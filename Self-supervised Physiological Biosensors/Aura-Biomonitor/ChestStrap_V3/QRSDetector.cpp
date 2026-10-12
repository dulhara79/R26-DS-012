#include "QRSDetector.h"

/************************************************
 * Constructor
 ************************************************/

QRSDetector::QRSDetector()
{
    reset();
}

/************************************************
 * Reset
 ************************************************/

void QRSDetector::reset()
{
    /**********************************************
     * Filter states
     **********************************************/

    filterInitialized = false;

    hp_x1 = 0.0f;
    hp_x2 = 0.0f;
    hp_y1 = 0.0f;
    hp_y2 = 0.0f;

    lp_x1 = 0.0f;
    lp_x2 = 0.0f;
    lp_y1 = 0.0f;
    lp_y2 = 0.0f;

    previousFiltered = 0.0f;

    /**********************************************
     * MWI
     **********************************************/

    mwiIndex = 0;
    mwiSum = 0.0f;

    for (int i = 0; i < MWI_SIZE; i++)
    {
        mwiBuffer[i] = 0.0f;
    }

    /**********************************************
     * Threshold
     **********************************************/

    signalPeak = 0.0f;
    noisePeak = 0.0f;
    threshold = MIN_THRESHOLD;

    startupTime = 0;
    initialized = false;

    /**********************************************
     * Peak detection
     **********************************************/

    previousMWISample = 0.0f;
    mwiLocalPeak = 0.0f;

    candidatePeakTime = 0;
    candidatePeakValue = 0.0f;
    candidateStartTime = 0;

    aboveThreshold = false;

    /**********************************************
     * RR timing
     **********************************************/

    lastPeakTime = 0;
    currentRR = 0;

    hasLastPeak = false;

    refractoryUntil = 0;
}

/************************************************
 * Band-pass filter
 *
 * Butterworth:
 *
 * High-pass = 5 Hz
 * Low-pass  = 15 Hz
 * Fs        = 500 Hz
 *
 * This removes:
 *
 * - baseline wander
 * - respiration/DC movement
 * - high-frequency noise
 ************************************************/

float QRSDetector::bandPassFilter(float sample)
{
    /**********************************************
     * 2nd order Butterworth 5 Hz HIGH-PASS
     **********************************************/

    const float hp_b0 = 0.95654323f;
    const float hp_b1 = -1.91308645f;
    const float hp_b2 = 0.95654323f;

    const float hp_a1 = -1.91119707f;
    const float hp_a2 = 0.91497583f;

    float hp_y =
        hp_b0 * sample +
        hp_b1 * hp_x1 +
        hp_b2 * hp_x2 -
        hp_a1 * hp_y1 -
        hp_a2 * hp_y2;

    hp_x2 = hp_x1;
    hp_x1 = sample;

    hp_y2 = hp_y1;
    hp_y1 = hp_y;

    /**********************************************
     * 2nd order Butterworth 15 Hz LOW-PASS
     **********************************************/

    const float lp_b0 = 0.00782021f;
    const float lp_b1 = 0.01564042f;
    const float lp_b2 = 0.00782021f;

    const float lp_a1 = -1.73472577f;
    const float lp_a2 = 0.76600660f;

    float lp_y =
        lp_b0 * hp_y +
        lp_b1 * lp_x1 +
        lp_b2 * lp_x2 -
        lp_a1 * lp_y1 -
        lp_a2 * lp_y2;

    lp_x2 = lp_x1;
    lp_x1 = hp_y;

    lp_y2 = lp_y1;
    lp_y1 = lp_y;

    return lp_y;
}

/************************************************
 * Preprocess ECG sample
 ************************************************/

float QRSDetector::preprocessSample(int32_t sample)
{
    /*
     * The ADS1292R produces a 24-bit signed value.
     * Scale down before floating-point processing.
     */

    float x = (float)sample / 1024.0f;

    if (!filterInitialized)
    {
        hp_x1 = x;
        hp_x2 = x;
        hp_y1 = 0.0f;
        hp_y2 = 0.0f;

        lp_x1 = 0.0f;
        lp_x2 = 0.0f;
        lp_y1 = 0.0f;
        lp_y2 = 0.0f;

        previousFiltered = 0.0f;
        filterInitialized = true;
    }

    return bandPassFilter(x);
}

/************************************************
 * Moving Window Integral
 ************************************************/

float QRSDetector::movingWindowIntegral(float sample)
{
    /*
     * Store squared ECG derivative energy.
     */

    float oldValue = mwiBuffer[mwiIndex];

    mwiSum -= oldValue;

    mwiBuffer[mwiIndex] = sample;

    mwiSum += sample;

    mwiIndex++;

    if (mwiIndex >= MWI_SIZE)
        mwiIndex = 0;

    return mwiSum / (float)MWI_SIZE;
}

/************************************************
 * Adaptive threshold update
 ************************************************/

void QRSDetector::updateThreshold(
    float peakValue,
    bool isSignal)
{
    if (peakValue <= 0.0f)
        return;

    if (isSignal)
    {
        /*
         * Signal peak follows actual QRS amplitude.
         */

        if (signalPeak == 0.0f)
        {
            signalPeak = peakValue;
        }
        else
        {
            signalPeak =
                0.125f * peakValue +
                0.875f * signalPeak;
        }
    }
    else
    {
        /*
         * Noise estimate.
         */

        if (noisePeak == 0.0f)
        {
            noisePeak = peakValue;
        }
        else
        {
            noisePeak =
                0.125f * peakValue +
                0.875f * noisePeak;
        }
    }

    /*
     * Standard adaptive threshold concept:
     *
     * threshold = noise + 25% of signal-noise range
     */

    if (signalPeak > noisePeak)
    {
        threshold =
            noisePeak +
            0.25f * (signalPeak - noisePeak);
    }
    else
    {
        threshold = noisePeak;
    }

    if (threshold < MIN_THRESHOLD)
    {
        threshold = MIN_THRESHOLD;
    }
}

/************************************************
 * Accept R peak
 ************************************************/

bool QRSDetector::acceptPeak(uint32_t peakTime)
{
    /**********************************************
     * First peak
     **********************************************/

    if (!hasLastPeak)
    {
        lastPeakTime = peakTime;

        hasLastPeak = true;

        currentRR = 0;

        refractoryUntil =
            peakTime + REFRACTORY_MS;

        return false;
    }

    /**********************************************
     * Calculate RR
     **********************************************/

    uint32_t rr = peakTime - lastPeakTime;

    /**********************************************
     * Physiological validation
     **********************************************/

    if (rr < MIN_RR_MS)
    {
        /*
         * Too short (T-wave, duplicate spike).
         */

        return false;
    }

    if (rr > MAX_RR_MS)
    {
        /*
         * Very long interval (strap shift or missed beats).
         * Update baseline reference peak so next beat is valid.
         */

        lastPeakTime = peakTime;

        currentRR = 0;

        refractoryUntil =
            peakTime + REFRACTORY_MS;

        return false;
    }

    /**********************************************
     * Accept RR
     **********************************************/

    currentRR = rr;

    lastPeakTime = peakTime;

    refractoryUntil =
        peakTime + REFRACTORY_MS;

    return true;
}

/************************************************
 * Process ECG sample
 ************************************************/

bool QRSDetector::processSample(
    int32_t sample,
    uint32_t timestamp)
{
    /**********************************************
     * Filter
     **********************************************/

    float filtered = preprocessSample(sample);

    /**********************************************
     * Derivative
     **********************************************/

    float derivative =
        filtered - previousFiltered;

    previousFiltered = filtered;

    /**********************************************
     * Square
     **********************************************/

    float squared =
        derivative * derivative;

    /**********************************************
     * Moving Window Integration (150 ms = 75 samples)
     **********************************************/

    float mwi =
        movingWindowIntegral(squared);

    /**********************************************
     * Startup period
     **********************************************/

    if (startupTime == 0)
    {
        startupTime = timestamp;
    }

    uint32_t startupElapsed =
        timestamp - startupTime;

    /**********************************************
     * Startup learning phase
     *
     * 0 to 1000 ms: Let filters prime and settle
     * 1000 to 2500 ms: Learn initial signal and noise peaks
     **********************************************/

    if (!initialized)
    {
        if (startupElapsed >= 1000)
        {
            if (mwi > signalPeak)
            {
                signalPeak = mwi;
            }

            if (mwi > 0.0f)
            {
                if (noisePeak == 0.0f)
                {
                    noisePeak = mwi;
                }
                else
                {
                    noisePeak =
                        0.02f * mwi +
                        0.98f * noisePeak;
                }
            }
        }

        if (startupElapsed >= 2500)
        {
            if (signalPeak <= noisePeak)
            {
                signalPeak = noisePeak * 2.0f;
            }

            threshold =
                noisePeak +
                0.25f *
                (signalPeak - noisePeak);

            if (threshold < MIN_THRESHOLD)
            {
                threshold = MIN_THRESHOLD;
            }

            initialized = true;
        }

        previousMWISample = mwi;
        return false;
    }

    /**********************************************
     * Adaptive threshold decay (Searchback mechanism)
     *
     * If no peak detected for > 1500 ms, gradually
     * decay threshold and signal peak toward floor.
     * At 500 Hz, 0.9995 decay reduces threshold ~22%/s.
     **********************************************/

    uint32_t timeSinceLastPeak = (hasLastPeak && lastPeakTime > 0) ? (timestamp - lastPeakTime) : startupElapsed;

    if (timeSinceLastPeak > 1500)
    {
        threshold *= 0.9995f;
        signalPeak *= 0.9995f;

        if (threshold < MIN_THRESHOLD)
        {
            threshold = MIN_THRESHOLD;
        }

        if (signalPeak < threshold * 1.5f)
        {
            signalPeak = threshold * 2.0f;
        }
    }

    /**********************************************
     * Refractory check
     **********************************************/

    if (timestamp < refractoryUntil)
    {
        previousMWISample = mwi;
        return false;
    }

    /**********************************************
     * Local maximum detection
     **********************************************/

    bool result = false;

    /**********************************************
     * Above threshold
     **********************************************/

    if (mwi > threshold)
    {
        if (!aboveThreshold)
        {
            // Start of QRS candidate
            aboveThreshold = true;

            candidatePeakValue = mwi;
            candidatePeakTime = timestamp;
            candidateStartTime = timestamp;
        }
        else
        {
            // Track highest MWI value inside candidate
            if (mwi > candidatePeakValue)
            {
                candidatePeakValue = mwi;
                candidatePeakTime = timestamp;
            }
        }
    }

    /**********************************************
     * Candidate ended
     **********************************************/

    else if (aboveThreshold)
    {
        aboveThreshold = false;

        uint32_t candidateDuration =
            timestamp - candidateStartTime;

        /******************************************
         * Validate candidate pulse width
         ******************************************/

        if (candidateDuration >= MIN_PEAK_SEARCH_MS && candidateDuration <= MAX_PEAK_SEARCH_MS)
        {
            // Update signal estimate
            updateThreshold(
                candidatePeakValue,
                true);

            // Accept actual R peak
            result =
                acceptPeak(candidatePeakTime);
        }
        else if (candidateDuration > MAX_PEAK_SEARCH_MS)
        {
            // Pulse was too wide (motion artifact / DC swing) -> update noise estimate
            updateThreshold(
                candidatePeakValue,
                false);
        }

        candidatePeakValue = 0.0f;
        candidatePeakTime = 0;
        candidateStartTime = 0;
        mwiLocalPeak = 0.0f;
    }

    /**********************************************
     * Noise estimation (track local peaks below threshold)
     **********************************************/

    else
    {
        if (mwi > previousMWISample)
        {
            if (mwi > mwiLocalPeak)
                mwiLocalPeak = mwi;
        }
        else if (previousMWISample > mwi && mwiLocalPeak > 0.0f)
        {
            // Local peak occurred below threshold -> update noise level
            updateThreshold(
                mwiLocalPeak,
                false);

            mwiLocalPeak = 0.0f;
        }
    }

    previousMWISample = mwi;

    /**********************************************
     * Return only when a VALID RR was generated
     **********************************************/

    return result;
}

/************************************************
 * Get last RR
 ************************************************/

uint32_t QRSDetector::getLastRR() const
{
    return currentRR;
}
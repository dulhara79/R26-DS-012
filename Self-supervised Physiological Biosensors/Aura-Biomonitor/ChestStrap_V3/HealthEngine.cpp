#include "HealthEngine.h"
#include "Config.h"

/**********************************************************
 * Constants
 **********************************************************/

/*
 * MPU6500 is configured for ±2 g.
 *
 * Therefore:
 *
 * 1 g ≈ 16384 counts
 *
 * We use vector magnitude rather than AZ so that
 * rotation of the chest strap does not change the
 * respiration baseline.
 */

static constexpr float ACC_COUNTS_PER_G = 16384.0f;


/**********************************************************
 * Wear detection
 **********************************************************/

static constexpr int32_t ECG_SATURATION_THRESHOLD = 7500000;

/*
 * ECG is sampled at 500 Hz.
 *
 * 100 samples ≈ 200 ms
 */

static constexpr uint16_t OFF_BODY_LIMIT = 1500;

/*
 * Require approximately 100 ms of good signal (50 samples)
 * before declaring the strap worn again.
 */

static constexpr uint16_t ON_BODY_LIMIT = 50;


/**********************************************************
 * Respiration (ADS1292R Channel 1 at 500 Hz)
 **********************************************************/

/*
 * Respiration low-pass filter:
 * Cutoff fc ~ 1.0 Hz at Fs = 500 Hz.
 * alpha = 2 * pi * fc / Fs ≈ 0.0125
 * Removes 50/60 Hz line noise, EMG, and ECG QRS bleed-through.
 */
static constexpr float RESP_LP_ALPHA = 0.0125f;

/*
 * DC baseline impedance tracking filter:
 * Cutoff fc ~ 0.008 Hz at Fs = 500 Hz.
 * alpha = 0.0001f (tau ~ 20 s) preserves respiratory frequencies
 * (0.1 to 0.7 Hz / 6 to 42 bpm) without canceling natural breaths.
 */
static constexpr float RESP_BASELINE_ALPHA = 0.0001f;

/*
 * Minimum time between breaths.
 * 1.5 seconds corresponds to 40 breaths/min.
 * This prevents multiple detections from one breath.
 */
static constexpr uint32_t MIN_BREATH_INTERVAL_MS = 1500;

/*
 * Maximum physiologically useful interval.
 * 15 seconds = 4 breaths/min.
 */
static constexpr uint32_t MAX_BREATH_INTERVAL_MS = 15000;


/**********************************************************
 * Constructor
 **********************************************************/

HealthEngine::HealthEngine()
{
    memset(&packet, 0, sizeof(packet));

    lastPacketTime = 0;

    stepCounter = 0;

    moving = false;

    /******************************************************
     * Wear state
     ******************************************************/

    isWorn = false;
    wasWorn = false;

    offBodyCount = OFF_BODY_LIMIT;

    /******************************************************
     * Respiration state
     ******************************************************/

    respInitialized = false;

    respLP = 0.0f;
    respBaseline = 0.0f;

    respirationSignal = 0.0f;
    previousRespirationSignal = 0.0f;

    respirationPeak = 0.0f;
    respirationTrough = 0.0f;
    adaptiveP2P = 1.0f;

    isRising = true;

    respirationPeakTime = 0;
    respirationTroughTime = 0;

    lastBreathTime = 0;

    hasFirstBreath = false;
}


/**********************************************************
 * Reset physiological state
 **********************************************************/

void HealthEngine::resetPhysiologicalState()
{
    /******************************************************
     * QRS
     ******************************************************/

    qrs.reset();

    /******************************************************
     * HRV
     ******************************************************/

    hrv.reset();

    /******************************************************
     * Respiration state
     ******************************************************/

    brStats = StatsWindow<float, 5>();

    respInitialized = false;

    respLP = 0.0f;
    respBaseline = 0.0f;

    respirationSignal = 0.0f;
    previousRespirationSignal = 0.0f;

    respirationPeak = 0.0f;
    respirationTrough = 0.0f;
    adaptiveP2P = 1.0f;

    isRising = true;

    respirationPeakTime = 0;
    respirationTroughTime = 0;

    lastBreathTime = 0;

    hasFirstBreath = false;

    /******************************************************
     * Motion / statistics
     ******************************************************/

    moving = false;

    stepCounter = 0;
}


/**********************************************************
 * Wear / lead detection
 **********************************************************/

void HealthEngine::updateWearState(
    const ECGData& ecg)
{
    if (!ecg.newData)
        return;

    /*
     * Check ECG saturation on Channel 2 (ECG channel).
     */

    bool ecgSaturated =
        (ecg.ch2 > ECG_SATURATION_THRESHOLD) ||
        (ecg.ch2 < -ECG_SATURATION_THRESHOLD);

    /******************************************************
     * Lead OFF
     ******************************************************/

    if (ecg.leadOff)
    {
        if (offBodyCount < 3000)
            offBodyCount++;

        if (offBodyCount >= OFF_BODY_LIMIT)
        {
            isWorn = false;
        }
    }

    /******************************************************
     * Lead ON
     ******************************************************/

    else
    {
        /*
         * Saturation is not treated as a complete
         * disconnect because an attached ECG can
         * temporarily saturate during movement.
         */

        if (ecgSaturated)
        {
            if (offBodyCount < 100)
                offBodyCount++;
        }
        else
        {
            if (offBodyCount > 0)
                offBodyCount--;
        }

        if (offBodyCount <= ON_BODY_LIMIT)
        {
            isWorn = true;
        }
    }

    /******************************************************
     * Detect transition from worn → not worn
     ******************************************************/

    if (wasWorn && !isWorn)
    {
        resetPhysiologicalState();
    }

    wasWorn = isWorn;
}


/**********************************************************
 * Main update
 **********************************************************/

void HealthEngine::update(
    const ECGData& ecg,
    const IMUData& imu,
    const TemperatureData& temp,
    uint32_t timestamp)
{
    /******************************************************
     * 1. Update lead / worn status
     ******************************************************/

    updateWearState(ecg);
    modelFeatures.advance(timestamp, isWorn && !ecg.leadOff);


    /******************************************************
     * 2. ADS1292R Acquisition (500 Hz)
     *
     * Channel 2 = ECG / HRV
     * Channel 1 = Respiration Impedance Pneumography
     *
     * Only process when the electrodes are attached and worn.
     ******************************************************/

    if (ecg.newData && isWorn && !ecg.leadOff)
    {
        // 2a. ECG / QRS detection on Channel 2
        if (qrs.processSample(
                ecg.ch2,
                timestamp))
        {
            uint32_t rr = qrs.getLastRR();

            /*
             * Extra sanity check.
             *
             * QRSDetector already validates this, but
             * keeping this boundary here protects HRV.
             */

            if (rr >= 300 && rr <= 2000)
            {
                hrv.addRRInterval(
                    (float)rr
                );
                modelFeatures.addRR((float)rr);
            }
        }

        // 2b. Respiration detection on Channel 1 (ADS1292R Impedance)
        detectRespiration(
            ecg.ch1,
            timestamp
        );
    }


    /******************************************************
     * 3. IMU (100 Hz)
     ******************************************************/

    if (imu.newData)
    {
        detectMotion(imu);

        /**************************************************
         * Acceleration magnitude
         *
         * IMPORTANT:
         *
         * Old code used:
         *
         * abs(ax) + abs(ay) + abs(az)
         *
         * which is not acceleration magnitude.
         *
         * We now calculate:
         *
         * sqrt(ax² + ay² + az²)
         *
         * and convert it to g.
         **************************************************/

        float ax = (float)imu.ax;
        float ay = (float)imu.ay;
        float az = (float)imu.az;

        float magnitude =
            sqrt(
                ax * ax +
                ay * ay +
                az * az
            );

        float magnitudeG =
            magnitude / ACC_COUNTS_PER_G;

        accStats.add(magnitudeG);
        modelFeatures.addAcceleration(ax / ACC_COUNTS_PER_G,
                                      ay / ACC_COUNTS_PER_G,
                                      az / ACC_COUNTS_PER_G);
    }


    /******************************************************
     * 4. Temperature
     ******************************************************/

    if (temp.newData && isWorn)
    {
        tempStats.add(
            temp.bodyTemperature
        );
        modelFeatures.addTemperature(temp.bodyTemperature);
    }


    /******************************************************
     * 5. Build health packet
     ******************************************************/

    packet.timestamp = timestamp;

    packet.isWorn = isWorn;


    /******************************************************
     * Physiological values
     ******************************************************/

    if (isWorn)
    {
        /**************************************************
         * Heart rate
         **************************************************/

        packet.meanHR =
            hrv.getMeanHR();

        packet.meanRR =
            hrv.getMeanRR();

        packet.sdnn =
            hrv.getSDNN();

        packet.rmssd =
            hrv.getRMSSD();


        /**************************************************
         * Respiration
         **************************************************/

        float meanBreathInterval =
            brStats.getMean();

        if (meanBreathInterval > 0.0f)
        {
            packet.meanBR =
                60000.0f /
                meanBreathInterval;
        }
        else
        {
            packet.meanBR = 0.0f;
        }


        /**************************************************
         * BR standard deviation
         *
         * Convert interval SD into approximate
         * breathing-rate SD using:
         *
         * BR = 60000 / RR
         **************************************************/

        float stdBreathInterval =
            brStats.getStdDev();

        if (meanBreathInterval > 0.0f)
        {
            packet.stdBR =
                (60000.0f /
                 (meanBreathInterval *
                  meanBreathInterval))
                *
                stdBreathInterval;
        }
        else
        {
            packet.stdBR = 0.0f;
        }


        /**************************************************
         * Temperature
         **************************************************/

        packet.meanTemp =
            tempStats.getMean();

        packet.stdTemp =
            tempStats.getStdDev();
    }

    else
    {
        /*
         * Never report physiological values when the
         * electrodes are disconnected.
         */

        packet.meanHR = 0.0f;
        packet.meanRR = 0.0f;

        packet.sdnn = 0.0f;
        packet.rmssd = 0.0f;

        packet.meanBR = 0.0f;
        packet.stdBR = 0.0f;

        packet.meanTemp = 0.0f;
        packet.stdTemp = 0.0f;
    }


    /******************************************************
     * Acceleration statistics
     ******************************************************/

    packet.meanAccMag =
        accStats.getMean();

    packet.stdAccMag =
        accStats.getStdDev();
}


/**********************************************************
 * Respiration detection (ADS1292R Channel 1 at 500 Hz)
 *
 * Channel 1 provides thoracic impedance pneumography data.
 * Breathing modulates this impedance (0.1 to 0.7 Hz).
 **********************************************************/

void HealthEngine::detectRespiration(
    int32_t respSample,
    uint32_t timestamp)
{
    /******************************************************
     * 1. Scale 24-bit raw impedance sample
     ******************************************************/

    float raw = (float)respSample / 1024.0f;

    /******************************************************
     * 2. Initialize filter state on first sample
     ******************************************************/

    if (!respInitialized)
    {
        respLP = raw;
        respBaseline = raw;
        respirationSignal = 0.0f;
        previousRespirationSignal = 0.0f;
        respirationPeak = 0.0f;
        respirationTrough = 0.0f;
        adaptiveP2P = 1.0f;
        isRising = true;
        respirationPeakTime = timestamp;
        respirationTroughTime = timestamp;
        respInitialized = true;
        return;
    }

    /******************************************************
     * 3. Low-pass filter (fc ~ 1.0 Hz at 500 Hz)
     *
     * Removes 50/60 Hz mains interference, muscle EMG,
     * and ECG QRS spikes that bleed into impedance.
     ******************************************************/

    respLP += RESP_LP_ALPHA * (raw - respLP);

    /******************************************************
     * 4. DC baseline tracking (fc ~ 0.008 Hz at 500 Hz)
     *
     * alpha = 0.0001f preserves respiratory frequencies
     * without canceling natural slow/deep breaths.
     ******************************************************/

    respBaseline += RESP_BASELINE_ALPHA * (respLP - respBaseline);

    /******************************************************
     * 5. Dynamic respiration signal
     *
     * Zero-mean respiratory waveform.
     ******************************************************/

    respirationSignal = respLP - respBaseline;

    /******************************************************
     * 6. Peak-to-Peak Amplitude & Extrema Detection
     *
     * Detects breath cycles by tracking trough-to-peak
     * and peak-to-trough transitions with adaptive
     * amplitude thresholding and hysteresis.
     ******************************************************/

    // Hysteresis threshold to confirm direction change (prevents noise ripple)
    float hysteresis = 0.25f * adaptiveP2P;
    if (hysteresis < 0.10f)
        hysteresis = 0.10f;

    // Minimum peak-to-peak amplitude required for a valid breath
    float minP2P = 0.35f * adaptiveP2P;
    if (minP2P < 0.15f)
        minP2P = 0.15f;

    if (isRising)
    {
        // Track local maximum during rising phase
        if (respirationSignal > respirationPeak)
        {
            respirationPeak = respirationSignal;
            respirationPeakTime = timestamp;
        }

        // Peak confirmed when signal drops below (peak - hysteresis)
        if (respirationSignal < (respirationPeak - hysteresis))
        {
            float p2p = respirationPeak - respirationTrough;

            if (p2p >= minP2P)
            {
                uint32_t breathTime = respirationPeakTime;

                if (!hasFirstBreath)
                {
                    lastBreathTime = breathTime;
                    hasFirstBreath = true;
                }
                else
                {
                    uint32_t interval = breathTime - lastBreathTime;

                    if (interval >= MIN_BREATH_INTERVAL_MS &&
                        interval <= MAX_BREATH_INTERVAL_MS)
                    {
                        brStats.add((float)interval);
                        modelFeatures.addBreathInterval((float)interval, timestamp);
                        lastBreathTime = breathTime;

                        // Adapt amplitude estimate
                        adaptiveP2P = 0.2f * p2p + 0.8f * adaptiveP2P;
                    }
                    else if (interval > MAX_BREATH_INTERVAL_MS)
                    {
                        // Re-anchor timing on prolonged pause or deep breath
                        lastBreathTime = breathTime;
                    }
                }
            }

            // Switch to falling phase
            isRising = false;
            respirationTrough = respirationSignal;
            respirationTroughTime = timestamp;
        }
    }
    else
    {
        // Track local minimum during falling phase
        if (respirationSignal < respirationTrough)
        {
            respirationTrough = respirationSignal;
            respirationTroughTime = timestamp;
        }

        // Trough confirmed when signal rises above (trough + hysteresis)
        if (respirationSignal > (respirationTrough + hysteresis))
        {
            // Switch to rising phase
            isRising = true;
            respirationPeak = respirationSignal;
            respirationPeakTime = timestamp;
        }
    }

    previousRespirationSignal = respirationSignal;
}


/**********************************************************
 * Motion detection
 **********************************************************/

void HealthEngine::detectMotion(
    const IMUData& imu)
{
    float ax = (float)imu.ax;
    float ay = (float)imu.ay;
    float az = (float)imu.az;

    float magnitude =
        sqrt(
            ax * ax +
            ay * ay +
            az * az
        );

    /*
     * Compare magnitude against gravity.
     *
     * A stationary sensor should be approximately
     * 1 g.
     */

    float dynamicMagnitude =
        fabs(
            magnitude -
            ACC_COUNTS_PER_G
        );

    /*
     * Approximately 0.15 g dynamic acceleration.
     *
     * This is only a basic motion indicator, not a
     * medical-grade activity classifier.
     */

    if (dynamicMagnitude >
        (0.15f * ACC_COUNTS_PER_G))
    {
        moving = true;

        stepCounter++;
    }
    else
    {
        moving = false;
    }
}


/**********************************************************
 * Get packet
 **********************************************************/

const HealthPacket&
HealthEngine::getPacket()
{
    return packet;
}


/**********************************************************
 * Get worn status
 **********************************************************/

bool HealthEngine::getIsWorn() const
{
    return isWorn;
}


/**********************************************************
 * CSV output
 **********************************************************/

String HealthEngine::toCSV()
{
    String csv;

    csv.reserve(320);


    /******************************************************
     * Time
     ******************************************************/

    csv += String(packet.timestamp);
    csv += ",";


    /******************************************************
     * HR
     ******************************************************/

    csv += String(packet.meanHR, 2);
    csv += ",";

    csv += String(packet.meanRR, 2);
    csv += ",";

    csv += String(packet.sdnn, 2);
    csv += ",";

    csv += String(packet.rmssd, 2);
    csv += ",";


    /******************************************************
     * Respiration
     ******************************************************/

    csv += String(packet.meanBR, 2);
    csv += ",";

    csv += String(packet.stdBR, 2);
    csv += ",";


    /******************************************************
     * Temperature
     ******************************************************/

    csv += String(packet.meanTemp, 2);
    csv += ",";

    csv += String(packet.stdTemp, 2);
    csv += ",";


    /******************************************************
     * Acceleration
     ******************************************************/

    csv += String(packet.meanAccMag, 2);
    csv += ",";

    csv += String(packet.stdAccMag, 2);
    csv += ",";


    /******************************************************
     * Lead connected
     *
     * 1 = connected
     * 0 = disconnected
     ******************************************************/

    csv += String(
        packet.isWorn ? 1 : 0
    );

    // Keep the original 12 display fields, then append the versioned model
    // minute. The same completed block may be repeated at 1 Hz; the app dedups
    // its end timestamp instead of averaging it again.
    const auto &model = modelFeatures.latest();
    csv += ",";
    csv += String(ModelMinuteFeatures::SCHEMA_VERSION);
    csv += ",";
    csv += String(model.startMs);
    csv += ",";
    csv += String(model.endMs);
    csv += ",";
    csv += String(model.status);
    for (float value : model.values) {
        csv += ",";
        csv += String(value, 6);
    }
    csv += "\n";

    return csv;
}
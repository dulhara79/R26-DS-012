#ifndef QRSDETECTOR_H
#define QRSDETECTOR_H

#include <Arduino.h>
#include <math.h>

class QRSDetector
{
public:

    QRSDetector();

    // Process one ECG sample.
    // sample    = raw signed ADS1292R CH2 sample (ECG channel)
    // timestamp = timestamp in milliseconds
    //
    // Returns true only when a valid RR interval has been detected.
    bool processSample(int32_t sample, uint32_t timestamp);

    // Last accepted RR interval in milliseconds.
    uint32_t getLastRR() const;

    // Reset detector state.
    void reset();

private:

    /************************************************
     * Sampling
     ************************************************/

    static constexpr float SAMPLE_RATE = 500.0f;

    /************************************************
     * ECG Band-pass
     *
     * 5 Hz high-pass
     * 15 Hz low-pass
     ************************************************/

    bool filterInitialized;

    float hp_x1;
    float hp_x2;
    float hp_y1;
    float hp_y2;

    float lp_x1;
    float lp_x2;
    float lp_y1;
    float lp_y2;

    float previousFiltered;

    float bandPassFilter(float sample);

    /************************************************
     * QRS processing
     *
     * 75 samples @ 500 Hz = 150 ms window
     ************************************************/

    static constexpr int MWI_SIZE = 75;

    float mwiBuffer[MWI_SIZE];
    int mwiIndex;
    float mwiSum;

    float movingWindowIntegral(float sample);

    /************************************************
     * Adaptive threshold
     ************************************************/

    static constexpr float MIN_THRESHOLD = 0.05f;

    float signalPeak;
    float noisePeak;
    float threshold;

    uint32_t startupTime;
    bool initialized;

    /************************************************
     * Peak detection
     ************************************************/

    float previousMWISample;
    float mwiLocalPeak;

    uint32_t candidatePeakTime;

    bool aboveThreshold;

    /************************************************
     * R-peak timing
     ************************************************/

    uint32_t lastPeakTime;
    uint32_t currentRR;

    bool hasLastPeak;

    /************************************************
     * Refractory period
     ************************************************/

    uint32_t refractoryUntil;

    static constexpr uint32_t REFRACTORY_MS = 250;

    /************************************************
     * Physiological limits
     ************************************************/

    static constexpr uint32_t MIN_RR_MS = 300;
    static constexpr uint32_t MAX_RR_MS = 2000;

    /************************************************
     * Peak confirmation
     *
     * We wait for the MWI to come back down before
     * confirming the local maximum.
     ************************************************/

    static constexpr uint32_t MIN_PEAK_SEARCH_MS = 20;
    static constexpr uint32_t MAX_PEAK_SEARCH_MS = 250;

    float candidatePeakValue;
    uint32_t candidateStartTime;

    /************************************************
     * Helpers
     ************************************************/

    void updateThreshold(float peakValue, bool isSignal);

    bool acceptPeak(uint32_t peakTime);

    float preprocessSample(int32_t sample);
};

#endif
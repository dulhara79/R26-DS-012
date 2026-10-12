#include "HRVProcessor.h"

HRVProcessor::HRVProcessor() {
    head = 0;
    count = 0;
    meanRR = 0;
    sdnn = 0;
    rmssd = 0;
    meanHR = 0;
    for(int i = 0; i < HRV_WINDOW_SIZE; i++) rr_buffer[i] = 0.0f;
}

void HRVProcessor::addRRInterval(float rr_ms) {
    if (rr_ms < 300 || rr_ms > 2000) return;

    rr_buffer[head] = rr_ms;
    head = (head + 1) % HRV_WINDOW_SIZE;
    if (count < HRV_WINDOW_SIZE) count++;
    recalculate();
}

void HRVProcessor::recalculate() {
    if (count == 0) return;

    // Reconstruct the RR series in chronological order. The previous
    // implementation iterated through the physical ring-buffer order, which
    // becomes non-chronological after the first wrap and corrupts RMSSD.
    const int oldest = (count == HRV_WINDOW_SIZE) ? head : 0;

    float sumRR = 0.0f;
    for(int i = 0; i < count; i++) {
        const int index = (oldest + i) % HRV_WINDOW_SIZE;
        sumRR += rr_buffer[index];
    }

    meanRR = sumRR / count;
    meanHR = (meanRR > 0.0f) ? 60000.0f / meanRR : 0.0f;

    if(count > 1) {
        float sumSqDiff = 0.0f;
        float sumSuccDiffSq = 0.0f;

        float previous = rr_buffer[oldest];
        for(int i = 0; i < count; i++) {
            const int index = (oldest + i) % HRV_WINDOW_SIZE;
            const float current = rr_buffer[index];
            const float diff = current - meanRR;
            sumSqDiff += diff * diff;

            if(i > 0) {
                const float succDiff = current - previous;
                sumSuccDiffSq += succDiff * succDiff;
            }
            previous = current;
        }

        sdnn = sqrt(sumSqDiff / (count - 1));
        rmssd = sqrt(sumSuccDiffSq / (count - 1));
    } else {
        sdnn = 0.0f;
        rmssd = 0.0f;
    }
}

float HRVProcessor::getMeanHR() const { return meanHR; }
float HRVProcessor::getMeanRR() const { return meanRR; }
float HRVProcessor::getSDNN() const { return sdnn; }
float HRVProcessor::getRMSSD() const { return rmssd; }

void HRVProcessor::reset() {
    head = 0;
    count = 0;
    meanRR = 0;
    sdnn = 0;
    rmssd = 0;
    meanHR = 0;
    for(int i = 0; i < HRV_WINDOW_SIZE; i++) rr_buffer[i] = 0.0f;
}

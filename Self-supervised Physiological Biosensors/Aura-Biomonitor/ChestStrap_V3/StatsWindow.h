#ifndef STATSWINDOW_H
#define STATSWINDOW_H

#include <Arduino.h>
#include <math.h>

template<typename T, int SIZE>
class StatsWindow {
public:
    StatsWindow() { reset(); }

    void reset() {
        head = 0;
        count = 0;
        mean = 0;
        stdDev = 0;
        for(int i = 0; i < SIZE; i++) buffer[i] = 0;
    }

    void add(T val) {
        buffer[head] = val;
        head = (head + 1) % SIZE;
        if (count < SIZE) count++;
        recalculate();
    }

    float getMean() const { return mean; }
    float getStdDev() const { return stdDev; }

private:
    T buffer[SIZE];
    int head;
    int count;
    float mean;
    float stdDev;

    void recalculate() {
        if (count == 0) return;

        float sum = 0;
        for(int i = 0; i < count; i++) sum += (float)buffer[i];
        mean = sum / count;

        if (count > 1) {
            float sumSqDiff = 0;
            for(int i = 0; i < count; i++) {
                float diff = (float)buffer[i] - mean;
                sumSqDiff += diff * diff;
            }
            stdDev = sqrt(sumSqDiff / (count - 1));
        } else {
            stdDev = 0;
        }
    }
};

#endif

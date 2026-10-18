#ifndef MODEL_MINUTE_FEATURES_H
#define MODEL_MINUTE_FEATURES_H

#include <cmath>
#include <cstdint>

// Separate model inputs from the existing live display statistics. Each block
// contains one non-overlapping minute, never an average of rolling SD/RMSSD.
class ModelMinuteFeatures {
public:
    static constexpr int SCHEMA_VERSION = 2;
    static constexpr uint32_t WINDOW_MS = 60000;
    enum Status { WARMING = 0, READY = 1, INSUFFICIENT_RR = 2,
                  INCOMPLETE = 3, NOT_WORN = 4 };

    struct Block {
        uint32_t startMs = 0, endMs = 0;
        int status = WARMING;
        float values[10] = {};
    };

    void advance(uint32_t now, bool worn) {
        if (!worn) {
            active = false;
            block = Block();
            block.status = NOT_WORN;
            return;
        }
        if (!active) {
            active = true;
            block = Block();
            warming = true;
            startMs = now;
            lastRateMs = now;
            hasBreath = false;
            for (auto &axis : highPass) axis = HighPass();
            clearMinute();
        }
        if (uint32_t(now - startMs) >= WINDOW_MS) {
            const bool timely = uint32_t(now - startMs) <= WINDOW_MS + 1000;
            finishMinute(timely);
            startMs = timely ? startMs + WINDOW_MS : now;
            clearMinute();
            warming = false;
        }
        if (uint32_t(now - lastRateMs) >= 1000) {
            lastRateMs = now;
            // Do not perpetuate a cached breathing rate after breath detection
            // stops. This previously made "normal BR" look fresh indefinitely.
            if (hasBreath && uint32_t(now - lastBreathMs) <= 15000)
                breathing.add(breathingRate);
        }
    }

    void addRR(float milliseconds) {
        if (active && milliseconds >= 300 && milliseconds <= 2000 && rrCount < 201)
            rr[rrCount++] = milliseconds;
    }

    void addAcceleration(float xG, float yG, float zG) {
        if (!active) return;
        float x = highPass[0].apply(xG), y = highPass[1].apply(yG);
        float z = highPass[2].apply(zG);
        acceleration.add(std::sqrt(x*x + y*y + z*z));
    }

    void addTemperature(float degreesC) {
        if (active && std::isfinite(degreesC) && degreesC >= 10 && degreesC <= 45)
            temperature.add(degreesC);
    }

    void addBreathInterval(float milliseconds, uint32_t now) {
        if (active && milliseconds >= 1500 && milliseconds <= 15000) {
            breathingRate = 60000.0f / milliseconds;
            lastBreathMs = now;
            hasBreath = true;
        }
    }

    const Block &latest() const { return block; }

private:
    struct Moments {
        unsigned count = 0;
        double mean = 0, m2 = 0;
        void add(float value) {
            ++count;
            double difference = value - mean;
            mean += difference / count;
            m2 += difference * (value - mean);
        }
        float sd() const {
            return count > 1 ? float(std::sqrt(m2 / (count - 1))) : 0.0f;
        }
    };

    struct HighPass {
        // scipy.signal.butter(4, .5, 'high', fs=100, output='sos').
        // Apply TWO forward passes: training filtfilt applies this filter twice.
        // This matches its steady-state magnitude response without buffering or
        // blocking ECG acquisition. Phase and window-edge behavior remain causal
        // and are not claimed to be bit-identical to offline filtfilt.
        double delay[4][2] = {};
        double origin = 0;
        bool initialized = false;
        float apply(float value) {
            if (!initialized) { origin = value; initialized = true; }
            double sample = double(value) - origin; // seed DC gravity at steady state
            const double sos[2][5] = {
                {0.959782230087, -1.91956446017, 0.959782230087,
                 -1.94263823054, 0.94359727847},
                {1.0, -2.0, 1.0, -1.97526963485, 0.976244792359}
            };
            for (int section = 0; section < 4; ++section) {
                const double *s = sos[section % 2];
                double output = s[0] * sample + delay[section][0];
                delay[section][0] = s[1] * sample - s[3] * output + delay[section][1];
                delay[section][1] = s[2] * sample - s[4] * output;
                sample = output;
            }
            return float(sample);
        }
    };

    bool active = false, warming = true, hasBreath = false;
    uint32_t startMs = 0, lastRateMs = 0, lastBreathMs = 0;
    float breathingRate = 0, rr[201] = {};
    unsigned rrCount = 0;
    Moments acceleration, temperature, breathing;
    HighPass highPass[3];
    Block block;

    void clearMinute() {
        acceleration = Moments(); temperature = Moments(); breathing = Moments();
        rrCount = 0;
    }

    void finishMinute(bool timely) {
        block = Block();
        block.startMs = startMs;
        block.endMs = startMs + WINDOW_MS;
        if (warming) return; // discard filter/QRS startup minute
        if (!timely || acceleration.count < 4800 || temperature.count < 20 ||
            breathing.count < 48) { block.status = INCOMPLETE; return; }
        if (rrCount < 30) { block.status = INSUFFICIENT_RR; return; }
        Moments intervals;
        double successiveSquares = 0;
        for (unsigned i = 0; i < rrCount; ++i) {
            intervals.add(rr[i]);
            if (i) {
                double difference = rr[i] - rr[i-1];
                successiveSquares += difference * difference;
            }
        }
        block.values[0] = 60000.0f / float(intervals.mean);
        block.values[1] = float(intervals.mean);
        block.values[2] = intervals.sd();
        block.values[3] = float(std::sqrt(successiveSquares / (rrCount - 1)));
        block.values[4] = float(breathing.mean);
        block.values[5] = breathing.sd();
        block.values[6] = float(temperature.mean);
        block.values[7] = temperature.sd();
        block.values[8] = float(acceleration.mean);
        block.values[9] = acceleration.sd();
        block.status = READY;
    }
};

#endif

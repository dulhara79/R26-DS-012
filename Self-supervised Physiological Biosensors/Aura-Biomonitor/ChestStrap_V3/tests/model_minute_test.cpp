#include "../ModelMinuteFeatures.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

static ModelMinuteFeatures simulate(bool motion, bool temperature = true,
                                    bool beats = true, bool breaths = true, float frequency = 1.0f) {
    ModelMinuteFeatures model;
    for (uint32_t time = 0; time <= 120000; time += 10) {
        model.advance(time, true);
        if (time == 120000) break;
        const float seconds = time / 1000.0f;
        model.addAcceleration(motion ? .03f * std::sin(6.283185307f * frequency * seconds) : 0,
                              0, .9f);
        if (time % 1000 == 0) {
            if (temperature) model.addTemperature(37 + .1f * std::sin(seconds));
            if (beats) model.addRR(time % 2000 == 0 ? 980 : 1020);
        }
        if (breaths && time % 4000 == 0) model.addBreathInterval(4000, time);
    }
    return model;
}

int main(int argc, char **argv) {
    if (argc == 2) {
        auto block = simulate(true, true, true, true, float(std::atof(argv[1]))).latest();
        std::printf("%.9g %.9g\n", block.values[8], block.values[9]);
        return 0;
    }
    auto stationary = simulate(false);
    auto block = stationary.latest();
    assert(block.status == ModelMinuteFeatures::READY);
    assert(block.startMs == 60000 && block.endMs == 120000);
    assert(std::abs(block.values[0] - 60) < .001f);
    assert(std::abs(block.values[1] - 1000) < .001f);
    assert(std::abs(block.values[2] - std::sqrt(24000.0f / 59)) < .001f);
    assert(std::abs(block.values[3] - 40) < .001f);
    assert(block.values[8] < 1e-6f && block.values[9] < 1e-6f);
    assert(std::abs(block.values[4] - 15) < .001f);
    assert(block.values[5] < 1e-6f);

    auto moving = simulate(true);
    block = moving.latest();
    assert(block.status == ModelMinuteFeatures::READY);
    assert(block.values[8] > .018f && block.values[8] < .020f);
    assert(block.values[9] > .008f && block.values[9] < .011f);
    // At the 0.5 Hz cutoff, offline filtfilt has gain 0.5 (two passes),
    // not sqrt(0.5). Check RMS from the full-minute moments, independent of phase.
    auto cutoff = simulate(true, true, true, true, .5f).latest();
    float rms = std::sqrt(cutoff.values[8] * cutoff.values[8] +
                         cutoff.values[9] * cutoff.values[9] * 5999.0f / 6000.0f);
    assert(std::abs(rms - .03f * .5f / std::sqrt(2.0f)) < 1e-6f);
    // Output permits an independent comparison with SciPy's reference filter.
    std::printf("%.9g %.9g\n", block.values[8], block.values[9]);

    assert(simulate(false, false).latest().status == ModelMinuteFeatures::INCOMPLETE);
    assert(simulate(false, true, false).latest().status == ModelMinuteFeatures::INSUFFICIENT_RR);
    assert(simulate(false, true, true, false).latest().status == ModelMinuteFeatures::INCOMPLETE);
    // Real loops can notice a boundary a millisecond late. Block IDs must
    // remain contiguous rather than accumulating that scheduling jitter.
    moving.advance(180001, true);
    assert(moving.latest().startMs == 120000);
    assert(moving.latest().endMs == 180000);
    moving.advance(240002, true);
    assert(moving.latest().startMs == 180000);
    assert(moving.latest().endMs == 240000);
    stationary.advance(120010, false);
    assert(stationary.latest().status == ModelMinuteFeatures::NOT_WORN);
    stationary.advance(120020, true);
    assert(stationary.latest().status != ModelMinuteFeatures::READY);
    std::puts("minute feature tests passed");
}

# Chest Strap V3

Firmware sketch for **Aura Biomonitor** (`v1.1.0`). Start here to flash; docs live one level up.

For the full write-up: hardware, protocol, architecture, build- see the [root README](../README.md).


## Physiological model feature contract v2

Flash the whole `ChestStrap_V3` sketch on the ESP32-C3 Super Mini, not just an
individual `.cpp` file. Install the matching app and deploy both `main.py` and
`calibration.py` from `HF-Anxiety` before collecting a fresh resting baseline.
Old calibration and data remain stored but cannot be used with the new features.

The first 12 CSV fields still describe the live display. Fields 13–26 are:

`schema_version,start_ms,end_ms,status,mean_hr,mean_rr,sdnn,rmssd,mean_br,std_br,mean_temp,std_temp,mean_acc_mag,std_acc_mag`

Status: 0 = warming, 1 = complete minute, 2 = too few RR intervals, 3 = incomplete
sensor coverage, 4 = not worn. A completed minute is repeated until the next
boundary. The phone uploads it once and uses the same blocks for calibration.
Keep wearing the strap for at least two minutes before starting resting setup.
Setup collects ten full consecutive minutes; boundary alignment may add up to
one minute. Keep the same resting position and let HR settle before starting.

Model features use chronological RR intervals (at least 30 per minute), sample
SDNN, successive-difference RMSSD, and full-minute temperature and breathing-rate
statistics. Breathing rate is sampled at 1 Hz from the most recent valid breath;
it expires after 15 seconds without another detected breath. Acceleration is
filtered on each axis before magnitude using two fourth-order 0.5 Hz Butterworth
high-pass passes at 100 Hz. Two causal passes match the steady-state magnitude
response of the notebook's forward/backward `filtfilt`; phase and window edges
still differ. Therefore this corrects the gravity/definition mismatch but does not
claim bit-identical preprocessing or validated anxiety prediction on hardware.
New recordings must be checked after flashing. The model output remains a
physiological anomaly index, not an anxiety probability. Model weights and score
formula are unchanged.

Host regression check:

```sh
g++ -std=c++11 -O2 -Wall -Wextra -Werror tests/model_minute_test.cpp -o /tmp/model-minute-test
/tmp/model-minute-test
```

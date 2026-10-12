# Aura Biomonitor 

### Your chest already knows. Aura just listens.

**Aura Biomonitor** is onboard firmware for a wearable chest strap that turns raw biosignals into clear, second-by-second health metrics: heart rhythm, breathing, temperature, and motion- then hands them to your phone over Bluetooth Low Energy.

<p align="center">
  <img src="Aura Chest Strap.png" alt="Aura Chest Strap" width="400"/>
</p>

This repository contains **Chest Strap V3** (`v1.0.2`): ESP32-C3 firmware that samples ECG and thoracic impedance at clinical-ish rates, runs QRS / HRV / respiration pipelines on-device, and streams a compact CSV health packet once per second.

> Not a medical device. Metrics are for research, prototyping, and personal exploration- not diagnosis or treatment.

---

## What it feels like

Strap on. Electrodes find skin. Within a few seconds the strap decides you’re *on body*, R-peaks start lining up, and a quiet 1 Hz stream begins:

```
timestamp, meanHR, meanRR, SDNN, RMSSD, meanBR, stdBR, meanTemp, stdTemp, meanAccMag, stdAccMag, isWorn
```

Heart rate from real RR intervals. Breathing from impedance pneumography. Skin temperature with a rolling window. Motion as accel magnitude stats. Wear state as a hard gate- when you’re off body, physiology zeros out instead of inventing numbers.

---

## Hardware

| Piece | Role |
|-------|------|
| **ESP32-C3 Super Mini** | Brain + BLE radio |
| **ADS1292R** | Dual-channel AFE — CH2 ECG, CH1 respiration impedance |
| **MPU6500** | 6-axis IMU (accel + gyro) |
| **DS18B20** | Body temperature (OneWire) |

### Pin map

| Function | Pins |
|----------|------|
| ADS1292R SPI | CS `10`, START `0`, RESET `1`, DRDY `3`, SCLK `4`, MISO `5`, MOSI `6` |
| MPU6500 I²C | SDA `9`, SCL `8`, address `0x69` |
| DS18B20 | `2` |

ECG SPI and IMU I²C both run at **100 kHz**. ADS1292R samples at **500 SPS** with DRDY-driven reads.

---

## Signal path

```
ADS1292R ──► ECG (CH2) + Resp impedance (CH1)
MPU6500  ──► Accel / gyro
DS18B20  ──► Body temp
              │
              ▼
         Health Engine
    ┌─────────┼─────────┐
    │         │         │
 QRS / HRV  Breathing  Wear + stats
    │         │         │
    └─────────┼─────────┘
              ▼
     HealthPacket → CSV
              ▼
     BLE Nordic UART  (+ Serial debug)
              ▼
         Mobile / host
```

| Module | Job |
|--------|-----|
| `ADS1292R` | SPI driver, DRDY IRQ, 24-bit conversion, lead-off status |
| `MPU6500` | I²C IMU, WHO_AM_I check |
| `DS18B20` | Non-blocking OneWire temperature |
| `QRSDetector` | Pan–Tompkins-style R-peak pipeline |
| `HRVProcessor` | Rolling RR window → mean HR / RR, SDNN, RMSSD |
| `HealthEngine` | Wear gating, respiration, stats, CSV assembly |
| `BLE_UART` | Nordic UART Service, 20-byte notify chunks |
| `Scheduler` | Period gates for IMU / temp / BLE tasks |

---

## Sampling cadence

| Stream | Rate | How |
|--------|------|-----|
| ECG + impedance | **500 Hz** | ADS1292R DRDY interrupt |
| IMU | **100 Hz** | Scheduler |
| Temperature | **~1 Hz** | Scheduler + DS18B20 conversion |
| BLE health CSV | **1 Hz** | Scheduler |

---

## On-device intelligence

### Heart (QRS → HRV)
- Band-pass **5–15 Hz** → derivative → square → **150 ms** moving-window integral
- Adaptive peak threshold, **250 ms** refractory, RR clamp **300–2000 ms**
- Last **20** RR intervals → **mean HR**, **mean RR**, **SDNN**, **RMSSD**

### Breath
- CH1 thoracic impedance: low-pass vs slow baseline, adaptive peak-to-peak
- Breath intervals **1.5–15 s**, rate from a **5-breath** rolling window (`meanBR`, `stdBR`)

### Wear / lead-off
- Lead-off from ADS1292R status bits
- Off-body after ~3 s of disconnect; on-body after a short run of good samples
- Saturation and disconnect reset QRS / HRV / respiration so stale beats don’t linger

### Motion & temperature
- Accel magnitude mean / std over ~1 s
- Body temp mean / std over ~10 s
- Internal motion / step heuristics exist but are **not** in the BLE CSV yet

---

## BLE protocol

Advertises as **`ChestStrap_V3`** using the **Nordic UART Service (NUS)**:

| Role | UUID |
|------|------|
| Service | `6E400001-B5A3-F393-E0A9-E50E24DCCA9E` |
| RX (write) | `…0002…` |
| TX (notify) | `…0003…` |

Payloads are newline-terminated CSV, chunked at **20 bytes**. Reassemble on the phone until `\n`.

**Field order**

```
timestamp,meanHR,meanRR,sdnn,rmssd,meanBR,stdBR,meanTemp,stdTemp,meanAccMag,stdAccMag,isWorn
```

Floats use two decimal places. `isWorn` is `1` or `0`. When off-body, physiological fields are forced to `0`.

---

## Build & flash

**Board:** MakerGO / ESP32-C3 Super Mini (`esp32:esp32:makergo_c3_supermini`)  
**Sketch:** `ChestStrap_V3/ChestStrap_V3.ino`  
**Typical settings:** CPU 160 MHz · Flash 80 MHz · Upload 921600

### Libraries
- ESP32 Arduino core (project tooling references ~3.0.x)
- `SPI`, `Wire`
- ESP32 **BLE** (`BLEDevice`, NUS-style UART)
- **OneWire** + **DallasTemperature**

### Quick start
1. Open `ChestStrap_V3/ChestStrap_V3.ino` in Arduino IDE or VS Code + Arduino extension.
2. Select the C3 Super Mini board and your serial port.
3. Install the libraries above if needed.
4. Upload, open Serial Monitor at **115200**.
5. Connect a NUS-capable BLE client to `ChestStrap_V3` and subscribe to TX notifies.


---


Treat electrode contact, hygiene, and electrical safety seriously. This firmware is a biomonitoring **prototype**, not a clinical instrument.

# Self-Supervised Physiological Biosensors

**R26-DS-012 · Component 1 · Sendanayake H.D. (IT22107596)**  
*Personalized anomaly detection and short-horizon escalation forecasting from chest-strap physiology — without labelled panic episodes.*

[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](.)
[![Feature Contract](https://img.shields.io/badge/Features-10%20locked-blue)](.)
[![AE](https://img.shields.io/badge/AE-c1--unmasked--lstm--ae--wesad--v2-EE4C2C)](.)
[![Forecast](https://img.shields.io/badge/Forecast-c1--direct--ridge--wesad--v5-orange)](.)
[![Stack](https://img.shields.io/badge/Stack-PyTorch%20%7C%20FastAPI%20%7C%20InfluxDB%20%7C%20HF-lightgrey)](.)

---

## The question

**Can we warn someone before acute autonomic escalation — without ever teaching the model what anxiety looks like?**

Most stress detectors need labelled “anxious” windows. Those labels are rare, subjective, and hard to collect in the wild. This component flips the problem:

1. Learn only **calm / baseline** physiology (self-supervised reconstruction).
2. Treat **reconstruction failure** as the anomaly signal.
3. Forecast that signal **5 and 10 minutes ahead** so the mobile stack can act before the peak.

Target hardware is a low-cost **ESP32-C3 chest strap** (ECG, respiration, motion, and skin temperature). The 10-feature contract, window lengths, and deployment path are all constrained by what that wearable can reliably produce.

> Research use only. Outputs are physiological risk indices / scores — **not** a clinical diagnosis.

---

## Pipeline at a glance

```text
Chest strap → 60s feature windows (10 dims)
                    │
                    ▼
         per-user baseline z-norm
                    │
                    ▼
     LSTM Autoencoder (T = 5 min)
                    │
                    ▼
      reconstruction error → C1 score
         score = err / (err + baseline_p95)
                    │
                    ▼
   Direct Ridge forecaster
   history: last 2 × 5-min blocks
   horizons: +5 min, +10 min
                    │
                    ▼
   FastAPI → Flutter app / fusion (C4)
```

Deployed model versions:

| Stage | Version ID | Role |
|-------|------------|------|
| Autoencoder | `c1-unmasked-lstm-ae-wesad-v2` | Population AE; personalized weights override when available |
| Forecaster | `c1-direct-ridge-score-forecast-wesad-v5` | Direct +5 / +10 logit-score delta (Ridge, α = 100) |

---

## Locked 10-feature contract

Extracted over **non-overlapping 60-second** windows:

| # | Feature | Meaning |
|---|---------|---------|
| 1 | `mean_hr` | Mean heart rate (bpm) |
| 2 | `mean_rr` | Mean R–R interval (ms) |
| 3 | `sdnn` | HRV — SDNN (ms) |
| 4 | `rmssd` | HRV — RMSSD (ms) |
| 5 | `mean_br` | Mean breathing rate |
| 6 | `std_br` | Breathing-rate variability |
| 7 | `mean_temp` | Mean skin temperature |
| 8 | `std_temp` | Temperature variability |
| 9 | `mean_acc_mag` | Mean acceleration magnitude |
| 10 | `std_acc_mag` | Acceleration variability |

`pNN50` was dropped from earlier 11-feature experiments — it is unreliable on many wearable PPG paths and is **not** part of the deployed contract.

---

## What's in this folder

```text
Self-supervised Physiological Biosensors/
├── Notebooks/          # Benchmark EDA + model training / validation
├── HF-Anxiety/         # Production FastAPI inference service (HF Space)
├── scripts/            # Weekly archive + personalization (GitHub Actions)
├── dashboard/          # React + Vite interactive demo
├── frontend/           # Static Aura Bio Monitor demo
└── requirements.txt    # Research / notebook dependencies
```

### Research notebooks

| Notebook | What it does | Status |
|----------|--------------|--------|
| `01_WESAD_EDA.ipynb` | WESAD chest-signal EDA → 10-feature windows, baseline z-norm | Complete |
| `02_AffectiveROAD_EDA.ipynb` | AffectiveROAD → same feature contract (external validation set) | Complete |
| `03_LSTM_AE_Training.ipynb` | Unmasked LSTM-AE, WESAD LOSO, AffectiveROAD external test | Complete · **deployed AE** |
| `04_Masked_LSTM_AE_Training.ipynb` | 20% input-mask ablation under the same protocol | Complete · ablation |
| `05_Forecasting_Module.ipynb` | Direct Ridge score forecast at +5 / +10 min | Complete · **deployed forecaster** |
| `09_Data_Hoarder_Pipeline.ipynb` | Prototype: Influx → Parquet → HF sensor archive | Scaffold |

### Live system

| Piece | Path / location | Role |
|-------|-----------------|------|
| Inference API | `HF-Anxiety/main.py` | Ingest, calibrate, predict, history, feedback |
| Model weights | `HF-Anxiety/models/` | Frozen AE + Ridge forecaster + metadata |
| Personalization | `scripts/personalize.py` | Weekly fine-tune → `unmasked_v2_{user_id}.pth` |
| Sensor archive | `scripts/archive_sensor_data.py` | Weekly Parquet dump to HF |
| CI schedule | `.github/workflows/weekly-physiological-pipeline.yml` | Sundays 02:15 UTC |
| Demo UI | `dashboard/` (React) · `frontend/` (static) | Simulated physiology + risk visuals |

---

## Phase 1 results (current 10-feature pipeline)

Evaluation design: train / LOSO on **WESAD** baseline-only windows, then **AffectiveROAD** as external stress test. Subjects **S3** and **S6** excluded from WESAD training.

### Unmasked LSTM-AE *(deployed)*

| Setting | AUROC | F1 | Notes |
|---------|------:|---:|-------|
| WESAD LOSO (mean) | **0.9979** | **0.9233** | Strong in-lab separation |
| AffectiveROAD pooled external | **0.8650** | 0.5783 | FAR ≈ **0.028** |
| AffectiveROAD macro (drives) | 0.8757 | 0.5362 | Cross-domain drop expected |

### Masked LSTM-AE *(ablation, not deployed)*

| Setting | AUROC | F1 |
|---------|------:|---:|
| WESAD LOSO (mean) | 1.0000 | 0.9233 |
| AffectiveROAD pooled | 0.8562 | 0.5419 |

Masking did **not** improve external F1 under the revised 10-feature protocol (unlike an earlier 11-feature / multi-dataset experiment). The live stack therefore stays on the **unmasked** AE.

### Direct Ridge forecasting *(deployed)*

Takes two consecutive 5-minute C1 score blocks and predicts scores at **+5** and **+10** minutes (logit-space residual vs persistence).

| Dataset | Horizon | Mean MAE improvement vs persistence |
|---------|---------|-------------------------------------|
| WESAD | +5 min | +0.0138 |
| WESAD | +10 min | +0.0307 |
| AffectiveROAD | +5 min | +0.0055 |
| AffectiveROAD | +10 min | +0.0406 |

Scores are **monotonic anomaly indices**, not calibrated probabilities. Bootstrap CIs on WESAD improvement include zero at both horizons — report that honestly in the paper.

---

## Deployment architecture

```text
ESP32-C3 chest strap
        │  10 features / minute
        ▼
Flutter research app
        │
        ▼
HF Space  Dewdu/physiological-anxiety-escalation
   FastAPI  (HF-Anxiety/)
        │
        ├── InfluxDB  physiological_metrics, norm_params, anxiety_feedback
        ├── Global AE + Ridge  (frozen, validated path)
        └── Personalized AE   (HF dataset Dewdu/physiological-anxiety-weights)
                ▲
                │ weekly GitHub Action
        scripts/personalize.py + archive_sensor_data.py
                │
                ▼
        Dewdu/physiological-sensor-archive  (Parquet)
```

### API surface (`HF-Anxiety`)

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/` | Health + model versions |
| `POST` | `/ingest` | Accept a worn, fresh 10-feature window |
| `POST` | `/set_norm_params/{user_id}` | Store baseline mean/std (+ optional calm calibration) |
| `GET` | `/predict/{user_id}` | C1 score + +5/+10 forecast (needs 10 contiguous minutes) |
| `GET` | `/history/{user_id}` | Daily aggregates / risk index history |
| `POST` | `/feedback/anxiety` | User confirmation / intervention feedback |
| `GET` | `/feedback/weekly/{user_id}` | 7-day feedback summary |

Predict statuses: `success` · `not_calibrated` · `buffering` · `stale`.

---

## Quick start

### Research notebooks

```bash
cd "Self-supervised Physiological Biosensors"
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Notebooks/
```

### Inference service (local)

```bash
cd HF-Anxiety
pip install -r requirements.txt
# set INFLUX_* and HF_TOKEN as needed
uvicorn main:app --reload --port 7860
```

### Demo dashboard

```bash
cd dashboard
npm install
npm run dev
```

Or open `frontend/index.html` for the static Aura Bio Monitor mock.

### Weekly personalization (manual)

```bash
cd "Self-supervised Physiological Biosensors"
pip install -r scripts/requirements.txt
# export INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, HF_TOKEN
python scripts/archive_sensor_data.py
python scripts/personalize.py
```

---

## Hardware target

| Sensing | Typical part | Measures |
|---------|--------------|----------|
| ECG / HRV | AD8232 | Cardiac rhythm, R–R |
| Respiration | BF350-3AA strain gauge | Thoracic expansion / BR |
| Motion | IMU (e.g. BMI160 / MPU6500 class) | Acc magnitude |
| Skin temp | DS18B20 | Peripheral temperature |
| MCU | ESP32-C3 | Acquisition + BLE / Wi-Fi uplink |

Feature quality gates in the API reject unworn straps, stale timestamps, out-of-range values, and inconsistent HR ↔ R–R pairs.

---

## What's next

- [ ] Ethics clearance (SLIIT) for naturalistic chest-strap collection
- [ ] Finalize wearable form factor (breadboard → strap)
- [ ] 4–6 week pilot with ~10–15 young adults
- [ ] Quantify real-world lift from weekly `unmasked_v2` personalization
- [ ] Keep fusion (Component 4) on the validated unmasked + Ridge path only

---

## Citation

Component of **R26-DS-012** — *A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders*, SLIIT, 2026.

**Student:** Sendanayake H.D. · IT22107596  
**Supervisors:** Prof. Samantha Thelijjagoda · Dr. Mahima Weerasinghe · Dr. Chathurie Suraweera

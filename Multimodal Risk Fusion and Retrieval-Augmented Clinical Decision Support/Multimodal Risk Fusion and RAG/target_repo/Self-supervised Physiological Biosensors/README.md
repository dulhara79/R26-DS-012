# 🧠 Personalized Self-Supervised Forecasting of Acute Anxiety Episodes Using Wearable Biosensors

![Project Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)
![Research Phase: Phase 1](https://img.shields.io/badge/Phase-1%20Completed-blue)

Welcome to the central repository for **Component 1 of Research Project R26-DS-012**. 

## 📌 The Core Vision

**Can we predict a panic attack before it happens, without ever teaching a machine what panic looks like?**

The objective of this research is to build a physiological anomaly detection and early warning forecasting system. Instead of relying on rare, hard-to-annotate "stress" data, this project takes a **self-supervised** approach. The model learns exclusively what *resting* physiology looks like. When acute autonomic destabilisation (a physiological precursor to anxiety) occurs, the model fails to reconstruct the signal. That calculated failure is our anomaly signal. 

Our target deployment is a custom, low-cost **ESP32-C3 chest-strap wearable** collecting ECG, respiration, motion, and skin temperature. Every algorithmic decision made here is strictly constrained by this specific hardware capabilities.

---

## ✅ What's Done (Phase 1)

Phase 1 focused on validating our approach using four massive benchmark datasets, ensuring robust cross-dataset generalization.

### 1️⃣ Data Pipeline & Feature Extraction
- **Locked 11-Feature Set**: Successfully engineered an immutable set of 11 physiological features (HRV, breathing rate, motion, temp) extracted over 60-second rolling windows.
- **Multi-Dataset Harmonization**: Standardized processing pipelines for **WESAD** (lab stress), **AffectiveROAD** (real-world driving stress), **PPG-DaLiA** (daily activities), and **EmoWear** (video-elicited emotions).

### 2️⃣ Model Development & Evaluation
- **Global LSTM Autoencoder (LSTM-AE)**: Trained exclusively on baseline data to flag physiological deviations. 
  - *Results:* **0.99 AUROC** on WESAD with **0% False Alarm Rate**. Combined AUROC of **0.97** across 25 subjects/drives.
- **Masked LSTM-AE Variant**: Introduced a masking technique during training.
  - *Impact:* Improved F1 scores by **20.7%** on noisier wearable data (wrist PPG), documenting a clear sensitivity-specificity tradeoff.
- **Strict Leave-One-Subject-Out (LOSO)** evaluation framework implemented for rigorous, unbiased validation.

### 3️⃣ Early Warning Forecasting Module
- Designed a secondary LSTM module taking recent anomaly scores to forecast stress onset probability.
- Achieved a combined forecasting AUROC of **0.88**.
- Demonstrated an average **Early Warning Time (EWT) of 11.83 minutes**, successfully exceeding our 5-10 minute target window for 24% of tested subjects.

---

## 🚀 What's to be Done

### ⏳ Immediate Next Steps (Wrapping up Phase 1)
- [ ] **Personalisation (Notebook 08)**: Fine-tune the global model weights for each individual subject. This will prove our central empirical claim: *Personalized self-supervised models outperform generic ones for individual anomaly detection.* 

### 🔮 Phase 2: Into the Real World (Future Work)
Phase 2 shifts from benchmark validation to real-world, clinical hardware deployment.
- [ ] **Ethics Approval**: Secure official SLIIT ethics clearance for human data collection.
- [ ] **Hardware Finalization**: Transition the custom ESP32-C3 chest strap from breadboard to a deployable, wearable strap format.
- [ ] **Participant Recruitment & Trial**: Conduct a 4-6 week naturalistic monitoring study with 10-15 young adults.
- [ ] **Real-World Fine-Tuning**: Apply our validated Phase 1 personalization pipelines to the newly collected real-world data to capture the gradual escalation of real-world anxiety.

---

## 📂 Repository Structure

Our workflow is contained within a series of modular Jupyter notebooks:

| Notebook | Purpose | Status |
|----------|---------|--------|
| `01_WESAD_EDA.ipynb` | WESAD dataset exploration and preprocessing | 🟢 Complete |
| `02_AffectiveROAD_EDA.ipynb` | AffectiveROAD dataset exploration and preprocessing | 🟢 Complete |
| `03_PPGDaLiA_EDA.ipynb` | PPG-DaLiA dataset exploration and preprocessing | 🟢 Complete |
| `04_EmoWear_EDA.ipynb` | EmoWear dataset exploration and preprocessing | 🟢 Complete |
| `05_LSTM_AE_Training.ipynb` | Global LSTM-AE, LOSO evaluation, EmoWear validation | 🟢 Complete |
| `06_Masked_LSTM_AE_Training.ipynb` | Masked variant training and ablation comparison | 🟢 Complete |
| `07_Forecasting_Module.ipynb` | Forecasting LSTM and Early Warning Time analysis | 🟢 Complete |
| `08_Personalisation.ipynb` | Fine-tuning per subject: Global vs Personalised | 🟡 Pending |

---
*R26-DS-012 | Component 1 | Sendanayake H.D. | IT22107596 | SLIIT | April 2026*

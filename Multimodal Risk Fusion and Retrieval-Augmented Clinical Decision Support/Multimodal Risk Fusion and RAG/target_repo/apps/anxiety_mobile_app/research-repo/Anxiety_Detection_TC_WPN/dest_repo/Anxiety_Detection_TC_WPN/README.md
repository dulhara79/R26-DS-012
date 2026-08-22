# TC-WPN: Temporal-Confidence Weighted Prototypical Network (v5)

[![Research](https://img.shields.io/badge/Research-Clinical--NLP-blue)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TC-WPN** is a specialized framework for **Few-Shot Learning (FSL)** on clinical notes, engineered for anxiety detection using the MIMIC-IV dataset. It addresses the unique challenges of clinical text—sparse labels, temporal dependencies, and high linguistic variability—through a multi-stage weighting and relation-based architecture.

---

## 🚀 Key Research Innovations

TC-WPN (v5) enhances traditional Prototypical Networks (ProtoNets) with three core modules:

### 1. Learnable Relation Module
Uses a **Neural Relation Module (MLP)** instead of simple geometric distances. This allows the model to learn a non-linear similarity metric, capturing the "entangled" nuances of clinical context that Cosine Similarity might miss.

### 2. Temporal Weighting Module
Clinical history is a sequence. TC-WPN computes **Recency** and **Regularity** weights:
- **Recency Decay**: Prioritizes notes closer to the current clinical event.
- **Visit Regularity**: Weights context based on the consistency of the patient's medical history.

### 3. Confidence Weighting Module
Refines prototypes using **Entropy-based weighting**. It automatically down-weights support examples that yield uncertain (high-entropy) predictions, ensuring the "class anchors" represent only the most confident clinical patterns.

### 4. Clinical Data Augmentation (New)
To prevent overfitting in low-resource settings, we've implemented a **ClinicalAugmenter**. This module performs **sentence shuffling** within notes while strictly **protecting sensitive medical terms** (e.g., "denies", "anxiety", "no_...") to preserve the ground truth label.

---

## 📂 Project Structure

```text
tc_wpn/
├── config/              # Centralized settings and path management
│   └── settings.py      # Environment and model configurations
├── mimic_processed/     # Processed .pkl files (ignored by git, metadata included)
├── notebooks/           # Analysis and Jupyter exploration
│   └── explore_clinical_notes.py  # Analysis of note quality
├── scripts/             # Operational entry points
│   ├── extract_data.py  # Cohort extraction from MIMIC-IV
│   └── explore_data.py  # Dataset statistics and temporal features
├── src/                 # Main source code package
│   └── tc_wpn/
│       ├── data/        # Clinical cohort identification & cleaning
│       ├── models/      # Core architecture (Embedder, Relation Module)
│       ├── sampler/     # Episode-based sampling with Augmentation
│       └── utils/       # Utility functions and validation logic
├── .env                 # Local environment variables
└── requirements.txt     # Python dependencies
```

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dulhara79/tc_wpn.git
cd tc_wpn
```

### 2. Setup Environment
We recommend using a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Data Pipeline

### 1. Extraction
Extract anxiety-related cohorts and clean clinical text:
```bash
python scripts/extract_data.py
```
*Note: Ensure your MIMIC-IV paths are set in `config/settings.py` or `.env`.*

### 2. Exploration
Analyze the extracted dataset and temporal distributions:
```bash
python scripts/explore_data.py
```

---

## 🧠 Model Training

The model uses `emilyalsentzer/Bio_ClinicalBERT` as its backbone and is optimized for 2-way 5-shot episodes.

- **Sampler Config**: See `src/tc_wpn/sampler/episode.py` for meta-learning settings.
- **Kaggle Notebook**: A complete training pipeline is available in `notebooks/tc-wpn-complete-kaggle-training-notebook-v4.ipynb`.

---

## 📝 Citation
> Kaushalya, D. (2026). TC-WPN: Temporal-Confidence Weighted Prototypical Networks for Clinical Anxiety Detection.

## 📄 License
This project is licensed under the MIT License.
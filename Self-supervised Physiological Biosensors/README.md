# Graph-Based Spatio-Temporal Behavioral Phenotyping
### For Personalized Anxiety Vulnerability Mapping

A graph neural network pipeline that models each participant's daily
movement and behavioral patterns as a directed graph, then predicts
anxiety vulnerability scores and clusters users into interpretable
behavioral phenotypes.

---

## Overview

| Stage | What it does |
|-------|-------------|
| **Preprocessing** | Loads GPS, activity, stress EMA, conversation, and phone-lock data |
| **GPS Cleaning** | Removes noisy points; detects stay-point clusters via DBSCAN |
| **Contextual States** | Merges modalities into composite state labels (location × time × activity) |
| **Graph Building** | Constructs a directed behavioral graph per participant |
| **Risk Profiling** | Computes hourly risk profiles and detects high-risk windows |
| **GATv2 Model** | Two-head graph attention network: vulnerability score + hourly risk vector |
| **Cross-Validation** | Stratified 5-fold CV with SMOTE oversampling inside each fold |
| **Phenotyping** | K-Means clustering of GNN embeddings → behavioral phenotypes |
| **Baselines** | Logistic Regression, Random Forest, Gradient Boosting comparison |
| **Ablation** | Feature-group ablation study |

---

## Folder structure

```
graph-behavioral-phenotyping/
├── config.py                       # All paths and hyperparameters
├── main.py                         # Full pipeline runner
├── requirements.txt
│
├── preprocessing/
│   ├── data_loader.py              # Per-modality loaders + batch loader
│   ├── gps_cleaner.py              # Speed filter + DBSCAN stay-points
│   └── contextual_states.py       # Temporal merge → contextual state labels
│
├── graph/
│   ├── graph_builder.py            # NetworkX behavioral graph construction
│   ├── graph_to_pyg.py             # NetworkX → PyTorch Geometric Data
│   └── risk_profiler.py            # Hourly risk profiles + window detection
│
├── models/
│   ├── gatv2_model.py              # AnxietyGATv2 architecture
│   └── loss.py                     # BCE + MSE combined loss
│
├── training/
│   ├── trainer.py                  # train_epoch, eval_epoch, train_fold (+ SMOTE)
│   ├── cross_validation.py         # Stratified k-fold CV
│   └── baselines.py                # Classical ML baselines
│
├── phenotyping/
│   ├── phenotyper.py               # Final model training, embedding extraction, K-Means
│   └── inference.py                # Single-user inference
│
├── evaluation/
│   ├── ablation.py                 # Feature-group ablation
│   └── population_heatmap.py      # Cluster × hour risk matrix
│
├── visualization/
│   ├── plots.py                    # Model comparison, UMAP, confusion matrix, heatmap
│   └── risk_profile_plot.py       # Per-user daily risk profile plots
│
├── notebooks/
│   └── digital_phenotyping_latest.ipynb   # Original Colab notebook
│
├── data/
│   └── README.md                   # Dataset download & layout instructions
│
├── outputs/                        # Auto-created — plots & CSV saved here
└── saved_models/                   # Auto-created — model weights saved here
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
# PyTorch Geometric requires a separate install step:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

### 2. Download the dataset

See [`data/README.md`](data/README.md) for the required folder layout.

### 3. Set your paths

Either edit `config.py` directly, or use environment variables:

```bash
export DATASET_PATH="/path/to/studentlife/dataset/"
export OUTPUT_DIR="/path/to/outputs/"
export MODELS_DIR="/path/to/saved_models/"
```

### 4. Run the full pipeline

```bash
python main.py
```

This will run all 20 pipeline stages and save model weights, CV results,
per-user vulnerability JSON, and all plots to `OUTPUT_DIR` / `MODELS_DIR`.

### 5. Run inference on a single participant

```python
from phenotyping.inference import predict_user

result = predict_user(
    uid="u00",
    model=final_model,          # loaded AnxietyGATv2
    user_graphs=user_graphs,
    all_profiles=all_profiles,
    label_map=label_map,
    uid_list=uid_list,
    clusters=clusters,
    phenotypes=PHENOTYPES,
    all_conversation=all_conversation,
    all_phonelock=all_phonelock,
    device=device,
)
```

---

## Model architecture

```
Node features (9) ──► GATv2Conv (4 heads) ──► BN ──► GATv2Conv (1 head) ──► BN
                                                              │
                                                    GlobalMeanPool + GlobalMaxPool
                                                              │
                                            ┌─────────────────┴──────────────────┐
                                     vuln_head (scalar)                  hrw_head (24-dim)
                               vulnerability score [0,1]           hourly risk probabilities
```

---

## Outputs

| File | Description |
|------|-------------|
| `gatv2_final_<ts>.pt` | Final trained model weights |
| `kmeans_<ts>.pkl` | K-Means model + phenotype assignments |
| `embeddings_<ts>.npy` | Graph-level embeddings (N × hidden*2) |
| `cv_results_<ts>.csv` | Per-fold CV metrics |
| `vulnerability_output_<ts>.json` | Per-user vulnerability scores + phenotype |
| `model_comparison.png` | GNN vs baselines bar chart |
| `phenotypes_umap.png` | UMAP scatter of behavioral phenotypes |
| `confusion_matrix.png` | Combined CV confusion matrix |
| `ablation_study.png` | Feature-group ablation bar chart |
| `population_risk_heatmap.png` | Cluster × hour risk heatmap |
| `risk_profile_<uid>.png` | Per-user daily risk profile |

---

## Citation



> R. Wang et al., "StudentLife: Assessing Mental Health, Academic Performance
> and Behavioral Trends of College Students using Smartphones," *UbiComp 2014*.

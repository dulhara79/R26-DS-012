# Component 2 — Spatio-Temporal Graph Learning for Anxiety Vulnerability Mapping

Final Component 2 implementation for **R26-DS-012**.

This repository evaluates whether passive smartphone sensing represented as a
spatio-temporal graph can predict anxiety vulnerability under a strict,
participant-grouped and cross-cohort evaluation protocol.

> **Important:** This final v8 pipeline replaces the earlier StudentLife,
> GPS/DBSCAN, hourly-risk and phenotype-deployment pipeline as the source of
> truth for the dissertation/paper.

---

## Final methodology

### Dataset

**GLOBEM multi-year dataset**

- `INS-W_1` — 2018
- `INS-W_2` — 2019
- `INS-W_3` — 2020
- `INS-W_4` — 2021

`INS-W_1` is used only for model/hyperparameter selection.  
`INS-W_2`, `INS-W_3` and `INS-W_4` are the final held-out evaluation cohorts.

### Target

Released binary:

`anx_weekly_subscale`

This is the PHQ-4 anxiety subscale / GAD-2-derived binary label supplied in
GLOBEM.

### Behavioral representation

For every labelled week:

1. Take the preceding **28 days** of passive sensing.
2. Use four daily segments:
   - morning
   - afternoon
   - evening
   - night
3. Select common RAPIDS feature bases across all four segments.
4. Cap selection at 10 feature bases per sensor family.
5. Build a temporal lattice graph:
   - node = one available `day × time-segment` cell
   - within-day edges connect adjacent time segments
   - across-day edges connect the same time segment on consecutive days
6. Node features include both behavioral values and an explicit missingness mask.

The executed final notebook selected **40 feature bases**, producing
**80 raw node features** (40 values + 40 missingness indicators).

---

## Leakage-free evaluation

The final protocol prevents the same participant from appearing in both train
and test folds.

Within every training fold:

- population normalization is fitted on training data only
- an inner validation set is created from training data only
- early stopping uses inner validation only
- calibration uses inner validation only
- classification threshold selection uses inner validation only
- the test fold is evaluated once after all choices are fixed

The final model is compared against a **prevalence-preserving,
participant-level permutation null**.

---

## Final result

| Quantity | Value |
|---|---:|
| Held-out GATv2 AUROC | **0.5205** |
| 95% participant-clustered CI | **0.485–0.560** |
| 50-permutation null mean | **0.4991** |
| Permutation SD | 0.0333 |
| Empirical p-value | **0.255** |
| AUPRC | 0.2270 |
| Brier score | 0.1847 |
| Brier skill vs constant predictor | **-0.0752** |
| Calibration resolution | 0.00064 |
| ECE | 0.0840 |

### Interpretation

The final held-out GATv2 model is **not distinguishable from chance**.

---

## Graph vs flat behavioral baselines

| Model | AUROC |
|---|---:|
| Logistic Regression | 0.5458 |
| Random Forest | 0.5617 |
| Gradient Boosting | **0.5681** |
| GATv2 | 0.5205 |

The graph representation did not outperform the simpler flat behavioral
representations.

---

## Other final checks

### Leave-one-cohort-out

| Model | Pooled AUROC | LOCO AUROC | Generalization gap |
|---|---:|---:|---:|
| GATv2 | 0.5529 | 0.5036 | +0.0493 |
| Logistic Regression | 0.5832 | 0.5553 | +0.0279 |
| Random Forest | 0.5654 | 0.5417 | +0.0237 |
| Gradient Boosting | 0.5503 | 0.5385 | +0.0117 |

The pre-registered cohort-shift robustness hypothesis for the graph was
**not supported**.

### Feature-breadth control

A histogram gradient boosting model using **5,952 engineered features** achieved:

- AUROC = **0.5335**
- 95% CI = **0.496–0.571**

This is treated as a **feature-breadth control**, not a mathematical ceiling.

### Compliance ablation

| Representation | AUROC |
|---|---:|
| Availability + volume only | 0.5172 |
| Behavioral values only | 0.5616 |
| Both | 0.5681 |

---

## Deployment decision

The earlier repository exposed:

- vulnerability score
- risk level
- hourly high-risk window
- phenotype cluster

Those are historical exploratory outputs and **must not be treated as validated
clinical outputs of the final v8 study**.

The final decision-curve/calibration evidence supports:

```text
Component 2 fusion weight = 0.0
```

The Component 2 score may be retained for research logging and provenance, but
should not be surfaced as a clinical anxiety-risk score.

---

## Folder structure

```text
graph-behavioral-phenotyping/
├── data/
│   └── README.md
├── evaluation/
│   ├── __init__.py
│   ├── ablation.py
│   ├── bootstrap_ci.py
│   ├── cohort_shift.py
│   ├── compliance_ablation.py
│   ├── decision_curve.py
│   ├── delong.py
│   ├── feature_breadth_control.py
│   ├── label_reliability.py
│   ├── permutation_test.py
│   └── population_heatmap.py
├── graph/
│   ├── __init__.py
│   ├── graph_builder.py
│   ├── graph_to_pyg.py
│   └── risk_profiler.py
├── models/
│   ├── __init__.py
│   ├── gatv2_model.py
│   └── loss.py
├── notebooks/
│   ├── component2_consolidated_v8.ipynb
│   └── archive/
│       └── README.md
├── phenotyping/
│   ├── __init__.py
│   ├── inference.py
│   └── phenotyper.py
├── preprocessing/
│   ├── __init__.py
│   ├── contextual_states.py
│   ├── data_loader.py
│   ├── feature_selector.py
│   └── gps_cleaner.py
├── training/
│   ├── __init__.py
│   ├── baselines.py
│   ├── cross_validation.py
│   └── trainer.py
├── visualization/
│   ├── __init__.py
│   ├── plots.py
│   └── risk_profile_plot.py
├── results/
│   ├── README.md
│   └── results_v8.json
├── tests/
│   ├── test_graph_construction.py
│   ├── test_no_participant_leakage.py
│   └── test_permutation_prevalence.py
├── .gitignore
├── LEGACY.md
├── README.md
├── config.py
├── main.py
└── requirements.txt
```

---

## Source of truth

The final numerical results come from:

`notebooks/component2_consolidated_v8.ipynb`

Do **not** re-tune the model after inspecting the final held-out cohorts.

---

## Dataset citation

X. Xu et al., “GLOBEM: Multi-year datasets for longitudinal human behavior
modeling generalization,” *Advances in Neural Information Processing Systems*,
vol. 35, 2022.

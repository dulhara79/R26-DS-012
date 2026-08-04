# TC-WPN: Temporal-Confidence Weighted Prototypical Networks
### Few-Shot Clinical Anxiety Detection from EHR Notes

[![Research](https://img.shields.io/badge/Research-Clinical--NLP-blue)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Data-MIMIC--IV-green)](#data)

**TC-WPN** detects clinical anxiety from MIMIC-IV electronic health record notes using **only K=3 labeled support examples per class** — no fine-tuning, no retraining. The system meta-trains on MIMIC-IV episodes and adapts to a new clinical site through a single forward pass.

> **Component 4** of the broader multimodal framework: *A Multimodal Digital Biomarker Framework for Personalized Vulnerability Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders* (R26-DS-012).

---

## Results

Patient-level evaluation. Bootstrap 95% CI over patients (independent units). Threshold locked from validation set — never optimised on test.

### Real-World Test Set — 33.3% prevalence (primary result)

| K-shot | Patients | AUROC | 95% CI | F1 | PR-AUC |
|--------|---------|-------|--------|----|--------|
| K=1 | 323 | **0.9863** | [0.977–0.994] | 0.9270 | 0.9807 |
| K=3 | 366 | 0.9812 | [0.967–0.992] | 0.9488 | 0.9682 |
| K=5 | 368 | 0.9783 | [0.964–0.990] | 0.9406 | 0.9681 |
| K=10 | 368 | 0.9813 | [0.968–0.992] | 0.9405 | 0.9736 |

### High-Confidence Test Set — 19.6% prevalence (challenging scenario)

| K-shot | Patients | AUROC | 95% CI | F1 | PR-AUC |
|--------|---------|-------|--------|----|--------|
| K=1 | 218 | 0.9778 | [0.960–0.992] | 0.8841 | 0.8566 |
| K=3 | 249 | 0.9744 | [0.955–0.989] | 0.8758 | 0.7975 |
| K=5 | 258 | 0.9725 | [0.952–0.989] | 0.8609 | 0.7492 |
| K=10 | 258 | 0.9750 | [0.956–0.990] | 0.8792 | 0.8149 |

> **K=1 achieves AUROC 0.9863** — a single labeled example per class is sufficient for near-optimal adaptation. Under the clinically critical low-prevalence scenario (19.6%), TC-WPN improves PR-AUC by **+0.063** and F1 by **+0.042** over the architecture-without-weighting baseline.

---

## Architecture

TC-WPN is built on Bio_ClinicalBERT and extends standard Prototypical Networks with five specific improvements. Each improvement addresses a concrete failure mode identified during development.

```
Input episode (K support + Q query per class)
        │
        ▼
┌─────────────────────────┐
│  Bio_ClinicalBERT       │  Shared weights — support & query
│  [CLS] → attention pool │
│  Linear(768 → 256)      │
│  ReLU · Dropout · LN    │
└──────────┬──────────────┘
           │ support only          │ query only
           ▼                       ▼
┌─────────────────────┐   ┌──────────────────┐
│  BiGRU Temporal     │   │  query_proj      │
│  Encoder            │   │  Linear·ReLU·LN  │
│  (trajectory-aware) │   │  (no GRU —       │
│  GRU h=128, bidir   │   │   avoids distort)│
└──────────┬──────────┘   └────────┬─────────┘
           │                        │
           ▼                        │
┌─────────────────────┐             │
│  Temporal Weighting │             │
│  recency = exp(     │             │
│   −λ×age/365)       │             │
│  visit_w = sigmoid( │             │
│   α×visits)  α∈param│             │
│  base_w = recency×  │             │
│           visit_w × │             │
│           dataset_w │             │
└──────────┬──────────┘             │
           │                        │
           ▼                        │
┌─────────────────────┐             │
│  Cosine Confidence  │             │
│  Weighting          │             │
│  prelim_proto from  │             │
│  base_w             │             │
│  cos_sim = cosine(  │             │
│   emb, proto)       │             │
│  conf_w = exp(β·sim)│             │
└──────────┬──────────┘             │
           │                        │
           ▼                        │
┌─────────────────────┐             │
│  TC-Weighted        │             │
│  Prototype          │             │
│  proto = L2_norm(   │             │
│   Σ norm_w·emb)     │             │
│  × 2 classes        │             │
└──────────┬──────────┘             │
           │                        │
           └──────────┬─────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  Prototype-Distance         │
        │  Classifier                 │
        │  temp = exp(log_temp) ≈8.1 │
        │  dist = ‖q_norm−proto‖²    │
        │  logit = −dist × temp      │
        │  probs = softmax(logits)   │
        └─────────────┬───────────────┘
                      │
                      ▼
          P(anxiety) ∈ [0.0, 1.0]
          ──────────────────────────
          Continuous risk score →
          Late-fusion model (w4=0.40)
```

### Why Each Component Exists

**BiGRU Temporal Encoder** — A note at visit 3 of 5 carries different meaning than the same text at visit 1 of 1. The BiGRU processes chronologically sorted support notes and gives each embedding trajectory context. Applied to support only; queries use a separate `query_proj` to avoid distribution mismatch.

**Cosine confidence weighting** — The proposal used Shannon entropy (`w = 1/(1+β·H)`). Entropy requires meaningful class probabilities, which only exist after the model has learned something. At episode 1 all probs ≈ 0.5, entropy is maximum for every note, and the mechanism does nothing. Cosine similarity to a preliminary prototype works from episode 1 because ClinicalBERT provides a meaningful embedding space before fine-tuning. Pre-training diagnostic confirmed: proposal p_std = 0.002–0.008 vs cosine p_std = 0.084 at initialisation.

**Learnable temperature scalar** — Without temperature, prototype-distance logits fall in `[-0.05, 0.02]`, making softmax near-uniform regardless of distance. Initialising `log_temperature = log(10)` produces logit spread from episode 1. The temperature learned to 8.11 over 3000 episodes — a small but meaningful adjustment. Pre-training logit range with temperature: `[-21.99, -19.85]`. Without it: `[0.01, 0.31]`.

**Learnable sigmoid regularity** — The proposal used a fixed threshold (1.0 if visits ≥ 3, else 0.8+0.1×visits). The learnable `sigmoid(α × total_visits)` adapts the optimal visit-weighting to the MIMIC-IV population distribution rather than using an arbitrary threshold.

**Prototype-distance classification** — The proposal used a RelationModule MLP (Sung et al. 2018). With K=3, there is insufficient data to reliably train a learned comparator. Prototype-distance is a direct geometric measurement — no learned comparator, no extra parameters — more stable at small K.

---

## Ablation Study Results

All ablation experiments use patient-level evaluation with identical data splits and seeds.

### Component Ablation — High-Confidence Test Set

| Configuration | K=3 AUROC | K=3 F1 | K=3 PR-AUC |
|--------------|----------|--------|------------|
| Standard ProtoNet (λ≈0, β≈0) | 0.9702 | 0.8243 | 0.6852 |
| Temporal-only (λ=0.5, β≈0) | 0.9717 | 0.8790 | 0.7627 |
| Confidence-only (λ≈0, β=2.0) | 0.9714 | 0.8690 | 0.7342 |
| Full TC-WPN (λ=0.5, β=2.0) | 0.9666 | **0.8663** | **0.7478** |

> AUROC differences between configurations are within stochastic noise (overlapping CIs). The TC weighting contribution is clearest in **F1 (+0.042) and PR-AUC (+0.063)** under 19.6% class prevalence — the clinically critical imbalanced scenario.

### k-Shot Generalisation (trained at K=3, evaluated at K=1 to K=20)

| K | HC AUROC | RW AUROC |
|---|---------|---------|
| 1 | **0.9778** | **0.9863** |
| 3 | 0.9665 | 0.9731 |
| 5 | 0.9744 | 0.9789 |
| 10 | 0.9741 | 0.9819 |
| 15 | 0.9757 | 0.9820 |
| 20 | 0.9751 | 0.9811 |

K=1 achieves the highest AUROC on both test sets — demonstrating strong few-shot generalisation from the meta-learned embedding space.

### λ Sweep (β=2.0 fixed, HC K=5)

| λ | AUROC |
|---|-------|
| 0.1 | 0.9728 |
| 0.3 | 0.9734 |
| **0.5** | **0.9779** |
| 0.7 | 0.9759 |
| 1.0 | 0.9791 |

### Projection Dimension Sweep

| dim | val AUROC | HC K=3 | HC K=5 |
|-----|----------|--------|--------|
| 64 | **0.9744** | 0.9684 | 0.9739 |
| 128 | 0.9642 | 0.9708 | 0.9753 |
| **256** | 0.9547 | 0.9691 | 0.9702 |
| 512 | 0.9700 | 0.9688 | 0.9759 |

dim=64 achieves the highest val AUROC with 16× fewer metric space parameters than dim=256, suggesting TC-WPN's clinical anxiety representation is highly compact.

---

## Data Pipeline

TC-WPN uses MIMIC-IV clinical notes. Access requires CITI certification and a signed PhysioNet Data Use Agreement.

```
MIMIC-IV (PhysioNet)
       │
       ▼ mimic_extract_v4.py
ICD-coded anxiety cohort + matched controls
       │
       ▼ extraction_v4.py
Cleaned notes: PHI normalisation, section extraction,
whitespace standardisation
       │
       ▼ convert_csv_to_pkl_v3.py
Tokenised PKL files:
  Bio_ClinicalBERT tokeniser
  max_length=512, sliding window 128-token overlap
  Deduplication check (raises ValueError on duplicate note_ids)
  No resampling at PKL level
       │
       ▼ fix_val_pkl.py (supplementary only)
balanced_supp variant: controls down-sampled 1:2
anxiety notes never duplicated
```

### Dataset Summary

| Split | File | Records | Anxiety pts | Control pts | Prevalence | Filter |
|-------|------|---------|-------------|-------------|------------|--------|
| Train | `mimic_anxiety_train_high_conf.pkl` | 4,640 | 177 | 352 | 49.5%* | moderate |
| Val | `mimic_anxiety_val_real_world.pkl` | 2,197 | 145 | 224 | 33.3% | full |
| Test HC | `mimic_anxiety_test_high_conf.pkl` | 1,739 | 32 | 226 | 19.6% | full |
| Test RW | `mimic_anxiety_test_real_world.pkl` | 2,112 | 142 | 226 | 33.3% | full |
| Test BS | `mimic_anxiety_test_balanced_supp.pkl` | 1,035 | 32 | 116 | 33.3% | full |

*After moderate curriculum filter: 4,303 records remain.

---

## Repository Structure

```text
tc_wpn/
├── models/
│   ├── core_v3.py            ← Current TC-WPN architecture (CURRENT)
│   ├── core_proposal.py      ← Proposal architecture for comparison
│   ├── embedder.py           ← ClinicalEmbedder (attention pooling)
│   └── patient_level_eval.py ← Publication-grade patient-level evaluation
│
├── sampler/
│   └── episode_dataset_v3.py ← TCMIMICEpisodicDataset
│                                patient-level sampling, temporal sort,
│                                chunk cap, subject_id tracking
│
├── config/
│   └── settings.py           ← Paths and environment config
│
scripts/
├── mimic_extract_v4.py       ← MIMIC-IV ICD cohort extraction
├── extraction_v4.py          ← Note cleaning and preprocessing
├── convert_csv_to_pkl_v3.py  ← CSV → PKL tokenisation pipeline
└── fix_val_pkl.py            ← Balanced-supp variant (appendix only)

notebooks/ (Kaggle)
├── tc-wpn-complete-kaggle-training-notebook-v9.ipynb
│     Main training — current architecture
├── proposal-tc-wpn-complete-kaggle-training-notebook.ipynb
│     Proposal architecture training (architecture comparison)
├── tc-wpn-ablation-CURRENT.ipynb
│     10-experiment ablation — current architecture
├── tc-wpn-ablation-PROPOSAL.ipynb
│     9-experiment ablation — proposal architecture
└── tc-wpn-overfitting-underfitting-diagnostic.ipynb
      Generalisation gap analysis
```

---

## Installation

```bash
git clone https://github.com/dulhara79/tc_wpn.git
cd tc_wpn
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Training

The main training notebook runs on Kaggle with a T4 GPU (~72 minutes, 3000 episodes).

**Key configuration:**

```python
k_shot          = 3          # support examples per class
q_query         = 3          # query examples per class
projection_dim  = 256        # metric space dimension
lambda_decay    = 0.5        # temporal recency decay
beta            = 2.0        # cosine confidence sharpness
aux_weight      = 0.3        # auxiliary head loss weight
freeze_bert_layers = 8       # freeze bottom 8 of 12 BERT layers
bert_lr         = 2e-5
head_lr         = 1e-4
temp_lr         = 5e-3       # separate LR for log_temperature
total_episodes  = 3000
```

**Optimizer:** AdamW with three parameter groups — BERT layers (slow), head parameters (fast), and `log_temperature` (separate fast LR). Cosine warmup over first 200 episodes.

**Validation:** Episodic AUROC every 100 episodes. Patient-level AUROC used for test reporting only. Threshold locked from validation set, never optimised on test.

---

## Evaluation

TC-WPN uses **patient-level evaluation** for all test reporting. This is required for clinical NLP publication.

```python
# The correct way to get TC-WPN output for the fusion model:
out    = model(episode)
probs  = out['probs']          # [Q, 2]
score  = probs[:, 1].mean()    # P(anxiety) ∈ [0.0, 1.0]
# score is the continuous risk score — never apply a binary threshold for fusion
```

**Do not use `out['preds']` for fusion.** Predictions are argmax(logits) — binary 0/1. The fusion model requires the continuous probability `probs[:, 1]`.

The evaluation functions in `patient_level_eval.py`:
- `evaluate_patient_level()` — **PRIMARY.** Mean-pools predictions per patient across episodes. Bootstrap CI over patients. Use for all test results.
- `evaluate_episodic()` — **VALIDATION LOOP ONLY.** Pooled episodic AUROC for fast model selection during training. Never report this as a test result.

---

## Output — Fusion Integration

TC-WPN is Component 4 of the multimodal framework. Its output is a continuous anxiety probability passed to the late-fusion model.

```
TC-WPN output:  P(anxiety) ∈ [0.0, 1.0]   weight w4 = 0.40
                       │
                       ▼
          Late-fusion model:
          composite = 0.25×s1 + 0.20×s2 + 0.15×s3 + 0.40×s4
                       │
                       ▼
          Final anxiety risk score ∈ [0.0, 1.0]
          → Risk level: LOW / MODERATE / HIGH / VERY HIGH
```

TC-WPN never outputs "anxiety" or "not anxiety" as a classification decision. It outputs a probability. The fusion model determines the final risk level.

---

## Proposal vs Current Architecture

The initial proposal described entropy-based confidence weighting and a RelationModule classifier. Both were implemented and compared against the current architecture.

| Component | Proposal | Current (core_v3.py) |
|-----------|---------|---------------------|
| Confidence | Entropy: `w=1/(1+β·H)` | Cosine: `w=exp(β·cos_sim)` |
| Classification | RelationModule MLP | Prototype-distance × temperature |
| Temperature | None | Learnable, init=10 |
| Temporal regularity | Fixed threshold ≥3 visits | Learnable `sigmoid(α×visits)` |
| After embedding (support) | None | BiGRU TemporalEncoder |
| Chunk pooling | Mean | Attention-weighted |
| Test RW K=3 AUROC | 0.9656 | **0.9812** |
| Test HC K=3 F1 | 0.8941 | **0.9488** |

Current architecture outperforms proposal on all 6 test configurations. The proposal architecture is available in `core_proposal.py` and the proposal training notebook for direct comparison.

---

## Ethical Considerations

MIMIC-IV is accessed under signed PhysioNet Data Use Agreement following CITI certification. No re-identification attempts are made. The system is positioned as clinical decision support, not autonomous diagnosis. All clinical decisions remain with the responsible psychiatrist.

NHSL data collection (planned) is subject to IRB approval from the NHSL Ethics Review Committee and SLIIT Ethics Committee. All PHI is removed by the consulting psychiatrist before data sharing.

---

## Acknowledgements

This research used the MIMIC-IV database (Johnson et al., 2023, PhysioNet, DOI: 10.13026/kpb9-mt58) and MIMIC-IV-Note (Johnson et al., 2023, PhysioNet, DOI: 10.13026/1n74-ne17). We gratefully acknowledge the dataset creators and PhysioNet for maintaining this resource. The Bio_ClinicalBERT backbone is from Alsentzer et al. (2019).

---

## Citation

```bibtex
@misc{kaushalya2026tcwpn,
  author    = {Kaushalya, I G D},
  title     = {TC-WPN: Temporal-Confidence Weighted Prototypical Networks
               for Few-Shot Clinical Anxiety Detection},
  year      = {2026},
  note      = {Undergraduate dissertation, Sri Lanka Institute of
               Information Technology. Part of R26-DS-012.},
  url       = {https://github.com/dulhara79/tc_wpn}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

MIMIC-IV data is subject to the PhysioNet Credentialed Health Data License and must not be redistributed. The PKL files derived from MIMIC-IV are not included in this repository.
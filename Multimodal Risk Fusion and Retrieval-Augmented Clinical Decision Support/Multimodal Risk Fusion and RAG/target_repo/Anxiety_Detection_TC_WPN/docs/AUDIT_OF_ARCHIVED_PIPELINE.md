# Audit of the archived TC-WPN pipeline

Audited commit: `0804faa` on `main`
Audit date: 2026-08-05
Scope: `Anxiety_Detection_TC_WPN/` at that commit

Every finding below cites the file and construct it comes from. Findings are
graded by whether they affect the *validity* of a reported number (V) or the
*interpretation* of it (I).

---

## V1 — The label was a function of the note text

`scripts/extract_data_final.py`, STEP 11:

```python
anx_rows = merged[anxiety_mask].copy()
anx_has_content = anx_rows.apply(gate_anxiety_note, axis=1)
anx_kept = anx_rows[anx_has_content]
```

`gate_anxiety_note` calls `has_psychiatric_content()`
(`src/tc_wpn/data/extraction.py`), which requires 1–2 hits from a keyword list
whose first six entries are `anxiety, anxious, panic, generalized anxiety, gad,
worry`.

The gate is applied to positives only; the comment states the intent plainly:
*"Controls are NOT filtered."*

Consequence: a positive note that does not contain anxiety vocabulary was
deleted from the corpus. A negative note was not. The presence of the word
became close to a sufficient statistic for the label, in train **and in test**,
because this ran before the split at STEP 14.

## V2 — The test set was filtered by a text-derived score

Same file, STEP 15:

```python
test_hc = test_df[
    ((test_df["has_anxiety"] == 1) & (test_df["label_confidence"] >= 0.7))
    | ((test_df["has_anxiety"] == 0) & (test_df["label_confidence"] >= 0.9))
]
```

`label_confidence` comes from `assign_anxiety_confidence()` (regex over the
note) for positives and `penalize_control_noise()` (regex over the note) for
negatives. Reading the two functions together:

- a positive note reaches ≥ 0.7 by matching an anxiety regex (or by the
  prescription/OMR override);
- a negative note falls to 0.2 or 0.6 precisely *when it mentions anxiety*, so
  the ≥ 0.9 filter removes negatives that mention it.

The `test_high_conf` set is therefore, to a first approximation, *positives that
say "anxiety" versus negatives that do not*. `mimic_anxiety_test_high_conf.pkl`
is the split the headline numbers were computed on.

## V3 — Support/query patient leakage inside every episode

`src/tc_wpn/sampler/episode_dataset.py`, `_build_class_examples()`:

```python
selected.extend(chosen)          # up to max_notes_per_patient=3 notes per patient
...
return selected[: self.k_shot], selected[self.k_shot :]
```

The support/query boundary is a positional cut through a list built by walking
patients and taking up to three consecutive notes each. Whenever a patient's
block straddles index `k_shot`, that patient is in both sides.

The 20-attempt retry in `sample_episode()` does not catch this: it compares
`note_id` sets **across classes**, and the classes are disjoint by construction,
so the check passes trivially.

Replaying the exact selection logic (5,000 episodes/class, `q_query=5`,
`max_notes_per_patient=3`, notes-per-patient drawn uniformly from 2–8):

| K | episodes with a patient on both sides | query notes from a support patient |
|---:|---:|---:|
| 1 | 100.0 % | 36.8 % |
| 3 | 16.7 % | 6.1 % |
| 5 | **71.3 %** | 14.7 % |
| 10 | 89.3 % | 28.1 % |

K = 5 is the reported configuration. The exact percentages depend on the real
notes-per-patient distribution, but the defect is structural, not distributional.
Reproduce with `tests/test_episode_leakage.py::test_archived_sampler_leaks`.

## V4 — Medication- and score-derived labels

`load_prescription_confirmed_subjects()` sets `label_confidence = 1.0` for any
patient prescribed one of 18 drugs including `lorazepam`, `diazepam`,
`midazolam`, and `hydroxyzine`. These are given for alcohol withdrawal,
procedural sedation, nausea, and pruritus far more often than for an anxiety
disorder. Combined with V2, drug names in the note become label-predictive in
the test set. `load_omr_gold_subjects()` has the same structure with GAD-7/PHQ.

## V5 — MIMIC-III was in the training corpus

`extract_data_final.py` STEP 6/7 concatenates MIMIC-III cohorts into
`all_cases`, and STEP 8 loads MIMIC-III notes into the same corpus that
produces `mimic_anxiety_train_*.pkl`.

`notebooks/tc-wpn-mimic-iii-external-validation (3).ipynb` then describes itself
as *"Trains on MIMIC-IV only … the model never sees MIMIC-III during training"*
and reports AUROC 0.9396 as external validation.

Whether the specific Kaggle PKLs used by that notebook were built by this
version of the extractor cannot be determined from the repository alone — the
PKLs are not committed. **This must be resolved before any external-validation
claim is made.** If the PKLs came from this extractor, the claim is void.

Independently of that: MIMIC-III and MIMIC-IV overlap in patients and time at
BIDMC. "Cross-dataset transfer" is the defensible framing; "external validation"
is not.

## I1 — Note type is confounded with class

MIMIC-IV contributes discharge summaries; MIMIC-III contributes Psychiatry,
Social Work, Physician and Nursing notes. The two sources supply the classes in
different proportions, and `compute_section_quality()` then multiplies weights
by a per-source factor from 0.85 to 1.3. A model can partly separate the classes
by recognising note *format*.

## I2 — The blinded/unblinded comparison was not paired

`tc-wpn-complete-kaggle-training-notebook-blinded.ipynb` reports validation
prevalence 13.3 %; `training_notebook_final.ipynb` reports 33.3 %. The two arms
ran over different record sets, so the AUROC difference between them (best val
0.8301 blinded vs 0.9547 unblinded) mixes a shortcut effect with a
prevalence-and-sample effect. The direction is almost certainly right; the
magnitude is not attributable.

## I3 — "Confidence" is a misnomer

`ConfidenceWeightingModule` computes `exp(beta * cos(z_i, prototype))`. This is
similarity to the class centroid — typicality, not confidence. See the renaming
note in `README_CLEAN_BENCHMARK.md`.

---

## The number that motivates the whole rebuild

From `notebooks/tc-wpn-baseline-comparison-notebook.ipynb`:

| Model | Test HC AUROC | Test RW AUROC |
|---|---:|---:|
| TF-IDF + logistic regression | 0.9126 | 0.9634 |
| Standard ProtoNet | — | 0.9689 |
| TC-WPN | — | 0.9794 |

A bag-of-words model with no clinical knowledge, no temporal reasoning and no
pretrained encoder reaches 0.9634 on the real-world test split. The gap from
TF-IDF to TC-WPN is ~0.016 AUROC. On a benchmark where positives were selected
for containing anxiety vocabulary (V1) that is the expected result, and it means
the reported 0.97–0.98 figures measure the benchmark, not the method.

---

## How this becomes a contribution rather than a retraction

The rebuilt benchmark answers a question the archived one could not:

> How much of reported few-shot clinical anxiety-detection performance survives
> when the label cannot be read off the page?

Reporting the archived numbers alongside the clean numbers, with V1–V5 named,
is a stronger and more publishable paper than another 0.98. Shortcut learning in
clinical NLP is an active area with a receptive audience, and a paper that
diagnoses its own benchmark and rebuilds it has a defensible novelty claim even
if the clean AUROC lands near 0.70.

What must not happen is reporting the clean numbers as though the archived
pipeline never existed. The audit is the contribution.

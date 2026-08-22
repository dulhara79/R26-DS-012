# TC-WPN — publication-clean benchmark

Branch: `develop/publication-clean-benchmark`
Component: R26-DS-012 / TC-WPN
Author: Dulhara Kaushalya (IT22130648)

This tree replaces the archived pipeline. The archived code is not deleted; it
stays on `main` and is cited in `docs/AUDIT_OF_ARCHIVED_PIPELINE.md` as the
"before" condition of the benchmark-contamination analysis.

---

## The one-line statement of what changed

In the archived benchmark, a note's **label was partly a function of its
text**, and a patient could appear in the **support and query set of the same
episode**. Both are removed here, and both removals are enforced by tests that
run in CI rather than by a convention.

---

## Run order

**Correction (2026-08-05):** the command lines previously in this section did
not match the scripts' actual `argparse` flags — they used `--task/--out-dir/
--stem` where the scripts take `--arm/--out`, and invoked `evaluate.py` and
`compare_models.py` with arguments those scripts do not accept. Anyone who
pasted them got an immediate argparse error. The block below is generated
against the real CLIs and is the authoritative version.

On Kaggle, use `notebooks/kaggle_stage_a_data_prep.ipynb` (CPU) and
`notebooks/kaggle_stage_b_phase_a_run.ipynb` (GPU) instead of running these by
hand; they wrap exactly these commands.

Every stage writes a JSON report next to its output. Do not proceed past a stage
whose report shows a non-zero violation count.

### 0. Environment
```bash
git checkout main
pip install -r requirements.txt
export PYTHONPATH=src
export MIMIC_IV_DATASET_PATH=/path/to/mimic-iv
export MIMIC_IV_NOTE_DATASET_PATH=/path/to/mimic-iv-note
```

### 1. Contract tests, before touching data
```bash
pytest tests/ -q
```

### 2. Build the cohort
```bash
python -m scripts.build_clean_cohort \
    --out data/clean --source mimic4 --arm psych \
    --age-min 18 --age-max 50 \
    --max-notes-per-patient 8 --train-control-ratio 3
```
`--arm psych` is the primary experiment (anxiety vs other psychiatric illness);
`--arm clean` produces the easier secondary benchmark.
Writes `cohort_psych_mimic4.csv`, `patients_psych_mimic4.csv`, `audit_psych_mimic4.json`.

### 3. Index-time protocol
```bash
python -m scripts.apply_index_time \
    --cohort   data/clean/cohort_psych_mimic4.csv \
    --patients data/clean/patients_psych_mimic4.csv \
    --policy at_or_before --source mimic4 \
    --out data/clean/cohort_psych_mimic4idx.csv
```
Fixes one index time per patient from structured tables and drops notes outside
the admitted window. `at_or_before` = concurrent detection; `strictly_before` =
prospective. Read `differential_patient_retention_pp` in the report before
trusting anything downstream.

### 4. Audit
```bash
python -m scripts.audit_cohort --cohort data/clean/cohort_psych_mimic4idx.csv
```

### 5. Tokenise, unblinded and blinded, from the same file
```bash
python -m scripts.tokenize_cohort --cohort data/clean/cohort_psych_mimic4idx.csv \
    --out data/clean/pkl --blind none
python -m scripts.tokenize_cohort --cohort data/clean/cohort_psych_mimic4idx.csv \
    --out data/clean/pkl --blind dx_meds
```
Use `tokenize_cohort --blind`, not `blind_cohort.py`, for the robustness arms:
the plan fingerprint requires both pkls to come from identical cohort rows in
identical order. `blind_cohort.py` remains useful only for inspecting how many
terms each class contained.

The stem is derived from the filename, so `cohort_psych_mimic4idx.csv` gives
stem `psych_mimic4idx`.

### 6. Freeze episode plans and the leakage certificate
```bash
python -m scripts.make_episode_plans \
    --pkl-dir data/clean/pkl --stem psych_mimic4idx \
    --out data/clean/plans --k 1 3 5 10 --q-query 5 \
    --train-episodes 3000 --eval-repeats 3 \
    --leakage-episodes 5000 --seed 42
```

### 7. Shallow baselines — and stop to read them
```bash
python -m scripts.run_shallow_baselines --stem psych_mimic4idx --k 5 \
    --baseline tfidf_lr   --split test --pkl-dir data/clean/pkl --plan-dir data/clean/plans
python -m scripts.run_shallow_baselines --stem psych_mimic4idx --k 5 \
    --baseline bert_probe --split test --pkl-dir data/clean/pkl --plan-dir data/clean/plans
```
If TF-IDF matches the neural models, that is the headline and the paper changes
shape. Do not run the grid before looking.

### 8. Train — one K, one seed first
```bash
python -m scripts.train --config configs/protonet.yaml   --k 5 --seed 42 --stem psych_mimic4idx
python -m scripts.train --config configs/tcwpn_full.yaml --k 5 --seed 42 --stem psych_mimic4idx
```
Runs land in `results/psych_mimic4idx/<config-name>_k<K>_seed<seed>/`.

### 9. Evaluate
```bash
python -m scripts.evaluate --run results/psych_mimic4idx/tcwpn_full_k5_seed42 \
    --split test --bootstrap 2000
python -m scripts.evaluate --run results/psych_mimic4idx/tcwpn_full_k5_seed42 \
    --split test --blind dx_meds --bootstrap 2000
```

### 10. Compare
```bash
python -m scripts.compare_models table --results results/psych_mimic4idx --split test
python -m scripts.compare_models pair \
    --a results/psych_mimic4idx/protonet_k5_seed42/predictions_test.csv \
    --b results/psych_mimic4idx/tcwpn_full_k5_seed42/predictions_test.csv
```

### 11. Only then: ablations and extra seeds
```bash
for CFG in protonet protonet_temp temporal_only pcw_only temporal_pcw tcwpn_full; do
  python -m scripts.train --config configs/$CFG.yaml --k 5 --seed 42 --stem psych_mimic4idx
done
```

---

## Two task definitions, named separately

The index policy decides which task the paper is reporting. Use the right words:

| policy | task | what the abstract may claim |
|---|---|---|
| `at_or_before` | concurrent detection | identifies anxiety presentations from the record up to and including the index admission |
| `strictly_before` | prospective detection | anticipates an anxiety diagnosis from the prior record |
| `none` | retrospective association | nothing predictive; kept only to compare against the archived pipeline |

The temporal weight uses `days_before_index` under any policy. `apply_index_time.py`
writes that value into the `days_before_patient_last_note` column because that is
the key `tokenize_cohort.py` reads; the substitution is recorded in the stage's
JSON report. To make it explicit in code instead, add a `--temporal-field`
argument to `tokenize_cohort.py` and point it at `days_before_index`.

## Naming change you must carry into the paper

`confidence_weight` is renamed `prototype_consistency_weight` throughout.

The quantity is `w_i^C = exp(beta * cos(z_i, p~_c))` — the cosine similarity of
a support note to the preliminary prototype of its own class. That measures how
typical a note is of its class centroid. It is not a confidence in the label, it
is not calibrated, and it carries no uncertainty semantics. Calling it
"confidence" invites exactly one reviewer question, and there is no good answer
to it.

The architecture keeps the name TC-WPN for continuity with the registered
project. Define the C as *consistency* on first use and never use the word
"confidence" for this term again.

---

## What is deliberately absent

| Archived construct | Status |
|---|---|
| `has_psychiatric_content()` keyword gate | removed |
| `assign_anxiety_confidence()` | removed |
| `penalize_control_noise()` | removed |
| `label_confidence`, `training_weight`, `section_quality` | removed from every record |
| `curriculum_filter()` purity phases | removed |
| `*_high_conf` splits | removed |
| prescription- and OMR-derived labels | removed |
| MIMIC-III mixed into training | removed (MIMIC-III reserved for cross-dataset transfer) |

`tests/test_no_text_derived_filtering.py` fails the build if any of these
reappears in executable code.

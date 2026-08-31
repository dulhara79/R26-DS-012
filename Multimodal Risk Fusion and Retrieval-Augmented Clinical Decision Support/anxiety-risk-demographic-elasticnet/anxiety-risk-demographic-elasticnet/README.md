# Demographic/Lifestyle Anxiety Vulnerability Model

This is the revised final component that **removes ISI/sleep** and uses only demographic/lifestyle predictors.

## Final architecture

**Inputs**

- age
- gender
- education
- smoking
- drinking

**Model**

- Elastic-Net Logistic Regression
- selected by 5-fold development cross-validation
- sigmoid/Platt probability calibration

**Training target**

- `1` if observed GAD-7 score is `>= 10`
- `0` otherwise

**Outputs**

- `demographic_probability`: calibrated probability of `GAD-7 >= 10`
- `demographic_fusion_score`: relative vulnerability score for the fusion model

The fusion score is not an anxiety probability.

## Current model results

Clean modeling records: **24,268**

Development set: **19,414**, including **434 positives**

Held-out test set: **4,854**, including **108 positives**

Development prevalence: **2.2355%**

Selected hyperparameters:

- `C = 1.0`
- `l1_ratio = 0.75`
- `class_weight = None`

Held-out performance:

- ROC-AUC: **0.614**
- PR-AUC: **0.043**
- Brier score: **0.0217**
- Log loss: **0.1046**

This is deliberately a **baseline vulnerability component**. It is not strong enough to be presented as a standalone anxiety diagnosis model.

## Real held-out examples

Low predicted vulnerability:

- age 18
- male
- master's degree
- never smoker
- never drinker
- probability about **0.68%**
- fusion score about **0.23**

High predicted vulnerability within the common age range:

- age 22
- female
- bachelor's degree
- current smoker
- current regular drinker
- probability about **19.66%**
- fusion score about **0.915**
- this held-out participant actually had `GAD-7 >= 10`

See `reports/results/real_case_examples.csv` for exact values.

## Fusion-score conversion

The model's development prevalence is the reference point.

```text
probability == development prevalence  -> fusion score 0.50
probability below baseline             -> fusion score <0.50
probability above baseline             -> fusion score >0.50
```

Formula:

```text
evidence = logit(probability) - logit(development_prevalence)
fusion_score = sigmoid(evidence)
```

See `docs/FUSION_INTEGRATION.md` for integration details.

## Run everything

Create/activate a Python environment, install requirements, then:

```bash
pip install -r requirements.txt
./scripts/run_all.sh
```

This performs final calibrated training, held-out evaluation, coefficient/odds-ratio export, fusion-score export, plots, and automated tests using the saved tuned parameters.

To re-run hyperparameter tuning first:

```bash
PYTHONPATH=src python scripts/tune_model.py
./scripts/run_all.sh
```

## Run a prediction

```bash
PYTHONPATH=src python scripts/predict_example.py
```

Or in Python:

```python
from anxiety_risk.predict import load_bundle, predict_patient

bundle = load_bundle("models/demographic_elasticnet_calibrated.joblib")

result = predict_patient(
    bundle,
    age=22,
    gender="female",
    education="bachelor's degree",
    smoking="current smoker (cumulative smoking >10 packs)",
    drinking="current regular drinker (more than once a week)",
)

print(result)
```

## Directory structure

```text
data/raw/                   demographic.csv + gad7.csv used by model
data/reference_unused/      isi.csv retained but not used
src/anxiety_risk/            reusable data/model/fusion/prediction code
scripts/                     tuning, training, example prediction, run-all
models/                      trained calibrated model bundle
reports/results/             metrics, predictions, coefficients, fusion mapping
reports/figures/             ROC, PR, calibration, fusion-score plots
tests/                       automated tests
docs/                        model card + fusion integration guide
```

# Model Card — Demographic/Lifestyle Anxiety Vulnerability Model

## Model

Calibrated Elastic-Net Logistic Regression.

Best development hyperparameters:

- `C = 1.0`
- `l1_ratio = 0.75`
- `class_weight = None`
- sigmoid/Platt probability calibration

## Inputs

- age
- gender
- education
- smoking habit
- drinking habit

ISI/sleep is intentionally excluded from the final component.

## Outcome

Binary screening outcome derived from observed GAD-7:

- positive: `GAD-7 >= 10`
- negative: `GAD-7 < 10`

The model estimates a screening probability. It does not diagnose an anxiety disorder.

## Data

Original demographic and GAD-7 files each contain 24,292 participants. GAD-7 totals were independently reconstructed from the seven item scores and checked against the provided `score` field during the data pipeline.

For this model, age values outside 15-65 were removed as data-quality outliers, leaving 24,268 records. The 1st-99th percentile development age range is 18-27, so predictions for much older or younger participants have weaker empirical support.

Positive prevalence in development data: 2.2355%.

## Held-out test results

- ROC-AUC: 0.614
- PR-AUC: 0.043
- Brier score: 0.0217
- Log loss: 0.1046
- test participants: 4,854
- test positives: 108

The model has modest discrimination and should be treated as a baseline vulnerability signal for multimodal fusion, not as a standalone clinical screening or diagnosis tool.

## Outputs

- calibrated probability of `GAD-7 >= 10`
- baseline-adjusted demographic fusion score between 0 and 1

Fusion score 0.50 corresponds to the development population baseline. The fusion score is not a probability.

## Limitations

- Student population; generalization to psychiatric patients is not established here.
- Severe class imbalance.
- Demographic/lifestyle features alone provide weak standalone discrimination.
- Associations must not be interpreted as causal effects.
- Age representation outside the dense training range is limited.
- External clinical validation remains necessary.

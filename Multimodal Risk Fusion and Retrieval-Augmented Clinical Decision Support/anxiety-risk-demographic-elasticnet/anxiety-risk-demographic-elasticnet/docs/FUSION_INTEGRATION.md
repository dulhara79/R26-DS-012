# Fusion Integration

## Component purpose

This model contributes a **baseline demographic/lifestyle vulnerability signal**. It does not use ISI, physiology, phone behavior, clinical notes, GAD-7 items, or GAD-7 response times as predictors.

Inputs:

- `age`
- `gender`
- `education`
- `smoking`
- `drinking`

Target used during training:

- `GAD-7 >= 10`

## Two outputs

The predictor returns two different numeric outputs:

1. `demographic_probability`: calibrated probability of `GAD-7 >= 10`.
2. `demographic_fusion_score`: a baseline-adjusted 0-1 evidence index for the multimodal fusion layer.

The fusion score is **not another probability**.

## Conversion

Let `p` be the calibrated demographic probability and `pi` the positive prevalence in the model's development data.

`evidence = logit(p) - logit(pi)`

`fusion_score = sigmoid(evidence)`

For this trained model:

- development prevalence `pi = 0.022355` (2.2355%)
- therefore a probability equal to 2.2355% maps to fusion score `0.50`
- probabilities below baseline map below `0.50`
- probabilities above baseline map above `0.50`

Examples from the mapping table:

| Calibrated probability | Fusion score meaning |
|---:|---:|
| 1.0% | ~0.306 |
| 2.2355% | 0.500 |
| 5.0% | ~0.697 |
| 10.0% | ~0.829 |
| 15.0% | ~0.885 |

Use `reports/results/fusion_score_reference.csv` for exact generated values.

## Recommended fusion payload

```json
{
  "component": "demographic_lifestyle",
  "model": "calibrated_elastic_net_logistic_regression",
  "demographic_probability": 0.1966,
  "demographic_fusion_score": 0.9145,
  "available": true
}
```

The multimodal fusion model should consume `demographic_fusion_score`. The UI/research report should retain `demographic_probability` separately.

## Important rule

Do not interpret `demographic_fusion_score = 0.91` as "91% probability of anxiety." It means the demographic evidence is strongly above the development-population baseline.

The final fusion component weights should be learned from patient-level multimodal ground truth. Do not assign fixed manual weights solely from the individual models' raw score ranges.

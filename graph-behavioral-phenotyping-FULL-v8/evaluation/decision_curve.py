"""Decision-curve utilities."""
import numpy as np


def net_benefit(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    predicted_positive = probabilities >= threshold
    tp = np.sum(predicted_positive & (y_true == 1))
    fp = np.sum(predicted_positive & (y_true == 0))
    n = len(y_true)

    odds = threshold / (1 - threshold)
    return float(tp / n - fp / n * odds)


def recommend_fusion_weight(brier_skill, n_thresholds_better_than_trivial):
    if brier_skill < 0 or n_thresholds_better_than_trivial <= 3:
        return 0.0
    return None

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold_for_sensitivity(y_true, probabilities, target_sensitivity: float = 0.80) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity must be in (0, 1]")
    thresholds = np.unique(p)
    best = None
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        if sensitivity + 1e-12 >= target_sensitivity:
            candidate = {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
            }
            if best is None or (candidate["specificity"], candidate["threshold"]) > (best["specificity"], best["threshold"]):
                best = candidate
    if best is None:
        raise ValueError("No threshold satisfies target sensitivity")
    return best


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": float(threshold),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

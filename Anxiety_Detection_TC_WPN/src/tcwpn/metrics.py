"""
metrics.py — Patient-level metrics, calibration, and inference.

Everything here operates on two aligned 1-D arrays:
    y  : patient-level binary labels
    p  : patient-level predicted P(anxiety)

The unit of analysis is the PATIENT, never the episode and never the note.
Bootstrapping resamples patients because patients are the independent units;
resampling episodes (as the archived pipeline effectively did) produces
intervals that are too narrow because the same patient recurs across episodes.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from .delong import delong_roc_test  # reused unchanged from fixes/delong.py


# =============================================================================
# CALIBRATION
# =============================================================================
def expected_calibration_error(y, p, n_bins=10):
    """
    Standard equal-width ECE. Returns (ece, per-bin table).
    Bins with no samples are skipped rather than counted as zero error.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, table = 0.0, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += (n / len(p)) * abs(acc - conf)
        table.append({"bin_lo": lo, "bin_hi": hi, "n": n,
                      "mean_pred": conf, "observed_rate": acc})
    return float(ece), table


# =============================================================================
# THRESHOLD SELECTION (validation only)
# =============================================================================
def select_threshold(y_val, p_val, objective="f1"):
    """
    Choose an operating point on the VALIDATION patients. The returned value is
    then frozen and passed to every test evaluation. Never call this on test.
    """
    y_val = np.asarray(y_val)
    p_val = np.asarray(p_val, dtype=float)
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if len(thr) == 0:
        return 0.5
    if objective == "f1":
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        return float(thr[int(np.argmax(f1[:-1]))])
    if objective == "youden":
        from sklearn.metrics import roc_curve

        fpr, tpr, roc_thr = roc_curve(y_val, p_val)
        return float(roc_thr[int(np.argmax(tpr - fpr))])
    raise ValueError(f"unknown objective {objective!r}")


# =============================================================================
# FULL METRIC BUNDLE
# =============================================================================
def compute_metrics(y, p, threshold, n_bootstrap=2000, seed=42, n_bins=10):
    """
    Returns a flat dict with point estimates and patient-bootstrap 95% CIs for
    AUROC and PR-AUC. Threshold-dependent metrics use the supplied (locked)
    threshold.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0 or y.sum() == 0 or (y == 0).sum() == 0:
        raise ValueError("need at least one patient of each class")

    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()

    res = {
        "n_patients": int(n),
        "n_case": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1_positive": float(f1_score(y, yhat, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "brier": float(brier_score_loss(y, p)),
    }
    res["ece"], res["_ece_table"] = expected_calibration_error(y, p, n_bins=n_bins)

    if n_bootstrap and n >= 20:
        rng = np.random.default_rng(seed)
        au, pr = [], []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            yb, pb = y[idx], p[idx]
            if yb.sum() == 0 or (yb == 0).sum() == 0:
                continue
            au.append(roc_auc_score(yb, pb))
            pr.append(average_precision_score(yb, pb))
        if au:
            res["auroc_ci_lower"] = float(np.percentile(au, 2.5))
            res["auroc_ci_upper"] = float(np.percentile(au, 97.5))
            res["pr_auc_ci_lower"] = float(np.percentile(pr, 2.5))
            res["pr_auc_ci_upper"] = float(np.percentile(pr, 97.5))
            res["n_bootstrap_used"] = len(au)
    return res


# =============================================================================
# PAIRED COMPARISON
# =============================================================================
def compare_models(y, p_a, p_b, name_a="model_a", name_b="model_b"):
    """
    Paired DeLong test on the SAME patients. Only valid when both models were
    evaluated on the same episode plan, which is why plans are serialised.
    """
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if not (len(y) == len(p_a) == len(p_b)):
        raise ValueError("paired test requires aligned patient vectors")
    auc_a, auc_b, z, pval = delong_roc_test(y, p_a, p_b)
    return {
        "model_a": name_a, "model_b": name_b,
        "auroc_a": float(auc_a), "auroc_b": float(auc_b),
        "delta_auroc": float(auc_a - auc_b),
        "z": float(z), "p_value": float(pval),
        "significant_at_0.05": bool(pval < 0.05),
        "n_patients": int(len(y)),
    }


def align_patient_vectors(res_a: dict, res_b: dict):
    """
    Given two {patient_id: prob} dicts plus {patient_id: label}, return the
    intersection in a stable order so DeLong gets genuinely paired inputs.

    res_* must have keys "probs" (dict) and "labels" (dict).
    """
    shared = sorted(set(res_a["probs"]) & set(res_b["probs"]))
    if not shared:
        raise ValueError("no shared patients between the two result sets")
    y = np.array([res_a["labels"][k] for k in shared], dtype=int)
    y_b = np.array([res_b["labels"][k] for k in shared], dtype=int)
    if not np.array_equal(y, y_b):
        raise ValueError("label mismatch between result sets for shared patients")
    return (
        y,
        np.array([res_a["probs"][k] for k in shared], dtype=float),
        np.array([res_b["probs"][k] for k in shared], dtype=float),
        shared,
    )


def summarise_seeds(per_seed_metrics, keys=("auroc", "pr_auc", "f1_positive")):
    """mean +/- SD across seeds, for the main results table."""
    out = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics if k in m]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
                "values": [float(v) for v in vals],
            }
    return out

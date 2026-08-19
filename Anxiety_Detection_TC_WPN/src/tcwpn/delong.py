"""
delong.py
Paired DeLong test for two correlated ROC AUCs (same patients, two models).
Author: (for Dulhara Kaushalya's TC-WPN pipeline)

Implements the fast DeLong method (Sun & Xu, 2014, IEEE SPL) for the variance
of an AUC and the covariance between two AUCs computed on the SAME samples.

USE THIS to test, honestly, whether TC-WPN's AUROC is significantly higher
than Standard ProtoNet's. With your reported gap (~0.004) and overlapping CIs,
expect p to be LARGE (not significant). Report whatever it shows.

INPUT: patient-level vectors (one score per patient), NOT episodic predictions.
You already produce these in evaluate_patient_level(); save the per-patient
(y_true, prob) arrays for each model and feed them here.

    auc1, auc2, z, p = delong_roc_test(y_true, prob_tcwpn, prob_protonet)

No scipy dependency (normal tail via math.erfc).
"""

import math
import numpy as np


def _compute_midrank(x):
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0  # 1-based midrank, averaged over ties
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: [k, n] with the m positive samples in the first m columns.
       Returns (aucs[k], delong_cov[k,k])."""
    k, n_total = preds_sorted.shape
    n = n_total - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, n_total])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)                      # [k,k]
    sy = np.cov(v10)
    if k == 1:                            # np.cov returns scalar for 1 row
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_roc_test(y_true, p1, p2):
    """Two-sided paired DeLong test: H0 = AUC(p1) == AUC(p2).
    Returns (auc1, auc2, z, p_value)."""
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    assert y_true.shape == p1.shape == p2.shape, "shape mismatch"
    assert set(np.unique(y_true)).issubset({0, 1}), "y_true must be binary"
    order = np.argsort(-y_true, kind="mergesort")  # positives (label 1) first
    m = int(y_true.sum())
    preds = np.vstack((p1, p2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var <= 0:
        z = 0.0 if abs(aucs[0] - aucs[1]) < 1e-12 else float("inf")
    else:
        z = (aucs[0] - aucs[1]) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))  # two-sided normal tail
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def delong_auc_ci(y_true, p, alpha=0.05):
    """Single-AUC DeLong 95% CI (analytic, not bootstrap)."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-y_true, kind="mergesort")
    m = int(y_true.sum())
    preds = p[order][None, :]
    aucs, cov = _fast_delong(preds, m)
    se = math.sqrt(max(cov[0, 0], 0.0))
    zc = 1.959963984540054  # ~ qnorm(0.975)
    lo = max(0.0, aucs[0] - zc * se)
    hi = min(1.0, aucs[0] + zc * se)
    return float(aucs[0]), float(lo), float(hi)


if __name__ == "__main__":
    # ---- Self-test 1: DeLong AUC must equal sklearn roc_auc_score ----
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    s1 = y * 0.6 + rng.normal(0, 1, n)          # informative
    s2 = y * 0.58 + rng.normal(0, 1, n)         # slightly worse, correlated
    a_dl, lo, hi = delong_auc_ci(y, s1)
    a_sk = roc_auc_score(y, s1)
    print(f"AUC check  DeLong={a_dl:.6f}  sklearn={a_sk:.6f}  "
          f"match={abs(a_dl - a_sk) < 1e-9}   95%CI=[{lo:.4f},{hi:.4f}]")

    # ---- Self-test 2: identical models -> z=0, p=1 ----
    a1, a2, z, p = delong_roc_test(y, s1, s1)
    print(f"Identical models: auc1={a1:.4f} auc2={a2:.4f} z={z:.3f} p={p:.3f} "
          f"(expect z=0, p=1)")

    # ---- Self-test 3: tiny realistic gap (mimics TC-WPN vs ProtoNet) ----
    a1, a2, z, p = delong_roc_test(y, s1, s2)
    print(f"Tiny gap:         auc1={a1:.4f} auc2={a2:.4f} dz={a1-a2:+.4f} "
          f"z={z:.3f} p={p:.4f}")

    # ---- Self-test 4: large gap -> significant ----
    s_bad = rng.normal(0, 1, n)  # pure noise
    a1, a2, z, p = delong_roc_test(y, s1, s_bad)
    print(f"Large gap:        auc1={a1:.4f} auc2={a2:.4f} z={z:.3f} p={p:.2e} "
          f"(expect p<0.001)")

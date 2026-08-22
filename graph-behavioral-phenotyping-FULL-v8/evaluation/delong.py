"""Paired DeLong test for two correlated ROC curves."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    rank = np.zeros(n)

    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        rank[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    out = np.empty(n)
    out[order] = rank
    return out


def delong_test(pred1, pred2, y):
    pred = np.vstack([np.asarray(pred1, float), np.asarray(pred2, float)])
    y = np.asarray(y, int)

    order = np.argsort(-y)
    y = y[order]
    pred = pred[:, order]

    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        raise ValueError("Both classes are required for DeLong testing.")

    tx = np.array([_midrank(pred[r, :m]) for r in range(2)])
    ty = np.array([_midrank(pred[r, m:]) for r in range(2)])
    tz = np.array([_midrank(pred[r]) for r in range(2)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    cov = (
        np.cov(v01, ddof=1).reshape(2, 2) / m
        + np.cov(v10, ddof=1).reshape(2, 2) / n
    )

    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ cov @ contrast)
    diff = float(aucs[0] - aucs[1])

    if variance <= 0:
        return float(aucs[0]), float(aucs[1]), diff, 1.0

    z = abs(diff / np.sqrt(variance))
    p = float(2 * (1 - norm.cdf(z)))
    return float(aucs[0]), float(aucs[1]), diff, p

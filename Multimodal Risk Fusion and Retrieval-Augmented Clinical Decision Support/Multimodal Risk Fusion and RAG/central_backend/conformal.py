"""
Conformal vulnerability banding — a tier SET with a coverage guarantee, instead
of a bare point tier.

WHY: with the tiny labelled samples this project will have at NHSL, a point tier
("High") communicates far more certainty than the model possesses. A conformal
set makes the uncertainty explicit and honest:

    {"High"}                  the model commits
    {"Medium", "High"}        genuine ambiguity between adjacent tiers
    {"Low", "Medium", "High"} the model knows nothing useful right now

CONSTRUCTION — split conformal on the scalar composite, interval nonconformity:

    nonconformity(s, k) = distance from composite s to tier k's band interval
                          (0 if s lies inside the interval)

    q = the ceil((n+1)(1-alpha))-th smallest calibration nonconformity
    prediction set = every tier whose interval lies within q of s

Coverage guarantee (P(true tier in set) >= 1-alpha on exchangeable data) follows
from the standard split-conformal argument; no distributional assumption.

SMALL-SAMPLE HONESTY — this is the part that matters most here. The finite-
sample quantile index ceil((n+1)(1-alpha)) exceeds n whenever n < (1-alpha)/alpha
(n < 9 at alpha=0.10), which forces q = infinity and the set becomes ALL tiers.
That is not a bug — it is split conformal telling you it cannot certify anything
yet. We surface it explicitly (`calibrated: false` + a reason) rather than hide
it, and additionally refuse to claim calibration below MIN_CALIBRATION_N even
when the arithmetic technically produces a finite q, because a quantile from a
dozen points is noise wearing a formula.

Labels come from the `verdicts` table — the clinician's HITL tier judgements —
which is also this project's label-collection mechanism for the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Band intervals — MUST stay in sync with fusion_service/fusion.py BANDS
# ([(0.33,"Low"), (0.66,"Medium"), (1.01,"High")]). test_backend.py asserts this.
TIER_INTERVALS: Dict[str, Tuple[float, float]] = {
    "Low": (0.0, 0.33),
    "Medium": (0.33, 0.66),
    "High": (0.66, 1.0),
}
TIERS: List[str] = ["Low", "Medium", "High"]

DEFAULT_ALPHA = 0.10
MIN_CALIBRATION_N = 20     # below this we refuse to claim calibration at all


def nonconformity(composite: float, tier: str) -> float:
    lo, hi = TIER_INTERVALS[tier]
    return max(lo - composite, composite - hi, 0.0)


@dataclass
class ConformalResult:
    prediction_set: List[str]
    alpha: float
    calibrated: bool
    n_calibration: int
    quantile: Optional[float]
    reason: Optional[str] = None

    def to_wire(self) -> dict:
        return {
            "conformal_set": self.prediction_set,
            "conformal_alpha": self.alpha,
            "conformal_calibrated": self.calibrated,
            "conformal_n": self.n_calibration,
            "conformal_quantile": (round(self.quantile, 4)
                                   if self.quantile is not None and math.isfinite(self.quantile)
                                   else None),
            "conformal_note": self.reason,
        }


def predict_set(composite: Optional[float],
                calibration_pairs: Sequence[Tuple[float, str]],
                alpha: float = DEFAULT_ALPHA) -> ConformalResult:
    """calibration_pairs: (composite_at_verdict_time, clinician_tier_label).

    Pairs with unknown tier labels or missing composites are dropped, not
    guessed. If the composite itself is None (gate blocked), there is nothing
    to band and the caller shouldn't be asking.
    """
    pairs = [(float(s), t) for s, t in calibration_pairs
             if s is not None and t in TIER_INTERVALS]
    n = len(pairs)

    if composite is None:
        return ConformalResult(prediction_set=[], alpha=alpha, calibrated=False,
                               n_calibration=n, quantile=None,
                               reason="no composite to band (gate blocked)")

    if n < MIN_CALIBRATION_N:
        return ConformalResult(
            prediction_set=list(TIERS), alpha=alpha, calibrated=False,
            n_calibration=n, quantile=None,
            reason=(f"only {n} clinician verdicts available; conformal calibration "
                    f"requires >= {MIN_CALIBRATION_N}. Returning the full tier set "
                    f"rather than an uncertifiable narrower one."))

    scores = sorted(nonconformity(s, t) for s, t in pairs)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        q: float = math.inf
        reason = (f"n={n} is too small for alpha={alpha:g} "
                  f"(needs quantile index {k} of {n}); set defaults to all tiers")
    else:
        q = scores[k - 1]
        reason = None

    pred = [t for t in TIERS if nonconformity(composite, t) <= q]
    if not pred:                                  # cannot happen with finite q >= 0,
        pred = list(TIERS)                        # but never return an empty set

    return ConformalResult(prediction_set=pred, alpha=alpha,
                           calibrated=math.isfinite(q), n_calibration=n,
                           quantile=q if math.isfinite(q) else None, reason=reason)


def coverage_report(calibration_pairs: Sequence[Tuple[float, str]],
                    test_pairs: Sequence[Tuple[float, str]],
                    alpha: float = DEFAULT_ALPHA) -> dict:
    """The two numbers the paper needs: empirical coverage vs nominal, and the
    average set size. Perfect coverage with constant size-3 sets is useless —
    report both, always together."""
    covered, sizes = 0, []
    for s, t in test_pairs:
        res = predict_set(s, calibration_pairs, alpha)
        sizes.append(len(res.prediction_set))
        covered += int(t in res.prediction_set)
    n = max(len(test_pairs), 1)
    return {
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": round(covered / n, 4),
        "mean_set_size": round(sum(sizes) / n, 3),
        "n_test": len(test_pairs),
        "n_calibration": len(calibration_pairs),
    }

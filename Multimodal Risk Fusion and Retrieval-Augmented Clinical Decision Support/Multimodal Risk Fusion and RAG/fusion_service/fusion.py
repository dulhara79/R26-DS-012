"""
Recency- and Reliability-Weighted Late Fusion
Component 4 · R26-DS-012 · https://github.com/dulhara79/R26-DS-012

Combines four independently trained modality risk scores into one calibrated
composite and a three-level vulnerability tier.

    w_m(t)  =  omega_m  ·  rho_m(dt)  ·  c_m
    alpha   =  w / sum(w)          over AVAILABLE modalities only
    S       =  sum_m alpha_m · p_m

    omega_m : base weight   — informativeness above chance (validation AUROC - 0.5)
    rho_m   : recency       — 2^(-dt / half_life_m), matched to update cadence
    c_m     : reliability   — 0.5 + 0.5 · confidence · coverage

Rationale for every constant is in FUSION_DESIGN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# Calibration constants — CHANGE THESE ONLY WITH A RECORDED REASON
# ─────────────────────────────────────────────────────────────────────────────

MODALITIES = ["c1_physiological", "c2_behavioral", "c3_clinical_nlp", "c4_demographic"]

LABELS = {
    "c1_physiological": "Physiological",
    "c2_behavioral": "Behavioural",
    "c3_clinical_nlp": "Clinical notes",
    "c4_demographic": "Demographic prior",
}

# Deployment-realistic validation AUROC for each component.
# NOT best-case laboratory figures — see FUSION_DESIGN.md §3.
VALIDATION_AUROC = {
    "c1_physiological": 0.6191,   # AffectiveROAD, recalibrated, real-world driving
    "c2_behavioral": 0.5205,      # GLOBEM held-out; did not clear its permutation null
    "c3_clinical_nlp": 0.7380,    # MIMIC-IV five-shot held-out cohort
    "c4_demographic": 0.6600,     # <-- REPLACE with the test AUROC from your notebook
}

# A component that does not exceed its own permutation null contributes nothing.
# This is a pre-registered exclusion rule, not a post-hoc judgement.
CLEARS_PERMUTATION_NULL = {
    "c1_physiological": True,
    "c2_behavioral": False,       # AUROC 0.5205 vs null 0.4991, p = 0.255
    "c3_clinical_nlp": True,
    "c4_demographic": True,       # set False if your notebook's permutation p >= .05
}

# Recency half-life in minutes, matched to each stream's update cadence.
# None = the score does not decay (a prior, not an observation).
HALF_LIFE_MIN = {
    "c1_physiological": 30,          # arrives every minute; 2h stale -> weight ~ 6%
    "c2_behavioral": 24 * 60,
    "c3_clinical_nlp": 30 * 24 * 60, # notes arrive daily to monthly
    "c4_demographic": None,          # scored once at enrolment; never decays
}

PRIOR_CAP = 0.35        # max share of the composite a non-changing stream may hold
BANDS = [(0.33, "Low"), (0.66, "Medium"), (1.01, "High")]

EWMA_ALPHA = 0.20       # physiological smoothing, ~15-minute effective window
HYSTERESIS_N = 3        # consecutive readings required before a tier change is emitted


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Reading:
    score: Optional[float] = None          # harmonised probability in [0, 1]
    available: bool = False
    confidence: float = 0.5
    coverage: float = 1.0
    captured_at: Optional[datetime] = None
    note: Optional[str] = None


@dataclass
class FusionOutput:
    composite: Optional[float]
    tier: Optional[str]
    weights: Dict[str, float]
    contributions: Dict[str, float]
    scores: Dict[str, Optional[float]]
    modalities_available: int
    renormalised: bool
    confidence: float
    reason: Optional[str] = None
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_wire(self) -> dict:
        """Shape consumed by FusionResult.fromJson in the clinician Flutter app."""
        band = {"Low": "GREEN", "Medium": "AMBER", "High": "RED", None: "GREY"}[self.tier]
        return {
            "composite_score": self.composite,
            "tier": self.tier,
            "alert_level": band,
            "weights": self.weights,
            "scores": self.scores,
            "contributions": self.contributions,
            "modalities_available": self.modalities_available,
            "renormalised": self.renormalised,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "computed_at": self.computed_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
def base_weights() -> Dict[str, float]:
    """omega_m proportional to discriminative power above chance, zeroed for any
    component that failed its permutation null. Normalised to sum to 1."""
    raw = {
        m: max(VALIDATION_AUROC[m] - 0.5, 0.0) if CLEARS_PERMUTATION_NULL[m] else 0.0
        for m in MODALITIES
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("no component clears its permutation null — nothing to fuse")
    return {m: v / total for m, v in raw.items()}


def recency(modality: str, captured_at: Optional[datetime], now: datetime) -> float:
    half = HALF_LIFE_MIN[modality]
    if half is None:
        return 1.0                       # a prior does not go stale
    if captured_at is None:
        return 0.0
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    dt_min = max((now - captured_at).total_seconds() / 60.0, 0.0)
    return 0.5 ** (dt_min / half)


def reliability(r: Reading) -> float:
    """Bounded in [0.5, 1.0]: low confidence halves a stream's weight but never
    silences it, because a confidently wrong zero is worse than a hedged input."""
    return 0.5 + 0.5 * min(max(r.confidence, 0.0), 1.0) * min(max(r.coverage, 0.0), 1.0)


def fuse(readings: Dict[str, Reading], now: Optional[datetime] = None) -> FusionOutput:
    now = now or datetime.now(timezone.utc)
    omega = base_weights()

    raw_w, present = {}, []
    for m in MODALITIES:
        r = readings.get(m, Reading())
        if not r.available or r.score is None or omega[m] == 0.0:
            raw_w[m] = 0.0
            continue
        raw_w[m] = omega[m] * recency(m, r.captured_at, now) * reliability(r)
        if raw_w[m] > 0:
            present.append(m)

    scores = {m: (readings.get(m, Reading()).score if m in present else None) for m in MODALITIES}

    # Guard: a demographic prior alone is not a vulnerability assessment.
    dynamic = [m for m in present if HALF_LIFE_MIN[m] is not None]
    if not dynamic:
        return FusionOutput(
            composite=None, tier=None, weights={m: 0.0 for m in MODALITIES},
            contributions={m: 0.0 for m in MODALITIES}, scores=scores,
            modalities_available=len(present), renormalised=True, confidence=0.0,
            reason="insufficient modalities: no time-varying evidence available; "
                   "the demographic prior alone cannot support a tier",
        )

    total = sum(raw_w.values())
    alpha = {m: (raw_w[m] / total if total > 0 else 0.0) for m in MODALITIES}

    # Cap the static prior's share and redistribute the excess across dynamic streams.
    if alpha["c4_demographic"] > PRIOR_CAP:
        excess = alpha["c4_demographic"] - PRIOR_CAP
        alpha["c4_demographic"] = PRIOR_CAP
        dyn_total = sum(alpha[m] for m in dynamic)
        for m in dynamic:
            alpha[m] += excess * (alpha[m] / dyn_total) if dyn_total > 0 else excess / len(dynamic)

    composite = sum(alpha[m] * (scores[m] or 0.0) for m in MODALITIES)
    contributions = {m: round(alpha[m] * (scores[m] or 0.0), 4) for m in MODALITIES}

    # Composite confidence: how much of the available base weight is actually
    # reporting, scaled by the mean reliability of the streams that are.
    covered = sum(omega[m] for m in present)
    mean_rel = sum(reliability(readings[m]) for m in present) / len(present)
    conf = covered * mean_rel

    tier = next(name for edge, name in BANDS if composite < edge)

    return FusionOutput(
        composite=round(composite, 4), tier=tier,
        weights={m: round(alpha[m], 4) for m in MODALITIES},
        contributions=contributions, scores=scores,
        modalities_available=len(present),
        renormalised=len(present) < sum(1 for m in MODALITIES if omega[m] > 0),
        confidence=conf,
    )


# ─────────────────────────────────────────────────────────────────────────────
class LiveFusion:
    """Stateful wrapper for the 1-per-minute physiological stream.

    Two problems solved here, both of which make the difference between a
    usable ward tool and one the nurses mute on day two:

    1. EWMA smoothing — a single anxious minute is not an escalation. Raw
       minute-level scores are noisy; the composite should follow the trend.
    2. Tier hysteresis — a tier change is only emitted after HYSTERESIS_N
       consecutive readings agree, so the badge does not oscillate.
    """

    def __init__(self):
        self._ewma: Optional[float] = None
        self._pending: Optional[str] = None
        self._pending_n: int = 0
        self.tier: Optional[str] = None

    def update(self, readings: Dict[str, Reading], now: Optional[datetime] = None) -> FusionOutput:
        r = readings.get("c1_physiological")
        if r and r.available and r.score is not None:
            self._ewma = r.score if self._ewma is None else \
                EWMA_ALPHA * r.score + (1 - EWMA_ALPHA) * self._ewma
            readings = dict(readings)
            readings["c1_physiological"] = Reading(
                score=self._ewma, available=True, confidence=r.confidence,
                coverage=r.coverage, captured_at=r.captured_at, note="EWMA-smoothed",
            )

        out = fuse(readings, now)

        if out.tier != self.tier:
            if out.tier == self._pending:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = out.tier, 1
            if self._pending_n >= HYSTERESIS_N:
                self.tier = out.tier
                self._pending, self._pending_n = None, 0
        else:
            self._pending, self._pending_n = None, 0

        out.tier = self.tier if self.tier is not None else out.tier
        return out

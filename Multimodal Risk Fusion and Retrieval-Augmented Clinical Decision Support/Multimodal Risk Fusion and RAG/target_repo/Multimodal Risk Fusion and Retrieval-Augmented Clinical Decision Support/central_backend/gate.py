"""
The fusion gate — step 28 of the sequence diagram.

    "gate: >=2 fresh modalities? weights defined? semantics compatible?"

This runs BEFORE anything is fused. It is the difference between a composite
that means something and a number that merely exists.

Five conditions, each rejecting a specific failure:

  FRESHNESS       A reading older than its modality's useful life is not
                  evidence about now. A physiological score from this morning
                  says nothing about this afternoon. Windows below match the
                  frozen service contract (R26-DS-012_service_contracts.md §5):
                  C1 15 min (60s windows, 5-10 min forecast horizon), C2 7 days
                  (daily graph recompute over a 42-day span), C3 90 days (notes
                  are episodic; a 6-month-old note is not current risk).

  EFFECTIVE WEIGHT A reading can be younger than its MAX_AGE cutoff and still be
                  almost worthless once the fusion service's recency decay is
                  applied. A modality whose post-decay weight has fallen below
                  a floor is dropped even if it is technically "fresh" by the
                  cutoff above. This is a deliberate second check, independent
                  of MAX_AGE, so a future change to one cutoff cannot silently
                  let a near-zero-weight reading count as real evidence again —
                  which is exactly what happened when this gate's C1 cutoff was
                  6 hours against a 30-minute fusion half-life: a 3-hour-old
                  reading passed the age check while contributing ~1.6% of the
                  composite. Two "modalities" where one is a rounding error is
                  one modality wearing a disguise.

  ENOUGH STREAMS  At least 2 usable modalities. One stream is not multimodal
                  fusion, it is that stream with extra steps.

  DYNAMIC         At least one TIME-VARYING modality. The demographic prior is
                  constant per patient forever, so a "composite" built from it
                  alone would be a demographic stereotype wearing a clinical
                  label. This is the day-one guard.

  SEMANTICS       Status must be `ok` and the score finite. A component that
                  reported any other status — not_validated, warming_up,
                  insufficient_data, poor_signal, no_support_set, error — is
                  stored and visible in the clinician timeline but never enters
                  the composite. This gate does not enumerate the vocabulary;
                  anything other than exactly "ok" is rejected, so a new status
                  value a component starts sending needs no change here.

A note on timebases, which no gate condition can fix and belongs in the paper
instead: C1 forecasts 5-10 minutes ahead, C2 scores vulnerability over a 42-day
window, C3 scores whether one note describes anxiety, C4 is a static prior. A
weighted average of these is a modelling decision, not an arithmetic fact —
state it as an assumption in the methods section rather than let an examiner
find it first.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

# How long a reading from each modality remains usable as evidence AT ALL.
# Matches R26-DS-012_service_contracts.md §5. Deliberately independent of the
# fusion HALF-LIVES below: the half-life decays a reading's weight smoothly,
# this is the hard cut-off past which it is dropped regardless of decay.
MAX_AGE_MINUTES: Dict[str, Optional[int]] = {
    "c1_physiological": 15,            # 60s windows, 5-10 min forecast horizon
    "c2_behavioral": 7 * 24 * 60,      # daily graph recompute over a 42-day span
    "c3_clinical_nlp": 90 * 24 * 60,   # notes are episodic; not current risk past this
    "c4_demographic": None,            # never expires; it is a prior, not an observation
}

# MUST match fusion_service/fusion.py HALF_LIFE_MIN. Duplicated here rather than
# imported so the gate has zero runtime dependency on the fusion service package
# (it must be able to reject stale readings even if fusion.py is unreachable).
# test_backend.py asserts these two stay in sync.
HALF_LIFE_MINUTES: Dict[str, Optional[int]] = {
    "c1_physiological": 30,
    "c2_behavioral": 24 * 60,
    "c3_clinical_nlp": 30 * 24 * 60,
    "c4_demographic": None,
}

# Below this, a reading is treated as decayed-out even if still under MAX_AGE.
# 0.05 = roughly 4.3 half-lives; chosen so it never binds under the CURRENT
# freshness windows above (they were tightened specifically to make this a
# defensive backstop, not the primary control) but still catches a future
# loosened window before it reintroduces the 6h/30min bug this replaced.
EFFECTIVE_WEIGHT_FLOOR = 0.05

# Which modalities carry information about CHANGE.
DYNAMIC_MODALITIES = {"c1_physiological", "c2_behavioral", "c3_clinical_nlp"}

# Excluded by pre-registered rule, not by preference.
EXCLUDED_MODALITIES = {"c2_behavioral"}

MIN_USABLE_MODALITIES = 2


@dataclass
class GateDecision:
    passed: bool
    usable: Dict[str, dict] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    reason: Optional[str] = None

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "usable_modalities": sorted(self.usable),
            "rejected": self.rejected,
            "reason": self.reason,
        }


def _age_minutes(captured_at: dt.datetime, now: dt.datetime) -> float:
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=dt.timezone.utc)
    return max((now - captured_at).total_seconds() / 60.0, 0.0)


def effective_weight(modality: str, age_minutes: float) -> float:
    """Same decay curve fusion.py applies, evaluated here for the gate check.
    A prior (half-life None) never decays, so it always returns 1.0."""
    half = HALF_LIFE_MINUTES.get(modality)
    if half is None:
        return 1.0
    return 0.5 ** (age_minutes / half)


def evaluate(readings: Dict[str, dict], now: Optional[dt.datetime] = None) -> GateDecision:
    """readings: {modality: {raw_score, status, confidence, coverage, captured_at}}"""
    now = now or dt.datetime.now(dt.timezone.utc)
    usable, rejected = {}, {}

    for modality, r in readings.items():
        if modality in EXCLUDED_MODALITIES:
            rejected[modality] = "excluded by pre-registered rule (did not clear permutation null)"
            continue
        if r is None:
            rejected[modality] = "no reading"
            continue

        status = r.get("status", "ok")
        if status != "ok":
            rejected[modality] = f"status={status}"
            continue

        score = r.get("raw_score")
        if score is None or not isinstance(score, (int, float)) or not math.isfinite(score):
            rejected[modality] = "score missing or not finite"
            continue

        age = _age_minutes(r["captured_at"], now)

        max_age = MAX_AGE_MINUTES.get(modality)
        if max_age is not None and age > max_age:
            rejected[modality] = f"stale: {age/60:.1f}h old, limit {max_age/60:.1f}h"
            continue

        eff = effective_weight(modality, age)
        if eff < EFFECTIVE_WEIGHT_FLOOR:
            rejected[modality] = (f"decayed below usable floor: {age:.0f}min old carries "
                                  f"only {eff:.1%} of full weight (floor {EFFECTIVE_WEIGHT_FLOOR:.0%})")
            continue

        usable[modality] = r

    # condition: enough streams
    if len(usable) < MIN_USABLE_MODALITIES:
        return GateDecision(
            passed=False, usable=usable, rejected=rejected,
            reason=(f"insufficient evidence: {len(usable)} usable modality"
                    f"{'' if len(usable) == 1 else 'ies'}, need {MIN_USABLE_MODALITIES}"),
        )

    # condition: at least one time-varying stream
    if not (set(usable) & DYNAMIC_MODALITIES):
        return GateDecision(
            passed=False, usable=usable, rejected=rejected,
            reason=("insufficient evidence: no time-varying modality available; "
                    "the demographic prior alone cannot support a vulnerability tier"),
        )

    return GateDecision(passed=True, usable=usable, rejected=rejected)

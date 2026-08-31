"""Temporary, opt-in investor demo simulation helpers.

This module is intentionally isolated from the research fusion implementation.
Nothing here is used unless INVESTOR_DEMO_SIMULATION=1 in central_backend/main.py.
Remove this module after the investor demo path is no longer needed.
"""

from __future__ import annotations

from typing import Optional

import modality_clients as mc

LOW_SCORE = 0.2033
HIGH_SCORE = 0.8967
NEUTRAL_SCORE = 0.4967

_LOW_TERMS = (
    "low anxiety",
    "lower end",
    "mild",
    "stable",
    "calm",
    "minimal anxiety",
    "no anxiety",
    "well controlled",
)
_HIGH_TERMS = (
    "high anxiety",
    "severe",
    "escalating",
    "panic",
    "acute anxiety",
    "very anxious",
    "worsening anxiety",
)


def physiological_composite(raw_score: Optional[float]) -> Optional[float]:
    """Map C1's documented 0..100 current_risk_index to fusion's 0..1 scale."""
    if raw_score is None:
        return None
    return round(min(max(float(raw_score), 0.0), 100.0) / 100.0, 4)


def tier_and_band(score: Optional[float]) -> tuple[Optional[str], str]:
    if score is None:
        return None, "GREY"
    if score < 0.33:
        return "Low", "GREEN"
    if score < 0.66:
        return "Medium", "AMBER"
    return "High", "RED"


def clinical_score(note_text: str) -> float:
    """Deterministic keyword simulation for the broken external clinical path.

    High-risk wording wins when both high and low terms occur in the same note,
    because escalation/severity language should not be masked by a phrase such
    as "previously stable" during the demo.
    """
    text = " ".join((note_text or "").lower().split())
    if any(term in text for term in _HIGH_TERMS):
        return HIGH_SCORE
    if any(term in text for term in _LOW_TERMS):
        return LOW_SCORE
    return NEUTRAL_SCORE


def clinical_result(note_text: str) -> mc.ComponentResult:
    score = clinical_score(note_text)
    tier, _ = tier_and_band(score)
    prediction = "ANXIETY" if score >= 0.5 else "NO ANXIETY"
    detail = {
        "prediction": prediction,
        "risk_score": score,
        "calibrated_probability": score,
        "probability": score,
        "confidence": 1.0,
        "entropy": 0.0,
        "threshold": 0.5,
        "support_k": 0,
        "attention_spans": [],
        "support_contributions": [],
        "temporal_context": "Investor demo simulation",
        "model_version": "investor-demo-clinical-v1",
        "latency_ms": 0,
        "used_default_support_set": False,
        "simulation": True,
        "simulated_tier": tier,
    }
    return mc.ComponentResult(
        raw_score=score,
        status="ok",
        confidence=1.0,
        coverage=1.0,
        model_version="investor-demo-clinical-v1",
        detail=detail,
        note="investor demo simulation: deterministic note keyword score",
    )

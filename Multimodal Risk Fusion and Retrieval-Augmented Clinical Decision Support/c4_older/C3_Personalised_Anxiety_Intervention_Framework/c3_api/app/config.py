"""C3 API configuration — all constants centralised here.

This module contains NO logic — only constants.
Environment variables are read at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Security / JWT
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.getenv(
    "C3_SECRET_KEY",
    "CHANGE_ME_IN_RAILWAY_ENV_this_is_a_development_fallback_only_32_bytes",
)
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24h

# ---------------------------------------------------------------------------
# Firebase (Phase 7 — not used yet but wired)
# ---------------------------------------------------------------------------
FIREBASE_CREDENTIALS_JSON: str = os.getenv("FIREBASE_CREDENTIALS_JSON", "")

# ---------------------------------------------------------------------------
# Feature schema — order matters. Must match training order exactly.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "age_norm",
    "gender_enc",
    "marital_enc",
    "education_enc",
    "income_enc",
    "physiological_risk",
    "behavioral_risk",
    "textual_risk",
    "composite_risk",
    "risk_tier_enc",          # ALWAYS 0.0 — leakage fix
    "interaction_count_norm",
    "last_reward_norm",
    "escalation_count_norm",
]

# ---------------------------------------------------------------------------
# Risk tiers and interventions
# ---------------------------------------------------------------------------
TIER_LABELS: dict[int, str] = {0: "Low", 1: "Medium", 2: "High"}

TIER_INTERVENTIONS: dict[int, list[str]] = {
    0: ["routine_monitoring", "light_followup"],
    1: ["targeted_nudge", "priority_followup"],
    2: ["urgent_outreach"],
}

INTERVENTION_PRIORITY: dict[str, str] = {
    "routine_monitoring": "P5",
    "light_followup":     "P4",
    "targeted_nudge":     "P3",
    "priority_followup":  "P2",
    "urgent_outreach":    "P1",
    "manual_review":      "P1",
}

# ---------------------------------------------------------------------------
# Reward function weights (Phase 6 MRT design)
#     R = w_c*completion + w_r*rating_norm + w_g*gad7_delta - w_e*escalation
#     clipped to [-1, 1]
# ---------------------------------------------------------------------------
REWARD_WEIGHTS: dict[str, float] = {
    "w_completion":          0.35,
    "w_rating":              0.30,
    "w_gad7_improvement":    0.25,
    "w_escalation_penalty":  0.10,
}

# ---------------------------------------------------------------------------
# SHAP weights for FAISS retrieval — sourced from Phase 2B global importance.
# Used to weight features before cosine similarity in the FAISS fallback.
# ---------------------------------------------------------------------------
SHAP_WEIGHTS: dict[str, float] = {
    "physiological_risk":      2.832,
    "composite_risk":          1.085,
    "behavioral_risk":         0.287,
    "age_norm":                0.150,
    "textual_risk":            0.079,
    "gender_enc":              0.072,
    "income_enc":              0.060,
    "marital_enc":             0.040,
    "education_enc":           0.030,
    "interaction_count_norm":  0.025,
    "last_reward_norm":        0.020,
    "escalation_count_norm":   0.015,
    "risk_tier_enc":           0.0,    # leakage feature — contributes nothing
}

# ---------------------------------------------------------------------------
# Uncertainty thresholds
# ---------------------------------------------------------------------------
CONFORMAL_SET_MAX_SIZE_FOR_SINGLETON: int = 1
UNCERTAINTY_CONFIDENCE_THRESHOLD: float = 0.55  # below this, flag for review

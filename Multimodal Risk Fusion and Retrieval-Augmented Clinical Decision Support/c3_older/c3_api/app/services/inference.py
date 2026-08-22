"""C3 core inference: classify, explain, recommend, compute_reward.

All four functions share a pattern:
  1. Convert FeatureVector → ordered numpy row
  2. FORCE risk_tier_enc = 0.0 (leakage fix, layer 2 of 2)
  3. Run the appropriate ML pipeline
  4. Return a response-shaped dict
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    INTERVENTION_PRIORITY,
    REWARD_WEIGHTS,
    TIER_INTERVENTIONS,
    TIER_LABELS,
    UNCERTAINTY_CONFIDENCE_THRESHOLD,
)
from app.models.schemas import FeatureVector
from app.services.model_loader import ModelRegistry

logger = logging.getLogger("c3.inference")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vector_to_row(features: FeatureVector, feature_cols: list[str]) -> np.ndarray:
    """Convert FeatureVector to a 1×13 numpy array in training column order.

    LEAKAGE FIX — risk_tier_enc is forced to 0.0 here even though the Pydantic
    validator already does it. Belt-and-braces: two layers.
    """
    fd = features.model_dump()
    fd["risk_tier_enc"] = 0.0  # layer 2 of leakage protection
    return np.array([[fd[c] for c in feature_cols]], dtype=np.float32)


def _row_to_dataframe(row: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Wrap a 1×13 row as a DataFrame for sklearn-style predict_proba."""
    return pd.DataFrame(row, columns=feature_cols)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------
def classify(features: FeatureVector, registry: ModelRegistry) -> dict[str, Any]:
    """Run XGBoost + calibration + conformal → return tier, probs, conformal set."""
    X = _row_to_dataframe(
        _vector_to_row(features, registry.feature_cols),
        registry.feature_cols,
    )

    # Raw probabilities
    raw_proba = registry.xgboost.predict_proba(X)[0]  # shape (3,)

    # Calibrated probabilities — isotonic regression
    try:
        cal_proba = registry.calibrator.predict_proba(X)[0]
    except Exception as e:
        logger.warning(f"Calibrator predict_proba failed ({e}), using raw proba")
        cal_proba = raw_proba

    tier = int(np.argmax(cal_proba))
    tier_label = TIER_LABELS[tier]

    # Conformal prediction set (APS)
    try:
        cp_set_mask = registry.conformal.predict(X)
        # cp_set_mask is shape (1, 3) boolean — map to tier labels
        mask = np.asarray(cp_set_mask)[0] if np.ndim(cp_set_mask) > 1 else cp_set_mask
        cp_labels = [TIER_LABELS[i] for i, m in enumerate(mask) if bool(m)]
        if not cp_labels:  # safety — never return empty
            cp_labels = [tier_label]
    except Exception as e:
        logger.warning(f"Conformal predictor failed ({e}), falling back to singleton")
        cp_labels = [tier_label]

    singleton = len(cp_labels) == 1
    max_conf = float(np.max(cal_proba))
    uncertainty = (not singleton) or (max_conf < UNCERTAINTY_CONFIDENCE_THRESHOLD)

    if uncertainty:
        intervention = "manual_review"
    else:
        intervention = TIER_INTERVENTIONS[tier][0]

    return {
        "risk_tier": tier,
        "risk_label": tier_label,
        "probabilities": {
            TIER_LABELS[i]: float(raw_proba[i]) for i in range(len(raw_proba))
        },
        "calibrated_probabilities": {
            TIER_LABELS[i]: float(cal_proba[i]) for i in range(len(cal_proba))
        },
        "conformal_set": cp_labels,
        "conformal_singleton": singleton,
        "uncertainty_flag": uncertainty,
        "intervention_type": intervention,
        "priority": INTERVENTION_PRIORITY[intervention],
    }


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------
def explain(features: FeatureVector, registry: ModelRegistry) -> dict[str, Any]:
    """Run SHAP → return feature importances + NL summary + counterfactual."""
    X = _row_to_dataframe(
        _vector_to_row(features, registry.feature_cols),
        registry.feature_cols,
    )

    # Predicted tier — needed to pick the right SHAP class
    try:
        cal_proba = registry.calibrator.predict_proba(X)[0]
    except Exception:
        cal_proba = registry.xgboost.predict_proba(X)[0]
    tier = int(np.argmax(cal_proba))

    # SHAP values
    shap_items: list[dict] = []
    top_factors: list[str] = []
    if registry.shap_explainer is not None:
        try:
            shap_out = registry.shap_explainer.shap_values(X)
            # For multiclass TreeExplainer, shap_out can be list[n_classes] of (1,n_features)
            # or a 3D array (1, n_features, n_classes). Handle both.
            if isinstance(shap_out, list):
                sv = np.asarray(shap_out[tier])[0]
            else:
                arr = np.asarray(shap_out)
                if arr.ndim == 3:
                    sv = arr[0, :, tier]
                else:
                    sv = arr[0]
            # Build feature-wise items
            for feat, val in zip(registry.feature_cols, sv):
                shap_items.append({
                    "feature": feat,
                    "shap_value": float(val),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                })
            # Top 5 by absolute SHAP value
            top_factors = [
                it["feature"]
                for it in sorted(shap_items, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
            ]
        except Exception as e:
            logger.warning(f"SHAP failed ({e}), using fallback importance")

    if not shap_items:
        # Fallback — use SHAP-weights from config as a proxy
        from app.config import SHAP_WEIGHTS
        fv = features.model_dump()
        for feat in registry.feature_cols:
            w = SHAP_WEIGHTS.get(feat, 0.0)
            val = float(fv.get(feat, 0.0)) * w
            shap_items.append({
                "feature": feat,
                "shap_value": val,
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            })
        top_factors = [
            it["feature"]
            for it in sorted(shap_items, key=lambda x: abs(x["shap_value"]), reverse=True)[:5]
        ]

    nl_summary = _build_nl_summary(tier, top_factors, registry.nl_template)
    cf_suggestion = _build_counterfactual_suggestion(tier, top_factors)

    return {
        "risk_tier": tier,
        "shap_values": shap_items,
        "top_risk_factors": top_factors,
        "nl_summary": nl_summary,
        "counterfactual_suggestion": cf_suggestion,
        "lime_top_features": None,      # LIME is cached from Phase 2B, not live
        "jaccard_agreement": 0.663,
    }


def _build_nl_summary(tier: int, top: list[str], tmpl: dict) -> str:
    label = TIER_LABELS[tier]
    if tmpl and str(tier) in tmpl:
        try:
            return tmpl[str(tier)].format(top_factors=", ".join(top[:3]))
        except Exception:
            pass
    # Default fallback templates
    top_str = ", ".join(f.replace("_", " ") for f in top[:3]) or "overall profile"
    if tier == 0:
        return f"Low anxiety risk. Main contributors: {top_str}. Continue routine monitoring."
    if tier == 1:
        return f"Medium anxiety risk. Main drivers: {top_str}. Targeted nudge recommended."
    return f"High anxiety risk. Key drivers: {top_str}. Urgent outreach recommended."


def _build_counterfactual_suggestion(tier: int, top: list[str]) -> str | None:
    if tier == 0:
        return None
    if not top:
        return None
    lever = top[0].replace("_", " ")
    return (
        f"Reducing {lever} through targeted intervention would likely shift "
        f"this patient to a lower risk tier."
    )


# ---------------------------------------------------------------------------
# recommend — FAISS retrieval with euclidean fallback
# ---------------------------------------------------------------------------
def recommend(
    features: FeatureVector,
    registry: ModelRegistry,
    k: int = 5,
) -> dict[str, Any]:
    """Retrieve k most similar cases and recommend intervention by majority vote."""
    row = _vector_to_row(features, registry.feature_cols)[0]  # shape (13,)

    if registry.faiss_available:
        return _recommend_via_faiss(row, registry, k)
    return _recommend_via_euclidean_fallback(row, registry, k)


def _recommend_via_faiss(
    row: np.ndarray, registry: ModelRegistry, k: int
) -> dict[str, Any]:
    """SHAP-weighted cosine similarity via FAISS IndexFlatIP."""
    w = registry.get_shap_weight_vector()           # (13,)
    q = (row * w).astype(np.float32)
    # L2-normalise for cosine similarity
    norm = np.linalg.norm(q) + 1e-12
    q_norm = (q / norm).reshape(1, -1)

    D, I = registry.faiss_index.search(q_norm, k)
    similar: list[dict] = []
    for idx, sim in zip(I[0], D[0]):
        if idx < 0 or idx >= len(registry.faiss_metadata):
            continue
        meta = registry.faiss_metadata[idx]
        similar.append({
            "case_id":      int(idx),
            "similarity":   float(sim),
            "risk_tier":    int(meta.get("risk_tier", 0)),
            "intervention": str(meta.get("intervention", "routine_monitoring")),
            "source":       meta.get("source"),
        })
    return _assemble_recommendation(similar, "faiss_shap_weighted")


def _recommend_via_euclidean_fallback(
    row: np.ndarray, registry: ModelRegistry, k: int
) -> dict[str, Any]:
    """Fallback retrieval: SHAP-weighted L2 distance on seed_case_base."""
    df = registry.seed_case_base
    # Align columns — seed base must contain feature_cols
    missing = [c for c in registry.feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"seed_case_base missing columns: {missing}")

    X = df[registry.feature_cols].to_numpy(dtype=np.float32)
    w = registry.get_shap_weight_vector()
    diff = (X - row) * w
    dist = np.linalg.norm(diff, axis=1)

    # Top-k smallest distances
    top_idx = np.argsort(dist)[:k]
    similar: list[dict] = []
    for idx in top_idx:
        rec = df.iloc[int(idx)]
        # Similarity = 1/(1+d) to stay in (0,1]
        sim = 1.0 / (1.0 + float(dist[idx]))
        tier_val = _extract_tier(rec)
        iv_val = _extract_intervention(rec, tier_val)
        similar.append({
            "case_id":      int(idx),
            "similarity":   float(sim),
            "risk_tier":    int(tier_val),
            "intervention": str(iv_val),
            "source":       str(rec["source"]) if "source" in rec else None,
        })
    return _assemble_recommendation(similar, "euclidean_fallback")


def _extract_tier(rec) -> int:
    for col in ("risk_tier", "tier", "y", "label"):
        if col in rec and pd.notna(rec[col]):
            return int(rec[col])
    return 0


def _extract_intervention(rec, tier: int) -> str:
    if "intervention" in rec and isinstance(rec["intervention"], str):
        return rec["intervention"]
    return TIER_INTERVENTIONS.get(tier, ["routine_monitoring"])[0]


def _assemble_recommendation(
    similar: list[dict], retriever_used: str
) -> dict[str, Any]:
    if not similar:
        return {
            "recommended_intervention": "manual_review",
            "priority": INTERVENTION_PRIORITY["manual_review"],
            "rationale": "No similar cases found — flagged for manual review.",
            "similar_cases": [],
            "retriever_used": retriever_used,
        }
    # Majority vote weighted by similarity
    votes: dict[str, float] = {}
    for s in similar:
        votes[s["intervention"]] = votes.get(s["intervention"], 0.0) + s["similarity"]
    winner = max(votes, key=votes.get)
    rationale = (
        f"Recommended '{winner}' based on {len(similar)} similar cases "
        f"(retriever: {retriever_used}, top similarity: {similar[0]['similarity']:.3f})."
    )
    return {
        "recommended_intervention": winner,
        "priority": INTERVENTION_PRIORITY.get(winner, "P3"),
        "rationale": rationale,
        "similar_cases": similar,
        "retriever_used": retriever_used,
    }


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------
def compute_reward(
    completion_flag: float,
    user_rating: float,
    gad7_pre: float,
    gad7_post: float,
    escalation_occurred: bool,
) -> dict[str, Any]:
    """Composite reward for continuous learning.

        R = w_c*completion + w_r*rating_norm + w_g*gad7_delta − w_e*escalation
        clipped to [−1, 1]

    rating_norm   = (rating − 1) / 4          # maps 1..5 → 0..1
    gad7_delta    = (pre − post) / 21         # positive = improvement
    escalation    = 1.0 if escalated else 0.0
    """
    rating_norm = (user_rating - 1.0) / 4.0
    gad7_delta = (gad7_pre - gad7_post) / 21.0
    escalation = 1.0 if escalation_occurred else 0.0

    w = REWARD_WEIGHTS
    raw_reward = (
        w["w_completion"] * float(completion_flag)
        + w["w_rating"] * rating_norm
        + w["w_gad7_improvement"] * gad7_delta
        - w["w_escalation_penalty"] * escalation
    )
    reward = float(np.clip(raw_reward, -1.0, 1.0))

    # Map [-1, 1] → [0, 1] for storage as last_reward_norm (F12)
    updated_last_reward_norm = float((reward + 1.0) / 2.0)

    return {
        "composite_reward": reward,
        "components": {
            "completion_component":  w["w_completion"] * float(completion_flag),
            "rating_component":      w["w_rating"] * rating_norm,
            "gad7_component":        w["w_gad7_improvement"] * gad7_delta,
            "escalation_penalty":    w["w_escalation_penalty"] * escalation,
        },
        "updated_last_reward_norm": updated_last_reward_norm,
    }

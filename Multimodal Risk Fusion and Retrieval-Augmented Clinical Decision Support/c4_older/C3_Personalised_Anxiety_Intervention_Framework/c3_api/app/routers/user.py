"""C3 user endpoints: register, GAD-7 submit, intervention assign."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    FeatureVector,
    GAD7SubmitRequest,
    GAD7SubmitResponse,
    InterventionAssignRequest,
    InterventionAssignResponse,
    RegisterRequest,
    RegisterResponse,
)
from app.services.auth import create_access_token, get_current_user

logger = logging.getLogger("c3.router.user")
router = APIRouter(prefix="/v3", tags=["User"])

# In-memory stores — Firebase in Phase 7
_users_by_email: dict[str, str] = {}       # email → user_id
_profiles: dict[str, dict[str, Any]] = {}  # user_id → profile


# ---------------------------------------------------------------------------
# Demographic encoding helpers
# ---------------------------------------------------------------------------
def _encode_demographics(req: RegisterRequest) -> dict[str, float]:
    return {
        "age_norm":      (req.age - 18) / 17.0,
        "gender_enc":    1.0 if req.gender == "Male" else 2.0,
        "marital_enc":   {"Married": 1.0, "Separated": 2.0, "Never": 3.0}[
            req.marital_status
        ],
        "education_enc": float(req.education_level),
        "income_enc":    min(req.income_pir / 5.0, 1.0),
    }


def _gad7_score_to_tier(score: int) -> int:
    if score <= 4:
        return 0   # minimal → Low
    if score <= 9:
        return 0   # mild → Low
    if score <= 14:
        return 1   # moderate → Medium
    return 2       # severe → High


# ---------------------------------------------------------------------------
# /v3/register
# ---------------------------------------------------------------------------
@router.post("/register", response_model=RegisterResponse)
async def register_user(req: RegisterRequest):
    """Register a new user with demographics. Returns JWT."""
    if req.email in _users_by_email:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    _users_by_email[req.email] = user_id
    _profiles[user_id] = {
        "email":                  req.email,
        "demographics":           _encode_demographics(req),
        "interaction_count_norm": 0.0,
        "last_reward_norm":       0.5,  # neutral start
        "escalation_count_norm":  0.0,
        "current_intervention":   None,
    }
    token = create_access_token({"sub": user_id})
    logger.info(f"registered user {user_id} (email hash={hash(req.email)})")
    return RegisterResponse(user_id=user_id, access_token=token)


# ---------------------------------------------------------------------------
# /v3/gad7/submit
# ---------------------------------------------------------------------------
@router.post("/gad7/submit", response_model=GAD7SubmitResponse)
async def gad7_submit(
    req: GAD7SubmitRequest,
    _user=Depends(get_current_user),
):
    """Assemble a 13-feature vector from GAD-7 + C1/C2/C4 external scores."""
    gad7_score = sum(req.gad7_answers)
    # Look up profile by patient_id (treated as user_id here)
    profile = _profiles.get(req.patient_id)
    if profile is None:
        # Default demographics — supervisor-safe fallback
        demo = {
            "age_norm": 0.35, "gender_enc": 2.0, "marital_enc": 3.0,
            "education_enc": 3.0, "income_enc": 0.5,
        }
        interaction = 0.0
        last_reward = 0.5
        escalation = 0.0
    else:
        demo = profile["demographics"]
        interaction = profile.get("interaction_count_norm", 0.0)
        last_reward = profile.get("last_reward_norm", 0.5)
        escalation = profile.get("escalation_count_norm", 0.0)

    composite = (
        0.25 * req.physiological_risk
        + 0.20 * req.behavioral_risk
        + 0.40 * req.textual_risk
    ) / 0.85

    # Build FeatureVector — risk_tier_enc will be forced to 0 by the validator
    features = FeatureVector(
        age_norm=demo["age_norm"],
        gender_enc=demo["gender_enc"],
        marital_enc=demo["marital_enc"],
        education_enc=demo["education_enc"],
        income_enc=demo["income_enc"],
        physiological_risk=req.physiological_risk,
        behavioral_risk=req.behavioral_risk,
        textual_risk=req.textual_risk,
        composite_risk=min(composite, 1.0),
        risk_tier_enc=0.0,                         # leakage fix
        interaction_count_norm=interaction,
        last_reward_norm=last_reward,
        escalation_count_norm=escalation,
    )

    return GAD7SubmitResponse(
        patient_id=req.patient_id,
        gad7_score=gad7_score,
        features=features,
        message=(
            f"GAD-7 score = {gad7_score}. Feature vector assembled. "
            f"Call /v3/risk/classify next."
        ),
    )


# ---------------------------------------------------------------------------
# /v3/intervention/assign
# ---------------------------------------------------------------------------
@router.post("/intervention/assign", response_model=InterventionAssignResponse)
async def intervention_assign(
    req: InterventionAssignRequest,
    _user=Depends(get_current_user),
):
    """Assign an intervention to a user profile (in-memory; Firebase in Phase 7)."""
    if req.patient_id not in _profiles:
        _profiles[req.patient_id] = {
            "demographics": {},
            "interaction_count_norm": 0.0,
            "last_reward_norm": 0.5,
            "escalation_count_norm": 0.0,
        }
    _profiles[req.patient_id]["current_intervention"] = {
        "intervention_type":  req.intervention_type,
        "priority":           req.priority,
        "clinician_approved": req.clinician_approved,
    }
    logger.info(
        f"intervention_assign({req.patient_id}) -> {req.intervention_type} "
        f"(approved={req.clinician_approved})"
    )
    return InterventionAssignResponse(
        patient_id=req.patient_id,
        assigned=True,
        firebase_path=f"/users/{req.patient_id}/current_intervention",
        message=f"Intervention '{req.intervention_type}' assigned.",
    )

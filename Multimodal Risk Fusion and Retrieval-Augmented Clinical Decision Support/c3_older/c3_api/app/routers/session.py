"""C3 /v3/session/complete and /v3/reward/compute endpoints."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    RewardRequest,
    RewardResponse,
    SessionCompleteRequest,
    SessionCompleteResponse,
)
from app.services.auth import get_current_user
from app.services.inference import compute_reward

logger = logging.getLogger("c3.router.session")
router = APIRouter(prefix="/v3", tags=["Session"])

# In-memory session store — replaced by Firebase in Phase 7
_sessions: dict[str, dict] = {}


@router.post("/session/complete", response_model=SessionCompleteResponse)
async def session_complete(
    req: SessionCompleteRequest,
    _user=Depends(get_current_user),
):
    """Record a completed intervention session."""
    try:
        _sessions[req.session_id] = {
            "patient_id":        req.patient_id,
            "intervention_type": req.intervention_type,
            "completion_flag":   req.completion_flag,
            "user_rating":       req.user_rating,
            "hr_mean":           req.hr_mean,
            "hrv_rmssd":         req.hrv_rmssd,
            "duration_seconds":  req.duration_seconds,
            "notes":             req.notes,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"session_complete({req.patient_id}, {req.session_id}) "
            f"completion={req.completion_flag} rating={req.user_rating}"
        )
        return SessionCompleteResponse(
            patient_id=req.patient_id,
            session_id=req.session_id,
            recorded=True,
            message="Session recorded. Call /v3/reward/compute to update reward signal.",
        )
    except Exception as e:
        logger.exception("session_complete failed")
        raise HTTPException(status_code=500, detail=f"Session record error: {e}") from e


@router.post("/reward/compute", response_model=RewardResponse)
async def reward_compute(
    req: RewardRequest,
    _user=Depends(get_current_user),
):
    """Compute composite reward R for continuous learning."""
    try:
        result = compute_reward(
            completion_flag=req.completion_flag,
            user_rating=req.user_rating,
            gad7_pre=req.gad7_pre,
            gad7_post=req.gad7_post,
            escalation_occurred=req.escalation_occurred,
        )
        # Persist the reward to the session if we have it
        if req.session_id in _sessions:
            _sessions[req.session_id]["reward"] = result["composite_reward"]
        logger.info(
            f"reward_compute({req.patient_id}) -> R={result['composite_reward']:.4f} "
            f"updated_last_reward_norm={result['updated_last_reward_norm']:.4f}"
        )
        return RewardResponse(
            patient_id=req.patient_id,
            session_id=req.session_id,
            composite_reward=result["composite_reward"],
            components=result["components"],
            updated_last_reward_norm=result["updated_last_reward_norm"],
        )
    except Exception as e:
        logger.exception("reward_compute failed")
        raise HTTPException(status_code=500, detail=f"Reward compute error: {e}") from e

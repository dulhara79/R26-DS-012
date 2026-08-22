"""C3 /v3/recommend endpoint — FAISS retrieval + intervention recommendation."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    RecommendRequest,
    RecommendResponse,
    SimilarCase,
)
from app.services.auth import get_current_user
from app.services.inference import recommend
from app.services.model_loader import get_registry

logger = logging.getLogger("c3.router.recommend")
router = APIRouter(prefix="/v3", tags=["Recommend"])


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_endpoint(
    req: RecommendRequest,
    _user=Depends(get_current_user),
):
    """Retrieve k similar cases and recommend an intervention."""
    try:
        registry = get_registry()
        result = recommend(req.features, registry, k=req.k)
        similar = [SimilarCase(**c) for c in result["similar_cases"]]
        logger.info(
            f"recommend({req.patient_id}) -> {result['recommended_intervention']} "
            f"via {result['retriever_used']} ({len(similar)} cases)"
        )
        return RecommendResponse(
            patient_id=req.patient_id,
            recommended_intervention=result["recommended_intervention"],
            priority=result["priority"],
            rationale=result["rationale"],
            similar_cases=similar,
            retriever_used=result["retriever_used"],
        )
    except Exception as e:
        logger.exception("recommend failed")
        raise HTTPException(status_code=500, detail=f"Recommend error: {e}") from e

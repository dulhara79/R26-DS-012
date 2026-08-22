"""C3 /v3/risk/* endpoints — classify and explain."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ExplainRequest,
    ExplainResponse,
    SHAPFeatureImportance,
)
from app.services.auth import get_current_user
from app.services.inference import classify, explain
from app.services.model_loader import get_registry

logger = logging.getLogger("c3.router.risk")
router = APIRouter(prefix="/v3/risk", tags=["Risk"])


@router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(
    req: ClassifyRequest,
    _user=Depends(get_current_user),
):
    """Classify a patient's anxiety risk tier."""
    try:
        registry = get_registry()
        result = classify(req.features, registry)
        logger.info(
            f"classify({req.patient_id}) -> tier={result['risk_tier']} "
            f"intervention={result['intervention_type']} uncertainty={result['uncertainty_flag']}"
        )
        return ClassifyResponse(patient_id=req.patient_id, **result)
    except Exception as e:
        logger.exception("classify failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e


@router.post("/explain", response_model=ExplainResponse)
async def explain_endpoint(
    req: ExplainRequest,
    _user=Depends(get_current_user),
):
    """Explain a classification via SHAP + NL + counterfactual."""
    try:
        registry = get_registry()
        result = explain(req.features, registry)
        shap_items = [SHAPFeatureImportance(**it) for it in result["shap_values"]]
        return ExplainResponse(
            patient_id=req.patient_id,
            risk_tier=result["risk_tier"],
            shap_values=shap_items,
            top_risk_factors=result["top_risk_factors"],
            nl_summary=result["nl_summary"],
            counterfactual_suggestion=result["counterfactual_suggestion"],
            lime_top_features=result["lime_top_features"],
            jaccard_agreement=result["jaccard_agreement"],
        )
    except Exception as e:
        logger.exception("explain failed")
        raise HTTPException(status_code=500, detail=f"Explain error: {e}") from e

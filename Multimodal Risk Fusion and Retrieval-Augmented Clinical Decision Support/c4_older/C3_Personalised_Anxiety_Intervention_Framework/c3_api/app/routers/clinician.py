"""C3 /v3/clinician/review endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import ClinicianReview, ClinicianReviewResponse
from app.services.auth import get_current_user

logger = logging.getLogger("c3.router.clinician")
router = APIRouter(prefix="/v3/clinician", tags=["Clinician"])

# In-memory pending review store
_pending_reviews: dict[str, dict] = {}


@router.post("/review", response_model=ClinicianReviewResponse)
async def clinician_review(
    review: ClinicianReview,
    _user=Depends(get_current_user),
):
    """Clinician approve / modify / reject an intervention."""
    if review.action == "reject":
        final = "manual_review"
        message = (
            f"Clinician {review.clinician_id} rejected the intervention. "
            f"Flagged for manual review."
        )
    elif review.action == "modify":
        if not review.modified_intervention:
            raise HTTPException(
                status_code=422,
                detail="modified_intervention is required when action='modify'",
            )
        final = review.modified_intervention
        message = (
            f"Clinician {review.clinician_id} modified intervention to '{final}'."
        )
    else:  # approve
        final = _pending_reviews.get(review.patient_id, {}).get(
            "intervention", "routine_monitoring"
        )
        message = f"Clinician {review.clinician_id} approved intervention '{final}'."

    _pending_reviews[review.patient_id] = {
        "clinician_id":  review.clinician_id,
        "action":        review.action,
        "final":         final,
        "notes":         review.clinical_notes,
    }
    logger.info(
        f"clinician_review({review.patient_id}): {review.action} -> {final}"
    )
    return ClinicianReviewResponse(
        patient_id=review.patient_id,
        status=review.action,
        final_intervention=final,
        message=message,
    )

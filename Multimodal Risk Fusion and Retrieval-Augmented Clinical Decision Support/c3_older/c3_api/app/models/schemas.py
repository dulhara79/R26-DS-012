"""C3 API Pydantic schemas — all request/response models.

Key safety: FeatureVector.risk_tier_enc is FORCED to 0.0 at validation time.
This is the first of TWO layers of leakage protection (the second is in the
inference handlers themselves).
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RiskTier(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


# ---------------------------------------------------------------------------
# Core feature vector — 13 features, F10 (risk_tier_enc) forced to 0
# ---------------------------------------------------------------------------
class FeatureVector(BaseModel):
    """13-dimensional input vector.

    risk_tier_enc (F10) is ALWAYS overwritten to 0.0 by the validator —
    this is the leakage fix, never relax it.
    """

    age_norm:               float = Field(..., ge=0.0, le=1.0)
    gender_enc:             float = Field(..., ge=1.0, le=2.0)
    marital_enc:            float = Field(..., ge=1.0, le=3.0)
    education_enc:          float = Field(..., ge=1.0, le=5.0)
    income_enc:             float = Field(..., ge=0.0, le=1.0)
    physiological_risk:     float = Field(..., ge=0.0, le=1.0)
    behavioral_risk:        float = Field(..., ge=0.0, le=1.0)
    textual_risk:           float = Field(..., ge=0.0, le=1.0)
    composite_risk:         float = Field(..., ge=0.0, le=1.0)
    risk_tier_enc:          float = Field(default=0.0)
    interaction_count_norm: float = Field(..., ge=0.0, le=1.0)
    last_reward_norm:       float = Field(..., ge=0.0, le=1.0)
    escalation_count_norm:  float = Field(..., ge=0.0, le=1.0)

    @field_validator("risk_tier_enc", mode="before")
    @classmethod
    def _force_zero(cls, v):
        """LEAKAGE FIX — F10 is always 0.0 at inference regardless of input."""
        return 0.0


# ---------------------------------------------------------------------------
# /v3/risk/classify
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    features: FeatureVector


class ClassifyResponse(BaseModel):
    patient_id: str
    risk_tier: int
    risk_label: str
    probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float]
    conformal_set: list[str]
    conformal_singleton: bool
    uncertainty_flag: bool
    intervention_type: str
    priority: str


# ---------------------------------------------------------------------------
# /v3/risk/explain
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    features: FeatureVector


class SHAPFeatureImportance(BaseModel):
    feature: str
    shap_value: float
    direction: str      # "increases_risk" | "decreases_risk"


class ExplainResponse(BaseModel):
    patient_id: str
    risk_tier: int
    shap_values: list[SHAPFeatureImportance]
    top_risk_factors: list[str]
    nl_summary: str
    counterfactual_suggestion: Optional[str] = None
    lime_top_features: Optional[list[str]] = None
    jaccard_agreement: float = 0.663  # Phase 2B reported mean


# ---------------------------------------------------------------------------
# /v3/recommend
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    features: FeatureVector
    k: int = Field(default=5, ge=1, le=20)


class SimilarCase(BaseModel):
    case_id: int
    similarity: float
    risk_tier: int
    intervention: str
    source: Optional[str] = None


class RecommendResponse(BaseModel):
    patient_id: str
    recommended_intervention: str
    priority: str
    rationale: str
    similar_cases: list[SimilarCase]
    retriever_used: str  # "faiss_shap_weighted" | "euclidean_fallback"


# ---------------------------------------------------------------------------
# /v3/session/complete
# ---------------------------------------------------------------------------
class SessionCompleteRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    intervention_type: str
    completion_flag: bool
    user_rating: float = Field(..., ge=1.0, le=5.0)
    hr_mean: Optional[float] = Field(default=None, ge=30.0, le=220.0)
    hrv_rmssd: Optional[float] = Field(default=None, ge=0.0, le=300.0)
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = Field(default=None, max_length=2000)


class SessionCompleteResponse(BaseModel):
    patient_id: str
    session_id: str
    recorded: bool
    message: str


# ---------------------------------------------------------------------------
# /v3/reward/compute
# ---------------------------------------------------------------------------
class RewardRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=128)
    completion_flag: float = Field(..., ge=0.0, le=1.0)
    user_rating: float = Field(..., ge=1.0, le=5.0)
    gad7_pre: float = Field(..., ge=0.0, le=21.0)
    gad7_post: float = Field(..., ge=0.0, le=21.0)
    escalation_occurred: bool = False


class RewardResponse(BaseModel):
    patient_id: str
    session_id: str
    composite_reward: float
    components: dict[str, float]
    updated_last_reward_norm: float


# ---------------------------------------------------------------------------
# /v3/register
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    age: int = Field(..., ge=18, le=35)
    gender: str = Field(..., pattern="^(Male|Female)$")
    marital_status: str = Field(..., pattern="^(Married|Separated|Never)$")
    education_level: int = Field(..., ge=1, le=5)
    income_pir: float = Field(..., ge=0.0, le=5.0)


class RegisterResponse(BaseModel):
    user_id: str
    access_token: str
    token_type: str = "bearer"


# ---------------------------------------------------------------------------
# /v3/gad7/submit
# ---------------------------------------------------------------------------
class GAD7SubmitRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    gad7_answers: list[int] = Field(..., min_length=7, max_length=7)
    physiological_risk: float = Field(..., ge=0.0, le=1.0)
    behavioral_risk: float = Field(..., ge=0.0, le=1.0)
    textual_risk: float = Field(..., ge=0.0, le=1.0)

    @field_validator("gad7_answers")
    @classmethod
    def _valid_answers(cls, v):
        if any(a < 0 or a > 3 for a in v):
            raise ValueError("each GAD-7 answer must be in [0, 3]")
        return v


class GAD7SubmitResponse(BaseModel):
    patient_id: str
    gad7_score: int
    features: FeatureVector
    message: str


# ---------------------------------------------------------------------------
# /v3/clinician/review
# ---------------------------------------------------------------------------
class ClinicianReview(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    clinician_id: str = Field(..., min_length=1, max_length=128)
    action: str = Field(..., pattern="^(approve|modify|reject)$")
    modified_intervention: Optional[str] = None
    clinical_notes: Optional[str] = Field(default=None, max_length=4000)


class ClinicianReviewResponse(BaseModel):
    patient_id: str
    status: str
    final_intervention: str
    message: str


# ---------------------------------------------------------------------------
# /v3/intervention/assign
# ---------------------------------------------------------------------------
class InterventionAssignRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=128)
    intervention_type: str
    priority: str
    clinician_approved: bool = False


class InterventionAssignResponse(BaseModel):
    patient_id: str
    assigned: bool
    firebase_path: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Internal token payload
# ---------------------------------------------------------------------------
class TokenData(BaseModel):
    user_id: Optional[str] = None

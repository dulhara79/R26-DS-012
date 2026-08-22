from __future__ import annotations

from datetime import datetime

from .models import EvidenceLevel, KnowledgeLayer, QueryAnalysis
from .util import clamp, freshness_score


EVIDENCE_SCORES: dict[EvidenceLevel, float] = {
    EvidenceLevel.CLINICAL_GUIDELINE: 1.00,
    EvidenceLevel.META_ANALYSIS: 0.94,
    EvidenceLevel.SYSTEMATIC_REVIEW: 0.90,
    EvidenceLevel.RANDOMIZED_CONTROLLED_TRIAL: 0.82,
    EvidenceLevel.COHORT_STUDY: 0.68,
    EvidenceLevel.CASE_CONTROL_STUDY: 0.62,
    EvidenceLevel.CROSS_SECTIONAL_STUDY: 0.52,
    EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION: 0.82,
    EvidenceLevel.RESEARCH_UPDATE: 0.48,
    EvidenceLevel.CASE_REPORT: 0.32,
    EvidenceLevel.GENERAL_INFORMATION: 0.35,
    EvidenceLevel.UNKNOWN: 0.25,
}


def evidence_score(level: EvidenceLevel) -> float:
    return EVIDENCE_SCORES[level]


def classify_evidence(
    default: EvidenceLevel,
    publication_types: list[str],
    title: str,
) -> EvidenceLevel:
    haystack = " ".join([title, *publication_types]).lower()
    if "retracted publication" in haystack or "retraction of publication" in haystack:
        return EvidenceLevel.UNKNOWN
    if "practice guideline" in haystack or "clinical guideline" in haystack or "guideline" in haystack:
        return EvidenceLevel.CLINICAL_GUIDELINE
    if "meta-analysis" in haystack or "meta analysis" in haystack:
        return EvidenceLevel.META_ANALYSIS
    if "systematic review" in haystack:
        return EvidenceLevel.SYSTEMATIC_REVIEW
    if any(term in haystack for term in ["randomized controlled trial", "randomised controlled trial", "clinical trial"]):
        return EvidenceLevel.RANDOMIZED_CONTROLLED_TRIAL
    if "cohort" in haystack:
        return EvidenceLevel.COHORT_STUDY
    if "case-control" in haystack or "case control" in haystack:
        return EvidenceLevel.CASE_CONTROL_STUDY
    if "cross-sectional" in haystack or "cross sectional" in haystack:
        return EvidenceLevel.CROSS_SECTIONAL_STUDY
    if "case report" in haystack:
        return EvidenceLevel.CASE_REPORT
    return default


def is_retracted(publication_types: list[str], metadata: dict[str, object]) -> bool:
    values = " ".join(publication_types).lower()
    status = str(metadata.get("publication_status", "")).lower()
    return any(
        marker in values or marker in status
        for marker in [
            "retracted publication",
            "retraction of publication",
            "withdrawn",
        ]
    )


def calculate_freshness(
    layer: KnowledgeLayer,
    updated_at: datetime | None,
    published_at: datetime | None,
    clinical_half_life_days: int,
    research_half_life_days: int,
) -> float:
    timestamp = updated_at or published_at
    half_life = (
        clinical_half_life_days
        if layer == KnowledgeLayer.CLINICAL_CORE
        else research_half_life_days
    )
    return freshness_score(timestamp, half_life_days=half_life)


def applicability_score(
    topics: list[str],
    query: QueryAnalysis,
    layer: KnowledgeLayer,
) -> float:
    normalized_topics = {topic.lower().replace(" ", "_") for topic in topics}
    subtype_match = 0.0
    if not query.anxiety_subtypes:
        subtype_match = 0.65
    elif normalized_topics.intersection(query.anxiety_subtypes):
        subtype_match = 1.0
    elif any("anxiety" in topic for topic in normalized_topics):
        subtype_match = 0.60
    else:
        subtype_match = 0.35

    if not query.preferred_layers:
        layer_match = 0.80
    elif layer in query.preferred_layers:
        layer_match = 1.0
    else:
        layer_match = 0.55
    return clamp((0.75 * subtype_match) + (0.25 * layer_match))

"""Deterministic evidence-question requirements for a clinician-facing prototype.

These are conservative English intent heuristics, not clinical validation.
They never generate medical content or relax exact-source citation checks.
"""
from __future__ import annotations

from datetime import UTC, datetime
import re

from .models import EvidenceLevel, QueryAnalysis, QueryIntent, SearchHit


def is_review_request(query: str) -> bool:
    return bool(re.search(r"\b(?:systematic reviews?|meta[- ]analys(?:is|es)|umbrella reviews?)\b", query, re.I))


def is_comparison_request(query: str) -> bool:
    return bool(re.search(r"\b(?:compar(?:e|ed|ison|ative)|versus|vs\.?)\b", query, re.I))


def is_conflict_request(query: str) -> bool:
    return bool(re.search(r"\b(?:contradict(?:s|ed|ing|ion|ory)?|conflict(?:s|ing)?|disagree(?:ment|s)?)\b", query, re.I))


def is_diagnosis_request(analysis: QueryAnalysis) -> bool:
    return analysis.intent == QueryIntent.DIAGNOSIS


def is_authoritative_medication_request(analysis: QueryAnalysis) -> bool:
    return (
        analysis.intent == QueryIntent.MEDICATION
        and bool(re.search(r"\b(?:authoritative|guidelines?|who|nice|official)\b", analysis.normalized_query))
    )


def is_treatment_review_request(analysis: QueryAnalysis) -> bool:
    return is_review_request(analysis.normalized_query) and bool(
        re.search(r"\b(?:treat(?:ment|ments|ing)?|therap(?:y|ies)|interventions?|medications?)\b", analysis.normalized_query)
    )


def publication_is_recent(hit: SearchHit, *, years: int = 3) -> bool:
    """Recent means published within last ~3 calendar years, not updated recently."""
    published = hit.chunk.published_at
    if published is None:
        return False
    published_utc = published.replace(tzinfo=UTC) if published.tzinfo is None else published.astimezone(UTC)
    now = datetime.now(UTC)
    return 0 <= (now - published_utc).days <= years * 366


def eligible_document_for_question(hit: SearchHit, analysis: QueryAnalysis) -> bool:
    """Enforce explicitly requested evidence type/recency before sentence selection."""
    if is_review_request(analysis.normalized_query):
        if hit.chunk.evidence_level not in {
            EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.META_ANALYSIS
        }:
            return False
        if analysis.wants_recent and not publication_is_recent(hit):
            return False
    if is_authoritative_medication_request(analysis):
        if hit.chunk.evidence_level not in {
            EvidenceLevel.CLINICAL_GUIDELINE,
            EvidenceLevel.GOVERNMENT_HEALTH_INFORMATION,
        }:
            return False
    return True

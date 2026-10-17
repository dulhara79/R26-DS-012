"""Strict bibliographic discovery, not clinical treatment-effectiveness inference.

For "Which recent systematic reviews address anxiety treatment?" the relevant
answer is an indexed, publication-dated source title. Do not treat that title
as evidence that its intervention works. No free-form source synthesis.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from .models import EvidenceLevel, QueryAnalysis, SearchHit


_REVIEW = re.compile(r"\b(?:systematic reviews?|meta[- ]analys(?:is|es))\b", re.I)
_DISCOVERY = re.compile(
    r"^\s*(?:what|which|list|identify|find|show me)\b", re.I
)
_EFFECT_ASK = re.compile(
    r"\b(?:show|say|conclude|prove|demonstrate|findings?|"
    r"results?|effective|efficacy|work(?:s|ed)?|improve|reduce)\b", re.I
)
_ANXIETY_TITLE = re.compile(
    r"\b(?:anxiety(?:[- ]disorders?)?|generaliz(?:ed|ised) anxiety|"
    r"panic disorder|social phobia|social anxiety|agoraphobia|phobi(?:a|as))\b",
    re.I,
)
_TREATMENT_TITLE = re.compile(
    r"\b(?:treatments?|therap(?:y|ies)|pharmacotherap(?:y|ies)|"
    r"pharmacological|psychotherap(?:y|ies)|CBT|"
    r"antidepressants?|SSRIs?|SNRIs?|exposure|"
    r"interventions?|brain stimulation|neuromodulation)\b",
    re.I,
)
_NON_TREATMENT_TITLE = re.compile(
    r"\b(?:biomarkers?|microbiome|prevention|risk factors?|"
    r"epidemiology|prevalence|genetics?)\b", re.I,
)


def is_review_discovery_query(analysis: QueryAnalysis) -> bool:
    """Narrow bibliographic question detection, excluding efficacy questions."""
    q = analysis.normalized_query
    return bool(
        _DISCOVERY.search(q)
        and _REVIEW.search(q)
        and not _EFFECT_ASK.search(q)
    )


def review_title_is_relevant(hit: SearchHit, analysis: QueryAnalysis) -> bool:
    """Filter metadata titles; never infer efficacy from a publication title."""
    if hit.excluded_due_to_conflict:
        return False
    if not is_review_discovery_query(analysis):
        return False
    if hit.chunk.evidence_level not in (
        EvidenceLevel.SYSTEMATIC_REVIEW, EvidenceLevel.META_ANALYSIS
    ):
        return False
    title = hit.chunk.title
    if not (_ANXIETY_TITLE.search(title) and _TREATMENT_TITLE.search(title)):
        return False
    if _NON_TREATMENT_TITLE.search(title):
        return False
    if analysis.wants_recent:
        published = hit.chunk.published_at
        if published is None:
            return False
        now = datetime.now(UTC)
        if not (now - timedelta(days=3 * 365) <= published <= now):
            return False
    return bool(hit.chunk.url)

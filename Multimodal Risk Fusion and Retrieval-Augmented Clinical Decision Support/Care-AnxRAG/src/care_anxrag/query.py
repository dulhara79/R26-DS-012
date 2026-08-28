from __future__ import annotations

import re

from .models import KnowledgeLayer, QueryAnalysis, QueryIntent, SafetyLevel
from .util import normalize_whitespace


_SUBTYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "generalized_anxiety_disorder": (
        "generalized anxiety",
        "generalised anxiety",
        " gad ",
        "constant worry",
        "excessive worry",
    ),
    "panic_disorder": (
        "panic attack",
        "panic disorder",
        "heart racing",
        "sudden fear",
    ),
    "social_anxiety_disorder": (
        "social anxiety",
        "social phobia",
        "fear of embarrassment",
        "public speaking",
    ),
    "agoraphobia": (
        "agoraphobia",
        "afraid to leave",
        "crowded places",
    ),
    "specific_phobia": (
        "specific phobia",
        "phobia",
        "fear of flying",
        "fear of spiders",
    ),
    "separation_anxiety": (
        "separation anxiety",
        "separation fear",
    ),
    "health_anxiety": (
        "health anxiety",
        "illness anxiety",
        "hypochondria",
    ),
}

_TREATMENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "cognitive_behavioral_therapy": (
        " cbt ",
        "cognitive behavioral therapy",
        "cognitive behavioural therapy",
        "cognitive-behavioral therapy",
        "cognitive-behavioural therapy",
    ),
}


class QueryAnalyzer:
    def analyze(
        self,
        query: str,
        safety_level: SafetyLevel = SafetyLevel.NORMAL,
        safety_reason: str | None = None,
    ) -> QueryAnalysis:
        normalized = normalize_whitespace(query).lower()
        padded = f" {normalized} "

        retrieval_terms = [normalized]

        if re.search(r"\bcbt\b", normalized):
            retrieval_terms.extend(
                [
                    "cognitive behavioral therapy",
                    "cognitive behavioural therapy",
                ]
            )

        if re.search(r"\bgad\b", normalized):
            retrieval_terms.append("generalized anxiety disorder")

        retrieval_query = normalize_whitespace(
            " ".join(dict.fromkeys(retrieval_terms))
        )

        subtypes = [
            subtype
            for subtype, patterns in _SUBTYPE_PATTERNS.items()
            if any(pattern in padded for pattern in patterns)
        ]

        treatments = [
            treatment
            for treatment, patterns in _TREATMENT_PATTERNS.items()
            if any(pattern in padded for pattern in patterns)
        ]

        intent = self._intent(normalized)

        wants_recent = bool(
            re.search(
                r"\b(latest|recent|new|newest|current|today|202[4-9])\b",
                normalized,
            )
        )

        if intent == QueryIntent.RECENT_RESEARCH:
            wants_recent = True

        preferred_layers = (
            [KnowledgeLayer.RESEARCH_FRONTIER, KnowledgeLayer.CLINICAL_CORE]
            if wants_recent
            else [KnowledgeLayer.CLINICAL_CORE, KnowledgeLayer.RESEARCH_FRONTIER]
        )

        population = None

        if re.search(
            r"\b(child|children|kid|adolescent|teen|teenager|youth)\b",
            normalized,
        ):
            population = "children_and_adolescents"
        elif re.search(
            r"\b(pregnant|pregnancy|postpartum|perinatal)\b",
            normalized,
        ):
            population = "perinatal"
        elif re.search(r"\b(older adult|elderly|senior)\b", normalized):
            population = "older_adults"
        elif re.search(r"\badult\b", normalized):
            population = "adults"

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized,
            retrieval_query=retrieval_query,
            intent=intent,
            anxiety_subtypes=subtypes,
            treatments=treatments,
            population=population,
            wants_recent=wants_recent,
            preferred_layers=preferred_layers,
            safety_level=safety_level,
            safety_reason=safety_reason,
        )

    @staticmethod
    def _intent(query: str) -> QueryIntent:
        if re.search(
            r"\b(latest|recent|new research|new study|current evidence)\b",
            query,
        ):
            return QueryIntent.RECENT_RESEARCH

        if re.search(
            r"\b(medicine|medication|drug|ssri|snri|benzodiazepine|dose|side effect)\b",
            query,
        ):
            return QueryIntent.MEDICATION

        if re.search(
            r"\b(treat|treatment|therapy|cbt|exposure|intervention|recommended)\b",
            query,
        ):
            return QueryIntent.TREATMENT

        if re.search(
            r"\b(symptom|sign|feel like|heart racing|dizzy|sweating)\b",
            query,
        ):
            return QueryIntent.SYMPTOMS

        if re.search(
            r"\b(diagnos|do i have|is this anxiety|test for)\b",
            query,
        ):
            return QueryIntent.DIAGNOSIS

        if re.search(
            r"\b(cause|why|risk factor|trigger)\b",
            query,
        ):
            return QueryIntent.CAUSES

        if re.search(
            r"\b(coping|cope|self help|self-help|breathing|relaxation|what can i do)\b",
            query,
        ):
            return QueryIntent.SELF_HELP

        return QueryIntent.GENERAL
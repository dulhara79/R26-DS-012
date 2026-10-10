from __future__ import annotations

import re
from collections.abc import Iterable


_SUBTYPE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "generalized_anxiety_disorder": (
        re.compile(r"\bgeneraliz(?:ed|ised) anxiety disorder\b", re.I),
        re.compile(r"\bGAD\b", re.I),
    ),
    "panic_disorder": (
        re.compile(r"\bpanic disorder\b", re.I),
        re.compile(r"\bpanic attacks?\b", re.I),
    ),
    "social_anxiety_disorder": (
        re.compile(r"\bsocial anxiety disorder\b", re.I),
        re.compile(r"\bsocial phobia\b", re.I),
    ),
    "agoraphobia": (re.compile(r"\bagoraphobia\b", re.I),),
    "specific_phobia": (
        re.compile(r"\bspecific phobia(?:s)?\b", re.I),
    ),
    "separation_anxiety": (
        re.compile(r"\bseparation anxiety(?: disorder)?\b", re.I),
    ),
    "health_anxiety": (
        re.compile(r"\bhealth anxiety\b", re.I),
        re.compile(r"\billness anxiety(?: disorder)?\b", re.I),
    ),
}

_TREATMENT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "cognitive_behavioral_therapy": (
        re.compile(r"\bCBT\b", re.I),
        re.compile(r"\bcognitive[- ]behaviou?ral therapy\b", re.I),
    ),
    "metacognitive_therapy": (
        re.compile(r"\bmetacognitive therapy\b", re.I),
        re.compile(r"\bMCT\b", re.I),
    ),
    "exposure_therapy": (
        re.compile(r"\bexposure[- ](?:based )?(?:therapy|treatment)\b", re.I),
        re.compile(r"\bexposure therapy\b", re.I),
    ),
    "virtual_reality_exposure_therapy": (
        re.compile(r"\bvirtual reality exposure therapy\b", re.I),
        re.compile(r"\bVRET\b", re.I),
    ),
    "acceptance_and_commitment_therapy": (
        re.compile(r"\bacceptance and commitment therapy\b", re.I),
    ),
    "internet_delivered_cbt": (
        re.compile(
            r"\binternet[- ](?:based|delivered) cognitive[- ]behaviou?ral therapy\b",
            re.I,
        ),
        re.compile(r"\biCBT\b", re.I),
    ),
    "ssri": (
        re.compile(r"\bSSRIs?\b", re.I),
        re.compile(r"\bselective serotonin reuptake inhibitors?\b", re.I),
    ),
    "snri": (
        re.compile(r"\bSNRIs?\b", re.I),
        re.compile(r"\bserotonin[- ]norepinephrine reuptake inhibitors?\b", re.I),
        re.compile(r"\bserotonin[- ]noradrenaline reuptake inhibitors?\b", re.I),
    ),
    "benzodiazepine": (
        re.compile(r"\bbenzodiazepines?\b", re.I),
    ),
}

_OUTCOME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "symptom_severity": (
        re.compile(r"\bsymptom severity\b", re.I),
        re.compile(r"\banxiety severity\b", re.I),
        re.compile(r"\bGAD-?7\b", re.I),
    ),
    "treatment_response": (
        re.compile(r"\btreatment response\b", re.I),
        re.compile(r"\bresponders?\b", re.I),
        re.compile(r"\bresponse rate\b", re.I),
    ),
    "remission": (
        re.compile(r"\bremission\b", re.I),
        re.compile(r"\bremission rate\b", re.I),
    ),
    "relapse": (
        re.compile(r"\brelapse\b", re.I),
        re.compile(r"\brecurrence\b", re.I),
    ),
    "quality_of_life": (
        re.compile(r"\bquality of life\b", re.I),
        re.compile(r"\bQoL\b", re.I),
    ),
    "functional_impairment": (
        re.compile(r"\bfunctional impairment\b", re.I),
        re.compile(r"\bfunctioning\b", re.I),
        re.compile(r"\bwork impairment\b", re.I),
    ),
    "adverse_effects": (
        re.compile(r"\badverse (?:effect|effects|event|events)\b", re.I),
        re.compile(r"\bside effects?\b", re.I),
        re.compile(r"\btolerability\b", re.I),
    ),
}

_COMORBIDITY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "major_depressive_disorder": (
        re.compile(r"\bmajor depressive disorder\b", re.I),
        re.compile(r"\bMDD\b", re.I),
        re.compile(r"\bcomorbid depression\b", re.I),
        re.compile(r"\bdepressive symptoms\b", re.I),
    ),
    "substance_use_disorder": (
        re.compile(r"\bsubstance use disorder\b", re.I),
        re.compile(r"\balcohol use disorder\b", re.I),
    ),
    "insomnia": (
        re.compile(r"\binsomnia\b", re.I),
    ),
    "post_traumatic_stress_disorder": (
        re.compile(r"\bpost[- ]traumatic stress disorder\b", re.I),
        re.compile(r"\bPTSD\b", re.I),
    ),
    "obsessive_compulsive_disorder": (
        re.compile(r"\bobsessive[- ]compulsive disorder\b", re.I),
        re.compile(r"\bOCD\b", re.I),
    ),
}

_POPULATION_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "children_and_adolescents": (
        re.compile(r"\bchildren?\b", re.I),
        re.compile(r"\badolescents?\b", re.I),
        re.compile(r"\bteenagers?\b", re.I),
        re.compile(r"\byouth\b", re.I),
        re.compile(r"\bpediatric\b", re.I),
        re.compile(r"\bpaediatric\b", re.I),
    ),
    "perinatal": (
        re.compile(r"\bpregnan(?:t|cy)\b", re.I),
        re.compile(r"\bpostpartum\b", re.I),
        re.compile(r"\bperinatal\b", re.I),
    ),
    "older_adults": (
        re.compile(r"\bolder adults?\b", re.I),
        re.compile(r"\belderly\b", re.I),
        re.compile(r"\bseniors?\b", re.I),
        re.compile(r"\baged 65\b", re.I),
        re.compile(r"\b65 years and (?:older|above)\b", re.I),
    ),
    "adults": (
        re.compile(r"(?<!older )\badults?\b", re.I),
    ),
}


def _extract(
    text: str,
    patterns: dict[str, tuple[re.Pattern[str], ...]],
) -> set[str]:
    return {
        concept
        for concept, aliases in patterns.items()
        if any(pattern.search(text or "") for pattern in aliases)
    }


def extract_subtype_concepts(text: str) -> set[str]:
    return _extract(text, _SUBTYPE_PATTERNS)


def extract_treatment_concepts(text: str) -> set[str]:
    return _extract(text, _TREATMENT_PATTERNS)


def extract_outcome_concepts(text: str) -> set[str]:
    return _extract(text, _OUTCOME_PATTERNS)


def extract_comorbidity_concepts(text: str) -> set[str]:
    return _extract(text, _COMORBIDITY_PATTERNS)


def extract_population_concepts(text: str) -> set[str]:
    concepts = _extract(text, _POPULATION_PATTERNS)
    if "older_adults" in concepts:
        concepts.discard("adults")
    return concepts


def _annotation_values(
    raw: object,
) -> list[str]:
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if not isinstance(
        raw,
        (list, tuple, set),
    ):
        return []
    values: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value and value not in values:
            values.append(value)
    return values


def build_clinical_evidence_facets(
    text: str,
    topics: Iterable[str],
    annotated_pico: object = None,
) -> dict[str, object]:
    """Build traceable clinical facets without model-generated medical facts."""
    normalized_topics = {
        str(topic).strip().lower().replace(" ", "_")
        for topic in topics
        if str(topic).strip()
    }
    subtype_topics = (
        normalized_topics
        & set(_SUBTYPE_PATTERNS)
    )

    pico_input = (
        annotated_pico
        if isinstance(annotated_pico, dict)
        else {}
    )
    pico = {
        "population": _annotation_values(
            pico_input.get("population")
        ),
        "intervention": _annotation_values(
            pico_input.get("intervention")
        ),
        "comparator": _annotation_values(
            pico_input.get("comparator")
        ),
        "outcome": _annotation_values(
            pico_input.get("outcome")
        ),
    }

    return {
        "anxiety_subtypes": sorted(
            extract_subtype_concepts(text)
            | subtype_topics
        ),
        "treatments": sorted(
            extract_treatment_concepts(text)
        ),
        "populations": sorted(
            extract_population_concepts(text)
        ),
        "outcomes": sorted(
            extract_outcome_concepts(text)
        ),
        "comorbidities": sorted(
            extract_comorbidity_concepts(text)
        ),
        "pico": pico,
        "provenance": {
            "normalized_concepts": (
                "deterministic_phrase_match"
            ),
            "pico": (
                "source_or_reviewer_metadata"
                if any(pico.values())
                else "not_annotated"
            ),
        },
    }


def concept_set_compatibility(
    requested: Iterable[str],
    evidence: Iterable[str],
    *,
    mismatch_score: float,
    unknown_score: float,
) -> float:
    requested_set = set(requested)
    if not requested_set:
        return 1.0

    evidence_set = set(evidence)
    if requested_set & evidence_set:
        return 1.0
    if evidence_set:
        return mismatch_score
    return unknown_score


def treatment_compatibility(
    requested_treatments: Iterable[str],
    evidence_text: str,
) -> float:
    requested = set(requested_treatments)
    if not requested:
        return 1.0
    evidence = extract_treatment_concepts(evidence_text)
    if requested & evidence:
        return 1.0
    if evidence:
        return 0.55
    return 0.80


def population_compatibility(
    requested_population: str | None,
    evidence_text: str,
) -> float:
    if not requested_population:
        return 1.0
    evidence = extract_population_concepts(evidence_text)
    if requested_population in evidence:
        return 1.0
    if evidence:
        return 0.60
    return 0.90


def supports_explicit_clinical_context(
    requested_subtypes: Iterable[str],
    requested_treatments: Iterable[str],
    requested_population: str | None,
    requested_outcomes: Iterable[str],
    requested_comorbidities: Iterable[str],
    topics: Iterable[str],
    evidence_text: str,
    evidence_facets: dict[str, object] | None = None,
) -> bool:
    """Require one evidence item to support the explicitly requested clinical context."""
    requested_treatments_set = set(requested_treatments)
    if requested_treatments_set:
        evidence_treatments = extract_treatment_concepts(evidence_text)
        if not requested_treatments_set.issubset(evidence_treatments):
            return False

    requested_subtypes_set = set(requested_subtypes)
    if requested_subtypes_set:
        normalized_topics = {
            str(topic).strip().lower().replace(" ", "_")
            for topic in topics
        }
        evidence_subtypes = extract_subtype_concepts(evidence_text) | (
            normalized_topics & set(_SUBTYPE_PATTERNS)
        )
        if not requested_subtypes_set.issubset(evidence_subtypes):
            return False

    if requested_population:
        evidence_populations = extract_population_concepts(evidence_text)
        if evidence_populations and requested_population not in evidence_populations:
            return False

    facets = evidence_facets if isinstance(evidence_facets, dict) else {}

    requested_outcomes_set = set(requested_outcomes)
    if requested_outcomes_set:
        facet_outcomes = facets.get("outcomes", [])
        evidence_outcomes = (
            {
                str(value)
                for value in facet_outcomes
                if str(value).strip()
            }
            if isinstance(facet_outcomes, list)
            else set()
        )
        evidence_outcomes |= extract_outcome_concepts(evidence_text)
        if not requested_outcomes_set.issubset(evidence_outcomes):
            return False

    requested_comorbidities_set = set(requested_comorbidities)
    if requested_comorbidities_set:
        facet_comorbidities = facets.get("comorbidities", [])
        evidence_comorbidities = (
            {
                str(value)
                for value in facet_comorbidities
                if str(value).strip()
            }
            if isinstance(facet_comorbidities, list)
            else set()
        )
        evidence_comorbidities |= extract_comorbidity_concepts(evidence_text)
        if not requested_comorbidities_set.issubset(evidence_comorbidities):
            return False

    return True


def supports_explicit_treatment_query(
    requested_subtypes: Iterable[str],
    requested_treatments: Iterable[str],
    requested_population: str | None,
    topics: Iterable[str],
    evidence_text: str,
) -> bool:
    """Return True only when one evidence chunk supports the requested treatment context.

    Treatment support must come from the evidence text itself. Topics may help establish
    the anxiety subtype, but a treatment topic alone is not accepted as proof that the
    chunk directly discusses that treatment.
    """
    requested_treatments_set = set(requested_treatments)
    if not requested_treatments_set:
        return True

    evidence_treatments = extract_treatment_concepts(evidence_text)
    if not (requested_treatments_set & evidence_treatments):
        return False

    requested_subtypes_set = set(requested_subtypes)
    if requested_subtypes_set:
        normalized_topics = {
            str(topic).strip().lower().replace(" ", "_")
            for topic in topics
        }
        evidence_subtypes = extract_subtype_concepts(evidence_text) | (
            normalized_topics & set(_SUBTYPE_PATTERNS)
        )
        if not (requested_subtypes_set & evidence_subtypes):
            return False

    if requested_population:
        evidence_populations = extract_population_concepts(evidence_text)
        if evidence_populations and requested_population not in evidence_populations:
            return False

    return True

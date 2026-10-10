"""Conservative, deterministic selection of complete source sentences.

These English phrase rules filter obvious non-answers; they are not a clinical
correctness classifier. Retrieval quality and exact-source grounding still apply.
"""

from __future__ import annotations

import re

from .clinical_match import (
    extract_population_concepts,
    extract_subtype_concepts,
    supports_explicit_clinical_context,
)
from .models import QueryAnalysis, QueryIntent, SearchHit
from .util import content_tokens, source_sentences


class InsufficientAnswerEvidence(ValueError):
    """Relevant passages contained no eligible answer sentence."""

    def __init__(self) -> None:
        super().__init__("insufficient_answer_evidence")


_NON_FINDING = re.compile(
    r"\b(?:aims?|objectives?|purpose|goal)\b.{0,120}"
    r"\b(?:study|investigat\w*|assess\w*|evaluat\w*|compar\w*|determin\w*|"
    r"examin\w*|test\w*|was|were|is|are)\b"
    r"|\b(?:aimed|sought|intend(?:ed)?|plan(?:ned)?) to\b"
    r"|\b(?:study|trial|review|we) (?:investigat\w*|evaluat\w*|assess\w*|compar\w*|examin\w*)\b"
    r"|\b(?:was|were) (?:evaluated|assessed|measured|examined)\b"
    r"|\bhypothes(?:is|es|iz\w*|is\w*)\b"
    r"|\b(?:will|expected to)\b"
    r"|^(?:objectives?|aims?|methods?|design|protocol)\s*:"
    r"|\b(?:randomly assigned|randomi[sz]ed to|recruited|enrolled)\b",
    re.I,
)
_FINDING = re.compile(
    r"\b(?:reduc(?:e[sd]?|ing|tion)|improv(?:e[sd]?|ing|ement)|benefit(?:s|ed)?|"
    r"effective(?:ness)?|efficacy|efficacious|remission|response rate|relapse|"
    r"adverse (?:effects?|events?)|side effects?|recommended|first[- ]line|"
    r"uncertain|inconclusive|insufficient|no (?:significant )?(?:difference|effect)|"
    r"not differ|equivalent|most evidence)\b"
    r"|\bevidence (?:supports?|suggests?)\b"
    r"|\bevidence[- ](?:based|supported) (?:treatments?|approaches?|interventions?|options?)\b"
    r"|\b(?:can|may) help\b",
    re.I,
)
_EFFECTIVENESS_QUESTION = re.compile(
    r"\b(?:effective(?:ness)?|efficacy|works?|helps?|helpful|benefits?|"
    r"reduc\w*|improv\w*|outcomes?|results?|findings?)\b",
    re.I,
)
_DESCRIPTIVE_QUESTION = re.compile(r"\b(?:discussed|mentioned|listed|delivered|offered|used)\b", re.I)
_RESULT_STATEMENT = re.compile(
    r"\b(?:reduced|improved|benefited|found|observed|showed|results)\b"
    r"|\b(?:did not|had no effect|no (?:significant )?difference|"
    r"not effective|uncertain|inconclusive|insufficient)\b",
    re.I,
)


def select_answer_sentence(hit: SearchHit, analysis: QueryAnalysis) -> str | None:
    """Return an eligible source sentence, retaining its wording and scope.

    Treatment/effectiveness questions require a finding or evidence statement,
    rather than an objective or a delivery description. Explicit clinical facets
    must be supported by the sentence itself, not borrowed from another sentence.
    Known population scope must remain in the selected text; otherwise omit it.
    """
    if hit.excluded_due_to_conflict:
        return None

    candidates = [
        part for part in source_sentences(hit.chunk.text)
        if not part.lstrip().startswith("#")
    ]
    query_tokens = set(content_tokens(analysis.normalized_query))
    asks_for_findings = bool(_EFFECTIVENESS_QUESTION.search(analysis.normalized_query)) or (
        "evidence" in query_tokens
        and not _DESCRIPTIVE_QUESTION.search(analysis.normalized_query)
    )
    requires_finding = (
        analysis.intent in {QueryIntent.TREATMENT, QueryIntent.MEDICATION}
        and asks_for_findings
    ) or (
        bool(analysis.treatments)
        and analysis.intent == QueryIntent.RECENT_RESEARCH
    )
    evidence_populations = extract_population_concepts(
        f"{hit.chunk.title}\n{hit.chunk.text}"
    )
    requested_populations = extract_population_concepts(analysis.normalized_query)
    if analysis.population:
        requested_populations.add(analysis.population)
    facets = hit.chunk.metadata.get("clinical_facets", {})
    if isinstance(facets, dict):
        populations = facets.get("populations", [])
        if isinstance(populations, list):
            evidence_populations.update(
                value for value in populations
                if isinstance(value, str)
                and value in {"children_and_adolescents", "adults", "older_adults", "perinatal"}
            )

    eligible: list[str] = []
    for sentence in candidates:
        if _NON_FINDING.search(sentence):
            continue
        if requires_finding and not _FINDING.search(sentence):
            continue
        if not supports_explicit_clinical_context(
            analysis.anxiety_subtypes,
            analysis.treatments,
            analysis.population,
            analysis.outcomes,
            analysis.comorbidities,
            [],
            sentence,
        ):
            continue
        sentence_populations = extract_population_concepts(sentence)
        if evidence_populations and not (sentence_populations & evidence_populations):
            continue
        if requested_populations and sentence_populations and not (
            requested_populations & sentence_populations
        ):
            continue
        sentence_tokens = set(content_tokens(sentence))
        if query_tokens and not query_tokens.intersection(sentence_tokens) and not analysis.treatments:
            continue
        if "anxiety" in query_tokens and not (
            "anxiety" in sentence_tokens or extract_subtype_concepts(sentence)
        ):
            continue
        eligible.append(sentence)

    if not eligible:
        return None

    def score(sentence: str) -> tuple[bool, float, int]:
        overlap = len(query_tokens & set(content_tokens(sentence))) / max(1, len(query_tokens))
        return requires_finding and bool(_RESULT_STATEMENT.search(sentence)), overlap, -len(sentence)

    return max(eligible, key=score) if query_tokens else eligible[0]

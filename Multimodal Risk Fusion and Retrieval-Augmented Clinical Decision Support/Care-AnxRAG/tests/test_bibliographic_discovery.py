"""Bibliographic discovery is explicitly distinct from claiming clinical efficacy."""
from datetime import UTC, datetime, timedelta

import pytest

from care_anxrag.answer_selection import InsufficientAnswerEvidence
from care_anxrag.bibliography import is_review_discovery_query, review_title_is_relevant
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import EvidenceLevel, GeneratedPayload
from care_anxrag.nli import HeuristicNliClassifier
from care_anxrag.query import QueryAnalyzer
from test_answer_selection import hit, retrieval, rag_for

DISCOVERY = "What recent systematic reviews address anxiety-disorder treatment?"
TITLE = "Psychological therapies for anxiety disorders: a systematic review."


def review_hit(title=TITLE, *, abstract="The abstract discusses methods and participants."):
    result = hit(abstract, title=title)
    result.chunk.url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    result.chunk.evidence_level = EvidenceLevel.SYSTEMATIC_REVIEW
    result.chunk.published_at = datetime.now(UTC) - timedelta(days=30)
    return result


def test_clinician_review_discovery_question_recognized_but_effectiveness_is_not():
    assert is_review_discovery_query(QueryAnalyzer().analyze(DISCOVERY))
    assert not is_review_discovery_query(QueryAnalyzer().analyze(
        "What do systematic reviews say about CBT effectiveness for anxiety?"
    ))


def test_review_title_is_bibliographic_answer_not_a_clinical_finding(settings):
    first = review_hit()
    result = rag_for(settings, retrieval(DISCOVERY, [first])).answer(DISCOVERY)
    assert not result.abstained
    assert result.citations[0].title == TITLE
    assert result.answer.startswith(f"- {TITLE} [S1]")
    assert "Bibliographic matches only." in result.answer
    assert result.citations[0].published_at is not None


def test_source_title_only_grounded_in_explicit_bibliography_mode():
    first = review_hit()
    payload = GeneratedPayload(
        answer=f"- {TITLE} [S1]",
        cited_source_ids=["S1"],
    )
    verifier = ClaimGroundingVerifier(HeuristicNliClassifier())
    assert not verifier.verify(payload, [first]).supported
    assert verifier.verify(payload, [first], allow_metadata_titles=True).supported


def test_biomarker_review_and_prevention_review_excluded():
    biomarker = review_hit(
        "Inflammatory biomarkers and oral microbiome changes in anxiety: a systematic review."
    )
    prevention = review_hit(
        "Sleep interventions for the prevention of pediatric anxiety: systematic review."
    )
    analysis = QueryAnalyzer().analyze(DISCOVERY)
    assert not review_title_is_relevant(biomarker, analysis)
    assert not review_title_is_relevant(prevention, analysis)


def test_old_or_unknown_date_cannot_be_called_recent():
    old = review_hit()
    old.chunk.published_at = datetime(2021, 1, 1, tzinfo=UTC)
    unknown = review_hit()
    unknown.chunk.published_at = None
    analysis = QueryAnalyzer().analyze(DISCOVERY)
    assert not review_title_is_relevant(old, analysis)
    assert not review_title_is_relevant(unknown, analysis)


def test_non_review_medical_article_not_falsely_listed_as_systematic_review():
    article = review_hit()
    article.chunk.evidence_level = EvidenceLevel.RANDOMIZED_CONTROLLED_TRIAL
    with pytest.raises(InsufficientAnswerEvidence):
        EvidenceOnlyGenerator().generate(
            DISCOVERY, [article], retrieval(DISCOVERY, [article])
        )


def test_review_discovery_avoids_duplicate_chunks_from_one_publication():
    a = review_hit()
    b = review_hit()
    b.chunk.chunk_id = "a-second-chunk-of-the-same-paper"
    b.chunk.document_id = a.chunk.document_id
    result = EvidenceOnlyGenerator().generate(
        DISCOVERY, [a, b], retrieval(DISCOVERY, [a, b])
    )
    assert result.cited_source_ids == ["S1"]

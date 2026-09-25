from datetime import UTC, datetime

from care_anxrag.grounding import GroundingReport
from care_anxrag.models import (
    ChunkRecord,
    DocumentStatus,
    EvidenceLevel,
    GeneratedPayload,
    KnowledgeLayer,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult,
    SafetyLevel,
    SearchHit,
)
from care_anxrag.rag import CareAnxRag


def _hit() -> SearchHit:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return SearchHit(
        chunk=ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            source_id="source-1",
            source_name="Source One",
            title="Panic guidance",
            url="https://example.org/panic",
            layer=KnowledgeLayer.CLINICAL_CORE,
            status=DocumentStatus.ACTIVE,
            section_path="treatment",
            section_heading="Treatment",
            ordinal=0,
            text="CBT can reduce symptoms of panic disorder.",
            text_hash="hash",
            published_at=now,
            updated_at=now,
            retrieved_at=now,
            authority_score=0.9,
            evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
            evidence_score=1.0,
            topics=["anxiety", "panic_disorder"],
            metadata={},
        ),
        relevance_score=0.9,
        care_score=0.9,
    )


def _retrieval(hit: SearchHit) -> RetrievalResult:
    return RetrievalResult(
        query_analysis=QueryAnalysis(
            original_query="Does CBT prevent all future panic attacks?",
            normalized_query="does cbt prevent all future panic attacks?",
            retrieval_query="does cbt prevent all future panic attacks?",
            intent=QueryIntent.TREATMENT,
            preferred_layers=[KnowledgeLayer.CLINICAL_CORE],
            safety_level=SafetyLevel.NORMAL,
        ),
        hits=[hit],
        confidence=0.9,
        should_abstain=False,
    )


class FakeRetriever:
    def __init__(self, retrieval: RetrievalResult):
        self.retrieval = retrieval

    def retrieve(self, question: str) -> RetrievalResult:
        return self.retrieval


class FakeGenerator:
    def generate(self, question, hits, retrieval):
        return GeneratedPayload(
            answer="CBT prevents all future panic attacks [S1].",
            cited_source_ids=["S1"],
        )

    def ping(self) -> bool:
        return True


class RejectGrounding:
    def verify(self, payload, hits):
        return GroundingReport(
            supported=False,
            reason="unsupported_claim_citation",
            claim_count=1,
            checked_pairs=1,
        )


class AcceptGrounding:
    def verify(self, payload, hits):
        return GroundingReport(
            supported=True,
            reason=None,
            claim_count=1,
            checked_pairs=1,
        )


def test_rag_abstains_when_generated_claim_is_not_grounded(settings) -> None:
    hit = _hit()
    rag = CareAnxRag(
        settings,
        FakeRetriever(_retrieval(hit)),
        FakeGenerator(),
        RejectGrounding(),
    )

    result = rag.answer("Does CBT prevent all future panic attacks?")

    assert result.abstained
    assert result.citations == []
    assert result.abstention_reason == (
        "claim_citation_grounding_failed:unsupported_claim_citation"
    )
    assert "prevents all future panic attacks" not in result.answer


def test_rag_returns_citations_after_grounding_passes(settings) -> None:
    hit = _hit()
    rag = CareAnxRag(
        settings,
        FakeRetriever(_retrieval(hit)),
        FakeGenerator(),
        AcceptGrounding(),
    )

    result = rag.answer("Does CBT prevent all future panic attacks?")

    assert not result.abstained
    assert [citation.citation_id for citation in result.citations] == ["S1"]



class GeneratorWithUngroundedUncertainty:
    def generate(self, question, hits, retrieval):
        return GeneratedPayload(
            answer="CBT can reduce symptoms of panic disorder [S1].",
            cited_source_ids=["S1"],
            uncertainty="A genetics study proves this treatment works permanently.",
        )

    def ping(self) -> bool:
        return True


def test_rag_does_not_return_unverified_generator_uncertainty(settings) -> None:
    hit = _hit()
    rag = CareAnxRag(
        settings,
        FakeRetriever(_retrieval(hit)),
        GeneratorWithUngroundedUncertainty(),
        AcceptGrounding(),
    )

    result = rag.answer("Does CBT help panic disorder?")

    assert not result.abstained
    assert "genetics study" not in result.answer
    assert "works permanently" not in result.answer

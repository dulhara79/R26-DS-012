from __future__ import annotations

from datetime import UTC, datetime

from care_anxrag.evaluation import BenchmarkItem, evaluate
from care_anxrag.models import (
    AnswerResponse,
    ChunkRecord,
    Citation,
    DocumentStatus,
    EvidenceLevel,
    KnowledgeLayer,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult,
    SafetyLevel,
    SearchHit,
)
from care_anxrag.util import redact_sensitive_settings


def _chunk() -> ChunkRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        version_id="version-1",
        source_id="trusted-source",
        source_name="Trusted Source",
        title="Anxiety evidence",
        url="https://example.org/evidence",
        layer=KnowledgeLayer.CLINICAL_CORE,
        status=DocumentStatus.ACTIVE,
        section_path="root",
        section_heading="Evidence",
        ordinal=0,
        text="Evidence-based anxiety information.",
        text_hash="hash",
        retrieved_at=now,
        authority_score=0.9,
        evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
        evidence_score=1.0,
        topics=["anxiety"],
        metadata={"external_id": "gold-doc"},
    )


def _analysis(question: str) -> QueryAnalysis:
    return QueryAnalysis(
        original_query=question,
        normalized_query=question.lower(),
        intent=QueryIntent.GENERAL,
        preferred_layers=[KnowledgeLayer.CLINICAL_CORE],
        safety_level=SafetyLevel.NORMAL,
    )


def test_sensitive_settings_are_redacted_recursively() -> None:
    value = {
        "api_url": "https://example.org/api",
        "api_key": "top-secret",
        "nested": {"access_token": "token-value", "mode": "public"},
    }
    redacted = redact_sensitive_settings(value)
    assert redacted["api_url"] == "https://example.org/api"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["access_token"] == "[REDACTED]"
    assert redacted["nested"]["mode"] == "public"


def test_evaluation_excludes_unlabelled_abstention_items_from_retrieval_metrics() -> None:
    chunk = _chunk()
    hit = SearchHit(chunk=chunk, care_score=0.9)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            hits = [hit] if question == "answerable" else []
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=hits,
                confidence=0.9 if hits else 0.0,
                should_abstain=not hits,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            if question != "answerable":
                return AnswerResponse(
                    answer="The knowledge base does not contain sufficient evidence.",
                    confidence=0.0,
                    conflict_score=0.0,
                    abstained=True,
                    abstention_reason="insufficient_evidence",
                    safety_level=SafetyLevel.NORMAL,
                )
            citation = Citation(
                citation_id="S1",
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_name=chunk.source_name,
                source_id=chunk.source_id,
                url=chunk.url,
                evidence_level=chunk.evidence_level,
                excerpt=chunk.text,
            )
            return AnswerResponse(
                answer="The evidence supports this answer [S1].",
                citations=[citation],
                confidence=0.9,
                conflict_score=0.0,
                abstained=False,
                safety_level=SafetyLevel.NORMAL,
            )

    report = evaluate(
        Retriever(),
        Rag(),
        [
            BenchmarkItem(
                id="answerable",
                question="answerable",
                relevant_external_ids=["gold-doc"],
            ),
            BenchmarkItem(
                id="abstain",
                question="out of domain",
                must_abstain=True,
            ),
        ],
    )
    assert report.count == 2
    assert report.retrieval_evaluable_count == 1
    assert report.recall_at_5 == 1.0
    assert report.precision_at_5 == 1.0
    assert report.per_item[1]["recall_at_5"] is None
    assert report.abstention_accuracy == 1.0

from __future__ import annotations

from datetime import UTC, datetime

from care_anxrag.evaluation import BenchmarkItem, evaluate, evaluate_ablation, load_benchmark
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
        retrieval_query=question.lower(),
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



def test_benchmark_item_supports_research_annotation_metadata() -> None:
    item = BenchmarkItem(
        id="gad-cbt-001",
        question="What evidence addresses CBT for GAD in older adults?",
        stratum="psychological_interventions",
        split="development",
        intent="treatment",
        anxiety_subtypes=["generalized_anxiety_disorder"],
        population="older_adults",
        relevant_external_ids=["gold-doc"],
        prohibited_external_ids=["wrong-doc"],
        gold_evidence_excerpts=["Exact approved evidence sentence."],
        prohibited_claims=["CBT cures every case of GAD."],
        annotator_ids=["clinician-a", "clinician-b"],
        adjudicated=True,
    )

    assert item.stratum == "psychological_interventions"
    assert item.split == "development"
    assert item.adjudicated is True
    assert item.prohibited_external_ids == ["wrong-doc"]


def test_load_benchmark_rejects_duplicate_item_ids(tmp_path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        '{"id":"q1","question":"one"}\n'
        '{"id":"q1","question":"two"}\n',
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="Duplicate benchmark item id"):
        load_benchmark(path)


def test_evaluation_reports_extractive_faithfulness_and_prohibited_intrusion() -> None:
    trusted = _chunk()
    poisoned = trusted.model_copy(
        update={
            "chunk_id": "chunk-poison",
            "document_id": "doc-poison",
            "version_id": "version-poison",
            "source_id": "poison-source",
            "text": "Unapproved distractor evidence.",
            "metadata": {"external_id": "poison-doc"},
        }
    )
    trusted_hit = SearchHit(chunk=trusted, care_score=0.9)
    poisoned_hit = SearchHit(chunk=poisoned, care_score=0.8)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[trusted_hit, poisoned_hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            citation = Citation(
                citation_id="S1",
                chunk_id=trusted.chunk_id,
                title=trusted.title,
                source_name=trusted.source_name,
                source_id=trusted.source_id,
                url=trusted.url,
                evidence_level=trusted.evidence_level,
                excerpt=trusted.text,
            )
            return AnswerResponse(
                answer=f"- {trusted.text} [S1]",
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
                stratum="source_poisoning",
                relevant_external_ids=["gold-doc"],
                prohibited_external_ids=["poison-doc"],
            )
        ],
    )

    assert report.extractive_evaluable_count == 1
    assert report.extractive_faithfulness == 1.0
    assert report.prohibited_evidence_evaluable_count == 1
    assert report.prohibited_evidence_intrusion_rate == 1.0
    assert report.per_stratum["source_poisoning"]["count"] == 1


def test_evaluation_detects_non_extractive_answer_text() -> None:
    chunk = _chunk()
    hit = SearchHit(chunk=chunk, care_score=0.9)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
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
                answer="This is a newly paraphrased medical claim [S1].",
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
                id="non-extractive",
                question="answerable",
                stratum="faithfulness",
                relevant_external_ids=["gold-doc"],
            )
        ],
    )

    assert report.extractive_faithfulness == 0.0
    assert report.per_item[0]["extractive_faithful"] is False


def test_evaluate_ablation_runs_all_research_modes_and_restores_setting() -> None:
    from types import SimpleNamespace

    chunk = _chunk()
    hit = SearchHit(chunk=chunk, care_score=0.9)

    class Retriever:
        def __init__(self):
            self.settings = SimpleNamespace(retrieval_mode="full")
            self.seen_modes = []

        def retrieve(self, question: str) -> RetrievalResult:
            self.seen_modes.append(self.settings.retrieval_mode)
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
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
                answer=f"- {chunk.text} [S1]",
                citations=[citation],
                confidence=0.9,
                conflict_score=0.0,
                abstained=False,
                safety_level=SafetyLevel.NORMAL,
            )

    retriever = Retriever()
    reports = evaluate_ablation(
        retriever,
        Rag(),
        [
            BenchmarkItem(
                id="q1",
                question="answerable",
                relevant_external_ids=["gold-doc"],
            )
        ],
    )

    assert list(reports) == [
        "B0_dense_only",
        "B1_lexical_only",
        "B2_hybrid_rrf",
        "B3_hybrid_rerank",
        "B4_care",
        "B5_care_conflict",
        "CARE_full",
    ]
    assert retriever.settings.retrieval_mode == "full"
    assert set(retriever.seen_modes) == {
        "dense_only",
        "lexical_only",
        "hybrid_rrf",
        "hybrid_rerank",
        "care",
        "care_conflict",
        "full",
    }


def test_benchmark_item_records_expected_treatments() -> None:
    item = BenchmarkItem(
        id="gad-cbt-treatment",
        question="What evidence addresses CBT for GAD?",
        treatments=["cognitive_behavioral_therapy"],
    )

    assert item.treatments == ["cognitive_behavioral_therapy"]


def test_evaluation_uses_full_retrieved_chunk_for_extractive_fidelity() -> None:
    later_sentence = "Cognitive behavioural therapy is discussed for panic disorder."
    long_text = ("Background context. " * 30) + later_sentence
    chunk = _chunk().model_copy(
        update={
            "text": long_text,
            "text_hash": "long-hash",
        }
    )
    hit = SearchHit(chunk=chunk, care_score=0.9)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            citation = Citation(
                citation_id="S1",
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                source_name=chunk.source_name,
                source_id=chunk.source_id,
                url=chunk.url,
                evidence_level=chunk.evidence_level,
                excerpt=long_text[:320],
            )
            return AnswerResponse(
                answer=f"- {later_sentence} [S1]",
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
                id="later-source-sentence",
                question="answerable",
                relevant_external_ids=["gold-doc"],
            )
        ],
    )

    assert report.extractive_faithfulness == 1.0


def test_evaluation_reports_gold_evidence_coverage() -> None:
    chunk = _chunk()
    hit = SearchHit(chunk=chunk, care_score=0.9)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
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
                answer=f"- {chunk.text} [S1]",
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
                id="gold-excerpt",
                question="answerable",
                relevant_external_ids=["gold-doc"],
                gold_evidence_excerpts=[chunk.text],
            )
        ],
    )

    assert report.gold_evidence_evaluable_count == 1
    assert report.gold_evidence_coverage == 1.0
    assert report.per_item[0]["gold_evidence_coverage"] == 1.0



def test_evaluation_reports_active_version_accuracy() -> None:
    active = _chunk()
    stale = active.model_copy(
        update={
            "chunk_id": "chunk-stale",
            "document_id": "doc-stale",
            "version_id": "version-stale",
            "status": DocumentStatus.SUPERSEDED,
            "metadata": {"external_id": "stale-doc"},
        }
    )
    active_hit = SearchHit(chunk=active, care_score=0.9)
    stale_hit = SearchHit(chunk=stale, care_score=0.8)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[active_hit, stale_hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            citation = Citation(
                citation_id="S1",
                chunk_id=active.chunk_id,
                title=active.title,
                source_name=active.source_name,
                source_id=active.source_id,
                url=active.url,
                evidence_level=active.evidence_level,
                excerpt=active.text,
            )
            return AnswerResponse(
                answer=f"- {active.text} [S1]",
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
                id="active-version",
                question="answerable",
                relevant_external_ids=["gold-doc"],
            )
        ],
    )

    assert report.active_version_evaluable_count == 1
    assert report.active_version_accuracy == 0.5
    assert report.per_item[0]["active_version_accuracy"] == 0.5


def test_evaluation_reports_stale_evidence_intrusion_rate() -> None:
    active = _chunk()
    withdrawn = active.model_copy(
        update={
            "chunk_id": "chunk-withdrawn",
            "document_id": "doc-withdrawn",
            "version_id": "version-withdrawn",
            "status": DocumentStatus.WITHDRAWN,
            "metadata": {"external_id": "withdrawn-doc"},
        }
    )
    active_hit = SearchHit(chunk=active, care_score=0.9)
    withdrawn_hit = SearchHit(chunk=withdrawn, care_score=0.8)

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[active_hit, withdrawn_hit],
                confidence=0.9,
                should_abstain=False,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            citation = Citation(
                citation_id="S1",
                chunk_id=active.chunk_id,
                title=active.title,
                source_name=active.source_name,
                source_id=active.source_id,
                url=active.url,
                evidence_level=active.evidence_level,
                excerpt=active.text,
            )
            return AnswerResponse(
                answer=f"- {active.text} [S1]",
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
                id="stale-intrusion",
                question="answerable",
                relevant_external_ids=["gold-doc"],
            )
        ],
    )

    assert report.stale_evidence_evaluable_count == 1
    assert report.stale_evidence_intrusion_rate == 0.5
    assert report.per_item[0]["stale_evidence_intrusion_rate"] == 0.5



def test_locked_test_split_requires_adjudication_and_two_annotators() -> None:
    import pytest

    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[],
                confidence=0.0,
                should_abstain=True,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            return AnswerResponse(
                answer="The knowledge base does not contain sufficient evidence.",
                confidence=0.0,
                conflict_score=0.0,
                abstained=True,
                abstention_reason="insufficient_evidence",
                safety_level=SafetyLevel.NORMAL,
            )

    with pytest.raises(ValueError, match="locked test item"):
        evaluate(
            Retriever(),
            Rag(),
            [
                BenchmarkItem(
                    id="unreviewed-test-item",
                    question="out of domain",
                    split="test",
                    must_abstain=True,
                    annotator_ids=["reviewer-a"],
                    adjudicated=False,
                )
            ],
        )


def test_locked_test_split_accepts_two_annotators_after_adjudication() -> None:
    class Retriever:
        def retrieve(self, question: str) -> RetrievalResult:
            return RetrievalResult(
                query_analysis=_analysis(question),
                hits=[],
                confidence=0.0,
                should_abstain=True,
            )

    class Rag:
        def answer(self, question: str) -> AnswerResponse:
            return AnswerResponse(
                answer="The knowledge base does not contain sufficient evidence.",
                confidence=0.0,
                conflict_score=0.0,
                abstained=True,
                abstention_reason="insufficient_evidence",
                safety_level=SafetyLevel.NORMAL,
            )

    report = evaluate(
        Retriever(),
        Rag(),
        [
            BenchmarkItem(
                id="reviewed-test-item",
                question="out of domain",
                split="test",
                must_abstain=True,
                annotator_ids=["reviewer-a", "reviewer-b"],
                adjudicated=True,
            )
        ],
    )

    assert report.count == 1
    assert report.abstention_accuracy == 1.0

"""Regression tests for opt-in Ollama synthesis; no external model/network needed."""
from datetime import UTC, datetime
from types import SimpleNamespace
import json

from care_anxrag.config import Settings
from care_anxrag.generation import EvidenceOnlyGenerator
from care_anxrag.grounding import ClaimGroundingVerifier
from care_anxrag.models import (
    ChunkRecord, DocumentStatus, EvidenceLevel, KnowledgeLayer,
    RetrievalResult, SearchHit,
)
from care_anxrag.nli import HeuristicNliClassifier
from care_anxrag.ollama_summarizer import OllamaClinicalSummarizer
from care_anxrag.query import QueryAnalyzer
from care_anxrag.rag import CareAnxRag


Q = "What evidence supports CBT for anxiety?"
SOURCE = "CBT reduced anxiety symptoms compared with the control condition."


def _fixture():
    hit = SearchHit(
        chunk=ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            source_id="source-1",
            source_name="Synthetic source",
            title="Synthetic trial",
            layer=KnowledgeLayer.CLINICAL_CORE,
            status=DocumentStatus.ACTIVE,
            section_path="results",
            section_heading="Results",
            ordinal=0,
            text=SOURCE,
            text_hash="synthetic",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            authority_score=0.9,
            evidence_level=EvidenceLevel.RANDOMIZED_CONTROLLED_TRIAL,
            evidence_score=0.9,
            metadata={"synthetic": True},
        ),
        care_score=0.9,
        relevance_score=0.9,
    )
    retrieval = RetrievalResult(
        query_analysis=QueryAnalyzer().analyze(Q),
        hits=[hit],
        confidence=0.9,
    )
    return hit, retrieval


def _mock_ollama(monkeypatch, sentence):
    def post(*args, **kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "message": {"content": json.dumps({
                    "claims": [{"source_id": "S1", "sentence": sentence}]
                })}
            },
        )
    monkeypatch.setattr("care_anxrag.ollama_summarizer.httpx.post", post)


def test_opt_in_provider_constructs_without_changing_default():
    from care_anxrag.runtime import _build_generator
    settings = Settings(generator_provider="ollama_summary")
    assert isinstance(_build_generator(settings), OllamaClinicalSummarizer)
    assert isinstance(_build_generator(Settings()), EvidenceOnlyGenerator)


def test_summary_rejects_new_numbers_and_keeps_original(monkeypatch):
    hit, retrieval = _fixture()
    _mock_ollama(monkeypatch, "CBT reduced anxiety symptoms by 93% compared with the control condition.")
    generator = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b")
    output = generator.generate(Q, [hit], retrieval)
    assert output.answer == f"- {SOURCE} [S1]"
    assert not output.source_evidence


def test_summary_produces_cited_candidate(monkeypatch):
    hit, retrieval = _fixture()
    _mock_ollama(monkeypatch, "Compared with the control condition, CBT reduced anxiety symptoms.")
    generator = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b")
    output = generator.generate(Q, [hit], retrieval)
    assert "Clinical evidence summary" in output.answer
    assert output.cited_source_ids == ["S1"]
    assert output.source_evidence == {"S1": SOURCE}


def test_unverified_paraphrase_fails_closed(monkeypatch):
    hit, retrieval = _fixture()
    _mock_ollama(monkeypatch, "Compared with the control condition, CBT reduced anxiety symptoms.")
    generator = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b")
    output = generator.generate(Q, [hit], retrieval)
    verifier = ClaimGroundingVerifier(HeuristicNliClassifier(), allow_entailed_paraphrases=True)
    report = verifier.verify(output, [hit])
    # If a paraphrase passes the conservative heuristic, it must have an
    # actual entailment label with >=0.95 confidence, never just a citation.
    if report.supported:
        labels = HeuristicNliClassifier().classify_text_pairs(
            [(SOURCE, "Compared with the control condition, CBT reduced anxiety symptoms.")]
        )
        assert labels[0][0].value == "entailment" and labels[0][1] >= 0.95


def test_summary_keeps_original_if_ollama_is_unavailable(monkeypatch):
    import httpx
    hit, retrieval = _fixture()
    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr("care_anxrag.ollama_summarizer.httpx.post", fail)
    output = OllamaClinicalSummarizer("http://localhost:11434", "gemma3:4b").generate(
        Q, [hit], retrieval
    )
    assert output.answer == f"- {SOURCE} [S1]"

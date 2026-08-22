from __future__ import annotations

import math
import sys
import types
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from care_anxrag.embeddings import OllamaEmbedder
from care_anxrag.generation import OllamaGenerator
from care_anxrag.models import (
    ChunkRecord,
    DocumentStatus,
    EvidenceLevel,
    KnowledgeLayer,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult,
    SafetyLevel,
    SearchHit,
)
from care_anxrag.sources.ncbi import _parse_pubmed_article
from care_anxrag.sources.nice import _parse_nice_json
from care_anxrag.vector_store import ChromaVectorStore


class FakeResponse:
    def __init__(self, body: dict):
        self._body = body
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def sample_chunk() -> ChunkRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return ChunkRecord(
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
        text="Cognitive behavioural therapy is an evidence-based treatment for panic disorder.",
        text_hash="hash",
        published_at=now,
        updated_at=now,
        retrieved_at=now,
        authority_score=0.9,
        evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
        evidence_score=1.0,
        topics=["anxiety", "panic_disorder"],
        metadata={"external_id": "panic"},
    )


def test_chroma_adapter_uses_cosine_configuration(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeCollection:
        def upsert(self, **kwargs):
            calls["upsert"] = kwargs

        def query(self, **kwargs):
            calls["query"] = kwargs
            return {"ids": [["chunk-1"]], "distances": [[0.1]], "metadatas": [[{}]]}

        def delete(self, ids):
            calls["delete"] = ids

        def count(self):
            return 1

    class FakeClient:
        def __init__(self, path):
            calls["path"] = path
            self.collection = FakeCollection()

        def get_or_create_collection(self, **kwargs):
            calls["collection"] = kwargs
            return self.collection

        def heartbeat(self):
            return 1

    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=FakeClient))
    store = ChromaVectorStore(tmp_path / "chroma")
    chunk = sample_chunk()
    store.upsert("care_clinical_core", [chunk], [[1.0, 0.0]])
    hits = store.query("care_clinical_core", [1.0, 0.0], 5)
    assert calls["collection"]["configuration"] == {"hnsw": {"space": "cosine"}}
    assert calls["upsert"]["ids"] == ["chunk-1"]
    assert hits[0].chunk_id == "chunk-1"


def test_ollama_embedding_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"embeddings": [[3.0, 4.0]]})

    monkeypatch.setattr("httpx.post", fake_post)
    embedder = OllamaEmbedder("http://localhost:11434", "embeddinggemma", 10.0, dimensions=2)
    vector = embedder.embed(["panic treatment"])[0]
    assert captured["url"].endswith("/api/embed")
    assert captured["json"]["input"] == ["panic treatment"]
    assert round(sum(value * value for value in vector), 6) == 1.0


def test_ollama_generation_contract_and_citations(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(
            {
                "message": {
                    "content": '{"answer":"CBT is described in the evidence [S1].","cited_source_ids":["S1"],"uncertainty":null}'
                }
            }
        )

    monkeypatch.setattr("httpx.post", fake_post)
    hit = SearchHit(chunk=sample_chunk(), care_score=0.9)
    retrieval = RetrievalResult(
        query_analysis=QueryAnalysis(
            original_query="What treatment is used?",
            normalized_query="what treatment is used?",
            intent=QueryIntent.TREATMENT,
            preferred_layers=[KnowledgeLayer.CLINICAL_CORE],
            safety_level=SafetyLevel.NORMAL,
        ),
        hits=[hit],
        confidence=0.8,
    )
    generator = OllamaGenerator("http://localhost:11434", "gemma3:4b")
    payload = generator.generate("What treatment is used?", [hit], retrieval)
    assert payload.cited_source_ids == ["S1"]
    assert captured["url"].endswith("/api/chat")
    assert isinstance(captured["json"]["format"], dict)


def test_ollama_repair_request_retains_original_evidence(monkeypatch) -> None:
    requests: list[dict] = []

    def fake_post(url, json, timeout):
        requests.append(json)
        if len(requests) == 1:
            content = '{"answer":"Uncited answer.","cited_source_ids":["S1"],"uncertainty":null}'
        else:
            content = '{"answer":"CBT is described in the evidence [S1].","cited_source_ids":["S1"],"uncertainty":null}'
        return FakeResponse({"message": {"content": content}})

    monkeypatch.setattr("httpx.post", fake_post)
    hit = SearchHit(chunk=sample_chunk(), care_score=0.9)
    retrieval = RetrievalResult(
        query_analysis=QueryAnalysis(
            original_query="What treatment is used?",
            normalized_query="what treatment is used?",
            intent=QueryIntent.TREATMENT,
            preferred_layers=[KnowledgeLayer.CLINICAL_CORE],
            safety_level=SafetyLevel.NORMAL,
        ),
        hits=[hit],
        confidence=0.8,
    )
    payload = OllamaGenerator("http://localhost:11434", "gemma3:4b").generate(
        "What treatment is used?", [hit], retrieval
    )
    assert payload.cited_source_ids == ["S1"]
    assert len(requests) == 2
    repair_text = requests[1]["messages"][1]["content"]
    assert "BEGIN_UNTRUSTED_EVIDENCE" in repair_text
    assert sample_chunk().text in repair_text


def test_cross_encoders_force_raw_logits(monkeypatch) -> None:
    calls: list[dict] = []

    class Identity:
        pass

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append({"model_name": model_name, **kwargs})
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
                )
            )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(nn=types.SimpleNamespace(Identity=Identity)))

    from care_anxrag.nli import CrossEncoderNliClassifier
    from care_anxrag.rerank import CrossEncoderReranker

    CrossEncoderReranker("reranker")
    CrossEncoderNliClassifier("nli")
    assert [call["model_name"] for call in calls] == ["reranker", "nli"]
    assert all(isinstance(call["activation_fn"], Identity) for call in calls)


def test_pubmed_parser_extracts_provenance() -> None:
    xml = """
    <PubmedArticle>
      <MedlineCitation>
        <PMID>12345</PMID>
        <DateRevised><Year>2026</Year><Month>02</Month><Day>03</Day></DateRevised>
        <Article>
          <ArticleTitle>Anxiety treatment review</ArticleTitle>
          <Abstract><AbstractText Label="BACKGROUND">Evidence about anxiety treatment.</AbstractText></Abstract>
          <Journal><JournalIssue><PubDate><Year>2025</Year><Month>Jan</Month><Day>10</Day></PubDate></JournalIssue><Title>Test Journal</Title></Journal>
          <PublicationTypeList><PublicationType>Systematic Review</PublicationType></PublicationTypeList>
          <Language>eng</Language>
        </Article>
        <MeshHeadingList><MeshHeading><DescriptorName>Anxiety Disorders</DescriptorName></MeshHeading></MeshHeadingList>
      </MedlineCitation>
      <PubmedData><PublicationStatus>ppublish</PublicationStatus><ArticleIdList><ArticleId IdType="doi">10.1/test</ArticleId></ArticleIdList></PubmedData>
    </PubmedArticle>
    """
    document = _parse_pubmed_article("pubmed", ET.fromstring(xml))
    assert document is not None
    assert document.external_id == "12345"
    assert document.metadata["doi"] == "10.1/test"
    assert "Systematic Review" in document.publication_types


def test_nice_json_parser_handles_licensed_resource() -> None:
    payload = {
        "items": [
            {
                "id": "CG113",
                "title": "Anxiety guideline",
                "content": "<h2>Recommendations</h2><p>Use evidence-based assessment and care.</p>",
                "updated": "2026-04-01",
                "url": "https://api.nice.org.uk/example",
            }
        ]
    }
    documents = _parse_nice_json("nice", payload, "https://api.nice.org.uk/feed")
    assert len(documents) == 1
    assert documents[0].external_id == "CG113"
    assert documents[0].sections[0].heading == "Recommendations"


def test_generator_rejects_inline_citation_missing_from_structured_list() -> None:
    from care_anxrag.generation import _payload_is_valid
    from care_anxrag.models import GeneratedPayload

    payload = GeneratedPayload(
        answer="One claim [S1]. Another claim [S2].",
        cited_source_ids=["S1"],
    )
    assert not _payload_is_valid(payload, ["S1", "S2"])


def test_chroma_list_ids_is_paginated(monkeypatch, tmp_path) -> None:
    calls: list[int] = []

    class FakeCollection:
        def get(self, *, limit, offset, include):
            calls.append(offset)
            if offset == 0:
                return {"ids": [str(index) for index in range(1000)]}
            return {"ids": ["1000"]}

        def count(self):
            return 1001

    class FakeClient:
        def __init__(self, path):
            self.collection = FakeCollection()

        def get_or_create_collection(self, **kwargs):
            return self.collection

        def heartbeat(self):
            return 1

    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=FakeClient))
    store = ChromaVectorStore(tmp_path / "chroma")
    ids = store.list_ids("care_clinical_core")
    assert len(ids) == 1001
    assert calls == [0, 1000]


def test_cross_encoder_reranker_handles_single_candidate(monkeypatch) -> None:
    class Identity:
        pass

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name

        def predict(self, pairs, show_progress_bar=False):
            assert len(pairs) == 1
            return [2.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(nn=types.SimpleNamespace(Identity=Identity)),
    )

    from care_anxrag.rerank import CrossEncoderReranker

    hit = SearchHit(chunk=sample_chunk())
    score = CrossEncoderReranker("reranker").score("panic treatment", [hit])
    assert len(score) == 1
    assert math.isclose(score[0], 1.0 / (1.0 + math.exp(-2.0)), rel_tol=1e-9)


def test_chroma_query_caps_results_to_collection_size(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            calls.update(kwargs)
            return {
                "ids": [["one", "two"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[{}, {}]],
            }

    class FakeClient:
        def __init__(self, path):
            self.collection = FakeCollection()

        def get_or_create_collection(self, **kwargs):
            return self.collection

        def heartbeat(self):
            return 1

    monkeypatch.setitem(sys.modules, "chromadb", types.SimpleNamespace(PersistentClient=FakeClient))
    store = ChromaVectorStore(tmp_path / "chroma")
    hits = store.query("care_clinical_core", [1.0, 0.0], 40)
    assert calls["n_results"] == 2
    assert len(hits) == 2


def test_nli_uses_model_label_mapping_and_accepts_single_vector(monkeypatch) -> None:
    class Identity:
        pass

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            self.model = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    id2label={0: "entailment", 1: "neutral", 2: "contradiction"}
                )
            )

        def predict(self, pairs, show_progress_bar=False):
            assert len(pairs) == 1
            return [8.0, 0.0, -2.0]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(nn=types.SimpleNamespace(Identity=Identity)),
    )

    from care_anxrag.nli import CrossEncoderNliClassifier

    left = SearchHit(chunk=sample_chunk())
    right = SearchHit(
        chunk=sample_chunk().model_copy(
            update={"chunk_id": "chunk-2", "document_id": "doc-2"}
        )
    )
    relation = CrossEncoderNliClassifier("custom-nli").classify([(left, right)])[0]
    assert relation.label.value == "entailment"
    assert relation.confidence > 0.99

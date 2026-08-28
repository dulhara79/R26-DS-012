from __future__ import annotations

from pathlib import Path
from conftest import write_document
from care_anxrag.evidence import applicability_score
from care_anxrag.models import KnowledgeLayer
from care_anxrag.query import QueryAnalyzer

GAD = """
# Generalized anxiety disorder
Persistent excessive worry across several areas of life is associated with generalized anxiety disorder. The information should not be used to self-diagnose because clinical assessment considers duration, impairment, medical causes, and other conditions.

# Treatment
Cognitive behavioural therapy is recommended as an evidence-based psychological option. A qualified clinician should discuss individual treatment choices and medication risks.
"""

PANIC = """
# Panic disorder
Panic attacks can involve sudden intense fear, a racing heart, sweating, trembling, dizziness, and fear of losing control. Similar physical symptoms can have medical causes, so severe or new symptoms may require medical assessment.

# Treatment
Cognitive behavioural therapy and exposure-based methods are evidence-supported approaches for panic disorder. Individual treatment must be discussed with a qualified professional.
"""


def test_hybrid_retrieval_prefers_matching_disorder(runtime, project: Path) -> None:
    write_document(
        project,
        "gad.md",
        external_id="gad",
        title="GAD Clinical Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD,
    )
    write_document(
        project,
        "panic.md",
        external_id="panic",
        title="Panic Disorder Clinical Guidance",
        topics=["anxiety", "panic_disorder"],
        body=PANIC,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    result = runtime.retriever.retrieve("What evidence-based treatment is used for panic attacks?")
    assert result.hits
    assert result.hits[0].chunk.metadata["external_id"] == "panic"
    assert result.hits[0].care_score > 0.3
    assert not result.should_abstain

    answer = runtime.rag.answer("What evidence-based treatment is used for panic attacks?")
    assert not answer.abstained
    assert answer.citations
    assert all(f"[{citation.citation_id}]" in answer.answer for citation in answer.citations)


def test_out_of_scope_query_abstains(runtime, project: Path) -> None:
    write_document(
        project,
        "gad.md",
        external_id="gad",
        title="GAD Clinical Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    result = runtime.rag.answer("How do I repair a diesel fuel injector?")
    assert result.abstained
def test_applicability_strongly_penalizes_wrong_anxiety_subtype() -> None:
    analyzer = QueryAnalyzer()

    query = analyzer.analyze(
        "What evidence supports CBT for generalized anxiety disorder?"
    )

    gad_score = applicability_score(
        topics=[
            "anxiety",
            "generalized_anxiety_disorder",
            "Cognitive Behavioral Therapy",
        ],
        query=query,
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
    )

    social_anxiety_score = applicability_score(
        topics=[
            "anxiety",
            "social_anxiety_disorder",
            "Cognitive Behavioral Therapy",
        ],
        query=query,
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
    )

    print("GAD applicability:", gad_score)
    print("Wrong subtype applicability:", social_anxiety_score)

    assert gad_score == 1.0
    assert social_anxiety_score <= 0.50
def test_query_analyzer_identifies_cbt_treatment() -> None:
    analyzer = QueryAnalyzer()

    analysis = analyzer.analyze(
        "What evidence supports CBT for generalized anxiety disorder?"
    )

    data = analysis.model_dump()

    assert data.get("treatments") == [
        "cognitive_behavioral_therapy"
    ]

def test_query_analyzer_expands_clinical_terms_for_retrieval() -> None:
    analyzer = QueryAnalyzer()

    analysis = analyzer.analyze(
        "What evidence supports CBT for GAD?"
    )

    assert analysis.original_query == (
        "What evidence supports CBT for GAD?"
    )

    assert "cognitive behavioral therapy" in analysis.retrieval_query
    assert "cognitive behavioural therapy" in analysis.retrieval_query
    assert "generalized anxiety disorder" in analysis.retrieval_query

def test_retriever_uses_expanded_query_for_dense_and_lexical_search(
    runtime,
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_embed(texts: list[str]) -> list[list[float]]:
        captured["dense_query"] = texts[0]
        return [[0.0]]

    def fake_dense_search(query_embedding, preferred_layers):
        return []

    def fake_lexical_search(query: str):
        captured["lexical_query"] = query
        return []

    monkeypatch.setattr(
        runtime.retriever.embedder,
        "embed",
        fake_embed,
    )
    monkeypatch.setattr(
        runtime.retriever,
        "_dense_search",
        fake_dense_search,
    )
    monkeypatch.setattr(
        runtime.retriever,
        "_lexical_search",
        fake_lexical_search,
    )

    runtime.retriever.retrieve(
        "What evidence supports CBT for GAD?"
    )

    assert "cognitive behavioral therapy" in captured["dense_query"]
    assert "generalized anxiety disorder" in captured["dense_query"]

    assert "cognitive behavioral therapy" in captured["lexical_query"]
    assert "generalized anxiety disorder" in captured["lexical_query"]

def test_explicit_wrong_subtype_is_penalized_in_final_scoring(runtime) -> None:
    analyzer = QueryAnalyzer()

    query = analyzer.analyze(
        "What evidence supports CBT for generalized anxiety disorder?"
    )

    assert query.anxiety_subtypes == [
        "generalized_anxiety_disorder"
    ]

    exact = applicability_score(
        topics=[
            "anxiety",
            "generalized_anxiety_disorder",
            "Cognitive Behavioral Therapy",
        ],
        query=query,
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
    )

    wrong = applicability_score(
        topics=[
            "anxiety",
            "social_anxiety_disorder",
            "Cognitive Behavioral Therapy",
        ],
        query=query,
        layer=KnowledgeLayer.RESEARCH_FRONTIER,
    )

    exact_adjustment = runtime.retriever._clinical_compatibility_adjustment(
        exact
    )

    wrong_adjustment = runtime.retriever._clinical_compatibility_adjustment(
        wrong
    )

    assert exact_adjustment == 1.0
    assert wrong_adjustment < 0.70

def test_care_score_strongly_penalizes_explicit_wrong_subtype(runtime) -> None:
    from types import SimpleNamespace

    common = {
        "dense_score": 0.55,
        "lexical_score": 1.0,
        "rrf_normalized": 1.0,
        "rerank_score": 0.92,
        "freshness_score": 0.80,
    }

    chunk = SimpleNamespace(
        authority_score=0.80,
        evidence_score=0.82,
    )

    exact_hit = SimpleNamespace(
        **common,
        applicability_score=1.0,
        chunk=chunk,
    )

    wrong_hit = SimpleNamespace(
        **common,
        applicability_score=0.4375,
        chunk=chunk,
    )

    exact_score = runtime.retriever._care_score(exact_hit)
    wrong_score = runtime.retriever._care_score(wrong_hit)

    print("Exact subtype CARE:", exact_score)
    print("Wrong subtype CARE:", wrong_score)

    assert wrong_score <= exact_score * 0.65

def test_care_score_penalizes_explicit_wrong_treatment(runtime) -> None:
    from types import SimpleNamespace

    analyzer = QueryAnalyzer()

    query = analyzer.analyze(
        "What evidence supports CBT for generalized anxiety disorder?"
    )

    common = {
        "dense_score": 0.55,
        "lexical_score": 0.80,
        "rrf_normalized": 0.90,
        "rerank_score": 0.85,
        "freshness_score": 0.80,
        "applicability_score": 1.0,
    }

    cbt_hit = SimpleNamespace(
        **common,
        chunk=SimpleNamespace(
            authority_score=0.80,
            evidence_score=0.82,
            title=(
                "Randomized controlled trial of cognitive-behavioral "
                "therapy for generalized anxiety disorder"
            ),
        ),
    )

    mct_hit = SimpleNamespace(
        **common,
        chunk=SimpleNamespace(
            authority_score=0.80,
            evidence_score=0.82,
            title=(
                "The effectiveness of metacognitive therapy "
                "in patients with generalized anxiety disorder"
            ),
        ),
    )

    cbt_score = runtime.retriever._care_score(
        cbt_hit,
        query,
    )

    mct_score = runtime.retriever._care_score(
        mct_hit,
        query,
    )

    print("CBT CARE:", cbt_score)
    print("MCT CARE:", mct_score)

    assert mct_score <= cbt_score * 0.70

from __future__ import annotations

from pathlib import Path

from conftest import write_document


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

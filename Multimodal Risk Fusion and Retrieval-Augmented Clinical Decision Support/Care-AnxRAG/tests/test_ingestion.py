from __future__ import annotations

import time
from pathlib import Path

from care_anxrag.models import DocumentStatus, KnowledgeLayer

from conftest import write_document


GAD_BODY = """
# Generalized anxiety disorder
Generalized anxiety disorder involves persistent and excessive worry that is difficult to control and can interfere with daily functioning. Assessment considers symptom duration, severity, impairment, physical symptoms, and alternative explanations.

# Evidence-based care
Cognitive behavioural therapy is an evidence-based psychological treatment. Treatment choices should reflect severity, preferences, prior response, accessibility, and clinical assessment. Medication decisions require an appropriately qualified prescriber and should not be changed solely because of an automated answer.
"""


def test_incremental_ingestion_is_idempotent_and_versions(runtime, project: Path) -> None:
    path = write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    first = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert first.promoted == 1
    assert runtime.database.count_chunks(DocumentStatus.ACTIVE) >= 1
    assert runtime.database.pending_outbox() == []

    second = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert second.unchanged == 1
    assert second.promoted == 0

    time.sleep(0.01)
    path.write_text(path.read_text(encoding="utf-8") + "\nAn updated section adds monitoring and follow-up.\n", encoding="utf-8")
    third = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert third.promoted == 1

    with runtime.database.connect() as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM document_versions GROUP BY status"
        ).fetchall()
    counts = {row["status"]: row["count"] for row in rows}
    assert counts["active"] == 1
    assert counts["superseded"] == 1
    assert runtime.database.integrity_check() == "ok"


def test_dry_run_does_not_ingest_or_advance_source_state(runtime, project: Path) -> None:
    write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    before = runtime.database.get_source_state("test_core")
    summary = runtime.ingestion.sync(source_ids=["test_core"], dry_run=True, force=True)
    after = runtime.database.get_source_state("test_core")
    assert summary.staged == 1
    assert runtime.database.count_chunks(status=None) == 0
    assert before.last_attempt_at is None and after.last_attempt_at is None
    assert before.last_success_at is None and after.last_success_at is None
    assert runtime.database.last_successful_sync_at() is None
    with runtime.database.connect() as connection:
        status = connection.execute(
            "SELECT status FROM sync_runs WHERE run_id=?", (summary.run_id,)
        ).fetchone()["status"]
    assert status == "dry_run"


def test_reconcile_reindexes_active_chunks(runtime, project: Path) -> None:
    write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    result = runtime.ingestion.reconcile_active_vectors(batch_size=2)
    assert result["active_chunks"] == result["indexed"]
    assert runtime.vector_store.count(runtime.settings.clinical_collection) >= result["active_chunks"]


def test_superseded_vectors_are_physically_removed(runtime, project: Path) -> None:
    path = write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    first_ids = runtime.vector_store.list_ids(runtime.settings.clinical_collection)
    assert first_ids

    path.write_text(
        path.read_text(encoding="utf-8") + "\nA materially updated follow-up recommendation.\n",
        encoding="utf-8",
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    second_ids = runtime.vector_store.list_ids(runtime.settings.clinical_collection)
    assert second_ids
    assert first_ids.isdisjoint(second_ids)
    assert second_ids == {
        chunk.chunk_id for chunk in runtime.database.list_chunks(DocumentStatus.ACTIVE)
    }


def test_reconcile_deletes_orphan_vector(runtime, project: Path) -> None:
    write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    active_chunk = runtime.database.list_chunks(DocumentStatus.ACTIVE)[0]
    orphan = active_chunk.model_copy(
        update={
            "chunk_id": "orphan-vector",
            "document_id": "orphan-document",
            "version_id": "orphan-version",
        }
    )
    runtime.vector_store.upsert(
        runtime.settings.clinical_collection,
        [orphan],
        [runtime.embedder.embed([orphan.text])[0]],
    )
    result = runtime.ingestion.reconcile_active_vectors()
    assert result["deleted_stale"] == 1
    assert "orphan-vector" not in runtime.vector_store.list_ids(
        runtime.settings.clinical_collection
    )


def test_rollback_reactivates_and_reindexes_superseded_version(runtime, project: Path) -> None:
    path = write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    first = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    first_version_id = runtime.database.list_chunks(DocumentStatus.ACTIVE)[0].version_id
    first_chunk_ids = {
        chunk.chunk_id for chunk in runtime.database.list_chunks_for_version(first_version_id)
    }
    assert first.promoted == 1

    path.write_text(
        path.read_text(encoding="utf-8") + "\nA second-version follow-up recommendation.\n",
        encoding="utf-8",
    )
    second = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    second_version_id = runtime.database.list_chunks(DocumentStatus.ACTIVE)[0].version_id
    assert second.promoted == 1
    assert second_version_id != first_version_id

    runtime.ingestion.approve(first_version_id)
    assert runtime.database.get_version(first_version_id).status == DocumentStatus.ACTIVE
    assert runtime.database.get_version(second_version_id).status == DocumentStatus.SUPERSEDED
    assert runtime.vector_store.list_ids(runtime.settings.clinical_collection) == first_chunk_ids


def test_rejecting_staged_version_physically_removes_vectors(runtime, project: Path) -> None:
    runtime.ingestion.sources_by_id["test_core"].auto_promote = False
    write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    staging = runtime.database.list_staging_versions()
    assert summary.staged == 1
    assert len(staging) == 1
    assert runtime.vector_store.list_ids(runtime.settings.clinical_collection)

    runtime.ingestion.reject(staging[0].version_id, "review_failed")
    assert runtime.database.get_version(staging[0].version_id).status == DocumentStatus.REJECTED
    assert runtime.vector_store.list_ids(runtime.settings.clinical_collection) == set()


def test_research_frontier_requires_global_auto_promote_gate(runtime, project: Path) -> None:
    source = runtime.ingestion.sources_by_id["test_core"]
    source.layer = KnowledgeLayer.RESEARCH_FRONTIER
    source.auto_promote = True
    write_document(
        project,
        "trial.md",
        external_id="trial-anxiety",
        title="Recent Anxiety Trial",
        topics=["anxiety", "recent_research"],
        publication_types=["Randomized Controlled Trial"],
        body=GAD_BODY
        + "\nThis synthetic randomized controlled trial is included only to validate the "
        "research-frontier publication gate. It must remain staged until explicitly reviewed.",
    )
    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert summary.staged == 1
    assert summary.promoted == 0
    assert runtime.database.count_chunks(DocumentStatus.ACTIVE) == 0
    assert runtime.vector_store.list_ids(runtime.settings.research_collection)


def test_local_connector_accepts_scalar_metadata_and_pattern(runtime, project: Path) -> None:
    runtime.ingestion.sources_by_id["test_core"].settings["patterns"] = "**/*.md"
    path = project / "docs" / "scalar.md"
    path.write_text(
        "---\n"
        "external_id: scalar-guideline\n"
        "title: Scalar Metadata Guideline\n"
        "topics: anxiety\n"
        "publication_types: Clinical Guideline\n"
        "authors: Example Author\n"
        "updated_at: 2026-01-01\n"
        "---\n\n"
        + GAD_BODY,
        encoding="utf-8",
    )

    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert summary.promoted == 1
    active = runtime.database.list_chunks(DocumentStatus.ACTIVE)[0]
    version = runtime.database.get_version(active.version_id)
    assert version is not None
    assert version.topics == ["anxiety"]
    assert version.publication_types == ["Clinical Guideline"]
    assert version.metadata["authors"] == ["Example Author"]


def test_embedding_identity_mismatch_requires_explicit_rebuild(runtime, project: Path) -> None:
    from care_anxrag.embeddings import CachedEmbedder, HashEmbedder

    write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert runtime.database.get_embedding_identity() == "hash:256:v2"

    replacement = HashEmbedder(128)
    runtime.ingestion.embedder = CachedEmbedder(replacement, runtime.database)
    runtime.retriever.embedder = replacement

    try:
        runtime.retriever.retrieve("What care is discussed for generalized anxiety disorder?")
    except RuntimeError as exc:
        assert "Embedding model/index mismatch" in str(exc)
    else:
        raise AssertionError("Expected retrieval to reject an embedding/index mismatch")

    result = runtime.ingestion.reconcile_active_vectors(reset_embedding_identity=True)
    assert result["identity_reset"] is True
    assert result["embedding_identity"] == "hash:128:v2"
    assert runtime.database.get_embedding_identity() == "hash:128:v2"
    assert runtime.retriever.retrieve(
        "What care is discussed for generalized anxiety disorder?"
    ).hits


def test_manual_withdrawal_removes_active_evidence_and_vectors(runtime, project: Path) -> None:
    write_document(
        project,
        "gad.md",
        external_id="guideline-gad",
        title="Generalized Anxiety Disorder Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=GAD_BODY,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    active_chunk = runtime.database.list_chunks(DocumentStatus.ACTIVE)[0]

    withdrawn_version_id = runtime.ingestion.withdraw(
        active_chunk.version_id,
        "source_withdrawn_by_publisher",
    )

    assert withdrawn_version_id == active_chunk.version_id
    version = runtime.database.get_version(active_chunk.version_id)
    assert version is not None
    assert version.status == DocumentStatus.WITHDRAWN
    assert version.rejection_reason == "source_withdrawn_by_publisher"
    assert runtime.database.count_chunks(DocumentStatus.ACTIVE) == 0
    assert runtime.vector_store.list_ids(runtime.settings.clinical_collection) == set()
    result = runtime.rag.answer("What care is discussed for generalized anxiety disorder?")
    assert result.abstained is True



def test_pubmed_retraction_relation_withdraws_active_target(runtime) -> None:
    from care_anxrag.models import EvidenceLevel, RawDocument, Section
    from care_anxrag.util import utc_now

    source = runtime.ingestion.sources_by_id["test_core"]

    original = RawDocument(
        source_id=source.id,
        external_id="80001",
        title="Anxiety trial",
        text=GAD_BODY,
        retrieved_at=utc_now(),
        publication_types=["Randomized Controlled Trial"],
        topics=["anxiety", "generalized_anxiety_disorder"],
        sections=[
            Section(
                path="abstract",
                heading="Abstract",
                text=GAD_BODY,
                ordinal=0,
            )
        ],
        metadata={
            "pmid": "80001",
            "publication_status": "ppublish",
        },
    )
    staged = runtime.ingestion.ingest_document(
        source,
        original,
    )
    if runtime.database.get_version(staged.version_id).status == DocumentStatus.STAGING:
        runtime.ingestion.approve(staged.version_id)
    source.connector = "pubmed"

    active = runtime.database.get_version(staged.version_id)
    assert active is not None
    assert active.status == DocumentStatus.ACTIVE
    assert runtime.vector_store.list_ids(
        runtime.settings.clinical_collection
    )

    notice = RawDocument(
        source_id=source.id,
        external_id="90001",
        title="Retraction notice",
        text="Retraction notice for the linked anxiety trial.",
        retrieved_at=utc_now(),
        publication_types=["Retraction Notice"],
        topics=["anxiety"],
        metadata={
            "pmid": "90001",
            "publication_status": "ppublish",
            "pubmed_relations": [
                {
                    "ref_type": "RetractionOf",
                    "pmid": "80001",
                    "ref_source": "Test Journal. 2025;1:1-5.",
                    "note": "",
                }
            ],
            "retraction_of_pmids": ["80001"],
        },
    )

    affected = runtime.ingestion.apply_pubmed_relations(
        source,
        notice,
        dry_run=False,
    )

    assert affected == ["80001"]
    withdrawn = runtime.database.get_version(staged.version_id)
    assert withdrawn is not None
    assert withdrawn.status == DocumentStatus.WITHDRAWN
    assert withdrawn.rejection_reason == "pubmed_retraction_notice:90001"
    assert runtime.vector_store.list_ids(
        runtime.settings.clinical_collection
    ) == set()


def test_pubmed_expression_of_concern_does_not_auto_withdraw(runtime) -> None:
    from care_anxrag.models import EvidenceLevel, RawDocument, Section
    from care_anxrag.util import utc_now

    source = runtime.ingestion.sources_by_id["test_core"]

    original = RawDocument(
        source_id=source.id,
        external_id="81001",
        title="Anxiety trial under review",
        text=GAD_BODY,
        retrieved_at=utc_now(),
        publication_types=["Randomized Controlled Trial"],
        topics=["anxiety"],
        sections=[
            Section(
                path="abstract",
                heading="Abstract",
                text=GAD_BODY,
                ordinal=0,
            )
        ],
        metadata={"pmid": "81001"},
    )
    staged = runtime.ingestion.ingest_document(source, original)
    if runtime.database.get_version(staged.version_id).status == DocumentStatus.STAGING:
        runtime.ingestion.approve(staged.version_id)
    source.connector = "pubmed"

    notice = RawDocument(
        source_id=source.id,
        external_id="91001",
        title="Expression of concern",
        text="Expression of concern for the linked anxiety trial.",
        retrieved_at=utc_now(),
        publication_types=["Expression of Concern"],
        topics=["anxiety"],
        metadata={
            "expression_of_concern_for_pmids": ["81001"],
            "pubmed_relations": [
                {
                    "ref_type": "ExpressionOfConcernFor",
                    "pmid": "81001",
                    "ref_source": "Test Journal.",
                    "note": "",
                }
            ],
        },
    )

    affected = runtime.ingestion.apply_pubmed_relations(
        source,
        notice,
        dry_run=False,
    )

    assert affected == []
    active = runtime.database.get_version(staged.version_id)
    assert active is not None
    assert active.status == DocumentStatus.ACTIVE

    alerts = runtime.database.list_evidence_alerts()
    assert len(alerts) == 1
    assert alerts[0]["relation_type"] == "ExpressionOfConcernFor"
    assert alerts[0]["target_external_id"] == "81001"
    assert alerts[0]["notice_external_id"] == "91001"
    assert alerts[0]["resolved_at"] is None


def test_pubmed_retraction_relation_dry_run_does_not_mutate(runtime) -> None:
    from care_anxrag.models import EvidenceLevel, RawDocument, Section
    from care_anxrag.util import utc_now

    source = runtime.ingestion.sources_by_id["test_core"]

    original = RawDocument(
        source_id=source.id,
        external_id="82001",
        title="Anxiety trial",
        text=GAD_BODY,
        retrieved_at=utc_now(),
        publication_types=["Randomized Controlled Trial"],
        topics=["anxiety"],
        sections=[
            Section(
                path="abstract",
                heading="Abstract",
                text=GAD_BODY,
                ordinal=0,
            )
        ],
        metadata={"pmid": "82001"},
    )
    staged = runtime.ingestion.ingest_document(source, original)
    if runtime.database.get_version(staged.version_id).status == DocumentStatus.STAGING:
        runtime.ingestion.approve(staged.version_id)
    source.connector = "pubmed"

    notice = RawDocument(
        source_id=source.id,
        external_id="92001",
        title="Retraction notice",
        text="Retraction notice for the linked anxiety trial.",
        retrieved_at=utc_now(),
        publication_types=["Retraction Notice"],
        topics=["anxiety"],
        metadata={"retraction_of_pmids": ["82001"]},
    )

    affected = runtime.ingestion.apply_pubmed_relations(
        source,
        notice,
        dry_run=True,
    )

    assert affected == ["82001"]
    active = runtime.database.get_version(staged.version_id)
    assert active is not None
    assert active.status == DocumentStatus.ACTIVE
    assert runtime.database.list_evidence_alerts() == []



def test_ingestion_propagates_clinical_facets_to_chunks(runtime) -> None:
    from care_anxrag.models import RawDocument
    from care_anxrag.util import utc_now

    source = runtime.ingestion.sources_by_id["test_core"]
    raw = RawDocument(
        source_id=source.id,
        external_id="pico-gad-cbt",
        title="CBT for generalized anxiety disorder",
        text=(
            GAD_BODY
            + "\nOlder adults with generalized anxiety disorder received "
            "cognitive behavioural therapy."
        ),
        retrieved_at=utc_now(),
        publication_types=["Randomized Controlled Trial"],
        topics=["generalized_anxiety_disorder"],
        metadata={
            "pico": {
                "population": ["Adults aged 65 years and older"],
                "intervention": ["Cognitive behavioural therapy"],
                "comparator": ["Treatment as usual"],
                "outcome": ["Anxiety symptom severity"],
            }
        },
    )

    result = runtime.ingestion.ingest_document(source, raw)
    version = runtime.database.get_version(result.version_id)
    assert version is not None

    chunks = runtime.database.list_chunks_for_version(result.version_id)
    assert chunks
    facets = chunks[0].metadata["clinical_facets"]

    assert facets["anxiety_subtypes"] == [
        "generalized_anxiety_disorder"
    ]
    assert "cognitive_behavioral_therapy" in facets["treatments"]
    assert facets["pico"]["comparator"] == ["Treatment as usual"]
    assert facets["pico"]["outcome"] == ["Anxiety symptom severity"]

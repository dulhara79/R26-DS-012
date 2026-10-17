from __future__ import annotations

from care_anxrag.corpus_freeze import build_corpus_freeze


def test_corpus_freeze_is_ready_for_clean_runtime(runtime) -> None:
    report = build_corpus_freeze(runtime, code_revision="abc123")

    assert report["ready_to_freeze"] is True
    assert report["blockers"] == []
    assert report["database_integrity"] == "ok"
    assert report["snapshot"]["code_revision"] == "abc123"


def test_corpus_freeze_blocks_unresolved_staging(runtime, project) -> None:
    from conftest import write_document

    write_document(
        project,
        "staged.md",
        external_id="staged-doc",
        title="Staged evidence",
        topics=["anxiety"],
        body=(
            "This research evidence discusses anxiety symptoms, assessment, "
            "functional impact, psychological care, monitoring, follow-up, "
            "and evidence interpretation in sufficient detail for ingestion "
            "validation. It deliberately contains enough substantive clinical "
            "text to pass the repository quality gate while remaining held for "
            "manual review instead of being promoted into the active corpus. "
            "The fixture describes persistent worry, impairment, assessment, "
            "treatment context, and follow-up without asserting fabricated outcomes."
        ),
    )
    runtime.ingestion.sources_by_id["test_core"].auto_promote = False
    runtime.ingestion.sync(source_ids=["test_core"], force=True)

    report = build_corpus_freeze(runtime, code_revision="abc123")

    assert report["ready_to_freeze"] is False
    assert report["staging_count"] == 1
    assert "unresolved_staging_versions=1" in report["blockers"]


def test_corpus_freeze_blocks_open_evidence_alert(runtime) -> None:
    from conftest import write_document

    write_document(
        runtime.settings.project_root,
        "active.md",
        external_id="target-doc",
        title="Target evidence",
        topics=["anxiety"],
        body=(
            "This active guidance discusses anxiety assessment, persistent "
            "symptoms, functional impairment, evidence-based psychological care, "
            "monitoring, follow-up, and individualized interpretation in enough "
            "detail to pass ingestion validation and become active evidence. "
            "It also describes the importance of evaluating symptom burden, "
            "functioning, treatment context, and change over time using qualified "
            "clinical judgment rather than relying on a single isolated signal."
        ),
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    active = runtime.database.list_active_version_fingerprints()[0]
    runtime.database.record_evidence_alert(
        source_id=active["source_id"],
        target_document_id=active["document_id"],
        target_external_id=active["external_id"],
        notice_external_id="notice-1",
        relation_type="correction",
    )

    report = build_corpus_freeze(runtime, code_revision="abc123")

    assert report["ready_to_freeze"] is False
    assert report["open_evidence_alert_count"] == 1
    assert "open_evidence_alerts=1" in report["blockers"]

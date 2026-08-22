from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from care_anxrag.models import RawDocument
from care_anxrag.sources.base import FetchResult


def test_failed_document_does_not_advance_incremental_cursor(
    runtime, monkeypatch, project: Path
) -> None:
    document = RawDocument(
        source_id="test_core",
        external_id="will-fail",
        title="Synthetic anxiety evidence",
        text=("Anxiety evidence and treatment information. " * 20),
        retrieved_at=datetime.now(tz=UTC),
        metadata={},
    )

    class Connector:
        def fetch(self, source, state, since, until):
            return FetchResult(
                documents=[document],
                changed=True,
                cursor="cursor-that-must-not-commit",
                etag="etag-that-must-not-commit",
            )

    monkeypatch.setattr("care_anxrag.ingestion.build_connector", lambda *args, **kwargs: Connector())
    monkeypatch.setattr(
        runtime.ingestion,
        "ingest_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )

    result = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    state = runtime.database.get_source_state("test_core")
    assert result.failed == 1
    assert state.last_success_at is None
    assert state.cursor is None
    assert state.etag is None
    assert "synthetic failure" in (state.last_error or "")


def test_sync_rejects_empty_enabled_source_selection(runtime) -> None:
    runtime.ingestion.sources_by_id["test_core"].enabled = False
    try:
        runtime.ingestion.sync(source_ids=["test_core"])
    except ValueError as exc:
        assert "No enabled sources selected" in str(exc)
    else:
        raise AssertionError("Expected an empty source selection to be rejected")
    assert runtime.database.last_successful_sync_at() is None


def test_sync_lock_rejects_overlapping_run(runtime) -> None:
    acquired = runtime.ingestion._sync_lock.acquire(blocking=False)
    assert acquired
    try:
        try:
            runtime.ingestion.sync(source_ids=["test_core"])
        except RuntimeError as exc:
            assert "already in progress" in str(exc)
        else:
            raise AssertionError("Expected an overlapping synchronization to be rejected")
    finally:
        runtime.ingestion._sync_lock.release()

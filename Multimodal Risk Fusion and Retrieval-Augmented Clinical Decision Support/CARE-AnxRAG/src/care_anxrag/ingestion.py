from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from .chunking import SectionAwareChunker
from .config import Settings
from .db import Database
from .embeddings import CachedEmbedder
from .evidence import classify_evidence, evidence_score, is_retracted
from .models import (
    ChunkRecord,
    DocumentStatus,
    DocumentVersion,
    KnowledgeLayer,
    RawDocument,
    SourceConfig,
    SourceState,
    SyncSummary,
)
from .registry import source_by_id
from .sources import build_connector
from .util import canonical_json, normalize_whitespace, sha256_text, stable_id, utc_now
from .validation import DocumentValidator
from .vector_store import VectorIndexer, VectorStore, collection_for_layer


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentIngestResult:
    outcome: str
    version_id: str | None = None
    reason: str | None = None


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        sources: Sequence[SourceConfig],
        embedder: CachedEmbedder,
        vector_store: VectorStore,
        chunker: SectionAwareChunker | None = None,
        validator: DocumentValidator | None = None,
    ):
        self.settings = settings
        self.database = database
        self.sources = list(sources)
        self.sources_by_id = source_by_id(self.sources)
        self.embedder = embedder
        self.vector_store = vector_store
        self.vector_indexer = VectorIndexer(database, vector_store)
        self.chunker = chunker or SectionAwareChunker()
        self.validator = validator or DocumentValidator()
        self._sync_lock = threading.Lock()

    def sync(
        self,
        source_ids: Sequence[str] | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncSummary:
        if not self._sync_lock.acquire(blocking=False):
            raise RuntimeError("A synchronization run is already in progress")
        try:
            return self._sync_unlocked(
                source_ids=source_ids,
                dry_run=dry_run,
                force=force,
            )
        finally:
            self._sync_lock.release()

    def _sync_unlocked(
        self,
        source_ids: Sequence[str] | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> SyncSummary:
        selected = self._select_sources(source_ids)
        started_at = utc_now()
        run_id = stable_id("sync", started_at.isoformat(), *(source.id for source in selected), length=24)
        self.database.start_sync_run(run_id, [source.id for source in selected], started_at)
        summary = SyncSummary(
            run_id=run_id,
            started_at=started_at,
            finished_at=started_at,
            source_ids=[source.id for source in selected],
        )
        try:
            for source in selected:
                try:
                    counters = self._sync_source(source, dry_run=dry_run, force=force)
                    summary.discovered += counters.get("discovered", 0)
                    summary.unchanged += counters.get("unchanged", 0)
                    summary.staged += counters.get("staged", 0)
                    summary.promoted += counters.get("promoted", 0)
                    summary.rejected += counters.get("rejected", 0)
                    summary.failed += counters.get("failed", 0)
                    summary.errors.extend(counters.get("errors", []))
                except Exception as exc:
                    summary.failed += 1
                    message = f"{source.id}: {type(exc).__name__}: {exc}"
                    summary.errors.append(message)
                    logger.exception("Source sync failed", extra={"source_id": source.id, "run_id": run_id})
                    if not dry_run:
                        state = self.database.get_source_state(source.id)
                        state.last_attempt_at = utc_now()
                        state.last_error = message
                        self.database.save_source_state(state)
            summary.finished_at = utc_now()
            if dry_run:
                status = "dry_run" if summary.failed == 0 else "dry_run_partial"
            else:
                status = "success" if summary.failed == 0 else "partial"
            self.database.finish_sync_run(
                run_id,
                status,
                summary.finished_at,
                summary.model_dump(mode="json"),
            )
            return summary
        except Exception as exc:
            summary.finished_at = utc_now()
            self.database.finish_sync_run(
                run_id,
                "failed",
                summary.finished_at,
                summary.model_dump(mode="json"),
                str(exc),
            )
            raise

    def _sync_source(
        self,
        source: SourceConfig,
        dry_run: bool,
        force: bool,
    ) -> dict[str, Any]:
        if source.connector != "local_files" and not self.settings.allow_network_sync:
            raise RuntimeError("Network synchronization is disabled by CARE_ALLOW_NETWORK_SYNC")
        state = self.database.get_source_state(source.id)
        state.last_attempt_at = utc_now()
        state.last_error = None
        if not dry_run:
            self.database.save_source_state(state)
        connector = build_connector(
            source,
            self.settings.project_root,
            self.settings.source_user_agent,
            self.settings.request_timeout_seconds,
        )
        until = utc_now()
        since = None if force else state.last_success_at
        fetched = connector.fetch(source, state, since, until)
        counters: dict[str, Any] = {
            "discovered": len(fetched.documents),
            "unchanged": 0,
            "staged": 0,
            "promoted": 0,
            "rejected": 0,
            "failed": 0,
            "errors": list(fetched.warnings),
        }
        for document in fetched.documents:
            try:
                result = self.ingest_document(source, document, dry_run=dry_run)
                if result.outcome in counters:
                    counters[result.outcome] += 1
                elif result.outcome == "withdrawn":
                    counters["rejected"] += 1
                else:
                    counters["failed"] += 1
            except Exception as exc:
                counters["failed"] += 1
                counters["errors"].append(
                    f"{source.id}/{document.external_id}: {type(exc).__name__}: {exc}"
                )
                logger.exception(
                    "Document ingestion failed",
                    extra={"source_id": source.id, "document_id": document.external_id},
                )

        if not dry_run:
            if counters["failed"] == 0:
                state.last_success_at = until
                if fetched.changed:
                    state.last_changed_at = until
                state.etag = fetched.etag or state.etag
                state.last_modified = fetched.last_modified or state.last_modified
                state.cursor = fetched.cursor or state.cursor
                state.last_error = None
            else:
                # Preserve the previous successful cursor/validators. Advancing them after a
                # partial ingestion failure can permanently skip records on the next poll.
                state.last_error = "; ".join(counters["errors"])[-2000:] or (
                    f"{counters['failed']} document(s) failed ingestion"
                )
            self.database.save_source_state(state)
        return counters

    def ingest_document(
        self,
        source: SourceConfig,
        raw: RawDocument,
        dry_run: bool = False,
    ) -> DocumentIngestResult:
        cleaned_text = normalize_whitespace(raw.text)
        cleaned_title = normalize_whitespace(raw.title)
        document_id = stable_id(source.id, raw.external_id)
        version_fingerprint = canonical_json(
            {
                "title": cleaned_title,
                "text": cleaned_text,
                "publication_types": sorted(raw.publication_types),
                "publication_status": raw.metadata.get("publication_status"),
                "license": raw.metadata.get("license_text"),
                "updated_at": raw.updated_at.isoformat() if raw.updated_at else None,
            }
        )
        content_hash = sha256_text(version_fingerprint)
        existing = self.database.get_version_by_hash(document_id, content_hash)
        if existing is not None:
            return DocumentIngestResult("unchanged", existing.version_id)

        duplicate = self.database.find_version_by_content_hash(content_hash)
        validation = self.validator.validate(raw, source)
        duplicate_reason = None
        if duplicate is not None and duplicate.document_id != document_id:
            duplicate_reason = f"duplicate_content_of:{duplicate.document_id}"

        level = classify_evidence(source.evidence_level, raw.publication_types, cleaned_title)
        active = self.database.get_active_version(document_id)
        version_id = stable_id(document_id, content_hash)
        status = DocumentStatus.STAGING
        rejection_reasons = list(validation.reasons)
        if duplicate_reason:
            rejection_reasons.append(duplicate_reason)
        if is_retracted(raw.publication_types, raw.metadata):
            status = DocumentStatus.WITHDRAWN
        elif rejection_reasons:
            status = DocumentStatus.REJECTED

        version = DocumentVersion(
            document_id=document_id,
            version_id=version_id,
            source_id=source.id,
            external_id=raw.external_id,
            title=cleaned_title,
            text=cleaned_text,
            url=raw.url,
            published_at=raw.published_at,
            updated_at=raw.updated_at,
            retrieved_at=raw.retrieved_at,
            content_hash=content_hash,
            status=status,
            layer=source.layer,
            evidence_level=level,
            authority_score=source.authority_score,
            evidence_score=evidence_score(level),
            topics=sorted(set(raw.topics or ["anxiety"])),
            publication_types=raw.publication_types,
            supersedes_version_id=active.version_id if active else None,
            rejection_reason=",".join(rejection_reasons) or None,
            metadata={
                **raw.metadata,
                "authors": raw.authors,
                "language": raw.language,
                "validation_relevance_score": validation.relevance_score,
                "publish_to_rag": source.publish_to_rag,
            },
        )
        sections = self.chunker.sections(cleaned_title, cleaned_text, raw.sections)
        previous_sections = self.database.list_sections(active.version_id) if active else []
        previous_chunks = self.database.list_chunks_for_version(active.version_id) if active else []
        chunks: list[ChunkRecord] = []
        if source.publish_to_rag and status == DocumentStatus.STAGING:
            chunks = self.chunker.build_chunks(
                version,
                source,
                sections,
                previous_sections=previous_sections,
                previous_chunks=previous_chunks,
            )
        if dry_run:
            if status == DocumentStatus.REJECTED:
                return DocumentIngestResult("rejected", version_id, version.rejection_reason)
            if status == DocumentStatus.WITHDRAWN:
                return DocumentIngestResult("withdrawn", version_id, "retracted_or_withdrawn")
            return DocumentIngestResult("staged", version_id)

        if chunks:
            self.database.ensure_embedding_identity(self.embedder.model_id)
        self.database.insert_version(version, sections, chunks)
        if status == DocumentStatus.WITHDRAWN:
            old_active_chunks = (
                self.database.list_chunks_for_version(active.version_id) if active else []
            )
            self.database.withdraw_document(document_id, version_id)
            self._delete_vectors(old_active_chunks)
            return DocumentIngestResult("withdrawn", version_id, "retracted_or_withdrawn")
        if status == DocumentStatus.REJECTED:
            return DocumentIngestResult("rejected", version_id, version.rejection_reason)
        if not source.publish_to_rag:
            return DocumentIngestResult("staged", version_id, "monitor_only")

        self._index_chunks(chunks)
        if self.database.outbox_has_pending_for_version(version_id):
            return DocumentIngestResult("failed", version_id, "vector_index_not_committed")
        auto_promote = source.auto_promote and (
            source.layer != KnowledgeLayer.RESEARCH_FRONTIER
            or self.settings.auto_promote_research
        )
        if auto_promote:
            self._activate_version(version_id, ensure_indexed=False)
            return DocumentIngestResult("promoted", version_id)
        return DocumentIngestResult("staged", version_id)

    def approve(self, version_id: str) -> None:
        version = self.database.get_version(version_id)
        if version is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        if version.status not in {DocumentStatus.STAGING, DocumentStatus.SUPERSEDED}:
            raise ValueError(f"Version {version_id} cannot be approved from status {version.status}")
        chunks = self.database.list_chunks_for_version(version_id)
        if bool(version.metadata.get("publish_to_rag", True)) and not chunks:
            raise RuntimeError("Publishable version has no chunks")
        self._activate_version(version_id, ensure_indexed=True)

    def reject(self, version_id: str, reason: str) -> None:
        version = self.database.get_version(version_id)
        if version is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        if version.status not in {DocumentStatus.STAGING, DocumentStatus.SUPERSEDED}:
            raise ValueError(f"Version {version_id} cannot be rejected from status {version.status}")
        chunks = self.database.list_chunks_for_version(version_id)
        self.database.set_version_status(version_id, DocumentStatus.REJECTED, reason)
        self._delete_vectors(chunks)

    def withdraw(self, version_id: str, reason: str) -> str:
        version = self.database.get_version(version_id)
        if version is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        active = self.database.get_active_version(version.document_id)
        if active is None:
            raise ValueError(f"Document {version.document_id} has no active version to withdraw")
        chunks = self.database.list_chunks_for_version(active.version_id)
        self.database.withdraw_document(active.document_id, reason=reason)
        self._delete_vectors(chunks)
        return active.version_id

    def reconcile_active_vectors(
        self,
        batch_size: int = 128,
        reset_embedding_identity: bool = False,
    ) -> dict[str, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if reset_embedding_identity:
            self.database.clear_vector_outbox()
            for collection in (
                self.settings.clinical_collection,
                self.settings.research_collection,
            ):
                existing_ids = sorted(self.vector_store.list_ids(collection))
                if existing_ids:
                    self.vector_store.delete(collection, existing_ids)
            self.database.set_metadata("vector_embedding_model_id", self.embedder.model_id)
        else:
            self.database.assert_embedding_identity(self.embedder.model_id)
        chunks = self.database.list_chunks(status=DocumentStatus.ACTIVE)
        indexed = 0
        expected_by_collection: dict[str, set[str]] = {
            self.settings.clinical_collection: set(),
            self.settings.research_collection: set(),
        }
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed_with_hashes(
                [chunk.text for chunk in batch],
                [chunk.text_hash for chunk in batch],
            )
            by_collection: dict[str, list[tuple[ChunkRecord, list[float]]]] = {}
            for chunk, vector in zip(batch, vectors, strict=True):
                collection = collection_for_layer(
                    chunk.layer,
                    self.settings.clinical_collection,
                    self.settings.research_collection,
                )
                expected_by_collection[collection].add(chunk.chunk_id)
                by_collection.setdefault(collection, []).append((chunk, vector))
            for collection, values in by_collection.items():
                self.vector_store.upsert(
                    collection,
                    [item[0] for item in values],
                    [item[1] for item in values],
                )
                indexed += len(values)

        deleted = 0
        for collection, expected_ids in expected_by_collection.items():
            stale_ids = sorted(self.vector_store.list_ids(collection) - expected_ids)
            if stale_ids:
                self.vector_store.delete(collection, stale_ids)
                deleted += len(stale_ids)
        return {
            "active_chunks": len(chunks),
            "indexed": indexed,
            "deleted_stale": deleted,
            "embedding_identity": self.embedder.model_id,
            "identity_reset": reset_embedding_identity,
        }

    def _activate_version(self, version_id: str, ensure_indexed: bool) -> None:
        version = self.database.get_version(version_id)
        if version is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        current_active = self.database.get_active_version(version.document_id)
        old_chunks = (
            self.database.list_chunks_for_version(current_active.version_id)
            if current_active and current_active.version_id != version_id
            else []
        )
        if ensure_indexed:
            chunks = self.database.list_chunks_for_version(version_id)
            self._index_chunks(chunks)
        self.vector_indexer.drain(limit=10000)
        if self.database.outbox_has_pending_for_version(version_id):
            raise RuntimeError("Vector indexing is incomplete; version was not promoted")
        self.database.promote_version(version_id)
        # Retrieval already checks authoritative DB status, so a failed cleanup cannot
        # surface superseded evidence. The durable outbox retries the physical delete.
        self._delete_vectors(old_chunks)

    def _delete_vectors(self, chunks: Sequence[ChunkRecord]) -> None:
        for chunk in chunks:
            collection = collection_for_layer(
                chunk.layer,
                self.settings.clinical_collection,
                self.settings.research_collection,
            )
            self.database.enqueue_vector_delete(collection, chunk.chunk_id)
        while chunks:
            result = self.vector_indexer.drain(limit=500)
            if result["processed"] == 0:
                break
            if not self.database.pending_outbox(limit=1):
                break

    def _index_chunks(self, chunks: Sequence[ChunkRecord]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed_with_hashes(
            [chunk.text for chunk in chunks],
            [chunk.text_hash for chunk in chunks],
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            collection = collection_for_layer(
                chunk.layer,
                self.settings.clinical_collection,
                self.settings.research_collection,
            )
            self.database.enqueue_vector_upsert(collection, chunk, vector)
        while True:
            result = self.vector_indexer.drain(limit=500)
            if result["processed"] == 0:
                break

    def _select_sources(self, source_ids: Sequence[str] | None) -> list[SourceConfig]:
        if source_ids:
            missing = sorted(set(source_ids) - set(self.sources_by_id))
            if missing:
                raise KeyError(f"Unknown source IDs: {', '.join(missing)}")
            selected = [self.sources_by_id[source_id] for source_id in source_ids]
        else:
            selected = [source for source in self.sources if source.enabled]
        enabled = [source for source in selected if source.enabled]
        if not enabled:
            raise ValueError("No enabled sources selected")
        return enabled

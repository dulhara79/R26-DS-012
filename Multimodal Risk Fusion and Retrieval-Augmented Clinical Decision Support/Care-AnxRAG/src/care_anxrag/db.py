from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .models import (
    ChunkRecord,
    DocumentStatus,
    DocumentVersion,
    EvidenceLevel,
    KnowledgeLayer,
    Section,
    SourceConfig,
    SourceState,
)
from .util import canonical_json, parse_datetime, utc_now


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    connector TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    publish_to_rag INTEGER NOT NULL,
    auto_promote INTEGER NOT NULL,
    layer TEXT NOT NULL,
    authority_score REAL NOT NULL,
    evidence_level TEXT NOT NULL,
    check_interval_minutes INTEGER NOT NULL,
    settings_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_state (
    source_id TEXT PRIMARY KEY REFERENCES sources(source_id) ON DELETE CASCADE,
    last_attempt_at TEXT,
    last_success_at TEXT,
    last_changed_at TEXT,
    etag TEXT,
    last_modified TEXT,
    cursor TEXT,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS sync_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    source_ids_json TEXT NOT NULL,
    summary_json TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    external_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, external_id)
);

CREATE TABLE IF NOT EXISTS document_versions (
    version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT,
    published_at TEXT,
    updated_at TEXT,
    retrieved_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    layer TEXT NOT NULL,
    evidence_level TEXT NOT NULL,
    authority_score REAL NOT NULL,
    evidence_score REAL NOT NULL,
    topics_json TEXT NOT NULL,
    publication_types_json TEXT NOT NULL,
    supersedes_version_id TEXT,
    rejection_reason TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(document_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_versions_document_status
ON document_versions(document_id, status);

CREATE INDEX IF NOT EXISTS idx_versions_source_status
ON document_versions(source_id, status);

CREATE TABLE IF NOT EXISTS sections (
    version_id TEXT NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    section_path TEXT NOT NULL,
    heading TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY(version_id, section_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    version_id TEXT NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    source_name TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    layer TEXT NOT NULL,
    status TEXT NOT NULL,
    section_path TEXT NOT NULL,
    section_heading TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    published_at TEXT,
    updated_at TEXT,
    retrieved_at TEXT NOT NULL,
    authority_score REAL NOT NULL,
    evidence_level TEXT NOT NULL,
    evidence_score REAL NOT NULL,
    topics_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version_id);
CREATE INDEX IF NOT EXISTS idx_chunks_status_layer ON chunks(status, layer);
CREATE INDEX IF NOT EXISTS idx_chunks_document_status ON chunks(document_id, status);
CREATE INDEX IF NOT EXISTS idx_chunks_text_hash ON chunks(text_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id UNINDEXED,
    title,
    section_heading,
    text,
    topics,
    source_name,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    model_id TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(model_id, text_hash)
);

CREATE TABLE IF NOT EXISTS vector_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_outbox_status ON vector_outbox(status, outbox_id);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO app_metadata(key, value) VALUES('schema_version', '1')"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ping(self) -> bool:
        try:
            with self.connect() as connection:
                value = connection.execute("SELECT 1").fetchone()[0]
            return value == 1
        except sqlite3.Error:
            return False

    def integrity_check(self) -> str:
        with self.connect() as connection:
            return str(connection.execute("PRAGMA integrity_check").fetchone()[0])

    def set_metadata(self, key: str, value: str, connection: sqlite3.Connection | None = None) -> None:
        owns_connection = connection is None
        connection = connection or self.connect()
        try:
            connection.execute(
                "INSERT INTO app_metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            if owns_connection:
                connection.commit()
        finally:
            if owns_connection:
                connection.close()

    def get_metadata(self, key: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_metadata WHERE key=?", (key,)
            ).fetchone()
        return None if row is None else str(row["value"])

    def get_embedding_identity(self) -> str | None:
        return self.get_metadata("vector_embedding_model_id")

    def ensure_embedding_identity(self, model_id: str) -> None:
        stored = self.get_embedding_identity()
        if stored is None:
            if self.count_chunks(status=None) > 0:
                raise RuntimeError(
                    "Indexed chunks exist without a recorded embedding identity. "
                    "Run reconcile with reset_embedding_identity=True before retrieval or indexing."
                )
            self.set_metadata("vector_embedding_model_id", model_id)
            return
        if stored != model_id:
            raise RuntimeError(
                "Embedding model/index mismatch: "
                f"index uses {stored!r}, runtime uses {model_id!r}. "
                "Run reconcile with reset_embedding_identity=True to rebuild all vectors."
            )

    def assert_embedding_identity(self, model_id: str) -> None:
        stored = self.get_embedding_identity()
        if stored is None:
            if self.count_chunks(status=None) == 0:
                return
            raise RuntimeError(
                "Indexed chunks exist without a recorded embedding identity. "
                "Run reconcile with reset_embedding_identity=True before retrieval."
            )
        if stored != model_id:
            raise RuntimeError(
                "Embedding model/index mismatch: "
                f"index uses {stored!r}, runtime uses {model_id!r}. "
                "Run reconcile with reset_embedding_identity=True to rebuild all vectors."
            )

    def upsert_sources(self, sources: Sequence[SourceConfig]) -> None:
        now = utc_now().isoformat()
        with self.transaction() as connection:
            for source in sources:
                connection.execute(
                    """
                    INSERT INTO sources(
                        source_id, name, connector, enabled, publish_to_rag, auto_promote,
                        layer, authority_score, evidence_level, check_interval_minutes,
                        settings_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        name=excluded.name,
                        connector=excluded.connector,
                        enabled=excluded.enabled,
                        publish_to_rag=excluded.publish_to_rag,
                        auto_promote=excluded.auto_promote,
                        layer=excluded.layer,
                        authority_score=excluded.authority_score,
                        evidence_level=excluded.evidence_level,
                        check_interval_minutes=excluded.check_interval_minutes,
                        settings_json=excluded.settings_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        source.id,
                        source.name,
                        source.connector,
                        int(source.enabled),
                        int(source.publish_to_rag),
                        int(source.auto_promote),
                        source.layer.value,
                        source.authority_score,
                        source.evidence_level.value,
                        source.check_interval_minutes,
                        canonical_json(source.settings),
                        now,
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO source_state(source_id) VALUES(?)", (source.id,)
                )

    def list_source_rows(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sources"
        params: tuple[Any, ...] = ()
        if enabled_only:
            sql += " WHERE enabled=1"
        sql += " ORDER BY source_id"
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_source_state(self, source_id: str) -> SourceState:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_state WHERE source_id=?", (source_id,)
            ).fetchone()
        if row is None:
            return SourceState(source_id=source_id)
        return SourceState(
            source_id=source_id,
            last_attempt_at=parse_datetime(row["last_attempt_at"]),
            last_success_at=parse_datetime(row["last_success_at"]),
            last_changed_at=parse_datetime(row["last_changed_at"]),
            etag=row["etag"],
            last_modified=row["last_modified"],
            cursor=row["cursor"],
            last_error=row["last_error"],
        )

    def save_source_state(self, state: SourceState) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO source_state(
                    source_id, last_attempt_at, last_success_at, last_changed_at,
                    etag, last_modified, cursor, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    last_attempt_at=excluded.last_attempt_at,
                    last_success_at=excluded.last_success_at,
                    last_changed_at=excluded.last_changed_at,
                    etag=excluded.etag,
                    last_modified=excluded.last_modified,
                    cursor=excluded.cursor,
                    last_error=excluded.last_error
                """,
                (
                    state.source_id,
                    _iso(state.last_attempt_at),
                    _iso(state.last_success_at),
                    _iso(state.last_changed_at),
                    state.etag,
                    state.last_modified,
                    state.cursor,
                    state.last_error,
                ),
            )

    def start_sync_run(self, run_id: str, source_ids: list[str], started_at: datetime) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO sync_runs(run_id, started_at, status, source_ids_json) VALUES(?, ?, 'running', ?)",
                (run_id, started_at.isoformat(), canonical_json(source_ids)),
            )

    def finish_sync_run(
        self,
        run_id: str,
        status: str,
        finished_at: datetime,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE sync_runs
                SET finished_at=?, status=?, summary_json=?, error=?
                WHERE run_id=?
                """,
                (
                    finished_at.isoformat(),
                    status,
                    None if summary is None else canonical_json(summary),
                    error,
                    run_id,
                ),
            )
            if status == "success":
                self.set_metadata("last_successful_sync_at", finished_at.isoformat(), connection)

    def get_active_version(self, document_id: str) -> DocumentVersion | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id=? AND status='active'
                ORDER BY created_at DESC LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return None if row is None else _row_to_version(row)

    def find_version_by_content_hash(self, content_hash: str) -> DocumentVersion | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_versions WHERE content_hash=? ORDER BY created_at DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
        return None if row is None else _row_to_version(row)

    def get_version_by_hash(self, document_id: str, content_hash: str) -> DocumentVersion | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_versions WHERE document_id=? AND content_hash=?",
                (document_id, content_hash),
            ).fetchone()
        return None if row is None else _row_to_version(row)

    def get_version(self, version_id: str) -> DocumentVersion | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM document_versions WHERE version_id=?", (version_id,)
            ).fetchone()
        return None if row is None else _row_to_version(row)

    def insert_version(
        self,
        version: DocumentVersion,
        sections: Sequence[Section],
        chunks: Sequence[ChunkRecord],
    ) -> None:
        now = utc_now().isoformat()
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO documents(document_id, source_id, external_id, created_at) "
                "VALUES(?, ?, ?, ?)",
                (version.document_id, version.source_id, version.external_id, now),
            )
            connection.execute(
                """
                INSERT INTO document_versions(
                    version_id, document_id, source_id, external_id, title, text, url,
                    published_at, updated_at, retrieved_at, content_hash, status, layer,
                    evidence_level, authority_score, evidence_score, topics_json,
                    publication_types_json, supersedes_version_id, rejection_reason,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.document_id,
                    version.source_id,
                    version.external_id,
                    version.title,
                    version.text,
                    version.url,
                    _iso(version.published_at),
                    _iso(version.updated_at),
                    _iso(version.retrieved_at),
                    version.content_hash,
                    version.status.value,
                    version.layer.value,
                    version.evidence_level.value,
                    version.authority_score,
                    version.evidence_score,
                    canonical_json(version.topics),
                    canonical_json(version.publication_types),
                    version.supersedes_version_id,
                    version.rejection_reason,
                    canonical_json(version.metadata),
                    now,
                ),
            )
            for section in sections:
                connection.execute(
                    """
                    INSERT INTO sections(version_id, section_path, heading, ordinal, text, content_hash)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version.version_id,
                        section.path,
                        section.heading,
                        section.ordinal,
                        section.text,
                        section.content_hash,
                    ),
                )
            self._insert_chunks(connection, chunks)

    def _insert_chunks(
        self, connection: sqlite3.Connection, chunks: Sequence[ChunkRecord]
    ) -> None:
        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO chunks(
                    chunk_id, document_id, version_id, source_id, source_name, title, url,
                    layer, status, section_path, section_heading, ordinal, text, text_hash,
                    published_at, updated_at, retrieved_at, authority_score, evidence_level,
                    evidence_score, topics_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.version_id,
                    chunk.source_id,
                    chunk.source_name,
                    chunk.title,
                    chunk.url,
                    chunk.layer.value,
                    chunk.status.value,
                    chunk.section_path,
                    chunk.section_heading,
                    chunk.ordinal,
                    chunk.text,
                    chunk.text_hash,
                    _iso(chunk.published_at),
                    _iso(chunk.updated_at),
                    _iso(chunk.retrieved_at),
                    chunk.authority_score,
                    chunk.evidence_level.value,
                    chunk.evidence_score,
                    canonical_json(chunk.topics),
                    canonical_json(chunk.metadata),
                ),
            )
            connection.execute(
                "INSERT INTO fts_chunks(chunk_id, title, section_heading, text, topics, source_name) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.title,
                    chunk.section_heading,
                    chunk.text,
                    " ".join(chunk.topics),
                    chunk.source_name,
                ),
            )

    def list_sections(self, version_id: str) -> list[Section]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sections WHERE version_id=? ORDER BY ordinal", (version_id,)
            ).fetchall()
        return [
            Section(
                path=row["section_path"],
                heading=row["heading"],
                ordinal=row["ordinal"],
                text=row["text"],
                content_hash=row["content_hash"],
            )
            for row in rows
        ]

    def list_chunks_for_version(self, version_id: str) -> list[ChunkRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chunks WHERE version_id=? ORDER BY section_path, ordinal",
                (version_id,),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def set_version_status(
        self,
        version_id: str,
        status: DocumentStatus,
        rejection_reason: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE document_versions SET status=?, rejection_reason=? WHERE version_id=?",
                (status.value, rejection_reason, version_id),
            )
            connection.execute(
                "UPDATE chunks SET status=? WHERE version_id=?",
                (status.value, version_id),
            )

    def promote_version(self, version_id: str) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT document_id FROM document_versions WHERE version_id=?",
                (version_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown version_id: {version_id}")
            document_id = str(row["document_id"])
            connection.execute(
                "UPDATE document_versions SET status='superseded' "
                "WHERE document_id=? AND status='active' AND version_id<>?",
                (document_id, version_id),
            )
            connection.execute(
                "UPDATE chunks SET status='superseded' "
                "WHERE document_id=? AND status='active' AND version_id<>?",
                (document_id, version_id),
            )
            connection.execute(
                "UPDATE document_versions SET status='active', rejection_reason=NULL WHERE version_id=?",
                (version_id,),
            )
            connection.execute(
                "UPDATE chunks SET status='active' WHERE version_id=?", (version_id,)
            )

    def withdraw_document(
        self,
        document_id: str,
        withdrawal_version_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE document_versions SET status='withdrawn', "
                "rejection_reason=COALESCE(?, rejection_reason) "
                "WHERE document_id=? AND status='active'",
                (reason, document_id),
            )
            connection.execute(
                "UPDATE chunks SET status='withdrawn' WHERE document_id=? AND status='active'",
                (document_id,),
            )
            if withdrawal_version_id:
                connection.execute(
                    "UPDATE document_versions SET status='withdrawn', "
                    "rejection_reason=COALESCE(?, rejection_reason) WHERE version_id=?",
                    (reason, withdrawal_version_id),
                )
                connection.execute(
                    "UPDATE chunks SET status='withdrawn' WHERE version_id=?",
                    (withdrawal_version_id,),
                )

    def rollback_to_version(self, version_id: str) -> None:
        version = self.get_version(version_id)
        if version is None:
            raise KeyError(f"Unknown version_id: {version_id}")
        self.promote_version(version_id)

    def get_chunks_by_ids(
        self, chunk_ids: Sequence[str], active_only: bool = True
    ) -> dict[str, ChunkRecord]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        sql = f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})"
        params: list[Any] = list(chunk_ids)
        if active_only:
            sql += " AND status='active'"
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return {str(row["chunk_id"]): _row_to_chunk(row) for row in rows}

    def list_chunks(
        self,
        status: DocumentStatus | None = DocumentStatus.ACTIVE,
        layer: KnowledgeLayer | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ChunkRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status.value)
        if layer is not None:
            clauses.append("layer=?")
            params.append(layer.value)
        sql = "SELECT * FROM chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY chunk_id"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def count_chunks(self, status: DocumentStatus | None = None) -> int:
        sql = "SELECT COUNT(*) FROM chunks"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status=?"
            params = (status.value,)
        with self.connect() as connection:
            return int(connection.execute(sql, params).fetchone()[0])

    def search_fts(self, query: str, limit: int = 40) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, bm25(fts_chunks, 0.0, 1.2, 0.8, 1.0, 0.4, 0.2) AS score
                FROM fts_chunks
                WHERE fts_chunks MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [(str(row["chunk_id"]), float(row["score"])) for row in rows]

    def get_cached_embedding(self, model_id: str, text_hash: str) -> list[float] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT embedding_json FROM embedding_cache WHERE model_id=? AND text_hash=?",
                (model_id, text_hash),
            ).fetchone()
        return None if row is None else [float(x) for x in json.loads(row["embedding_json"])]

    def save_cached_embedding(
        self, model_id: str, text_hash: str, embedding: Sequence[float]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache(model_id, text_hash, dimension, embedding_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(model_id, text_hash) DO UPDATE SET
                    dimension=excluded.dimension,
                    embedding_json=excluded.embedding_json
                """,
                (
                    model_id,
                    text_hash,
                    len(embedding),
                    canonical_json(list(embedding)),
                    utc_now().isoformat(),
                ),
            )

    def enqueue_vector_upsert(
        self,
        collection_name: str,
        chunk: ChunkRecord,
        embedding: Sequence[float],
    ) -> None:
        payload = {
            "chunk": chunk.model_dump(mode="json"),
            "embedding": list(embedding),
        }
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO vector_outbox(operation, collection_name, chunk_id, payload_json, created_at)
                VALUES('upsert', ?, ?, ?, ?)
                """,
                (
                    collection_name,
                    chunk.chunk_id,
                    canonical_json(payload),
                    utc_now().isoformat(),
                ),
            )

    def enqueue_vector_delete(self, collection_name: str, chunk_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO vector_outbox(operation, collection_name, chunk_id, payload_json, created_at)
                VALUES('delete', ?, ?, '{}', ?)
                """,
                (collection_name, chunk_id, utc_now().isoformat()),
            )

    def clear_vector_outbox(self) -> int:
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM vector_outbox")
            return int(cursor.rowcount)

    def pending_outbox(self, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM vector_outbox
                WHERE status IN ('pending', 'failed')
                ORDER BY outbox_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_outbox_processed(self, outbox_ids: Iterable[int]) -> None:
        ids = list(outbox_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.transaction() as connection:
            connection.execute(
                f"UPDATE vector_outbox SET status='processed', processed_at=?, last_error=NULL "
                f"WHERE outbox_id IN ({placeholders})",
                [utc_now().isoformat(), *ids],
            )

    def mark_outbox_failed(self, outbox_id: int, error: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                UPDATE vector_outbox
                SET status='failed', attempts=attempts+1, last_error=?
                WHERE outbox_id=?
                """,
                (error[:2000], outbox_id),
            )

    def outbox_has_pending_for_version(self, version_id: str) -> bool:
        chunks = self.list_chunks_for_version(version_id)
        if not chunks:
            return False
        ids = [chunk.chunk_id for chunk in chunks]
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            count = connection.execute(
                f"SELECT COUNT(*) FROM vector_outbox WHERE chunk_id IN ({placeholders}) "
                "AND status<>'processed'",
                ids,
            ).fetchone()[0]
        return int(count) > 0

    def list_staging_versions(self) -> list[DocumentVersion]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM document_versions WHERE status='staging' ORDER BY created_at"
            ).fetchall()
        return [_row_to_version(row) for row in rows]

    def last_successful_sync_at(self) -> datetime | None:
        return parse_datetime(self.get_metadata("last_successful_sync_at"))

    def stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            version_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM document_versions GROUP BY status"
            ).fetchall()
            chunk_rows = connection.execute(
                "SELECT status, layer, COUNT(*) AS count FROM chunks GROUP BY status, layer"
            ).fetchall()
            source_rows = connection.execute(
                """
                SELECT s.source_id, s.name,
                       COUNT(DISTINCT CASE WHEN v.status='active' THEN v.version_id END) AS active_documents,
                       COUNT(DISTINCT CASE WHEN v.status='staging' THEN v.version_id END) AS staging_documents,
                       COUNT(DISTINCT CASE WHEN c.status='active' THEN c.chunk_id END) AS active_chunks
                FROM sources s
                LEFT JOIN document_versions v ON v.source_id=s.source_id
                LEFT JOIN chunks c ON c.source_id=s.source_id
                GROUP BY s.source_id, s.name
                ORDER BY s.source_id
                """
            ).fetchall()
            outbox = connection.execute(
                "SELECT status, COUNT(*) AS count FROM vector_outbox GROUP BY status"
            ).fetchall()
        return {
            "versions_by_status": {row["status"]: row["count"] for row in version_rows},
            "chunks_by_status_layer": [dict(row) for row in chunk_rows],
            "sources": [dict(row) for row in source_rows],
            "vector_outbox": {row["status"]: row["count"] for row in outbox},
            "last_successful_sync_at": _iso(self.last_successful_sync_at()),
            "integrity_check": self.integrity_check(),
        }


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _row_to_version(row: sqlite3.Row) -> DocumentVersion:
    return DocumentVersion(
        document_id=row["document_id"],
        version_id=row["version_id"],
        source_id=row["source_id"],
        external_id=row["external_id"],
        title=row["title"],
        text=row["text"],
        url=row["url"],
        published_at=parse_datetime(row["published_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        retrieved_at=parse_datetime(row["retrieved_at"]),
        content_hash=row["content_hash"],
        status=DocumentStatus(row["status"]),
        layer=KnowledgeLayer(row["layer"]),
        evidence_level=EvidenceLevel(row["evidence_level"]),
        authority_score=float(row["authority_score"]),
        evidence_score=float(row["evidence_score"]),
        topics=json.loads(row["topics_json"]),
        publication_types=json.loads(row["publication_types_json"]),
        supersedes_version_id=row["supersedes_version_id"],
        rejection_reason=row["rejection_reason"],
        metadata=json.loads(row["metadata_json"]),
    )


def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
    retrieved_at = parse_datetime(row["retrieved_at"])
    if retrieved_at is None:
        raise ValueError(f"Chunk {row['chunk_id']} has invalid retrieved_at")
    return ChunkRecord(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        version_id=row["version_id"],
        source_id=row["source_id"],
        source_name=row["source_name"],
        title=row["title"],
        url=row["url"],
        layer=KnowledgeLayer(row["layer"]),
        status=DocumentStatus(row["status"]),
        section_path=row["section_path"],
        section_heading=row["section_heading"],
        ordinal=int(row["ordinal"]),
        text=row["text"],
        text_hash=row["text_hash"],
        published_at=parse_datetime(row["published_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        retrieved_at=retrieved_at,
        authority_score=float(row["authority_score"]),
        evidence_level=EvidenceLevel(row["evidence_level"]),
        evidence_score=float(row["evidence_score"]),
        topics=json.loads(row["topics_json"]),
        metadata=json.loads(row["metadata_json"]),
    )

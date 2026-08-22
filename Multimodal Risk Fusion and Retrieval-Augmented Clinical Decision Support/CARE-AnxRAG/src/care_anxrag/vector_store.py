from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .db import Database
from .models import ChunkRecord, KnowledgeLayer
from .util import batched, canonical_json, clamp, cosine_similarity


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    distance: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert(
        self,
        collection_name: str,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def delete(self, collection_name: str, chunk_ids: Sequence[str]) -> None: ...

    def query(
        self,
        collection_name: str,
        query_embedding: Sequence[float],
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]: ...

    def count(self, collection_name: str) -> int: ...

    def list_ids(self, collection_name: str) -> set[str]: ...

    def ping(self) -> bool: ...


class SQLiteVectorStore:
    """Portable exact cosine store used for tests and offline development.

    Production should use Chroma. This backend keeps the complete pipeline executable
    without external model/vector services and is intentionally not optimized for scale.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS care_vectors(
                    collection_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(collection_name, chunk_id)
                )
                """
            )

    def upsert(
        self,
        collection_name: str,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO care_vectors(collection_name, chunk_id, embedding_json, metadata_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(collection_name, chunk_id) DO UPDATE SET
                    embedding_json=excluded.embedding_json,
                    metadata_json=excluded.metadata_json
                """,
                [
                    (
                        collection_name,
                        chunk.chunk_id,
                        canonical_json(list(embedding)),
                        canonical_json(_chroma_metadata(chunk)),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )

    def delete(self, collection_name: str, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as connection:
            for batch in batched(list(chunk_ids), 500):
                placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"DELETE FROM care_vectors WHERE collection_name=? "
                    f"AND chunk_id IN ({placeholders})",
                    [collection_name, *batch],
                )

    def query(
        self,
        collection_name: str,
        query_embedding: Sequence[float],
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM care_vectors WHERE collection_name=?", (collection_name,)
            ).fetchall()
        hits: list[VectorHit] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if where and not _matches_where(metadata, where):
                continue
            similarity = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            hits.append(
                VectorHit(
                    chunk_id=str(row["chunk_id"]),
                    distance=clamp(1.0 - similarity, 0.0, 2.0),
                    metadata=metadata,
                )
            )
        hits.sort(key=lambda hit: hit.distance)
        return hits[:limit]

    def count(self, collection_name: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM care_vectors WHERE collection_name=?",
                    (collection_name,),
                ).fetchone()[0]
            )

    def list_ids(self, collection_name: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM care_vectors WHERE collection_name=?",
                (collection_name,),
            ).fetchall()
        return {str(row["chunk_id"]) for row in rows}

    def ping(self) -> bool:
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False


class ChromaVectorStore:
    def __init__(self, path: Path | str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Install the 'production' extra or set "
                "CARE_VECTOR_BACKEND=sqlite for an offline smoke test."
            ) from exc
        self._chromadb = chromadb
        self.client = chromadb.PersistentClient(path=str(path))
        self._collections: dict[str, Any] = {}

    def _collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"application": "CARE-AnxRAG", "metric": "cosine"},
            )
        return self._collections[name]

    def upsert(
        self,
        collection_name: str,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        collection = self._collection(collection_name)
        for chunk_batch, embedding_batch in zip(
            batched(list(chunks), 256), batched(list(embeddings), 256), strict=True
        ):
            collection.upsert(
                ids=[chunk.chunk_id for chunk in chunk_batch],
                embeddings=[list(vector) for vector in embedding_batch],
                documents=[chunk.text for chunk in chunk_batch],
                metadatas=[_chroma_metadata(chunk) for chunk in chunk_batch],
            )

    def delete(self, collection_name: str, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        collection = self._collection(collection_name)
        for batch in batched(list(chunk_ids), 500):
            collection.delete(ids=batch)

    def query(
        self,
        collection_name: str,
        query_embedding: Sequence[float],
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        collection = self._collection(collection_name)
        collection_count = int(collection.count())
        if collection_count == 0:
            return []
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(limit, collection_count),
            where=where,
            include=["metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            VectorHit(
                chunk_id=str(chunk_id),
                distance=float(distance),
                metadata=dict(metadata or {}),
            )
            for chunk_id, distance, metadata in zip(ids, distances, metadatas, strict=True)
        ]

    def count(self, collection_name: str) -> int:
        return int(self._collection(collection_name).count())

    def list_ids(self, collection_name: str) -> set[str]:
        collection = self._collection(collection_name)
        output: set[str] = set()
        offset = 0
        page_size = 1000
        while True:
            result = collection.get(limit=page_size, offset=offset, include=[])
            ids = [str(value) for value in (result.get("ids") or [])]
            output.update(ids)
            if len(ids) < page_size:
                break
            offset += len(ids)
        return output

    def ping(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False


class VectorIndexer:
    def __init__(self, database: Database, vector_store: VectorStore):
        self.database = database
        self.vector_store = vector_store

    def drain(self, limit: int = 500) -> dict[str, int]:
        rows = self.database.pending_outbox(limit=limit)
        processed = 0
        failed = 0
        for row in rows:
            outbox_id = int(row["outbox_id"])
            try:
                operation = str(row["operation"])
                collection_name = str(row["collection_name"])
                if operation == "delete":
                    self.vector_store.delete(collection_name, [str(row["chunk_id"])])
                elif operation == "upsert":
                    payload = json.loads(row["payload_json"])
                    chunk = ChunkRecord.model_validate(payload["chunk"])
                    embedding = [float(value) for value in payload["embedding"]]
                    self.vector_store.upsert(collection_name, [chunk], [embedding])
                else:
                    raise ValueError(f"Unknown vector outbox operation: {operation}")
                self.database.mark_outbox_processed([outbox_id])
                processed += 1
            except Exception as exc:
                self.database.mark_outbox_failed(outbox_id, str(exc))
                failed += 1
        return {"processed": processed, "failed": failed}


def collection_for_layer(
    layer: KnowledgeLayer,
    clinical_collection: str = "care_clinical_core",
    research_collection: str = "care_research_frontier",
) -> str:
    return clinical_collection if layer == KnowledgeLayer.CLINICAL_CORE else research_collection


def _chroma_metadata(chunk: ChunkRecord) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "document_id": chunk.document_id,
        "version_id": chunk.version_id,
        "source_id": chunk.source_id,
        "source_name": chunk.source_name,
        "title": chunk.title,
        "layer": chunk.layer.value,
        "status_at_index_time": chunk.status.value,
        "section_path": chunk.section_path,
        "section_heading": chunk.section_heading,
        "ordinal": chunk.ordinal,
        "authority_score": chunk.authority_score,
        "evidence_level": chunk.evidence_level.value,
        "evidence_score": chunk.evidence_score,
        "topics": "|".join(chunk.topics),
        "retrieved_at": chunk.retrieved_at.isoformat(),
    }
    if chunk.url:
        metadata["url"] = chunk.url
    if chunk.published_at:
        metadata["published_at"] = chunk.published_at.isoformat()
    if chunk.updated_at:
        metadata["updated_at"] = chunk.updated_at.isoformat()
    return metadata


def _matches_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    for key, expected in where.items():
        if key == "$and":
            return all(_matches_where(metadata, item) for item in expected)
        if key == "$or":
            return any(_matches_where(metadata, item) for item in expected)
        actual = metadata.get(key)
        if isinstance(expected, dict):
            for operator, value in expected.items():
                if operator == "$eq" and actual != value:
                    return False
                if operator == "$ne" and actual == value:
                    return False
                if operator == "$gte" and not (actual is not None and actual >= value):
                    return False
                if operator == "$lte" and not (actual is not None and actual <= value):
                    return False
        elif actual != expected:
            return False
    return True

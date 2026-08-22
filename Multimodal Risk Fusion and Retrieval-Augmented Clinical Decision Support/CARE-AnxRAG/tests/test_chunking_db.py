from __future__ import annotations

from care_anxrag.chunking import ChunkingConfig, SectionAwareChunker
from care_anxrag.util import fts_query, normalize_whitespace


def test_chunker_preserves_overlap_and_limits() -> None:
    chunker = SectionAwareChunker(ChunkingConfig(max_words=30, overlap_words=5, min_words=5))
    text = " ".join(f"Sentence {index} contains several anxiety treatment words." for index in range(20))
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 60 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_fts_query_is_safe_and_normalized() -> None:
    expression = fts_query('panic "attack" OR treatment*')
    assert '"panic"' in expression
    assert '"attack"' in expression
    assert "*" not in expression
    assert normalize_whitespace("a   b\n\n\n c") == "a b\n\nc"


def test_sqlite_vector_delete_batches_large_id_sets(tmp_path) -> None:
    from care_anxrag.vector_store import SQLiteVectorStore

    store = SQLiteVectorStore(tmp_path / "vectors.sqlite3")
    with store._connect() as connection:
        connection.executemany(
            "INSERT INTO care_vectors(collection_name, chunk_id, embedding_json, metadata_json) "
            "VALUES(?, ?, ?, ?)",
            [("collection", f"chunk-{index}", "[1.0]", "{}") for index in range(1200)],
        )
    ids = [f"chunk-{index}" for index in range(1200)]
    store.delete("collection", ids)
    assert store.count("collection") == 0

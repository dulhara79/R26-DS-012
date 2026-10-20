"""Offline tests: backfill operator must preserve corpus integrity and provenance."""
from pathlib import Path
import runpy
import sqlite3

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = runpy.run_path(str(ROOT / "scripts" / "pubmed_backfill.py"))


def test_profile_is_nonempty_and_anxiety_specific():
    profile = OPERATOR["read_profile"](
        ROOT / "config" / "pubmed_psychiatrist_profile.yaml"
    )
    assert "generalized anxiety disorder" in profile["query"].lower()


def test_effective_registry_does_not_mutate_source_registry(tmp_path):
    registry = {"sources": [{
        "id": "pubmed_anxiety", "connector": "pubmed",
        "enabled": True, "publish_to_rag": True, "auto_promote": False,
        "settings": {"query": "original", "max_records_per_sync": 250},
    }]}
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    profile = {"query": "psychiatrist study evidence", "max_records_per_sync": 500}
    staged = OPERATOR["effective_registry"](path, profile, False)
    demo = OPERATOR["effective_registry"](path, profile, True)
    assert staged["sources"][0]["auto_promote"] is False
    assert demo["sources"][0]["auto_promote"] is True
    assert staged["sources"][0]["settings"]["query"] == profile["query"]
    assert yaml.safe_load(path.read_text()) == registry


def test_counts_distinct_active_documents_not_chunks_or_staging(tmp_path):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE document_versions(document_id TEXT, status TEXT)")
        conn.executemany("INSERT INTO document_versions VALUES (?, ?)", [
            ("a", "active"), ("a", "active"), ("b", "active"), ("c", "staging")
        ])
    assert OPERATOR["active_documents"](db) == 2


def test_missing_database_does_not_initialize(tmp_path):
    with pytest.raises(FileNotFoundError):
        OPERATOR["active_documents"](tmp_path / "nonexistent.sqlite")


def test_sqlite_online_backup(tmp_path):
    db = tmp_path / "source.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sample (v TEXT)")
        conn.execute("INSERT INTO sample VALUES ('preserved')")
    copy = tmp_path / "backup.sqlite"
    OPERATOR["backup_database"](db, copy)
    with sqlite3.connect(copy) as conn:
        assert conn.execute("SELECT v FROM sample").fetchone()[0] == "preserved"

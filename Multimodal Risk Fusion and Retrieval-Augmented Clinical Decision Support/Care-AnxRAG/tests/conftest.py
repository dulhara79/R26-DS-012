from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from care_anxrag.config import RetrievalWeights, Settings
from care_anxrag.runtime import Runtime, build_runtime


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "docs").mkdir()
    registry = {
        "sources": [
            {
                "id": "test_core",
                "name": "Test Clinical Core",
                "connector": "local_files",
                "enabled": True,
                "publish_to_rag": True,
                "auto_promote": True,
                "layer": "clinical_core",
                "authority_score": 0.95,
                "evidence_level": "clinical_guideline",
                "check_interval_minutes": 60,
                "settings": {
                    "path": str(tmp_path / "docs"),
                    "patterns": ["**/*.md"],
                },
            }
        ]
    }
    (tmp_path / "config" / "sources.yaml").write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    return tmp_path


def make_settings(project: Path, **overrides: object) -> Settings:
    values = dict(
        project_root=project,
        care_home=project / "var",
        database_path=project / "var" / "care.sqlite3",
        vector_path=project / "var" / "vectors",
        source_registry_path=project / "config" / "sources.yaml",
        vector_backend="sqlite",
        embedding_provider="hash",
        embedding_model="hash",
        embedding_dimensions=256,
        generator_provider="rule",
        generation_model="rule",
        reranker_provider="heuristic",
        nli_provider="heuristic",
        allow_network_sync=False,
        minimum_care_score=0.28,
        minimum_confidence=0.35,
        contradiction_threshold=0.70,
        unresolved_conflict_threshold=0.20,
        dominance_margin=0.15,
        weights=RetrievalWeights(),
    )
    values.update(overrides)
    settings = Settings(**values)
    settings.validate()
    return settings


@pytest.fixture()
def settings(project: Path) -> Settings:
    return make_settings(project)


@pytest.fixture()
def runtime(settings: Settings) -> Runtime:
    return build_runtime(settings)


def write_document(
    project: Path,
    filename: str,
    *,
    external_id: str,
    title: str,
    topics: list[str],
    body: str,
    publication_types: list[str] | None = None,
) -> Path:
    metadata = {
        "external_id": external_id,
        "title": title,
        "topics": topics,
        "publication_types": publication_types or ["Clinical Guideline"],
        "published_at": "2025-01-01",
        "updated_at": "2026-01-01",
        "url": f"https://example.org/{external_id}",
    }
    path = project / "docs" / filename
    path.write_text(
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False).strip()
        + "\n---\n\n"
        + body.strip()
        + "\n",
        encoding="utf-8",
    )
    return path

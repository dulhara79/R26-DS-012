from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from care_anxrag.config import RetrievalWeights, Settings


def test_weights_sum_to_one() -> None:
    weights = RetrievalWeights()
    weights.validate()
    assert round(sum(asdict(weights).values()), 8) == 1.0


def test_settings_from_environment(project, monkeypatch) -> None:
    monkeypatch.setenv("CARE_HOME", str(project / "custom-var"))
    monkeypatch.setenv("CARE_VECTOR_BACKEND", "sqlite")
    monkeypatch.setenv("CARE_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("CARE_GENERATOR_PROVIDER", "rule")
    monkeypatch.setenv("CARE_RERANKER_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_NLI_PROVIDER", "heuristic")
    settings = Settings.from_env(project_root=project)
    assert settings.care_home == (project / "custom-var").resolve()
    assert settings.vector_backend == "sqlite"


def test_relative_paths_are_project_relative(project: Path) -> None:
    settings = Settings.from_env(
        project_root=project,
        environ={
            "CARE_HOME": "runtime",
            "CARE_DATABASE_PATH": "state/care.sqlite3",
            "CARE_VECTOR_PATH": "state/vectors",
            "CARE_SOURCE_REGISTRY": "config/sources.yaml",
            "CARE_VECTOR_BACKEND": "sqlite",
            "CARE_EMBEDDING_PROVIDER": "hash",
            "CARE_GENERATOR_PROVIDER": "rule",
            "CARE_RERANKER_PROVIDER": "heuristic",
            "CARE_NLI_PROVIDER": "heuristic",
        },
    )
    assert settings.care_home == (project / "runtime").resolve()
    assert settings.database_path == (project / "state" / "care.sqlite3").resolve()
    assert settings.vector_path == (project / "state" / "vectors").resolve()
    assert settings.source_registry_path == (project / "config" / "sources.yaml").resolve()


def test_invalid_boolean_is_rejected(project: Path) -> None:
    with pytest.raises(ValueError, match="Invalid boolean value"):
        Settings.from_env(
            project_root=project,
            environ={
                "CARE_ALLOW_NETWORK_SYNC": "tru",
                "CARE_VECTOR_BACKEND": "sqlite",
                "CARE_EMBEDDING_PROVIDER": "hash",
                "CARE_GENERATOR_PROVIDER": "rule",
                "CARE_RERANKER_PROVIDER": "heuristic",
                "CARE_NLI_PROVIDER": "heuristic",
            },
        )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CARE_MIN_DISTINCT_SOURCES", "0", "must be positive"),
        ("CARE_CLINICAL_HALF_LIFE_DAYS", "0", "must be positive"),
        ("CARE_REQUEST_TIMEOUT_SECONDS", "0", "must be positive"),
        ("CARE_RERANK_CANDIDATES", "31", "cannot exceed"),
    ],
)
def test_invalid_operational_settings_are_rejected(
    project: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    environment = {
        "CARE_VECTOR_BACKEND": "sqlite",
        "CARE_EMBEDDING_PROVIDER": "hash",
        "CARE_GENERATOR_PROVIDER": "rule",
        "CARE_RERANKER_PROVIDER": "heuristic",
        "CARE_NLI_PROVIDER": "heuristic",
        name: value,
    }
    with pytest.raises(ValueError, match=message):
        Settings.from_env(project_root=project, environ=environment)


def test_embeddinggemma_uses_supported_default_dimension(project: Path) -> None:
    settings = Settings.from_env(project_root=project, environ={})
    assert settings.embedding_model == "embeddinggemma"
    assert settings.embedding_dimensions == 256


def test_embeddinggemma_rejects_unsupported_dimension(project: Path) -> None:
    with pytest.raises(ValueError, match="embeddinggemma supports"):
        Settings.from_env(
            project_root=project,
            environ={
                "CARE_EMBEDDING_PROVIDER": "ollama",
                "CARE_EMBEDDING_MODEL": "embeddinggemma",
                "CARE_EMBEDDING_DIMENSIONS": "384",
            },
        )

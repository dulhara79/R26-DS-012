from __future__ import annotations

import json

from care_anxrag.cli import _json
from care_anxrag.models import HealthStatus
from care_anxrag.registry import load_source_registry
from care_anxrag.scaffold import scaffold_project


def test_scaffold_creates_standalone_project_and_preserves_operator_edits(tmp_path) -> None:
    result = scaffold_project(tmp_path)
    registry = tmp_path / "config" / "sources.yaml"
    env_example = tmp_path / ".env.example"
    assert result[str(registry)] == "written"
    assert result[str(env_example)] == "written"
    assert load_source_registry(registry)
    assert (tmp_path / "data" / "local").is_dir()

    registry.write_text("sources: []\n", encoding="utf-8")
    second = scaffold_project(tmp_path)
    assert second[str(registry)] == "preserved"
    assert registry.read_text(encoding="utf-8") == "sources: []\n"


def test_cli_json_serializes_nested_pydantic_models() -> None:
    payload = json.loads(
        _json(
            {
                "scaffold": {"config/sources.yaml": "written"},
                "health": HealthStatus(
                    status="ok",
                    database=True,
                    vector_store=True,
                    ollama=None,
                    details={"embedding_identity_ok": True},
                ),
            }
        )
    )
    assert payload["health"]["status"] == "ok"
    assert payload["health"]["details"]["embedding_identity_ok"] is True

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from care_anxrag.api import create_app
from conftest import write_document


BODY = """
# Panic symptoms
Panic attacks may include a sudden surge of fear, rapid heartbeat, sweating, trembling, dizziness, and an urge to escape. Severe new physical symptoms require medical assessment rather than automatic attribution to anxiety.

# Care
Evidence-based psychological care can include cognitive behavioural and exposure-based approaches delivered with appropriate clinical assessment.
"""


def test_api_health_ask_and_stats(runtime, project: Path) -> None:
    write_document(
        project,
        "panic.md",
        external_id="panic",
        title="Panic Information",
        topics=["anxiety", "panic_disorder"],
        body=BODY,
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["database"] is True
        assert "details" not in health.json()
        assert "database_path" not in health.text

        response = client.post("/v1/ask", json={"question": "What symptoms can occur in a panic attack?"})
        assert response.status_code == 200
        assert response.json()["citations"]

        stats = client.get("/v1/stats")
        assert stats.status_code == 200
        assert stats.json()["integrity_check"] == "ok"


def test_debug_and_raw_retrieval_require_admin_key(runtime) -> None:
    runtime.settings.admin_key = "test-secret"
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        denied_debug = client.post(
            "/v1/ask",
            json={"question": "What is anxiety?", "include_debug": True},
        )
        assert denied_debug.status_code == 401

        denied_retrieve = client.post(
            "/v1/retrieve",
            json={"question": "What is anxiety?"},
        )
        assert denied_retrieve.status_code == 401

        allowed_retrieve = client.post(
            "/v1/retrieve",
            headers={"X-Admin-Key": "test-secret"},
            json={"question": "What is anxiety?"},
        )
        assert allowed_retrieve.status_code == 200


def test_admin_api_is_disabled_when_key_is_not_configured(runtime) -> None:
    runtime.settings.admin_key = ""
    app = create_app(runtime=runtime)
    with TestClient(app) as client:
        response = client.post(
            "/v1/retrieve",
            json={"question": "What is anxiety?"},
        )
        assert response.status_code == 503

        debug = client.post(
            "/v1/ask",
            json={"question": "What is anxiety?", "include_debug": True},
        )
        assert debug.status_code == 503

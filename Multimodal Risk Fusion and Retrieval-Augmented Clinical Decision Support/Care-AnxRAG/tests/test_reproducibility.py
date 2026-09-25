from __future__ import annotations

from conftest import make_settings, write_document

from care_anxrag.reproducibility import build_experiment_snapshot
from care_anxrag.runtime import build_runtime
from care_anxrag.util import sha256_text


def test_snapshot_records_active_evidence_and_runtime_configuration(
    project,
) -> None:
    settings = make_settings(
        project,
        admin_key="must-not-leak",
    )
    runtime = build_runtime(settings)
    write_document(
        project,
        "gad.md",
        external_id="gad-guidance",
        title="GAD Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=(
            "Generalized anxiety disorder is characterized by persistent and "
            "excessive worry that can interfere with daily functioning. "
            "Clinical guidance discusses assessment, functional impairment, "
            "and evidence-based psychological treatment. Cognitive behavioural "
            "therapy is discussed as a treatment option for generalized anxiety "
            "disorder, with care decisions interpreted by qualified clinicians. "
            "Follow-up should consider symptoms, functioning, and individual context."
        ),
    )
    runtime.ingestion.sync(
        source_ids=["test_core"],
        force=True,
    )

    snapshot = build_experiment_snapshot(
        runtime,
        code_revision="abc123",
    )

    assert snapshot["snapshot_version"] == 1
    assert snapshot["code_revision"] == "abc123"
    assert snapshot["database_schema_version"] == "1"
    assert snapshot["corpus"]["active_version_count"] == 1

    active = snapshot["corpus"]["active_versions"][0]
    assert active["external_id"] == "gad-guidance"
    assert active["version_id"]
    assert active["content_hash"]
    assert active["status"] == "active"

    assert snapshot["models"]["embedding_runtime_id"] == "hash:256:v2"
    assert snapshot["models"]["answer_provider"] == "extractive"
    assert snapshot["retrieval"]["weights"]["semantic"] == 0.20
    assert snapshot["chunking"] == {
        "max_words": 180,
        "overlap_words": 35,
        "min_words": 35,
    }


def test_snapshot_hashes_registry_and_optional_benchmark(
    runtime,
    project,
) -> None:
    benchmark = project / "benchmark.jsonl"
    benchmark.write_text(
        '{"id":"q1","question":"synthetic"}\n',
        encoding="utf-8",
    )

    snapshot = build_experiment_snapshot(
        runtime,
        code_revision="abc123",
        benchmark_path=benchmark,
    )

    registry_text = (
        project / "config" / "sources.yaml"
    ).read_text(encoding="utf-8")

    assert snapshot["source_registry_sha256"] == sha256_text(
        registry_text
    )
    assert snapshot["benchmark"]["path"] == str(
        benchmark.resolve()
    )
    assert snapshot["benchmark"]["sha256"] == sha256_text(
        benchmark.read_text(encoding="utf-8")
    )


def test_snapshot_does_not_expose_secrets(project) -> None:
    settings = make_settings(
        project,
        admin_key="top-secret-admin-key",
    )
    runtime = build_runtime(settings)

    snapshot = build_experiment_snapshot(
        runtime,
        code_revision="abc123",
    )
    serialized = str(snapshot)

    assert "top-secret-admin-key" not in serialized
    assert "admin_key" not in snapshot["retrieval"]
    assert "admin_key" not in snapshot["models"]


def test_database_lists_only_active_version_fingerprints(
    runtime,
    project,
) -> None:
    write_document(
        project,
        "active.md",
        external_id="active-doc",
        title="Active document",
        topics=["anxiety"],
        body=(
            "This clinical guidance discusses anxiety disorders, including "
            "assessment, symptom burden, functional impact, and evidence-based "
            "care. It describes the importance of evaluating persistent anxiety "
            "in clinical context and considering psychological interventions, "
            "ongoing monitoring, and individualized professional assessment. "
            "The document is intentionally long enough to pass the normal "
            "ingestion quality gate used by CARE-AnxRAG."
        ),
    )
    runtime.ingestion.sync(
        source_ids=["test_core"],
        force=True,
    )

    rows = runtime.database.list_active_version_fingerprints()

    assert len(rows) == 1
    assert rows[0]["external_id"] == "active-doc"
    assert set(rows[0]) == {
        "version_id",
        "document_id",
        "source_id",
        "external_id",
        "content_hash",
        "status",
        "layer",
        "evidence_level",
        "published_at",
        "updated_at",
        "retrieved_at",
    }



def test_snapshot_experiment_cli_writes_json(project, monkeypatch) -> None:
    import json

    from typer.testing import CliRunner

    from care_anxrag.cli import app

    monkeypatch.setenv("CARE_VECTOR_BACKEND", "sqlite")
    monkeypatch.setenv("CARE_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("CARE_EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("CARE_GENERATOR_PROVIDER", "extractive")
    monkeypatch.setenv("CARE_RERANKER_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_NLI_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_ALLOW_NETWORK_SYNC", "false")

    output = project / "experiment-snapshot.json"

    result = CliRunner().invoke(
        app,
        [
            "snapshot-experiment",
            str(output),
            "--code-revision",
            "abc123",
            "--project-root",
            str(project),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["code_revision"] == "abc123"
    assert payload["models"]["answer_policy"] == "extractive_only"

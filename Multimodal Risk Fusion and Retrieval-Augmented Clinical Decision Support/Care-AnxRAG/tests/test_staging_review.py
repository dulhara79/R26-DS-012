from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from care_anxrag.staging_review import (
    apply_staging_review,
    write_staging_review_package,
)

from conftest import write_document


def _body(label: str) -> str:
    return (
        "# Evidence\n"
        f"{label} discusses anxiety disorders and evidence relevant to clinical "
        "research review. This fixture contains enough detail to pass the normal "
        "ingestion quality gate and exercise staging review without depending on "
        "network access or model inference. Review decisions remain explicit and "
        "must never be inferred by the software."
    )


def _stage_one(runtime, project: Path, *, title: str = "Review candidate"):
    runtime.sources[0].auto_promote = False
    write_document(
        project,
        "candidate.md",
        external_id="candidate-1",
        title=title,
        topics=["anxiety"],
        body=_body(title),
    )
    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert summary.staged == 1
    versions = runtime.database.list_staging_versions()
    assert len(versions) == 1
    return versions[0]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_review_package_uses_historical_plan_only_as_hint(
    runtime,
    project: Path,
    tmp_path: Path,
) -> None:
    version = _stage_one(runtime, project, title="Historical candidate")
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            [
                {
                    "version_id": version.version_id,
                    "title": version.title,
                    "role": "out_of_domain",
                    "evidence_role": "review_evidence",
                    "score": 0.1,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.csv"

    report = write_staging_review_package(
        runtime,
        output,
        hints_path=hints,
    )

    rows = _read_rows(output)
    assert report["count"] == 1
    assert rows[0]["version_id"] == version.version_id
    assert rows[0]["historical_hint_role"] == "out_of_domain"
    assert rows[0]["historical_hint_evidence_role"] == "review_evidence"
    assert rows[0]["historical_hint_score"] == "0.1"
    assert rows[0]["decision"] == ""
    assert rows[0]["decision_reason"] == ""
    assert rows[0]["reviewer_id"] == ""


def test_apply_review_refuses_blank_decision_without_mutating(
    runtime,
    project: Path,
    tmp_path: Path,
) -> None:
    version = _stage_one(runtime, project)
    review = tmp_path / "review.csv"
    write_staging_review_package(runtime, review)

    with pytest.raises(ValueError, match="decision"):
        apply_staging_review(runtime, review, apply=True)

    unchanged = runtime.database.get_version(version.version_id)
    assert unchanged is not None
    assert unchanged.status.value == "staging"


def test_apply_review_refuses_title_mismatch(
    runtime,
    project: Path,
    tmp_path: Path,
) -> None:
    version = _stage_one(runtime, project)
    review = tmp_path / "review.csv"
    write_staging_review_package(runtime, review)
    rows = _read_rows(review)
    rows[0]["title"] = "Different title"
    rows[0]["decision"] = "reject"
    rows[0]["decision_reason"] = "Human review: outside scope"
    rows[0]["reviewer_id"] = "reviewer-1"
    _write_rows(review, rows)

    with pytest.raises(ValueError, match="title"):
        apply_staging_review(runtime, review, apply=True)

    unchanged = runtime.database.get_version(version.version_id)
    assert unchanged is not None
    assert unchanged.status.value == "staging"


def test_apply_review_dry_run_reports_without_mutating(
    runtime,
    project: Path,
    tmp_path: Path,
) -> None:
    version = _stage_one(runtime, project)
    review = tmp_path / "review.csv"
    write_staging_review_package(runtime, review)
    rows = _read_rows(review)
    rows[0]["decision"] = "reject"
    rows[0]["decision_reason"] = "Human review: outside scope"
    rows[0]["reviewer_id"] = "reviewer-1"
    _write_rows(review, rows)

    report = apply_staging_review(runtime, review, apply=False)

    assert report["validated"] == 1
    assert report["applied"] == 0
    unchanged = runtime.database.get_version(version.version_id)
    assert unchanged is not None
    assert unchanged.status.value == "staging"


def test_apply_review_rejects_only_after_explicit_apply(
    runtime,
    project: Path,
    tmp_path: Path,
) -> None:
    version = _stage_one(runtime, project)
    review = tmp_path / "review.csv"
    write_staging_review_package(runtime, review)
    rows = _read_rows(review)
    rows[0]["decision"] = "reject"
    rows[0]["decision_reason"] = "Human review: outside scope"
    rows[0]["reviewer_id"] = "reviewer-1"
    _write_rows(review, rows)

    report = apply_staging_review(runtime, review, apply=True)

    assert report["validated"] == 1
    assert report["applied"] == 1
    assert report["rejected"] == 1
    updated = runtime.database.get_version(version.version_id)
    assert updated is not None
    assert updated.status.value == "rejected"


def test_apply_review_approves_only_after_explicit_apply(
    runtime,
    project: Path,
    tmp_path: Path,
) -> None:
    version = _stage_one(runtime, project)
    review = tmp_path / "review.csv"
    write_staging_review_package(runtime, review)
    rows = _read_rows(review)
    rows[0]["decision"] = "approve"
    rows[0]["decision_reason"] = "Human review: in scope"
    rows[0]["reviewer_id"] = "reviewer-1"
    _write_rows(review, rows)

    report = apply_staging_review(runtime, review, apply=True)

    assert report["validated"] == 1
    assert report["applied"] == 1
    assert report["approved"] == 1
    updated = runtime.database.get_version(version.version_id)
    assert updated is not None
    assert updated.status.value == "active"



def test_staging_review_package_cli_writes_blank_decisions(
    runtime,
    project: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from typer.testing import CliRunner

    from care_anxrag.cli import app

    version = _stage_one(runtime, project, title="CLI review candidate")
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            [
                {
                    "version_id": version.version_id,
                    "title": version.title,
                    "role": "out_of_domain",
                    "evidence_role": "result_evidence",
                    "score": 0.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.csv"

    monkeypatch.setenv("CARE_HOME", str(runtime.settings.care_home))
    monkeypatch.setenv("CARE_DATABASE_PATH", str(runtime.settings.database_path))
    monkeypatch.setenv("CARE_VECTOR_PATH", str(runtime.settings.vector_path))
    monkeypatch.setenv(
        "CARE_SOURCE_REGISTRY",
        str(runtime.settings.source_registry_path),
    )
    monkeypatch.setenv("CARE_VECTOR_BACKEND", "sqlite")
    monkeypatch.setenv("CARE_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("CARE_EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("CARE_EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("CARE_GENERATOR_PROVIDER", "extractive")
    monkeypatch.setenv("CARE_RERANKER_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_NLI_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_ALLOW_NETWORK_SYNC", "false")

    result = CliRunner().invoke(
        app,
        [
            "staging-review-package",
            str(output),
            "--hints",
            str(hints),
            "--project-root",
            str(project),
        ],
    )

    assert result.exit_code == 0, result.output
    rows = _read_rows(output)
    assert rows[0]["decision"] == ""
    assert rows[0]["historical_hint_role"] == "out_of_domain"


def test_staging_review_apply_cli_is_dry_run_without_apply_flag(
    runtime,
    project: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json as json_module

    from typer.testing import CliRunner

    from care_anxrag.cli import app

    version = _stage_one(runtime, project)
    review = tmp_path / "review.csv"
    write_staging_review_package(runtime, review)
    rows = _read_rows(review)
    rows[0]["decision"] = "reject"
    rows[0]["decision_reason"] = "Human review: outside scope"
    rows[0]["reviewer_id"] = "reviewer-1"
    _write_rows(review, rows)

    monkeypatch.setenv("CARE_HOME", str(runtime.settings.care_home))
    monkeypatch.setenv("CARE_DATABASE_PATH", str(runtime.settings.database_path))
    monkeypatch.setenv("CARE_VECTOR_PATH", str(runtime.settings.vector_path))
    monkeypatch.setenv(
        "CARE_SOURCE_REGISTRY",
        str(runtime.settings.source_registry_path),
    )
    monkeypatch.setenv("CARE_VECTOR_BACKEND", "sqlite")
    monkeypatch.setenv("CARE_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("CARE_EMBEDDING_MODEL", "hash")
    monkeypatch.setenv("CARE_EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("CARE_GENERATOR_PROVIDER", "extractive")
    monkeypatch.setenv("CARE_RERANKER_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_NLI_PROVIDER", "heuristic")
    monkeypatch.setenv("CARE_ALLOW_NETWORK_SYNC", "false")

    result = CliRunner().invoke(
        app,
        [
            "staging-review-apply",
            str(review),
            "--project-root",
            str(project),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json_module.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["applied"] == 0
    unchanged = runtime.database.get_version(version.version_id)
    assert unchanged is not None
    assert unchanged.status.value == "staging"

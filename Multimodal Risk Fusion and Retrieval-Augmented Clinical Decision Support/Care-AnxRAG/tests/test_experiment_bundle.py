from __future__ import annotations

import json

import pytest

from conftest import write_document
from care_anxrag.experiment_bundle import run_experiment_bundle


def _prepare_benchmark(runtime, project):
    write_document(
        project,
        "gad-cbt.md",
        external_id="gad-cbt-guidance",
        title="GAD CBT Guidance",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=(
            "Generalized anxiety disorder involves persistent excessive worry and "
            "functional impairment. Cognitive behavioural therapy is an evidence-based "
            "psychological treatment discussed for generalized anxiety disorder. "
            "Clinical care should consider symptoms, functioning, patient preferences, "
            "and appropriate professional follow-up. This document contains sufficient "
            "detail to satisfy the controlled ingestion quality threshold for testing."
        ),
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    benchmark = project / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "What evidence discusses CBT for generalized anxiety disorder?",
                "relevant_external_ids": ["gad-cbt-guidance"],
                "must_abstain": False,
                "expects_conflict": False,
                "stratum": "psychological_interventions",
                "split": "development",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return benchmark


def test_experiment_bundle_writes_required_artifacts(runtime, project) -> None:
    benchmark = _prepare_benchmark(runtime, project)
    output = project / "artifacts" / "run-001"

    result = run_experiment_bundle(
        runtime,
        benchmark,
        output,
        code_revision="abc123",
    )

    expected = {
        "ablation.json",
        "comparisons.json",
        "comparisons.csv",
        "coverage.json",
        "timings.json",
        "snapshot.json",
        "manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert result["manifest"]["code_revision"] == "abc123"
    assert result["manifest"]["item_count"] == 1
    assert result["manifest"]["ablation_modes"] == [
        "B0_dense_only",
        "B1_lexical_only",
        "B2_hybrid_rrf",
        "B3_hybrid_rerank",
        "B4_care",
        "B5_care_conflict",
        "CARE_full",
    ]

    comparisons = json.loads((output / "comparisons.json").read_text(encoding="utf-8"))
    assert comparisons["reference"] == "CARE_full"
    assert "B0_dense_only" in comparisons["comparisons"]
    assert (output / "comparisons.csv").read_text(encoding="utf-8").startswith("baseline,reference,analysis,metric")

    timings = json.loads((output / "timings.json").read_text(encoding="utf-8"))
    assert timings["count"] == 1
    assert timings["per_item"][0]["id"] == "q1"


def test_experiment_bundle_refuses_non_empty_output(runtime, project) -> None:
    benchmark = _prepare_benchmark(runtime, project)
    output = project / "artifacts" / "existing"
    output.mkdir(parents=True)
    (output / "do-not-overwrite.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        run_experiment_bundle(
            runtime,
            benchmark,
            output,
            code_revision="abc123",
        )

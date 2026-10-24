from __future__ import annotations

import json

import pytest

from care_anxrag.corpus_freeze import build_corpus_freeze
from care_anxrag.experiment_preflight import validate_final_experiment


def _seed_active_guidance(runtime, project) -> None:
    from conftest import write_document

    write_document(
        project,
        "guidance.md",
        external_id="guidance-1",
        title="Anxiety guidance",
        topics=["anxiety"],
        body=(
            "This clinical guidance discusses anxiety assessment, persistent "
            "symptoms, functional impairment, evidence-based psychological care, "
            "monitoring, follow-up, and individualized interpretation. It contains "
            "enough substantive information to pass ingestion quality validation "
            "and become active evidence. Qualified clinical judgment should "
            "consider symptoms, functioning, treatment context, patient preference, "
            "change over time, uncertainty, and the limits of any single evidence source."
        ),
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)


def _write_reviewed_benchmark(path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "reviewed-1",
                "question": "What anxiety evidence is available?",
                "relevant_external_ids": ["guidance-1"],
                "relevant_source_ids": [],
                "prohibited_external_ids": [],
                "prohibited_source_ids": [],
                "must_abstain": False,
                "expects_conflict": False,
                "stratum": "generalized_anxiety_disorder",
                "split": "test",
                "intent": "general",
                "anxiety_subtypes": [],
                "treatments": [],
                "population": None,
                "gold_evidence_excerpts": [],
                "prohibited_claims": [],
                "annotator_ids": ["reviewer-a", "reviewer-b"],
                "adjudicated": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_final_experiment_preflight_accepts_matching_freeze(
    runtime,
    project,
    tmp_path,
) -> None:
    _seed_active_guidance(runtime, project)
    freeze = build_corpus_freeze(runtime, code_revision="abc123")
    assert freeze["ready_to_freeze"] is True

    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    benchmark = tmp_path / "benchmark.jsonl"
    _write_reviewed_benchmark(benchmark)

    report = validate_final_experiment(
        runtime,
        corpus_freeze_path=freeze_path,
        benchmark_path=benchmark,
        code_revision="abc123",
    )

    assert report["status"] == "ready"
    assert report["benchmark_item_count"] == 1
    assert report["active_version_count"] == 1


def test_final_experiment_preflight_rejects_code_revision_mismatch(
    runtime,
    project,
    tmp_path,
) -> None:
    _seed_active_guidance(runtime, project)
    freeze = build_corpus_freeze(runtime, code_revision="abc123")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    benchmark = tmp_path / "benchmark.jsonl"
    _write_reviewed_benchmark(benchmark)

    with pytest.raises(ValueError, match="code revision"):
        validate_final_experiment(
            runtime,
            corpus_freeze_path=freeze_path,
            benchmark_path=benchmark,
            code_revision="different",
        )


def test_final_experiment_preflight_rejects_unadjudicated_item(
    runtime,
    project,
    tmp_path,
) -> None:
    _seed_active_guidance(runtime, project)
    freeze = build_corpus_freeze(runtime, code_revision="abc123")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "id": "unreviewed",
                "question": "Synthetic question",
                "split": "development",
                "annotator_ids": [],
                "adjudicated": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two distinct reviewers"):
        validate_final_experiment(
            runtime,
            corpus_freeze_path=freeze_path,
            benchmark_path=benchmark,
            code_revision="abc123",
        )

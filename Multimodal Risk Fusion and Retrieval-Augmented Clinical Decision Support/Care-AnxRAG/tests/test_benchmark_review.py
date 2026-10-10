from __future__ import annotations

import csv
import json

import pytest

from care_anxrag.benchmark_review import (
    compile_adjudicated_benchmark,
    write_annotation_sheet,
)


def test_annotation_sheet_contains_blank_human_judgment_fields(tmp_path) -> None:
    path = tmp_path / "review.csv"
    write_annotation_sheet(path, split="test")

    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))

    assert rows
    assert all(row["split"] == "test" for row in rows)
    assert all(row["reviewer_1_id"] == "" for row in rows)
    assert all(row["reviewer_2_id"] == "" for row in rows)
    assert all(row["final_must_abstain"] == "" for row in rows)
    assert all(row["final_expects_conflict"] == "" for row in rows)
    assert all(row["adjudicated"] == "" for row in rows)


def test_compile_benchmark_requires_two_distinct_reviewers(runtime, tmp_path) -> None:
    sheet = tmp_path / "review.csv"
    write_annotation_sheet(sheet, split="test")

    rows = list(csv.DictReader(sheet.open(encoding="utf-8", newline="")))
    row = rows[0]
    row["reviewer_1_id"] = "reviewer-a"
    row["reviewer_2_id"] = "reviewer-a"
    row["final_must_abstain"] = "true"
    row["final_expects_conflict"] = "false"
    row["adjudicated"] = "true"

    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="two distinct reviewer IDs"):
        compile_adjudicated_benchmark(runtime, sheet, tmp_path / "out.jsonl")


def test_compile_benchmark_validates_active_evidence_and_exact_excerpt(
    runtime,
    project,
    tmp_path,
) -> None:
    from conftest import write_document

    excerpt = (
        "Cognitive behavioural therapy is discussed as an evidence-based "
        "psychological treatment option for generalized anxiety disorder."
    )
    write_document(
        project,
        "gad.md",
        external_id="gad-guideline",
        title="GAD guideline",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=(
            "Generalized anxiety disorder involves persistent excessive worry "
            "and can impair daily functioning. Assessment considers symptoms, "
            "duration, functional impact, and relevant clinical context. "
            + excerpt
            + " Ongoing follow-up should consider symptoms, functioning, "
            "treatment response, adverse effects, patient preference, and "
            "individual circumstances under qualified clinical care."
        ),
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert runtime.database.list_active_version_fingerprints()

    sheet = tmp_path / "review.csv"
    fields = [
        "id",
        "question",
        "stratum",
        "split",
        "intent",
        "anxiety_subtypes_json",
        "treatments_json",
        "population",
        "reviewer_1_id",
        "reviewer_2_id",
        "final_relevant_external_ids_json",
        "final_relevant_source_ids_json",
        "final_prohibited_external_ids_json",
        "final_prohibited_source_ids_json",
        "final_gold_evidence_excerpts_json",
        "final_prohibited_claims_json",
        "final_must_abstain",
        "final_expects_conflict",
        "adjudicated",
    ]
    row = {
        "id": "gad-cbt-reviewed",
        "question": "What evidence discusses CBT for GAD?",
        "stratum": "psychological_interventions",
        "split": "test",
        "intent": "treatment",
        "anxiety_subtypes_json": json.dumps(["generalized_anxiety_disorder"]),
        "treatments_json": json.dumps(["cognitive_behavioral_therapy"]),
        "population": "",
        "reviewer_1_id": "reviewer-a",
        "reviewer_2_id": "reviewer-b",
        "final_relevant_external_ids_json": json.dumps(["gad-guideline"]),
        "final_relevant_source_ids_json": json.dumps([]),
        "final_prohibited_external_ids_json": json.dumps([]),
        "final_prohibited_source_ids_json": json.dumps([]),
        "final_gold_evidence_excerpts_json": json.dumps([excerpt]),
        "final_prohibited_claims_json": json.dumps([]),
        "final_must_abstain": "false",
        "final_expects_conflict": "false",
        "adjudicated": "true",
    }
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    output = tmp_path / "benchmark.jsonl"
    items = compile_adjudicated_benchmark(runtime, sheet, output)

    assert len(items) == 1
    assert items[0].annotator_ids == ["reviewer-a", "reviewer-b"]
    assert items[0].adjudicated is True
    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["relevant_external_ids"] == ["gad-guideline"]


def test_compile_benchmark_rejects_non_active_external_id(runtime, tmp_path) -> None:
    sheet = tmp_path / "review.csv"
    fields = [
        "id",
        "question",
        "stratum",
        "split",
        "intent",
        "anxiety_subtypes_json",
        "treatments_json",
        "population",
        "reviewer_1_id",
        "reviewer_2_id",
        "final_relevant_external_ids_json",
        "final_relevant_source_ids_json",
        "final_prohibited_external_ids_json",
        "final_prohibited_source_ids_json",
        "final_gold_evidence_excerpts_json",
        "final_prohibited_claims_json",
        "final_must_abstain",
        "final_expects_conflict",
        "adjudicated",
    ]
    row = {
        field: "" for field in fields
    }
    row.update(
        {
            "id": "bad-ref",
            "question": "Synthetic reviewed question",
            "stratum": "test",
            "split": "test",
            "anxiety_subtypes_json": "[]",
            "treatments_json": "[]",
            "reviewer_1_id": "reviewer-a",
            "reviewer_2_id": "reviewer-b",
            "final_relevant_external_ids_json": '["missing-doc"]',
            "final_relevant_source_ids_json": "[]",
            "final_prohibited_external_ids_json": "[]",
            "final_prohibited_source_ids_json": "[]",
            "final_gold_evidence_excerpts_json": "[]",
            "final_prohibited_claims_json": "[]",
            "final_must_abstain": "false",
            "final_expects_conflict": "false",
            "adjudicated": "true",
        }
    )
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ValueError, match="non-active external IDs"):
        compile_adjudicated_benchmark(runtime, sheet, tmp_path / "out.jsonl")

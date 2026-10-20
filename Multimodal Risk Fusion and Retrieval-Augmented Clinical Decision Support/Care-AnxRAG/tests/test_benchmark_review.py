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


def test_review_package_has_development_candidates_and_empty_test_template(tmp_path) -> None:
    from care_anxrag import benchmark_review

    output = tmp_path / "package"
    assert hasattr(benchmark_review, "write_review_package")
    manifest = benchmark_review.write_review_package(output)
    development = list(csv.DictReader((output / "development.csv").open()))
    test = list(csv.DictReader((output / "test.csv").open()))
    assert len(development) == 17
    assert test == []
    assert all(row["split"] == "development" for row in development)
    human_fields = [key for key in development[0] if key.startswith(("reviewer_", "final_"))]
    human_fields.append("adjudicated")
    assert all(row[field] == "" for row in development for field in human_fields)
    assert manifest["status"] == "awaiting_human_review"
    assert len(manifest["strata"]) == 17
    assert json.loads((output / "manifest.json").read_text()) == manifest
    before = (output / "development.csv").read_bytes()
    with pytest.raises(ValueError, match="empty"):
        benchmark_review.write_review_package(output)
    assert (output / "development.csv").read_bytes() == before


def _write_split_fixture(path, *, split, item_id, question, adjudicated=True):
    """Synthetic metadata solely for split validation; never a research benchmark."""
    from care_anxrag.evaluation import BenchmarkItem

    item = BenchmarkItem(
        id=item_id, question=question, split=split, stratum="unit_test",
        annotator_ids=["unit-test-a", "unit-test-b"], adjudicated=adjudicated,
    )
    path.write_text(item.model_dump_json() + "\n", encoding="utf-8")


@pytest.mark.parametrize("failure", ["id", "question", "split", "review", "empty"])
def test_split_validation_rejects_leakage_and_unreviewed_items(tmp_path, failure) -> None:
    from care_anxrag import benchmark_review

    development = tmp_path / "development.jsonl"
    test = tmp_path / "test.jsonl"
    _write_split_fixture(development, split="development", item_id="dev", question="First question?")
    _write_split_fixture(
        test, split="development" if failure == "split" else "test",
        item_id="dev" if failure == "id" else "test",
        question="  FIRST   question?  " if failure == "question" else "Second question?",
        adjudicated=failure != "review",
    )
    if failure == "empty":
        test.write_text("")
    assert hasattr(benchmark_review, "validate_benchmark_splits")
    with pytest.raises(ValueError):
        benchmark_review.validate_benchmark_splits(development, test)


def test_split_validation_reports_counts_without_running_models(tmp_path) -> None:
    from care_anxrag import benchmark_review

    development = tmp_path / "development.jsonl"
    test = tmp_path / "test.jsonl"
    _write_split_fixture(development, split="development", item_id="dev", question="First question?")
    _write_split_fixture(test, split="test", item_id="test", question="Second question?")
    assert hasattr(benchmark_review, "validate_benchmark_splits")
    report = benchmark_review.validate_benchmark_splits(development, test)
    assert report["status"] == "passed"
    assert report["development"]["count"] == report["test"]["count"] == 1
    assert report["development"]["strata"] == {"unit_test": 1}


def test_split_validation_rejects_whitespace_padded_unspecified_stratum(tmp_path) -> None:
    from care_anxrag.benchmark_review import validate_benchmark_splits

    development = tmp_path / "development.jsonl"
    test = tmp_path / "test.jsonl"
    _write_split_fixture(development, split="development", item_id="dev", question="First question?")
    _write_split_fixture(test, split="test", item_id="test", question="Second question?")
    payload = json.loads(test.read_text())
    payload["stratum"] = " unspecified "
    test.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="stratum"):
        validate_benchmark_splits(development, test)

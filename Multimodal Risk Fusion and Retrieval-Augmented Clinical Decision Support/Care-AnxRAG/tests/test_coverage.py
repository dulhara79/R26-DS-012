from __future__ import annotations

from pathlib import Path

from care_anxrag.coverage import audit_corpus_coverage

from conftest import write_document


def _long_body(sentence: str) -> str:
    return (
        "# Clinical evidence\n"
        + sentence
        + " This evidence summary includes sufficient contextual detail for "
        + "controlled retrieval, versioning, and corpus coverage validation. "
        + "Clinical interpretation remains the responsibility of qualified reviewers. "
        + "The fixture is deliberately longer than the document-quality minimum so "
        + "coverage tests exercise clinical concept handling rather than length rejection."
    )


def test_corpus_coverage_counts_direct_subtype_treatment_support(
    runtime,
    project: Path,
) -> None:
    write_document(
        project,
        "gad-cbt.md",
        external_id="gad-cbt",
        title="CBT evidence for generalized anxiety disorder",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=_long_body(
            "Cognitive behavioural therapy was evaluated for generalized anxiety disorder."
        ),
    )
    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert summary.promoted == 1

    report = audit_corpus_coverage(runtime.database)

    row = next(
        item
        for item in report.combinations
        if item.subtype == "generalized_anxiety_disorder"
        and item.treatment == "cognitive_behavioral_therapy"
    )
    assert row.supporting_chunks >= 1
    assert row.supporting_documents == 1
    assert row.supporting_sources == 1
    assert report.active_chunks >= 1
    assert report.active_documents == 1


def test_corpus_coverage_does_not_use_treatment_topic_as_evidence(
    runtime,
    project: Path,
) -> None:
    write_document(
        project,
        "gad-mct.md",
        external_id="gad-mct",
        title="Metacognitive therapy for generalized anxiety disorder",
        topics=[
            "anxiety",
            "generalized_anxiety_disorder",
            "Cognitive Behavioral Therapy",
        ],
        body=_long_body(
            "Metacognitive therapy was evaluated for generalized anxiety disorder."
        ),
    )
    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert summary.promoted == 1

    report = audit_corpus_coverage(runtime.database)

    assert not any(
        item.subtype == "generalized_anxiety_disorder"
        and item.treatment == "cognitive_behavioral_therapy"
        for item in report.combinations
    )
    assert any(
        item.subtype == "generalized_anxiety_disorder"
        and item.treatment == "metacognitive_therapy"
        for item in report.combinations
    )


def test_corpus_coverage_ignores_superseded_evidence(
    runtime,
    project: Path,
) -> None:
    path = write_document(
        project,
        "gad-treatment.md",
        external_id="gad-treatment",
        title="Treatment evidence for generalized anxiety disorder",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=_long_body(
            "Cognitive behavioural therapy was evaluated for generalized anxiety disorder."
        ),
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "Cognitive behavioural therapy",
            "Metacognitive therapy",
        ),
        encoding="utf-8",
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)

    report = audit_corpus_coverage(runtime.database)

    assert not any(
        item.subtype == "generalized_anxiety_disorder"
        and item.treatment == "cognitive_behavioral_therapy"
        for item in report.combinations
    )
    assert any(
        item.subtype == "generalized_anxiety_disorder"
        and item.treatment == "metacognitive_therapy"
        for item in report.combinations
    )


def test_corpus_coverage_can_filter_requested_combination(
    runtime,
    project: Path,
) -> None:
    write_document(
        project,
        "panic-cbt.md",
        external_id="panic-cbt",
        title="CBT evidence for panic disorder",
        topics=["anxiety", "panic_disorder"],
        body=_long_body(
            "Cognitive behavioral therapy was evaluated for panic disorder."
        ),
    )
    runtime.ingestion.sync(source_ids=["test_core"], force=True)

    report = audit_corpus_coverage(
        runtime.database,
        subtype="generalized_anxiety_disorder",
        treatment="cognitive_behavioral_therapy",
    )

    assert report.requested_subtype == "generalized_anxiety_disorder"
    assert report.requested_treatment == "cognitive_behavioral_therapy"
    assert report.requested_supporting_chunks == 0



def test_coverage_cli_reports_requested_direct_support(
    runtime,
    project: Path,
) -> None:
    import json

    from typer.testing import CliRunner

    from care_anxrag.cli import app

    write_document(
        project,
        "gad-cbt-cli.md",
        external_id="gad-cbt-cli",
        title="CBT evidence for generalized anxiety disorder",
        topics=["anxiety", "generalized_anxiety_disorder"],
        body=_long_body(
            "Cognitive behavioural therapy was evaluated for generalized anxiety disorder."
        ),
    )
    summary = runtime.ingestion.sync(source_ids=["test_core"], force=True)
    assert summary.promoted == 1

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "coverage",
            "--project-root",
            str(project),
            "--subtype",
            "generalized_anxiety_disorder",
            "--treatment",
            "cognitive_behavioral_therapy",
        ],
        env={
            "CARE_HOME": str(runtime.settings.care_home),
            "CARE_DATABASE_PATH": str(runtime.settings.database_path),
            "CARE_VECTOR_PATH": str(runtime.settings.vector_path),
            "CARE_SOURCE_REGISTRY": str(runtime.settings.source_registry_path),
            "CARE_VECTOR_BACKEND": "sqlite",
            "CARE_EMBEDDING_PROVIDER": "hash",
            "CARE_EMBEDDING_MODEL": "hash",
            "CARE_EMBEDDING_DIMENSIONS": "256",
            "CARE_GENERATOR_PROVIDER": "extractive",
            "CARE_RERANKER_PROVIDER": "heuristic",
            "CARE_NLI_PROVIDER": "heuristic",
            "CARE_ALLOW_NETWORK_SYNC": "false",
        },
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["requested_subtype"] == "generalized_anxiety_disorder"
    assert payload["requested_treatment"] == "cognitive_behavioral_therapy"
    assert payload["requested_supporting_chunks"] >= 1

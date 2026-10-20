from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .coverage import audit_corpus_coverage
from .evaluation import BenchmarkItem, evaluate_ablation, load_benchmark
from .experiment_preflight import validate_final_experiment
from .reproducibility import build_experiment_snapshot
from .runtime import Runtime
from .statistics import mcnemar_exact, paired_bootstrap_difference


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_timings(runtime: Runtime, items: list[BenchmarkItem]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        answer = runtime.rag.answer(item.question, include_debug=True)
        rows.append(
            {
                "id": item.id,
                "answer_timings_ms": dict(answer.timings_ms),
                "retrieval_timings_ms": (
                    dict(answer.retrieval.timings_ms)
                    if answer.retrieval is not None
                    else {}
                ),
                "abstained": answer.abstained,
                "safety_level": answer.safety_level.value,
            }
        )

    stage_values: dict[str, list[float]] = {}
    for row in rows:
        for prefix, timings_key in (
            ("answer", "answer_timings_ms"),
            ("retrieval", "retrieval_timings_ms"),
        ):
            for stage, value in row[timings_key].items():
                stage_values.setdefault(f"{prefix}.{stage}", []).append(float(value))

    summary: dict[str, dict[str, float | int]] = {}
    for stage, values in sorted(stage_values.items()):
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
        summary[stage] = {
            "count": len(values),
            "mean_ms": sum(values) / len(values),
            "median_ms": ordered[len(ordered) // 2],
            "p95_ms": ordered[p95_index],
        }

    return {"count": len(rows), "per_item": rows, "stage_summary": summary}


_BOOTSTRAP_METRICS = (
    "recall_at_5",
    "precision_at_5",
    "reciprocal_rank",
    "ndcg_at_5",
    "gold_evidence_coverage",
    "active_version_accuracy",
    "stale_evidence_intrusion_rate",
)


def _paired_report_rows(left: dict[str, Any], right: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    right_by_id = {row["id"]: row for row in right["per_item"]}
    return [
        (row, right_by_id[row["id"]])
        for row in left["per_item"]
        if row["id"] in right_by_id
    ]


def _build_comparisons(ablation: dict[str, Any]) -> dict[str, Any]:
    reference_label = "CARE_full"
    reference = ablation[reference_label]
    comparisons: dict[str, Any] = {}

    for label, report in ablation.items():
        if label == reference_label:
            continue
        pairs = _paired_report_rows(report, reference)
        metrics: dict[str, Any] = {}
        for metric in _BOOTSTRAP_METRICS:
            evaluable = [
                (float(left[metric]), float(right[metric]))
                for left, right in pairs
                if left.get(metric) is not None and right.get(metric) is not None
            ]
            if evaluable:
                metrics[metric] = paired_bootstrap_difference(
                    [left for left, _ in evaluable],
                    [right for _, right in evaluable],
                )

        binary: dict[str, Any] = {}
        binary_pairs = {
            "abstention_correct": [
                (
                    left["predicted_abstain"] == left["expected_abstain"],
                    right["predicted_abstain"] == right["expected_abstain"],
                )
                for left, right in pairs
            ],
            "conflict_correct": [
                (
                    (left["conflict_score"] > 0.0) == left["expected_conflict"],
                    (right["conflict_score"] > 0.0) == right["expected_conflict"],
                )
                for left, right in pairs
            ],
            "citation_valid": [
                (bool(left["citation_valid"]), bool(right["citation_valid"]))
                for left, right in pairs
            ],
        }
        for name, outcomes in binary_pairs.items():
            if outcomes:
                binary[name] = mcnemar_exact(
                    [left for left, _ in outcomes],
                    [right for _, right in outcomes],
                )

        comparisons[label] = {
            "reference": reference_label,
            "paired_item_count": len(pairs),
            "bootstrap_metrics": metrics,
            "mcnemar_outcomes": binary,
        }

    return {
        "comparison_version": 1,
        "reference": reference_label,
        "bootstrap_difference_definition": "CARE_full minus baseline",
        "comparisons": comparisons,
    }


def _write_comparisons_csv(path: Path, comparisons: dict[str, Any]) -> None:
    fieldnames = [
        "baseline",
        "reference",
        "analysis",
        "metric",
        "n",
        "difference",
        "ci95_low",
        "ci95_high",
        "p_value",
        "left_only_correct",
        "right_only_correct",
        "discordant_pairs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for baseline, comparison in comparisons["comparisons"].items():
            for metric, result in comparison["bootstrap_metrics"].items():
                writer.writerow(
                    {
                        "baseline": baseline,
                        "reference": comparison["reference"],
                        "analysis": "paired_bootstrap",
                        "metric": metric,
                        "n": result["n"],
                        "difference": result["difference"],
                        "ci95_low": result["ci95_low"],
                        "ci95_high": result["ci95_high"],
                    }
                )
            for metric, result in comparison["mcnemar_outcomes"].items():
                writer.writerow(
                    {
                        "baseline": baseline,
                        "reference": comparison["reference"],
                        "analysis": "mcnemar_exact",
                        "metric": metric,
                        "n": result["n"],
                        "p_value": result["p_value"],
                        "left_only_correct": result["left_only_correct"],
                        "right_only_correct": result["right_only_correct"],
                        "discordant_pairs": result["discordant_pairs"],
                    }
                )


def run_experiment_bundle(
    runtime: Runtime,
    benchmark_path: Path,
    output_dir: Path,
    *,
    code_revision: str,
) -> dict[str, Any]:
    """Run the reproducible CARE-AnxRAG research bundle.

    The target directory is immutable-by-default. A pre-existing non-empty
    directory is rejected so a completed research run cannot be silently
    overwritten.
    """
    benchmark_path = benchmark_path.resolve()
    output_dir = output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Experiment output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_benchmark(benchmark_path)
    ablation_reports = evaluate_ablation(
        runtime.retriever,
        runtime.rag,
        items,
    )
    ablation = {
        label: report.as_dict()
        for label, report in ablation_reports.items()
    }
    comparisons = _build_comparisons(ablation)
    coverage = audit_corpus_coverage(runtime.database).as_dict()
    timings = _collect_timings(runtime, items)
    snapshot = build_experiment_snapshot(
        runtime,
        code_revision=code_revision,
        benchmark_path=benchmark_path,
    )

    payloads = {
        "ablation.json": ablation,
        "comparisons.json": comparisons,
        "coverage.json": coverage,
        "timings.json": timings,
        "snapshot.json": snapshot,
    }
    for filename, payload in payloads.items():
        _write_json(output_dir / filename, payload)
    _write_comparisons_csv(output_dir / "comparisons.csv", comparisons)

    artifact_names = [*payloads, "comparisons.csv"]
    artifacts = {
        filename: {
            "sha256": _sha256_file(output_dir / filename),
            "size_bytes": (output_dir / filename).stat().st_size,
        }
        for filename in sorted(artifact_names)
    }
    manifest = {
        "bundle_version": 1,
        "code_revision": code_revision,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": _sha256_file(benchmark_path),
        "item_count": len(items),
        "ablation_modes": list(ablation),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)

    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
    }



def run_final_experiment_bundle(
    runtime: Runtime,
    benchmark_path: Path,
    output_dir: Path,
    *,
    code_revision: str,
    corpus_freeze_path: Path,
) -> dict[str, Any]:
    """Run the locked final bundle only when freeze and adjudication gates pass."""
    preflight = validate_final_experiment(
        runtime,
        corpus_freeze_path=corpus_freeze_path,
        benchmark_path=benchmark_path,
        code_revision=code_revision,
    )
    result = run_experiment_bundle(
        runtime,
        benchmark_path,
        output_dir,
        code_revision=code_revision,
    )

    output_dir = output_dir.resolve()
    preflight_path = output_dir / "preflight.json"
    _write_json(preflight_path, preflight)

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_experiment"] = True
    manifest["corpus_freeze_path"] = str(Path(corpus_freeze_path).resolve())
    manifest["corpus_freeze_sha256"] = _sha256_file(Path(corpus_freeze_path))
    manifest["artifacts"]["preflight.json"] = {
        "sha256": _sha256_file(preflight_path),
        "size_bytes": preflight_path.stat().st_size,
    }
    _write_json(manifest_path, manifest)

    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
        "preflight": preflight,
    }

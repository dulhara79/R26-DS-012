from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import load_benchmark
from .reproducibility import build_experiment_snapshot
from .runtime import Runtime


def validate_final_experiment(
    runtime: Runtime,
    *,
    corpus_freeze_path: Path | str,
    benchmark_path: Path | str,
    code_revision: str,
) -> dict[str, Any]:
    freeze_path = Path(corpus_freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    if freeze.get("ready_to_freeze") is not True:
        raise ValueError("Corpus freeze is not marked ready_to_freeze=true")
    if freeze.get("code_revision") != code_revision:
        raise ValueError("Current code revision does not match corpus freeze")

    frozen_snapshot = freeze.get("snapshot")
    if not isinstance(frozen_snapshot, dict):
        raise ValueError("Corpus freeze is missing a reproducibility snapshot")

    current_snapshot = build_experiment_snapshot(
        runtime,
        code_revision=code_revision,
        benchmark_path=benchmark_path,
    )

    for key in ("source_registry_sha256", "models", "retrieval", "chunking"):
        if frozen_snapshot.get(key) != current_snapshot.get(key):
            raise ValueError(f"Current {key} does not match corpus freeze")

    frozen_corpus = frozen_snapshot.get("corpus", {})
    current_corpus = current_snapshot.get("corpus", {})
    if frozen_corpus != current_corpus:
        raise ValueError("Current active evidence corpus does not match corpus freeze")

    items = load_benchmark(benchmark_path)
    if not items:
        raise ValueError("Final benchmark contains no items")

    for item in items:
        reviewers = {
            reviewer.strip()
            for reviewer in item.annotator_ids
            if reviewer.strip()
        }
        if not item.adjudicated or len(reviewers) < 2:
            raise ValueError(
                f"Final benchmark item {item.id!r} must be adjudicated "
                "by at least two distinct reviewers"
            )

    return {
        "preflight_version": 1,
        "code_revision": code_revision,
        "corpus_freeze_path": str(freeze_path.resolve()),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "benchmark_item_count": len(items),
        "active_version_count": current_corpus.get("active_version_count", 0),
        "status": "ready",
    }

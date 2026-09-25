from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runtime import Runtime
from .util import sha256_text, utc_now


def _file_fingerprint(path: Path) -> dict[str, str] | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_text(text),
    }


def build_experiment_snapshot(
    runtime: Runtime,
    *,
    code_revision: str | None,
    benchmark_path: Path | str | None = None,
) -> dict[str, Any]:
    settings = runtime.settings
    chunking = runtime.ingestion.chunker.config

    active_versions = runtime.database.list_active_version_fingerprints()

    benchmark = None
    if benchmark_path is not None:
        benchmark_file = Path(benchmark_path)
        benchmark = _file_fingerprint(benchmark_file)
        if benchmark is None:
            raise FileNotFoundError(
                f"Benchmark file does not exist: {benchmark_file}"
            )

    dependency_manifest = _file_fingerprint(
        settings.project_root / "pyproject.toml"
    )

    return {
        "snapshot_version": 1,
        "created_at": utc_now().isoformat(),
        "code_revision": code_revision,
        "database_schema_version": runtime.database.get_metadata(
            "schema_version"
        ),
        "source_registry_sha256": sha256_text(
            settings.source_registry_path.read_text(
                encoding="utf-8"
            )
        ),
        "benchmark": benchmark,
        "dependency_manifest": dependency_manifest,
        "corpus": {
            "active_version_count": len(active_versions),
            "active_versions": active_versions,
        },
        "models": {
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "embedding_runtime_id": runtime.embedder.model_id,
            "embedding_stored_id": runtime.database.get_embedding_identity(),
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
            "nli_provider": settings.nli_provider,
            "nli_model": settings.nli_model,
            "answer_provider": (
                "extractive"
                if settings.generator_provider in {"extractive", "rule"}
                else settings.generator_provider
            ),
            "configured_answer_provider": settings.generator_provider,
            "answer_policy": "extractive_only",
        },
        "retrieval": {
            "dense_candidates": settings.dense_candidates,
            "lexical_candidates": settings.lexical_candidates,
            "fused_candidates": settings.fused_candidates,
            "rerank_candidates": settings.rerank_candidates,
            "final_context_chunks": settings.final_context_chunks,
            "rrf_k": settings.rrf_k,
            "minimum_care_score": settings.minimum_care_score,
            "minimum_relevance_score": settings.minimum_relevance_score,
            "minimum_confidence": settings.minimum_confidence,
            "contradiction_threshold": settings.contradiction_threshold,
            "unresolved_conflict_threshold": (
                settings.unresolved_conflict_threshold
            ),
            "dominance_margin": settings.dominance_margin,
            "grounding_entailment_threshold": (
                settings.grounding_entailment_threshold
            ),
            "min_distinct_sources": settings.min_distinct_sources,
            "clinical_half_life_days": settings.clinical_half_life_days,
            "research_half_life_days": settings.research_half_life_days,
            "weights": asdict(settings.weights),
        },
        "chunking": {
            "max_words": chunking.max_words,
            "overlap_words": chunking.overlap_words,
            "min_words": chunking.min_words,
        },
    }

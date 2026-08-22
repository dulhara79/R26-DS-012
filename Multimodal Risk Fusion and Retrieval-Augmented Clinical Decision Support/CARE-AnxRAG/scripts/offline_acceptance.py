#!/usr/bin/env python3
"""Deterministic end-to-end acceptance test for CARE-AnxRAG."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from care_anxrag.config import Settings
from care_anxrag.evaluation import evaluate, load_benchmark
from care_anxrag.models import SafetyLevel
from care_anxrag.runtime import build_runtime


def run(project_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="care-anxrag-acceptance-") as temporary:
        state = Path(temporary)
        settings = Settings.from_env(
            project_root=project_root,
            environ={
                "CARE_HOME": str(state),
                "CARE_DATABASE_PATH": str(state / "care.sqlite3"),
                "CARE_VECTOR_PATH": str(state / "vectors"),
                "CARE_SOURCE_REGISTRY": str(
                    project_root / "examples" / "local_corpus" / "sources.yaml"
                ),
                "CARE_VECTOR_BACKEND": "sqlite",
                "CARE_EMBEDDING_PROVIDER": "hash",
                "CARE_EMBEDDING_MODEL": "hash",
                "CARE_EMBEDDING_DIMENSIONS": "256",
                "CARE_GENERATOR_PROVIDER": "rule",
                "CARE_RERANKER_PROVIDER": "heuristic",
                "CARE_NLI_PROVIDER": "heuristic",
                "CARE_ALLOW_NETWORK_SYNC": "false",
                "CARE_AUTO_PROMOTE_RESEARCH": "false",
                "CARE_MINIMUM_CARE_SCORE": "0.28",
                "CARE_MINIMUM_RELEVANCE_SCORE": "0.20",
                "CARE_MINIMUM_CONFIDENCE": "0.35",
            },
        )
        runtime = build_runtime(settings)
        sync = runtime.ingestion.sync(source_ids=["synthetic_demo"], force=True)
        if sync.failed or sync.promoted != 1:
            raise AssertionError(f"Synthetic ingestion failed: {sync.model_dump(mode='json')}")

        answerable_question = (
            "What evidence-based psychological intervention is discussed for panic disorder?"
        )
        answerable = runtime.rag.answer(answerable_question)
        if answerable.abstained:
            raise AssertionError(f"Answerable query abstained: {answerable.abstention_reason}")
        if not answerable.citations:
            raise AssertionError("Answerable query returned no citations")
        if not any(term in answerable.answer.lower() for term in ("cognitive", "exposure")):
            raise AssertionError("Deterministic answer did not select the query-relevant sentence")
        if any(citation.source_id != "synthetic_demo" for citation in answerable.citations):
            raise AssertionError("Acceptance answer cited an unexpected source")

        out_of_domain = runtime.rag.answer("How do I repair a diesel fuel injector?")
        if not out_of_domain.abstained or out_of_domain.citations:
            raise AssertionError("Out-of-domain query was not safely rejected")

        crisis = runtime.rag.answer("I want to kill myself tonight")
        if crisis.safety_level != SafetyLevel.CRISIS or not crisis.abstained:
            raise AssertionError("Crisis query did not bypass normal RAG")

        benchmark = load_benchmark(project_root / "data" / "benchmark" / "example.jsonl")
        evaluation = evaluate(runtime.retriever, runtime.rag, benchmark)
        if evaluation.abstention_accuracy != 1.0:
            raise AssertionError("Acceptance benchmark abstention accuracy was not 1.0")
        if evaluation.citation_validity != 1.0:
            raise AssertionError("Acceptance benchmark citation validity was not 1.0")

        reconciliation = runtime.ingestion.reconcile_active_vectors(batch_size=2)
        if reconciliation["active_chunks"] != reconciliation["indexed"]:
            raise AssertionError("Vector reconciliation did not index every active chunk")
        if runtime.database.integrity_check() != "ok":
            raise AssertionError("SQLite integrity check failed")
        if runtime.database.pending_outbox():
            raise AssertionError("Vector outbox is not empty after acceptance run")

        return {
            "status": "passed",
            "sync": sync.model_dump(mode="json"),
            "database_integrity": runtime.database.integrity_check(),
            "clinical_vector_count": runtime.vector_store.count(settings.clinical_collection),
            "research_vector_count": runtime.vector_store.count(settings.research_collection),
            "answerable": {
                "abstained": answerable.abstained,
                "citation_count": len(answerable.citations),
                "confidence": answerable.confidence,
            },
            "out_of_domain": {
                "abstained": out_of_domain.abstained,
                "reason": out_of_domain.abstention_reason,
            },
            "crisis": {
                "abstained": crisis.abstained,
                "safety_level": crisis.safety_level.value,
            },
            "evaluation": evaluation.as_dict(),
            "reconciliation": reconciliation,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.project_root.resolve())
    rendered = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

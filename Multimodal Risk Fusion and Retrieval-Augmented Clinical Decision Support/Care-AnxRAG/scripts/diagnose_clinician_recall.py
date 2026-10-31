#!/usr/bin/env python3
"""Read-only diagnosis of question-specific evidence recall for clinician viva.

Runs the real retriever and existing sentence selector; never invents labels,
changes corpus entries, bypasses evidence rules, or promotes staged versions.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

QUESTIONS = (
    ("P01", "What treatments are supported for generalized anxiety disorder?"),
    ("P02", "What does the evidence say about CBT for social anxiety disorder?"),
    ("P03", "Compare exposure therapy and CBT for panic disorder."),
    ("P04", "What recent systematic reviews address anxiety-disorder treatment?"),
    ("P05", "What do authoritative sources say about medication for anxiety?"),
    ("P06", "What evidence contradicts an intervention recommendation for GAD?"),
    ("P07", "How is panic disorder diagnosed?"),
)


def inspect(retrieval, *, max_hits: int = 12) -> dict:
    from care_anxrag.answer_selection import select_answer_sentence
    records = []
    for rank, hit in enumerate(retrieval.hits[:max_hits], 1):
        eligible = select_answer_sentence(hit, retrieval.query_analysis)
        records.append({
            "rank": rank,
            "document_id": hit.chunk.document_id,
            "chunk_id": hit.chunk.chunk_id,
            "title": hit.chunk.title,
            "url": hit.chunk.url,
            "published_at": hit.chunk.published_at.isoformat()
                if hit.chunk.published_at else None,
            "evidence_level": hit.chunk.evidence_level.value,
            "care_score": round(hit.care_score, 4),
            "relevance_score": round(hit.relevance_score, 4),
            "excluded_due_to_conflict": hit.excluded_due_to_conflict,
            "eligible_source_sentence": eligible,
            "sample_source_text": hit.chunk.text[:600],
        })
    return {
        "retrieval_abstained": retrieval.should_abstain,
        "retrieval_abstention_reason": retrieval.abstention_reason,
        "retrieval_confidence": round(retrieval.confidence, 4),
        "intent": retrieval.query_analysis.intent.value,
        "requested_subtypes": retrieval.query_analysis.anxiety_subtypes,
        "requested_treatments": retrieval.query_analysis.treatments,
        "eligible_hit_count": sum(bool(hit["eligible_source_sentence"]) for hit in records),
        "examined_hits": len(records),
        "hits": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, choices=range(1, 8), metavar="1..7", default=7)
    args = parser.parse_args()
    from care_anxrag.config import Settings
    from care_anxrag.runtime import build_runtime

    root = args.project_root.resolve()
    settings = Settings.from_env(project_root=root)
    runtime = build_runtime(settings)
    started = datetime.now(UTC)
    result = {
        "created_at": started.isoformat(),
        "mode": "read_only_retrieval_diagnostic",
        "review_status": "not_independently_adjudicated",
        "cases": [],
    }
    for case_id, question in QUESTIONS[:args.limit]:
        print(f"Inspecting {case_id}: {question}", flush=True)
        try:
            retrieval = runtime.retriever.retrieve(question)
            report = inspect(retrieval)
            report.update({"case_id": case_id, "question": question})
            print(
                f"  ELIGIBLE={report['eligible_hit_count']}/"
                f"{report['examined_hits']} RETRIEVAL_ABSTAINED="
                f"{report['retrieval_abstained']}",
                flush=True,
            )
        except Exception as exc:
            report = {
                "case_id": case_id, "question": question,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["cases"].append(report)
    output = (
        root / "artifacts" / "final-viva" / "clinician-recall" /
        started.strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    target = output / "diagnostic.json"
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"DIAGNOSTIC_FILE={target}", flush=True)
    print("CLINICAL_VALIDATION=NOT_PERFORMED", flush=True)
    return 1 if any("error" in row for row in result["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())

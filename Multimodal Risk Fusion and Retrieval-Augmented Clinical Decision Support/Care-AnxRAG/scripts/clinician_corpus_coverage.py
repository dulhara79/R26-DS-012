#!/usr/bin/env python3
"""Read-only source-coverage audit of active CARE-AnxRAG literature.

All counts are keyword heuristics, NOT validated gold labels or evidence of
clinical treatment effectiveness. This script does not ingest or promote data.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import sqlite3

TOPICS = {
    "P02_social_anxiety_CBT": (
        r"\b(?:social anxiety|social phobia)\b",
        r"\b(?:CBT|cognitive[- ]behaviou?ral therap(?:y|ies))\b",
    ),
    "P03_panic_exposure_CBT": (
        r"\bpanic disorder\b",
        r"\b(?:exposure[- ](?:based )?therap(?:y|ies)|exposure[- ]based)\b",
        r"\b(?:CBT|cognitive[- ]behaviou?ral therap(?:y|ies))\b",
    ),
    "P06_GAD_negative_treatment_finding": (
        r"\b(?:generaliz(?:ed|ised) anxiety disorder|GAD)\b",
        r"\b(?:CBT|therap(?:y|ies)|pharmacotherap(?:y|ies)|SSRIs?|treatment)\b",
        r"\b(?:not effective|no (?:significant )?(?:difference|effect|benefit)|"
        r"did not reduc|inconclusive|mixed results|contradictory findings)\b",
    ),
    "P07_panic_diagnostic_assessment": (
        r"\bpanic disorder\b",
        r"\b(?:diagnos(?:is|tic|ed)|clinical assessment|DSM[- ]?[45]|"
        r"recurrent unexpected panic attacks)\b",
    ),
}
TITLE_ANXIETY = re.compile(
    r"\b(?:anxiety|panic disorder|social phobia|agoraphobia)\b", re.I,
)
TITLE_TREATMENT = re.compile(
    r"\b(?:treat|therap|pharmac|CBT|exposure|intervention|stimulation)\w*", re.I,
)


def summarize(rows: list[dict], *, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    report: dict = {
        "active_document_versions": len(rows),
        "disclaimer": (
            "Keyword coverage only; matching abstracts may still lack a "
            "claim-grounded answer or be inapplicable to the target population."
        ),
        "by_source": {},
        "topics": {},
        "recent_treatment_reviews": [],
    }
    for row in rows:
        source = row["source_id"]
        report["by_source"][source] = report["by_source"].get(source, 0) + 1
    for topic, expressions in TOPICS.items():
        matches = []
        for row in rows:
            text = f"{row['title']}\n{row['text']}"
            if all(re.search(expr, text, flags=re.I) for expr in expressions):
                matches.append({
                    "title": row["title"], "source_id": row["source_id"],
                    "publication_date": row["published_at"],
                    "url": row["url"],
                })
        report["topics"][topic] = {
            "heuristic_matching_document_count": len(matches),
            "candidate_titles": matches[:12],
        }

    cutoff = (now - timedelta(days=3 * 365)).date()
    for row in rows:
        if row["evidence_level"] not in {"systematic_review", "meta_analysis"}:
            continue
        published = row["published_at"]
        if not published:
            continue
        try:
            date = datetime.fromisoformat(published.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        if not cutoff <= date <= now.date():
            continue
        title = row["title"]
        if TITLE_ANXIETY.search(title) and TITLE_TREATMENT.search(title):
            report["recent_treatment_reviews"].append({
                "title": title,
                "publication_date": published,
                "source_id": row["source_id"],
                "url": row["url"],
            })
    report["recent_treatment_reviews"].sort(
        key=lambda row: row["publication_date"], reverse=True
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    from care_anxrag.config import Settings
    settings = Settings.from_env(project_root=args.project_root.resolve())
    path = settings.database_path
    if not path.is_file():
        parser.error(f"Missing existing evidence database: {path}")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(record)
            for record in connection.execute(
                "SELECT source_id, title, text, published_at, evidence_level, url "
                "FROM document_versions WHERE status='active'"
            )
        ]
    result = summarize(rows)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        args.project_root.resolve() / "artifacts" / "final-viva" /
        "coverage-audit" / stamp / "report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"ACTIVE_DOCUMENTS={result['active_document_versions']}")
    for name, outcome in result["topics"].items():
        print(f"{name}={outcome['heuristic_matching_document_count']}")
    print(f"RECENT_TREATMENT_REVIEWS={len(result['recent_treatment_reviews'])}")
    print(f"REPORT={output}")
    print("CLINICAL_RELEVANCE_NOT_ADJUDICATED=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

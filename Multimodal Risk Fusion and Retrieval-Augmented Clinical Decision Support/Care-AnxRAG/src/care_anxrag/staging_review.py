from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .runtime import Runtime


_FIELDS = [
    "version_id",
    "source_id",
    "external_id",
    "title",
    "url",
    "evidence_level",
    "topics_json",
    "historical_hint_role",
    "historical_hint_evidence_role",
    "historical_hint_score",
    "decision",
    "decision_reason",
    "reviewer_id",
]


def _load_hints(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Historical hint file must contain a JSON list")

    hints: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Historical hint entries must be JSON objects")
        version_id = str(item.get("version_id", "")).strip()
        if not version_id:
            raise ValueError("Historical hint entry is missing version_id")
        if version_id in hints:
            raise ValueError(f"Duplicate historical hint version_id: {version_id}")
        hints[version_id] = item
    return hints


def write_staging_review_package(
    runtime: Runtime,
    output: Path,
    *,
    hints_path: Path | None = None,
) -> dict[str, Any]:
    """Write a human-review CSV for all currently staged evidence.

    Historical plans are copied only into hint columns. They never populate the
    reviewer decision fields and therefore cannot authorize a mutation.
    """
    if output.exists() and output.stat().st_size > 0:
        raise ValueError(f"Review package already exists and is non-empty: {output}")

    hints = _load_hints(hints_path)
    versions = runtime.database.list_staging_versions()
    rows: list[dict[str, str]] = []
    hinted = 0

    for version in versions:
        hint = hints.get(version.version_id)
        if hint is not None and str(hint.get("title", "")) != version.title:
            hint = None
        if hint is not None:
            hinted += 1

        rows.append(
            {
                "version_id": version.version_id,
                "source_id": version.source_id,
                "external_id": version.external_id,
                "title": version.title,
                "url": version.url,
                "evidence_level": version.evidence_level.value,
                "topics_json": json.dumps(version.topics, ensure_ascii=False),
                "historical_hint_role": (
                    str(hint.get("role", "")) if hint is not None else ""
                ),
                "historical_hint_evidence_role": (
                    str(hint.get("evidence_role", "")) if hint is not None else ""
                ),
                "historical_hint_score": (
                    str(hint.get("score", "")) if hint is not None else ""
                ),
                "decision": "",
                "decision_reason": "",
                "reviewer_id": "",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "output": str(output),
        "count": len(rows),
        "hinted": hinted,
        "decisions_prefilled": 0,
    }


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Review CSV is missing a header")
        missing = [field for field in _FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                "Review CSV is missing required columns: " + ", ".join(missing)
            )
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def apply_staging_review(
    runtime: Runtime,
    review_path: Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate a completed staging review and optionally apply its decisions."""
    rows = _read_review_rows(review_path)
    staging = runtime.database.list_staging_versions()
    staging_by_id = {version.version_id: version for version in staging}

    if not rows and staging:
        raise ValueError("Review CSV contains no decision rows")

    seen: set[str] = set()
    actions: list[tuple[str, str, str]] = []

    for line_number, row in enumerate(rows, start=2):
        version_id = row["version_id"]
        if not version_id:
            raise ValueError(f"Row {line_number} is missing version_id")
        if version_id in seen:
            raise ValueError(f"Duplicate version_id in review CSV: {version_id}")
        seen.add(version_id)

        version = staging_by_id.get(version_id)
        if version is None:
            raise ValueError(
                f"Row {line_number} version_id is not currently staging: {version_id}"
            )

        if row["title"] != version.title:
            raise ValueError(
                f"Row {line_number} title does not match current staging version"
            )
        if row["source_id"] != version.source_id:
            raise ValueError(
                f"Row {line_number} source_id does not match current staging version"
            )
        if row["external_id"] != version.external_id:
            raise ValueError(
                f"Row {line_number} external_id does not match current staging version"
            )

        decision = row["decision"].lower()
        if decision not in {"approve", "reject"}:
            raise ValueError(
                f"Row {line_number} decision must be 'approve' or 'reject'"
            )
        if not row["decision_reason"]:
            raise ValueError(f"Row {line_number} decision_reason is required")
        if not row["reviewer_id"]:
            raise ValueError(f"Row {line_number} reviewer_id is required")

        actions.append((version_id, decision, row["decision_reason"]))

    missing_ids = sorted(set(staging_by_id) - seen)
    if missing_ids:
        raise ValueError(
            "Review CSV does not cover every current staging version; "
            f"missing {len(missing_ids)} version(s)"
        )

    approved = sum(decision == "approve" for _, decision, _ in actions)
    rejected = sum(decision == "reject" for _, decision, _ in actions)

    if apply:
        for version_id, decision, reason in actions:
            if decision == "approve":
                runtime.ingestion.approve(version_id)
            else:
                runtime.ingestion.reject(version_id, reason)

    return {
        "review_path": str(review_path),
        "validated": len(actions),
        "approved": approved,
        "rejected": rejected,
        "applied": len(actions) if apply else 0,
        "dry_run": not apply,
    }

from __future__ import annotations

from pathlib import Path
from typing import Any

from .coverage import audit_corpus_coverage
from .reproducibility import build_experiment_snapshot
from .runtime import Runtime
from .util import utc_now


def build_corpus_freeze(
    runtime: Runtime,
    *,
    code_revision: str,
    benchmark_path: Path | str | None = None,
) -> dict[str, Any]:
    """Audit and fingerprint the current corpus without inventing research evidence."""
    integrity = runtime.database.integrity_check()
    health = runtime.health()
    staging = runtime.database.list_staging_versions()
    open_alerts = runtime.database.list_evidence_alerts(open_only=True)

    reconciliation: dict[str, Any] | None = None
    reconciliation_error: str | None = None
    try:
        reconciliation = runtime.ingestion.reconcile_active_vectors()
    except Exception as exc:
        reconciliation_error = str(exc)

    coverage = audit_corpus_coverage(runtime.database).as_dict()
    snapshot = build_experiment_snapshot(
        runtime,
        code_revision=code_revision,
        benchmark_path=benchmark_path,
    )

    blockers: list[str] = []
    if integrity != "ok":
        blockers.append(f"database_integrity={integrity}")
    if health.status != "ok":
        blockers.append(f"runtime_health={health.status}")
    if staging:
        blockers.append(f"unresolved_staging_versions={len(staging)}")
    if open_alerts:
        blockers.append(f"open_evidence_alerts={len(open_alerts)}")
    if reconciliation_error is not None:
        blockers.append("vector_reconciliation_failed")

    return {
        "freeze_version": 1,
        "created_at": utc_now().isoformat(),
        "code_revision": code_revision,
        "ready_to_freeze": not blockers,
        "blockers": blockers,
        "database_integrity": integrity,
        "health": health.model_dump(mode="json"),
        "staging_count": len(staging),
        "staging_versions": staging,
        "open_evidence_alert_count": len(open_alerts),
        "open_evidence_alerts": open_alerts,
        "vector_reconciliation": reconciliation,
        "vector_reconciliation_error": reconciliation_error,
        "coverage": coverage,
        "snapshot": snapshot,
    }

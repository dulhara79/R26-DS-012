#!/usr/bin/env python3
"""PubMed backfill for a NONCLINICAL psychiatrist-evidence research demonstration.

Preview: python scripts/pubmed_backfill.py preview
Stage:   python scripts/pubmed_backfill.py sync --days 30 --cap 500
Demo:    python scripts/pubmed_backfill.py sync --days 30 --cap 500 --machine-screened-demo --confirm-research-only
No clinician review is implied by machine screening or activation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import httpx
import yaml

WINDOWS = (7, 14, 30, 45, 60, 90, 120, 180)
NCBI_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def read_profile(path: Path) -> dict:
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(p, dict) or not str(p.get("query", "")).strip():
        raise ValueError("PubMed profile must provide a nonempty query")
    return p


def effective_registry(path: Path, profile: dict, machine_screened: bool) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError("Malformed source registry")
    result = deepcopy(raw)
    matches = [s for s in result["sources"] if s.get("id") == "pubmed_anxiety"]
    if len(matches) != 1 or matches[0].get("connector") != "pubmed":
        raise ValueError("Expected exactly one PubMed connector")
    source = matches[0]
    if not source.get("enabled", True) or not source.get("publish_to_rag", True):
        raise ValueError("pubmed_anxiety must be enabled and publishable")
    source["auto_promote"] = machine_screened
    keys = {"query", "initial_lookback_days", "overlap_days",
            "max_records_per_sync", "search_batch_size", "fetch_batch_size"}
    source["settings"].update({key: value for key, value in profile.items() if key in keys})
    return result


def active_documents(db: Path) -> int:
    if not db.exists():
        raise FileNotFoundError(f"Missing CARE SQLite ledger: {db}")
    with sqlite3.connect(db) as conn:
        return int(conn.execute(
            "SELECT COUNT(DISTINCT document_id) FROM document_versions "
            "WHERE status='active'"
        ).fetchone()[0])


def candidates(query: str, days: int, email: str, user_agent: str) -> int:
    now = datetime.now(UTC)
    params = {
        "db": "pubmed", "term": query, "datetype": "mdat",
        "mindate": (now - timedelta(days=days)).strftime("%Y/%m/%d"),
        "maxdate": now.strftime("%Y/%m/%d"), "retmode": "json",
        "retmax": "0", "tool": "CARE-AnxRAG", "email": email,
    }
    key = os.environ.get("NCBI_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    reply = httpx.get(NCBI_URL, params=params, timeout=60,
                      follow_redirects=True, headers={"User-Agent": user_agent})
    reply.raise_for_status()
    return int(reply.json()["esearchresult"]["count"])


def backup_database(db: Path, destination: Path) -> None:
    with sqlite3.connect(db) as original, sqlite3.connect(destination) as backup:
        original.backup(backup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("preview")
    sync = sub.add_parser("sync")
    sync.add_argument("--days", type=int, required=True)
    sync.add_argument("--cap", type=int, default=500)
    sync.add_argument("--machine-screened-demo", action="store_true")
    sync.add_argument("--confirm-research-only", action="store_true")
    args = parser.parse_args()

    from care_anxrag.config import Settings
    root = args.project_root.resolve()
    settings = Settings.from_env(project_root=root)
    if not settings.database_path.exists():
        parser.error("Existing database required; refusing to initialize an empty corpus")
    profile = read_profile(root / "config" / "pubmed_psychiatrist_profile.yaml")
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if "@" not in email or email.endswith("example.org"):
        parser.error("Set a real NCBI_EMAIL privately in .env or your shell")
    before = active_documents(settings.database_path)
    print(f"ACTIVE_DOCUMENTS={before}; ADDITIONAL_TO_100={max(0, 100-before)}", flush=True)

    if args.action == "preview":
        for days in WINDOWS:
            n = candidates(profile["query"], days, email, settings.source_user_agent)
            print(f"WINDOW_DAYS={days} MATCHES={n}", flush=True)
            time.sleep(0.4)
        return 0

    if not 1 <= args.days <= 3650 or not 1 <= args.cap <= 3000:
        parser.error("Valid --days: 1..3650; --cap: 1..3000")
    if args.machine_screened_demo != args.confirm_research_only:
        parser.error("Activation requires BOTH --machine-screened-demo and --confirm-research-only")
    if not settings.allow_network_sync:
        parser.error("Enable CARE_ALLOW_NETWORK_SYNC in the operator environment")
    n = candidates(profile["query"], args.days, email, settings.source_user_agent)
    if n == 0 or n > args.cap:
        parser.error(f"Candidates={n}; requires nonzero count <= cap={args.cap}")

    data = effective_registry(settings.source_registry_path, profile, args.machine_screened_demo)
    source = next(s for s in data["sources"] if s["id"] == "pubmed_anxiety")
    source["settings"]["initial_lookback_days"] = args.days
    source["settings"]["max_records_per_sync"] = args.cap
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "artifacts" / "final-viva" / "pubmed-backfill" / stamp
    out.mkdir(parents=True, exist_ok=False)
    backup_database(settings.database_path, out / "before.sqlite3")
    temporary_registry = out / "effective_registry.yaml"
    temporary_registry.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    report = {
        "screening": "automatic_non_adjudicated",
        "mode": "machine-screened-research-demo" if args.machine_screened_demo else "staging-only",
        "human_reviewed": False, "active_before": before,
        "candidates": n, "days": args.days, "cap": args.cap,
        "chroma_in_sqlite_backup": False,
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=False,
        ).stdout.strip(),
    }
    os.environ["CARE_SOURCE_REGISTRY"] = str(temporary_registry)
    os.environ["CARE_AUTO_PROMOTE_RESEARCH"] = "true" if args.machine_screened_demo else "false"
    failed = False
    try:
        from care_anxrag.runtime import build_runtime
        runtime = build_runtime(Settings.from_env(project_root=root))
        summary = runtime.ingestion.sync(source_ids=["pubmed_anxiety"], force=True)
        report["sync"] = summary.model_dump(mode="json")
        report["health"] = runtime.health().model_dump(mode="json")
        report["active_after"] = active_documents(settings.database_path)
        failed = summary.failed > 0 or report["health"]["status"] != "ok"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"AUDIT_FILE={out / 'report.json'}", flush=True)
        print(f"ACTIVE_DOCUMENTS_AFTER={report.get('active_after', 'unknown')}", flush=True)
        print(f"HUMAN_REVIEWED=false; CORPUS_FREEZE_NOT_ESTABLISHED", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

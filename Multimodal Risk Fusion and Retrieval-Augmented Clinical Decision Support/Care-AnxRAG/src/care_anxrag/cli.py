from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from .config import Settings
from .evaluation import evaluate as run_evaluation
from .evaluation import load_benchmark
from .logging_utils import configure_logging
from .runtime import build_runtime
from .scaffold import scaffold_project
from .util import redact_sensitive_settings, utc_now


app = typer.Typer(
    name="care-anxrag",
    help="CARE-AnxRAG research pipeline and service commands.",
    no_args_is_help=True,
)


def _json_default(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _json(value: object) -> str:
    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        default=_json_default,
    )


def _runtime(project_root: Path | None = None):
    settings = Settings.from_env(project_root=project_root)
    return build_runtime(settings)


@app.command("init")
def initialize(
    project_root: Annotated[Path | None, typer.Option(help="Project root containing config/sources.yaml")] = None,
) -> None:
    """Scaffold operator files, then initialize the database and vector collections."""
    root = (project_root or Path.cwd()).resolve()
    scaffold = scaffold_project(root)
    runtime = _runtime(root)
    typer.echo(_json({"scaffold": scaffold, "health": runtime.health()}))


@app.command()
def sync(
    source: Annotated[list[str] | None, typer.Option("--source", "-s", help="Source ID; repeatable")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Fetch and validate without ingesting content or advancing source state"
        ),
    ] = False,
    force: Annotated[bool, typer.Option(help="Ignore the last-success cursor")] = False,
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    result = runtime.ingestion.sync(source_ids=source, dry_run=dry_run, force=force)
    typer.echo(_json(result))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Anxiety information question")],
    debug: Annotated[bool, typer.Option(help="Include retrieval internals")] = False,
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    typer.echo(_json(runtime.rag.answer(question, include_debug=debug)))


@app.command()
def retrieve(
    question: Annotated[str, typer.Argument()],
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    typer.echo(_json(runtime.retriever.retrieve(question)))


@app.command()
def stats(project_root: Annotated[Path | None, typer.Option()] = None) -> None:
    runtime = _runtime(project_root)
    typer.echo(_json(runtime.database.stats()))


@app.command()
def sources(project_root: Annotated[Path | None, typer.Option()] = None) -> None:
    runtime = _runtime(project_root)
    values = []
    for source in runtime.sources:
        source_payload = source.model_dump(mode="json")
        source_payload["settings"] = redact_sensitive_settings(source.settings)
        values.append(
            {
                **source_payload,
                "state": runtime.database.get_source_state(source.id).model_dump(mode="json"),
            }
        )
    typer.echo(_json(values))


@app.command()
def staging(project_root: Annotated[Path | None, typer.Option()] = None) -> None:
    runtime = _runtime(project_root)
    typer.echo(_json(runtime.database.list_staging_versions()))


@app.command()
def approve(
    version_id: Annotated[str, typer.Argument()],
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    runtime.ingestion.approve(version_id)
    typer.echo(_json({"version_id": version_id, "status": "active"}))


@app.command()
def reject(
    version_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option(prompt=True)],
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    runtime.ingestion.reject(version_id, reason)
    typer.echo(_json({"version_id": version_id, "status": "rejected"}))


@app.command()
def withdraw(
    version_id: Annotated[str, typer.Argument(help="Any version ID for the document")],
    reason: Annotated[str, typer.Option(prompt=True)],
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    withdrawn_version_id = runtime.ingestion.withdraw(version_id, reason)
    typer.echo(
        _json(
            {
                "requested_version_id": version_id,
                "withdrawn_version_id": withdrawn_version_id,
                "status": "withdrawn",
            }
        )
    )


@app.command()
def reconcile(
    reset_embedding_index: Annotated[
        bool,
        typer.Option(
            help="Delete and rebuild all vectors while rebinding the index to the configured embedding model"
        ),
    ] = False,
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    typer.echo(
        _json(
            runtime.ingestion.reconcile_active_vectors(
                reset_embedding_identity=reset_embedding_index
            )
        )
    )


@app.command()
def evaluate(
    benchmark: Annotated[Path, typer.Argument(help="Benchmark JSONL file")],
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    runtime = _runtime(project_root)
    report = run_evaluation(runtime.retriever, runtime.rag, load_benchmark(benchmark))
    typer.echo(_json(report.as_dict()))


@app.command()
def selfcheck(
    offline: Annotated[
        bool,
        typer.Option(help="Use deterministic local components instead of Chroma/Ollama/models"),
    ] = False,
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    if offline:
        os.environ["CARE_VECTOR_BACKEND"] = "sqlite"
        os.environ["CARE_EMBEDDING_PROVIDER"] = "hash"
        os.environ["CARE_GENERATOR_PROVIDER"] = "rule"
        os.environ["CARE_RERANKER_PROVIDER"] = "heuristic"
        os.environ["CARE_NLI_PROVIDER"] = "heuristic"
        os.environ["CARE_ALLOW_NETWORK_SYNC"] = "false"
    runtime = _runtime(project_root)
    health = runtime.health()
    payload = {
        "health": health.model_dump(mode="json"),
        "database_integrity": runtime.database.integrity_check(),
        "source_count": len(runtime.sources),
        "weights_sum": sum(asdict(runtime.settings.weights).values()),
    }
    typer.echo(_json(payload))
    if health.status != "ok" or payload["database_integrity"] != "ok":
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option()] = False,
    log_level: Annotated[str, typer.Option()] = "info",
) -> None:
    configure_logging(log_level.upper())
    uvicorn.run("care_anxrag.api:app", host=host, port=port, reload=reload, log_level=log_level)


@app.command()
def scheduler(
    poll_seconds: Annotated[int, typer.Option(min=10)] = 60,
    project_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Run the source-aware polling loop. Use a process supervisor in production."""
    runtime = _runtime(project_root)
    while True:
        now = utc_now()
        due: list[str] = []
        for source in runtime.sources:
            if not source.enabled:
                continue
            state = runtime.database.get_source_state(source.id)
            if state.last_attempt_at is None or (
                now - state.last_attempt_at
                >= timedelta(minutes=source.check_interval_minutes)
            ):
                due.append(source.id)
        if due:
            result = runtime.ingestion.sync(source_ids=due)
            typer.echo(_json(result))
        time.sleep(poll_seconds)


if __name__ == "__main__":
    app()

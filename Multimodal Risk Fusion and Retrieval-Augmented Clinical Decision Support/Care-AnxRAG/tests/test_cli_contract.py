from __future__ import annotations

from typer.main import get_command

from care_anxrag.cli import app


def test_cli_command_surface_contract() -> None:
    command = get_command(app)
    registered = set(command.commands)

    expected = {
        "init",
        "sync",
        "ask",
        "benchmark-scaffold",
        "benchmark-compile",
        "retrieve",
        "stats",
        "coverage",
        "freeze-corpus",
        "sources",
        "staging",
        "approve",
        "reject",
        "withdraw",
        "reconcile",
        "evaluate",
        "evaluate-ablation",
        "experiment-bundle",
        "experiment-final",
        "evaluate-safety",
        "snapshot-experiment",
        "selfcheck",
        "serve",
        "scheduler",
    }

    assert expected <= registered

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
        "benchmark-review-package",
        "benchmark-validate-splits",
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


def test_cli_command_registrations_are_unique() -> None:
    registered_names: list[str] = []
    for command in app.registered_commands:
        if command.name is not None:
            registered_names.append(command.name)
            continue
        assert command.callback is not None
        registered_names.append(
            command.callback.__name__.replace("_", "-")
        )

    duplicates = sorted(
        name
        for name in set(registered_names)
        if registered_names.count(name) > 1
    )

    assert duplicates == []

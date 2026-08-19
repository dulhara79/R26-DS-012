"""
test_call_arity.py — catch wrong-argument-count calls without importing torch.

WHY THIS EXISTS
===============
`scripts/diagnose_collapse.py` shipped with:

    def gradient_norms(model, batch): ...
    ...
    gnorms.append(gradient_norms(model, batch, device))    # 3 args, takes 2

It compiled cleanly. `python -m py_compile` only checks syntax, and the test
suite never imported the module because it needs torch, which is absent from
CPU environments. So the mistake survived every check and surfaced only on a
GPU session, after Bio_ClinicalBERT had downloaded and 100 validation episodes
had run — five times over, once per checkpoint.

The bug was introduced by a scripted edit that renamed a function's parameters
but whose companion replacement for the call site did not match the text and
silently did nothing. That failure mode is invisible to review and guaranteed
to recur.

WHAT THIS CHECKS
================
For every module in src/tcwpn/ and every script in scripts/, parse the file and
compare each call to a function DEFINED IN THAT SAME FILE against the function's
signature:

    - too many positional arguments -> error
    - too few (after accounting for defaults) -> error
    - unknown keyword argument -> error

Deliberately limited to same-file calls. Cross-module resolution would need
import following and produce false positives on decorators, dynamic dispatch and
re-exports; same-file calls are where scripted edits do their damage, and the
check stays cheap and free of false alarms.

Calls with *args or **kwargs at the call site are skipped, since the count is
not statically known.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET_DIRS = [ROOT / "src" / "tcwpn", ROOT / "scripts"]


def _python_files():
    for d in TARGET_DIRS:
        if d.exists():
            yield from sorted(d.glob("*.py"))


def _signature(fn: ast.FunctionDef) -> dict:
    a = fn.args
    positional = [p.arg for p in (a.posonlyargs + a.args)]
    n_defaults = len(a.defaults)
    return {
        "positional": positional,
        "min_positional": len(positional) - n_defaults,
        "max_positional": None if a.vararg else len(positional),
        "kwonly": {p.arg for p in a.kwonlyargs},
        "has_kwargs": a.kwarg is not None,
        "lineno": fn.lineno,
        "is_method": False,   # set by the collector
    }


def _collect(tree: ast.AST) -> dict[str, dict]:
    """Module-level functions only. Methods are skipped: `self` binding makes
    the arity comparison unreliable without type inference."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = _signature(node)
    return out


def _check_file(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    defs = _collect(tree)
    if not defs:
        return []

    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        sig = defs.get(name)
        if sig is None:
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if any(k.arg is None for k in node.keywords):      # **kwargs at call site
            continue

        n_pos = len(node.args)
        kw_names = {k.arg for k in node.keywords}
        loc = f"{path.relative_to(ROOT)}:{node.lineno}"

        if sig["max_positional"] is not None and n_pos > sig["max_positional"]:
            problems.append(
                f"{loc}: {name}() takes at most {sig['max_positional']} "
                f"positional arguments but {n_pos} were given "
                f"(defined at line {sig['lineno']})")
            continue

        supplied = set(sig["positional"][:n_pos]) | kw_names
        required = set(sig["positional"][:sig["min_positional"]])
        missing = required - supplied
        if missing:
            problems.append(
                f"{loc}: {name}() missing required argument(s) "
                f"{sorted(missing)} (defined at line {sig['lineno']})")
            continue

        if not sig["has_kwargs"]:
            known = set(sig["positional"]) | sig["kwonly"]
            unknown = kw_names - known
            if unknown:
                problems.append(
                    f"{loc}: {name}() got unexpected keyword argument(s) "
                    f"{sorted(unknown)} (defined at line {sig['lineno']})")

    return problems


@pytest.mark.parametrize(
    "path", list(_python_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_same_file_calls_match_their_signatures(path):
    problems = _check_file(path)
    assert not problems, "\n".join(problems)


def test_the_checker_actually_catches_the_bug_it_was_written_for():
    """
    Guard the guard. If this stops failing on the known-bad snippet, the checker
    has been weakened and the diagnose_collapse class of bug can return.
    """
    bad = (
        "def gradient_norms(model, batch):\n"
        "    return {}\n"
        "\n"
        "def main():\n"
        "    gnorms = []\n"
        "    gnorms.append(gradient_norms(model, batch, device))\n"
    )
    tmp = ROOT / "tests" / "_arity_probe.py"
    tmp.write_text(bad, encoding="utf-8")
    try:
        problems = _check_file(tmp)
        assert problems, "checker failed to flag a 3-arg call to a 2-arg function"
        assert "gradient_norms" in problems[0]
    finally:
        tmp.unlink(missing_ok=True)

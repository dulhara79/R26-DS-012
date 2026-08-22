"""
test_repo_layout.py — catch broken repository state in two seconds, not after
eight minutes of GPU tokenisation.

WHY THIS EXISTS
===============
Stage C died on every training and evaluation call with:

    ModuleNotFoundError: No module named 'tcwpn.model'

The code was fine. The repository was not. Uploading files through the GitHub
web UI had produced:

    src/tcwpn/model.py       -> renamed to src/tcwpn/old_model_1.py
    scripts/train.py         -> a second copy at src/tcwpn/train.py
    scripts/train_1.py       -> the superseded copy, left in place

So `tcwpn.model` did not exist, and two stale duplicates of train.py sat where
an import could pick the wrong one. Nothing in the test suite noticed, because
every existing test imports `tcwpn.cohort`, `tcwpn.sampler`, `tcwpn.blinding`
and `tcwpn.indexing` — none of them touches `tcwpn.model`, which needs torch and
is therefore skipped in a CPU environment.

These tests import nothing heavy. They check that the files the pipeline calls
are present, in one place, and free of the `_1` / `old_` / `copy` suffixes that
web-UI uploads leave behind.

Run them FIRST, before any data stage.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tcwpn"
SCRIPTS = ROOT / "scripts"

# Every module the pipeline imports at runtime.
REQUIRED_MODULES = [
    "__init__", "blinding", "cohort", "collate", "delong", "evaluation",
    "indexing", "io_paths", "metrics", "model", "sampler", "splits",
]

# Every script a notebook or the README invokes.
REQUIRED_SCRIPTS = [
    "apply_index_time", "audit_cohort", "blind_cohort", "build_clean_cohort",
    "compare_models", "evaluate", "make_episode_plans", "run_shallow_baselines",
    "tokenize_cohort", "train",
]

# Suffixes and prefixes a web-UI upload leaves when a file is added under a
# nudged name instead of replacing the original.
STRAY_PATTERN = re.compile(
    r"(_\d+|_old|_new|_copy|_backup|_bak|_final|_v\d+|\(\d+\))\.py$", re.IGNORECASE)
STRAY_PREFIX = re.compile(r"^(old_|copy_of_|new_)", re.IGNORECASE)


# ===========================================================================
@pytest.mark.parametrize("name", REQUIRED_MODULES)
def test_required_module_exists(name):
    p = SRC / f"{name}.py"
    assert p.is_file(), (
        f"src/tcwpn/{name}.py is missing. Every script that imports "
        f"tcwpn.{name} will fail with ModuleNotFoundError. Check whether the "
        f"file was uploaded under a different name."
    )


@pytest.mark.parametrize("name", REQUIRED_SCRIPTS)
def test_required_script_exists(name):
    assert (SCRIPTS / f"{name}.py").is_file(), f"scripts/{name}.py is missing"


def test_no_stray_duplicates():
    """
    A file named model_1.py or old_model.py next to model.py means an upload
    landed beside the original rather than replacing it. Which of the two is
    live then depends on import order, and the answer is never obvious from a
    traceback.
    """
    offenders = []
    for d in (SRC, SCRIPTS):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.py")):
            if STRAY_PATTERN.search(p.name) or STRAY_PREFIX.match(p.name):
                offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, (
        f"stray duplicate files: {offenders}. Delete them — they shadow or "
        f"compete with the real module and make tracebacks misleading."
    )


def test_scripts_are_not_duplicated_inside_the_package():
    """
    scripts/ holds entry points; src/tcwpn/ holds importable modules. A script
    copied into the package is dead weight at best, and at worst is the file
    that actually gets imported.
    """
    script_names = {p.stem for p in SCRIPTS.glob("*.py")}
    module_names = {p.stem for p in SRC.glob("*.py")}
    overlap = (script_names & module_names) - {"__init__"}
    assert not overlap, (
        f"these exist in BOTH scripts/ and src/tcwpn/: {sorted(overlap)}. "
        f"Keep entry points in scripts/ only."
    )


def test_every_intra_package_import_resolves():
    """
    Parse each module's imports without executing it, so this works on a machine
    with no torch or transformers installed. Catches a renamed or deleted module
    that some other module still imports.
    """
    available = {p.stem for p in SRC.glob("*.py")}
    missing = []
    for p in sorted(SRC.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level and node.level > 0:          # from .x import y
                    target = mod.split(".")[0] if mod else None
                elif mod.startswith("tcwpn."):             # from tcwpn.x import y
                    target = mod.split(".")[1]
                else:
                    continue
                if target and target not in available:
                    missing.append(f"{p.name} imports tcwpn.{target} (not found)")
    assert not missing, missing


def test_scripts_import_only_modules_that_exist():
    available = {p.stem for p in SRC.glob("*.py")}
    missing = []
    for p in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tcwpn"):
                parts = node.module.split(".")
                if len(parts) > 1 and parts[1] not in available:
                    missing.append(f"scripts/{p.name} imports {node.module} (not found)")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("tcwpn."):
                        sub = a.name.split(".")[1]
                        if sub not in available:
                            missing.append(f"scripts/{p.name} imports {a.name} (not found)")
    assert not missing, missing


def test_every_config_referenced_by_the_ablation_exists():
    """The Stage C ladder names six configs; a missing YAML fails only after the
    previous config has finished training."""
    needed = ["protonet", "protonet_temp", "temporal_only", "pcw_only",
              "temporal_pcw", "tcwpn_full"]
    missing = [n for n in needed if not (ROOT / "configs" / f"{n}.yaml").is_file()]
    assert not missing, f"missing configs: {missing}"


def test_mechanism_logging_patch_is_present():
    """
    The lambda/beta logging your supervisor asked for lives in two files. If
    model.py is restored from an older copy, the patch silently disappears and
    the training logs go back to reporting tau only.
    """
    model_src = (SRC / "model.py").read_text(encoding="utf-8")
    assert "lambda_decay" in model_src and "beta_consistency" in model_src, (
        "src/tcwpn/model.py does not expose lambda_decay / beta_consistency — "
        "it is probably an older copy of the file."
    )
    train_src = (SCRIPTS / "train.py").read_text(encoding="utf-8")
    assert "lambda_decay" in train_src, (
        "scripts/train.py does not log lambda_decay — probably an older copy."
    )


def test_every_ablation_preset_has_a_config():
    """
    Stage C ran five of six presets. `temporal_pcw` had a config but was never
    launched, and it is the one preset that separates 'the mechanisms do
    nothing' from 'the auxiliary head does everything'. A preset with no config
    is a row of the ablation table that quietly never gets run.
    """
    src = (SRC / "model.py").read_text(encoding="utf-8")
    block = re.search(r"ABLATION_PRESETS\s*=\s*\{(.*?)\n\}", src, re.S)
    assert block, "ABLATION_PRESETS not found in model.py"
    presets = re.findall(r'"([a-z_0-9]+)"\s*:', block.group(1))
    assert presets, "no presets parsed"
    missing = [p for p in presets
               if not (ROOT / "configs" / f"{p}.yaml").is_file()]
    assert not missing, f"presets with no config file: {missing}"


def test_exactly_one_preset_family_uses_the_auxiliary_head():
    """
    Pins the Stage C confound so it cannot silently return: if `tcwpn_full` is
    once again the only preset with aux>0, then any comparison against it is
    confounded by the auxiliary loss and the ablation cannot attribute anything
    to w^T or w^C.
    """
    src = (SRC / "model.py").read_text(encoding="utf-8")
    block = re.search(r"ABLATION_PRESETS\s*=\s*\{(.*?)\n\}", src, re.S)
    rows = re.findall(
        r'"([a-z_0-9]+)"\s*:\s*\(\s*(True|False)\s*,\s*(True|False)\s*,'
        r'\s*(True|False)\s*,\s*([0-9.]+)\s*\)', block.group(1))
    with_aux = [name for name, _t, _p, _tau, aux in rows if float(aux) > 0]
    assert len(with_aux) >= 2, (
        f"only {with_aux} use the auxiliary head. At least one preset without "
        f"w^T/w^C must also use it, or the auxiliary loss stays confounded "
        f"with the TC-WPN mechanisms in every comparison."
    )

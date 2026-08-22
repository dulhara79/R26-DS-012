#!/usr/bin/env python3
"""
export_checkpoint.py — Phase 4. Upload the EXACT checkpoint.

§16 names the risk this script exists to close: the research repo and the HF
model repo are two separate version-control locations, so it is possible for
everything to run while the deployed model is not the model that was evaluated.

This script does four things and refuses to do any of them halfway:

  1. Reads the run directory produced by scripts/train.py for the frozen run.
  2. Verifies the checkpoint's cfg block matches deployment_config.json.
  3. Computes the sha256 and copies the locked threshold out of the manifest.
  4. Uploads to the HF model repo and writes every identifier back into
     deployment_config.json, so /health can publish the whole chain (§17).

Usage:
    python deployment/huggingface/scripts/export_checkpoint.py \
        --run-dir results/psych_mimic4idx/tcwpn_full_k5_seed42 \
        --out-name tcwpn_clean_v1_full_k5_seed42.pt \
        [--dry-run]

scripts/train.py saves best.pt as:
    {"model": state_dict, "cfg": cfg, "k": k, "seed": seed, "step": step}
and writes locked_threshold into the run manifest. Both are read here rather
than retyped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent
CONFIG_PATH = DEPLOY / "deployment_config.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg: str):
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir", required=True, help="run directory written by scripts/train.py"
    )
    ap.add_argument(
        "--out-name",
        required=True,
        help="filename to upload, e.g. tcwpn_clean_v1_full_k5_seed42.pt",
    )
    ap.add_argument("--ckpt-name", default="best.pt")
    ap.add_argument("--manifest-name", default="manifest.json")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.ckpt_name
    manifest_path = run_dir / args.manifest_name

    if not ckpt_path.exists():
        fail(f"{ckpt_path} not found.")
    if not manifest_path.exists():
        fail(
            f"{manifest_path} not found. The manifest carries locked_threshold "
            f"and the training config; without it the deployment is not verifiable."
        )

    cfg = json.loads(CONFIG_PATH.read_text())
    prov, inf = cfg["provenance"], cfg["inference_configuration"]

    if str(prov.get("tcwpn_git_commit", "")).startswith("<FILL"):
        fail(
            "provenance.tcwpn_git_commit is unfilled. Run vendor.sh first "
            "(Phase 3) so the deployment is pinned to a frozen commit."
        )

    # ---- 2. architecture check (§15.6) --------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model", "cfg"):
        if key not in ckpt:
            fail(
                f"checkpoint has no {key!r} key; it was not written by "
                f"scripts/train.py at the pinned commit."
            )

    train_model_cfg = ckpt["cfg"].get("model", {})
    kmap = inf["_keyword_map"]
    checks = [
        ("preset", inf["preset"], train_model_cfg.get("preset")),
        ("encoder_name", inf["encoder_name"], train_model_cfg.get("encoder_name")),
        (
            "projection_dim",
            inf["projection_dim"],
            train_model_cfg.get("projection_dim"),
        ),
        (
            "lambda_decay",
            inf["lambda_decay"],
            train_model_cfg.get(kmap["lambda_decay"]),
        ),
        ("beta", inf["beta"], train_model_cfg.get(kmap["beta"])),
        (
            "init_temperature",
            inf["init_temperature"],
            train_model_cfg.get("init_temperature"),
        ),
        (
            "consistency_passes",
            inf["consistency_passes"],
            train_model_cfg.get("consistency_passes"),
        ),
    ]
    bad = [
        f"{n}: config {w!r} vs checkpoint {g!r}"
        for n, w, g in checks
        if g is not None and g != w
    ]
    if bad:
        fail(
            "checkpoint architecture disagrees with deployment_config.json:\n  - "
            + "\n  - ".join(bad)
        )

    # ---- 3. threshold and hash ---------------------------------------------
    manifest = json.loads(manifest_path.read_text())
    threshold = manifest.get("locked_threshold")
    if threshold is None:
        fail(
            "manifest has no locked_threshold. §7: the operating point must "
            "come from validation, not be carried over from a previous "
            "deployment."
        )

    staged = run_dir / args.out_name
    shutil.copy2(ckpt_path, staged)
    digest = sha256_of(staged)

    print(f"run_name            : {manifest.get('run_name')}")
    print(f"k_shot / seed       : {ckpt.get('k')} / {ckpt.get('seed')}")
    print(f"best_step           : {manifest.get('best_step')}")
    print(f"best_val_auroc_quick: {manifest.get('best_val_auroc_quick')}")
    print(f"locked_threshold    : {threshold:.4f}")
    print(f"sha256              : {digest}")
    print(f"artefact            : {staged}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded, deployment_config.json unchanged.")
        return

    # ---- 4. upload and stamp -----------------------------------------------
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = prov["hf_model_repo"]
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(staged),
        path_in_repo=args.out_name,
        repo_id=repo_id,
        repo_type="model",
    )
    revision = api.model_info(repo_id).sha

    prov["checkpoint_filename"] = args.out_name
    prov["checkpoint_sha256"] = digest
    prov["hf_model_repo_revision"] = revision
    prov["run_name"] = manifest.get("run_name", prov.get("run_name"))
    cfg["operating_point"]["threshold"] = round(float(threshold), 4)

    CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"\nuploaded to {repo_id}@{revision}")
    print(
        "deployment_config.json updated: checkpoint_filename, checkpoint_sha256, "
        "hf_model_repo_revision, operating_point.threshold"
    )
    print(
        "\nStill manual: set research_metrics.metrics_verified to true only "
        "after confirming the published metrics were measured on THIS "
        "checkpoint (§17)."
    )


if __name__ == "__main__":
    main()

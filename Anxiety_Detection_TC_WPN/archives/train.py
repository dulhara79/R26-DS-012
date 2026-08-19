"""
train.py — Stage 5. Meta-train one configuration.

    python -m scripts.train --config configs/tcwpn_full.yaml --k 5 --seed 42

One script trains every arm of the paper (ProtoNet baseline, each ablation,
full TC-WPN) because they differ only in `preset`. Checkpoints and a run
manifest go to results/<run_name>/.

Selection: validation AUROC on a slice of the frozen val plan. The operating
threshold is chosen on the FULL val plan once, at the end, and written to the
manifest so evaluate.py can consume it without ever touching test.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn.collate import collate_episode          # noqa: E402
from tcwpn.evaluation import quick_val_auroc, run_plan, to_arrays  # noqa: E402
from tcwpn.metrics import select_threshold          # noqa: E402
from tcwpn.model import build_model                 # noqa: E402
from tcwpn.sampler import EpisodePlan, RecordStore  # noqa: E402


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pkl-dir", default="data/clean/pkl")
    ap.add_argument("--plan-dir", default="data/clean/plans")
    ap.add_argument("--stem", default="psych_mimic4")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_all_seeds(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    run_name = f"{cfg['name']}_k{args.k}_seed{args.seed}"
    run_dir = Path(args.results) / args.stem / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ---------------------------------------------------------------
    pkl_dir, plan_dir = Path(args.pkl_dir), Path(args.plan_dir)
    train_store = RecordStore.from_pkl(pkl_dir / f"{args.stem}_train.pkl", "train")
    val_store = RecordStore.from_pkl(pkl_dir / f"{args.stem}_val.pkl", "val")
    train_plan = EpisodePlan.load(plan_dir / f"{args.stem}_train_k{args.k}.json")
    val_plan = EpisodePlan.load(plan_dir / f"{args.stem}_val_k{args.k}.json")

    print(f"train: {len(train_store):,} notes / {train_store.n_patients():,} patients")
    print(f"val:   {len(val_store):,} notes / {val_store.n_patients():,} patients")
    print(f"episodes: {len(train_plan):,} train, {len(val_plan):,} val")

    # ---- model --------------------------------------------------------------
    model = build_model(cfg["model"]).to(device)
    groups = model.parameter_groups(
        encoder_lr=cfg["optim"].get("encoder_lr", 2e-5),
        head_lr=cfg["optim"].get("head_lr", 1e-3),
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
    )
    opt = torch.optim.AdamW(groups)
    total_steps = len(train_plan)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[g["lr"] for g in groups],
        total_steps=total_steps,
        pct_start=cfg["optim"].get("warmup_frac", 0.1),
    )
    use_amp = device.type == "cuda" and cfg["optim"].get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum = int(cfg["optim"].get("grad_accum", 1))
    eval_every = int(cfg["optim"].get("eval_every", 250))
    clip = float(cfg["optim"].get("grad_clip", 1.0))

    best = {"auroc": -1.0, "step": -1}
    history = []
    t0 = time.time()

    for step, ep in enumerate(train_plan, start=1):
        model.train()
        batch = collate_episode(ep, train_store, device)
        with torch.autocast(device_type=device.type, enabled=use_amp,
                            dtype=torch.float16):
            out = model(batch)
            loss = out["loss"] / accum

        scaler.scale(loss).backward()
        if step % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        sched.step()

        if step % 50 == 0:
            lam = out.get("lambda_decay")
            beta = out.get("beta_consistency")
            mech = f"tau {out['temperature'].item():.2f}"
            if lam is not None:
                mech += f"  lambda {lam.item():.3f}"
            if beta is not None:
                mech += f"  beta {beta.item():.3f}"
            print(f"  step {step:>6}/{total_steps}  loss {out['loss'].item():.4f}  "
                  f"{mech}  {(time.time()-t0)/step:.2f}s/step")

        if step % eval_every == 0 or step == total_steps:
            auroc = quick_val_auroc(model, val_store, val_plan, device,
                                    max_episodes=cfg["optim"].get("val_episodes", 60))
            lam = out.get("lambda_decay")
            beta = out.get("beta_consistency")
            history.append({
                "step": step, "val_auroc": auroc,
                "loss": float(out["loss"].item()),
                "tau": float(out["temperature"].item()),
                "lambda_decay": float(lam.item()) if lam is not None else None,
                "beta_consistency": float(beta.item()) if beta is not None else None,
            })
            print(f"  [val] step {step}  AUROC {auroc:.4f}"
                  f"{'  <- best' if auroc > best['auroc'] else ''}")
            if auroc > best["auroc"]:
                best = {"auroc": auroc, "step": step}
                torch.save({"model": model.state_dict(), "cfg": cfg,
                            "k": args.k, "seed": args.seed, "step": step},
                           run_dir / "best.pt")

    # ---- lock the operating threshold on the FULL val plan ------------------
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    val_res = run_plan(model, val_store, val_plan, device)
    _, y_val, p_val = to_arrays(val_res)
    threshold = select_threshold(y_val, p_val,
                                 objective=cfg.get("threshold_objective", "f1"))
    print(f"\nlocked threshold from validation: {threshold:.4f}")

    manifest = {
        "run_name": run_name,
        "config": cfg,
        "k_shot": args.k,
        "seed": args.seed,
        "stem": args.stem,
        "best_val_auroc_quick": best["auroc"],
        "best_step": best["step"],
        "locked_threshold": float(threshold),
        "val_plan_meta": val_plan.meta,
        "train_plan_meta": train_plan.meta,
        "history": history,
        "device": str(device),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {run_dir/'manifest.json'}")
    print("\nNEXT: python -m scripts.evaluate --run "
          f"{run_dir} --split test")


if __name__ == "__main__":
    main()

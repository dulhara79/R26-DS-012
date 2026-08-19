"""
collect_seeds.py — Aggregate repeated runs into mean +/- SD, and assemble the
proto-loss mechanism table.

    python -m scripts.collect_seeds \
        --results results/psych_mimic4idx \
        --configs aux_only temporal_aux pcw_aux tcwpn_full \
        --seeds 42 43 44 45 46 --k 5 \
        --out-dir results/summary

WHY MEAN +/- SD AND NOT THE BEST RUN
====================================
The Phase 2 single-seed ladder produced AUROC 0.7415 / 0.7400 / 0.7378 / 0.7335
across four configurations whose 95% bootstrap CIs are roughly 0.041 wide. The
entire spread between the best and worst configuration is 0.0080 -- about a
fifth of one CI width. Nothing can be concluded from that ordering, and picking
the top row would be selecting on noise.

Seed variance is the missing quantity. If the between-seed SD of a single
configuration is comparable to the between-configuration spread, then the
mechanisms are indistinguishable and the paper must say so.

THE PROTO-LOSS TABLE
====================
For runs with aux_head_weight = 0, the logged `loss` IS the prototypical loss,
because the auxiliary term is identically zero. For runs with an auxiliary head
and the newer logging, `proto_loss` is recorded separately. That makes the two
comparable across the whole ladder, which is what licenses the mechanism claim.

`proto_loss_min` is a minimum over logged steps and therefore favours noise, so
this script also reports `proto_loss_final_mean`: the mean over the last
`--tail` logged evaluations. Quote the tail mean in the paper and keep the
minimum only as a secondary column.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

METRICS = ["auroc", "pr_auc", "f1_positive", "sensitivity", "specificity",
           "brier", "ece"]

AUX_PRESETS = {"tcwpn_full", "aux_only", "temporal_aux", "pcw_aux"}


def _run_dir(results: Path, cfg: str, k: int, seed: int) -> Path:
    return results / f"{cfg}_k{k}_seed{seed}"


def _load_run(run: Path) -> dict | None:
    ev, mf = run / "eval_test.json", run / "manifest.json"
    if not ev.exists():
        return None
    m = json.loads(ev.read_text())["metrics"]
    out = {k: m.get(k) for k in METRICS}

    if mf.exists():
        man = json.loads(mf.read_text())
        preset = man.get("config", {}).get("model", {}).get("preset", "?")
        hist = man.get("history", [])
        out["preset"] = preset
        out["has_aux"] = preset in AUX_PRESETS
        if hist:
            has_proto = "proto_loss" in hist[-1] and hist[-1]["proto_loss"] is not None
            if has_proto:
                series = [e["proto_loss"] for e in hist if e.get("proto_loss") is not None]
                out["proto_source"] = "logged"
            elif not out["has_aux"]:
                # aux weight is zero, so total loss == prototypical loss
                series = [e["loss"] for e in hist]
                out["proto_source"] = "total_loss (aux=0)"
            else:
                series = []
                out["proto_source"] = "unavailable (pre-patch run with aux)"
            if series:
                out["proto_loss_min"] = min(series)
                out["_proto_series"] = series
    return out


def _tail_mean(series, tail: int) -> float:
    s = series[-tail:] if len(series) >= tail else series
    return sum(s) / len(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/<stem>")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tail", type=int, default=4,
                    help="how many trailing evaluations to average for the "
                         "proto-loss tail mean")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    results = Path(args.results)
    per_run, missing = [], []

    for cfg in args.configs:
        for seed in args.seeds:
            run = _run_dir(results, cfg, args.k, seed)
            rec = _load_run(run)
            if rec is None:
                missing.append(f"{cfg}_k{args.k}_seed{seed}")
                continue
            series = rec.pop("_proto_series", None)
            if series:
                rec["proto_loss_final_mean"] = _tail_mean(series, args.tail)
            rec.update({"config": cfg, "seed": seed})
            per_run.append(rec)

    if not per_run:
        print("No completed runs found under", results)
        if missing:
            print("missing:", missing)
        return 1

    df = pd.DataFrame(per_run)
    ordered = ["config", "seed", "preset", "has_aux"] + METRICS + \
              ["proto_loss_final_mean", "proto_loss_min", "proto_source"]
    df = df[[c for c in ordered if c in df.columns]]

    # ---------------- per-seed table ----------------
    print("=" * 100)
    print("PER-SEED RESULTS")
    print("=" * 100)
    print(df.to_string(index=False))

    # ---------------- mean +/- SD ----------------
    agg_rows = []
    for cfg, g in df.groupby("config", sort=False):
        row = {"config": cfg, "n_seeds": len(g)}
        for m in METRICS:
            if m not in g:
                continue
            vals = g[m].dropna()
            mean = vals.mean()
            sd = vals.std(ddof=1) if len(vals) > 1 else float("nan")
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_sd"] = None if math.isnan(sd) else round(sd, 4)
            if len(vals) > 1:
                # Normal-approximation CI on the mean across seeds. With five
                # seeds this is indicative, not authoritative; the per-run
                # bootstrap CI remains the primary uncertainty estimate.
                half = 1.96 * sd / math.sqrt(len(vals))
                row[f"{m}_ci95"] = f"[{mean-half:.4f}, {mean+half:.4f}]"
        if "proto_loss_final_mean" in g:
            v = g["proto_loss_final_mean"].dropna()
            if len(v):
                row["proto_loss_final_mean"] = round(v.mean(), 4)
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows)
    keep = ["config", "n_seeds", "auroc_mean", "auroc_sd", "auroc_ci95",
            "pr_auc_mean", "pr_auc_sd", "f1_positive_mean", "sensitivity_mean",
            "specificity_mean", "brier_mean", "ece_mean", "proto_loss_final_mean"]
    agg_view = agg[[c for c in keep if c in agg.columns]]

    print()
    print("=" * 100)
    print("MEAN +/- SD ACROSS SEEDS")
    print("=" * 100)
    print(agg_view.to_string(index=False))

    # ---------------- the interpretation gate ----------------
    print()
    print("-" * 100)
    if "auroc_mean" in agg.columns and len(agg) > 1:
        spread = agg["auroc_mean"].max() - agg["auroc_mean"].min()
        sds = agg["auroc_sd"].dropna()
        typical_sd = float(sds.mean()) if len(sds) else float("nan")
        print(f"between-configuration AUROC spread : {spread:.4f}")
        print(f"typical between-seed SD            : {typical_sd:.4f}")
        if not math.isnan(typical_sd):
            if spread < typical_sd:
                print()
                print("  The spread between configurations is SMALLER than the")
                print("  variation caused by changing the seed alone. The mechanisms")
                print("  are not distinguishable at this sample size. Report them as")
                print("  equivalent; do not rank them.")
            elif spread < 2 * typical_sd:
                print()
                print("  The spread is within twice the seed SD. Treat any ordering as")
                print("  provisional and lean on the DeLong tests rather than the means.")
            else:
                print()
                print("  The spread exceeds twice the seed SD. An ordering may be real;")
                print("  confirm with paired DeLong plus Holm before claiming it.")

    if missing:
        print()
        print(f"MISSING RUNS ({len(missing)}):")
        for m in missing:
            print("   ", m)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_seed_results.csv", index=False)
        agg.to_csv(out / "seed_summary.csv", index=False)
        print(f"\nwrote {out/'per_seed_results.csv'} and {out/'seed_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

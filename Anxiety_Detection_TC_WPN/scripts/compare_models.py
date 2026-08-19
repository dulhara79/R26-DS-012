"""
compare_models.py — Stage 7. Paired significance tests and the results tables.

    # one pairwise test
    python -m scripts.compare_models pair \
        --a results/psych_mimic4/tcwpn_full_k5_seed42/predictions_test.csv \
        --b results/psych_mimic4/protonet_k5_seed42/predictions_test.csv

    # the whole main table
    python -m scripts.compare_models table --results results/psych_mimic4 \
        --split test --out results/psych_mimic4/main_table.csv

    # robustness: same run, unblinded vs blinded
    python -m scripts.compare_models pair \
        --a results/.../predictions_test.csv \
        --b results/.../predictions_test_blind-anx_meds.csv

`pair` intersects on patient_id, so it can only report a paired test on
patients both runs actually scored. If the intersection is much smaller than
either input, the two runs did not use the same episode plan and the comparison
is not valid — the script says so rather than quietly proceeding.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn.metrics import compare_models, summarise_seeds   # noqa: E402


def load_preds(path):
    df = pd.read_csv(path)
    return {
        "probs": dict(zip(df["patient_id"].astype(str), df["p_anxiety"])),
        "labels": dict(zip(df["patient_id"].astype(str), df["label"].astype(int))),
        "name": Path(path).parent.name,
    }


def cmd_pair(args):
    a, b = load_preds(args.a), load_preds(args.b)
    shared = sorted(set(a["probs"]) & set(b["probs"]))
    cov = len(shared) / max(len(a["probs"]), len(b["probs"]), 1)
    if cov < 0.95:
        print(f"WARNING: only {len(shared)} shared patients "
              f"({cov:.1%} of the larger run). The two runs were probably not "
              f"scored on the same episode plan; a paired test is not valid "
              f"unless this is ~100%.")

    y = np.array([a["labels"][k] for k in shared], dtype=int)
    yb = np.array([b["labels"][k] for k in shared], dtype=int)
    if not np.array_equal(y, yb):
        raise SystemExit("label mismatch between the two prediction files")
    pa = np.array([a["probs"][k] for k in shared], dtype=float)
    pb = np.array([b["probs"][k] for k in shared], dtype=float)

    res = compare_models(y, pa, pb, name_a=a["name"], name_b=b["name"])
    res["shared_patient_fraction"] = round(cov, 4)
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


def cmd_table(args):
    root = Path(args.results)
    rows = []
    for eval_file in sorted(root.glob(f"*/eval_{args.split}*.json")):
        with open(eval_file) as f:
            blob = json.load(f)
        m = blob.get("metrics", blob)
        rows.append({
            "run": m.get("run", eval_file.parent.name),
            "model": eval_file.parent.name.rsplit("_k", 1)[0],
            "k_shot": m.get("k_shot"),
            "seed": m.get("seed"),
            "blind": m.get("blind", "none"),
            "n_patients": m.get("n_patients"),
            "prevalence": m.get("prevalence"),
            "auroc": m.get("auroc"),
            "auroc_ci_lower": m.get("auroc_ci_lower"),
            "auroc_ci_upper": m.get("auroc_ci_upper"),
            "pr_auc": m.get("pr_auc"),
            "f1_positive": m.get("f1_positive"),
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "brier": m.get("brier"),
            "ece": m.get("ece"),
        })
    if not rows:
        raise SystemExit(f"no eval_{args.split}*.json found under {root}")

    df = pd.DataFrame(rows).sort_values(["blind", "model", "k_shot", "seed"])
    print(df.to_string(index=False))

    agg = (
        df.groupby(["blind", "model", "k_shot"])
        .agg(n_seeds=("seed", "nunique"),
             auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
             pr_auc_mean=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
             f1_mean=("f1_positive", "mean"), f1_sd=("f1_positive", "std"))
        .reset_index().round(4)
    )
    print("\nmean +/- SD across seeds:")
    print(agg.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        agg.to_csv(str(args.out).replace(".csv", "_aggregated.csv"), index=False)
        print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pair")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_pair)

    t = sub.add_parser("table")
    t.add_argument("--results", required=True)
    t.add_argument("--split", default="test")
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_table)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

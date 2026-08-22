"""
evaluate.py — Stage 6. Score a trained run on a frozen plan.

    python -m scripts.evaluate --run results/psych_mimic4/tcwpn_full_k5_seed42 \
                               --split test
    # robustness: same run, same plan, blinded text
    python -m scripts.evaluate --run <run> --split test --blind anx_meds

Writes results/<...>/eval_<split><blind>.json (metrics) and
predictions_<split><blind>.csv (patient_id, label, p_anxiety, n_episodes).

The predictions CSV is the input to compare_models.py, which runs the paired
DeLong test. Keeping the per-patient vectors on disk means the significance
tests are reproducible without re-running any model.

The threshold is read from the run manifest, where train.py locked it using
the validation split only. This script never selects a threshold.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn.evaluation import run_plan, to_arrays   # noqa: E402
from tcwpn.metrics import compute_metrics          # noqa: E402
from tcwpn.model import build_model                # noqa: E402
from tcwpn.sampler import EpisodePlan, RecordStore # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--blind", default=None,
                    help="blinding level suffix, e.g. anx_meds; uses the pkl "
                         "<stem>_<split>_blind-<level>.pkl with the SAME plan")
    ap.add_argument("--pkl-dir", default="data/clean/pkl")
    ap.add_argument("--plan-dir", default="data/clean/plans")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    stem = args.stem or manifest["stem"]
    k = manifest["k_shot"]
    threshold = manifest["locked_threshold"]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    suffix = f"_blind-{args.blind}" if args.blind else ""
    pkl = Path(args.pkl_dir) / f"{stem}_{args.split}{suffix}.pkl"
    plan_path = Path(args.plan_dir) / f"{stem}_{args.split}_k{k}.json"
    if not pkl.exists():
        raise SystemExit(f"missing {pkl}")

    store = RecordStore.from_pkl(pkl, split_name=args.split)
    plan = EpisodePlan.load(plan_path)

    # The plan indexes into record positions. A blinded pkl is produced from the
    # same cohort rows in the same order, so positions line up — but verify by
    # comparing note_ids rather than trusting it.
    if plan.meta.get("store_fingerprint"):
        from tcwpn.sampler import store_fingerprint

        got = store_fingerprint(store)
        if got != plan.meta["store_fingerprint"]:
            raise SystemExit(
                f"plan fingerprint {plan.meta['store_fingerprint']} does not "
                f"match store {got}. The blinded pkl must be generated from the "
                f"same cohort CSV with the same row order (re-run "
                f"tokenize_cohort.py on the identical --cohort file)."
            )

    model = build_model(manifest["config"]["model"]).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    res = run_plan(model, store, plan, device)
    ids, y, p = to_arrays(res)
    metrics = compute_metrics(y, p, threshold=threshold,
                              n_bootstrap=args.bootstrap)
    ece_table = metrics.pop("_ece_table")

    metrics.update({
        "run": run_dir.name, "split": args.split, "k_shot": k,
        "seed": manifest["seed"], "blind": args.blind or "none",
        "n_episodes": len(plan),
        "mean_episodes_per_patient": round(
            sum(res["coverage"].values()) / max(len(res["coverage"]), 1), 2),
        "query_coverage_fraction": round(
            len(res["coverage"]) / max(store.n_patients(), 1), 4),
    })

    tag = f"{args.split}{suffix}"
    with open(run_dir / f"eval_{tag}.json", "w") as f:
        json.dump({"metrics": metrics, "calibration_bins": ece_table}, f, indent=2)

    pd.DataFrame({
        "patient_id": ids,
        "label": y,
        "p_anxiety": p,
        "n_episodes": [res["coverage"][i] for i in ids],
    }).to_csv(run_dir / f"predictions_{tag}.csv", index=False)

    print(json.dumps(
        {k_: v for k_, v in metrics.items() if not k_.startswith("_")}, indent=2))
    print(f"\nwrote {run_dir/f'eval_{tag}.json'}")
    print(f"wrote {run_dir/f'predictions_{tag}.csv'}")


if __name__ == "__main__":
    main()

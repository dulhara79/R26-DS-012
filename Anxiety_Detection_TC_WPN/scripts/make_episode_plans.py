"""
make_episode_plans.py — Stage 4. Freeze the episodes every model will see.

    python -m scripts.make_episode_plans \
        --pkl-dir data/clean/pkl --stem psych_mimic4 \
        --k 1 3 5 10 --q-query 5 --out data/clean/plans

Why plans are files and not a live sampler:
  * TF-IDF, linear probe, ProtoNet and TC-WPN are then scored on byte-identical
    episodes, which is the precondition for the paired DeLong test.
  * The blinded run reuses the plan built for the unblinded pkl (same note_ids,
    same order), so the robustness comparison holds patients fixed.
  * A reviewer can re-run validate_episodes.py against the shipped plans.

Also emits the 5,000-episode leakage certificate at every K.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn.sampler import (   # noqa: E402
    RecordStore, build_eval_plan, build_train_plan,
    format_leakage_report, validate_plan,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", default="data/clean/pkl")
    ap.add_argument("--stem", required=True,
                    help="e.g. psych_mimic4 (matches <stem>_<split>.pkl)")
    ap.add_argument("--out", default="data/clean/plans")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--q-query", type=int, default=5)
    ap.add_argument("--train-episodes", type=int, default=3000)
    ap.add_argument("--eval-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leakage-episodes", type=int, default=5000)
    args = ap.parse_args()

    pkl_dir, out_dir = Path(args.pkl_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stores = {}
    for split in ("train", "val", "test"):
        p = pkl_dir / f"{args.stem}_{split}.pkl"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        stores[split] = RecordStore.from_pkl(p, split_name=split)
        print(json.dumps(stores[split].describe()))

    certificate = {"stem": args.stem, "q_query": args.q_query,
                   "seed": args.seed, "per_k": {}}

    print("\n" + "=" * 66)
    print("EPISODE LEAKAGE CERTIFICATE")
    print("=" * 66)

    for k in args.k:
        # ---- training plan ---------------------------------------------------
        tr = build_train_plan(stores["train"], k, args.q_query,
                              args.train_episodes, seed=args.seed + k)
        tr.save(out_dir / f"{args.stem}_train_k{k}.json")

        # ---- evaluation plans (coverage guaranteed) --------------------------
        for split in ("val", "test"):
            ev = build_eval_plan(stores[split], k, args.q_query,
                                 seed=args.seed + 1000 + k,
                                 n_repeats=args.eval_repeats)
            ev.save(out_dir / f"{args.stem}_{split}_k{k}.json")
            st = validate_plan(ev, stores[split])
            certificate["per_k"].setdefault(str(k), {})[split] = st

        # ---- stress test on the training pool -------------------------------
        stress = build_train_plan(stores["train"], k, args.q_query,
                                  args.leakage_episodes, seed=args.seed + 7 * k)
        st = validate_plan(stress, stores["train"])
        certificate["per_k"].setdefault(str(k), {})["train_stress"] = st
        print(format_leakage_report(st, k))

    all_clean = all(
        v[s]["episodes_with_support_query_patient_overlap"] == 0
        for v in certificate["per_k"].values() for s in v
    )
    certificate["all_clean"] = all_clean
    print("RESULT:", "ZERO LEAKAGE AT ALL K" if all_clean else "LEAKAGE DETECTED")

    with open(out_dir / f"leakage_certificate_{args.stem}.json", "w") as f:
        json.dump(certificate, f, indent=2)
    print(f"wrote {out_dir / f'leakage_certificate_{args.stem}.json'}")

    if not all_clean:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

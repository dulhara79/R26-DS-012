"""
analyse_mechanisms.py — Is TC-WPN's weighting doing anything at all?

Supervisor's Phase 4A, items 16 and 17. Inference only; no training.

    python -m scripts.analyse_mechanisms \
        --run results/psych_mimic4idx/tcwpn_full_k5_seed42 \
        --reference results/psych_mimic4idx/aux_only_k5_seed42 \
        --pkl-dir data/clean/pkl --plan-dir data/clean/plans \
        --split test --episodes 300

THE QUANTITY THAT MATTERS
=========================
Both weights are normalised across the K support notes of a class, so they sum
to 1. With K=5 the uninformative solution is w = (0.2, 0.2, 0.2, 0.2, 0.2). The
question is therefore not "what are the weights" but "how far from uniform are
they", and the cleanest scalar for that is normalised entropy:

    H_norm = -sum(w_i log w_i) / log(K)

    H_norm = 1.000  ->  exactly uniform; the mechanism cannot change the
                        prototype at all, and its AUROC must equal the
                        unweighted model up to optimisation noise
    H_norm = 0.99   ->  weights vary by a few percent; the prototype moves by
                        a fraction of the distance between support embeddings
    H_norm < 0.90   ->  the mechanism is genuinely reweighting

This is reported alongside the raw spread (max/min ratio, SD) because entropy
alone hides whether one note dominates or all differ slightly.

PROTOTYPE DIVERGENCE
====================
With --reference, the same episodes are run through a second checkpoint and the
prototypes compared:

    ||p_this - p_ref||     Euclidean, on L2-normalised embeddings so the scale
                           is interpretable: 0 = identical, 2 = antipodal
    cos(p_this, p_ref)     1.0 = same direction

Two models trained from different objectives will never produce identical
prototypes, so a non-zero distance here is expected and does NOT by itself show
the mechanism is active. Read it together with the entropy: near-uniform
weights plus large prototype divergence means the difference comes from the
encoder having trained differently, not from the weighting.

WHAT THIS CANNOT TELL YOU
=========================
It cannot tell you whether a *better* weighting scheme would help. It only tells
you whether the current one is active, and if active, whether it correlates with
anything useful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcwpn.collate import collate_episode           # noqa: E402
from tcwpn.model import build_model                 # noqa: E402
from tcwpn.sampler import RecordStore, EpisodePlan  # noqa: E402


def normalised_entropy(w: np.ndarray) -> float:
    """w sums to 1. Returns H(w)/log(K); 1.0 means uniform."""
    k = len(w)
    if k < 2:
        return float("nan")
    p = np.clip(w, 1e-12, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(k))


def load_run(run_dir: Path, device: str):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    model = build_model(manifest["config"]["model"]).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, manifest


@torch.no_grad()
def episode_record(model, batch):
    """Per-class weights and prototypes for one episode."""
    sup = batch["support"]
    sup_emb = model.embedder(sup["input_ids"], sup["attention_mask"],
                             sup["note_index"])
    labels = sup["labels"]
    days = sup["days"]

    per_class = {}
    protos = {}
    for c in sorted(set(int(v) for v in labels.tolist())):
        m = labels == c
        proto, w = model.build_prototype(sup_emb[m], days[m])
        protos[c] = proto
        wv = w.detach().float().cpu().numpy().reshape(-1)
        wv = wv / max(wv.sum(), 1e-12)
        per_class[c] = {
            "weights": wv,
            "days": days[m].detach().float().cpu().numpy().reshape(-1),
        }
    return per_class, protos, sup_emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", default=None,
                    help="second run dir; prototypes compared on the same episodes")
    ap.add_argument("--pkl-dir", required=True)
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run)
    model, manifest = load_run(run_dir, device)
    stem, K = manifest["stem"], manifest["k_shot"]
    preset = manifest["config"]["model"].get("preset", "?")

    store = RecordStore.from_pkl(
        Path(args.pkl_dir) / f"{stem}_{args.split}.pkl", args.split)
    plan = EpisodePlan.load(Path(args.plan_dir) / f"{stem}_{args.split}_k{K}.json")

    ref_model = None
    if args.reference:
        ref_model, _ = load_run(Path(args.reference), device)

    n = min(args.episodes, len(plan.episodes))
    print(f"{run_dir.name}  preset={preset}  K={K}  {n} {args.split} episodes\n")

    ent, cv, ratio, wmin, wmax = [], [], [], [], []
    corr_days = []
    ent_by_class = {0: [], 1: []}
    proto_dist, proto_cos = [], []

    for i in range(n):
        batch = collate_episode(plan.episodes[i], store, device)
        per_class, protos, _ = episode_record(model, batch)

        for c, rec in per_class.items():
            w, d = rec["weights"], rec["days"]
            e = normalised_entropy(w)
            ent.append(e)
            if c in ent_by_class:
                ent_by_class[c].append(e)
            cv.append(float(w.std() / max(w.mean(), 1e-12)))
            ratio.append(float(w.max() / max(w.min(), 1e-12)))
            wmin.append(float(w.min()))
            wmax.append(float(w.max()))
            if len(w) > 2 and np.std(d) > 0 and np.std(w) > 0:
                corr_days.append(float(np.corrcoef(w, d)[0, 1]))

        if ref_model is not None:
            _, ref_protos, _ = episode_record(ref_model, batch)
            for c in protos:
                if c not in ref_protos:
                    continue
                a = F.normalize(protos[c], dim=-1)
                b = F.normalize(ref_protos[c], dim=-1)
                proto_dist.append(float((a - b).norm()))
                proto_cos.append(float(F.cosine_similarity(a, b, dim=0)))

        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{n}")

    def s(v):
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        if not len(v):
            return None
        return {"mean": float(v.mean()), "sd": float(v.std()),
                "p05": float(np.percentile(v, 5)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95))}

    report = {
        "run": run_dir.name, "preset": preset, "split": args.split,
        "n_episodes": n, "k_shot": K,
        "uniform_weight": 1.0 / K,
        "normalised_entropy": s(ent),
        "weight_cv": s(cv),
        "weight_max_over_min": s(ratio),
        "weight_min": s(wmin),
        "weight_max": s(wmax),
        "corr_weight_vs_days_before_index": s(corr_days),
        "entropy_cases": s(ent_by_class[1]),
        "entropy_controls": s(ent_by_class[0]),
    }
    if ref_model is not None:
        report["reference"] = Path(args.reference).name
        report["prototype_l2_distance"] = s(proto_dist)
        report["prototype_cosine"] = s(proto_cos)

    # ------------------------------ printed report
    print("\n" + "=" * 74)
    print(f"MECHANISM DIAGNOSIS — {run_dir.name}")
    print("=" * 74)
    print(f"  uniform weight at K={K}: {1.0/K:.4f}\n")
    for key in ("normalised_entropy", "weight_cv", "weight_max_over_min",
                "weight_min", "weight_max", "corr_weight_vs_days_before_index"):
        v = report[key]
        if v:
            print(f"  {key:<34} mean {v['mean']:>9.5f}   sd {v['sd']:.5f}   "
                  f"p05 {v['p05']:.5f}  p95 {v['p95']:.5f}")
    if ref_model is not None:
        print()
        for key in ("prototype_l2_distance", "prototype_cosine"):
            v = report[key]
            if v:
                print(f"  {key:<34} mean {v['mean']:>9.5f}   sd {v['sd']:.5f}")

    print("\n" + "-" * 74)
    print("READING")
    print("-" * 74)
    e = report["normalised_entropy"]["mean"] if report["normalised_entropy"] else float("nan")
    r = report["weight_max_over_min"]["mean"] if report["weight_max_over_min"] else float("nan")
    print(f"  normalised entropy {e:.5f}  (1.000 = perfectly uniform)")
    if e > 0.999:
        print("    -> the weights are uniform to three decimals. The mechanism")
        print("       cannot be moving the prototype, and an AUROC equal to the")
        print("       unweighted model is the expected result, not a surprise.")
    elif e > 0.99:
        print("    -> weights deviate from uniform by roughly a percent. The")
        print("       prototype shifts, but only slightly.")
    else:
        print("    -> the weights are genuinely non-uniform; if AUROC is still")
        print("       unchanged, the reweighting is active but uninformative.")
    print(f"  largest/smallest support weight {r:.3f}  (1.0 = all equal)")

    c = report["corr_weight_vs_days_before_index"]
    if c:
        print(f"  corr(weight, days_before_index) = {c['mean']:+.4f}")
        if abs(c["mean"]) < 0.1:
            print("    -> the temporal weight is barely tracking recency, which is")
            print("       what it is defined to do. Worth checking lambda.")

    out = Path(args.out) if args.out else run_dir / f"mechanism_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

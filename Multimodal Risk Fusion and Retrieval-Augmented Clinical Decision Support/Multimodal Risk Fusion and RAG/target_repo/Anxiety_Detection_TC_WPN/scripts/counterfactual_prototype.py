"""
counterfactual_prototype.py — Isolate the weighting effect from the encoder effect.

WHY THIS IS NEEDED
==================
`analyse_mechanisms.py --reference` compares tcwpn_full's prototype against
aux_only's prototype and reports cosine 0.567. That looks like a large effect,
but the two prototypes come from two SEPARATELY TRAINED models, so the number
mixes together:

    (a) the encoder having converged to a different representation, and
    (b) the support weights being non-uniform.

Only (b) is the mechanism under test. A cosine of 0.567 between two independently
trained BERT encoders is unremarkable on its own and cannot be attributed to
weighting.

THE CLEAN MEASUREMENT
=====================
Inside ONE model, on ONE set of embeddings, build the prototype twice:

    p_weighted = sum_i w_i z_i        with the model's learned weights
    p_uniform  = (1/K) sum_i z_i      with the weights forced to 1/K

Everything except the weights is held fixed, so any difference is caused by the
weighting and nothing else. This is the counterfactual the ablation table
implicitly asks about: what would this exact model do if its weighting were
switched off at inference time?

WHAT IS REPORTED
================
    proto_cos_weighted_vs_uniform   1.0 = weighting changed nothing
    proto_l2_weighted_vs_uniform    0.0 = weighting changed nothing
    effective_K = exp(H)            how many support notes the prototype
                                    effectively averages; K means all of them
    decision_flip_rate              fraction of QUERY notes whose predicted
                                    class changes when weighting is switched off

The last one is the bottom line. If the flip rate is near zero, the mechanism
cannot affect AUROC no matter how different the prototypes look geometrically,
and the flat ablation table is explained.

    python -m scripts.counterfactual_prototype \
        --run results/psych_mimic4idx/tcwpn_full_k5_seed42 \
        --pkl-dir data/clean/pkl --plan-dir data/clean/plans \
        --split val --episodes 300
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


def effective_k(w: np.ndarray) -> float:
    """exp(Shannon entropy). Equals K for uniform weights."""
    p = np.clip(np.asarray(w, dtype=float), 1e-12, None)
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def compare_one_episode(model, batch):
    sup, qry = batch["support"], batch["query"]
    sup_emb = model.embedder(sup["input_ids"], sup["attention_mask"],
                             sup["note_index"])
    qry_emb = model.embedder(qry["input_ids"], qry["attention_mask"],
                             qry["note_index"])
    labels, days = sup["labels"], sup["days"]

    classes = sorted(set(int(v) for v in labels.tolist()))
    p_w, p_u, keff, cos, l2 = [], [], [], [], []

    for c in classes:
        m = labels == c
        emb = sup_emb[m]

        proto_w, w = model.build_prototype(emb, days[m])

        # Counterfactual: identical embeddings, weights forced uniform.
        # Matches build_prototype's normalise-then-combine convention.
        z = F.normalize(emb, dim=-1)
        proto_u = F.normalize(z.mean(dim=0), dim=-1)
        proto_w_n = F.normalize(proto_w, dim=-1)

        p_w.append(proto_w_n)
        p_u.append(proto_u)
        cos.append(float(F.cosine_similarity(proto_w_n, proto_u, dim=0)))
        l2.append(float((proto_w_n - proto_u).norm()))
        wv = w.detach().float().cpu().numpy().reshape(-1)
        keff.append(effective_k(wv))

    if len(classes) != 2:
        return None

    q = F.normalize(qry_emb, dim=-1)
    Pw = torch.stack(p_w, 0)
    Pu = torch.stack(p_u, 0)
    dw = torch.cdist(q.unsqueeze(0), Pw.unsqueeze(0)).squeeze(0) ** 2
    du = torch.cdist(q.unsqueeze(0), Pu.unsqueeze(0)).squeeze(0) ** 2

    pred_w = dw.argmin(dim=1).cpu().numpy()
    pred_u = du.argmin(dim=1).cpu().numpy()

    idx = {c: i for i, c in enumerate(classes)}
    pos = idx.get(1, len(classes) - 1)
    neg = idx.get(0, 0)
    tau = float(torch.exp(model.log_temperature))
    score_w = torch.softmax(-dw * tau, dim=1)[:, pos].cpu().numpy()
    score_u = torch.softmax(-du * tau, dim=1)[:, pos].cpu().numpy()

    return {
        "cos": float(np.mean(cos)),
        "l2": float(np.mean(l2)),
        "keff": float(np.mean(keff)),
        "flip": float(np.mean(pred_w != pred_u)),
        "score_shift": float(np.mean(np.abs(score_w - score_u))),
        "score_w": score_w,
        "score_u": score_u,
        "labels": qry["labels"].cpu().numpy(),
    }


def auroc(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--pkl-dir", required=True)
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stem, K = manifest["stem"], manifest["k_shot"]
    preset = manifest["config"]["model"].get("preset", "?")

    model = build_model(manifest["config"]["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    model.eval()

    store = RecordStore.from_pkl(
        Path(args.pkl_dir) / f"{stem}_{args.split}.pkl", args.split)
    plan = EpisodePlan.load(Path(args.plan_dir) / f"{stem}_{args.split}_k{K}.json")

    n = min(args.episodes, len(plan.episodes))
    print(f"{run_dir.name}  preset={preset}  K={K}  {n} {args.split} episodes\n")

    recs, sw, su, ys = [], [], [], []
    for i in range(n):
        r = compare_one_episode(model, collate_episode(plan.episodes[i], store, device))
        if r is None:
            continue
        recs.append(r)
        sw.append(r["score_w"]); su.append(r["score_u"]); ys.append(r["labels"])
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{n}")

    if not recs:
        print("no usable episodes"); return 1

    def agg(key):
        v = np.array([r[key] for r in recs], float)
        return {"mean": float(v.mean()), "sd": float(v.std()),
                "p05": float(np.percentile(v, 5)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95))}

    sw, su, ys = np.concatenate(sw), np.concatenate(su), np.concatenate(ys)
    a_w, a_u = auroc(sw, ys), auroc(su, ys)

    report = {
        "run": run_dir.name, "preset": preset, "split": args.split,
        "n_episodes": len(recs), "k_shot": K,
        "proto_cos_weighted_vs_uniform": agg("cos"),
        "proto_l2_weighted_vs_uniform": agg("l2"),
        "effective_K": agg("keff"),
        "decision_flip_rate": agg("flip"),
        "score_shift_abs": agg("score_shift"),
        "episode_auroc_weighted": a_w,
        "episode_auroc_uniform": a_u,
        "episode_auroc_delta": a_w - a_u,
    }

    print("\n" + "=" * 72)
    print(f"COUNTERFACTUAL — {run_dir.name}   (same model, weights on vs off)")
    print("=" * 72)
    for k in ("proto_cos_weighted_vs_uniform", "proto_l2_weighted_vs_uniform",
              "effective_K", "decision_flip_rate", "score_shift_abs"):
        v = report[k]
        print(f"  {k:<32} mean {v['mean']:>9.5f}  sd {v['sd']:.5f}  "
              f"p05 {v['p05']:.5f}  p95 {v['p95']:.5f}")
    print()
    print(f"  episode-pooled AUROC, weights ON  : {a_w:.5f}")
    print(f"  episode-pooled AUROC, weights OFF : {a_u:.5f}")
    print(f"  delta                             : {a_w - a_u:+.5f}")

    print("\n" + "-" * 72)
    print("READING")
    print("-" * 72)
    flip = report["decision_flip_rate"]["mean"]
    ke = report["effective_K"]["mean"]
    print(f"  effective support size {ke:.2f} of {K}")
    if ke < K - 0.2:
        print(f"    -> the weighting discards about {K-ke:.2f} of a support example.")
        print("       In few-shot learning that is a real cost, and it has to be")
        print("       repaid by the recency information for the mechanism to break even.")
    print(f"  decision flip rate {flip:.4f}")
    if flip < 0.01:
        print("    -> switching the weighting off changes almost no predictions,")
        print("       so it cannot move AUROC. The flat ablation table follows.")
    else:
        print("    -> the weighting does change predictions; if AUROC is unchanged,")
        print("       it is changing them in both directions about equally.")

    out = Path(args.out) if args.out else run_dir / f"counterfactual_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

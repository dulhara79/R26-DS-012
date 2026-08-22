"""
diagnose_collapse.py — Why does a model emit a constant 0.5?

Supervisor's Phase 1, items 1 and 3. Answers a specific question:

    ProtoNet, ProtoNet+tau, temporal_only and pcw_only all produced
    predictions with p_sd <= 0.0013 and mean_p_case == mean_p_control == 0.5000.
    Is that a weak model, a broken objective, or a collapsed representation?

    python -m scripts.diagnose_collapse \
        --run results/psych_mimic4idx/protonet_k5_seed42 \
        --pkl-dir data/clean/pkl --plan-dir data/clean/plans \
        --split val --episodes 100 --grad-episodes 20

WHAT IT MEASURES, AND WHAT EACH ANSWER MEANS
============================================

1. PROTOTYPE SEPARATION      cos(p_case, p_control) per episode
   ~1.0  -> the two prototypes are the same vector. Every query is equidistant,
            logits are equal, softmax gives exactly 0.5. This is the signature
            of representation collapse and is the most likely explanation.
   <0.9  -> prototypes differ; the problem is downstream, not here.

2. SUPPORT EMBEDDING SPREAD  mean pairwise cos within a class, and across all
   ~1.0  -> the encoder maps every note to nearly the same point. Nothing can
            be separated. Check whether the encoder is receiving gradient.
   lower -> embeddings vary but do not align with the label.

3. DISTANCE GAP              d(q, p_control) - d(q, p_case), split by true label
   The quantity the logits are built from. If its magnitude is ~1e-4 the
   softmax cannot produce spread whatever tau does. If the gap is healthy but
   has the WRONG SIGN, the class-to-column mapping is inverted and AUROC would
   sit just below 0.5 -- which is worth ruling out explicitly, since protonet
   and protonet_temp both landed slightly under 0.5.

4. LOGIT AND PROBABILITY SPREAD
   tau * gap is the logit separation. Reported so `tau` can be exonerated or
   implicated directly rather than by argument.

5. GRADIENT NORMS by group   encoder / projection / temperature / weights
   ~0    on the encoder -> the episodic loss is not training BERT at all, which
         would explain why only the auxiliary-head configuration learned.
   healthy -> gradients flow but the objective has no useful descent direction.

INTERPRETATION IS NOT AUTOMATED. The script prints the numbers and a short
reading guide; it does not decide the diagnosis for you.
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

from tcwpn.collate import collate_episode          # noqa: E402
from tcwpn.model import build_model                # noqa: E402
from tcwpn.sampler import RecordStore, EpisodePlan  # noqa: E402


# ---------------------------------------------------------------------------
def _group_of(param_name: str) -> str:
    if param_name.startswith("embedder.bert."):
        return "encoder"
    if param_name.startswith("embedder."):
        return "projection"
    if "log_temperature" in param_name:
        return "temperature"
    if param_name.startswith("temporal_w.") or param_name.startswith("pcw."):
        return "weight_modules"
    if param_name.startswith("aux_head."):
        return "aux_head"
    return "other"


def _pairwise_cos(x: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity of rows of x."""
    if x.size(0) < 2:
        return float("nan")
    xn = F.normalize(x, dim=-1)
    sim = xn @ xn.t()
    n = sim.size(0)
    off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    return float(off.mean())


@torch.no_grad()
def episode_geometry(model, batch):
    """Reproduce forward() up to the logits, keeping the intermediate geometry."""
    sup, qry = batch["support"], batch["query"]
    sup_emb = model.embedder(sup["input_ids"], sup["attention_mask"],
                             sup["note_index"])
    qry_emb = model.embedder(qry["input_ids"], qry["attention_mask"],
                             qry["note_index"])

    sup_labels = sup["labels"]
    classes = sorted(set(int(v) for v in sup_labels.tolist()))
    protos, per_class_spread = [], {}
    for c in classes:
        mask = sup_labels == c
        proto, _w = model.build_prototype(sup_emb[mask], sup["days"][mask])
        protos.append(proto)
        per_class_spread[c] = _pairwise_cos(sup_emb[mask])

    P = torch.stack(protos, dim=0)
    q = F.normalize(qry_emb, dim=-1)
    dist = torch.cdist(q.unsqueeze(0), P.unsqueeze(0)).squeeze(0) ** 2
    tau = float(torch.exp(model.log_temperature))
    logits = -dist * tau
    probs = F.softmax(logits, dim=-1)

    idx = {c: i for i, c in enumerate(classes)}
    pos = idx.get(1, len(classes) - 1)
    neg = idx.get(0, 0)

    # d(query, negative prototype) - d(query, positive prototype).
    # Positive for a case query means the query sits nearer the case prototype,
    # which is the correct orientation.
    gap = (dist[:, neg] - dist[:, pos]).detach().cpu().numpy()
    qlab = qry["labels"].detach().cpu().numpy()

    return {
        "proto_cos": float(F.cosine_similarity(protos[pos], protos[neg], dim=0))
        if len(protos) == 2 else float("nan"),
        "support_spread_case": per_class_spread.get(1, float("nan")),
        "support_spread_control": per_class_spread.get(0, float("nan")),
        "support_spread_all": _pairwise_cos(sup_emb),
        "query_spread_all": _pairwise_cos(qry_emb),
        "emb_norm_mean": float(sup_emb.norm(dim=-1).mean()),
        "gap_case": float(np.mean(gap[qlab == 1])) if (qlab == 1).any() else float("nan"),
        "gap_control": float(np.mean(gap[qlab == 0])) if (qlab == 0).any() else float("nan"),
        "gap_abs_mean": float(np.mean(np.abs(gap))),
        "logit_spread": float((logits.max(dim=1).values - logits.min(dim=1).values).mean()),
        "p_pos_mean": float(probs[:, pos].mean()),
        "p_pos_sd": float(probs[:, pos].std()),
        "tau": tau,
    }


def gradient_norms(model, batch):
    """One forward/backward; returns L2 grad norm per parameter group."""
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out["loss"].backward()
    acc: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = _group_of(name)
        acc[g] = acc.get(g, 0.0) + float(p.grad.detach().pow(2).sum())
    model.zero_grad(set_to_none=True)
    return {k: float(np.sqrt(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a results/<stem>/<name>_kK_seedS dir")
    ap.add_argument("--pkl-dir", required=True)
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--grad-episodes", type=int, default=20)
    ap.add_argument("--blind", default="none")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg, stem, K = manifest["config"], manifest["stem"], manifest["k_shot"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    suffix = "" if args.blind == "none" else f"_blind-{args.blind}"
    store = RecordStore.from_pkl(
        Path(args.pkl_dir) / f"{stem}_{args.split}{suffix}.pkl", args.split)
    plan = EpisodePlan.load(Path(args.plan_dir) / f"{stem}_{args.split}_k{K}.json")

    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = min(args.episodes, len(plan.episodes))
    print(f"{run_dir.name}: {n} {args.split} episodes on {device}\n")

    geo = []
    for i in range(n):
        batch = collate_episode(plan.episodes[i], store, device)
        geo.append(episode_geometry(model, batch))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{n}")

    def agg(key):
        v = np.array([g[key] for g in geo], dtype=float)
        v = v[~np.isnan(v)]
        return {"mean": float(v.mean()), "sd": float(v.std()),
                "p05": float(np.percentile(v, 5)),
                "p95": float(np.percentile(v, 95))} if len(v) else None

    keys = ["proto_cos", "support_spread_case", "support_spread_control",
            "support_spread_all", "query_spread_all", "emb_norm_mean",
            "gap_case", "gap_control", "gap_abs_mean", "logit_spread",
            "p_pos_mean", "p_pos_sd", "tau"]
    summary = {k: agg(k) for k in keys}

    # gradients need grad enabled
    model.train()
    gnorms: list[dict] = []
    for i in range(min(args.grad_episodes, len(plan.episodes))):
        batch = collate_episode(plan.episodes[i], store, device)
        gnorms.append(gradient_norms(model, batch))
    groups = sorted({g for d in gnorms for g in d})
    grad_summary = {
        g: float(np.mean([d.get(g, 0.0) for d in gnorms])) for g in groups
    }

    report = {"run": run_dir.name, "split": args.split, "blind": args.blind,
              "n_episodes": n, "geometry": summary, "grad_norms": grad_summary}

    # ---------------- printed report ----------------
    print("\n" + "=" * 72)
    print(f"COLLAPSE DIAGNOSTIC — {run_dir.name}")
    print("=" * 72)
    for k in keys:
        s = summary[k]
        if s:
            print(f"  {k:<26} {s['mean']:>10.5f}  (sd {s['sd']:.5f}, "
                  f"p05 {s['p05']:.5f}, p95 {s['p95']:.5f})")
    print("\n  gradient L2 norm by parameter group")
    for g, v in grad_summary.items():
        print(f"    {g:<24} {v:.6e}")

    print("\n" + "-" * 72)
    print("READING GUIDE")
    print("-" * 72)
    pc = summary["proto_cos"]["mean"] if summary["proto_cos"] else float("nan")
    sa = summary["support_spread_all"]["mean"] if summary["support_spread_all"] else float("nan")
    gap = summary["gap_abs_mean"]["mean"] if summary["gap_abs_mean"] else float("nan")
    gcase = summary["gap_case"]["mean"] if summary["gap_case"] else float("nan")
    gctrl = summary["gap_control"]["mean"] if summary["gap_control"] else float("nan")
    enc = grad_summary.get("encoder", float("nan"))

    print(f"  cos(p_case, p_control) = {pc:.5f}")
    if pc > 0.99:
        print("    -> the two prototypes are effectively the same vector.")
        print("       Every query is equidistant, so p = 0.5 exactly. This is")
        print("       representation collapse, not a weak classifier.")
    elif pc > 0.9:
        print("    -> prototypes are close but distinguishable.")
    else:
        print("    -> prototypes are well separated; look further down.")

    print(f"  mean pairwise cos across all support notes = {sa:.5f}")
    if sa > 0.99:
        print("    -> the encoder maps every note to nearly one point.")

    print(f"  |distance gap| = {gap:.3e}")
    if gap < 1e-3:
        print("    -> the quantity the logits are built from is numerically")
        print("       negligible; no value of tau can rescue this.")

    if not (np.isnan(gcase) or np.isnan(gctrl)):
        print(f"  gap by class: case {gcase:+.3e}, control {gctrl:+.3e}")
        if gcase < gctrl:
            print("    -> WRONG SIGN: cases sit nearer the CONTROL prototype.")
            print("       Check the class-to-column mapping before anything else;")
            print("       this alone produces AUROC just below 0.5.")

    print(f"  encoder gradient norm = {enc:.3e}")
    if enc < 1e-6:
        print("    -> the encoder receives essentially no gradient from this")
        print("       objective. That is consistent with only the auxiliary-head")
        print("       configuration having learned anything.")

    out = Path(args.out) if args.out else run_dir / f"collapse_diagnostic_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

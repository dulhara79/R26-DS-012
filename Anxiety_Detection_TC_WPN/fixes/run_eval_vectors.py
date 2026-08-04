"""
run_eval_vectors.py
Generate patient-level (y_true, y_prob) vectors for the DeLong test — LOCALLY.
Author: (for Dulhara Kaushalya's TC-WPN pipeline)

WHY: delong_pipeline.py needs one .npz per model containing the per-patient
score vector. That requires running the trained model over the test episodes.
This script does exactly that — on CPU if you have no GPU (slow but fine for a
one-off, ~10-30 min). It reproduces evaluate_patient_level()'s pooling EXACTLY
(mean prob per patient, key = "label_subjectid").

RUN (from the local project root, see folder structure in the chat answer):
  python fixes/run_eval_vectors.py \
      --repo  ./tc_wpn \
      --pkl   ./data/mimic_anxiety_test_real_world.pkl \
      --ckpt  ./checkpoints/best_model.pt \
      --out   ./vectors/tcwpn_rw_k5.npz \
      --k 5 --n_episodes 600

For the Standard ProtoNet baseline, point --ckpt at proto_baseline_best.pt
ONLY IF that checkpoint was built from the same TCWPN class (your ablation
builds ProtoNet as TCWPN with lambda/beta -> 0, so it loads). If your baseline
notebook used a different class, generate its .npz inside that notebook instead
(same 3-line np.savez), then just run delong_pipeline locally.
"""
import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict


def collect_patient_vectors(model, dataset, n_episodes, device, k_shot=5):
    """Faithful copy of evaluate_patient_level()'s pooling.
    Returns (patient_ids, y_true, y_prob) aligned per patient."""
    model.eval()
    dataset.k_shot = k_shot
    dataset.q_query = k_shot
    dataset.total_needed = k_shot * 2

    probs_by_patient = defaultdict(list)
    label_by_patient = {}

    with torch.no_grad():
        for ep_idx in range(n_episodes):
            try:
                ep = dataset.sample_episode()
                out = model(ep)
            except Exception as e:
                if ep_idx < 3:
                    print(f"  [warn] episode {ep_idx} failed: {e}")
                continue
            probs = out["probs"].cpu().float()
            targets = out["targets"].cpu()
            classes = ep["classes"]
            anx_idx = classes.index(1) if 1 in classes else 0
            class_q_counts = {c: 0 for c in classes}

            for i, t in enumerate(targets.tolist()):
                true_label = classes[t]
                prob = probs[i, anx_idx].item()
                q_data = ep.get("query", {}).get(true_label, {})
                sids = q_data.get("subject_ids", None)
                local_idx = class_q_counts[true_label]
                if sids is not None and local_idx < len(sids):
                    sid = f"{true_label}_{sids[local_idx]}"
                else:
                    sid = f"ep{ep_idx}_class{true_label}_q{local_idx}"
                class_q_counts[true_label] += 1
                probs_by_patient[sid].append(prob)
                label_by_patient[sid] = true_label

            if (ep_idx + 1) % 100 == 0:
                print(f"  ...{ep_idx+1}/{n_episodes} episodes  "
                      f"({len(label_by_patient)} patients so far)")

    patient_ids = list(label_by_patient.keys())
    y_prob = np.array([float(np.mean(probs_by_patient[s])) for s in patient_ids])
    y_true = np.array([label_by_patient[s] for s in patient_ids], dtype=int)
    print(f"  Collected {len(patient_ids)} patients "
          f"({int(y_true.sum())} anxiety / {int((y_true==0).sum())} control)")
    return patient_ids, y_true, y_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to cloned tc_wpn repo")
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_episodes", type=int, default=600)
    args = ap.parse_args()

    # Make the repo importable (it has src/tc_wpn/...)
    sys.path.insert(0, args.repo)
    sys.path.insert(0, os.path.join(args.repo, "src"))
    from src.tc_wpn.models.core import TCWPN
    from src.tc_wpn.sampler.episode_dataset import TCMIMICEpisodicDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    model = TCWPN().to(device)
    model.load_state_dict(state, strict=False)  # strict=False tolerates head naming
    print("Model loaded.")

    ds = TCMIMICEpisodicDataset(
        args.pkl, k_shot=args.k, q_query=args.k, phase="full",
        min_notes_per_patient=2, max_notes_per_patient=3, max_chunks_per_note=1,
    )

    pids, yt, yp = collect_patient_vectors(model, ds, args.n_episodes, device, k_shot=args.k)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, patient_ids=pids, y_true=yt, y_prob=yp)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

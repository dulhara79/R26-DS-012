"""
evaluation.py — Run an episode plan and pool predictions to patient level.

Contract: given (model, store, plan) the output is a dict of
    {"probs": {patient_id: mean P(anxiety)},
     "labels": {patient_id: 0/1},
     "coverage": {patient_id: n_episodes_queried}}
plus the raw per-query rows for auditing.

Because the plan is fixed, two different models produce dicts over the SAME
patient keys, which is what makes the paired DeLong test in metrics.py valid.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .collate import collate_episode


@torch.no_grad()
def run_plan(model, store, plan, device, progress=True, max_episodes=None):
    model.eval()
    probs_by_patient = defaultdict(list)
    labels_by_patient = {}
    rows = []

    episodes = list(plan)
    if max_episodes:
        episodes = episodes[:max_episodes]

    iterator = enumerate(episodes)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = enumerate(tqdm(episodes, desc=f"eval[{store.split_name}]"))
        except ImportError:
            pass

    for ep_i, ep in iterator:
        batch = collate_episode(ep, store, device)
        out = model(batch)
        p_anx = out["p_anxiety"].detach().cpu().numpy()

        q = batch["query"]
        for j, sid in enumerate(q["subject_ids"]):
            label = int(q["labels"][j].item())
            probs_by_patient[sid].append(float(p_anx[j]))
            if sid in labels_by_patient and labels_by_patient[sid] != label:
                raise RuntimeError(
                    f"patient {sid} appears with two different labels; the "
                    f"cohort is inconsistent"
                )
            labels_by_patient[sid] = label
            rows.append({
                "episode": ep_i,
                "subject_id": sid,
                "note_id": q["note_ids"][j],
                "label": label,
                "p_anxiety": float(p_anx[j]),
            })

    return {
        "probs": {k: float(np.mean(v)) for k, v in probs_by_patient.items()},
        "labels": labels_by_patient,
        "coverage": {k: len(v) for k, v in probs_by_patient.items()},
        "rows": rows,
        "plan_meta": plan.meta,
    }


def to_arrays(result: dict):
    """Stable-ordered (patient_ids, y, p) from a run_plan result."""
    ids = sorted(result["probs"].keys())
    y = np.array([result["labels"][k] for k in ids], dtype=int)
    p = np.array([result["probs"][k] for k in ids], dtype=float)
    return ids, y, p


@torch.no_grad()
def quick_val_auroc(model, store, plan, device, max_episodes=60):
    """
    Cheap model-selection signal for the training loop. Deliberately uses a
    small slice of the validation plan; the reported validation numbers in the
    paper come from the full plan via run_plan().
    """
    from sklearn.metrics import roc_auc_score

    res = run_plan(model, store, plan, device, progress=False,
                   max_episodes=max_episodes)
    _, y, p = to_arrays(res)
    if y.sum() == 0 or (y == 0).sum() == 0:
        return 0.5
    return float(roc_auc_score(y, p))

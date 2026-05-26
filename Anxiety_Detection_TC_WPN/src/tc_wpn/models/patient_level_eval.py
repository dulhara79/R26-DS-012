"""
patient_level_eval.py
TC-WPN Publication-Grade Patient-Level Evaluation
Author: Dulhara Kaushalya

WHY PATIENT-LEVEL EVALUATION IS REQUIRED FOR PUBLICATION:
==========================================================
The previous episodic evaluation (pooled AUROC across 600 episodes) has
a critical flaw: the same patient can appear in the query set of many
different episodes. With 1,739 test records and 600 episodes each drawing
~6 query samples, each patient appears in ~2-3 episodes on average.

Consequence: bootstrap CI is computed over dependent (non-iid) samples.
The CI appears tight but is overconfident. Clinical NLP reviewers at CHIL,
AMIA, and JAMIA universally require patient-level metrics.

THIS MODULE IMPLEMENTS:
  evaluate_patient_level() — runs N episodes, aggregates predictions per
                             patient (mean pooling), computes AUROC/F1/PR-AUC
                             once over the patient-level vector.
                             Bootstrap CI resamples patients (independent units).
  evaluate_episodic()      — legacy episodic pooling. Use ONLY for val loop
                             during training (model selection). Never report
                             this as a test result.

USAGE:
    from tc_wpn.models.patient_level_eval import (
        evaluate_patient_level,
        evaluate_episodic,
    )
"""

import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)


# =============================================================================
# PATIENT-LEVEL EVALUATION — PRIMARY FUNCTION
# =============================================================================
@torch.no_grad()
def evaluate_patient_level(
    model,
    dataset,
    n_episodes: int,
    device: torch.device,
    fixed_threshold: float = None,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    label: str = "",
):
    """
    Publication-grade patient-level evaluation.

    Collects query predictions across n_episodes. For each patient,
    averages all probabilities from every episode they appeared in as a
    query (mean pooling). Then computes AUROC, F1, PR-AUC ONCE over the
    patient-level probability vector.

    Patient identification uses subject_ids from ep["query"][label]["subject_ids"]
    which are provided by episode_dataset_v3.py. If subject_ids are missing
    (e.g., old dataset code), falls back to episode-indexed keys — this still
    avoids double-counting within one episode but cannot pool across episodes.

    Args:
        model:           TCWPN model
        dataset:         TCMIMICEpisodicDataset (v3, with subject_id tracking)
        n_episodes:      number of episodes to collect predictions from
        device:          torch device
        fixed_threshold: classification threshold locked from val set
                         (do NOT pass None at test time — threshold must not
                          be tuned on the test set)
        bootstrap_ci:    whether to compute 95% CI via patient bootstrap
        n_bootstrap:     number of bootstrap iterations (1000 recommended)
        label:           optional string label for logging

    Returns dict with:
        auroc, pr_auc, macro_f1, threshold
        n_patients, n_anxiety, n_control, prevalence
        episode_coverage_mean (avg episodes a patient appeared in)
        auroc_ci_lower, auroc_ci_upper (if bootstrap_ci=True)
    """
    model.eval()

    all_probs_by_patient = defaultdict(list)
    all_labels_by_patient = {}

    for ep_idx in range(n_episodes):
        try:
            ep = dataset.sample_episode()
            out = model(ep)
        except Exception:
            continue

        probs = out["probs"].cpu().float()
        targets = out["targets"].cpu()
        classes = ep["classes"]
        anx_idx = classes.index(1) if 1 in classes else 0

        # Counter to correctly index into per-class subject_ids list
        class_query_counts = {c: 0 for c in classes}

        for i, t in enumerate(targets.tolist()):
            true_label = classes[t]
            prob = probs[i, anx_idx].item()

            # Retrieve subject_id from episode query metadata
            # ep["query"][true_label]["subject_ids"] is set by episode_dataset_v3
            q_data = ep.get("query", {}).get(true_label, {})
            sids = q_data.get("subject_ids", None)
            local_idx = class_query_counts[true_label]

            if sids is not None and local_idx < len(sids):
                # Prefix with label to prevent subject_id collision across classes
                # (same subject_id cannot appear as both anxiety and control,
                # but prefix makes the key unambiguous)
                sid = f"{true_label}_{sids[local_idx]}"
            else:
                # Fallback: unique per episode query slot — no cross-episode pooling
                sid = f"ep{ep_idx}_class{true_label}_q{local_idx}"

            class_query_counts[true_label] += 1
            all_probs_by_patient[sid].append(prob)
            all_labels_by_patient[sid] = true_label

    if len(all_labels_by_patient) < 10:
        return {
            "auroc": 0.5,
            "macro_f1": 0.0,
            "pr_auc": 0.5,
            "threshold": fixed_threshold if fixed_threshold is not None else 0.5,
            "n_patients": len(all_labels_by_patient),
            "n_anxiety": 0,
            "n_control": 0,
            "prevalence": 0.0,
            "episode_coverage_mean": 0.0,
        }

    # Aggregate: one probability per patient (mean across episodes)
    patient_ids = list(all_labels_by_patient.keys())
    p = np.array([float(np.mean(all_probs_by_patient[sid])) for sid in patient_ids])
    t = np.array([all_labels_by_patient[sid] for sid in patient_ids])
    coverage = np.array([len(all_probs_by_patient[sid]) for sid in patient_ids])

    n_anxiety = int(t.sum())
    n_control = int((t == 0).sum())
    prevalence = float(n_anxiety / max(len(t), 1))

    # Metrics computed ONCE over patient-level vectors
    try:
        auroc = float(roc_auc_score(t, p))
    except Exception:
        auroc = 0.5

    try:
        pr_auc = float(average_precision_score(t, p))
    except Exception:
        pr_auc = 0.5

    if fixed_threshold is not None:
        thr = fixed_threshold
    else:
        # Only used during val loop where threshold optimisation on val is acceptable
        precs, recs, thrs = precision_recall_curve(t, p)
        f1s = 2 * precs * recs / (precs + recs + 1e-10)
        thr = float(thrs[np.argmax(f1s[:-1])]) if len(thrs) > 0 else 0.5

    macro_f1 = float(
        f1_score(t, (p >= thr).astype(int), average="macro", zero_division=0)
    )

    res = {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "macro_f1": macro_f1,
        "threshold": thr,
        "n_patients": len(t),
        "n_anxiety": n_anxiety,
        "n_control": n_control,
        "prevalence": prevalence,
        "episode_coverage_mean": float(coverage.mean()),
    }

    # Bootstrap CI — resamples PATIENTS (independent units)
    if bootstrap_ci and len(t) >= 20:
        rng = np.random.default_rng(42)
        boot_aurocs = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(p), len(p))
            try:
                if t[idx].sum() > 0 and (t[idx] == 0).sum() > 0:
                    boot_aurocs.append(roc_auc_score(t[idx], p[idx]))
            except Exception:
                pass
        if boot_aurocs:
            res["auroc_ci_lower"] = float(np.percentile(boot_aurocs, 2.5))
            res["auroc_ci_upper"] = float(np.percentile(boot_aurocs, 97.5))

    return res


# =============================================================================
# EPISODE-LEVEL EVALUATE — val loop only, NOT for test reporting
# =============================================================================
@torch.no_grad()
def evaluate_episodic(
    model,
    dataset,
    n_episodes: int,
    device: torch.device,
    fixed_threshold: float = None,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 500,
):
    """
    Episodic pooled evaluation — for use in the VALIDATION LOOP ONLY.

    Pools all query predictions across n_episodes into one flat array,
    then computes AUROC/F1 over that array. Fast, but statistically
    incorrect for test reporting (dependent samples, inflated CI precision).

    USE FOR:    model selection during training (val every 100 episodes)
    DO NOT USE: test set reporting in the paper
    """
    model.eval()
    all_probs, all_targets = [], []

    for _ in range(n_episodes):
        try:
            ep = dataset.sample_episode()
            out = model(ep)
        except Exception:
            continue

        probs = out["probs"].cpu().float()
        targets = out["targets"].cpu()
        classes = ep["classes"]
        anx_idx = classes.index(1) if 1 in classes else 0

        for i, t in enumerate(targets.tolist()):
            all_probs.append(probs[i, anx_idx].item())
            all_targets.append(classes[t])

    if len(all_probs) < 20:
        return {
            "auroc": 0.5,
            "macro_f1": 0.0,
            "pr_auc": 0.5,
            "threshold": 0.5,
            "n_samples": 0,
        }

    p = np.array(all_probs)
    t = np.array(all_targets)

    try:
        auroc = float(roc_auc_score(t, p))
    except Exception:
        auroc = 0.5
    try:
        pr_auc = float(average_precision_score(t, p))
    except Exception:
        pr_auc = 0.5

    if fixed_threshold is not None:
        thr = fixed_threshold
    else:
        precs, recs, thrs = precision_recall_curve(t, p)
        f1s = 2 * precs * recs / (precs + recs + 1e-10)
        thr = float(thrs[np.argmax(f1s[:-1])]) if len(thrs) > 0 else 0.5

    macro_f1 = float(
        f1_score(t, (p >= thr).astype(int), average="macro", zero_division=0)
    )

    res = {
        "auroc": auroc,
        "macro_f1": macro_f1,
        "pr_auc": pr_auc,
        "threshold": thr,
        "n_samples": len(p),
    }

    if bootstrap_ci and len(p) >= 50:
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(p), len(p))
            try:
                boot.append(roc_auc_score(t[idx], p[idx]))
            except Exception:
                pass
        if boot:
            res["auroc_ci_lower"] = float(np.percentile(boot, 2.5))
            res["auroc_ci_upper"] = float(np.percentile(boot, 97.5))

    return res

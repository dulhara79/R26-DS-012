"""Participant-clustered bootstrap confidence interval."""
import numpy as np
from sklearn.metrics import roc_auc_score


def participant_cluster_bootstrap_auc(
    preds,
    trues,
    groups,
    n_boot=2000,
    seed=42,
):
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    groups = np.asarray(groups)

    rng = np.random.RandomState(seed)
    unique_users = np.unique(groups)
    idx_of = {u: np.where(groups == u)[0] for u in unique_users}

    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_users, len(unique_users), replace=True)
        idx = np.concatenate([idx_of[u] for u in sampled])
        if len(np.unique(trues[idx])) > 1:
            values.append(roc_auc_score(trues[idx], preds[idx]))

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )

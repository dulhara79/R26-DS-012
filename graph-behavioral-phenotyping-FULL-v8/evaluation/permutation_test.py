"""Prevalence-preserving participant-level permutation null."""
from __future__ import annotations

import numpy as np


def participant_label_permutation(data, seed):
    groups = np.array([d.uid for d in data])
    unique_users = np.unique(groups)

    user_labels = np.array(
        [
            int(round(np.mean([d.y.item() for d in data if d.uid == uid])))
            for uid in unique_users
        ],
        dtype=int,
    )

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(user_labels)

    return dict(zip(unique_users, shuffled))


def empirical_p_value(observed, null_draws):
    null_draws = np.asarray(null_draws, dtype=float)
    return float((1 + np.sum(null_draws >= observed)) / (1 + len(null_draws)))

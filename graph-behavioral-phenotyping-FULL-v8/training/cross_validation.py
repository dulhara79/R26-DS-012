"""Participant-grouped repeated cross-validation."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _participant_labels(data):
    y = np.array([int(d.y.item()) for d in data], dtype=int)
    groups = np.array([d.uid for d in data])
    unique_users = np.unique(groups)
    user_y = np.array(
        [int(round(y[groups == u].mean())) for u in unique_users],
        dtype=int,
    )
    return y, groups, unique_users, user_y


def participant_grouped_splits(data, n_splits=5, random_state=0):
    y, groups, unique_users, user_y = _participant_labels(data)
    idx_of = {u: np.where(groups == u)[0] for u in unique_users}

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for tr_u, te_u in splitter.split(unique_users, user_y):
        tr_idx = np.concatenate([idx_of[u] for u in unique_users[tr_u]])
        te_idx = np.concatenate([idx_of[u] for u in unique_users[te_u]])

        overlap = set(groups[tr_idx]) & set(groups[te_idx])
        if overlap:
            raise RuntimeError(f"Participant leakage detected: {overlap}")

        yield tr_idx, te_idx


def grouped_repeated_cv(data, evaluate_fold, n_splits=5, n_repeats=3):
    all_oof = []

    for repeat in range(n_repeats):
        oof = np.full(len(data), np.nan, dtype=float)

        for tr_idx, te_idx in participant_grouped_splits(
            data,
            n_splits=n_splits,
            random_state=repeat,
        ):
            train_data = [data[i] for i in tr_idx]
            test_data = [data[i] for i in te_idx]

            result = evaluate_fold(train_data, test_data, seed=repeat)
            pred = np.asarray(result["preds"], dtype=float).reshape(-1)

            if len(pred) != len(te_idx):
                raise ValueError("Prediction count does not match test fold size.")

            oof[te_idx] = pred

        all_oof.append(oof)

    return np.nanmean(np.vstack(all_oof), axis=0)

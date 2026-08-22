"""Flat behavioral baselines evaluated on participant-grouped folds."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import SEED


def flat_vec(d):
    x = np.nan_to_num(d.x_raw.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate(
        [
            x.mean(axis=0),
            x.std(axis=0),
            [d.num_nodes, d.edge_index.shape[1], getattr(d, "n_days", 0)],
        ]
    )


BASELINES = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=3000,
        C=0.2,
        class_weight="balanced",
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "Gradient Boosting": lambda: GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=SEED,
    ),
}


def fit_predict_baseline(train_data, test_data, make_model):
    X_train = np.asarray([flat_vec(d) for d in train_data])
    y_train = np.asarray([int(d.y.item()) for d in train_data])

    X_test = np.asarray([flat_vec(d) for d in test_data])

    scaler = StandardScaler().fit(X_train)
    model = make_model().fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]

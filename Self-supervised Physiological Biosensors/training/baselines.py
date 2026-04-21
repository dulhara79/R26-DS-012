"""
training/baselines.py
======================
Flattens behavioral graphs into hand-crafted feature vectors and runs
classical ML baselines (Logistic Regression, Random Forest,
Gradient Boosting) using the same k-fold splits as the GNN.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import TRAIN_CONFIG


def graph_to_flat(G, profile: dict) -> list | None:
    """
    Convert a behavioral graph + risk profile into a flat feature vector.

    Features (35 total):
      - 11 graph-level statistics
      - 24 hourly risk probabilities
    """
    all_sr = [s for n in G.nodes for s in G.nodes[n]["stress_readings"]]
    if not all_sr:
        return None

    feats = [
        G.number_of_nodes(),
        G.number_of_edges(),
        np.mean([G.nodes[n]["mean_stress"]        for n in G.nodes]),
        np.mean([G.nodes[n]["high_stress_ratio"]   for n in G.nodes]),
        np.mean([G.nodes[n]["visit_count"]         for n in G.nodes]),
        np.std( [G.nodes[n]["typical_hour"]        for n in G.nodes]),
        np.mean([G.nodes[n]["weekday_ratio"]       for n in G.nodes]),
        np.mean(all_sr),
        np.std(all_sr),
        np.max(all_sr),
        sum(1 for s in all_sr if s >= 3) / len(all_sr),
    ] + [profile[h]["risk_probability"] for h in range(24)]

    return feats


def run_baselines(
    uid_list: list,
    user_graphs: dict,
    all_profiles: dict,
    label_map: dict,
    n_splits: int = TRAIN_CONFIG["cv_folds"],
) -> dict:
    """
    Build flat features and run all baselines with stratified k-fold CV.

    Returns
    -------
    dict  {model_name: {auc, f1, mae}}
    """
    X_flat, y_flat = [], []
    for uid in uid_list:
        feats = graph_to_flat(user_graphs[uid], all_profiles[uid])
        if feats is not None:
            X_flat.append(feats)
            y_flat.append(label_map[uid])

    X_flat = np.array(X_flat)
    y_flat = np.array(y_flat)
    y_bin  = (y_flat >= 0.5).astype(int)

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    kf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=TRAIN_CONFIG["random_state"])
    scaler  = StandardScaler()
    results = {}

    print(f"\n{'Model':<24} {'AUC':>8} {'F1':>8} {'MAE':>8}")
    print("-" * 52)

    for name, clf in baselines.items():
        f_auc, f_f1, f_mae = [], [], []
        for tr_i, te_i in kf.split(X_flat, y_bin):
            Xtr = scaler.fit_transform(X_flat[tr_i])
            Xte = scaler.transform(X_flat[te_i])
            clf.fit(Xtr, y_bin[tr_i])
            probs = clf.predict_proba(Xte)[:, 1]
            preds = clf.predict(Xte)
            f_mae.append(mean_absolute_error(y_flat[te_i], probs))
            f_f1.append(f1_score(y_bin[te_i], preds, zero_division=0))
            if len(np.unique(y_bin[te_i])) > 1:
                f_auc.append(roc_auc_score(y_bin[te_i], probs))
        results[name] = {
            "auc": np.mean(f_auc) if f_auc else float("nan"),
            "f1" : np.mean(f_f1),
            "mae": np.mean(f_mae),
        }
        print(
            f"{name:<24} {results[name]['auc']:>8.4f} "
            f"{results[name]['f1']:>8.4f} {results[name]['mae']:>8.4f}"
        )

    return results

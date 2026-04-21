"""
training/cross_validation.py
=============================
Stratified 5-fold cross-validation and optimal-threshold selection.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, mean_absolute_error, precision_recall_curve, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold

from config import TRAIN_CONFIG
from training.trainer import train_fold


def best_threshold(trues: np.ndarray, preds: np.ndarray) -> float:
    """Find the decision threshold that maximises F1 on the given fold."""
    precisions, recalls, thresholds = precision_recall_curve(
        (trues >= 0.5).astype(int), preds
    )
    f1s    = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_t = thresholds[np.argmax(f1s[:-1])]
    return float(np.clip(best_t, 0.2, 0.8))


def run_cross_validation(
    dataset: list,
    pos_weight,
    device,
    n_splits: int = TRAIN_CONFIG["cv_folds"],
    epochs: int   = TRAIN_CONFIG["epochs"],
    verbose: bool = False,
) -> tuple[pd.DataFrame, list]:
    """
    Run stratified k-fold CV and collect per-fold metrics.

    Returns
    -------
    (cv_df, fold_models)
        cv_df       : DataFrame with columns [fold, auc, f1, mae, hrw_mae, win_acc]
        fold_models : list of trained AnxietyGATv2 instances (one per fold)
    """
    y_labels = np.array([(d.y.item() >= 0.5) for d in dataset]).astype(int)
    kf       = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=TRAIN_CONFIG["random_state"]
    )

    cv_results  = []
    fold_models = []

    print(f"Running {n_splits}-fold cross-validation...\n")
    print(f"{'Fold':<6} {'AUC':>7} {'F1':>7} {'MAE':>7} {'HRW MAE':>9} {'Win Acc':>9}")
    print("-" * 48)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(dataset, y_labels)):
        tr_data = [dataset[i] for i in tr_idx]
        te_data = [dataset[i] for i in te_idx]

        fold_model, preds, trues, hrw_p, hrw_t = train_fold(
            tr_data, te_data, pos_weight, device, epochs=epochs, verbose=verbose
        )
        fold_models.append(fold_model)

        thresh   = best_threshold(trues, preds)
        bin_pred = (preds >= thresh).astype(int)
        bin_true = (trues >= 0.5).astype(int)

        auc     = roc_auc_score(bin_true, preds) if len(np.unique(bin_true)) > 1 else float("nan")
        f1      = f1_score(bin_true, bin_pred, zero_division=0)
        mae     = mean_absolute_error(trues, preds)
        hrw_mae = mean_absolute_error(hrw_t.flatten(), hrw_p.flatten())
        win_acc = sum(
            1 for t, p in zip(hrw_t, hrw_p)
            if abs(int(np.argmax(t)) - int(np.argmax(p))) <= 2
        ) / len(hrw_t)

        cv_results.append({
            "fold": fold + 1, "auc": auc, "f1": f1,
            "mae": mae, "hrw_mae": hrw_mae, "win_acc": win_acc,
        })
        print(f"  optimal threshold: {thresh:.3f}")
        print(f"{fold+1:<6} {auc:>7.3f} {f1:>7.3f} {mae:>7.3f} {hrw_mae:>9.3f} {win_acc:>9.3f}")

    cv_df     = pd.DataFrame(cv_results)
    valid_auc = cv_df["auc"].dropna()

    print(f"\n{'='*52}")
    print(f"  {n_splits}-FOLD CV SUMMARY")
    print(f"{'='*52}")
    print(f"  AUC-ROC  : {valid_auc.mean():.4f} ± {valid_auc.std():.4f}")
    print(f"  F1-Score : {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
    print(f"  MAE      : {cv_df['mae'].mean():.4f} ± {cv_df['mae'].std():.4f}")
    print(f"  HRW MAE  : {cv_df['hrw_mae'].mean():.4f} ± {cv_df['hrw_mae'].std():.4f}")
    print(f"  Win Acc  : {cv_df['win_acc'].mean():.4f} ± {cv_df['win_acc'].std():.4f}")

    return cv_df, fold_models

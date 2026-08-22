"""
evaluation/ablation.py
=======================
Ablation study: systematically mask feature groups and measure
the impact on CV performance.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import TRAIN_CONFIG
from training.trainer import train_fold

FEATURE_GROUPS = {
    "Full model (all 9)"  : list(range(9)),
    "Without stress"      : [0, 1, 2, 3, 7, 8],    # drop idx 4,5,6
    "Without temporal"    : [0, 3, 4, 5, 6, 7, 8],  # drop idx 1,2
    "Without location"    : [1, 2, 4, 5, 6, 7, 8],  # drop idx 0,3
    "Without social"      : list(range(7)),           # drop idx 7,8
}


def run_ablation(
    dataset: list,
    pos_weight: torch.Tensor,
    device,
    epochs: int   = 150,
    n_splits: int = TRAIN_CONFIG["cv_folds"],
) -> dict:
    """
    Mask each feature group in turn and re-run k-fold CV.

    Returns
    -------
    dict  {config_name: {auc, f1, mae}}
    """
    y_labels = np.array([(d.y.item() >= 0.5) for d in dataset]).astype(int)
    kf       = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=TRAIN_CONFIG["random_state"]
    )
    results  = {}

    print("Running ablation study...\n")
    print(f"{'Configuration':<26} {'AUC':>7} {'F1':>7} {'MAE':>7}")
    print("-" * 50)

    for config_name, feat_idx in FEATURE_GROUPS.items():
        # Apply feature mask
        masked = []
        for d in dataset:
            d2   = d.clone()
            mask = torch.zeros(d.x.shape[1])
            mask[feat_idx] = 1.0
            d2.x = d.x * mask.unsqueeze(0)
            masked.append(d2.cpu())

        fold_auc, fold_f1, fold_mae = [], [], []
        for tr_idx, te_idx in kf.split(masked, y_labels):
            tr = [masked[i] for i in tr_idx]
            te = [masked[i] for i in te_idx]
            _, preds, trues, _, _ = train_fold(tr, te, pos_weight, device, epochs=epochs)
            bp = (preds >= 0.5).astype(int)
            bt = (trues >= 0.5).astype(int)
            fold_f1.append(f1_score(bt, bp, zero_division=0))
            fold_mae.append(mean_absolute_error(trues, preds))
            if len(np.unique(bt)) > 1:
                fold_auc.append(roc_auc_score(bt, preds))

        auc = np.mean(fold_auc) if fold_auc else float("nan")
        f1  = np.mean(fold_f1)
        mae = np.mean(fold_mae)
        results[config_name] = {"auc": auc, "f1": f1, "mae": mae}
        print(f"{config_name:<26} {auc:>7.3f} {f1:>7.3f} {mae:>7.3f}")

    return results

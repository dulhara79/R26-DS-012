"""Leakage-free trainer for final Component 2 v8."""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

from torch_geometric.loader import DataLoader

from config import (
    BATCH_SIZE,
    CALIBRATION,
    CALIB_MIN_ISOTONIC,
    INNER_VAL_FRAC,
    PATIENCE,
)
from models.gatv2_model import AnxietyGATv2


def personal_z(raw):
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    return np.clip((raw - mean) / np.where(std < 1e-6, 1.0, std), -4, 4)


def fit_population_stats(train_data):
    rows = np.concatenate([d.x_raw.numpy() for d in train_data], axis=0)
    mean = np.nanmean(rows, axis=0)
    std = np.nanstd(rows, axis=0)

    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((std < 1e-6) | ~np.isfinite(std), 1.0, std)
    return mean, std


def apply_view(data, mean, std, mode="dual"):
    output = []
    for d in data:
        d2 = d.clone()
        raw = np.nan_to_num(d.x_raw.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        personal = personal_z(raw)
        population = np.clip((raw - mean) / std, -4, 4)

        if mode == "dual":
            x = np.concatenate([personal, population], axis=1)
        elif mode == "population":
            x = population
        elif mode == "personal":
            x = personal
        else:
            raise ValueError(f"Unknown view mode: {mode}")

        d2.x = torch.tensor(x, dtype=torch.float)
        output.append(d2.cpu())
    return output


def _epoch(model, loader, pos_weight, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    losses = []
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for batch in loader:
            batch = batch.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(batch).flatten()
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.y.flatten(),
                pos_weight=pos_weight.expand(logits.shape[0]),
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []

    for batch in loader:
        batch = batch.to(device)
        p = torch.sigmoid(model(batch).squeeze(-1))
        preds.extend(p.detach().cpu().numpy().reshape(-1))
        trues.extend(batch.y.detach().cpu().numpy().reshape(-1))

    return np.asarray(preds), np.asarray(trues)


def fit_threshold(trues, preds):
    y = (np.asarray(trues) >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y, preds)
    if len(thresholds) == 0:
        return 0.5

    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(np.clip(thresholds[np.argmax(f1[:-1])], 0.2, 0.8))


def _fit_calibrator(val_preds, val_true):
    y = (np.asarray(val_true) >= 0.5).astype(int)

    if CALIBRATION == "none" or len(np.unique(y)) < 2 or len(y) < 30:
        return lambda p: p

    if len(y) >= CALIB_MIN_ISOTONIC:
        model = IsotonicRegression(out_of_bounds="clip").fit(val_preds, y)
        return lambda p: np.clip(model.predict(p), 1e-6, 1 - 1e-6)

    model = LogisticRegression(max_iter=1000)
    model.fit(np.asarray(val_preds).reshape(-1, 1), y)
    return lambda p: model.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def train_fold(
    raw_train,
    raw_test,
    *,
    device,
    node_feat,
    hidden=32,
    heads=2,
    drop=0.50,
    epochs=80,
    seed=42,
    view_mode="dual",
):
    """Train with test data untouched until the final prediction call."""
    raw_train = [d.cpu() for d in raw_train]
    raw_test = [d.cpu() for d in raw_test]

    y = np.array([int(d.y.item()) for d in raw_train], dtype=int)
    idx = np.arange(len(raw_train))

    strat = y if len(np.unique(y)) > 1 and np.bincount(y).min() >= 2 else None
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=INNER_VAL_FRAC,
        random_state=seed,
        stratify=strat,
    )

    inner_train_raw = [raw_train[i] for i in tr_idx]
    inner_val_raw = [raw_train[i] for i in va_idx]

    # Training-only population statistics.
    mean, std = fit_population_stats(inner_train_raw)

    inner_train = apply_view(inner_train_raw, mean, std, view_mode)
    inner_val = apply_view(inner_val_raw, mean, std, view_mode)
    test = apply_view(raw_test, mean, std, view_mode)

    train_y = y[tr_idx]
    n_pos = max(int(train_y.sum()), 1)
    n_neg = max(int(len(train_y) - n_pos), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float, device=device)

    torch.manual_seed(seed)
    model = AnxietyGATv2(
        node_feat=node_feat,
        hidden=hidden,
        heads=heads,
        drop=drop,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=15,
        factor=0.5,
    )

    train_loader = DataLoader(inner_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(inner_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    best_loss = float("inf")
    best_state = None
    stale = 0

    for _ in range(epochs):
        _epoch(model, train_loader, pos_weight, device, optimizer)
        val_loss = _epoch(model, val_loader, pos_weight, device)
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibration and threshold use only inner validation.
    val_pred, val_true = predict(model, val_loader, device)
    calibrate = _fit_calibrator(val_pred, val_true)
    threshold = fit_threshold(val_true, calibrate(val_pred))

    # First and only use of test fold for model assessment.
    test_pred_raw, test_true = predict(model, test_loader, device)
    test_pred = calibrate(test_pred_raw)

    return {
        "model": model,
        "preds": test_pred,
        "trues": test_true,
        "threshold": threshold,
    }

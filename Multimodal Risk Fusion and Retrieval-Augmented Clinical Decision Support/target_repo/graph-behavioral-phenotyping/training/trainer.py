"""
training/trainer.py
====================
Per-epoch training and evaluation loops, plus the single-fold training
function that incorporates SMOTE oversampling inside each fold to
prevent data leakage.
"""

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from config import TRAIN_CONFIG
from models.gatv2_model import AnxietyGATv2
from models.loss import compute_loss


def train_epoch(model: AnxietyGATv2, loader: DataLoader, optimizer, pos_weight, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        v, h  = model(batch)
        loss  = compute_loss(v, h, batch, pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model: AnxietyGATv2, loader: DataLoader, pos_weight, device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        v, h  = model(batch)
        total += compute_loss(v, h, batch, pos_weight).item()
    return total / max(len(loader), 1)


def train_fold(
    train_data: list,
    test_data: list,
    pos_weight: torch.Tensor,
    device,
    epochs: int = TRAIN_CONFIG["epochs"],
    verbose: bool = False,
) -> tuple:
    """
    Train one cross-validation fold with SMOTE oversampling inside the fold.

    Parameters
    ----------
    train_data : list of PyG Data objects for training
    test_data  : list of PyG Data objects for evaluation
    pos_weight : class-imbalance weight tensor
    device     : torch device
    epochs     : maximum training epochs
    verbose    : print loss every 40 epochs

    Returns
    -------
    (model, preds, trues, hrw_preds, hrw_trues)
    All numpy arrays.
    """
    from imblearn.over_sampling import SMOTE

    train_data = [d.cpu() for d in train_data]
    test_data  = [d.cpu() for d in test_data]

    # ── SMOTE: operate on a compact feature summary, then duplicate graphs ─
    X_fold = np.array([[
        d.x.mean(0)[4].item(),
        d.x.mean(0)[5].item(),
        d.x.shape[0] / 100,
        d.edge_index.shape[1] / 500,
        d.hourly_risk.max().item(),
        d.hourly_risk.mean().item(),
    ] for d in train_data])

    y_fold = np.array([(d.y.item() >= 0.5) for d in train_data]).astype(int)

    if y_fold.sum() >= 2 and (y_fold == 0).sum() >= 2:
        try:
            k         = min(3, int(y_fold.sum()) - 1)
            sm        = SMOTE(random_state=42, k_neighbors=k)
            _, y_res  = sm.fit_resample(X_fold, y_fold)
            n_needed  = int(y_res.sum()) - int(y_fold.sum())
            minority  = np.where(y_fold == 1)[0]
            extra     = [
                train_data[i].clone()
                for i in np.random.choice(minority, n_needed, replace=True)
            ]
            train_data = train_data + extra
        except Exception as e:
            if verbose:
                print(f"  SMOTE skipped: {e}")

    # ── Model, optimiser, scheduler ──────────────────────────────────────────
    model = AnxietyGATv2().to(device)
    opt   = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        patience=TRAIN_CONFIG["lr_patience"],
        factor=TRAIN_CONFIG["lr_factor"],
    )

    t_loader = DataLoader(train_data, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,  pin_memory=False)
    e_loader = DataLoader(test_data,  batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, pin_memory=False)

    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        tr = train_epoch(model, t_loader, opt, pos_weight, device)
        vl = eval_epoch(model, t_loader, pos_weight, device)
        sch.step(vl)
        if tr < best_loss:
            best_loss  = tr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= TRAIN_CONFIG["patience"]:
            break
        if verbose and epoch % 40 == 0:
            print(f"  epoch {epoch:3d}  loss={tr:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    preds, trues, hrw_p, hrw_t = [], [], [], []
    with torch.no_grad():
        for batch in e_loader:
            batch = batch.to(device)
            v, h  = model(batch)
            bs    = v.squeeze(-1).shape[0]
            preds.extend(torch.sigmoid(v.squeeze(-1)).cpu().numpy().reshape(-1))
            trues.extend(batch.y.squeeze(-1).cpu().numpy().reshape(-1))
            hrw_p.extend(h.cpu().numpy())
            hrw_t.extend(batch.hourly_risk.view(bs, 24).cpu().numpy())

    return (
        model,
        np.array(preds),
        np.array(trues),
        np.array(hrw_p),
        np.array(hrw_t),
    )

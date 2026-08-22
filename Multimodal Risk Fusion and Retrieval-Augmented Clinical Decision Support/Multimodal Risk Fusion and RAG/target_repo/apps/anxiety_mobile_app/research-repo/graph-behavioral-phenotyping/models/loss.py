"""
models/loss.py
===============
Combined loss function for the AnxietyGATv2 model:
  L = BCE(vuln_logit, y)  +  0.5 * MSE(hrw_pred, hourly_risk_true)
"""

import torch
import torch.nn.functional as F


def compute_loss(
    vuln_logit: torch.Tensor,
    hrw_pred: torch.Tensor,
    batch,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Parameters
    ----------
    vuln_logit : (B, 1) raw vulnerability logits
    hrw_pred   : (B, 24) predicted hourly risk probabilities
    batch      : PyG Batch object (must have .y and .hourly_risk)
    pos_weight : class-imbalance weight for the positive class

    Returns
    -------
    Scalar combined loss tensor.
    """
    vs = vuln_logit.squeeze(-1)
    y  = batch.y.squeeze(-1)
    bs = vs.shape[0]

    l_v = F.binary_cross_entropy_with_logits(
        vs, y, pos_weight=pos_weight.expand(bs)
    )

    hrw_true = batch.hourly_risk.view(bs, 24)
    l_h      = F.mse_loss(hrw_pred, hrw_true)

    return l_v + 0.5 * l_h

"""Final v8 uses class-weighted binary cross entropy."""

import torch.nn.functional as F


def weighted_bce_with_logits(logits, targets, pos_weight):
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight.expand(logits.shape[0]),
    )

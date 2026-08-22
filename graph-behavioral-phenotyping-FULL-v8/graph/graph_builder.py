"""28-day day × segment temporal-lattice construction."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from config import MIN_DAYS, MIN_NODES, WINDOW_DAYS


def build_temporal_lattice(
    feature_array: np.ndarray,
    feature_dates: np.ndarray,
    label_date,
    y: int,
    uid: str,
    cohort: str,
    n_segments: int,
    n_bases: int,
):
    """Create one graph for one labelled week.

    ``feature_array`` must be ordered as
    ``[day, segment_1_features..., segment_2_features..., ...]`` with
    ``n_segments * n_bases`` columns.
    """
    label_date = np.datetime64(label_date)
    mask = (
        (feature_dates > label_date - np.timedelta64(WINDOW_DAYS, "D"))
        & (feature_dates <= label_date)
    )
    n_days = int(mask.sum())
    if n_days < MIN_DAYS:
        return None

    window = feature_array[mask].reshape(n_days, n_segments, n_bases)
    valid = ~np.isnan(window)
    keep = valid.any(axis=2)
    if int(keep.sum()) < MIN_NODES:
        return None

    day_idx, seg_idx = np.nonzero(keep)

    values = np.nan_to_num(
        window[day_idx, seg_idx],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    missingness_mask = valid[day_idx, seg_idx].astype(np.float32)
    x_raw = np.concatenate([values, missingness_mask], axis=1).astype(np.float32)

    pos = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(day_idx, seg_idx))}
    src, dst = [], []

    for (day, seg), i in pos.items():
        # Adjacent segment in the same day.
        same_day = (day, seg + 1)
        # Same segment on the following day.
        next_day = (day + 1, seg)

        for neighbor in (same_day, next_day):
            j = pos.get(neighbor)
            if j is not None:
                src.extend([i, j])
                dst.extend([j, i])

    if len(src) < 2:
        return None

    data = Data(
        x=torch.zeros((x_raw.shape[0], 1), dtype=torch.float),
        x_raw=torch.from_numpy(x_raw),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        y=torch.tensor([float(y)], dtype=torch.float),
        num_nodes=x_raw.shape[0],
    )
    data.uid = uid
    data.cohort = cohort
    data.n_days = n_days
    data.label_date = str(label_date)
    return data

"""
graph/graph_to_pyg.py
======================
Converts a NetworkX behavioral graph into a PyTorch Geometric Data object
and computes social interaction features (conversation + phone-lock).
"""

import numpy as np
import torch
from torch_geometric.data import Data


def get_social_features(
    uid: str,
    typical_hour: float,
    all_conversation: dict,
    all_phonelock: dict,
) -> tuple[float, float]:
    """
    Compute normalized conversation-duration and phone-unlock features
    for the hour closest to *typical_hour*.

    Returns
    -------
    (conv_norm, lock_norm)  both in [0, 1]
    """
    h = int(round(typical_hour)) % 24

    conv_norm = 0.0
    if uid in all_conversation:
        conv_df    = all_conversation[uid]
        hour_conv  = conv_df[conv_df["hour"] == h]["duration_min"].sum()
        total_conv = conv_df["duration_min"].sum()
        conv_norm  = min(hour_conv / max(total_conv, 1), 1.0)

    lock_norm = 0.0
    if uid in all_phonelock:
        lock_df    = all_phonelock[uid]
        hour_lock  = len(lock_df[lock_df["hour"] == h])
        total_lock = len(lock_df)
        lock_norm  = min(hour_lock / max(total_lock, 1) * 10, 1.0)

    return conv_norm, lock_norm


def graph_to_pyg(
    G,
    uid: str,
    profile: dict,
    label: float,
    all_conversation: dict,
    all_phonelock: dict,
) -> Data | None:
    """
    Convert a NetworkX behavioral graph to a PyTorch Geometric Data object.

    Node features (9 total)
    -----------------------
    0  visit_count        (÷ 100)
    1  typical_hour       (÷ 24)
    2  hour_std           (÷ 12)
    3  weekday_ratio
    4  mean_stress        (÷ 4)
    5  high_stress_ratio
    6  std_stress         (÷ 3)
    7  conv_norm          (conversation duration ratio)
    8  lock_norm          (phone-unlock frequency ratio)

    Edge features (2 total)
    -----------------------
    0  transition weight  (÷ 10)
    1  avg_gap            (÷ 240 min)

    Returns
    -------
    torch_geometric.data.Data or None if the graph is too small.
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return None

    n2i  = {n: i for i, n in enumerate(nodes)}
    rows = []
    for n in nodes:
        conv_n, lock_n = get_social_features(
            uid, G.nodes[n]["typical_hour"], all_conversation, all_phonelock
        )
        rows.append([
            G.nodes[n]["visit_count"]       / 100.0,
            G.nodes[n]["typical_hour"]      / 24.0,
            G.nodes[n]["hour_std"]          / 12.0,
            G.nodes[n]["weekday_ratio"],
            G.nodes[n]["mean_stress"]       / 4.0,
            G.nodes[n]["high_stress_ratio"],
            G.nodes[n]["std_stress"]        / 3.0,
            conv_n,
            lock_n,
        ])
    x = torch.tensor(rows, dtype=torch.float)

    edges = list(G.edges(data=True))
    if not edges:
        return None

    edge_index = torch.tensor(
        [[n2i[e[0]], n2i[e[1]]] for e in edges],
        dtype=torch.long,
    ).t().contiguous()

    edge_attr = torch.tensor(
        [
            [e[2].get("weight", 1) / 10.0, e[2].get("avg_gap", 30) / 240.0]
            for e in edges
        ],
        dtype=torch.float,
    )

    y           = torch.tensor([label], dtype=torch.float)
    hourly_risk = torch.tensor(
        [profile[h]["risk_probability"] for h in range(24)],
        dtype=torch.float,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        hourly_risk=hourly_risk,
        num_nodes=len(nodes),
        user_id=uid,
    )

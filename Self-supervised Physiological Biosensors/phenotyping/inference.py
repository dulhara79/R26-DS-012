"""
phenotyping/inference.py
=========================
Single-participant inference: build graph → run model → return result dict.
"""

import json

import numpy as np
import torch
from torch_geometric.data import Batch

from graph.graph_to_pyg import graph_to_pyg
from graph.risk_profiler import risk_level


def predict_user(
    uid: str,
    model,
    user_graphs: dict,
    all_profiles: dict,
    label_map: dict,
    uid_list: list,
    clusters: np.ndarray,
    phenotypes: dict,
    all_conversation: dict,
    all_phonelock: dict,
    device,
) -> dict | None:
    """
    Run the full inference pipeline for a single participant.

    Parameters
    ----------
    uid           : participant ID
    model         : trained AnxietyGATv2
    user_graphs   : {uid: nx.DiGraph}
    all_profiles  : {uid: hourly risk profile dict}
    label_map     : {uid: vulnerability label}
    uid_list      : ordered list of participant IDs used during training
    clusters      : cluster assignment array (aligned with uid_list)
    phenotypes    : {cluster_id: name}
    all_conversation / all_phonelock : raw modality dicts
    device        : torch device

    Returns
    -------
    Result dict, or None on failure.
    """
    if uid not in user_graphs or uid not in all_profiles:
        print(f"User {uid} not found.")
        return None

    d = graph_to_pyg(
        user_graphs[uid], uid, all_profiles[uid],
        label_map.get(uid, 0.0),
        all_conversation, all_phonelock,
    )
    if d is None:
        print(f"Could not build graph for {uid}")
        return None

    model.eval()
    with torch.no_grad():
        batch  = Batch.from_data_list([d.to(device)])
        v, h   = model(batch)
        vscore = model.vulnerability_score(v)
        hrw    = model.high_risk_window(hrw_pred=h)

    i_in_list = uid_list.index(uid) if uid in uid_list else -1
    phenotype = phenotypes.get(int(clusters[i_in_list]), "Unknown") \
                if i_in_list >= 0 else "Unknown"

    result = {
        "participant_id"     : uid,
        "vulnerability_score": round(vscore, 4),
        "risk_level"         : risk_level(vscore),
        "high_risk_window"   : hrw,
        "phenotype_cluster"  : phenotype,
    }
    print(json.dumps(result, indent=2))
    return result

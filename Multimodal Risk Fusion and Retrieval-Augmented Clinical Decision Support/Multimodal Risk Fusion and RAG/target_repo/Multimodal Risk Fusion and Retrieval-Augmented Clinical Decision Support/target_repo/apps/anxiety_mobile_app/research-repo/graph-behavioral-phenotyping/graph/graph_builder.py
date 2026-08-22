"""
graph/graph_builder.py
=======================
Constructs per-participant directed behavioral graphs from contextual states.
Nodes = unique contextual states; edges = observed state transitions.
"""

from collections import defaultdict

import networkx as nx
import numpy as np

from config import MAX_EDGE_GAP_MIN


def build_behavioral_graph(
    uid: str,
    ctx_df,
) -> nx.DiGraph | None:
    """
    Build a directed behavioral graph for one participant.

    Parameters
    ----------
    uid    : participant ID
    ctx_df : output of build_contextual_states() — must contain columns
             [contextual_state, stress_level, hour, day_of_week, timestamp]

    Returns
    -------
    nx.DiGraph with node/edge attributes, or None if the graph is too sparse.

    Node attributes
    ---------------
    visit_count, typical_hour, hour_std, weekday_ratio,
    mean_stress, max_stress, std_stress, high_stress_ratio,
    n_stress_obs, stress_readings

    Edge attributes
    ---------------
    weight (transition count), avg_gap (minutes)
    """
    df = ctx_df.copy()
    df = df[df["location_cluster"] != -1]       # drop transit/noise points
    df = df.dropna(subset=["contextual_state"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 10:
        return None

    G = nx.DiGraph()
    G.graph["user_id"] = uid

    states      = df["contextual_state"].tolist()
    stress_vals = df["stress_level"].tolist() if "stress_level" in df.columns else [None] * len(df)
    hours       = df["hour"].tolist()
    days        = df["day_of_week"].tolist()

    # ── Aggregate node statistics ─────────────────────────────────────────────
    node_stats = defaultdict(lambda: {
        "visits": 0, "stress": [], "hours": [], "weekday": 0, "weekend": 0
    })

    for i, state in enumerate(states):
        ns = node_stats[state]
        ns["visits"] += 1
        ns["hours"].append(hours[i])
        if days[i] < 5:
            ns["weekday"] += 1
        else:
            ns["weekend"] += 1
        v = stress_vals[i]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ns["stress"].append(float(v))

    for state, ns in node_stats.items():
        sr = ns["stress"]
        G.add_node(
            state,
            visit_count       = ns["visits"],
            typical_hour      = float(np.mean(ns["hours"])),
            hour_std          = float(np.std(ns["hours"])),
            weekday_ratio     = ns["weekday"] / max(ns["visits"], 1),
            mean_stress       = float(np.mean(sr)) if sr else 0.0,
            max_stress        = float(np.max(sr))  if sr else 0.0,
            std_stress        = float(np.std(sr))  if sr else 0.0,
            high_stress_ratio = sum(1 for s in sr if s >= 3) / max(len(sr), 1),
            n_stress_obs      = len(sr),
            stress_readings   = sr,
        )

    # ── Build edges (temporal transitions) ───────────────────────────────────
    for i in range(len(states) - 1):
        src, dst = states[i], states[i + 1]
        gap_min  = (df["timestamp"].iloc[i + 1] - df["timestamp"].iloc[i]).seconds / 60
        if gap_min > MAX_EDGE_GAP_MIN:
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
        else:
            G.add_edge(src, dst, weight=1, avg_gap=gap_min)

    if G.number_of_nodes() >= 3 and G.number_of_edges() >= 2:
        return G
    return None

"""
graph/risk_profiler.py
=======================
Computes per-hour risk profiles and detects high-risk time windows
from a participant's behavioral graph.
"""

from collections import defaultdict

import numpy as np

from config import STRESS_THRESHOLD, RISK_THRESHOLDS


def risk_level(score: float) -> str:
    """Convert a numeric risk score to a categorical label."""
    if score >= RISK_THRESHOLDS["CRITICAL"]: return "CRITICAL"
    if score >= RISK_THRESHOLDS["HIGH"]:     return "HIGH"
    if score >= RISK_THRESHOLDS["MODERATE"]: return "MODERATE"
    return "LOW"


def compute_hourly_risk_profile(G, threshold: float = STRESS_THRESHOLD) -> dict:
    """
    Aggregate stress readings from graph nodes into an hour-by-hour risk profile.

    Parameters
    ----------
    G         : nx.DiGraph from build_behavioral_graph()
    threshold : stress level >= this is treated as a high-stress event

    Returns
    -------
    dict  {hour (0-23): {mean_stress, risk_probability, n_observations}}
    """
    hourly = defaultdict(list)
    for node, attrs in G.nodes(data=True):
        h = int(round(attrs.get("typical_hour", 12))) % 24
        for s in attrs.get("stress_readings", []):
            hourly[h].append(s)

    profile = {}
    for h in range(24):
        readings = hourly.get(h, [])
        profile[h] = {
            "mean_stress"     : np.mean(readings) if readings else 0.0,
            "risk_probability": (
                np.mean([1 if s >= threshold else 0 for s in readings])
                if readings else 0.0
            ),
            "n_observations"  : len(readings),
        }
    return profile


def detect_top_risk_windows(
    profile: dict,
    top_k: int = 2,
    window_hrs: int = 2,
) -> list[dict]:
    """
    Find the top-k non-overlapping time windows with the highest risk.

    Parameters
    ----------
    profile    : output of compute_hourly_risk_profile()
    top_k      : number of windows to return
    window_hrs : width of each window in hours

    Returns
    -------
    List of dicts: [{start, end, score, obs}, ...]
    """
    risk = np.array([profile[h]["risk_probability"] for h in range(24)])

    windows = []
    for start in range(24):
        hrs   = [(start + i) % 24 for i in range(window_hrs)]
        score = float(np.mean(risk[hrs]))
        obs   = sum(profile[h]["n_observations"] for h in hrs)
        windows.append({
            "start": start,
            "end"  : (start + window_hrs) % 24,
            "score": score,
            "obs"  : obs,
        })

    windows.sort(key=lambda x: x["score"], reverse=True)

    selected, used = [], set()
    for w in windows:
        if w["obs"] < 2:
            continue
        hrs_in_window = [(w["start"] + i) % 24 for i in range(window_hrs)]
        if not any(h in used for h in hrs_in_window):
            selected.append(w)
            used.update(hrs_in_window)
        if len(selected) == top_k:
            break

    return selected

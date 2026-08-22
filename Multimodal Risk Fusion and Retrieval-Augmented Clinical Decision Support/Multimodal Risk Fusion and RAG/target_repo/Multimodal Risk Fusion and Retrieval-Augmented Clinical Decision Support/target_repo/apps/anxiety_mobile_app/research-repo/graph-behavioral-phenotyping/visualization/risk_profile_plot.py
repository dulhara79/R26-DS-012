"""
visualization/risk_profile_plot.py
====================================
Per-participant daily risk profile plots (risk probability + mean stress
overlaid with detected high-risk windows).
"""

import os

import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from graph.risk_profiler import risk_level


def plot_risk_profile(
    uid: str,
    all_profiles: dict,
    all_windows: dict,
    save: bool = True,
):
    """
    Plot a two-panel daily risk profile for one participant.

    Top panel  : hourly risk probability with high-risk windows shaded.
    Bottom panel: observation count (bars) + mean stress (line).

    Parameters
    ----------
    uid          : participant ID
    all_profiles : {uid: hourly risk profile dict}
    all_windows  : {uid: list of high-risk window dicts}
    save         : whether to save the figure to OUTPUT_DIR
    """
    profile = all_profiles[uid]
    windows = all_windows.get(uid, [])
    hours   = list(range(24))
    risk_p  = [profile[h]["risk_probability"] for h in hours]
    mean_s  = [profile[h]["mean_stress"]       for h in hours]
    obs_c   = [profile[h]["n_observations"]    for h in hours]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # ── Top panel: risk probability ───────────────────────────────────────────
    ax1.fill_between(hours, risk_p, alpha=0.2, color="steelblue")
    ax1.plot(hours, risk_p, "o-", color="steelblue", linewidth=2.5, markersize=5)

    colors_w = ["red", "orange"]
    for i, w in enumerate(windows):
        lvl = risk_level(w["score"])
        ax1.axvspan(
            w["start"], w["end"], alpha=0.25, color=colors_w[i],
            label=f"Window {i+1}: {w['start']:02d}:00–{w['end']:02d}:00 [{lvl}] {w['score']:.2f}",
        )
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.5,
                linewidth=1.2, label="Risk threshold 0.5")
    ax1.set_title(f"Personalized Daily Risk Profile — {uid}",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Risk Probability")
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, fontsize=7)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Bottom panel: observations + mean stress ──────────────────────────────
    ax2b = ax2.twinx()
    ax2.bar(hours, obs_c, alpha=0.4, color="gray", label="Observations")
    ax2b.plot(hours, mean_s, "s-", color="crimson", linewidth=2,
              markersize=5, label="Mean stress")
    ax2.set_ylabel("# Observations")
    ax2b.set_ylabel("Mean Stress (1–5)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, fontsize=7)
    ax2.legend(loc="upper left", fontsize=8)
    ax2b.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"risk_profile_{uid}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    plt.show()

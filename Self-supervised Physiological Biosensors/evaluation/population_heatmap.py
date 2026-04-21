"""
evaluation/population_heatmap.py
==================================
Builds the cluster × hour risk heatmap matrix and prints the peak
risk hour for each behavioral phenotype.
"""

import numpy as np


def build_population_heatmap(
    uid_list: list,
    clusters: np.ndarray,
    all_profiles: dict,
    n_clusters: int,
    phenotypes: dict,
) -> np.ndarray:
    """
    Average risk_probability across participants for each (cluster, hour) cell.

    Returns
    -------
    heatmap_avg : numpy array of shape (n_clusters, 24)
    """
    heatmap_data  = np.zeros((n_clusters, 24))
    heatmap_count = np.zeros((n_clusters, 24))

    for i, uid in enumerate(uid_list):
        c_id    = int(clusters[i])
        profile = all_profiles[uid]
        for h in range(24):
            if profile[h]["n_observations"] > 0:
                heatmap_data[c_id, h]  += profile[h]["risk_probability"]
                heatmap_count[c_id, h] += 1

    heatmap_avg = np.where(
        heatmap_count > 0,
        heatmap_data / heatmap_count,
        0,
    )

    print("\nPeak risk hours by phenotype:")
    print("-" * 45)
    for c in range(n_clusters):
        peak_h = int(np.argmax(heatmap_avg[c]))
        peak_v = heatmap_avg[c, peak_h]
        name   = phenotypes.get(c, f"Cluster {c}")
        print(f"  {name:<30}: {peak_h:02d}:00  (risk={peak_v:.3f})")

    return heatmap_avg

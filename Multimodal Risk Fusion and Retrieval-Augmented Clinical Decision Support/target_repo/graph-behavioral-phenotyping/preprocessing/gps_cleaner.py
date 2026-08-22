"""
preprocessing/gps_cleaner.py
=============================
GPS cleaning and stay-point detection via DBSCAN.
"""

import numpy as np
import pandas as pd
from haversine import haversine, Unit
from sklearn.cluster import DBSCAN

from config import (
    GPS_ACCURACY_LIMIT,
    GPS_MAX_SPEED_MPS,
    STAY_POINT_RADIUS_M,
    STAY_POINT_MIN_PTS,
)


def clean_gps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove low-accuracy, out-of-range, and physically implausible GPS points.

    Parameters
    ----------
    df : raw GPS DataFrame from load_gps()

    Returns
    -------
    Cleaned DataFrame with an added `speed_mps` column.
    """
    df = df.copy()
    df = df[df["latitude"].between(-90, 90)]
    df = df[df["longitude"].between(-180, 180)]
    df = df[df["accuracy"] < GPS_ACCURACY_LIMIT]
    df = df.dropna(subset=["latitude", "longitude"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    speeds = [0.0]
    for i in range(1, len(df)):
        try:
            dist = haversine(
                (df.loc[i - 1, "latitude"], df.loc[i - 1, "longitude"]),
                (df.loc[i,     "latitude"], df.loc[i,     "longitude"]),
                unit=Unit.METERS,
            )
            dt = max((df.loc[i, "timestamp"] - df.loc[i - 1, "timestamp"]).seconds, 1)
            speeds.append(dist / dt)
        except Exception:
            speeds.append(0.0)

    df["speed_mps"] = speeds
    return df[df["speed_mps"] < GPS_MAX_SPEED_MPS].reset_index(drop=True)


def detect_stay_points(gps_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cluster GPS points into stay-point locations using DBSCAN (haversine).

    Parameters
    ----------
    gps_df : cleaned GPS DataFrame

    Returns
    -------
    (gps_df_with_clusters, cluster_centers)
        cluster_centers : dict  {cluster_id: {lat, lon, visit_count}}
        Noise points are labelled -1.
    """
    coords  = gps_df[["latitude", "longitude"]].values
    eps_rad = STAY_POINT_RADIUS_M / 6_371_000   # metres → radians

    labels = DBSCAN(
        eps=eps_rad,
        min_samples=STAY_POINT_MIN_PTS,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(np.radians(coords)).labels_

    gps_df = gps_df.copy()
    gps_df["location_cluster"] = labels

    centers = {}
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        centers[cid] = {
            "lat"        : coords[mask, 0].mean(),
            "lon"        : coords[mask, 1].mean(),
            "visit_count": int(mask.sum()),
        }

    return gps_df, centers

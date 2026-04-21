"""
preprocessing/contextual_states.py
====================================
Builds composite contextual-state labels by merging GPS clusters,
activity readings, stress readings, and time-of-day categories.
"""

import pandas as pd


def get_time_category(hour: int) -> str:
    """Map hour of day (0-23) to a named time category."""
    if   0  <= hour < 6:  return "NIGHT"
    elif 6  <= hour < 12: return "MORNING"
    elif 12 <= hour < 17: return "AFTERNOON"
    elif 17 <= hour < 21: return "EVENING"
    else:                 return "LATE_NIGHT"


def build_contextual_states(
    uid: str,
    gps_clustered: pd.DataFrame,
    activity: pd.DataFrame,
    stress: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge GPS, activity, and stress into a single contextual-state DataFrame.

    Each row in the output carries:
      - location_cluster, time_category, activity
      - stress_level (nearest reading within 3 hours)
      - contextual_state  (composite string key used as graph node id)

    Parameters
    ----------
    uid            : participant ID (used for logging only)
    gps_clustered  : output of detect_stay_points()
    activity       : output of load_activity()
    stress         : output of load_stress()

    Returns
    -------
    Merged DataFrame with a `contextual_state` column.
    """
    gps = gps_clustered.copy()
    gps["time_category"] = gps["hour"].apply(get_time_category)

    # Nearest activity reading within 5 minutes
    gps = pd.merge_asof(
        gps.sort_values("timestamp"),
        activity[["timestamp", "activity"]].sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("5min"),
        direction="nearest",
    )
    gps["activity"] = gps["activity"].fillna("UNKNOWN")

    # Nearest stress reading within 3 hours
    gps = pd.merge_asof(
        gps.sort_values("timestamp"),
        stress[["timestamp", "stress_level"]].sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("3hours"),
        direction="nearest",
    )

    gps["contextual_state"] = (
        "LOC_" + gps["location_cluster"].astype(str)
        + "__"  + gps["time_category"]
        + "__"  + gps["activity"]
    )

    return gps

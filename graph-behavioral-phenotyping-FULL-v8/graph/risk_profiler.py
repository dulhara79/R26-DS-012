"""Legacy risk-profile API.

Hourly risk profiling is not a validated output of the final v8 model.
"""

def compute_hourly_risk_profile(*args, **kwargs):
    raise RuntimeError(
        "Hourly risk profiles belong to the exploratory StudentLife pipeline "
        "and are not used by Component 2 v8."
    )

def detect_top_risk_windows(*args, **kwargs):
    raise RuntimeError("High-risk windows are not deployed in Component 2 v8.")

def risk_level(*args, **kwargs):
    raise RuntimeError("Clinical risk levels are not deployed in Component 2 v8.")

"""Legacy compatibility module.

Raw GPS DBSCAN stay-point clustering is not part of the final GLOBEM v8 method.
GLOBEM RAPIDS-derived behavioral features are used instead.
"""

def clean_gps(*args, **kwargs):
    raise RuntimeError("Raw GPS cleaning is not used in Component 2 v8.")

def detect_stay_points(*args, **kwargs):
    raise RuntimeError("DBSCAN stay-point detection is not used in Component 2 v8.")

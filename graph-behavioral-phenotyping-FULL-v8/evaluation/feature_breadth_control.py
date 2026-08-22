"""Feature-breadth control metadata.

The final notebook performs the full high-dimensional RAPIDS evaluation because
it must recover exact label-aligned 28-day windows from the source files.
"""


FINAL_FEATURE_BREADTH_RESULT = {
    "n_features": 5952,
    "auroc": 0.5335,
    "ci": [0.496, 0.571],
    "interpretation": "feature-breadth control; not a mathematical upper bound",
}

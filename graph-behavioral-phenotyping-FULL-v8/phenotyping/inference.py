"""Deployment guard for Component 2 v8."""

FINAL_FUSION_WEIGHT = 0.0


def predict_user(*args, **kwargs):
    raise RuntimeError(
        "Per-user vulnerability/risk inference from the old StudentLife model "
        "is disabled. Final Component 2 v8 recommends fusion weight 0.0."
    )

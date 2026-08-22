"""Final-v8 ablation summary.

The former StudentLife feature-group ablation is not the final analysis.
The final compliance ablation is exposed here for compatibility.
"""
from .compliance_ablation import FINAL_COMPLIANCE_ABLATION


def run_ablation(*args, **kwargs):
    return dict(FINAL_COMPLIANCE_ABLATION)

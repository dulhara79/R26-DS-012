"""Legacy module retained for repository compatibility.

Population phenotype risk heatmaps are not part of the validated final v8
pipeline because phenotype/risk-window deployment was retired.
"""

def build_population_heatmap(*args, **kwargs):
    raise RuntimeError(
        "Population risk heatmaps belong to the legacy StudentLife pipeline."
    )

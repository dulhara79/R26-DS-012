"""Legacy compatibility module.

The final v8 method no longer constructs location × time × activity contextual
states. It constructs a day × segment temporal lattice directly from RAPIDS
features. Use ``graph.graph_builder.build_temporal_lattice``.
"""

def build_contextual_states(*args, **kwargs):
    raise RuntimeError(
        "contextual_states.py belongs to the legacy StudentLife pipeline. "
        "Use graph.graph_builder.build_temporal_lattice for final v8."
    )

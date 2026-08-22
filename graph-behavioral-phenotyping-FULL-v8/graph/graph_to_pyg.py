"""Compatibility helper.

The final temporal-lattice builder already returns a PyTorch Geometric ``Data``
object, so no NetworkX conversion is required.
"""

def graph_to_pyg(graph, *args, **kwargs):
    return graph

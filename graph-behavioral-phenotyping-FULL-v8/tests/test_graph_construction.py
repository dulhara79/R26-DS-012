import numpy as np
from graph.graph_builder import build_temporal_lattice


def test_temporal_lattice_builds_edges_and_masks():
    n_days, n_segments, n_bases = 14, 4, 2
    dates = np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + np.timedelta64(n_days, "D"),
    )

    arr = np.ones((n_days, n_segments * n_bases), dtype=np.float32)

    graph = build_temporal_lattice(
        arr,
        dates,
        np.datetime64("2026-01-14"),
        y=1,
        uid="u1",
        cohort="INS-W_1",
        n_segments=n_segments,
        n_bases=n_bases,
    )

    assert graph is not None
    assert graph.x_raw.shape[1] == n_bases * 2
    assert graph.edge_index.shape[0] == 2

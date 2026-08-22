"""Helpers for leave-one-cohort-out reporting."""


def generalization_gap(pooled_auroc, loco_auroc):
    return float(pooled_auroc - loco_auroc)


def graph_shift_supported(graph_gap, best_baseline_gap, graph_significantly_worse):
    gap_advantage = best_baseline_gap - graph_gap
    return (gap_advantage > 0.02) and (not graph_significantly_worse)

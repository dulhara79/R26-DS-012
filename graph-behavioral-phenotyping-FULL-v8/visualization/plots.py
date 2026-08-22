"""Minimal plotting helpers for final v8 results."""
from pathlib import Path
import matplotlib.pyplot as plt


def plot_model_comparison(output_path=None):
    names = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "GATv2",
    ]
    values = [0.5458, 0.5617, 0.5681, 0.5205]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 0.60)
    ax.set_title("Held-out model comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig

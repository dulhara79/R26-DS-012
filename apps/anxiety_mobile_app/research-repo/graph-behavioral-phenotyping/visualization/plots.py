"""
visualization/plots.py
=======================
High-level plotting functions: model comparison bar chart, UMAP phenotype
scatter, combined confusion matrix, and population risk heatmap.
All figures are saved to OUTPUT_DIR.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import OUTPUT_DIR


def plot_model_comparison(bl_results: dict, gnn_auc: float, gnn_f1: float, gnn_mae: float):
    """Bar chart comparing GNN against classical baselines on AUC, F1, MAE."""
    models_list = list(bl_results.keys()) + ["Our GNN\n(GATv2)"]
    auc_list    = [bl_results[m]["auc"] for m in bl_results] + [gnn_auc]
    f1_list     = [bl_results[m]["f1"]  for m in bl_results] + [gnn_f1]
    mae_list    = [bl_results[m]["mae"] for m in bl_results] + [gnn_mae]
    colors      = ["#4a90d9"] * len(bl_results) + ["#e05c3a"]
    x           = np.arange(len(models_list))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title in zip(axes,
            [auc_list, f1_list, mae_list],
            ["AUC-ROC ↑", "F1-Score ↑", "MAE ↓"]):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.grid(alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Model Comparison — Anxiety Vulnerability Detection",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_phenotype_umap(emb2d: np.ndarray, clusters: np.ndarray, phenotypes: dict):
    """Scatter plot of UMAP-projected embeddings coloured by phenotype cluster."""
    best_k  = len(phenotypes)
    palette = ["#e05c3a", "#4a90d9", "#2ecc71", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(best_k):
        mask = clusters == c
        ax.scatter(
            emb2d[mask, 0], emb2d[mask, 1],
            c=palette[c], label=phenotypes.get(c, f"Cluster {c}"),
            s=100, alpha=0.85, edgecolors="white", linewidths=0.5,
        )
    ax.legend(fontsize=10)
    ax.set_title("Behavioral Phenotype Clusters (UMAP)", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phenotypes_umap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_confusion_matrix(all_true: list, all_pred: list):
    """Heatmap of the combined CV confusion matrix."""
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    ax.set_title("Confusion Matrix (5-fold CV)", fontsize=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_population_heatmap(
    heatmap_avg: np.ndarray,
    phenotypes: dict,
    n_clusters: int,
):
    """Cluster × hour risk heatmap for the full population."""
    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(heatmap_avg, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=8)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([phenotypes.get(c, f"Cluster {c}") for c in range(n_clusters)], fontsize=11)
    plt.colorbar(im, ax=ax, label="Average Risk Probability")
    ax.set_title("Population Risk Heatmap — Hour of Day × Phenotype Cluster",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day", fontsize=11)

    for ci in range(n_clusters):
        for h in range(24):
            val = heatmap_avg[ci, h]
            if val > 0:
                ax.text(h, ci, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.5 else "black")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "population_risk_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")


def plot_ablation(ablation_results: dict):
    """Horizontal bar chart of F1 scores for each ablation configuration."""
    names  = list(ablation_results.keys())
    f1s    = [ablation_results[n]["f1"] for n in names]
    colors = ["#2ecc71" if i == 0 else "#e05c3a" for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, f1s, color=colors, edgecolor="white", linewidth=0.8)
    ax.axvline(f1s[0], color="green", linestyle="--", alpha=0.5, label="Full model")
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Ablation Study — Feature Group Contribution",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, f1s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ablation_study.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {path}")

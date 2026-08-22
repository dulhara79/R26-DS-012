"""
phenotyping/phenotyper.py
==========================
Extracts graph-level embeddings from the trained GATv2 model via a
forward hook, then clusters them with K-Means to discover behavioral
phenotypes. UMAP is used for 2-D visualisation.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch_geometric.loader import DataLoader

from config import PHENOTYPE_LABELS, TRAIN_CONFIG
from models.gatv2_model import AnxietyGATv2
from models.loss import compute_loss


def train_final_model(
    dataset: list,
    pos_weight: torch.Tensor,
    device,
    epochs: int = TRAIN_CONFIG["final_epochs"],
) -> AnxietyGATv2:
    """Train a fresh model on the full dataset for embedding extraction."""
    loader      = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, pin_memory=False)
    final_model = AnxietyGATv2().to(device)
    opt         = torch.optim.Adam(
        final_model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    for epoch in range(epochs):
        final_model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            v, h = final_model(batch)
            compute_loss(v, h, batch, pos_weight).backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            opt.step()
        if epoch % 20 == 0:
            print(f"  epoch {epoch}")

    return final_model


def extract_embeddings(model: AnxietyGATv2, dataset: list, device) -> np.ndarray:
    """
    Extract graph-level embeddings using a forward hook on vuln_head[0].

    Returns
    -------
    numpy array of shape (N, hidden*2)
    """
    emb_list = []

    def _hook(module, input, output):
        emb_list.append(output.detach().cpu())

    handle = model.vuln_head[0].register_forward_hook(_hook)

    model.eval()
    loader = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)
    with torch.no_grad():
        for batch in loader:
            model(batch.to(device))

    handle.remove()
    embeddings = torch.cat(emb_list, dim=0).numpy()
    print(f"Embeddings shape : {embeddings.shape}")
    return embeddings


def cluster_phenotypes(embeddings: np.ndarray) -> tuple[KMeans, np.ndarray, int, float, dict]:
    """
    Fit K-Means for k ∈ {2, 3, 4} and select the best by silhouette score.

    Returns
    -------
    (km_final, clusters, best_k, best_sil, phenotype_labels)
    """
    best_k, best_sil = 3, -1
    for k in [2, 3, 4]:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, lbl) if len(np.unique(lbl)) > 1 else -1
        print(f"  k={k}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = km_final.fit_predict(embeddings)
    phenotypes = PHENOTYPE_LABELS.get(best_k, {i: f"Cluster {i}" for i in range(best_k)})

    print(f"\nBest k={best_k}  silhouette={best_sil:.4f}")
    for c in range(best_k):
        print(f"  {phenotypes.get(c, f'Cluster {c}'):30s}: {np.sum(clusters == c)} users")

    return km_final, clusters, best_k, best_sil, phenotypes


def umap_project(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2-D with UMAP for visualisation."""
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

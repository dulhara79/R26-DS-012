"""
main.py
========
End-to-end pipeline runner for Graph-Based Spatio-Temporal Behavioral
Phenotyping for Personalized Anxiety Vulnerability Mapping.

Run
---
    python main.py

Override paths via environment variables if needed:
    DATASET_PATH=/my/data OUTPUT_DIR=/my/outputs MODELS_DIR=/my/models python main.py
"""

import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch

# ── Config ───────────────────────────────────────────────────────────────────
from config import MODELS_DIR, OUTPUT_DIR, VULNERABILITY_CUTOFF

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}\n")

# ── 1. Data loading ───────────────────────────────────────────────────────────
from preprocessing.data_loader import get_users, load_all_users

USERS = get_users()
print(f"Participants found : {len(USERS)}")
data  = load_all_users(USERS)

all_gps          = data["gps"]
all_activity     = data["activity"]
all_stress       = data["stress"]
all_conversation = data["conversation"]
all_phonelock    = data["phonelock"]

valid_users = sorted(set(all_gps) & set(all_activity) & set(all_stress))
print(f"Valid users        : {len(valid_users)}\n")

# ── 2. GPS cleaning & stay-point detection ────────────────────────────────────
from preprocessing.gps_cleaner import clean_gps, detect_stay_points

print("Cleaning GPS & detecting stay points...")
all_gps_clustered   = {}
all_cluster_centers = {}

for uid in valid_users:
    cleaned = clean_gps(all_gps[uid])
    if len(cleaned) < 100:
        continue
    clustered, centers = detect_stay_points(cleaned)
    all_gps_clustered[uid]   = clustered
    all_cluster_centers[uid] = centers

valid_users = sorted(set(all_gps_clustered) & set(all_activity) & set(all_stress))
print(f"After GPS cleaning : {len(valid_users)} valid users\n")

# ── 3. Contextual states ──────────────────────────────────────────────────────
from preprocessing.contextual_states import build_contextual_states

print("Building contextual states...")
user_ctx = {}
for uid in valid_users:
    try:
        user_ctx[uid] = build_contextual_states(
            uid, all_gps_clustered[uid], all_activity[uid], all_stress[uid]
        )
    except Exception as e:
        print(f"  {uid} failed: {e}")
print(f"Contextual states built for {len(user_ctx)} users\n")

# ── 4. Behavioral graphs ──────────────────────────────────────────────────────
from graph.graph_builder import build_behavioral_graph

print("Building behavioral graphs...")
user_graphs = {}
for uid in valid_users:
    G = build_behavioral_graph(uid, user_ctx[uid])
    if G is not None:
        user_graphs[uid] = G

print(f"Graphs built : {len(user_graphs)} users\n")

# ── 5. Labels (stress EMA) ────────────────────────────────────────────────────
label_map = {}
for uid, G in user_graphs.items():
    all_sr = [s for n in G.nodes for s in G.nodes[n]["stress_readings"]]
    if all_sr:
        label_map[uid] = float(
            np.mean([1 if s >= 3 else 0 for s in all_sr])
        )
print(f"Users labelled : {len(label_map)}\n")

# ── 6. Risk profiles ──────────────────────────────────────────────────────────
from graph.risk_profiler import compute_hourly_risk_profile, detect_top_risk_windows

print("Computing risk profiles...")
all_profiles = {}
all_windows  = {}
for uid, G in user_graphs.items():
    all_profiles[uid] = compute_hourly_risk_profile(G)
    all_windows[uid]  = detect_top_risk_windows(all_profiles[uid])
print(f"Risk profiles computed for {len(all_profiles)} users\n")

# ── 7. PyG dataset ────────────────────────────────────────────────────────────
from graph.graph_to_pyg import graph_to_pyg

dataset, uid_list, skipped = [], [], []

for uid in sorted(user_graphs):
    if uid not in label_map or uid not in all_profiles:
        skipped.append(uid); continue
    d = graph_to_pyg(
        user_graphs[uid], uid, all_profiles[uid], label_map[uid],
        all_conversation, all_phonelock,
    )
    if d is not None:
        dataset.append(d)
        uid_list.append(uid)
    else:
        skipped.append(uid)

print(f"PyG dataset  : {len(dataset)} graphs")
print(f"Node features: {dataset[0].x.shape[1]}")
print(f"Skipped      : {skipped if skipped else 'none'}\n")

labels_arr     = np.array([d.y.item() for d in dataset])
binary_all     = (labels_arr >= VULNERABILITY_CUTOFF).astype(int)
n_high         = binary_all.sum()
n_low          = len(binary_all) - n_high
pos_weight_val = len(binary_all) / (2 * max(n_high, 1))
pos_weight     = torch.tensor([pos_weight_val]).to(device)
print(f"Class balance: {n_high} high / {n_low} low  (pos_weight={pos_weight_val:.3f})\n")

dataset = [d.cpu() for d in dataset]

# ── 8. Cross-validation ───────────────────────────────────────────────────────
from training.cross_validation import run_cross_validation

cv_df, fold_models = run_cross_validation(dataset, pos_weight, device)
valid_auc = cv_df["auc"].dropna()

# ── 9. Baseline comparison ────────────────────────────────────────────────────
from training.baselines import run_baselines

bl_results = run_baselines(uid_list, user_graphs, all_profiles, label_map)

gnn_auc = float(valid_auc.mean())
gnn_f1  = float(cv_df["f1"].mean())
gnn_mae = float(cv_df["mae"].mean())
print(f"\n{'Our GNN (GATv2)':<24} {gnn_auc:>8.4f} {gnn_f1:>8.4f} {gnn_mae:>8.4f}")

# ── 10. Final model + embeddings ──────────────────────────────────────────────
from phenotyping.phenotyper import (
    cluster_phenotypes,
    extract_embeddings,
    train_final_model,
    umap_project,
)

print("\nTraining final model for phenotyping...")
final_model = train_final_model(dataset, pos_weight, device)
embeddings  = extract_embeddings(final_model, dataset, device)

print("\nClustering phenotypes...")
km_final, clusters, best_k, best_sil, PHENOTYPES = cluster_phenotypes(embeddings)
emb2d = umap_project(embeddings)

# ── 11. Visualisations ────────────────────────────────────────────────────────
from visualization.plots import (
    plot_ablation,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_phenotype_umap,
    plot_population_heatmap,
)
from visualization.risk_profile_plot import plot_risk_profile

plot_model_comparison(bl_results, gnn_auc, gnn_f1, gnn_mae)
plot_phenotype_umap(emb2d, clusters, PHENOTYPES)

# Combined confusion matrix (quick re-run at 100 epochs)
from sklearn.model_selection import StratifiedKFold
from training.trainer import train_fold

y_labels                     = (labels_arr >= VULNERABILITY_CUTOFF).astype(int)
kf                           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_true_combined            = []
all_pred_combined            = []
for tr_idx, te_idx in kf.split(dataset, y_labels):
    _, preds, trues, _, _ = train_fold(
        [dataset[i] for i in tr_idx],
        [dataset[i] for i in te_idx],
        pos_weight, device, epochs=100,
    )
    all_true_combined.extend((trues >= 0.5).astype(int))
    all_pred_combined.extend((preds >= 0.5).astype(int))

plot_confusion_matrix(all_true_combined, all_pred_combined)

# Per-user risk profiles (first 3 users)
for uid in sorted(user_graphs.keys())[:3]:
    plot_risk_profile(uid, all_profiles, all_windows)

# Ablation
from evaluation.ablation import run_ablation

ablation_results = run_ablation(dataset, pos_weight, device)
plot_ablation(ablation_results)

# Population heatmap
from evaluation.population_heatmap import build_population_heatmap

heatmap_avg = build_population_heatmap(uid_list, clusters, all_profiles, best_k, PHENOTYPES)
plot_population_heatmap(heatmap_avg, PHENOTYPES, best_k)

# ── 12. Save everything ───────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Model weights
model_path = os.path.join(MODELS_DIR, f"gatv2_final_{ts}.pt")
torch.save({
    "model_state_dict": final_model.state_dict(),
    "model_config"    : {"node_feat": 9, "hidden": 64, "heads": 4, "drop": 0.3},
    "timestamp"       : ts,
}, model_path)

# K-Means
km_path = os.path.join(MODELS_DIR, f"kmeans_{ts}.pkl")
with open(km_path, "wb") as f:
    pickle.dump({
        "kmeans"    : km_final,
        "phenotypes": PHENOTYPES,
        "silhouette": best_sil,
        "clusters"  : clusters.tolist(),
        "uid_list"  : uid_list,
    }, f)

# Embeddings
import numpy as np
emb_path = os.path.join(MODELS_DIR, f"embeddings_{ts}.npy")
np.save(emb_path, embeddings)

# CV results CSV
cv_path = os.path.join(OUTPUT_DIR, f"cv_results_{ts}.csv")
cv_df.to_csv(cv_path, index=False)

# Per-user vulnerability JSON
from graph.risk_profiler import risk_level

output_json = []
for i, uid in enumerate(uid_list):
    G      = user_graphs[uid]
    all_sr = [s for n in G.nodes for s in G.nodes[n]["stress_readings"]]
    vscore = float(np.mean([1 if s >= 3 else 0 for s in all_sr])) if all_sr else 0.0
    ws     = all_windows.get(uid, [])
    hrw    = f"{ws[0]['start']:02d}:00" if ws else "unknown"
    output_json.append({
        "participant_id"    : uid,
        "vulnerability_score": round(vscore, 4),
        "risk_level"        : risk_level(vscore),
        "high_risk_window"  : hrw,
        "phenotype_cluster" : PHENOTYPES.get(int(clusters[i]), "Unknown"),
        "timestamp"         : ts,
    })

json_path = os.path.join(OUTPUT_DIR, f"vulnerability_output_{ts}.json")
with open(json_path, "w") as f:
    json.dump(output_json, f, indent=2)

# Pipeline metadata
meta = {
    "timestamp"   : ts,
    "dataset"     : "StudentLife",
    "n_participants": len(dataset),
    "model"       : "GATv2Conv + GlobalAttention",
    "cv_folds"    : 5,
    "performance" : {
        "auc_mean" : round(float(valid_auc.mean()), 4),
        "auc_std"  : round(float(valid_auc.std()),  4),
        "f1_mean"  : round(float(cv_df["f1"].mean()), 4),
        "mae_mean" : round(float(cv_df["mae"].mean()), 4),
        "silhouette": round(float(best_sil), 4),
    },
    "phenotypes"  : PHENOTYPES,
    "files"       : {
        "model"     : model_path,
        "kmeans"    : km_path,
        "embeddings": emb_path,
        "cv_results": cv_path,
        "json_output": json_path,
    },
}
meta_path = os.path.join(MODELS_DIR, f"pipeline_meta_{ts}.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 55)
print("PIPELINE COMPLETE")
print("=" * 55)
print(f"Model    : {model_path}")
print(f"K-Means  : {km_path}")
print(f"CV CSV   : {cv_path}")
print(f"JSON     : {json_path}")
print(f"Meta     : {meta_path}")
print(f"\nSample output:\n{json.dumps(output_json[0], indent=2)}")

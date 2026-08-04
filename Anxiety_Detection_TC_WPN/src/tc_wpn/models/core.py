# =============================================================================
# core.py
# Temporal-Confidence Weighted Prototypical Network (TC-WPN)
# Publication-Grade Version — AUROC Fix Release
#
# KEY CHANGES FROM core_v2.py:
# 1. RelationModule REMOVED — replaced with prototype-distance classification.
#    Per-example relation scoring was the primary cause of near-uniform logits.
#    Prototype distance is more stable with K=5 clinical notes.
# 2. Learnable temperature scalar added — forces logit spread before softmax.
#    Previous logits were in [-0.05, 0.02] range → softmax ≈ uniform.
# 3. TemporalEncoder applied to SUPPORT only (not query).
#    Query goes through a separate query_proj to avoid GRU distortion.
# 4. Confidence weighting now uses COSINE distance to current prototype
#    estimate rather than prediction entropy — more stable early in training
#    when entropy is uninformative (all predictions ≈ 0.5).
# 5. auxiliary head weight increased from 0.1 → 0.3 to provide a stronger
#    direct supervision signal through the embedding space.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from tc_wpn.models.embedder import ClinicalEmbedder


# =============================================================================
# TEMPORAL ENCODER
# Unchanged from v2 — processes chronologically sorted support trajectories
# =============================================================================
class TemporalEncoder(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        """
        x: [K, D] — K support notes, chronologically sorted
        returns: [K, D] — trajectory-aware embeddings
        """
        x = x.unsqueeze(0)  # [1, K, D]
        out, _ = self.gru(x)
        out = out.squeeze(0)  # [K, D]
        out = self.norm(out)
        out = self.fc(out)
        return out


# =============================================================================
# TEMPORAL WEIGHTING MODULE
# Unchanged from v2
# =============================================================================
class TemporalWeightingModule(nn.Module):
    def __init__(self, lambda_decay=0.5):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, temporal_metadata, device):
        ages = torch.tensor(
            [m.get("note_age_days", 0.0) for m in temporal_metadata],
            dtype=torch.float32,
            device=device,
        )
        visits = torch.tensor(
            [m.get("total_visits", 1.0) for m in temporal_metadata],
            dtype=torch.float32,
            device=device,
        )
        recency_weight = torch.exp(-self.lambda_decay * ages / 365.0)
        visit_weight = torch.sigmoid(self.alpha * visits)
        return recency_weight * visit_weight


# =============================================================================
# CONFIDENCE WEIGHTING MODULE  (v3 — cosine-distance based)
#
# WHY CHANGED: entropy-based confidence requires meaningful class probabilities,
# which only exist AFTER the model has learned something. Early in training all
# probs ≈ 0.5, so entropy ≈ max for every example — all notes get identical
# confidence weight and the mechanism is useless.
#
# NEW APPROACH: compute a preliminary prototype from temporal-weighted means,
# then weight each support note by its cosine similarity to that prototype.
# Notes that are close to the class centroid in embedding space are more
# representative → higher weight. This works from episode 1.
# =============================================================================
class ConfidenceWeightingModule(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        # beta controls how sharply low-similarity notes are down-weighted
        self.beta = beta

    def forward(self, embeddings, base_weights):
        """
        embeddings:   [K, D] — support embeddings after temporal encoding
        base_weights: [K]    — temporal × dataset weights (already computed)

        returns: [K] — refined confidence weights (normalized)
        """
        # Preliminary prototype using base weights
        w = base_weights / (base_weights.sum() + 1e-10)
        prototype = (embeddings * w.unsqueeze(1)).sum(0)  # [D]

        # Cosine similarity of each note to prototype
        emb_norm = F.normalize(embeddings, dim=1)  # [K, D]
        proto_norm = F.normalize(prototype.unsqueeze(0), dim=1)  # [1, D]
        cos_sim = (emb_norm * proto_norm).sum(dim=1)  # [K], in [-1, 1]

        # Convert to weight: higher similarity → higher weight
        # Clamp to [0, 1] so negative similarity notes get near-zero weight
        cos_sim = cos_sim.clamp(min=0.0)
        confidence_weight = torch.exp(self.beta * cos_sim)  # [K]

        return confidence_weight


# =============================================================================
# TC-WPN MAIN MODEL  (v3)
# =============================================================================
class TCWPN(nn.Module):
    def __init__(
        self,
        projection_dim=256,
        freeze_bert=False,
        lambda_decay=0.5,
        beta=2.0,
        aux_weight=0.3,  # v3: increased from 0.1 → 0.3
        refinement_passes=1,
    ):
        super().__init__()

        # -------------------------------------------------------
        # EMBEDDER
        # -------------------------------------------------------
        self.embedder = ClinicalEmbedder(
            projection_dim=projection_dim,
            freeze_bert=freeze_bert,
        )

        # -------------------------------------------------------
        # TEMPORAL ENCODER  (support only)
        # -------------------------------------------------------
        self.temporal_encoder = TemporalEncoder(projection_dim)

        # Query projection — separate from temporal GRU
        self.query_proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.LayerNorm(projection_dim),
        )

        # -------------------------------------------------------
        # WEIGHTING MODULES
        # -------------------------------------------------------
        self.temporal_w = TemporalWeightingModule(lambda_decay=lambda_decay)
        self.confidence_w = ConfidenceWeightingModule(beta=beta)

        # -------------------------------------------------------
        # TEMPERATURE SCALAR  (v3 — KEY FIX)
        # Initialized to 10.0 so initial logits have meaningful spread.
        # Without this, logit range ≈ [-0.05, 0.02] → softmax ≈ uniform.
        # -------------------------------------------------------
        self.log_temperature = nn.Parameter(torch.tensor(2.3026))  # log(10) ≈ 2.3

        # -------------------------------------------------------
        # AUXILIARY CLASSIFICATION HEAD
        # Provides direct supervision on embedding quality
        # -------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, 2),
        )

        self.aux_weight = aux_weight
        self.refinement_passes = refinement_passes

    # ===========================================================================
    # EMBEDDING HELPERS
    # ===========================================================================
    def _embed_note_list(self, ids_list, mask_list):
        device = next(self.parameters()).device
        return self.embedder.embed_batch(ids_list, mask_list, device=device)

    # ===========================================================================
    # SUPPORT PROTOTYPE CONSTRUCTION
    # ===========================================================================
    def build_prototypes(self, support):
        """
        Returns:
            prototypes: dict {label: [D] prototype tensor}
            all_embeddings: dict {label: [K, D] embedding tensor}
            all_weights: dict {label: [K] normalized weight tensor}
        """
        device = next(self.parameters()).device
        classes = list(support.keys())
        prototypes = {}
        all_embeddings = {}
        all_weights = {}

        for label in classes:
            ids_list = support[label]["input_ids"]
            mask_list = support[label]["attention_mask"]
            temporal = support[label]["temporal"]

            dataset_weights = torch.tensor(
                support[label].get("weights", [1.0] * len(ids_list)),
                dtype=torch.float32,
                device=device,
            )

            # --- Chronological sort (ascending note_age_days = oldest first) ---
            sorted_pack = sorted(
                zip(ids_list, mask_list, temporal, dataset_weights.tolist()),
                key=lambda x: x[2].get("note_age_days", 0),
                reverse=False,
            )
            ids_list, mask_list, temporal, dw_list = zip(*sorted_pack)

            # Reconstruct dataset_weights as a proper [K] float tensor
            dataset_weights = torch.tensor(
                list(dw_list),
                dtype=torch.float32,
                device=device,
            )

            # --- Embed ---
            embeddings = self._embed_note_list(ids_list, mask_list)  # [K, D]

            # --- Temporal encoding (trajectory-aware) ---
            embeddings = self.temporal_encoder(embeddings)  # [K, D]

            # --- Temporal weights ---
            temporal_weights = self.temporal_w(temporal, device)  # [K]

            # --- Combined base weights (temporal × dataset quality) ---
            base_weights = temporal_weights * dataset_weights  # [K]

            # --- Confidence refinement (cosine-based, works from episode 1) ---
            for _ in range(max(1, self.refinement_passes)):
                conf_weights = self.confidence_w(embeddings, base_weights)  # [K]
                base_weights = base_weights * conf_weights

            # --- Normalize ---
            norm_weights = base_weights / (base_weights.sum() + 1e-10)  # [K]

            # --- Weighted prototype ---
            prototype = (embeddings * norm_weights.unsqueeze(1)).sum(0)  # [D]
            prototype = F.normalize(prototype, dim=-1)

            prototypes[label] = prototype
            all_embeddings[label] = embeddings
            all_weights[label] = norm_weights

        return prototypes, all_embeddings, all_weights

    # ===========================================================================
    # PROTOTYPE-DISTANCE CLASSIFICATION  (v3 — replaces RelationModule)
    #
    # Uses negative squared Euclidean distance between L2-normalized query
    # and L2-normalized prototype, scaled by learnable temperature.
    # This directly maximizes inter-class distance in embedding space and
    # produces logits with meaningful spread from the first episode.
    # ===========================================================================
    def _classify_queries(self, query_emb, prototypes, classes):
        """
        query_emb:  [Nq, D]
        prototypes: dict {label: [D]}
        returns:    [Nq, N_classes] logits
        """
        temperature = torch.exp(self.log_temperature)  # always > 0

        query_norm = F.normalize(query_emb, dim=-1)  # [Nq, D]

        logits = []
        for c in classes:
            proto = prototypes[c]  # [D], already normalized
            # Negative squared Euclidean distance (equivalent to 2 * cosine - 2)
            dist = ((query_norm - proto.unsqueeze(0)) ** 2).sum(-1)  # [Nq]
            logits.append(-dist * temperature)

        return torch.stack(logits, dim=1)  # [Nq, N_classes]

    # ===========================================================================
    # FORWARD
    # ===========================================================================
    def forward(self, collated_episode):
        support = collated_episode["support"]
        query = collated_episode["query"]
        classes = collated_episode["classes"]

        # --- Build prototypes ---
        prototypes, all_embeddings, all_weights = self.build_prototypes(support)

        # --- Embed queries ---
        query_embeddings_all = []
        query_targets_all = []

        for idx, label in enumerate(classes):
            if label not in query:
                continue

            q_emb = self._embed_note_list(
                query[label]["input_ids"],
                query[label]["attention_mask"],
            )
            q_emb = self.query_proj(q_emb)  # [Nq, D]

            query_embeddings_all.append(q_emb)
            query_targets_all.append(
                torch.full((q_emb.size(0),), idx, device=q_emb.device, dtype=torch.long)
            )

        query_embeddings = torch.cat(query_embeddings_all, dim=0)  # [total_q, D]
        query_targets = torch.cat(query_targets_all, dim=0)  # [total_q]

        # --- Prototype-distance logits ---
        logits = self._classify_queries(query_embeddings, prototypes, classes)

        # --- Losses ---
        proto_loss = F.cross_entropy(logits, query_targets)
        aux_logits = self.classifier(query_embeddings)
        aux_loss = F.cross_entropy(aux_logits, query_targets)
        total_loss = proto_loss + self.aux_weight * aux_loss

        # --- Outputs ---
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        return {
            "loss": total_loss,
            "proto_loss": proto_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "targets": query_targets,
            "support_embeddings": all_embeddings,
            "support_weights": all_weights,
            "prototypes": prototypes,
            "temperature": torch.exp(self.log_temperature).item(),
        }

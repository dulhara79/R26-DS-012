# =============================================================================
# updated core.py
# core_v2.py
# Temporal-Confidence Weighted Prototypical Network (TC-WPN)
# Publication-Grade Version for Clinical Anxiety Detection
#
# KEY IMPROVEMENTS:
# - Proper support/query temporal consistency
# - Dataset confidence + model confidence integration
# - Source quality weighting
# - Multi-chunk clinical note embedding
# - Leakage-safe support prototype construction
# - Stable relation scoring
# - Auxiliary supervised head
# - Optional curriculum refinement
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from tc_wpn.models.embedder_v2 import ClinicalEmbedder


# =============================================================================
# TEMPORAL ENCODER
# =============================================================================
class TemporalEncoder(nn.Module):
    """
    Encodes longitudinal sequence patterns among support notes.
    Uses bidirectional GRU over patient trajectory.
    """

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

    def forward(self, x, is_query=False):
        """
        Support:
            x = [K, D]
        Query:
            x = [N, D]
        """

        if is_query:
            # Query note treated independently
            x = x.unsqueeze(1)  # [N,1,D]
            out, _ = self.gru(x)
            out = out.squeeze(1)

        else:
            # Support trajectory
            x = x.unsqueeze(0)  # [1,K,D]
            out, _ = self.gru(x)
            out = out.squeeze(0)

        out = self.norm(out)
        out = self.fc(out)

        return out


# =============================================================================
# RELATION MODULE
# =============================================================================
class RelationModule(nn.Module):
    """
    Learns richer distance than Euclidean/Cosine alone.
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()

        self.relation = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query, support):
        """
        query:   [Nq, D]
        support: [Ks, D]

        returns:
            [Nq, Ks]
        """

        query = F.layer_norm(query, query.shape[-1:])
        support = F.layer_norm(support, support.shape[-1:])

        Nq = query.size(0)
        Ks = support.size(0)

        query_exp = query.unsqueeze(1).expand(Nq, Ks, -1)
        support_exp = support.unsqueeze(0).expand(Nq, Ks, -1)

        abs_diff = torch.abs(query_exp - support_exp)
        mult = query_exp * support_exp

        combined = torch.cat(
            [query_exp, support_exp, abs_diff, mult],
            dim=-1,
        )

        scores = self.relation(combined).squeeze(-1)

        return scores


# =============================================================================
# TEMPORAL WEIGHTING
# =============================================================================
class TemporalWeightingModule(nn.Module):
    """
    Weights:
    - Recent notes higher
    - Higher visit regularity
    - Optional longitudinal richness
    """

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

        # More recent = higher
        recency_weight = torch.exp(-self.lambda_decay * ages / 365.0)

        # More longitudinal history = richer phenotype
        visit_weight = torch.sigmoid(self.alpha * visits)

        return recency_weight * visit_weight


# =============================================================================
# CONFIDENCE WEIGHTING
# =============================================================================
class ConfidenceWeightingModule(nn.Module):
    """
    Entropy-aware reliability weighting.
    """

    def __init__(self, beta=1.0, tau=2.0):
        super().__init__()

        self.beta = beta
        self.tau = tau

    def forward(self, logits):
        probs = F.softmax(logits / self.tau, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        weights = 1.0 / (1.0 + self.beta * entropy)

        return weights


# =============================================================================
# MAIN TC-WPN MODEL
# =============================================================================
class TCWPN(nn.Module):
    def __init__(
        self,
        projection_dim=256,
        freeze_bert=False,
        lambda_decay=0.5,
        beta=1.0,
        aux_weight=0.1,
        refinement_passes=1,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # EMBEDDER
        # ---------------------------------------------------------
        self.embedder = ClinicalEmbedder(
            projection_dim=projection_dim,
            freeze_bert=freeze_bert,
        )

        # ---------------------------------------------------------
        # TEMPORAL
        # ---------------------------------------------------------
        self.temporal_encoder = TemporalEncoder(projection_dim)

        # Query projection (avoids GRU distortion)
        self.query_proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.LayerNorm(projection_dim),
        )

        # ---------------------------------------------------------
        # WEIGHTING
        # ---------------------------------------------------------
        self.temporal_w = TemporalWeightingModule(lambda_decay=lambda_decay)

        self.confidence_w = ConfidenceWeightingModule(beta=beta)

        # ---------------------------------------------------------
        # RELATION
        # ---------------------------------------------------------
        self.relation_module = RelationModule(projection_dim)

        # ---------------------------------------------------------
        # AUXILIARY HEAD
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, 2),
        )

        self.aux_weight = aux_weight
        self.refinement_passes = refinement_passes

    # =========================================================================
    # EMBEDDING HELPERS
    # =========================================================================
    def _embed_note_list(self, ids_list, mask_list):
        device = next(self.parameters()).device

        return self.embedder.embed_batch(
            ids_list,
            mask_list,
            device=device,
        )

    # =========================================================================
    # SUPPORT BUILDING
    # =========================================================================
    def build_support_features(
        self,
        support,
    ):
        classes = list(support.keys())

        all_embeddings = {}
        all_base_weights = {}

        # ---------------------------------------------------------
        # STEP 1: Embed each class support set
        # ---------------------------------------------------------
        for label in classes:

            ids_list = support[label]["input_ids"]
            mask_list = support[label]["attention_mask"]
            temporal = support[label]["temporal"]

            dataset_weights = torch.tensor(
                support[label].get(
                    "weights",
                    [1.0] * len(ids_list),
                ),
                dtype=torch.float32,
                device=next(self.parameters()).device,
            )

            # Temporal sort
            sorted_pack = sorted(
                zip(
                    ids_list,
                    mask_list,
                    temporal,
                    dataset_weights,
                ),
                key=lambda x: x[2].get(
                    "note_age_days",
                    0,
                ),
                reverse=False,
            )

            ids_list, mask_list, temporal, dataset_weights = zip(*sorted_pack)

            embeddings = self._embed_note_list(
                ids_list,
                mask_list,
            )

            embeddings = self.temporal_encoder(
                embeddings,
                is_query=False,
            )

            temporal_weights = self.temporal_w(
                temporal,
                embeddings.device,
            )

            combined = temporal_weights * dataset_weights

            all_embeddings[label] = embeddings
            all_base_weights[label] = combined

        # ---------------------------------------------------------
        # STEP 2: Normalize
        # ---------------------------------------------------------
        all_weights = {c: w / (w.sum() + 1e-10) for c, w in all_base_weights.items()}

        # ---------------------------------------------------------
        # STEP 3: Confidence refinement
        # ---------------------------------------------------------
        for _ in range(min(self.refinement_passes, 2)):

            refined_weights = {}

            for label in classes:

                embeddings = all_embeddings[label]

                with torch.no_grad():
                    logits = self._classify_queries(
                        embeddings,
                        all_embeddings,
                        all_weights,
                        classes,
                    )

                model_conf = self.confidence_w(logits).to(embeddings.device)

                model_conf = model_conf.clamp(
                    0.5,
                    1.5,
                )

                refined = all_base_weights[label] * model_conf

                refined_weights[label] = refined / (refined.sum() + 1e-10)

            all_weights = refined_weights

        return all_embeddings, all_weights

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================
    def _classify_queries(
        self,
        query_embeddings,
        all_embeddings,
        all_weights,
        classes,
    ):
        logits = []

        for c in classes:
            support_emb = all_embeddings[c]
            weights = all_weights[c]

            relation_scores = self.relation_module(
                query_embeddings,
                support_emb,
            )

            weighted_scores = relation_scores * weights.unsqueeze(0)

            class_score = weighted_scores.sum(dim=1)

            logits.append(class_score)

        logits = torch.stack(
            logits,
            dim=1,
        )

        return logits

    # =========================================================================
    # FORWARD
    # =========================================================================
    def forward(
        self,
        collated_episode,
    ):
        support = collated_episode["support"]
        query = collated_episode["query"]
        classes = collated_episode["classes"]

        # ---------------------------------------------------------
        # SUPPORT
        # ---------------------------------------------------------
        all_embeddings, all_weights = self.build_support_features(support)

        # ---------------------------------------------------------
        # QUERY
        # ---------------------------------------------------------
        query_embeddings_all = []
        query_targets_all = []

        for idx, label in enumerate(classes):

            if label not in query:
                continue

            ids_list = query[label]["input_ids"]
            mask_list = query[label]["attention_mask"]

            q_emb = self._embed_note_list(
                ids_list,
                mask_list,
            )

            q_emb = self.query_proj(q_emb)

            query_embeddings_all.append(q_emb)

            query_targets_all.append(
                torch.full(
                    (q_emb.size(0),),
                    idx,
                    device=q_emb.device,
                    dtype=torch.long,
                )
            )

        query_embeddings = torch.cat(
            query_embeddings_all,
            dim=0,
        )

        query_targets = torch.cat(
            query_targets_all,
            dim=0,
        )

        # ---------------------------------------------------------
        # RELATION LOGITS
        # ---------------------------------------------------------
        rel_logits = self._classify_queries(
            query_embeddings,
            all_embeddings,
            all_weights,
            classes,
        )

        # ---------------------------------------------------------
        # LOSSES
        # ---------------------------------------------------------
        relation_loss = F.cross_entropy(
            rel_logits,
            query_targets,
        )

        aux_logits = self.classifier(query_embeddings)

        aux_loss = F.cross_entropy(
            aux_logits,
            query_targets,
        )

        total_loss = relation_loss + self.aux_weight * aux_loss

        # ---------------------------------------------------------
        # OUTPUTS
        # ---------------------------------------------------------
        probs = F.softmax(
            rel_logits,
            dim=-1,
        )

        preds = rel_logits.argmax(dim=-1)

        return {
            "loss": total_loss,
            "relation_loss": relation_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "logits": rel_logits,
            "probs": probs,
            "preds": preds,
            "targets": query_targets,
            "support_embeddings": all_embeddings,
            "support_weights": all_weights,
        }

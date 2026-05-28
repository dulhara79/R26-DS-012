# =============================================================================
# core_proposal.py
# Temporal-Confidence Weighted Prototypical Network — PROPOSAL ARCHITECTURE
#
# This file implements TC-WPN EXACTLY as described in the original proposal
# document (March 2026). It is the correct file to use with the proposal
# training notebook for a fair architecture comparison against core.py.
#
# DIFFERENCES FROM core.py:
# ┌─────────────────────────┬────────────────────────────┬───────────────────────────────┐
# │ Component               │ core.py (CURRENT)       │ core_proposal.py (PROPOSAL)   │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Confidence weighting    │ Cosine-distance to proto   │ Shannon entropy w=1/(1+β·H)   │
# │                         │ w = exp(β · cos_sim)       │ Proposal Section 3.3.3        │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Classification          │ Prototype-distance ×       │ RelationModule MLP            │
# │                         │ learnable temperature      │ (Sung et al. 2018)            │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Temperature scalar      │ Learnable nn.Parameter     │ None — not in proposal        │
# │                         │ init = log(10) ≈ 2.3026    │                               │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Temporal regularity     │ Learnable sigmoid          │ Threshold: 1.0 if visits≥3    │
# │                         │ sigmoid(α · visits)        │ else 0.8 + 0.1·visits         │
# │                         │ α is nn.Parameter          │ Proposal Section 3.3.2        │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Temporal encoder (GRU)  │ Present — trajectory-aware │ NOT in proposal — removed     │
# │                         │ BiGRU on support notes     │ Proposal has no GRU           │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Query projection        │ Separate query_proj MLP    │ None — queries use embedder   │
# │                         │ (avoids GRU distortion)    │ output directly               │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ Embedder pooling        │ Attention pooling          │ Simple mean pooling           │
# │                         │ (weighted by chunk norm)   │ Proposal Section 3.4.1        │
# ├─────────────────────────┼────────────────────────────┼───────────────────────────────┤
# │ aux_weight default      │ 0.3                        │ 0.1 (proposal Table 2)        │
# └─────────────────────────┴────────────────────────────┴───────────────────────────────┘
#
# WHY THIS FILE EXISTS:
# The previous proposal experiment (proposal training notebook) imported core.py
# from the GitHub repo, which means it used the CURRENT architecture's embedder
# (attention pooling, GRU temporal encoder, query_proj) instead of the proposal's
# simpler embedder (mean pooling, no GRU, direct query embedding). This contaminated
# the comparison. This file provides a clean, self-contained proposal implementation
# that does NOT depend on any components from core.py.
#
# USAGE:
#   Push this file to your GitHub repo at tc_wpn/models/core_proposal.py
#   Then in the proposal training notebook, change Cell 2 import to:
#       from tc_wpn.models.core_proposal import ProposalTCWPN
#   And remove the inline class definitions from Cell 5.
#
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


# =============================================================================
# PROPOSAL EMBEDDER
# Section 3.4.1: ClinicalBERT [CLS] → 768 → 256 linear projection
# Pooling: simple mean across chunks (NOT attention pooling from core.py)
# No GRU temporal encoder (not in proposal)
# =============================================================================
class ProposalClinicalEmbedder(nn.Module):
    """
    ClinicalBERT embedder as described in proposal Section 3.4.1.

    KEY DIFFERENCE FROM core.py ClinicalEmbedder:
    - core.py uses attention pooling: weighted sum by chunk norm
    - This file uses simple mean pooling across chunks (proposal spec)
    - core.py passes support through a BiGRU TemporalEncoder after embedding
    - This file does NOT — the proposal has no GRU on embeddings
    """

    def __init__(self, projection_dim=256):
        super().__init__()
        config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        config.use_cache = False
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", config=config
        )
        # Proposal Section 3.4.1: 768 → 256 projection
        self.projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(projection_dim),
        )

    def embed_note(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Embed one note (possibly multi-chunk).
        input_ids, attention_mask: [n_chunks, seq_len]
        Returns: [projection_dim] — single embedding for the note
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] token for each chunk: [n_chunks, 768]
        chunk_cls = outputs.last_hidden_state[:, 0, :]
        # PROPOSAL: simple mean pooling (NOT attention-weighted as in core.py)
        note_emb = chunk_cls.mean(dim=0)  # [768]
        return self.projection(note_emb)  # [projection_dim]

    def embed_batch(self, ids_list: list, mask_list: list, device=None) -> torch.Tensor:
        """
        Embed a list of notes.
        Returns: [N_notes, projection_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.stack(
            [
                self.embed_note(ids.to(device), mask.to(device))
                for ids, mask in zip(ids_list, mask_list)
            ],
            dim=0,
        )


# =============================================================================
# TEMPORAL WEIGHTING MODULE — PROPOSAL
# Section 3.3.2 — EXACT equations from proposal:
#
#   w_recency(t_i) = exp(−λ × (t_current − t_i) / 365)
#   w_regularity(p) = 1.0  if total_visits ≥ 3
#                   = 0.8 + 0.1 × total_visits  otherwise
#   w_temporal = w_recency × w_regularity
#
# KEY DIFFERENCE FROM core.py TemporalWeightingModule:
# - core.py: visit_weight = sigmoid(α × visits), α is a LEARNABLE parameter
# - This file: explicit threshold at 3 visits — NO learnable parameters
# - core.py: recency = exp(−λ × age / 365) relative to zero
# - This file: recency = exp(−λ × (t_current − t_i) / 365) relative to most recent
# =============================================================================
class ProposalTemporalWeighting(nn.Module):
    """
    Temporal weighting exactly as written in proposal Section 3.3.2.
    No learnable parameters — pure formula from the proposal text.
    """

    def __init__(self, lambda_decay: float = 0.5):
        super().__init__()
        self.lambda_decay = lambda_decay
        # No nn.Parameter here — proposal has no learnable temporal params

    def forward(self, temporal_metadata: list, device: torch.device) -> torch.Tensor:
        """
        temporal_metadata: list of dicts with 'note_age_days', 'total_visits'
        Returns: [K] weight tensor
        """
        ages = torch.tensor(
            [m.get("note_age_days", 0.0) for m in temporal_metadata],
            dtype=torch.float32,
            device=device,
        )
        visits = torch.tensor(
            [m.get("total_visits", 1) for m in temporal_metadata],
            dtype=torch.float32,
            device=device,
        )

        # t_current = most recent note in the episode (max age = oldest note
        # in calendar time → use max note_age_days as reference point)
        t_current = ages.max()

        # Proposal equation: exp(−λ × (t_current − t_i) / 365)
        # Most recent note gets weight 1.0; older notes decay exponentially
        recency_weight = torch.exp(-self.lambda_decay * (t_current - ages) / 365.0)

        # Proposal equation: threshold at 3 visits
        regularity_weight = torch.where(
            visits >= 3,
            torch.ones_like(visits),  # 1.0 for regular patients
            0.8 + 0.1 * visits,  # 0.9 for 1 visit, 1.0 for 2 visits, etc.
        )

        return recency_weight * regularity_weight  # [K]


# =============================================================================
# ENTROPY-BASED CONFIDENCE WEIGHTING MODULE — PROPOSAL
# Section 3.3.3 — EXACT equations from proposal:
#
#   H(x_i) = −∑_c P(y=c|x_i) × log P(y=c|x_i)
#   w_confidence(x_i) = 1 / (1 + β × H(x_i))
#
# KEY DIFFERENCE FROM core.py ConfidenceWeightingModule:
# - core.py: cosine similarity to preliminary prototype → exp(β × cos_sim)
#   Works from episode 1 because it only needs embedding geometry.
# - This file: Shannon entropy from class probabilities → 1/(1+β·H)
#   Requires a preliminary prototype to compute P(y=c|x_i) first.
#   Early in training: all probs ≈ 0.5 → H ≈ max → all weights ≈ 0.5
#   The mechanism becomes useful only after the model has learned something.
#   This is the limitation that motivated the change to cosine confidence.
# =============================================================================
class ProposalEntropyConfidence(nn.Module):
    """
    Entropy-based confidence weighting from proposal Section 3.3.3.
    No learnable parameters.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        embeddings: torch.Tensor,  # [K, D] support embeddings
        preliminary_prototypes: dict,  # {label: [D]} from temporal-only step
        classes: list,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Step 1: compute P(y=c|x_i) for each support note x_i
                using Euclidean distance to preliminary prototypes.
        Step 2: compute Shannon entropy H(x_i).
        Step 3: return w_confidence = 1 / (1 + β × H).
        Returns: [K] confidence weights
        """
        # Step 1: prototype-distance probabilities for each support note
        q_norm = F.normalize(embeddings, dim=-1)  # [K, D]
        logit_list = []
        for c in classes:
            proto = F.normalize(
                preliminary_prototypes[c].unsqueeze(0), dim=-1
            )  # [1, D]
            dist = ((q_norm - proto) ** 2).sum(-1)  # [K]
            logit_list.append(-dist)
        logits = torch.stack(logit_list, dim=1)  # [K, N_classes]
        probs = F.softmax(logits, dim=-1)  # [K, N_classes]

        # Step 2: Shannon entropy H(x_i) = −∑_c P_c × log(P_c)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [K]

        # Step 3: proposal confidence weight
        return 1.0 / (1.0 + self.beta * entropy)  # [K]


# =============================================================================
# RELATION MODULE — PROPOSAL
# Sung et al. (2018), cited in proposal Section 1.2.2.
# concat([query_emb, prototype]) → MLP → relation score per class
#
# KEY DIFFERENCE FROM core.py:
# - core.py REMOVED this and replaced with prototype-distance × temperature
# - Reason: RelationModule produces narrow logits early in training
#   (p_std ≈ 0.002–0.008 vs 0.084 with prototype-distance + temperature)
# =============================================================================
class ProposalRelationModule(nn.Module):
    """
    RelationModule as described in proposal Section 1.2.2.
    Input: concat([query_emb, prototype]) shape [Nq, 2D]
    Output: relation score per class shape [Nq, N_classes]
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_emb: torch.Tensor,  # [Nq, D]
        prototypes: dict,  # {label: [D]}
        classes: list,
    ) -> torch.Tensor:
        """Returns [Nq, N_classes] relation logits."""
        Nq = query_emb.size(0)
        scores = []
        for c in classes:
            proto = prototypes[c].unsqueeze(0).expand(Nq, -1)  # [Nq, D]
            combined = torch.cat([query_emb, proto], dim=-1)  # [Nq, 2D]
            score = self.net(combined).squeeze(-1)  # [Nq]
            scores.append(score)
        return torch.stack(scores, dim=1)  # [Nq, N_classes]


# =============================================================================
# PROPOSAL TC-WPN — MAIN MODEL
# Integrates all four proposal components above.
# Import this class into the proposal training notebook.
# =============================================================================
class ProposalTCWPN(nn.Module):
    """
    TC-WPN as described in the original proposal (March 2026).

    Architecture summary:
      Embedder:    ClinicalBERT [CLS] → mean-pool chunks → Linear(768→256)
      Temporal:    recency decay × regularity threshold (no learnable params)
      Confidence:  Shannon entropy  w = 1/(1+β·H)  (proposal Section 3.3.3)
      Prototype:   TC-weighted mean  (proposal Section 3.3.4)
      Classifier:  RelationModule MLP  (proposal Section 1.2.2)
      Temperature: None (not in proposal)
      aux_weight:  0.1 (proposal Table 2)

    Use this for a CLEAN comparison against core.py TCWPN.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        lambda_decay: float = 0.5,
        beta: float = 1.0,
        aux_weight: float = 0.1,
        relation_hidden: int = 256,
    ):
        super().__init__()

        # ── Embedder (mean-pool, no GRU) ──────────────────────────────────────
        self.embedder = ProposalClinicalEmbedder(projection_dim=projection_dim)

        # ── Temporal weighting (threshold regularity, no learnable α) ─────────
        self.temporal_w = ProposalTemporalWeighting(lambda_decay=lambda_decay)

        # ── Entropy confidence (proposal Section 3.3.3) ───────────────────────
        self.confidence_w = ProposalEntropyConfidence(beta=beta)

        # ── RelationModule classifier (proposal Section 1.2.2) ────────────────
        self.relation = ProposalRelationModule(
            input_dim=projection_dim * 2,
            hidden_dim=relation_hidden,
        )

        # ── Auxiliary head (same purpose as core.py, weight from proposal) ─
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim // 2, 2),
        )

        self.aux_weight = aux_weight
        self.projection_dim = projection_dim

        # NO self.log_temperature — proposal has no learnable temperature

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, ids_list: list, mask_list: list) -> torch.Tensor:
        device = next(self.parameters()).device
        return self.embedder.embed_batch(ids_list, mask_list, device=device)

    # ── Prototype construction (proposal Sections 3.3.2–3.3.4) ───────────────

    def build_prototypes(self, support: dict) -> dict:
        """
        Two-pass prototype construction as described in proposal Section 3.3.4:

        Pass 1 — temporal-only prototype (needed to compute entropy):
            preliminary_proto_c = Σ w_temporal(x_i) · f(x_i) / Σ w_temporal(x_i)

        Pass 2 — TC-weighted prototype:
            entropy H(x_i) computed from distances to preliminary prototypes
            final_proto_c = Σ [w_temporal · w_confidence] · f(x_i)
                           / Σ [w_temporal · w_confidence]

        Returns: dict {label: [D] prototype tensor}
        """
        device = next(self.parameters()).device
        classes = list(support.keys())

        # ── Pass 1: preliminary prototypes (temporal weights only) ────────────
        preliminary_prototypes = {}
        for label in classes:
            ids_l = support[label]["input_ids"]
            mask_l = support[label]["attention_mask"]
            temp = support[label]["temporal"]
            dw = torch.tensor(
                support[label].get("weights", [1.0] * len(ids_l)),
                dtype=torch.float32,
                device=device,
            )

            emb = self._embed(ids_l, mask_l)  # [K, D]
            tw = self.temporal_w(temp, device)  # [K]
            bw = tw * dw  # [K]
            nw = bw / (bw.sum() + 1e-10)

            preliminary_prototypes[label] = (emb * nw.unsqueeze(1)).sum(0)  # [D]

        # ── Pass 2: TC-weighted prototypes (temporal × entropy confidence) ─────
        final_prototypes = {}
        for label in classes:
            ids_l = support[label]["input_ids"]
            mask_l = support[label]["attention_mask"]
            temp = support[label]["temporal"]
            dw = torch.tensor(
                support[label].get("weights", [1.0] * len(ids_l)),
                dtype=torch.float32,
                device=device,
            )

            emb = self._embed(ids_l, mask_l)  # [K, D]
            tw = self.temporal_w(temp, device)  # [K]
            bw = tw * dw  # [K] — base temporal weights

            # Entropy confidence (proposal Section 3.3.3)
            cw = self.confidence_w(emb, preliminary_prototypes, classes, device)  # [K]

            # Proposal Section 3.3.4: combined TC-weighted prototype
            combined = bw * cw
            nw = combined / (combined.sum() + 1e-10)
            final_prototypes[label] = (emb * nw.unsqueeze(1)).sum(0)  # [D]

        return final_prototypes

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, collated_episode: dict) -> dict:
        support = collated_episode["support"]
        query = collated_episode["query"]
        classes = collated_episode["classes"]

        # Build TC-weighted prototypes
        prototypes = self.build_prototypes(support)

        # Embed queries — NO query_proj (not in proposal)
        q_embs, q_tgts = [], []
        for idx, label in enumerate(classes):
            if label not in query:
                continue
            qe = self._embed(
                query[label]["input_ids"],
                query[label]["attention_mask"],
            )  # [Nq, D]
            q_embs.append(qe)
            q_tgts.append(
                torch.full(
                    (qe.size(0),),
                    idx,
                    device=qe.device,
                    dtype=torch.long,
                )
            )

        query_embeddings = torch.cat(q_embs, dim=0)  # [total_q, D]
        query_targets = torch.cat(q_tgts, dim=0)  # [total_q]

        # RelationModule classification (proposal Section 1.2.2)
        logits = self.relation(query_embeddings, prototypes, classes)

        # Losses
        proto_loss = F.cross_entropy(logits, query_targets)
        aux_logits = self.classifier(query_embeddings)
        aux_loss = F.cross_entropy(aux_logits, query_targets)
        total_loss = proto_loss + self.aux_weight * aux_loss

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
            "prototypes": prototypes,
            # NOTE: no "temperature" key — proposal has no learnable temperature
            # NOTE: no "support_embeddings" key — proposal has no GRU encoder
            # NOTE: no "support_weights" key — for compatibility, add if needed
        }

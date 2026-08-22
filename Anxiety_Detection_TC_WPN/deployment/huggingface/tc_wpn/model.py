"""
model.py — One prototypical network with explicit ablation switches.

Rather than maintaining separate ProtoNet / TC-WPN classes that drift apart,
this module implements a single model whose components are toggled by config.
Every ablation row in the paper is then the SAME code path with a different
flag, which removes a whole class of "the ablation differs for another reason"
reviewer objections.

    use_temporal_weight              -> w^T
    use_prototype_consistency_weight -> w^C   (formerly "confidence weight")
    learn_temperature                -> tau
    aux_head_weight = 0              -> auxiliary CE head off

Setting all of them off/zero gives Snell et al. Prototypical Networks with a
cosine-distance head. That is the fair internal baseline.

THREE CORRECTIONS TO THE ARCHIVED core.py
=========================================
1. The BiGRU TemporalEncoder is REMOVED. It consumed the support set as if it
   were one patient's chronological trajectory, but a K-shot support set is K
   notes from K DIFFERENT patients. A recurrent pass over that sequence models
   an ordering that does not exist, and its output depends on which patients
   were sampled into the episode. The temporal signal is now a per-note scalar,
   `days_before_patient_last_note`, defined WITHIN a patient, which is what the
   quantity actually means.

2. `log_temperature` is a plain nn.Parameter and MUST be handed to the
   optimizer. In the archived training notebook it was excluded from the
   optimizer param groups, which froze it at its init value and disabled the
   sharpening mechanism the ablation was supposed to test. `parameter_groups()`
   below builds the groups so this cannot silently regress.

3. The `weights` input (training_weight = label_confidence x section_quality)
   is GONE. It was computed by regex over the note text and was strongly
   correlated with the label, so multiplying prototypes by it injected the
   label into prototype construction.

NOTATION FOR THE PAPER
======================
For support note i of class c:

    w_i^T = exp(-lambda * dt_i / 365)                   temporal recency
    w_i^C = exp(beta * cos(z_i, p~_c))                  prototype consistency
    w_i   = w_i^T * w_i^C  /  sum_j (w_j^T * w_j^C)
    p_c   = normalize( sum_i w_i * z_i )

where p~_c is the preliminary prototype built from w^T alone. Call w^C
"prototype consistency", not "confidence": it measures agreement with the
class centroid, not calibrated predictive confidence. The blinded evaluation
result (w^C increasing reliance on lexical shortcuts) is much easier to
discuss honestly under the accurate name.

Author: Dulhara Kaushalya (IT22130648)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ENCODER = "emilyalsentzer/Bio_ClinicalBERT"


# =============================================================================
# EMBEDDER
# =============================================================================
class ClinicalEmbedder(nn.Module):
    """
    Bio_ClinicalBERT -> [CLS] per chunk -> mean over chunks of a note ->
    projection to `projection_dim`.

    The archived embedder pooled chunks with softmax over the [CLS] L2 norms.
    That is not attention over anything meaningful (the "query" is the norm
    itself), it is unstable when a note has one chunk (softmax over a single
    element is always 1.0, so the weights did nothing), and it is hard to
    justify in a paper. Mean pooling over chunks is simple and defensible;
    with max_chunks_per_note=1 the two are identical anyway.

    All notes in a call are packed into ONE BERT forward pass instead of a
    Python loop over notes, which is the main speed difference on a T4.
    """

    def __init__(self, projection_dim=256, freeze_bert=False,
                 encoder_name=DEFAULT_ENCODER, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(encoder_name)
        config.use_cache = False
        self.bert = AutoModel.from_pretrained(encoder_name, config=config)
        self.hidden_size = self.bert.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_dim),
        )
        self.frozen = freeze_bert
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, note_index):
        """
        input_ids/attention_mask : [n_chunks_total, L]
        note_index               : [n_chunks_total] mapping each chunk -> note
        returns                  : [n_notes, projection_dim]
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]                       # [C, H]

        n_notes = int(note_index.max().item()) + 1
        pooled = torch.zeros(n_notes, cls.size(1), device=cls.device, dtype=cls.dtype)
        counts = torch.zeros(n_notes, 1, device=cls.device, dtype=cls.dtype)
        pooled.index_add_(0, note_index, cls)
        counts.index_add_(0, note_index, torch.ones_like(counts[note_index]))
        pooled = pooled / counts.clamp(min=1.0)

        return self.projection(pooled)


# =============================================================================
# WEIGHTING COMPONENTS
# =============================================================================
class TemporalRecencyWeight(nn.Module):
    """
    w_i^T = exp(-lambda * dt_i / 365), lambda > 0 learnable via log-parameter.

    dt_i = days_before_patient_last_note: how far note i sits before the LAST
    note observed for that same patient. It is defined entirely within a
    patient, so it does not depend on which other patients landed in the
    episode. Notes are the patient's most recent -> dt = 0 -> weight 1.
    """

    def __init__(self, init_lambda=0.5, learnable=True):
        super().__init__()
        val = torch.tensor(float(math.log(max(init_lambda, 1e-6))))
        self.log_lambda = nn.Parameter(val, requires_grad=learnable)

    def forward(self, days_before_last):
        lam = torch.exp(self.log_lambda)
        return torch.exp(-lam * days_before_last.clamp(min=0.0) / 365.0)


class PrototypeConsistencyWeight(nn.Module):
    """
    w_i^C = exp(beta * max(cos(z_i, p~_c), 0)), where p~_c is the preliminary
    prototype from the temporal weights alone.

    Detached from the graph on the prototype side would kill the gradient into
    beta, so p~ keeps its graph; the clamp at 0 is what stops a note that
    points away from the centroid from receiving negative weight.
    """

    def __init__(self, beta=2.0, learnable=True):
        super().__init__()
        self.log_beta = nn.Parameter(
            torch.tensor(float(math.log(max(beta, 1e-6)))), requires_grad=learnable
        )

    def forward(self, embeddings, base_weights):
        w = base_weights / (base_weights.sum() + 1e-10)
        prelim = (embeddings * w.unsqueeze(1)).sum(0)
        cos = F.cosine_similarity(embeddings, prelim.unsqueeze(0), dim=1).clamp(min=0.0)
        return torch.exp(torch.exp(self.log_beta) * cos)


# =============================================================================
# MAIN MODEL
# =============================================================================
class PrototypicalModel(nn.Module):
    def __init__(
        self,
        projection_dim: int = 256,
        freeze_bert: bool = False,
        encoder_name: str = DEFAULT_ENCODER,
        use_temporal_weight: bool = True,
        use_prototype_consistency_weight: bool = True,
        learn_temperature: bool = True,
        init_temperature: float = 10.0,
        init_lambda: float = 0.5,
        init_beta: float = 2.0,
        aux_head_weight: float = 0.0,
        consistency_passes: int = 1,
    ):
        super().__init__()
        self.embedder = ClinicalEmbedder(
            projection_dim=projection_dim,
            freeze_bert=freeze_bert,
            encoder_name=encoder_name,
        )
        self.use_temporal_weight = use_temporal_weight
        self.use_pcw = use_prototype_consistency_weight
        self.consistency_passes = max(1, int(consistency_passes))

        self.temporal_w = TemporalRecencyWeight(init_lambda) if use_temporal_weight else None
        self.pcw = PrototypeConsistencyWeight(init_beta) if use_prototype_consistency_weight else None

        self.log_temperature = nn.Parameter(
            torch.tensor(float(math.log(init_temperature))),
            requires_grad=learn_temperature,
        )

        self.aux_head_weight = float(aux_head_weight)
        self.aux_head = (
            nn.Sequential(
                nn.Linear(projection_dim, projection_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(projection_dim // 2, 2),
            )
            if aux_head_weight > 0
            else None
        )

    # -------------------------------------------------------------------------
    def parameter_groups(self, encoder_lr=2e-5, head_lr=1e-3, weight_decay=0.01):
        """
        Build optimizer param groups. EVERY trainable parameter appears in
        exactly one group; the assertion at the end is what prevents the
        archived bug where log_temperature was silently left out and frozen.
        """
        enc, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if name.startswith("embedder.bert.") else head).append((name, p))

        groups = [
            {"params": [p for _, p in enc], "lr": encoder_lr,
             "weight_decay": weight_decay, "name": "encoder"},
            {"params": [p for _, p in head], "lr": head_lr,
             "weight_decay": 0.0, "name": "head"},
        ]
        n_in_groups = sum(len(g["params"]) for g in groups)
        n_trainable = sum(1 for p in self.parameters() if p.requires_grad)
        assert n_in_groups == n_trainable, (
            f"{n_trainable - n_in_groups} trainable parameters are missing from "
            f"the optimizer groups (this is the log_temperature class of bug)"
        )
        return groups

    # -------------------------------------------------------------------------
    def build_prototype(self, embeddings, days_before_last):
        """
        embeddings       : [K, D]
        days_before_last : [K]
        returns (prototype [D], normalised weights [K])
        """
        K = embeddings.size(0)
        if self.use_temporal_weight:
            w = self.temporal_w(days_before_last)
        else:
            w = torch.ones(K, device=embeddings.device, dtype=embeddings.dtype)

        if self.use_pcw:
            for _ in range(self.consistency_passes):
                w = w * self.pcw(embeddings, w)

        w = w / (w.sum() + 1e-10)
        proto = (embeddings * w.unsqueeze(1)).sum(0)
        return F.normalize(proto, dim=-1), w

    def classify(self, query_emb, prototypes):
        """
        Cosine-distance head. prototypes: list of [D] in class order.
        Returns logits [Nq, n_classes].
        """
        tau = torch.exp(self.log_temperature)
        q = F.normalize(query_emb, dim=-1)
        P = torch.stack(prototypes, dim=0)                     # [C, D]
        # squared euclidean on the unit sphere == 2 - 2*cos
        dist = torch.cdist(q.unsqueeze(0), P.unsqueeze(0)).squeeze(0) ** 2
        return -dist * tau

    # -------------------------------------------------------------------------
    def forward(self, batch):
        """
        `batch` is produced by collate.collate_episode() and contains, for each
        of support and query, one packed tensor set plus per-note metadata:

            batch["support"]["input_ids"]      [C_chunks, L]
            batch["support"]["attention_mask"] [C_chunks, L]
            batch["support"]["note_index"]     [C_chunks]
            batch["support"]["labels"]         [N_support]   class index
            batch["support"]["days"]           [N_support]
            batch["query"][...] likewise, plus ["labels"] as the target
        """
        sup, qry = batch["support"], batch["query"]

        sup_emb = self.embedder(sup["input_ids"], sup["attention_mask"], sup["note_index"])
        qry_emb = self.embedder(qry["input_ids"], qry["attention_mask"], qry["note_index"])

        classes = sorted(set(sup["labels"].tolist()))
        prototypes, weights = [], {}
        for c in classes:
            mask = sup["labels"] == c
            proto, w = self.build_prototype(sup_emb[mask], sup["days"][mask])
            prototypes.append(proto)
            weights[c] = w.detach()

        logits = self.classify(qry_emb, prototypes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        targets = torch.tensor(
            [class_to_idx[int(v)] for v in qry["labels"].tolist()],
            device=logits.device, dtype=torch.long,
        )

        # The PROTOTYPE loss, kept separately from the total. Without this
        # split the training log cannot answer the question Phase 1 raised:
        # does the episodic objective ever descend, or does the auxiliary term
        # carry the whole decrease? ln(2) = 0.6931 is the chance level for a
        # balanced 2-way episode, so a proto_loss pinned at ~0.693 means the
        # prototypical objective learned nothing regardless of the total.
        proto_loss = F.cross_entropy(logits, targets)
        loss = proto_loss
        aux_loss = torch.zeros((), device=logits.device)
        if self.aux_head is not None:
            aux_loss = F.cross_entropy(self.aux_head(qry_emb), targets)
            loss = loss + self.aux_head_weight * aux_loss

        probs = F.softmax(logits, dim=-1)
        positive_col = class_to_idx.get(1, logits.size(1) - 1)

        return {
            "loss": loss,
            "proto_loss": proto_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "logits": logits,
            "probs": probs,
            "p_anxiety": probs[:, positive_col],
            "targets": targets,
            "classes": classes,
            "support_weights": weights,
            "temperature": torch.exp(self.log_temperature).detach(),
            # The two mechanism parameters, exposed so training can log what the
            # model actually learned rather than only that it learned something.
            # None when the corresponding module is switched off, which keeps
            # the ablation logs unambiguous.
            "lambda_decay": (torch.exp(self.temporal_w.log_lambda).detach()
                             if self.temporal_w is not None else None),
            "beta_consistency": (torch.exp(self.pcw.log_beta).detach()
                                 if self.pcw is not None else None),
        }


# =============================================================================
# CONFIG -> MODEL
# =============================================================================
ABLATION_PRESETS = {
    # name                       temporal  pcw    temp   aux
    "protonet":                  (False,   False, False, 0.0),
    "protonet_temp":             (False,   False, True,  0.0),
    "temporal_only":             (True,    False, True,  0.0),
    "pcw_only":                  (False,   True,  True,  0.0),
    "temporal_pcw":              (True,    True,  True,  0.0),
    "tcwpn_full":                (True,    True,  True,  0.3),
    # ------------------------------------------------------------------
    # Added after the Stage C ablation. Every preset with aux=0.0 collapsed
    # to chance (AUROC 0.498-0.522, predicted probabilities constant at
    # 0.500); the only preset that learned was the only one with aux>0.
    # `aux_only` isolates the auxiliary head from w^T and w^C so the two
    # explanations can be told apart:
    #     aux_only ~= tcwpn_full  -> the mechanisms contribute nothing and
    #                                the auxiliary loss is what trains the
    #                                encoder
    #     aux_only <  tcwpn_full  -> the mechanisms need a trained encoder
    #                                before they can help, which is a real
    #                                finding but a different claim
    # ------------------------------------------------------------------
    "aux_only":                  (False,   False, True,  0.3),
    # ------------------------------------------------------------------
    # Auxiliary-controlled ladder. The Phase 1 result showed that every
    # configuration WITHOUT the auxiliary head collapses (proto_cos >= 0.9997,
    # p_sd <= 0.0018) and every configuration WITH it does not. Comparing
    # tcwpn_full against a collapsed baseline therefore measures the auxiliary
    # head, not w^T or w^C.
    #
    # These two hold the auxiliary head CONSTANT and vary one mechanism each,
    # which is the only way to attribute a delta to that mechanism:
    #
    #   aux_only      = ProtoNet + tau + aux                 (already run: 0.7400)
    #   temporal_aux  = ProtoNet + tau + aux + w^T           <- delta_temporal
    #   pcw_aux       = ProtoNet + tau + aux + w^C           <- delta_PCW
    #   tcwpn_full    = ProtoNet + tau + aux + w^T + w^C     (already run: 0.7335)
    #
    # NOTE: `aux_only` IS the supervisor's `protonet_temp_aux`. Same tuple,
    # same numbers. It does not need re-running; relabel it in the table.
    # ------------------------------------------------------------------
    "temporal_aux":              (True,    False, True,  0.3),
    "pcw_aux":                   (False,   True,  True,  0.3),
    # ------------------------------------------------------------------
    # Auxiliary-weight sweep. Phase 3B established that essentially all of
    # the jump from ~0.50 to ~0.74 comes from the auxiliary CE head, not
    # from w^T or w^C. These isolate the dose-response of that head with
    # both mechanisms OFF, so nothing else varies.
    #
    # SELECT ON VALIDATION ONLY. The test set is locked.
    # ------------------------------------------------------------------
    "aux_w0.1":                  (False,   False, True,  0.1),
    "aux_w0.25":                 (False,   False, True,  0.25),
    "aux_w0.5":                  (False,   False, True,  0.5),
    "aux_w1.0":                  (False,   False, True,  1.0),
    "aux_w2.0":                  (False,   False, True,  2.0),
}


def build_model(cfg: dict) -> PrototypicalModel:
    """
    cfg keys: preset (optional) plus any explicit overrides.
    A preset sets the four ablation switches; explicit keys win over the preset.
    """
    kwargs = dict(
        projection_dim=cfg.get("projection_dim", 256),
        freeze_bert=cfg.get("freeze_bert", False),
        encoder_name=cfg.get("encoder_name", DEFAULT_ENCODER),
        init_temperature=cfg.get("init_temperature", 10.0),
        init_lambda=cfg.get("init_lambda", 0.5),
        init_beta=cfg.get("init_beta", 2.0),
        consistency_passes=cfg.get("consistency_passes", 1),
    )
    preset = cfg.get("preset")
    if preset:
        if preset not in ABLATION_PRESETS:
            raise ValueError(
                f"unknown preset {preset!r}; choose from {sorted(ABLATION_PRESETS)}"
            )
        t, p, temp, aux = ABLATION_PRESETS[preset]
        kwargs.update(
            use_temporal_weight=t,
            use_prototype_consistency_weight=p,
            learn_temperature=temp,
            aux_head_weight=aux,
        )
    for k in ("use_temporal_weight", "use_prototype_consistency_weight",
              "learn_temperature", "aux_head_weight"):
        if k in cfg:
            kwargs[k] = cfg[k]
    return PrototypicalModel(**kwargs)

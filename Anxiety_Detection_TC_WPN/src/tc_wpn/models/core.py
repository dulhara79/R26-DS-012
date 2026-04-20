# =============================================================================
# tc_wpn/models/core.py (Research-Grade Few-Shot Edition)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from tc_wpn.models.embedder import ClinicalEmbedder


class TemporalEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x, is_query=False):
        if is_query:
            # 🟢 Treats independent queries as sequences of length 1
            x = x.unsqueeze(1)
            out, _ = self.gru(x)
            out = self.fc(out)
            return out.squeeze(1)
        else:
            # 🟢 Treats Support Set as a single timeline of length K
            x = x.unsqueeze(0)
            out, _ = self.gru(x)
            out = self.fc(out)
            return out.squeeze(0)


# class RelationModule(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         # 🟢 High-Capacity Relation Network
#         self.relation = nn.Sequential(
#             nn.Linear(input_dim * 4, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )


class RelationModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Linear(input_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, query, support):
        # 🟢 STEP 3: BOOST SIGNAL (Add normalization here)
        query = F.normalize(query, p=2, dim=-1)
        support = F.normalize(support, p=2, dim=-1)

        Nq = query.size(0)
        K = support.size(0)

        query_exp = query.unsqueeze(1).expand(Nq, K, -1)
        support_exp = support.unsqueeze(0).expand(Nq, K, -1)

        diff = torch.abs(query_exp - support_exp)
        prod = query_exp * support_exp

        combined = torch.cat([query_exp, support_exp, diff, prod], dim=-1)
        return self.relation(combined).squeeze(-1)

    def forward(self, query, support):
        Nq = query.size(0)
        K = support.size(0)

        query_exp = query.unsqueeze(1).expand(Nq, K, -1)
        support_exp = support.unsqueeze(0).expand(Nq, K, -1)

        # 🟢 Explicit distance and interaction signals
        diff = torch.abs(query_exp - support_exp)
        prod = query_exp * support_exp

        combined = torch.cat([query_exp, support_exp, diff, prod], dim=-1)

        return self.relation(combined).squeeze(-1)


class TemporalWeightingModule(nn.Module):
    def __init__(self, lambda_decay=0.5):
        super().__init__()
        self.lambda_decay = lambda_decay

    def forward(self, temporal_metadata, device):
        ages = torch.tensor(
            [m["note_age_days"] for m in temporal_metadata],
            dtype=torch.float32,
            device=device,
        )
        visits = torch.tensor(
            [m["total_visits"] for m in temporal_metadata],
            dtype=torch.float32,
            device=device,
        )

        recency = torch.exp(-self.lambda_decay * ages / 365.0)
        regularity = torch.where(visits >= 3, 1.0, 0.8 + 0.1 * visits)

        return recency * regularity


class ConfidenceWeightingModule(nn.Module):
    def __init__(self, beta=1.0, tau=2.0):
        super().__init__()
        self.beta = beta
        self.tau = tau

    def forward(self, logits):
        probs = F.softmax(logits / self.tau, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return 1.0 / (1.0 + self.beta * entropy)


class TCWPN(nn.Module):
    def __init__(
        self,
        lambda_decay=0.5,
        beta=1.0,
        projection_dim=256,
        freeze_bert=False,
        aux_weight=0.0, 
    ):
        super().__init__()

        self.embedder = ClinicalEmbedder(
            projection_dim=projection_dim, freeze_bert=freeze_bert
        )
        self.temporal_encoder = TemporalEncoder(projection_dim)
        self.temporal_w = TemporalWeightingModule(lambda_decay)
        self.confidence_w = ConfidenceWeightingModule(beta)
        self.relation_module = RelationModule(projection_dim)
        self.classifier = nn.Linear(projection_dim, 2)
        self.aux_weight = aux_weight

    def _embed_note_list(self, ids_list, mask_list):
        device = next(self.parameters()).device
        return self.embedder.embed_batch(ids_list, mask_list, device=device)

    # 🟢 FULL MODE: GRU and Weighting are BACK ON
    def build_support_features(self, support, n_refinement_passes=1):
        classes = list(support.keys())
        all_embeddings = {}
        all_temporal_w = {}

        for label in classes:
            ids_list = support[label]["input_ids"]
            mask_list = support[label]["attention_mask"]
            temporal = support[label]["temporal"]

            sorted_data = sorted(
                zip(ids_list, mask_list, temporal),
                key=lambda x: x[2]["note_age_days"],
                reverse=True,
            )
            ids_list, mask_list, temporal = zip(*sorted_data)

            embeddings = self._embed_note_list(ids_list, mask_list)
            # 🟢 GRU applied to support
            # embeddings = self.temporal_encoder(embeddings)

            w_temp = self.temporal_w(temporal, embeddings.device)
            all_temporal_w[label] = w_temp
            all_embeddings[label] = embeddings

        all_final_weights = {
            label: w / (w.sum() + 1e-10) for label, w in all_temporal_w.items()
        }

        for _ in range(min(n_refinement_passes, 2)):
            new_weights = {}
            for label in classes:
                embeddings = all_embeddings[label]
                logits = self._classify_queries(
                    embeddings.detach(), all_embeddings, all_final_weights, classes
                ).detach()
                w_conf = self.confidence_w(logits).to(embeddings.device)
                w_combined = all_temporal_w[label] * (0.5 + 0.5 * w_conf)
                new_weights[label] = w_combined / (w_combined.sum() + 1e-10)
            all_final_weights = new_weights

        return all_embeddings, all_final_weights

    def _classify_queries(self, query_embeddings, all_embeddings, all_weights, classes):
        logits = []

        for c in classes:
            sup_embs = all_embeddings[c]
            weights = all_weights[c]
            rel_scores = self.relation_module(query_embeddings, sup_embs)
            weighted = rel_scores * weights.unsqueeze(0)
            logits.append(weighted.sum(dim=1))

        return torch.stack(logits, dim=1)

    def forward(self, collated_episode):
        support = collated_episode["support"]
        query = collated_episode["query"]
        classes = collated_episode["classes"]

        all_embeddings, all_weights = self.build_support_features(support)

        all_q_emb = []
        all_q_targets = []

        for idx, label in enumerate(classes):
            if label not in query:
                continue

            ids_list = query[label]["input_ids"]
            mask_list = query[label]["attention_mask"]

            q_emb = self._embed_note_list(ids_list, mask_list)
            # 🟢 GRU applied to queries (with anti-scrambling flag)
            # q_emb = self.temporal_encoder(q_emb, is_query=True)

            all_q_emb.append(q_emb)
            all_q_targets.append(
                torch.full((q_emb.shape[0],), idx, device=q_emb.device)
            )

        query_embeddings = torch.cat(all_q_emb, dim=0)
        query_targets = torch.cat(all_q_targets, dim=0)

        rel_logits = self._classify_queries(
            query_embeddings, all_embeddings, all_weights, classes
        )
        loss_relation = F.cross_entropy(rel_logits, query_targets)

        cls_logits = self.classifier(query_embeddings)
        loss_cls = F.cross_entropy(cls_logits, query_targets)

        loss = loss_relation + (self.aux_weight * loss_cls)

        probs = F.softmax(rel_logits, dim=-1)
        preds = rel_logits.argmax(dim=-1)

        return {
            "loss": loss,
            "logits": rel_logits,
            "probs": probs,
            "preds": preds,
            "targets": query_targets,
            "all_embeddings": all_embeddings,
        }

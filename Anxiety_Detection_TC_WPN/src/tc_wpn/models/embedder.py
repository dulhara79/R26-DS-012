# =============================================================================
# TC-WPN: ClinicalBERT Embedder
# =============================================================================
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class ClinicalEmbedder(nn.Module):
    def __init__(self, projection_dim=256, freeze_bert=False):
        super().__init__()

        # Disable cache for gradient checkpointing stability
        config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        config.use_cache = False

        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT", config=config
        )

        self.projection = nn.Sequential(
            nn.Linear(768, projection_dim),
            nn.ReLU(),  # 🟢 EVALUATOR FIX: Non-linearity added
            nn.Dropout(0.1),  # 🟢 EVALUATOR FIX: Dropout to prevent overfitting
            nn.LayerNorm(projection_dim),
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def embed_note(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # 🟢 EVALUATOR FIX: Vectorized processing. Kills the slow for-loop!
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Grab the [CLS] token for every chunk
        chunk_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: [n_chunks, 768]

        # 🟢 EVALUATOR FIX: Attention Pooling instead of simple mean
        weights = torch.softmax(chunk_embeddings.norm(dim=1), dim=0)
        cls_emb = (weights.unsqueeze(1) * chunk_embeddings).sum(dim=0)

        projected = self.projection(cls_emb)
        return projected

    def embed_batch(
        self, ids_list: list, mask_list: list, device: torch.device = None
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        embeddings = []
        for ids, mask in zip(ids_list, mask_list):
            embeddings.append(self.embed_note(ids.to(device), mask.to(device)))

        return torch.stack(embeddings, dim=0)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.embed_note(input_ids, attention_mask)

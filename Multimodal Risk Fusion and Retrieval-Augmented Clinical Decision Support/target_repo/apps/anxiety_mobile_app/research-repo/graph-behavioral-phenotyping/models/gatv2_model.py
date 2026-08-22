"""
models/gatv2_model.py
======================
GATv2-based anxiety vulnerability model with two output heads:
  - vuln_head  : scalar vulnerability logit (apply sigmoid for score)
  - hrw_head   : 24-dim hourly risk probability vector
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

from config import WINDOW_ORDER


class AnxietyGATv2(nn.Module):
    """
    Graph Attention Network v2 for anxiety vulnerability prediction.

    Parameters
    ----------
    node_feat : int  — number of input node features (default 9)
    hidden    : int  — hidden dimension per attention head
    heads     : int  — number of attention heads in conv1
    drop      : float — dropout probability
    """

    def __init__(
        self,
        node_feat: int = 9,
        hidden: int    = 64,
        heads: int     = 4,
        drop: float    = 0.3,
    ):
        super().__init__()
        self.drop = drop

        self.conv1 = GATv2Conv(
            node_feat, hidden, heads=heads, dropout=drop, edge_dim=2, concat=True
        )
        self.conv2 = GATv2Conv(
            hidden * heads, hidden, heads=1, dropout=drop, edge_dim=2, concat=False
        )

        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.vuln_head = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(32, 1),
        )
        self.hrw_head = nn.Sequential(
            nn.Linear(hidden * 2, 48),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(48, 24),
            nn.Sigmoid(),
        )

        self._last_attn  = None
        self._last_edges = None

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.bn1(F.elu(self.conv1(x, ei, ea)))
        x = F.dropout(x, p=self.drop, training=self.training)
        x, (ei2, attn) = self.conv2(x, ei, ea, return_attention_weights=True)
        x = self.bn2(F.elu(x))
        x = F.dropout(x, p=self.drop, training=self.training)

        self._last_attn  = attn.detach()
        self._last_edges = ei2.detach()

        xg = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        return self.vuln_head(xg), self.hrw_head(xg)

    def vulnerability_score(self, logit: torch.Tensor) -> float:
        """Sigmoid-transform a vulnerability logit to [0, 1]."""
        return float(torch.sigmoid(logit).item())

    def high_risk_window(self, hrw_pred: torch.Tensor | None = None) -> str:
        """
        Return the name of the highest-risk 2-hour time window.

        Uses hrw_head predictions when available; falls back to
        attention-weight heuristic otherwise.
        """
        if hrw_pred is not None:
            hrw_np = hrw_pred.detach().cpu().numpy()
            if hrw_np.ndim > 1:
                hrw_np = hrw_np[0]
            best_score, best_h = -1, 0
            for start in range(24):
                hrs   = [(start + i) % 24 for i in range(2)]
                score = float(np.mean(hrw_np[hrs]))
                if score > best_score:
                    best_score = score
                    best_h     = start
            if   6  <= best_h < 12: return "morning"
            elif 12 <= best_h < 18: return "afternoon"
            elif 18 <= best_h < 23: return "evening"
            else:                   return "night"

        # Fallback: attention weights
        if self._last_attn is None:
            return "unknown"
        attn   = self._last_attn.cpu().numpy().flatten()
        src    = self._last_edges[0].cpu().numpy()
        w_attn = np.zeros(4)
        w_cnt  = np.zeros(4)
        for node, a in zip(src, attn):
            h = int(node) % 4
            w_attn[h] += a
            w_cnt[h]  += 1
        w_avg = w_attn / (w_cnt + 1e-9)
        return WINDOW_ORDER[int(np.argmax(w_avg))]

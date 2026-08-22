"""Final binary GATv2 architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class AnxietyGATv2(nn.Module):
    def __init__(self, node_feat, hidden=32, heads=2, drop=0.50):
        super().__init__()
        self.drop = drop

        self.conv1 = GATv2Conv(
            node_feat,
            hidden,
            heads=heads,
            dropout=drop,
            concat=True,
        )
        self.bn1 = nn.BatchNorm1d(hidden * heads)

        self.conv2 = GATv2Conv(
            hidden * heads,
            hidden,
            heads=1,
            dropout=drop,
            concat=False,
        )
        self.bn2 = nn.BatchNorm1d(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 24),
            nn.ELU(),
            nn.Dropout(drop),
            nn.Linear(24, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(F.elu(x))
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(F.elu(x))

        pooled = torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )
        return self.head(pooled)

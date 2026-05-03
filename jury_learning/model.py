from __future__ import annotations

import torch
import torch.nn as nn


class MoralJuryDCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_response_features: int,
        num_group_features: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_cross_layers: int = 3,
        response_encoder_hidden: int = 64,
    ):
        super().__init__()

        h = response_encoder_hidden
        self.response_encoder = nn.Sequential(
            nn.Linear(num_response_features, h),
            nn.ReLU(),
            nn.Linear(h, h),
        )

        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.group_encoder = nn.Sequential(
            nn.Linear(num_group_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.input_dim = h + embed_dim + embed_dim

        self.cross_layers = nn.ModuleList([nn.Linear(self.input_dim, self.input_dim) for _ in range(num_cross_layers)])

        self.deep_layers = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.output_head = nn.Linear(self.input_dim + hidden_dim, 1)

    def forward(self, response_fts: torch.Tensor, user_ids: torch.Tensor, group_fts: torch.Tensor) -> torch.Tensor:
        x_feat = self.response_encoder(response_fts)
        x0 = torch.cat([x_feat, self.user_embed(user_ids), self.group_encoder(group_fts)], dim=-1)

        xl = x0
        for layer in self.cross_layers:
            xl = x0 * layer(xl) + xl

        xd = self.deep_layers(x0)
        combined = torch.cat([xl, xd], dim=-1)
        return torch.sigmoid(self.output_head(combined))

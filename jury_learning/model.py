from __future__ import annotations

import math

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
        return self.output_head(combined)


class MoralJuryDCNBaseline(nn.Module):
    """Same architecture as MoralJuryDCN but without the user-ID embedding.

    Serves as a baseline to measure how much the per-user embedding contributes.
    The forward signature is identical so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
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

        self.group_encoder = nn.Sequential(
            nn.Linear(num_group_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.input_dim = h + embed_dim

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
        x0 = torch.cat([x_feat, self.group_encoder(group_fts)], dim=-1)

        xl = x0
        for layer in self.cross_layers:
            xl = x0 * layer(xl) + xl

        xd = self.deep_layers(x0)
        combined = torch.cat([xl, xd], dim=-1)
        return self.output_head(combined)


class MoralJuryTransformer(nn.Module):
    """Scenario-only Transformer encoder replicating arXiv:2602.03351.

    Each of the N character columns for Stay and Swerve becomes one token:

        e_c = [E_char(char_type_id) ; E_card(count) ; E_team(outcome)]

    where d_char = d_model//2, d_card = d_model//4, d_team = d_model//4.

    A learnable [CLS] token is prepended; its final representation feeds a
    2-layer GELU MLP head that outputs a raw logit (no sigmoid — use
    BCEWithLogitsLoss during training).

    At evaluation time, ``forward_symmetric`` averages f(A,B) with 1-f(B,A)
    to enforce side-invariance, as described in the paper.

    The forward signature matches MoralJuryDCN (accepts user_ids, group_fts)
    so it is a drop-in replacement; those arguments are ignored.
    """

    def __init__(
        self,
        num_char_types: int,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_count: int = 20,
    ):
        super().__init__()

        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.num_char_types = num_char_types
        d_char = d_model // 2
        d_card = d_model // 4
        d_team = d_model // 4

        # Sub-embeddings that compose each token
        self.char_embed = nn.Embedding(num_char_types, d_char)
        self.card_embed = nn.Embedding(max_count + 1, d_card)
        self.team_embed = nn.Embedding(2, d_team)   # 0 = Stay, 1 = Swerve

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm; more stable than post-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: d_model → d_model//2 → 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._max_count = max_count
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _build_tokens(self, response_fts: torch.Tensor) -> torch.Tensor:
        """Convert flat response features [B, 2*T] into token sequence [B, 1+2*T, d_model]."""
        B = response_fts.shape[0]
        T = self.num_char_types

        stay_counts   = response_fts[:, :T]
        swerve_counts = response_fts[:, T:]

        char_ids = torch.arange(T, device=response_fts.device).unsqueeze(0).expand(B, -1)

        stay_card   = stay_counts.long().clamp(0, self._max_count)
        swerve_card = swerve_counts.long().clamp(0, self._max_count)

        stay_team   = torch.zeros(B, T, dtype=torch.long, device=response_fts.device)
        swerve_team = torch.ones( B, T, dtype=torch.long, device=response_fts.device)

        def _tok(card_ids: torch.Tensor, team_ids: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [self.char_embed(char_ids), self.card_embed(card_ids), self.team_embed(team_ids)],
                dim=-1,
            )

        tokens = torch.cat([_tok(stay_card, stay_team), _tok(swerve_card, swerve_team)], dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, tokens], dim=1)   # [B, 1+2T, d_model]

    def _logit(self, response_fts: torch.Tensor) -> torch.Tensor:
        tokens  = self._build_tokens(response_fts)
        cls_out = self.transformer(tokens)[:, 0, :]   # [B, d_model]
        return self.head(cls_out).squeeze(-1)          # [B]

    def forward(
        self,
        response_fts: torch.Tensor,
        user_ids: torch.Tensor,   # ignored — kept for API compatibility
        group_fts: torch.Tensor,  # ignored — kept for API compatibility
    ) -> torch.Tensor:
        return self._logit(response_fts).unsqueeze(-1)  # [B, 1]

    def forward_symmetric(
        self,
        response_fts: torch.Tensor,
        user_ids: torch.Tensor,
        group_fts: torch.Tensor,
    ) -> torch.Tensor:
        """Side-invariant prediction: p = ½[σ(f(A,B)) + 1 − σ(f(B,A))].

        Swaps Stay and Swerve halves of response_fts for the second pass.
        Use at eval/inference time for best accuracy.
        Returns a logit (threshold at 0.0, consistent with the rest of the codebase).
        """
        T = self.num_char_types
        flipped = torch.cat([response_fts[:, T:], response_fts[:, :T]], dim=1)

        prob = 0.5 * (torch.sigmoid(self._logit(response_fts))
                      + 1.0 - torch.sigmoid(self._logit(flipped)))
        prob = prob.clamp(1e-7, 1.0 - 1e-7)
        return torch.log(prob / (1.0 - prob)).unsqueeze(-1)  # logit [B, 1]

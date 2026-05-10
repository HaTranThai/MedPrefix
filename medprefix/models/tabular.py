"""Tabular encoders for clinical metadata (age, sex, localization).

Three variants:
    Baseline (per-field tokens), MLP (single fused token), FT-Transformer
    (3 contextualized tokens via a small Transformer encoder).

Age is discretized to integer bins clamped to [0, 100] (101 effective bins
+ 1 reserved for unknown).
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn


def _bin_age(age: torch.Tensor, device) -> torch.Tensor:
    if age.dtype.is_floating_point:
        a = torch.clamp(age.round().long(), 0, 100)
    else:
        a = torch.clamp(age.long(), 0, 100)
    return a.to(device)


class TabularTokenEncoder(nn.Module):
    """BASELINE: 3 independent field tokens (Age, Sex, Localization) + type embedding."""

    def __init__(self, sex_vocab: Dict[str, int], loc_vocab: Dict[str, int], d_model: int = 768):
        super().__init__()
        self.sex_emb = nn.Embedding(len(sex_vocab) + 1, d_model)
        self.loc_emb = nn.Embedding(len(loc_vocab) + 1, d_model)
        self.age_emb = nn.Embedding(101 + 1, d_model)
        self.type_emb = nn.Embedding(3, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, age, sex_id, loc_id):
        device = sex_id.device
        age_b = _bin_age(age, device)
        t_age = self.age_emb(age_b)
        t_sex = self.sex_emb(sex_id)
        t_loc = self.loc_emb(loc_id)
        tok = torch.stack([t_age, t_sex, t_loc], dim=1)  # (B, 3, d)
        type_ids = torch.arange(3, device=device).unsqueeze(0).expand(tok.size(0), -1)
        tok = tok + self.type_emb(type_ids)
        return self.ln(tok)


class TabularMLPEncoder(nn.Module):
    """MLP: small per-field embeddings concatenated, then projected to one token."""

    def __init__(self, sex_vocab: Dict[str, int], loc_vocab: Dict[str, int], d_model: int = 768, embed_dim: int = 64):
        super().__init__()
        self.sex_emb = nn.Embedding(len(sex_vocab) + 1, embed_dim)
        self.loc_emb = nn.Embedding(len(loc_vocab) + 1, embed_dim)
        self.age_emb = nn.Embedding(101 + 1, embed_dim)
        input_dim = embed_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, age, sex_id, loc_id):
        device = sex_id.device
        age_b = _bin_age(age, device)
        v_age = self.age_emb(age_b)
        v_sex = self.sex_emb(sex_id)
        v_loc = self.loc_emb(loc_id)
        concat_feat = torch.cat([v_age, v_sex, v_loc], dim=1)
        return self.mlp(concat_feat).unsqueeze(1)  # (B, 1, d)


class FTTransformerEncoder(nn.Module):
    """FT-Transformer: 3 field tokens with self-attention for feature interaction."""

    def __init__(self, sex_vocab: Dict[str, int], loc_vocab: Dict[str, int], d_model: int = 768, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.sex_emb = nn.Embedding(len(sex_vocab) + 1, d_model)
        self.loc_emb = nn.Embedding(len(loc_vocab) + 1, d_model)
        self.age_emb = nn.Embedding(101 + 1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, age, sex_id, loc_id):
        device = sex_id.device
        age_b = _bin_age(age, device)
        t_age = self.age_emb(age_b)
        t_sex = self.sex_emb(sex_id)
        t_loc = self.loc_emb(loc_id)
        x = torch.stack([t_age, t_sex, t_loc], dim=1)
        x = self.transformer(x)
        return self.ln(x)


def build_tabular_encoder(mode: str, sex_vocab, loc_vocab, d_model: int):
    mode = mode.upper()
    if mode == "BASELINE":
        return TabularTokenEncoder(sex_vocab, loc_vocab, d_model=d_model), 3
    if mode == "MLP":
        return TabularMLPEncoder(sex_vocab, loc_vocab, d_model=d_model), 1
    if mode == "FT_TRANSFORMER":
        return FTTransformerEncoder(sex_vocab, loc_vocab, d_model=d_model), 3
    if mode == "TEXT":
        return None, 0
    raise ValueError(f"Unknown tabular_mode: {mode}")

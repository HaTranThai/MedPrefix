"""Perceiver Resampler + dual-pathway projector.

The Perceiver Resampler compresses a variable-length token sequence into a
fixed-budget latent code via cross-attention from learnable query latents.
Two independent resamplers are used for the visual and tabular streams to
prevent the high-dimensional visual tokens from drowning out the compact
metadata signal.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_latents: int = 16,
        n_heads: int = 8,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "ln_q":  nn.LayerNorm(d_model),
                "ln_kv": nn.LayerNorm(d_model),
                "attn":  nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ln_ff": nn.LayerNorm(d_model),
                "ff":    nn.Sequential(
                    nn.Linear(d_model, int(d_model * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(d_model * mlp_ratio), d_model),
                ),
            }))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B = context.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q = blk["ln_q"](latents)
            kv = blk["ln_kv"](context)
            attn_out, _ = blk["attn"](q, kv, kv, need_weights=False)
            latents = latents + attn_out
            latents = latents + blk["ff"](blk["ln_ff"](latents))
        return latents


class TokenFusionProjector(nn.Module):
    """Dual-pathway projector. Returns ``[P_tab ; P_img]`` concatenated."""

    def __init__(
        self,
        img_dim: int,
        d_model: int,
        n_tab_prefix: int = 4,
        n_img_prefix: int = 12,
        perceiver_depth: int = 2,
        n_heads: int = 8,
    ):
        super().__init__()
        self.n_tab_prefix = int(n_tab_prefix)
        self.n_img_prefix = int(n_img_prefix)
        self.n_prefix = self.n_tab_prefix + self.n_img_prefix

        self.img_proj = nn.Linear(img_dim, d_model)
        self.resampler_tab = PerceiverResampler(
            d_model=d_model, n_latents=max(1, self.n_tab_prefix),
            n_heads=n_heads, depth=perceiver_depth,
        )
        self.resampler_img = PerceiverResampler(
            d_model=d_model, n_latents=max(1, self.n_img_prefix),
            n_heads=n_heads, depth=perceiver_depth,
        )

        # Learnable null prefix (used when image is dropped).
        self.null_img_prefix = nn.Parameter(torch.zeros(1, self.n_img_prefix, d_model))

        # Learnable per-modality scaling.
        self.tab_scale = nn.Parameter(torch.tensor(1.0))
        self.img_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        img_tokens: Optional[torch.Tensor],
        tab_tokens: Optional[torch.Tensor],
        type_img: Optional[torch.Tensor] = None,
        type_tab: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # We need a batch-size reference even if one stream is None.
        if tab_tokens is not None:
            B = tab_tokens.size(0)
        elif img_tokens is not None:
            B = img_tokens.size(0)
        else:
            raise ValueError("Both img_tokens and tab_tokens are None.")

        # --- Tabular ---
        if tab_tokens is not None and self.n_tab_prefix > 0:
            if type_tab is not None:
                tab_tokens = tab_tokens + type_tab
            tab_prefix = self.resampler_tab(tab_tokens) * self.tab_scale
        else:
            tab_prefix = None

        # --- Visual ---
        if img_tokens is not None:
            img = self.img_proj(img_tokens)
            if type_img is not None:
                img = img + type_img
            img_prefix = self.resampler_img(img)
        else:
            img_prefix = self.null_img_prefix.expand(B, -1, -1)
        img_prefix = img_prefix * self.img_scale

        if tab_prefix is None:
            return img_prefix
        return torch.cat([tab_prefix, img_prefix], dim=1)

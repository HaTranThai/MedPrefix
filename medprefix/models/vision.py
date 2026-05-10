"""Frozen Vision Transformer producing patch tokens (no [CLS])."""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


class VisionEncoderTokens(nn.Module):
    """timm-backed ViT that returns the spatial patch sequence.

    For ViT-Base/16 on 224x224 input the output is (B, 196, 768).
    """

    def __init__(self, name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        self.out_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:
            B, H, W, C = feats.shape
            feats = feats.view(B, -1, C)
        elif feats.dim() == 3:
            arch = self.backbone.default_cfg.get("architecture", "")
            is_swin = "swin" in arch.lower()
            # Drop [CLS] for plain ViTs (Swin doesn't have one).
            if not is_swin and feats.size(1) > 1:
                feats = feats[:, 1:, :]
        return feats

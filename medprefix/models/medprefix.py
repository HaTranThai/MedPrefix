"""Med-Prefix: tri-modal model conditioning a frozen LLM via prefix tokens + GXCA."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision import VisionEncoderTokens
from .tabular import build_tabular_encoder
from .perceiver import TokenFusionProjector
from .gxca import inject_gated_cross_attention


class MedPrefixModel(nn.Module):
    """Frozen-backbone tri-modal conditioning model.

    Args:
        llm_obj: pre-built CausalLM (Qwen, TinyLlama, …). Frozen by default.
        tokenizer_obj: matching tokenizer (already augmented with special tokens).
        vision_name: timm model name for the visual backbone.
        n_tab_prefix: # latent queries for the tabular Perceiver Resampler.
        n_img_prefix: # latent queries for the visual Perceiver Resampler.
        sex_vocab, loc_vocab: dicts {value -> id} (id 0 reserved for unknown).
        tabular_mode: ``BASELINE`` | ``MLP`` | ``FT_TRANSFORMER`` | ``TEXT``.
        use_gxca / gxca_layers / gxca_heads: GXCA injection settings.
    """

    def __init__(
        self,
        *,
        llm_obj,
        tokenizer_obj,
        vision_name: str = "vit_base_patch16_224",
        n_tab_prefix: int = 4,
        n_img_prefix: int = 12,
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        sex_vocab: Optional[Dict[str, int]] = None,
        loc_vocab: Optional[Dict[str, int]] = None,
        tabular_mode: str = "BASELINE",
        perceiver_depth: int = 2,
        perceiver_heads: int = 8,
        use_gxca: bool = True,
        gxca_layers: int = 4,
        gxca_heads: Optional[int] = None,
        aux_lambda: float = 0.0,
    ):
        super().__init__()
        if llm_obj is None or tokenizer_obj is None:
            raise ValueError("llm_obj and tokenizer_obj are required.")
        if sex_vocab is None or loc_vocab is None:
            raise ValueError("sex_vocab and loc_vocab are required.")

        self.tokenizer = tokenizer_obj
        self.llm = llm_obj
        self.tabular_mode = tabular_mode
        self.aux_lambda = float(aux_lambda)

        d_model = int(self.llm.get_input_embeddings().weight.shape[-1])
        self.d_model = d_model

        # --- Tabular branch ---
        self.tabtok, n_tab_tokens_out = build_tabular_encoder(
            tabular_mode, sex_vocab, loc_vocab, d_model
        )
        self.n_tab_tokens_out = n_tab_tokens_out
        self.n_tab_prefix = int(n_tab_prefix) if tabular_mode != "TEXT" else 0

        # --- Vision branch ---
        self.n_img_prefix = int(n_img_prefix)
        self.n_prefix = self.n_tab_prefix + self.n_img_prefix

        self.vision = VisionEncoderTokens(vision_name, pretrained=True)
        self.freeze_vision = bool(freeze_vision)
        if self.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.freeze_llm = bool(freeze_llm)
        if self.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            # Re-enable LoRA params if PEFT-wrapped.
            for name, p in self.llm.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True

        # Null tab token + type embeddings.
        self.null_tab = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_img = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_tab = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # --- Projector (Perceiver) ---
        self.projector = TokenFusionProjector(
            img_dim=self.vision.out_dim,
            d_model=d_model,
            n_tab_prefix=self.n_tab_prefix,
            n_img_prefix=self.n_img_prefix,
            perceiver_depth=perceiver_depth,
            n_heads=perceiver_heads,
        )

        # --- Auxiliary reconstruction heads (optional) ---
        # Force the tabular prefix to encode meaningful age/sex/loc info by
        # asking small heads to reconstruct them from the pooled prefix tokens.
        # Enabled when aux_lambda > 0 (used by ISIC 2019 setup).
        self.use_aux = (
            self.aux_lambda > 0.0
            and self.tabular_mode != "TEXT"
            and self.n_tab_prefix > 0
        )
        if self.use_aux:
            self.age_aux_head = nn.Linear(d_model, 1)
            self.sex_aux_head = nn.Linear(d_model, len(sex_vocab) + 1)
            self.loc_aux_head = nn.Linear(d_model, len(loc_vocab) + 1)
        else:
            self.age_aux_head = None
            self.sex_aux_head = None
            self.loc_aux_head = None

        # --- GXCA injection ---
        self.use_gxca = bool(use_gxca)
        self.gxca_layers = int(gxca_layers)
        self.gxca_heads = gxca_heads
        if self.use_gxca and self.gxca_layers > 0:
            llm_inject = self.llm
            if hasattr(llm_inject, "get_base_model"):
                try:
                    llm_inject = llm_inject.get_base_model()
                except Exception:
                    pass
            elif hasattr(llm_inject, "base_model"):
                llm_inject = llm_inject.base_model

            inject_gated_cross_attention(
                llm_inject,
                last_k_layers=self.gxca_layers,
                n_heads=self.gxca_heads,
                dropout=0.0,
            )
            for name, p in self.llm.named_parameters():
                if "mm_gxca" in name.lower():
                    p.requires_grad = True

    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        tab_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_image: bool = True,
        use_tab: bool = True,
    ):
        B = input_ids.size(0)
        vision_device = next(self.vision.parameters()).device
        vision_dtype = next(self.vision.parameters()).dtype

        # 1. Vision
        if use_image:
            images = images.to(device=vision_device, dtype=vision_dtype)
            if self.freeze_vision:
                with torch.no_grad():
                    img_tokens = self.vision(images)
            else:
                img_tokens = self.vision(images)
        else:
            img_tokens = None

        # 2. Tabular
        age, sex_id, loc_id = tab_batch
        age = age.to(vision_device)
        sex_id = sex_id.to(vision_device)
        loc_id = loc_id.to(vision_device)

        tab_tokens = None
        if self.tabular_mode != "TEXT":
            if use_tab and self.tabtok is not None:
                tab_tokens = self.tabtok(age, sex_id, loc_id)
            else:
                n_toks = self.n_tab_tokens_out or 1
                tab_tokens = self.null_tab.to(
                    device=vision_device, dtype=vision_dtype
                ).expand(B, n_toks, -1)

        # 3. Fusion
        prefix = self.projector(
            img_tokens,
            tab_tokens,
            type_img=self.type_img.to(device=vision_device, dtype=vision_dtype),
            type_tab=self.type_tab.to(device=vision_device, dtype=vision_dtype),
        )

        # 3b. Auxiliary reconstruction loss (only when training + tab present + use_aux)
        aux_loss = None
        if (
            self.training
            and self.use_aux
            and use_tab
            and tab_tokens is not None
            and self.n_tab_prefix > 0
        ):
            tab_prefix_pooled = prefix[:, : self.n_tab_prefix, :].mean(dim=1).float()
            age_pred = self.age_aux_head(tab_prefix_pooled).squeeze(-1)
            age_loss = F.mse_loss(age_pred, age.float() / 100.0)
            sex_logits = self.sex_aux_head(tab_prefix_pooled)
            sex_loss = F.cross_entropy(sex_logits, sex_id.long())
            loc_logits = self.loc_aux_head(tab_prefix_pooled)
            loc_loss = F.cross_entropy(loc_logits, loc_id.long())
            aux_loss = (age_loss + sex_loss + loc_loss) / 3.0

        # 4. Build LLM inputs
        emb = self.llm.get_input_embeddings()
        wdev = emb.weight.device
        wdtype = emb.weight.dtype
        act_dtype = wdtype if (wdtype.is_floating_point or wdtype.is_complex) else torch.float16

        prefix = prefix.to(device=wdev, dtype=act_dtype)
        input_ids = input_ids.to(device=wdev)
        tok_emb = emb(input_ids).to(dtype=act_dtype)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

        # 5. Attention mask + labels with prefix region
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=wdev)
            ones = torch.ones((B, self.n_prefix), dtype=attention_mask.dtype, device=wdev)
            attn = torch.cat([ones, attention_mask], dim=1)
        else:
            attn = None

        if labels is not None:
            labels = labels.to(device=wdev)
            ignore_prefix = torch.full((B, self.n_prefix), -100, dtype=labels.dtype, device=wdev)
            labels = torch.cat([ignore_prefix, labels], dim=1)

        # 6. GXCA bridge
        if self.use_gxca:
            self.llm._mm_prefix_raw = prefix
            self.llm._mm_n_prefix = prefix.size(1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
        )

        if self.use_gxca:
            for k in ("_mm_prefix_raw", "_mm_n_prefix"):
                if hasattr(self.llm, k):
                    delattr(self.llm, k)

        # Combine main loss with aux reconstruction loss when applicable.
        if aux_loss is not None and getattr(out, "loss", None) is not None:
            total = out.loss + self.aux_lambda * aux_loss
            return SimpleNamespace(loss=total, logits=getattr(out, "logits", None))
        return out

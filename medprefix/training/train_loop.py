"""Training loop: per-epoch runner, parameter groups, scheduler."""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm


# ---------- parameter groups ----------

def get_trainable_params(model, stage: str = "full") -> List[torch.nn.Parameter]:
    """Filter trainable params; in ``stage='tab_only'`` we exclude image-side modules."""
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if stage == "tab_only":
            if (
                "projector.img_proj" in name
                or "projector.resampler_img" in name
                or "projector.null_img_prefix" in name
                or "type_img" in name
                or ".vision." in name
            ):
                continue
        out.append(p)
    return out


def get_param_groups_full(
    model,
    base_lr: float = 2e-4,
    wd: float = 1e-2,
    lr_tab_multiplier: float = 1.0,
):
    """Group parameters so we can apply different LRs to image vs tabular branches.

    The ``aux_head`` weights (auxiliary reconstruction heads, present when
    ``aux_lambda > 0``) are grouped with the tabular params and benefit from
    the same LR multiplier.
    """
    tab_params, img_params, other_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in (
            "tabtok", "type_tab", "resampler_tab", "tab_scale", "aux_head"
        )):
            tab_params.append(p)
        elif any(k in name for k in (
            "vision", "img_proj", "resampler_img", "type_img", "img_scale"
        )):
            img_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": tab_params,   "lr": base_lr * float(lr_tab_multiplier), "weight_decay": wd},
        {"params": img_params,   "lr": base_lr,                             "weight_decay": wd},
        {"params": other_params, "lr": base_lr,                             "weight_decay": wd},
    ]


def make_scheduler(optimizer, n_steps: int, warmup_ratio: float = 0.1):
    return OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=max(1, n_steps),
        pct_start=warmup_ratio,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )


# ---------- run one epoch ----------

def run_epoch(
    model,
    loader,
    *,
    device: str,
    optimizer=None,
    scheduler=None,
    scaler=None,
    train: bool = True,
    use_image: bool = True,
    use_tab: bool = True,
    max_grad_norm: float = 1.0,
    modality_dropout: bool = False,
    p_img_drop: float = 0.0,
    p_tab_drop: float = 0.0,
    desc: str = "",
) -> float:
    model.train() if train else model.eval()
    losses = []

    pbar = tqdm(loader, desc=desc or ("Train" if train else "Eval"), leave=False)
    vis_device = next(model.vision.parameters()).device
    emb_device = model.llm.get_input_embeddings().weight.device

    for batch in pbar:
        images, tab_batch, input_ids, labels, attn_mask, _, _ = batch

        _use_img = use_image
        _use_tab = use_tab
        if train and modality_dropout:
            if torch.rand(1).item() < p_img_drop:
                _use_img = False
            if torch.rand(1).item() < p_tab_drop:
                _use_tab = False
            if (not _use_img) and (not _use_tab):
                _use_img = True

        images = images.to(vis_device, non_blocking=True)
        age, sex_id, loc_id = tab_batch
        tab_batch = (
            age.to(vis_device, non_blocking=True),
            sex_id.to(vis_device, non_blocking=True),
            loc_id.to(vis_device, non_blocking=True),
        )
        input_ids = input_ids.to(emb_device, non_blocking=True)
        if attn_mask is not None:
            attn_mask = attn_mask.to(emb_device, non_blocking=True)
        if labels is not None:
            labels = labels.to(emb_device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(
                    images,
                    tab_batch,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attn_mask,
                    use_image=_use_img,
                    use_tab=_use_tab,
                )
                loss = out.loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                params_for_clip = []
                for g in optimizer.param_groups:
                    params_for_clip.extend(g["params"])
                torch.nn.utils.clip_grad_norm_(params_for_clip, max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        losses.append(float(loss.item()))
        if len(losses) >= 10:
            pbar.set_postfix(loss=float(np.mean(losses[-50:])))

    return float(np.mean(losses)) if losses else float("nan")

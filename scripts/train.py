#!/usr/bin/env python3
"""Train Med-Prefix on HAM10000 or ISIC 2019.

Example:
    python scripts/train.py \\
        --dataset ham10000 \\
        --image_dirs data/HAM10000/HAM10000_images_part_1 data/HAM10000/HAM10000_images_part_2 \\
        --instructions_json data/instructions/ham10000_instructions.json \\
        --output_dir results/ham10000_full \\
        --llm_profile qwen0p5b \\
        --epochs_phase1 2 --epochs_phase2 6 \\
        --n_tab_prefix 2 --n_img_prefix 14 \\
        --ablation_mode FULL
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medprefix.config import TrainConfig, add_train_args
from medprefix.utils.seed import set_seed
from medprefix.llm.builder import build_tokenizer_and_llm, add_special_tokens
from medprefix.data.splits import load_instructions_dataframe, split_df, build_vocab
from medprefix.data.dataset import MedDermDataset, ImageTransforms, make_collate_fn
from medprefix.models.medprefix import MedPrefixModel
from medprefix.training.train_loop import (
    run_epoch,
    get_trainable_params,
    get_param_groups_full,
    make_scheduler,
)
from medprefix.utils.viz import plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Med-Prefix on HAM10000 / ISIC 2019.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_train_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(**vars(args))

    set_seed(cfg.seed)
    device = cfg.device

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "checkpoint").mkdir(exist_ok=True)
    cfg.save(str(out_dir / "config.json"))

    # ------------------------------------------------------------------
    # 1. LLM + tokenizer
    # ------------------------------------------------------------------
    print(f"[1/6] Loading LLM ({cfg.llm_profile}) ...")
    tokenizer, llm, llm_info = build_tokenizer_and_llm(cfg.llm_profile, device=device)
    add_special_tokens(tokenizer, llm)
    pad_id = tokenizer.pad_token_id
    print(f"      d_model={llm_info['d_model']}  llm={llm_info['llm_name']}")

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    print(f"[2/6] Loading instructions from {cfg.instructions_json} ...")
    df = load_instructions_dataframe(cfg.image_dirs, cfg.instructions_json)
    print(f"      total (image, QA) rows: {len(df)}")

    sex_vocab = build_vocab(df["sex"])
    loc_vocab = build_vocab(df["localization"])
    print(f"      sex_vocab={list(sex_vocab.keys())}  loc_vocab={len(loc_vocab)} entries")

    train_df, val_df, test_df = split_df(df, 0.85, 0.075, 0.075, seed=cfg.seed)
    print(f"      train/val/test = {len(train_df)} / {len(val_df)} / {len(test_df)}")

    img_tf_train = ImageTransforms.train(cfg.img_size)
    img_tf_val = ImageTransforms.val(cfg.img_size)
    train_ds = MedDermDataset(train_df, tokenizer, cfg.max_len, transform=img_tf_train)
    val_ds = MedDermDataset(val_df,   tokenizer, cfg.max_len, transform=img_tf_val)

    collate = make_collate_fn(sex_vocab, loc_vocab, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate)

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    print("[3/6] Building Med-Prefix model ...")
    model = MedPrefixModel(
        llm_obj=llm, tokenizer_obj=tokenizer,
        vision_name=cfg.vision_name,
        n_tab_prefix=cfg.n_tab_prefix,
        n_img_prefix=cfg.n_img_prefix,
        freeze_vision=True,
        freeze_llm=True,
        sex_vocab=sex_vocab, loc_vocab=loc_vocab,
        tabular_mode=cfg.tabular_mode,
        perceiver_depth=cfg.perceiver_depth,
        perceiver_heads=cfg.perceiver_heads,
        use_gxca=cfg.use_gxca,
        gxca_layers=cfg.gxca_layers,
        gxca_heads=cfg.gxca_heads,
        aux_lambda=cfg.aux_lambda,
    ).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      trainable params: {n_train / 1e6:.2f} M")

    # ------------------------------------------------------------------
    # 4. Ablation gating
    # ------------------------------------------------------------------
    if cfg.ablation_mode == "FULL":
        do_phase1 = cfg.epochs_phase1 > 0
        do_phase2 = True
        use_img_full, use_tab_full = True, True
        use_drop = cfg.use_modality_dropout
    elif cfg.ablation_mode == "NO_TAB":
        do_phase1 = False
        do_phase2 = True
        use_img_full, use_tab_full = True, False
        use_drop = False
    elif cfg.ablation_mode == "NO_IMG":
        do_phase1 = cfg.epochs_phase1 > 0
        do_phase2 = True
        use_img_full, use_tab_full = False, True
        use_drop = False
    else:
        raise ValueError(cfg.ablation_mode)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    history = []

    # ------------------------------------------------------------------
    # 5. Phase 1 — tabular-only warm-up
    # ------------------------------------------------------------------
    if do_phase1:
        print("[4/6] Phase 1 — tabular-only warm-up")
        opt = torch.optim.AdamW(
            get_trainable_params(model, "tab_only"),
            lr=cfg.lr_tab, weight_decay=cfg.weight_decay,
        )
        sched = make_scheduler(opt, len(train_loader) * cfg.epochs_phase1, cfg.warmup_ratio)
        for e in range(1, cfg.epochs_phase1 + 1):
            tl = run_epoch(model, train_loader, device=device, optimizer=opt,
                           scheduler=sched, scaler=scaler, train=True,
                           use_image=False, use_tab=True,
                           max_grad_norm=cfg.max_grad_norm,
                           desc=f"P1 ep{e}/{cfg.epochs_phase1}")
            vl_tab = run_epoch(model, val_loader, device=device, train=False,
                               use_image=False, use_tab=True,
                               desc=f"P1 val ep{e}")
            history.append({"phase": "tab_only", "epoch": e,
                            "train_loss": tl, "val_loss_tab": vl_tab})
            print(f"  [P1 ep{e}] train={tl:.4f}  val(tab)={vl_tab:.4f}")
    else:
        print("[4/6] Phase 1 — skipped")

    # ------------------------------------------------------------------
    # 6. Phase 2 — main / ablation
    # ------------------------------------------------------------------
    print(f"[5/6] Phase 2 — {cfg.ablation_mode}")
    opt = torch.optim.AdamW(get_param_groups_full(
        model,
        base_lr=cfg.lr_full,
        wd=cfg.weight_decay,
        lr_tab_multiplier=cfg.lr_tab_multiplier,
    ))
    sched = make_scheduler(opt, len(train_loader) * cfg.epochs_phase2, cfg.warmup_ratio)
    best_val = float("inf")
    wait = 0
    best_path = out_dir / "checkpoint" / f"ckpt_best_{cfg.ablation_mode.lower()}.pt"

    for e in range(1, cfg.epochs_phase2 + 1):
        tl = run_epoch(
            model, train_loader, device=device,
            optimizer=opt, scheduler=sched, scaler=scaler, train=True,
            use_image=use_img_full, use_tab=use_tab_full,
            max_grad_norm=cfg.max_grad_norm,
            modality_dropout=use_drop,
            p_img_drop=cfg.p_img_drop, p_tab_drop=cfg.p_tab_drop,
            desc=f"P2 ep{e}/{cfg.epochs_phase2}",
        )
        vl = run_epoch(
            model, val_loader, device=device, train=False,
            use_image=use_img_full, use_tab=use_tab_full,
            desc=f"P2 val ep{e}",
        )
        history.append({"phase": f"main_{cfg.ablation_mode}", "epoch": e,
                        "train_loss": tl, "val_loss_main": vl})
        print(f"  [P2 ep{e}] train={tl:.4f}  val={vl:.4f}")

        if vl < best_val - 1e-4:
            best_val = vl
            wait = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best ({best_val:.4f}) to {best_path}")
        else:
            wait += 1
            if wait >= cfg.early_stop_patience:
                print(f"  -> early stop (patience={cfg.early_stop_patience})")
                break

    # ------------------------------------------------------------------
    # 7. Save history + curves
    # ------------------------------------------------------------------
    print("[6/6] Saving history and figures ...")
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)
    plot_training_curves(history, str(out_dir / "figures" / "training_curves.png"))

    print(f"\n[done] best_val={best_val:.4f}")
    print(f"       output_dir={out_dir}")
    print(f"       checkpoint={best_path}")


if __name__ == "__main__":
    main()

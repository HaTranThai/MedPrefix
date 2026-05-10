#!/usr/bin/env python3
"""Evaluate a trained Med-Prefix checkpoint on HAM10000 / ISIC 2019.

Example:
    python scripts/evaluate.py \\
        --dataset ham10000 \\
        --image_dirs data/HAM10000/HAM10000_images_part_1 data/HAM10000/HAM10000_images_part_2 \\
        --instructions_json data/instructions/ham10000_instructions.json \\
        --checkpoint results/ham10000_full/checkpoint/ckpt_best_full.pt \\
        --output_dir results/ham10000_full/eval \\
        --eval_split test --eval_mode FULL
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medprefix.config import EvalConfig, add_eval_args
from medprefix.utils.seed import set_seed
from medprefix.llm.builder import build_tokenizer_and_llm, add_special_tokens
from medprefix.data.splits import load_instructions_dataframe, split_df, build_vocab
from medprefix.data.dataset import MedDermDataset, ImageTransforms, make_collate_fn
from medprefix.models.medprefix import MedPrefixModel
from medprefix.eval.evaluator import evaluate_split
from medprefix.utils.viz import plot_metrics_bar, plot_sample_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a Med-Prefix checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_eval_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(**vars(args))

    set_seed(cfg.seed)
    device = cfg.device
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    # 1. LLM
    print(f"[1/4] Loading LLM ({cfg.llm_profile}) ...")
    tokenizer, llm, _ = build_tokenizer_and_llm(cfg.llm_profile, device=device)
    add_special_tokens(tokenizer, llm)
    pad_id = tokenizer.pad_token_id

    # 2. Data + splits (must match training seed/ratio)
    print(f"[2/4] Loading data ...")
    df = load_instructions_dataframe(cfg.image_dirs, cfg.instructions_json)
    sex_vocab = build_vocab(df["sex"])
    loc_vocab = build_vocab(df["localization"])
    train_df, val_df, test_df = split_df(df, 0.85, 0.075, 0.075, seed=cfg.seed)

    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[cfg.eval_split]
    print(f"      eval_split={cfg.eval_split}  size={len(eval_df)}")

    img_tf = ImageTransforms.val(cfg.img_size)
    eval_ds = MedDermDataset(eval_df, tokenizer, cfg.max_len, transform=img_tf)
    collate = make_collate_fn(sex_vocab, loc_vocab, pad_id=pad_id)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate)

    # 3. Model + checkpoint
    print(f"[3/4] Building model + loading {cfg.checkpoint} ...")
    model = MedPrefixModel(
        llm_obj=llm, tokenizer_obj=tokenizer,
        vision_name=cfg.vision_name,
        n_tab_prefix=cfg.n_tab_prefix,
        n_img_prefix=cfg.n_img_prefix,
        freeze_vision=True, freeze_llm=True,
        sex_vocab=sex_vocab, loc_vocab=loc_vocab,
        tabular_mode=cfg.tabular_mode,
        perceiver_depth=cfg.perceiver_depth,
        perceiver_heads=cfg.perceiver_heads,
        use_gxca=cfg.use_gxca,
        gxca_layers=cfg.gxca_layers,
        gxca_heads=cfg.gxca_heads,
        aux_lambda=cfg.aux_lambda,
    ).to(device)

    state = torch.load(cfg.checkpoint, map_location=device)
    info = model.load_state_dict(state, strict=False)
    print(f"      missing={len(info.missing_keys)}  unexpected={len(info.unexpected_keys)}")
    model.eval()

    # 4. Generate + score
    print(f"[4/4] Evaluating (mode={cfg.eval_mode}) ...")
    metrics, samples = evaluate_split(
        model, tokenizer, eval_loader, device=device,
        max_new_tokens=cfg.max_new_tokens,
        max_batches=cfg.max_eval_batches,
        mode=cfg.eval_mode,
        n_samples_to_keep=cfg.n_samples_to_keep,
    )

    # Save outputs
    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "mode", cfg.eval_mode)
    metrics_df.insert(0, "split", cfg.eval_split)
    metrics_df.to_csv(out_dir / "eval_metrics.csv", index=False)

    sample_records = [
        {"instruction": ins, "ground_truth": gt, "prediction": pr}
        for (_img, ins, gt, pr) in samples
    ]
    pd.DataFrame(sample_records).to_csv(out_dir / "eval_samples.csv", index=False)

    plot_metrics_bar(metrics, str(out_dir / "figures" / "eval_metrics.png"))
    plot_sample_predictions(samples, str(out_dir / "figures" / "sample_predictions.png"))

    print("\n[done] Metrics:")
    for k, v in metrics.items():
        print(f"   {k:20} {v:.4f}")
    print(f"\n   saved to {out_dir}")


if __name__ == "__main__":
    main()

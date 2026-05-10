#!/usr/bin/env python3
"""Sanity-check the ISIC 2019 layout and (optionally) deduplicate against HAM10000.

The Kaggle archive ``nischaydnk/isic-2019-jpg-224x224-resized`` ships with the
training images already resized to 224x224. This script just verifies that the
metadata + ground-truth CSVs exist and lines up the columns Med-Prefix expects.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--isic_dir", required=True,
                   help="Directory containing ISIC_2019_Training_Metadata.csv etc.")
    p.add_argument("--ham_dir", default=None,
                   help="(Optional) HAM10000 root for deduplication report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    isic_dir = Path(args.isic_dir)

    meta_csv = isic_dir / "ISIC_2019_Training_Metadata.csv"
    gt_csv = isic_dir / "ISIC_2019_Training_GroundTruth.csv"
    train_dir_candidates = [isic_dir / "train", isic_dir]

    print("== ISIC 2019 layout check ==")
    for f in (meta_csv, gt_csv):
        ok = f.is_file()
        print(f"  {'OK' if ok else 'MISSING':7}  {f}")

    img_dir = next((d for d in train_dir_candidates
                    if d.is_dir() and any(p.suffix.lower() in (".jpg", ".jpeg", ".png")
                                          for p in d.iterdir())), None)
    print(f"  image_dir = {img_dir}")

    if not (meta_csv.is_file() and gt_csv.is_file() and img_dir is not None):
        raise SystemExit(
            "\nISIC 2019 not laid out as expected. Download from\n"
            "  https://www.kaggle.com/datasets/nischaydnk/isic-2019-jpg-224x224-resized\n"
            "and unzip into a single directory.\n"
        )

    meta = pd.read_csv(meta_csv)
    gt = pd.read_csv(gt_csv)
    img_col_meta = "image" if "image" in meta.columns else "image_name"
    img_col_gt = "image" if "image" in gt.columns else "image_name"
    print(f"  metadata rows : {len(meta):,}")
    print(f"  groundtruth rows: {len(gt):,}")
    print(f"  unique image_ids: {meta[img_col_meta].nunique():,}")

    # Deduplication report against HAM10000 if possible.
    if args.ham_dir:
        ham_dir = Path(args.ham_dir)
        ham_meta = ham_dir / "HAM10000_metadata.csv"
        if ham_meta.is_file():
            ham = pd.read_csv(ham_meta)
            shared = set(ham["image_id"]) & set(meta[img_col_meta])
            print(f"\n  HAM10000 image overlap : {len(shared):,} / {len(meta):,}")
            uniq = len(meta) - len(shared)
            print(f"  ISIC 2019 unique after dedup: {uniq:,}")
        else:
            print(f"\n  (HAM metadata not found at {ham_meta}, skipping dedup)")


if __name__ == "__main__":
    main()

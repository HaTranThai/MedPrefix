"""Plotting utilities for training curves and evaluation outputs."""
from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_training_curves(history: List[dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tab_df = hist_df[hist_df["phase"] == "tab_only"]
    full_df = hist_df[hist_df["phase"].str.startswith("main")] if "phase" in hist_df.columns else pd.DataFrame()

    if len(tab_df) > 0:
        ax.plot(tab_df["epoch"], tab_df["train_loss"], "o-", label="Train (tab-only)",
                color="#2196F3", markersize=4)
        if "val_loss_tab" in tab_df.columns:
            ax.plot(tab_df["epoch"], tab_df["val_loss_tab"], "s--", label="Val (tab-only)",
                    color="#64B5F6", markersize=4)

    if len(full_df) > 0:
        offset = len(tab_df)
        ep = full_df["epoch"].values + offset
        ax.plot(ep, full_df["train_loss"], "o-", label="Train (full)",
                color="#F44336", markersize=4)
        val_col = "val_loss_main" if "val_loss_main" in full_df.columns else "val_loss_full"
        if val_col in full_df.columns:
            ax.plot(ep, full_df[val_col], "s--", label="Val (full)",
                    color="#EF9A9A", markersize=4)

    if len(tab_df) > 0 and len(full_df) > 0:
        ax.axvline(x=len(tab_df) - 0.5, color="gray", linestyle=":", alpha=0.6,
                   label="Phase switch")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Right: summary table
    ax2 = axes[1]
    ax2.axis("off")
    summary_data = []
    if len(tab_df) > 0:
        summary_data.append([
            "Phase 1 (Tab-only)", f"{len(tab_df)} epochs",
            f"{tab_df['train_loss'].iloc[-1]:.4f}",
            f"{tab_df['val_loss_tab'].iloc[-1]:.4f}" if "val_loss_tab" in tab_df.columns else "N/A",
        ])
    if len(full_df) > 0:
        val_col = "val_loss_main" if "val_loss_main" in full_df.columns else "val_loss_full"
        val_last = f"{full_df[val_col].iloc[-1]:.4f}" if val_col in full_df.columns else "N/A"
        summary_data.append([
            "Phase 2 (Full)", f"{len(full_df)} epochs",
            f"{full_df['train_loss'].iloc[-1]:.4f}", val_last,
        ])
    if summary_data:
        tbl = ax2.table(
            cellText=summary_data,
            colLabels=["Phase", "Epochs", "Final Train Loss", "Final Val Loss"],
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
        ax2.set_title("Training Summary", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_bar(metrics: dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    preferred = ["ROUGE_L", "METEOR", "Token_F1", "Content_Recall", "Diagnosis_Accuracy"]
    names = [k for k in preferred if k in metrics]
    values = [metrics[k] for k in names]
    display_names = [n.replace("_", "\n") for n in names]

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = [cmap(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    bars = ax.bar(display_names, values, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Evaluation Metrics", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _denorm(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x * s + m).clamp(0, 1)


def plot_sample_predictions(samples: List[Tuple], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    n = len(samples)
    if n == 0:
        return
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5.5 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (img_t, ins, gt, pred) in enumerate(samples):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        img = _denorm(img_t[0]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])

        ins_short = ins[:60] + "..." if len(ins) > 60 else ins
        gt_short = gt[:50] + "..." if len(gt) > 50 else gt
        pred_short = pred[:50] + "..." if len(pred) > 50 else pred

        match = gt.strip().lower() == pred.strip().lower()
        border = "#4CAF50" if match else "#F44336"
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3)
        cap = f"Q: {ins_short}\nGT: {gt_short}\nPred: {pred_short}"
        ax.set_title(cap, fontsize=7, loc="left", pad=4, linespacing=1.4,
                     fontfamily="monospace")

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    plt.suptitle("Sample Predictions  (Green = Match, Red = Mismatch)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

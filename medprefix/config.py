"""Argparse-friendly config dataclasses for train/eval entry points."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import argparse
import json


# ---------------------- TRAIN ----------------------
@dataclass
class TrainConfig:
    # Data
    dataset: str = "ham10000"          # ham10000 | isic2019
    image_dirs: List[str] = field(default_factory=list)
    instructions_json: str = ""
    output_dir: str = "results/run"

    # LLM
    llm_profile: str = "qwen0p5b"      # qwen0p5b | qwen1p5b | tinyllama

    # Vision / Tabular
    vision_name: str = "vit_base_patch16_224"
    tabular_mode: str = "BASELINE"     # BASELINE | MLP | FT_TRANSFORMER | TEXT
    img_size: int = 224
    n_tab_prefix: int = 2              # paper: 2 for both HAM10000 and ISIC 2019
    n_img_prefix: int = 14
    perceiver_depth: int = 2
    perceiver_heads: int = 8

    # GXCA
    use_gxca: bool = True
    gxca_layers: int = 4
    gxca_heads: Optional[int] = None

    # Auxiliary reconstruction loss (optional; OFF in the paper for both datasets)
    aux_lambda: float = 0.0
    lr_tab_multiplier: float = 1.0     # multiply base LR for tabular-side params; paper uses 1.0

    # Training schedule
    batch_size: int = 8
    max_len: int = 192
    epochs_phase1: int = 2             # tabular-only warm-up
    epochs_phase2: int = 6             # full tri-modal
    lr_tab: float = 2e-4
    lr_full: float = 2e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    grad_accum: int = 2

    # Ablation / dropout
    ablation_mode: str = "FULL"        # FULL | NO_TAB | NO_IMG
    use_modality_dropout: bool = True
    p_img_drop: float = 0.15
    p_tab_drop: float = 0.30

    # Misc
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda:0"
    early_stop_patience: int = 3

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    # Data
    parser.add_argument("--dataset", choices=["ham10000", "isic2019"], default="ham10000")
    parser.add_argument("--image_dirs", nargs="+", required=True,
                        help="One or more directories containing the dermoscopic JPGs.")
    parser.add_argument("--instructions_json", required=True,
                        help="Path to the instruction-response JSON (download from the "
                             "HuggingFace datasets HaTranThai/HAM10000-Instruction or "
                             "HaTranThai/ISIC2019-Instructions).")
    parser.add_argument("--output_dir", default="results/run")

    # LLM
    parser.add_argument("--llm_profile", choices=["qwen0p5b", "qwen1p5b", "tinyllama"],
                        default="qwen0p5b")

    # Vision / Tabular
    parser.add_argument("--vision_name", default="vit_base_patch16_224")
    parser.add_argument("--tabular_mode",
                        choices=["BASELINE", "MLP", "FT_TRANSFORMER", "TEXT"],
                        default="BASELINE")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_tab_prefix", type=int, default=2)
    parser.add_argument("--n_img_prefix", type=int, default=14)
    parser.add_argument("--perceiver_depth", type=int, default=2)
    parser.add_argument("--perceiver_heads", type=int, default=8)

    # GXCA
    parser.add_argument("--use_gxca", type=str2bool, default=True)
    parser.add_argument("--gxca_layers", type=int, default=4)
    parser.add_argument("--gxca_heads", type=int, default=None)

    # Aux reconstruction loss (ISIC 2019 setup uses 0.2; HAM10000 uses 0.0)
    parser.add_argument("--aux_lambda", type=float, default=0.0)
    parser.add_argument("--lr_tab_multiplier", type=float, default=1.0)

    # Schedule
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--epochs_phase1", type=int, default=2)
    parser.add_argument("--epochs_phase2", type=int, default=6)
    parser.add_argument("--lr_tab", type=float, default=2e-4)
    parser.add_argument("--lr_full", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=2)

    # Ablation
    parser.add_argument("--ablation_mode",
                        choices=["FULL", "NO_TAB", "NO_IMG"], default="FULL")
    parser.add_argument("--use_modality_dropout", type=str2bool, default=True)
    parser.add_argument("--p_img_drop", type=float, default=0.15)
    parser.add_argument("--p_tab_drop", type=float, default=0.30)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--early_stop_patience", type=int, default=3)


# ---------------------- EVAL ----------------------
@dataclass
class EvalConfig:
    dataset: str = "ham10000"
    image_dirs: List[str] = field(default_factory=list)
    instructions_json: str = ""
    checkpoint: str = ""
    output_dir: str = "results/eval"

    llm_profile: str = "qwen0p5b"
    vision_name: str = "vit_base_patch16_224"
    tabular_mode: str = "BASELINE"
    img_size: int = 224
    n_tab_prefix: int = 2
    n_img_prefix: int = 14
    perceiver_depth: int = 2
    perceiver_heads: int = 8
    use_gxca: bool = True
    gxca_layers: int = 4
    gxca_heads: Optional[int] = None
    aux_lambda: float = 0.0

    batch_size: int = 1
    max_len: int = 192
    max_new_tokens: int = 64
    eval_split: str = "test"           # train | val | test
    eval_mode: str = "FULL"            # FULL | NO_TAB | NO_IMG
    max_eval_batches: Optional[int] = None
    n_samples_to_keep: int = 10

    seed: int = 42
    num_workers: int = 2
    device: str = "cuda:0"


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["ham10000", "isic2019"], default="ham10000")
    parser.add_argument("--image_dirs", nargs="+", required=True)
    parser.add_argument("--instructions_json", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .pt file saved by train.py")
    parser.add_argument("--output_dir", default="results/eval")

    parser.add_argument("--llm_profile", choices=["qwen0p5b", "qwen1p5b", "tinyllama"],
                        default="qwen0p5b")
    parser.add_argument("--vision_name", default="vit_base_patch16_224")
    parser.add_argument("--tabular_mode",
                        choices=["BASELINE", "MLP", "FT_TRANSFORMER", "TEXT"],
                        default="BASELINE")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--n_tab_prefix", type=int, default=2)
    parser.add_argument("--n_img_prefix", type=int, default=14)
    parser.add_argument("--perceiver_depth", type=int, default=2)
    parser.add_argument("--perceiver_heads", type=int, default=8)
    parser.add_argument("--use_gxca", type=str2bool, default=True)
    parser.add_argument("--gxca_layers", type=int, default=4)
    parser.add_argument("--gxca_heads", type=int, default=None)
    parser.add_argument("--aux_lambda", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--eval_mode", choices=["FULL", "NO_TAB", "NO_IMG"], default="FULL")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--n_samples_to_keep", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")

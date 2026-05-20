# Med-Prefix

**Tri-Modal Prefix Conditioning for Instruction-Following Dermatology Report Generation**

Official implementation of the Med-Prefix paper. The model conditions a frozen LLM (Qwen) on three modalities — dermoscopic images, structured patient metadata (age, sex, anatomical site), and a natural-language instruction — through a compact learnable prefix produced by a Dual Perceiver Resampler, plus Gated Cross-Attention adapters injected into the last decoder blocks.

---

## 1. Setup

```bash
git clone https://github.com/HaTranThai/Med-Prefix.git medprefix
cd medprefix
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tested with Python 3.10–3.12, PyTorch 2.x, CUDA 12.x, NVIDIA RTX 3090 / A100 (≥ 16 GB VRAM).

---

## 2. Data preparation
| # | Purpose | URL |
|---|---|---|
| 1 | HAM10000 images | [HAM1000](https://www.nature.com/articles/sdata2018161) |
| 2 | ISIC 2019 images | [ISIC2019](https://challenge.isic-archive.com/landing/2019/) |
| 3 | HAM10000 instruction–response JSON | [HAM10000-Instruction](https://huggingface.co/datasets/HaTranThai/HAM10000-Instruction) |
| 4 | ISIC 2019 instruction–response JSON | [ISIC2019-Instructions](https://huggingface.co/datasets/HaTranThai/ISIC2019-Instructions) |

---

## 3. Training

### Quick start (HAM10000, FULL Med-Prefix)

```bash
python scripts/train.py \
    --dataset ham10000 \
    --image_dirs data/HAM10000/HAM10000_images_part_1 data/HAM10000/HAM10000_images_part_2 \
    --instructions_json data/instructions/ham10000_instructions.json \
    --output_dir results/ham10000_full \
    --llm_profile qwen0p5b \
    --batch_size 8 --epochs_phase1 2 --epochs_phase2 6 \
    --n_tab_prefix 2 --n_img_prefix 14 \
    --ablation_mode FULL
```

### ISIC 2019

```bash
python scripts/train.py \
    --dataset isic2019 \
    --image_dirs data/ISIC_2019/train-image/image \
    --instructions_json data/instructions/isic2019_instruction.json \
    --output_dir results/isic2019_full \
    --llm_profile qwen0p5b \
    --batch_size 8 --epochs_phase1 2 --epochs_phase2 6 \
    --n_tab_prefix 2 --n_img_prefix 14 \
    --ablation_mode FULL
```

> **Note:** The Kaggle archive `nischaydnk/isic-2019-jpg-224x224-resized`
> extracts images into `train-image/image/` (not just `train/`). Adjust
> `--image_dirs` if your local layout is different.

### Ablation studies

| Mode | Flag | Description |
|---|---|---|
| Full Med-Prefix | `--ablation_mode FULL` | image + tabular + GXCA + curriculum + dropout |
| Image-only | `--ablation_mode NO_TAB` | drop the tabular branch |
| Tabular-only | `--ablation_mode NO_IMG` | drop the visual branch |

### LLM backbone variants

| Profile | Model | Notes |
|---|---|---|
| `qwen0p5b` | Qwen/Qwen2.5-0.5B-Instruct | default, fits in 16 GB |
| `qwen1p5b` | Qwen/Qwen2.5-1.5B-Instruct | uses LoRA (r=16); needs ≥ 24 GB |
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | alternative small backbone |

All other hyperparameters (LR, weight decay, dropout, GXCA layers, …) are exposed as CLI flags. See `python scripts/train.py --help` for the full list.

---

## 4. Evaluation

```bash
python scripts/evaluate.py \
    --dataset ham10000 \
    --image_dirs data/HAM10000/HAM10000_images_part_1 data/HAM10000/HAM10000_images_part_2 \
    --instructions_json data/instructions/ham10000_instructions.json \
    --checkpoint results/ham10000_full/checkpoint/ckpt_best_full.pt \
    --output_dir results/ham10000_full/eval \
    --llm_profile qwen0p5b \
    --max_new_tokens 64 \
    --eval_split test
```

Outputs:

* `results/.../eval/eval_metrics.csv` — ROUGE-L, METEOR, Token F1, Content Recall, Diagnostic Accuracy
* `results/.../eval/eval_samples.csv` — per-sample predictions
* `results/.../eval/figures/` — bar chart of metrics, sample-prediction grid

---

## 5. Project layout

```
medprefix/
├── medprefix/
│   ├── config.py              # dataclass mirroring all CLI flags
│   ├── data/
│   │   ├── io_utils.py        # prompt template & label masking
│   │   ├── dataset.py         # PyTorch Dataset + collate_fn
│   │   └── splits.py          # 85/7.5/7.5 stratified split, vocab build
│   ├── llm/builder.py         # build_tokenizer_and_llm (qwen0p5b/1p5b/tinyllama)
│   ├── models/
│   │   ├── vision.py          # frozen ViT patch-token encoder (timm)
│   │   ├── tabular.py         # 3 tabular encoders (Baseline/MLP/FT-Transformer)
│   │   ├── perceiver.py       # PerceiverResampler + TokenFusionProjector
│   │   ├── gxca.py            # GatedCrossAttnSDPA + injection helper
│   │   └── medprefix.py       # MultiModalPerceiverPrefix (full model)
│   ├── training/
│   │   ├── train_loop.py      # run_epoch, param groups, scheduler
│   │   └── metrics.py         # ROUGE/METEOR/TokenF1/ContentRecall/DiagAcc
│   ├── eval/
│   │   ├── generate.py        # greedy generation with prefix injection
│   │   └── evaluator.py       # evaluate_split → metrics dict + samples
│   └── utils/
│       ├── seed.py
│       └── viz.py             # training-curve & metric-bar plots
├── scripts/
│   ├── train.py               # main training entry (argparse)
│   ├── evaluate.py            # eval entry (argparse)
│   ├── generate_instructions.py   # build instruction–response JSON via teacher LLM
│   └── prepare_isic2019.py    # convert ISIC metadata → HAM-style schema
├── configs/default.yaml       # optional YAML config (overridden by CLI)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Citation

If you use this code, please cite:

```
@article{medprefix2026,
  title   = {Med-Prefix: Tri-Modal Prefix Conditioning for Instruction-Following Dermatology Report Generation},
  author  = {Tran Thai Ha and Bui Thanh Hung},
  journal = {The Visual Computer},
  year    = {2026}
}
```

---

## 7. License

Code: MIT. Datasets follow their respective licenses (HAM10000 — CC BY-NC; ISIC 2019 — CC BY-NC).

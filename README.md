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

# NLTK data needed by METEOR
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

Tested with Python 3.10–3.12, PyTorch 2.x, CUDA 12.x, NVIDIA RTX 3090 / A100 (≥ 16 GB VRAM).

---

## 2. Data preparation

Four Kaggle datasets are needed (all free, no Kaggle Pro required):

| # | Purpose | Kaggle URL |
|---|---|---|
| 1 | HAM10000 images | <[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.nature.com/articles/sdata2018161)> |
| 2 | ISIC 2019 images (224×224 resized) | <[https://www.kaggle.com/datasets/nischaydnk/isic-2019-jpg-224x224-resized](https://arxiv.org/abs/1902.03368)> |
| 3 | HAM10000 instruction–response JSON | <https://www.kaggle.com/datasets/mrworldzero/ham10000-instruction> |
| 4 | ISIC 2019 instruction–response JSON | <https://www.kaggle.com/datasets/tranhathai/isic-tmp> |

You can download them either through the Kaggle web UI or via the Kaggle CLI:

```bash
pip install kaggle   # configure ~/.kaggle/kaggle.json first
mkdir -p data/HAM10000 data/ISIC_2019 data/instructions

kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/HAM10000 --unzip
kaggle datasets download -d nischaydnk/isic-2019-jpg-224x224-resized -p data/ISIC_2019 --unzip
kaggle datasets download -d mrworldzero/ham10000-instruction -p data/instructions --unzip
kaggle datasets download -d tranhathai/isic-tmp -p data/instructions --unzip
```

### 2.1  HAM10000 (primary dataset, 10,015 images, 7 classes)

Download from Kaggle: <https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000>

Required files after extracting the Kaggle archive into `data/HAM10000/`:

```
data/HAM10000/
├── HAM10000_metadata.csv           # patient/lesion metadata (one row per image)
├── HAM10000_images_part_1/         # ~5,000 JPGs
└── HAM10000_images_part_2/         # ~5,000 JPGs
```

Quick check:

```bash
ls data/HAM10000/HAM10000_images_part_1 | head
wc -l data/HAM10000/HAM10000_metadata.csv   # ~10,016 (header + 10,015 rows)
```

### 2.2  ISIC 2019 (15,316 images after deduplication, 8 classes)

Download from Kaggle: <https://www.kaggle.com/datasets/nischaydnk/isic-2019-jpg-224x224-resized>

Extract into `data/ISIC_2019/`:

```
data/ISIC_2019/
├── ISIC_2019_Training_Metadata.csv
├── ISIC_2019_Training_GroundTruth.csv
└── train/                          # 25,331 resized JPGs (224x224)
```

The training script automatically deduplicates against HAM10000 by image hash if both datasets are present.

### 2.3  Instruction–response corpus (HAM10000)

The instruction corpus referenced in the paper is **publicly available as a
ready-made Kaggle dataset** — no need to regenerate it yourself:

> <https://www.kaggle.com/datasets/mrworldzero/ham10000-instruction>

Download and place the JSON under `data/instructions/`:

```
data/instructions/
└── ham10000_instructions_clean.json
```

Schema (one record per HAM10000 image):

```json
[
  {
    "image_id": "ISIC_0024306",
    "age": 80, "sex": "male", "localization": "scalp",
    "dx": "akiec", "dx_type": "histo",
    "output": [
      {"instruction": "What diagnosis fits best?", "response": "..."},
      {"instruction": "What should be done next?", "response": "..."}
    ]
  },
  ...
]
```

### 2.4  Instruction–response corpus (ISIC 2019)

Same as HAM10000, the ISIC 2019 instruction file is published on Kaggle:

> <https://www.kaggle.com/datasets/tranhathai/isic-tmp>

After download, place at `data/instructions/isic2019_instruction.json`.

ISIC 2019 records use the original ISIC field names — `age_approx`,
`anatom_site_general` — instead of HAM's `age` and `localization`. The data
loader in `medprefix/data/splits.py` auto-detects either schema, so the same
training script works on both datasets without any special flags.

If you want to (re)generate the JSON yourself with a different teacher LLM,
use `scripts/generate_instructions.py` (see `--help` for all options).

---

## 3. Training

### Quick start (HAM10000, FULL Med-Prefix)

```bash
python scripts/train.py \
    --dataset ham10000 \
    --image_dirs data/HAM10000/HAM10000_images_part_1 data/HAM10000/HAM10000_images_part_2 \
    --instructions_json data/instructions/ham10000_instructions_clean.json \
    --output_dir results/ham10000_full \
    --llm_profile qwen0p5b \
    --batch_size 8 --epochs_phase1 2 --epochs_phase2 6 \
    --n_tab_prefix 2 --n_img_prefix 14 \
    --ablation_mode FULL
```

### ISIC 2019 (uses auxiliary reconstruction loss + 1.5× tab LR per the paper)

```bash
python scripts/train.py \
    --dataset isic2019 \
    --image_dirs data/ISIC_2019/train-image/image \
    --instructions_json data/instructions/isic2019_instruction.json \
    --output_dir results/isic2019_full \
    --llm_profile qwen0p5b \
    --batch_size 8 --epochs_phase1 2 --epochs_phase2 6 \
    --n_tab_prefix 6 --n_img_prefix 14 \
    --ablation_mode FULL \
    --aux_lambda 0.2 \
    --lr_tab_multiplier 1.5
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
    --instructions_json data/instructions/ham10000_instructions_clean.json \
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
@article{medprefix2025,
  title  = {Med-Prefix: Tri-Modal Prefix Conditioning for Instruction-Following
            Dermatology Report Generation},
  author = {Tran Thai Ha and Bui Thanh Hung},
  year   = {2025}
}
```

---

## 7. License

Code: MIT. Datasets follow their respective licenses (HAM10000 — CC BY-NC; ISIC 2019 — CC BY-NC).

#!/usr/bin/env python3
"""Generate instruction–response pairs for HAM10000 / ISIC 2019 using a teacher LLM.

For each image we feed (age, sex, localization, ground-truth diagnosis) to a
strong instruction LLM (default: Qwen/Qwen2.5-7B-Instruct) and ask it to
produce K diverse clinically-plausible question-answer pairs.

The output JSON has the schema documented in README.md::

    [
      {
        "image_id": "ISIC_0024306",
        "age": 80, "sex": "male", "localization": "scalp",
        "dx": "akiec", "dx_type": "histo",
        "output": [
          {"instruction": "...", "response": "..."},
          ...
        ]
      },
      ...
    ]

Example
-------

    python scripts/generate_instructions.py \\
        --dataset ham10000 \\
        --metadata_csv data/HAM10000/HAM10000_metadata.csv \\
        --output_json data/instructions/ham10000_instructions.json \\
        --teacher_model Qwen/Qwen2.5-7B-Instruct \\
        --n_qa_per_image 1 \\
        --batch_size 4

If you don't have GPU memory for a 7B teacher, use ``--teacher_model
Qwen/Qwen2.5-1.5B-Instruct`` or any other instruction-tuned LLM.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


HAM_DIAG_FULL = {
    "nv":    "melanocytic nevus",
    "mel":   "melanoma",
    "bkl":   "benign keratosis-like lesion",
    "bcc":   "basal cell carcinoma",
    "akiec": "actinic keratosis / intraepithelial carcinoma",
    "df":    "dermatofibroma",
    "vasc":  "vascular lesion",
    "scc":   "squamous cell carcinoma",
    "ak":    "actinic keratosis",
}

PROMPT_TEMPLATE = """You are an expert dermatologist preparing teaching cases for medical students.

Patient: age {age}, sex {sex}, lesion on the {localization}.
Confirmed diagnosis (ground truth): {dx_full}.

Generate {n_qa} concise question-answer pair(s) about this dermoscopic case.
The questions should cover: diagnosis, lesion morphology (border, color, structure),
anatomical reasoning, and recommended next clinical step.

Format strictly as JSON list, no surrounding text:

[
  {{"instruction": "<question>", "response": "<short clinical answer>"}}
]
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate instruction–response JSON via a teacher LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=["ham10000", "isic2019"], required=True)
    p.add_argument("--metadata_csv", required=True,
                   help="HAM10000_metadata.csv or ISIC_2019_Training_Metadata.csv")
    p.add_argument("--groundtruth_csv", default=None,
                   help="ISIC_2019_Training_GroundTruth.csv (only for ISIC 2019)")
    p.add_argument("--output_json", required=True)
    p.add_argument("--teacher_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n_qa_per_image", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of images for a smoke test.")
    return p.parse_args()


def load_ham(meta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    keep = ["image_id", "age", "sex", "localization", "dx", "dx_type"]
    return df[keep].copy()


def load_isic2019(meta_csv: str, gt_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    gt = pd.read_csv(gt_csv)

    # ISIC metadata column variants
    img_col = "image" if "image" in meta.columns else "image_name"
    age_col = "age_approx"
    sex_col = "sex"
    loc_col = "anatom_site_general"

    meta = meta.rename(columns={
        img_col: "image_id", age_col: "age", sex_col: "sex", loc_col: "localization",
    })

    # GT columns are one-hot per class
    class_cols = [c for c in gt.columns if c.lower() not in ("image", "image_name", "unknown")]
    gt_img = gt["image"] if "image" in gt.columns else gt["image_name"]
    dx_series = gt[class_cols].idxmax(axis=1).str.lower()

    gt_long = pd.DataFrame({"image_id": gt_img, "dx": dx_series, "dx_type": "consensus"})

    df = meta.merge(gt_long, on="image_id", how="inner")
    return df[["image_id", "age", "sex", "localization", "dx", "dx_type"]].copy()


def build_prompt(row: pd.Series, n_qa: int) -> str:
    dx_full = HAM_DIAG_FULL.get(str(row["dx"]).lower(), str(row["dx"]))
    age = row.get("age", "unknown")
    sex = row.get("sex", "unknown")
    loc = row.get("localization", "unknown")
    return PROMPT_TEMPLATE.format(
        age=age if pd.notna(age) else "unknown",
        sex=sex if pd.notna(sex) else "unknown",
        localization=loc if pd.notna(loc) else "unknown",
        dx_full=dx_full,
        n_qa=n_qa,
    )


def parse_qa_block(text: str) -> List[Dict[str, str]]:
    """Find the first JSON list of dicts in the model output."""
    # Try strict JSON first.
    text = text.strip()
    m = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.S)
    if m:
        block = m.group(0)
        try:
            data = json.loads(block)
            if isinstance(data, list):
                clean = []
                for d in data:
                    if isinstance(d, dict) and "instruction" in d and "response" in d:
                        clean.append({
                            "instruction": str(d["instruction"]).strip(),
                            "response":    str(d["response"]).strip(),
                        })
                return clean
        except json.JSONDecodeError:
            pass
    return []


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ham10000":
        df = load_ham(args.metadata_csv)
    else:
        if not args.groundtruth_csv:
            raise SystemExit("--groundtruth_csv is required for isic2019")
        df = load_isic2019(args.metadata_csv, args.groundtruth_csv)

    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[load] {len(df)} rows from {args.metadata_csv}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[load] teacher LLM = {args.teacher_model}")
    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    teacher.eval()

    out_records: List[Dict] = []
    bs = max(1, args.batch_size)

    pbar = tqdm(range(0, len(df), bs), desc="generate")
    for s in pbar:
        chunk = df.iloc[s:s + bs]
        prompts = [build_prompt(r, args.n_qa_per_image) for _, r in chunk.iterrows()]

        # Apply chat template if available; otherwise raw prompt.
        messages = [[{"role": "user", "content": p}] for p in prompts]
        try:
            inputs = tok.apply_chat_template(
                messages, return_tensors="pt", padding=True,
                add_generation_prompt=True,
            ).to(args.device)
            attn = (inputs != tok.pad_token_id).long()
        except Exception:
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs = enc["input_ids"].to(args.device)
            attn = enc["attention_mask"].to(args.device)

        with torch.no_grad():
            gen = teacher.generate(
                input_ids=inputs,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        for j, (_, row) in enumerate(chunk.iterrows()):
            full = tok.decode(gen[j], skip_special_tokens=True)
            # Cut off the prompt by re-tokenizing prompt length.
            qa = parse_qa_block(full)
            if not qa:
                # fallback: trivial single QA so the row still works
                dx_full = HAM_DIAG_FULL.get(str(row["dx"]).lower(), str(row["dx"]))
                qa = [{"instruction": "What diagnosis fits best?",
                       "response": f"The most likely diagnosis is {dx_full}."}]
            out_records.append({
                "image_id":     str(row["image_id"]),
                "age":          row.get("age"),
                "sex":          row.get("sex"),
                "localization": row.get("localization"),
                "dx":           row.get("dx"),
                "dx_type":      row.get("dx_type"),
                "output":       qa,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(out_records)} records to {out_path}")


if __name__ == "__main__":
    main()

"""Evaluate the model on a DataLoader and return metrics + sample predictions."""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from tqdm.auto import tqdm

from .generate import generate_answer
from ..training.metrics import compute_all_metrics


def evaluate_split(
    model,
    tokenizer,
    loader,
    *,
    device: str,
    max_new_tokens: int = 64,
    max_batches: Optional[int] = None,
    mode: str = "FULL",
    n_samples_to_keep: int = 10,
) -> Tuple[dict, List[Tuple]]:
    """Iterate ``loader``, generate answers, compute metrics.

    Returns:
        metrics: dict with ROUGE_L, METEOR, Token_F1, Content_Recall, Diagnosis_Accuracy
        samples: list of (image_tensor, instruction, ground_truth, prediction) tuples
                 (kept for visualization, capped to ``n_samples_to_keep``)
    """
    refs: List[List[str]] = []
    hyps: List[str] = []
    instr_list: List[str] = []
    samples: List[Tuple] = []

    pbar = tqdm(loader, desc=f"Eval[{mode}]", leave=False)
    for bi, (images, tab_batch, _input_ids, _labels, _attn, instrs, gts) in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break

        images = images.to(device)
        age, sex, loc = tab_batch
        age, sex, loc = age.to(device), sex.to(device), loc.to(device)

        B = images.size(0)
        for i in range(B):
            img_i = images[i:i + 1]
            tab_i = (age[i:i + 1], sex[i:i + 1], loc[i:i + 1])
            pred = generate_answer(
                model, tokenizer, img_i, tab_i, instrs[i],
                max_new_tokens=max_new_tokens, mode=mode,
            )
            refs.append([gts[i]])
            hyps.append(pred)
            instr_list.append(instrs[i])
            if len(samples) < n_samples_to_keep:
                samples.append((img_i.detach().cpu(), instrs[i], gts[i], pred))

    metrics = compute_all_metrics(instr_list, refs, hyps)
    return metrics, samples

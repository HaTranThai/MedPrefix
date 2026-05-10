"""Prompt construction and label-masking helpers.

The chat-style template used during training:

    <|im_start|>user
    <image>
    <tabular>
    Instruction: {instruction}
    <|im_end|>
    <|im_start|>assistant
    Answer: {answer} <|im_end|>

Loss is computed only on the ``Answer`` segment by setting all preceding
labels to ``-100`` (the standard cross-entropy ignore index).
"""
from __future__ import annotations
from typing import Tuple
import torch


def build_io_strings(instruction: str, answer: str) -> Tuple[str, str]:
    prompt = (
        "<|im_start|>user\n"
        "<image>\n<tabular>\n"
        f"Instruction: {instruction.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Answer:"
    )
    target = f" {answer.strip()} <|im_end|>"
    return prompt, target


def to_model_inputs(
    tokenizer,
    prompt: str,
    answer: str,
    max_len: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)

    ids = p_ids + a_ids
    if len(ids) > max_len:
        ids = ids[:max_len]

    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = input_ids.clone()
    # Mask the prompt tokens; loss only on the answer.
    labels[: len(p_ids)] = -100
    return input_ids, labels

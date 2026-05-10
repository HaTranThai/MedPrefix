"""Build the LLM backbone + tokenizer used as the conditioned generator."""
from __future__ import annotations
from typing import Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


_PROFILES = {
    "qwen0p5b":  {"name": "Qwen/Qwen2.5-0.5B-Instruct",      "use_lora": False},
    "qwen1p5b":  {"name": "Qwen/Qwen2.5-1.5B-Instruct",      "use_lora": True},
    "tinyllama": {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "use_lora": False},
}


def build_tokenizer_and_llm(
    profile: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Return ``(tokenizer, llm, info_dict)``.

    Profiles:
        ``qwen0p5b``  — Qwen2.5-0.5B-Instruct, no LoRA
        ``qwen1p5b``  — Qwen2.5-1.5B-Instruct, LoRA wrapping
        ``tinyllama`` — TinyLlama-1.1B-Chat-v1.0, no LoRA
    """
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown profile '{profile}'. Available: {list(_PROFILES.keys())}"
        )

    spec = _PROFILES[profile]
    llm_name = spec["name"]
    use_lora = spec["use_lora"]

    tokenizer = AutoTokenizer.from_pretrained(
        llm_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm = llm.to(device)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_cfg)
        llm.print_trainable_parameters()

    d_model = llm.get_input_embeddings().embedding_dim
    info = {
        "use_lora": use_lora,
        "d_model": d_model,
        "llm_name": llm_name,
    }
    return tokenizer, llm, info


def add_special_tokens(tokenizer, llm) -> int:
    """Add prompt special tokens; return number added."""
    specials = [
        "<|im_start|>", "<|im_end|>",
        "<image>", "<tabular>",
        "Instruction:", "Answer:",
    ]
    to_add = [t for t in specials if t not in tokenizer.get_vocab()]
    added = tokenizer.add_tokens(to_add) if to_add else 0
    if added > 0:
        llm.resize_token_embeddings(len(tokenizer))

    if tokenizer.eos_token is None:
        for tk in ["<|im_end|>", "<|endoftext|>", "</s>"]:
            if tk in tokenizer.get_vocab():
                tokenizer.eos_token = tk
                break
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return added

"""Greedy generation with multimodal prefix injection."""
from __future__ import annotations
import torch

from ..data.io_utils import build_io_strings


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    image: torch.Tensor,
    tab_batch_1,
    instruction: str,
    *,
    max_new_tokens: int = 64,
    mode: str = "FULL",
) -> str:
    """Greedy answer generation for a single (image, tab, instruction) example.

    ``mode`` controls modality dropout at inference: ``FULL``, ``NO_TAB``, ``NO_IMG``.
    """
    model.eval()
    use_img = (mode != "NO_IMG")
    use_tab = (mode != "NO_TAB")

    device = next(model.llm.parameters()).device
    vis_dtype = next(model.vision.parameters()).dtype

    # Vision
    if use_img:
        image = image.to(device=device, dtype=vis_dtype)
        img_tokens = model.vision(image)
        type_img = model.type_img.to(device=device, dtype=vis_dtype)
    else:
        img_tokens = None
        type_img = None

    # Tabular
    if use_tab and getattr(model, "tabtok", None) is not None:
        age, sex, loc = tab_batch_1
        tab_batch_1 = (age.to(device), sex.to(device), loc.to(device))
        tab_tokens = model.tabtok(*tab_batch_1)
    else:
        tab_tokens = model.null_tab.to(device=device, dtype=vis_dtype).expand(1, -1, -1)
    type_tab = model.type_tab.to(device=device, dtype=vis_dtype)

    # Project to prefix
    prefix = model.projector(
        img_tokens, tab_tokens, type_img=type_img, type_tab=type_tab
    )

    emb_dtype = model.llm.get_input_embeddings().weight.dtype
    prefix = prefix.to(device=device, dtype=emb_dtype)

    prompt, _ = build_io_strings(instruction, "")
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    tok_emb = model.llm.get_input_embeddings()(input_ids).to(dtype=emb_dtype)

    inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
    attn = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=torch.long)

    # Make GXCA see the prefix during generation as well.
    if getattr(model, "use_gxca", False):
        model.llm._mm_prefix_raw = prefix
        model.llm._mm_n_prefix = prefix.size(1)

    gen_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    if getattr(model, "use_gxca", False):
        for k in ("_mm_prefix_raw", "_mm_n_prefix"):
            if hasattr(model.llm, k):
                delattr(model.llm, k)

    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1]
    text = text.strip()
    for s in ("Instruction:", "<|im_end|>", "</s>", "<|endoftext|>"):
        if s in text:
            text = text.split(s, 1)[0].strip()
    return text or "unknown"

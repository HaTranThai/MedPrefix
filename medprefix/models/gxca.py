"""Gated Cross-Attention adapter (Flamingo-style) and injection helper.

The injection wraps the ``forward`` of the LLM's last *k* decoder layers.
After the original layer produces hidden states ``h``, we split off the
prefix region, run cross-attention from text → prefix, then re-concat::

    h = [h_prefix ; h_text]
    h_text <- h_text + tanh(alpha) * CrossAttn(Q=h_text, K=P, V=P)

The ``alpha`` gate is initialized to zero so injection has no effect at
initialization (training-stable Flamingo trick).
"""
from __future__ import annotations
from typing import Optional
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttnSDPA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Scalar gate, init at 0 -> identity at start of training.
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, h_text: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        B, T, D = h_text.shape
        P = prefix.size(1)

        q = self.q_proj(self.ln_q(h_text))
        k = self.k_proj(self.ln_kv(prefix))
        v = self.v_proj(self.ln_kv(prefix))

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, P, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        gate = self.gate.to(dtype=h_text.dtype)
        return h_text + gate * out


# ---------- discovery utilities ----------

def _safe_get(obj, name):
    return getattr(obj, name, None) if obj is not None else None


def _infer_llm_hidden_size(llm) -> int:
    cfg = getattr(llm, "config", None)
    for k in ("hidden_size", "n_embd", "d_model", "dim"):
        v = getattr(cfg, k, None) if cfg is not None else None
        if isinstance(v, int) and v > 0:
            return int(v)
    return int(llm.get_input_embeddings().weight.shape[-1])


def _infer_llm_num_heads(llm, d_model: int) -> int:
    cfg = getattr(llm, "config", None)
    for k in ("num_attention_heads", "n_head", "num_heads", "n_heads"):
        v = getattr(cfg, k, None) if cfg is not None else None
        if isinstance(v, int) and v > 0 and d_model % v == 0:
            return int(v)
    for cand in (16, 12, 8, 6, 4):
        if d_model % cand == 0:
            return cand
    return 8


def _infer_train_dtype(llm) -> torch.dtype:
    dt = llm.get_input_embeddings().weight.dtype
    if dt.is_floating_point or dt.is_complex:
        return dt
    return torch.float16


def _get_decoder_layers(llm) -> nn.ModuleList:
    """Locate the ``ModuleList`` holding the decoder layers across HF/PEFT wrappers."""
    roots = []
    def _add(x):
        if x is not None and x not in roots:
            roots.append(x)

    _add(llm)
    _add(_safe_get(llm, "base_model"))
    _add(_safe_get(llm, "model"))
    _add(_safe_get(_safe_get(llm, "base_model"), "model"))
    _add(_safe_get(_safe_get(_safe_get(llm, "base_model"), "model"), "model"))

    cand_paths = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("layers",),
        ("decoder", "layers"),
    ]
    for r in roots:
        for path in cand_paths:
            obj = r
            ok = True
            for p in path:
                obj = _safe_get(obj, p)
                if obj is None:
                    ok = False
                    break
            if ok and isinstance(obj, (list, nn.ModuleList)) and len(obj) > 0:
                return obj

    # Last-resort: scan for the largest ModuleList that looks like layers.
    best, best_score = None, -1
    for name, module in llm.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            lname = name.lower()
            score = (2 if "layer" in lname else 0) + (1 if ("decoder" in lname or "model" in lname) else 0)
            if score > best_score:
                best_score = score
                best = module
    if best is not None:
        return best
    raise RuntimeError(
        "Cannot locate decoder layers. "
        "Inspect your model structure and update _get_decoder_layers()."
    )


def inject_gated_cross_attention(
    llm,
    last_k_layers: int = 4,
    n_heads: Optional[int] = None,
    dropout: float = 0.0,
):
    """Wrap the last ``last_k_layers`` decoder forwards with a GXCA adapter.

    Two attributes are set on the LLM at runtime by the parent model:
        ``llm._mm_prefix_raw`` : Tensor of shape (B, N_p, D)
        ``llm._mm_n_prefix``   : int N_p

    The wrapped forward reads them, applies cross-attention only on the text
    portion of the hidden states (positions >= N_p), then re-concatenates.
    """
    layers = _get_decoder_layers(llm)
    d_model = _infer_llm_hidden_size(llm)
    if n_heads is None:
        n_heads = _infer_llm_num_heads(llm, d_model)
    train_dtype = _infer_train_dtype(llm)

    k = int(last_k_layers)
    if k <= 0:
        return llm
    start = max(0, len(layers) - k)

    def _make_wrapped_forward(orig_forward, llm_ref):
        def wrapped_forward(self, *args, **kwargs):
            out = orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                h = out[0]
                rest = out[1:]
                is_tuple = True
            else:
                h = out
                rest = None
                is_tuple = False

            prefix = getattr(llm_ref, "_mm_prefix_raw", None)
            n_prefix = getattr(llm_ref, "_mm_n_prefix", None)
            if (prefix is not None) and (n_prefix is not None) and (h is not None):
                hp = int(n_prefix)
                if h.dim() == 3 and h.size(1) >= hp and prefix.size(-1) == h.size(-1):
                    if prefix.device != h.device or prefix.dtype != h.dtype:
                        prefix_local = prefix.to(device=h.device, dtype=h.dtype)
                    else:
                        prefix_local = prefix
                    h_pref = h[:, :hp, :]
                    h_text = h[:, hp:, :]
                    h_text = self.mm_gxca(h_text, prefix_local)
                    h = torch.cat([h_pref, h_text], dim=1)

            if is_tuple:
                return (h,) + rest
            return h
        return wrapped_forward

    for i in range(start, len(layers)):
        layer = layers[i]
        if getattr(layer, "_mm_gxca_injected", False):
            continue
        try:
            layer_device = next(layer.parameters()).device
        except StopIteration:
            layer_device = torch.device("cpu")

        layer.mm_gxca = GatedCrossAttnSDPA(
            d_model=d_model, n_heads=int(n_heads), dropout=float(dropout)
        ).to(device=layer_device, dtype=train_dtype)
        layer._mm_gxca_injected = True

        orig_forward = layer.forward
        layer.forward = types.MethodType(
            _make_wrapped_forward(orig_forward, llm), layer
        )
    return llm

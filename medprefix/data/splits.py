"""Dataset splits and vocabulary construction."""
from __future__ import annotations
from typing import Dict, List, Tuple
import os
import json
import numpy as np
import pandas as pd


def map_image_paths(image_dirs: List[str]) -> Dict[str, str]:
    """Build a {filename-or-stem -> absolute path} index from one or more dirs.

    Each image is registered under both its stem (``ISIC_0024306``) and its
    full filename (``ISIC_0024306.jpg``) so we can match instruction JSONs
    that store either form (HAM uses stems, ISIC uses full filenames).
    """
    mapping: Dict[str, str] = {}
    for d in image_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            name, ext = os.path.splitext(f)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                full = os.path.join(d, f)
                mapping[name] = full
                mapping[f] = full
    return mapping


def _first(d: dict, keys, default=None):
    """Return d[k] for the first key present (handles HAM/ISIC field-name diff)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def explode_instructions(records: List[dict], image_map: Dict[str, str]) -> pd.DataFrame:
    """Flatten the per-image instruction JSON into one row per (image, QA).

    Auto-detects field names so the same loader works for both schemas:

    HAM10000: ``age``, ``sex``, ``localization``, ``dx``, ``dx_type``
    ISIC 2019: ``age_approx``, ``sex``, ``anatom_site_general``, ``dx``, …

    Image lookup also tries ``image_id`` and ``image_id + .jpg`` because the
    ISIC dataset on Kaggle stores filenames with extensions.
    """
    rows = []
    for r in records:
        img_id = r.get("image_id") or r.get("image") or r.get("isic_id")
        if img_id is None:
            continue
        img_id = str(img_id)
        # Try id, id.jpg, id.jpeg, id.png in that order.
        img_path = image_map.get(img_id)
        if img_path is None:
            for ext in (".jpg", ".jpeg", ".png"):
                img_path = image_map.get(img_id + ext) or image_map.get(img_id.removesuffix(ext))
                if img_path:
                    break
        if (not img_path) or (not os.path.exists(img_path)):
            continue

        age = _first(r, ("age", "age_approx"))
        sex = _first(r, ("sex",))
        loc = _first(r, ("localization", "anatom_site_general", "anatom_site"))
        dx = _first(r, ("dx", "diagnosis", "label"))
        dx_type = _first(r, ("dx_type",), default="unknown")

        for qa in r.get("output", []):
            ins = qa.get("instruction", "").strip()
            resp = qa.get("response", "").strip()
            if not (ins and resp):
                continue
            rows.append({
                "image_path":   img_path,
                "age":          age,
                "sex":          sex,
                "localization": loc,
                "dx":           dx,
                "dx_type":      dx_type,
                "instruction":  ins,
                "response":     resp,
            })
    return pd.DataFrame(rows)


def load_instructions_dataframe(
    image_dirs: List[str],
    instructions_json: str,
) -> pd.DataFrame:
    image_map = map_image_paths(image_dirs)
    if not image_map:
        raise FileNotFoundError(
            f"No images found in any of {image_dirs}. Check the paths."
        )
    with open(instructions_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = explode_instructions(records, image_map)
    if len(df) == 0:
        raise ValueError(
            "Instruction JSON loaded but 0 (image, QA) rows produced. "
            "Check that 'image_id' values match the filenames in image_dirs."
        )
    return df


def split_df(
    df: pd.DataFrame,
    tr: float = 0.85,
    va: float = 0.075,
    te: float = 0.075,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(n * tr)
    n_va = int(n * va)
    return (
        df.iloc[idx[:n_tr]].reset_index(drop=True),
        df.iloc[idx[n_tr:n_tr + n_va]].reset_index(drop=True),
        df.iloc[idx[n_tr + n_va:]].reset_index(drop=True),
    )


def build_vocab(col: pd.Series) -> Dict[str, int]:
    """Map each distinct (string) value to an index >= 1; index 0 reserved for unknown/missing."""
    vals = sorted({str(x) for x in col.dropna().tolist()})
    return {v: i + 1 for i, v in enumerate(vals)}


def safe_id(vocab: Dict[str, int], x) -> int:
    if x is None:
        return 0
    if isinstance(x, float) and np.isnan(x):
        return 0
    return int(vocab.get(str(x), 0))

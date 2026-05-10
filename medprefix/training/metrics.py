"""Generation-quality and clinical metrics.

ROUGE-L, METEOR : sequence-level + alignment.
Token F1, Content Recall, Diagnostic Accuracy : clinical fidelity.
"""
from __future__ import annotations
from collections import Counter
from typing import List, Optional
import re
import numpy as np

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score


LOW_INFO_PATTERNS = {
    "unknown", "unclear", "not sure", "cannot determine", "can't determine",
    "insufficient information", "unable to determine", "no idea", "n a", "n/a",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to", "and",
    "or", "in", "on", "at", "for", "with", "by", "as", "from", "that", "this", "it", "its",
    "into", "than", "then", "their", "his", "her", "about", "such", "what", "which",
    "patient", "lesion",
}

DIAGNOSIS_LABELS = {
    "melanoma": ["melanoma", "malignant melanoma"],
    "nevus":    ["nevus", "melanocytic nevus", "mole", "benign mole"],
    "bkl":      ["benign keratosis", "benign keratosis-like lesion", "bkl", "seborrheic keratosis"],
    "bcc":      ["basal cell carcinoma", "bcc"],
    "akiec":    ["actinic keratosis", "intraepithelial carcinoma", "akiec", "bowen disease"],
    "df":       ["dermatofibroma", "df"],
    "vasc":     ["vascular lesion", "angioma", "vasc", "hemangioma"],
    "scc":      ["squamous cell carcinoma", "scc"],
}

_DIAG_KEYS = ["diagnosis", "diagnostic", "what condition", "what lesion", "fits best"]


def normalize_text(s) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokens(s: str) -> List[str]:
    return normalize_text(s).split()


def content_tokens(s: str) -> List[str]:
    return [t for t in simple_tokens(s) if t not in STOPWORDS]


# ---------- pair / batch helpers ----------

def _token_f1_pair(ref: str, hyp: str) -> float:
    ref_t, hyp_t = simple_tokens(ref), simple_tokens(hyp)
    if not ref_t and not hyp_t:
        return 1.0
    if not ref_t or not hyp_t:
        return 0.0
    common = Counter(ref_t) & Counter(hyp_t)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(hyp_t)
    r = n / len(ref_t)
    return 2 * p * r / (p + r)


def token_f1_score(refs: List[List[str]], hyps: List[str]) -> float:
    vals = [_token_f1_pair(r[0], h) for r, h in zip(refs, hyps)]
    return float(np.mean(vals)) if vals else 0.0


def _content_recall_pair(ref: str, hyp: str) -> float:
    ref_t, hyp_t = content_tokens(ref), content_tokens(hyp)
    if not ref_t:
        return 1.0
    common = Counter(ref_t) & Counter(hyp_t)
    return sum(common.values()) / len(ref_t)


def content_recall_score(refs: List[List[str]], hyps: List[str]) -> float:
    vals = [_content_recall_pair(r[0], h) for r, h in zip(refs, hyps)]
    return float(np.mean(vals)) if vals else 0.0


def is_low_info_answer(s: str) -> bool:
    n = normalize_text(s)
    toks = n.split()
    if n in LOW_INFO_PATTERNS:
        return True
    if not toks:
        return True
    if len(toks) == 1 and toks[0] not in {"yes", "no"}:
        return True
    if len(toks) <= 3 and any(p in n for p in ("unknown", "unclear", "not sure")):
        return True
    return False


def low_info_rate(hyps: List[str]) -> float:
    vals = [int(is_low_info_answer(h)) for h in hyps]
    return float(np.mean(vals)) if vals else 0.0


def extract_diagnosis_label(text: str) -> Optional[str]:
    s = normalize_text(text)
    for label, aliases in DIAGNOSIS_LABELS.items():
        for alias in aliases:
            if normalize_text(alias) in s:
                return label
    return None


def is_diagnosis_question(instruction: str) -> bool:
    s = normalize_text(instruction)
    return any(k in s for k in _DIAG_KEYS)


def diagnosis_accuracy(
    instrs: List[str],
    refs: List[List[str]],
    hyps: List[str],
) -> Optional[float]:
    vals = []
    for ins, ref, hyp in zip(instrs, refs, hyps):
        if not is_diagnosis_question(ins):
            continue
        gt = extract_diagnosis_label(ref[0])
        pr = extract_diagnosis_label(hyp)
        if gt is None:
            continue
        vals.append(int(gt == pr))
    return float(np.mean(vals)) if vals else None


def rouge_l_score(refs: List[List[str]], hyps: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(refs[i][0], hyps[i])["rougeL"].fmeasure for i in range(len(hyps))]
    return float(np.mean(vals)) if vals else 0.0


def meteor_avg(refs: List[List[str]], hyps: List[str]) -> float:
    vals = [
        meteor_score([simple_tokens(refs[i][0])], simple_tokens(hyps[i]))
        for i in range(len(hyps))
    ]
    return float(np.mean(vals)) if vals else 0.0


def compute_all_metrics(
    instrs: List[str],
    refs: List[List[str]],
    hyps: List[str],
) -> dict:
    diag = diagnosis_accuracy(instrs, refs, hyps)
    return {
        "ROUGE_L":            rouge_l_score(refs, hyps),
        "METEOR":             meteor_avg(refs, hyps),
        "Token_F1":           token_f1_score(refs, hyps),
        "Content_Recall":     content_recall_score(refs, hyps),
        "Diagnosis_Accuracy": diag if diag is not None else 0.0,
    }

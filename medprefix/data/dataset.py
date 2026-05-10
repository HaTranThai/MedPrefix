"""PyTorch Dataset and collate function for Med-Prefix."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from .io_utils import build_io_strings, to_model_inputs
from .splits import safe_id


class MedDermDataset(Dataset):
    """Multi-modal dermatology dataset.

    Each row of ``df`` must contain: image_path, age, sex, localization,
    instruction, response.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 192,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = int(max_len)
        if transform is None:
            raise ValueError("transform is required (use ImageTransforms.train/val).")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict:
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        instruction = row["instruction"]
        answer = row["response"]
        prompt, target = build_io_strings(instruction, answer)
        input_ids, labels = to_model_inputs(self.tok, prompt, target, self.max_len)

        return {
            "image":           img,
            "age":             row["age"],
            "sex":             str(row["sex"]),
            "loc":             str(row["localization"]),
            "input_ids":       input_ids,
            "labels":          labels,
            "instruction_txt": instruction,
            "response_txt":    answer,
        }


def make_collate_fn(sex_vocab: Dict[str, int], loc_vocab: Dict[str, int], pad_id: int):
    """Build a collate function that closes over the vocabularies + pad token id."""

    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch], dim=0)

        ids = [b["input_ids"] for b in batch]
        labs = [b["labels"] for b in batch]
        input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labs, batch_first=True, padding_value=-100)
        attn_mask = (input_ids != pad_id).long()

        age_list, sex_ids, loc_ids = [], [], []
        for b in batch:
            a = b["age"]
            if a is None or (isinstance(a, float) and np.isnan(a)):
                a = 0.0
            age_list.append(float(a))
            sex_ids.append(safe_id(sex_vocab, b["sex"]))
            loc_ids.append(safe_id(loc_vocab, b["loc"]))

        tab_batch = (
            torch.tensor(age_list, dtype=torch.float32),
            torch.tensor(sex_ids,   dtype=torch.long),
            torch.tensor(loc_ids,   dtype=torch.long),
        )

        instrs = [b["instruction_txt"] for b in batch]
        gts = [b["response_txt"] for b in batch]
        return images, tab_batch, input_ids, labels, attn_mask, instrs, gts

    return collate_fn


class ImageTransforms:
    """Standard 224x224 normalization. Augmentations only on train."""

    @staticmethod
    def train(img_size: int = 224):
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def val(img_size: int = 224):
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

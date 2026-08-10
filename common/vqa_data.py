"""VQAv2 loading for the LVLM reference build (LLaVA + encoder from checkpoints/vlm_weights)."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vlm.data import VQAJsonlDataset, collate_vqa_batch
from vlm.models import load_vqa_model_by_id


def resolve_vqa_device(model: nn.Module) -> torch.device:
    device_str = getattr(model, "_device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def load_vqa_model(model_id: str) -> nn.Module:
    return load_vqa_model_by_id(model_id)


def make_vqa_dataloader(
    *,
    data_jsonl: str,
    image_root: str,
    start_idx: int,
    end_idx: Optional[int],
    subset_file: Optional[str],
    batch_size: int,
    adv: bool = False,
) -> Tuple[DataLoader, VQAJsonlDataset]:
    ds = VQAJsonlDataset(
        data_jsonl,
        image_root=image_root,
        start_idx=start_idx,
        end_idx=end_idx,
        subset_file=subset_file,
        adv=adv,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_vqa_batch)
    return dl, ds


def make_vqa_dataloader_from_rows(
    rows: list,
    *,
    image_root: str,
    batch_size: int,
    adv: bool = False,
) -> Tuple[DataLoader, VQAJsonlDataset]:
    ds = VQAJsonlDataset(rows=rows, image_root=image_root, adv=adv)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_vqa_batch)
    return dl, ds


def iter_vqa_batches(
    dl: DataLoader,
    device: torch.device,
) -> Iterator[dict]:
    for batch in dl:
        yield {
            "id": batch["id"],
            "image": batch["image"].to(device),
            "question": batch["question"],
            "answer": batch["answer"],
            "answers": batch["answers"],
            "image_path": batch["image_path"],
        }

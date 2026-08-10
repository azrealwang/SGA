from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.utils.image import pad_images_to_batch_tensor


def safe_vqa_id_for_filename(qid: str) -> str:
    """Safe fragment for filenames (matches attack_pgd adv_images naming)."""
    return str(qid).replace("/", "_").replace("\\", "_").replace(":", "_")


def _resolve_image_path(image_field: str, image_root: str | Path | None) -> Path:
    p = Path(image_field)
    if p.is_file():
        return p
    if image_root:
        return Path(image_root) / image_field
    return p


def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    t = TF.pil_to_tensor(img.convert("RGB")).float() / 255.0
    return t


class VQAJsonlDataset(Dataset):
    """
    Generic VQA jsonl:
    {
      "id": "...",
      "image": "path.jpg",
      "question": "...",
      "answer": "canonical",
      "answers": ["a1", ...]
    }

    Rows are taken in file order. After optional ``subset_file`` filtering, keep accepted rows
    with index ``k`` such that ``start_idx <= k < end_idx`` when ``end_idx`` is set; if
    ``end_idx`` is None, keep all rows from ``start_idx`` onward. (``k`` = 0-based index among
    accepted rows; half-open range matches Python slicing.)
    """

    def __init__(
        self,
        data_jsonl: str | Path | None = None,
        image_root: str | Path | None = None,
        start_idx: int = 0,
        end_idx: int | None = None,
        subset_file: str | Path | None = None,
        *,
        rows: list[dict[str, Any]] | None = None,
        adv: bool = False,
    ):
        self.image_root = Path(image_root) if image_root else None
        self.adv = adv
        if rows is not None:
            if data_jsonl is not None or subset_file is not None:
                raise ValueError("Pass either rows= or data_jsonl (+ optional subset_file), not both.")
            self.samples = list(rows)
            return

        if data_jsonl is None:
            raise ValueError("data_jsonl is required when rows= is not set.")

        self.samples: list[dict[str, Any]] = []
        subset_ids: set[str] | None = None
        if subset_file:
            subset_ids = set()
            with open(subset_file, "r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    subset_ids.add(line.split()[0] if not line.startswith("{") else json.loads(line).get("id", line))

        accepted_idx = 0
        with open(data_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row["id"])
                if subset_ids is not None and sid not in subset_ids:
                    continue
                if accepted_idx < start_idx:
                    accepted_idx += 1
                    continue
                if end_idx is not None and accepted_idx >= end_idx:
                    break
                self.samples.append(row)
                accepted_idx += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        sid = str(row["id"])
        if self.adv:
            if self.image_root is None:
                raise ValueError("adv=True requires image_root (directory of per-question adv rasters).")
            ext = Path(row["image"]).suffix or ".png"
            img_path = self.image_root / f"{safe_vqa_id_for_filename(sid)}{ext}"
        else:
            img_path = _resolve_image_path(row["image"], self.image_root)
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image: {img_path}")
        image = Image.open(img_path).convert("RGB")
        tensor = pil_to_tensor01(image)
        answers = row.get("answers")
        if answers is None:
            answers = [row["answer"]]
        return {
            "id": sid,
            "image": tensor,
            "image_path": str(img_path.resolve()),
            "question": row["question"],
            "answer": row["answer"],
            "answers": list(answers),
        }


def collate_vqa_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    images = pad_images_to_batch_tensor([x["image"] for x in items])
    return {
        "id": [x["id"] for x in items],
        "image": images,
        "image_path": [x["image_path"] for x in items],
        "question": [x["question"] for x in items],
        "answer": [x["answer"] for x in items],
        "answers": [x["answers"] for x in items],
    }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

"""
Shared CLI and probe loading for the model pools.

``--task imagenet|cifar10``: classification cross-entropy + ``--input``.
``--task vqa``: LVLM teacher-forced NLL + ``--data-jsonl`` + ``--input``.

VQA probe pairing (``--conditioning``):
  ground_truth: test-set probes, the jsonl ``image`` resolved under ``--input``,
                 conditioned on the annotated answer.
  pseudo:       public probes, sorted images under ``--input`` index-aligned with
                 the jsonl rows (the jsonl ``image`` is ignored), conditioned on
                 each model's greedy-decoded answer for (image, question).

``start_idx`` / ``end_idx`` slice the jsonl question rows (after ``--subset-file``).
The range is half-open, so the probe count is ``end_idx - start_idx``. With
``pseudo`` the folder must hold that many images at those indices.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from common.utils import _sample_filepath, load_one_model
from vlm.data import VQAJsonlDataset, _resolve_image_path
from vlm.models import load_vqa_model_by_id

RANKING_TASKS = frozenset({"imagenet", "cifar10", "vqa"})

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def is_vqa_task(task: str) -> bool:
    return task == "vqa"


def classifier_backend_dataset(task: str) -> str:
    if task == "cifar10":
        return "cifar10"
    return "imagenet"


def add_ranking_task_args(
    p: argparse.ArgumentParser,
    *,
    default_task: str = "imagenet",
    default_conditioning: str = "pseudo",
) -> None:
    p.add_argument(
        "--task",
        type=str,
        default=default_task,
        choices=sorted(RANKING_TASKS),
        help="imagenet|cifar10: classification CE. vqa: LLaVA NLL.",
    )
    p.add_argument(
        "--conditioning",
        type=str,
        default=default_conditioning,
        choices=["pseudo", "ground_truth"],
        help=(
            "Conditioning output for the input gradient. ground_truth uses the "
            "labelled target of a test-set probe; pseudo uses the model's own "
            "prediction (argmax for classification, greedy-decoded answer for VQA), "
            "which is what public probes require."
        ),
    )
    p.add_argument(
        "--data-jsonl",
        type=str,
        default=None,
        help="VQAv2 jsonl (required for --task vqa).",
    )
    p.add_argument("--subset-file", type=str, default=None)
    add_grad_batch_size_arg(p)


def add_grad_batch_size_arg(
    p: argparse.ArgumentParser,
    *,
    default: Optional[int] = None,
) -> None:
    p.add_argument(
        "--grad_batch_size",
        type=int,
        default=default,
        help=(
            "GPU input chunk per model. Default: all loaded images (classification) "
            "or 1 (vqa). CLEVER: defaults to --nsamp when unset."
        ),
    )


def resolve_grad_batch_size(args: argparse.Namespace, n_samples: int) -> int:
    """Number of probes pushed through a model per backward pass."""
    if getattr(args, "grad_batch_size", None) is not None:
        bs = int(args.grad_batch_size)
    elif is_vqa_task(args.task):
        bs = 1
    else:
        bs = n_samples
    if bs <= 0:
        raise ValueError("--grad_batch_size must be positive")
    return bs


def validate_ranking_args(args: argparse.Namespace) -> None:
    if args.task == "vqa":
        if not args.data_jsonl:
            raise ValueError("--task vqa requires --data-jsonl.")
        if not args.input:
            raise ValueError("--task vqa requires --input (image directory).")
    elif not args.input:
        raise ValueError(f"--task {args.task} requires --input.")


def normalize_ranking_args(args: argparse.Namespace) -> None:
    validate_ranking_args(args)


@dataclass
class RankingSample:
    image: np.ndarray  # float32 CHW [0,1]
    class_id: Optional[int] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    answers_multi: Optional[List[str]] = None


def _load_image_chw01(path: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    with Image.open(path) as pil_img:
        if shape is not None:
            pil_img = pil_img.resize(shape)
        image = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
    chw = np.transpose(image, (2, 0, 1))
    return np.ascontiguousarray(chw / 255.0)


def _list_folder_images(path: str) -> List[str]:
    root = os.path.join(path, "")
    files: List[str] = []
    for name in os.listdir(root):
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMAGE_EXTS:
            files.append(os.path.join(root, name))
    if not files:
        raise FileNotFoundError(f"No images under {path}")
    return sorted(files, key=lambda p: os.path.basename(p).lower())


def _label_from_basename(basename: str, *, require_label: bool) -> Optional[int]:
    stem = os.path.splitext(basename)[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    if require_label:
        raise ValueError(f"Expected label suffix in filename, got {basename!r}")
    return None


def load_folder_ranking_samples(
    path: str,
    start_idx: int,
    end_idx: int,
    *,
    require_label: bool,
    spatial: Optional[Tuple[int, int]] = None,
) -> List[RankingSample]:
    samplepath = os.path.join(path, "")
    if end_idx is None:
        matches = glob.glob(os.path.join(samplepath, "[0-9][0-9][0-9][0-9][0-9]*"))
        if not matches:
            raise FileNotFoundError(f"No indexed samples under {path}")
        end_idx = max(int(os.path.basename(m)[:5]) for m in matches) + 1
    if end_idx <= start_idx:
        raise ValueError(f"end_idx ({end_idx}) must be > start_idx ({start_idx})")

    out: List[RankingSample] = []
    for i in range(start_idx, end_idx):
        filepath = _sample_filepath(samplepath, i)
        label = _label_from_basename(os.path.basename(filepath), require_label=require_label)
        img = _load_image_chw01(filepath, spatial)
        out.append(RankingSample(image=img, class_id=label))
    return out


def _rows_from_jsonl(
    data_jsonl: str,
    start_idx: int,
    end_idx: Optional[int],
    subset_file: Optional[str],
) -> List[dict]:
    ds = VQAJsonlDataset(
        data_jsonl,
        start_idx=start_idx,
        end_idx=end_idx,
        subset_file=subset_file,
    )
    return ds.samples


def load_vqa_gt_ranking_samples(
    data_jsonl: str,
    image_root: str,
    start_idx: int,
    end_idx: Optional[int],
    subset_file: Optional[str],
) -> List[RankingSample]:
    """VQA test-set probes: jsonl image + question + annotated answer, under ``image_root``."""
    rows = _rows_from_jsonl(data_jsonl, start_idx, end_idx, subset_file)
    out: List[RankingSample] = []
    for row in rows:
        p = _resolve_image_path(row["image"], image_root)
        img = _load_image_chw01(str(p))
        answers = row.get("answers") or [row["answer"]]
        out.append(
            RankingSample(
                image=img,
                question=row["question"],
                answer=row["answer"],
                answers_multi=list(answers),
            )
        )
    return out


def load_vqa_pred_ranking_samples(
    image_root: str,
    data_jsonl: str,
    start_idx: int,
    end_idx: Optional[int],
    subset_file: Optional[str],
) -> List[RankingSample]:
    """
    VQA pred: folder image[i] + jsonl question row[i]; json ``image`` field ignored.

    ``start_idx``/``end_idx`` slice jsonl rows; the same number of sorted folder images
    at indices ``[start_idx, start_idx + n)`` is required.
    """
    rows = _rows_from_jsonl(data_jsonl, start_idx, end_idx, subset_file)
    if not rows:
        raise ValueError("No jsonl rows in the selected index range.")
    files = _list_folder_images(image_root)
    n = len(rows)
    if start_idx + n > len(files):
        raise ValueError(
            f"conditioning=pseudo needs {n} images at indices [{start_idx}, {start_idx + n}) "
            f"under {image_root}, found {len(files)} images total."
        )

    out: List[RankingSample] = []
    for offset, row in enumerate(rows):
        img = _load_image_chw01(files[start_idx + offset])
        answers = row.get("answers") or [row["answer"]]
        out.append(
            RankingSample(
                image=img,
                question=row["question"],
                answer=row["answer"],
                answers_multi=list(answers),
            )
        )
    return out


def load_vqa_ranking_samples(
    data_jsonl: str,
    image_root: str,
    start_idx: int,
    end_idx: Optional[int],
    subset_file: Optional[str],
    *,
    conditioning: str,
) -> List[RankingSample]:
    if conditioning == "ground_truth":
        return load_vqa_gt_ranking_samples(
            data_jsonl, image_root, start_idx, end_idx, subset_file
        )
    if conditioning == "pseudo":
        return load_vqa_pred_ranking_samples(
            image_root, data_jsonl, start_idx, end_idx, subset_file
        )
    raise ValueError(f"conditioning must be 'ground_truth' or 'pseudo', got {conditioning!r}")


def _pad_images_chw01_to_batch(images: Sequence[np.ndarray], fill: float = 0.0) -> np.ndarray:
    """Center-pad variable-size CHW images to a common batch tensor (matches vlm collate)."""
    if not images:
        raise ValueError("empty image list")
    if len(images) == 1:
        return np.ascontiguousarray(images[0][np.newaxis, ...])
    max_h = max(int(img.shape[1]) for img in images)
    max_w = max(int(img.shape[2]) for img in images)
    if all(int(img.shape[1]) == max_h and int(img.shape[2]) == max_w for img in images):
        return np.stack(images, axis=0)
    padded: List[np.ndarray] = []
    for img in images:
        _, h, w = img.shape
        if int(h) == max_h and int(w) == max_w:
            padded.append(img)
            continue
        canvas = np.full((3, max_h, max_w), fill, dtype=img.dtype)
        top = (max_h - int(h)) // 2
        left = (max_w - int(w)) // 2
        canvas[:, top : top + int(h), left : left + int(w)] = img
        padded.append(np.ascontiguousarray(canvas))
    return np.stack(padded, axis=0)


def load_ranking_model(
    task: str,
    model_id: str,
    device: torch.device,
    *,
    threat_model: str = "Linf",
) -> torch.nn.Module:
    if is_vqa_task(task):
        return load_vqa_model_by_id(model_id).to(device).eval()
    backend = classifier_backend_dataset(task)
    return load_one_model(backend, model_id, threat_model=threat_model).to(device).eval()



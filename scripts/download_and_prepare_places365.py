#!/usr/bin/env python3
"""
Download Places365 val, shuffle, and export a public-probe image folder.

Same file naming as the ImageNet / CelebA probes: ``00000_<class>.jpg``, ``00001_<class>.jpg``, ...
Use with ranking scripts via ``--task imagenet --conditioning pseudo --input data/places365``
(or ``--task vqa --conditioning pseudo --input data/places365`` for LVLM).

Run on the server (do not download on a laptop):

  python scripts/download_and_prepare_places365.py
  python scripts/download_and_prepare_places365.py --out-dir data/places365 --num-images 1000 --seed 0
  python scripts/download_and_prepare_places365.py --large   # full-res val (much larger download)

Requires: torch, torchvision, Pillow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Places365 val and write indexed public-probe images.",
    )
    root = Path(__file__).resolve().parents[1]
    p.add_argument(
        "--data-root",
        type=str,
        default=str(root / "data"),
        help="Root for torchvision Places365 cache (default: <repo>/data).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "data" / "places365"),
        help="Output folder for indexed images (default: <repo>/data/places365).",
    )
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    p.add_argument(
        "--large",
        action="store_true",
        help="Download full-resolution Places365 (default: small 256px).",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing torchvision Places365 cache under --data-root.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise SystemExit("--num-images must be > 0")

    try:
        from torchvision.datasets import Places365
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required. Install torch + torchvision on the server."
        ) from exc

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Places365 val | small={not args.large} download={not args.skip_download} "
        f"cache={data_root} -> {out_dir} (n={args.num_images}, seed={args.seed})",
        flush=True,
    )
    ds = Places365(
        root=str(data_root),
        split="val",
        small=not args.large,
        download=not args.skip_download,
    )
    n_total = len(ds)
    if args.num_images > n_total:
        raise SystemExit(
            f"--num-images={args.num_images} exceeds Places365 val size {n_total}"
        )

    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(n_total)[: args.num_images]

    # Clear previous indexed outputs so a re-run cannot mix old files.
    for old in out_dir.glob("[0-9][0-9][0-9][0-9][0-9]_*"):
        old.unlink()

    for out_i, src_i in enumerate(chosen):
        path, class_id = ds.imgs[int(src_i)]
        class_id = int(class_id)
        src = Path(path)
        # Keep a common raster format; convert if needed.
        dst = out_dir / f"{out_i:05d}_{class_id}.jpg"
        if src.suffix.lower() in {".jpg", ".jpeg"} and src.is_file():
            shutil.copy2(src, dst)
        else:
            with Image.open(src) as im:
                im.convert("RGB").save(dst, quality=95)
        if (out_i + 1) % 100 == 0 or out_i + 1 == args.num_images:
            print(f"  wrote {out_i + 1}/{args.num_images}", flush=True)

    print(f"Done. {args.num_images} images in {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

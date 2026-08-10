from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_makedirs(path: str | Path, overwrite: bool = False) -> Path:
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {p}. Pass overwrite=True or use a new path."
        )
    if overwrite and p.exists():
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p

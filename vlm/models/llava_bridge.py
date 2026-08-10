"""Resolve LLaVA / Robust-LLaVA on the import path (not vendored in this repo)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def ensure_llava_on_path() -> Path:
    """
    Ensure `llava` can be imported.

    Resolution order:
    1. Environment variable ``LLAVA_ROOT`` or ``LLAVA_HOME`` (directory that *contains* ``llava/``).
    2. An already-discoverable ``llava`` (e.g. ``pip install -e /path/to/LLaVA``); prepends the parent of
       the ``llava`` package directory to ``sys.path`` when needed.
    """
    for key in ("LLAVA_ROOT", "LLAVA_HOME"):
        v = os.environ.get(key)
        if not v:
            continue
        root = Path(v).expanduser().resolve()
        conv = root / "llava" / "conversation.py"
        if conv.is_file():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            return root

    spec = importlib.util.find_spec("llava")
    if spec is None:
        raise ImportError(
            "The `llava` package is not available. This repository no longer vendors LLaVA. "
            "Install Robust-LLaVA or LLaVA (e.g. `pip install -e /path/to/LLaVA`) or set `LLAVA_ROOT` "
            "to the checkout directory that contains the `llava/` package."
        )

    locs = list(getattr(spec, "submodule_search_locations", None) or [])
    origin = getattr(spec, "origin", None)
    if origin and origin != "namespace":
        pkg = Path(origin).resolve().parent
    elif locs:
        pkg = Path(locs[0]).resolve()
    else:
        raise ImportError(
            "Could not resolve `llava` install location (no origin / search locations). "
            "Set LLAVA_ROOT to a LLaVA/Robust-LLaVA checkout."
        )

    if not (pkg / "conversation.py").is_file():
        raise ImportError(
            f"Resolved llava at {pkg} but conversation.py is missing; use a standard LLaVA/Robust-LLaVA tree."
        )
    top = pkg.parent
    if str(top) not in sys.path:
        sys.path.insert(0, str(top))
    return top

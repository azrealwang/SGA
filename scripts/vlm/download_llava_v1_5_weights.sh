#!/usr/bin/env bash
# Download LLaVA 1.5 7B merged weights (LlavaLlama format for the official LLaVA Python package)
# -> ./checkpoints/llava-v1.5-7b/
#
# Default repo is liuhaotian/llava-v1.5-7b (public, matches load_pretrained_model in LLaVA).
# The older liuhaotian/llava-v1.5-7b-hf repo often returns 401 for anonymous API access; do not use it unless you fix auth.
#
# If you see "401" / "Invalid username or password" while downloading a public model, you likely have a stale
# HUGGINGFACE_HUB_TOKEN in the environment. This script uses anonymous Hub access unless HF_TOKEN is set.
#
# Prereq: pip install huggingface_hub
# Usage:
#   bash scripts/vlm/download_llava_v1_5_weights.sh
#
# Optional env:
#   HF_REPO_ID=liuhaotian/llava-v1.5-7b   # default; llava-hf/llava-1.5-7b-hf needs Transformers-style loading, not this repo's LLaVA wrapper
#   HF_TOKEN=...                          # only if you use a private mirror or gated fork (overrides anonymous download)
#   HF_ENDPOINT=<url>                     # alternative Hub endpoint (export before running)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/checkpoints/llava-v1.5-7b"
REPO_ID="${HF_REPO_ID:-liuhaotian/llava-v1.5-7b}"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "Install huggingface_hub first:  pip install huggingface_hub"
  exit 1
fi

mkdir -p "${ROOT}/checkpoints"
echo "Downloading ${REPO_ID} -> ${OUT}"
export OUT
export REPO_ID
export HF_TOKEN="${HF_TOKEN:-}"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

out = os.environ["OUT"]
repo_id = os.environ["REPO_ID"]
token = os.environ.get("HF_TOKEN") or None
# Public Hub download unless HF_TOKEN is set (ignores stale HUGGINGFACE_HUB_TOKEN for public repos).
kwargs = {"token": token} if token else {"token": False}
snapshot_download(repo_id=repo_id, local_dir=out, **kwargs)
PY

echo "Done. LLaVA weights at: ${OUT}/"

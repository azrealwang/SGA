#!/usr/bin/env bash
# Download Madry Lab L2-robust ImageNet checkpoints used by reference/dnn_l2.xml
# (3 arch x 4 eps = 12). Source: https://huggingface.co/madrylab/robust-imagenet-models
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/checkpoints/madry_imagenet_l2"
BASE="https://huggingface.co/madrylab/robust-imagenet-models/resolve/main"

mkdir -p "$OUT"

# Matches Salman2020Do_{R18,R50,50_2}_L2_eps* in dnn_l2.xml (no wide_resnet50_4).
ARCHES=(resnet18 resnet50 wide_resnet50_2)
EPS_TAGS=(0.1 0.25 0.5 1)

for arch in "${ARCHES[@]}"; do
  for eps in "${EPS_TAGS[@]}"; do
    fname="${arch}_l2_eps${eps}.ckpt"
    dest="${OUT}/${fname}"
    if [[ -f "$dest" ]]; then
      echo "skip ${fname}"
      continue
    fi
    echo "download ${fname}"
    curl -fL --retry 3 --retry-delay 2 -o "$dest" "${BASE}/${fname}"
  done
done

echo "Done. weights in ${OUT}"

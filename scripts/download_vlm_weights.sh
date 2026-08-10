#!/usr/bin/env bash
# Download robust vision encoder weights into ./checkpoints/vlm_weights/.
# LEAF HF config (tracked): ./scripts/leaf_hf_config.json
#
# Usage:
#   bash scripts/download_vlm_weights.sh
#
# Optional:
#   DRY_RUN=1 bash scripts/download_vlm_weights.sh
#   VIT_L_ONLY=1 bash scripts/download_vlm_weights.sh   # skip ConvNeXt / ViT-B
#   WEIGHT_SOURCE=tuebingen bash scripts/download_vlm_weights.sh  # original author mirrors
#   HF_ENDPOINT=<url> bash scripts/download_vlm_weights.sh        # alternative HF endpoint

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ROOT}/checkpoints/vlm_weights"
LEAF_CONFIG="${SCRIPT_DIR}/leaf_hf_config.json"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-hf}"
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"

download() {
  local url="$1" out="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would fetch: $url -> $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"

  if [[ -s "$out" ]]; then
    echo "Already exists: $out (Skipping)"
    return 0
  fi

  echo "Downloading -> $out"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --continue-at - -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" -c "$url"
  else
    echo "Error: Need curl or wget to download."
    exit 1
  fi
}

mkdir -p "${DEST}"

if [[ "${WEIGHT_SOURCE}" == "hf" ]]; then
  echo "WEIGHT_SOURCE=hf (Hugging Face CDN via ${HF_BASE})"
  FARE_L14_URL="${HF_BASE}/chs20/fare4-clip/resolve/main/open_clip_pytorch_model.bin"
  TECOA_L14_URL="${HF_BASE}/chs20/tecoa4-clip/resolve/main/open_clip_pytorch_model.bin"
  SIMCLIP4_URL="${HF_BASE}/hossainzarif19/SimCLIP/resolve/main/simclip4.pt"
  CONVNEXT_TECOA_URL="${HF_BASE}/chs20/TeCoA4-convnext_base_w-laion2B-s13B-b82K-augreg/resolve/main/open_clip_pytorch_model.bin"
  CONVNEXT_FARE_URL="${HF_BASE}/chs20/FARE4-convnext_base_w-laion2B-s13B-b82K-augreg/resolve/main/open_clip_pytorch_model.bin"
  VIT_B_TECOA_URL="${HF_BASE}/chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin"
  VIT_B_FARE_URL="${HF_BASE}/chs20/FARE4-ViT-B-16-laion2B-s34B-b88K/resolve/main/open_clip_pytorch_model.bin"
else
  echo "WEIGHT_SOURCE=tuebingen (mirrors published with the original releases)"
  FARE_L14_URL="https://nc.mlcloud.uni-tuebingen.de/index.php/s/jnQ2qmp9tst8kyQ/download/fare_eps_4.pt"
  TECOA_L14_URL="https://nc.mlcloud.uni-tuebingen.de/index.php/s/92req4Pak5i56tX/download/tecoa_eps_4.pt"
  SIMCLIP4_URL="${HF_BASE}/hossainzarif19/SimCLIP/resolve/main/simclip4.pt"
  CONVNEXT_TECOA_URL="https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/zHKCC9aS7rf4qCt/?accept=zip"
  CONVNEXT_FARE_URL="https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/GCPXBDEE5PoCngy/?accept=zip"
  VIT_B_TECOA_URL="https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/ZBkmbMrAwgfeeSa/?accept=zip"
  VIT_B_FARE_URL="https://nc.mlcloud.uni-tuebingen.de/public.php/dav/files/fg7JHQzASiNnxCg/?accept=zip"
fi

echo "Saving core encoders under: ${DEST}/"
download "${FARE_L14_URL}" "${DEST}/fare_eps_4.pt"
download "${TECOA_L14_URL}" "${DEST}/tecoa_eps_4.pt"
download "${SIMCLIP4_URL}" "${DEST}/simclip4.pt"

LEAF_PT_OUT="${DEST}/leaf_vit_l.pt"

if [[ ! -s "${LEAF_PT_OUT}" ]]; then
  LEAF_HF_DIR="${DEST}/hf_leaf_temp"
  mkdir -p "${LEAF_HF_DIR}"

  LEAF_SAFETENSORS_URL="${HF_BASE}/LEAF-CLIP/CLIP-ViT-L-rho50-k1-constrained-FARE2/resolve/main/model.safetensors?download=true"

  download "${LEAF_SAFETENSORS_URL}" "${LEAF_HF_DIR}/model.safetensors"

  if [[ ! -f "${LEAF_CONFIG}" ]]; then
    echo "ERROR: Missing LEAF HF config: ${LEAF_CONFIG}"
    exit 1
  fi
  cp "${LEAF_CONFIG}" "${LEAF_HF_DIR}/config.json"

  python "${SCRIPT_DIR}/convert_to_openclip.py" \
    --hf_checkpoint "${LEAF_HF_DIR}" \
    --openclip_model "ViT-L-14-quickgelu" \
    --output "${LEAF_PT_OUT}"
fi

if [[ "${VIT_L_ONLY:-0}" == "1" ]]; then
  echo "VIT_L_ONLY=1: skipping ConvNeXt / ViT-B weights"
else
  download "${CONVNEXT_TECOA_URL}" "${DEST}/convnext_base_w-tecoa-eps4.pt"
  download "${CONVNEXT_FARE_URL}" "${DEST}/convnext_base_w-fare-eps4.pt"
  download "${VIT_B_TECOA_URL}" "${DEST}/vit-b-16-tecoa-eps4.pt"
  download "${VIT_B_FARE_URL}" "${DEST}/vit-b-16-fare-eps4.pt"
fi

echo ""
echo "All downloads complete. Weights are stored in: ${DEST}/"

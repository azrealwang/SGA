#!/usr/bin/env bash
# Download VQAv2 val (questions + annotations) and COCO val2014 images, then build jsonl for clean eval / attacks.
#
# Usage (from repo root):
#   bash scripts/vlm/download_and_prepare_vqav2.sh
#   bash scripts/vlm/download_and_prepare_vqav2.sh --data-root /path/to/data
#   bash scripts/vlm/download_and_prepare_vqav2.sh --max-samples 5000
#   bash scripts/vlm/download_and_prepare_vqav2.sh --skip-coco      # only VQA zips + jsonl (images already there)
#   bash scripts/vlm/download_and_prepare_vqav2.sh --skip-download # only run convert (zips already extracted)
#
# Disk: COCO val2014 ~6GB + VQA zips small. Requires: curl or wget, unzip, python3.
#
# Official sources:
#   COCO: https://cocodataset.org/#download
#   VQAv2: https://visualqa.org/download.html

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${ROOT}/data"
MAX_SAMPLES=""
SKIP_COCO=0
SKIP_DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --skip-coco) SKIP_COCO=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    -h|--help)
      echo "Usage: bash scripts/vlm/download_and_prepare_vqav2.sh [--data-root DIR] [--max-samples N] [--skip-coco] [--skip-download]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

DL="${DATA_ROOT}/_downloads"
VQA_RAW="${DATA_ROOT}/vqa/raw"
COCO_IMG="${DATA_ROOT}/coco/val2014"
OUT_JSONL="${DATA_ROOT}/vqa/vqav2_val.jsonl"

mkdir -p "${DL}" "${VQA_RAW}" "${COCO_IMG}" "$(dirname "${OUT_JSONL}")"

download() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "Exists (skip): $out"
    return 0
  fi
  echo "Downloading -> $out"
  if command -v curl >/dev/null 2>&1; then
    curl -fSL --retry 3 --retry-delay 2 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Need curl or wget"; exit 1
  fi
}

# VQAv2 val: AWS mirror, same files as the visualqa.org bundles
VQA_Q_ZIP_URL="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
VQA_A_ZIP_URL="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
COCO_VAL_ZIP_URL="http://images.cocodataset.org/zips/val2014.zip"

Q_ZIP="${DL}/v2_Questions_Val_mscoco.zip"
A_ZIP="${DL}/v2_Annotations_Val_mscoco.zip"
COCO_ZIP="${DL}/val2014.zip"

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  download "${VQA_Q_ZIP_URL}" "${Q_ZIP}"
  download "${VQA_A_ZIP_URL}" "${A_ZIP}"
  if [[ "${SKIP_COCO}" -eq 0 ]]; then
    download "${COCO_VAL_ZIP_URL}" "${COCO_ZIP}"
  else
    echo "Skipping COCO download (--skip-coco). Ensure images exist under: ${COCO_IMG}"
  fi

  echo "Unzipping VQA..."
  unzip -q -o "${Q_ZIP}" -d "${VQA_RAW}"
  unzip -q -o "${A_ZIP}" -d "${VQA_RAW}"

  if [[ "${SKIP_COCO}" -eq 0 ]]; then
    echo "Unzipping COCO val2014 (may take a few minutes)..."
    unzip -q -o "${COCO_ZIP}" -d "${DATA_ROOT}/coco"
  fi
else
  echo "Skipping downloads (--skip-download)."
fi

# Locate extracted JSON (zip layout can vary)
QUEST_JSON="$(find "${VQA_RAW}" -name 'v2_OpenEnded_mscoco_val2014_questions.json' -type f | head -1)"
ANN_JSON="$(find "${VQA_RAW}" -name 'v2_mscoco_val2014_annotations.json' -type f | head -1)"

if [[ -z "${QUEST_JSON}" || ! -f "${QUEST_JSON}" ]]; then
  echo "Could not find v2_OpenEnded_mscoco_val2014_questions.json under ${VQA_RAW}"
  echo "List: find ${VQA_RAW} -name '*.json'"
  exit 1
fi
if [[ -z "${ANN_JSON}" || ! -f "${ANN_JSON}" ]]; then
  echo "Could not find v2_mscoco_val2014_annotations.json under ${VQA_RAW}"
  exit 1
fi

echo "Questions: ${QUEST_JSON}"
echo "Annotations: ${ANN_JSON}"

CONV=(python3 scripts/vlm/convert_vqav2.py
  --questions-json "${QUEST_JSON}"
  --annotations-json "${ANN_JSON}"
  --output-jsonl "${OUT_JSONL}"
)
if [[ -n "${MAX_SAMPLES}" ]]; then
  CONV+=(--max-samples "${MAX_SAMPLES}")
fi

echo "Building jsonl..."
"${CONV[@]}"

echo ""
echo "Done."
echo "  jsonl:     ${OUT_JSONL}"
echo "  images:    ${COCO_IMG}/"

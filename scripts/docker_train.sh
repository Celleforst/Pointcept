#!/bin/bash
set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
DATASET=utonia
CONFIG=semseg-utonia-v1m1-0c-goose_ex-ft
EXP_NAME=goosex_run1       # output goes to exp/${DATASET}/${EXP_NAME}/
RESUME=true               # set to true to resume from last checkpoint
# Path to the downloaded Utonia pretrained checkpoint (.pth).
# Download from: https://huggingface.co/Pointcept/Utonia
WEIGHT=/workspace/weights/utonia_pretrain.pth

# ── host paths (edit these) ───────────────────────────────────────────────────
POINTCEPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # repo root
GOOSE_EX_DIR="${GOOSE_EX_DIR:-/scratch/mkrahforst/gooseEx_3d_val}"      # override with env var
WEIGHTS_DIR="${WEIGHTS_DIR:-$HOME/weights}"          # host dir holding .pth files
# ──────────────────────────────────────────────────────────────────────────────

IMAGE=pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel

RESUME_FLAG=""
if [ "$RESUME" = "true" ]; then
  RESUME_FLAG="-r true"
fi

docker run --gpus all --rm \
  -v "$POINTCEPT_DIR":/workspace/Pointcept \
  -v "$GOOSE_EX_DIR":/data \
  -v "$WEIGHTS_DIR":/workspace/weights \
  -w /workspace/Pointcept \
  "$IMAGE" \
  bash scripts/train.sh \
    -p python \
    -d "$DATASET" \
    -c "$CONFIG" \
    -n "$EXP_NAME" \
    -g 1 \
    $RESUME_FLAG \
    -w "$WEIGHT"

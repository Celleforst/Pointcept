#!/bin/bash
#SBATCH --job-name=pointcept_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --account=mkrahforst   # replace with your Euler account

# ── config ────────────────────────────────────────────────────────────────────
DATASET=utonia
CONFIG=semseg-utonia-v1m1-0c-goose_ex-ft
EXP_NAME=goosex_run1       # output goes to exp/${DATASET}/${EXP_NAME}/
RESUME=false               # set to true to resume from last checkpoint
# Path to the downloaded Utonia pretrained checkpoint (.pth).
# Download from: https://huggingface.co/Pointcept/Utonia
WEIGHT=/cluster/scratch/mkrahforst/weights/utonia_pretrain.pth
# ──────────────────────────────────────────────────────────────────────────────

module load stack/2024-06 cuda/12.1.1

# activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pointcept

mkdir -p logs

bash scripts/train.sh \
  -p python \
  -d "$DATASET" \
  -c "$CONFIG" \
  -n "$EXP_NAME" \
  -g 1 \
  -r "$RESUME" \
  -w "$WEIGHT"

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# train.sh – Launch fine-tuning (single GPU or multi-GPU)
#
# Usage:
#   # Single GPU
#   bash scripts/train.sh
#
#   # Multi-GPU (4 GPUs)
#   NUM_GPUS=4 bash scripts/train.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="${CONFIG:-configs/train_config.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
PYTHON=${PYTHON:-python3}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "=== Launching multi-GPU training (${NUM_GPUS} GPUs) ==="
    accelerate launch \
        --num_processes "$NUM_GPUS" \
        --mixed_precision fp16 \
        src/train.py --config "$CONFIG"
else
    echo "=== Launching single-GPU training ==="
    $PYTHON src/train.py --config "$CONFIG"
fi

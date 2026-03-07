#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# prepare_data.sh – Full data-preparation pipeline
#
# Edit the variables below to match your paths, then run:
#   bash scripts/prepare_data.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────
SOURCE_DIR="${SOURCE_DIR:-data/raw}"           # raw images (any structure)
DATASET_DIR="${DATASET_DIR:-data/interior}"    # processed dataset root
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
MODEL_TYPE="${MODEL_TYPE:-vit_h}"
DEVICE="${DEVICE:-cuda}"
VAL_RATIO="${VAL_RATIO:-0.1}"

PYTHON=${PYTHON:-python3}
# ─────────────────────────────────────────────────────────────

echo "=== Step 1: Split raw images into train / val ==="
$PYTHON src/prepare_data.py split \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$DATASET_DIR" \
    --val_ratio  "$VAL_RATIO"

echo ""
echo "=== Step 2: Generate captions with BLIP-2 ==="
$PYTHON src/prepare_data.py caption \
    --dataset_dir "$DATASET_DIR" \
    --device      "$DEVICE"

echo ""
echo "=== Step 3: Generate masks with SAM ==="
if [ -f "$SAM_CHECKPOINT" ]; then
    $PYTHON src/prepare_data.py mask \
        --dataset_dir    "$DATASET_DIR" \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --model_type     "$MODEL_TYPE" \
        --device         "$DEVICE"
else
    echo "SAM checkpoint not found at '$SAM_CHECKPOINT'."
    echo "Skipping mask generation; random masks will be used during training."
    echo "To download: wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
fi

echo ""
echo "Data preparation complete. Dataset ready at: $DATASET_DIR"

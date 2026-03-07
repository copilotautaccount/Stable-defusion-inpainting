#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# prepare_data.sh – Full data-preparation pipeline
#
# The dataset contains only raw interior images (no masks, no captions).
# This script runs all three annotation steps automatically.
#
# Captioner options : blip | blip2 | florence2
# Masker options    : sam | sam2 | grounded_sam | oneformer
#
# Edit the variables below to match your paths, then run:
#   bash scripts/prepare_data.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────
SOURCE_DIR="${SOURCE_DIR:-data/raw}"           # raw images (any structure)
DATASET_DIR="${DATASET_DIR:-data/interior}"    # processed dataset root
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
DEVICE="${DEVICE:-cuda}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# Caption model: blip (6 GB) | blip2 (15 GB) | florence2 (8 GB, recommended)
CAPTIONER="${CAPTIONER:-florence2}"

# Mask model: grounded_sam (recommended) | sam2 | sam | oneformer
MASKER="${MASKER:-grounded_sam}"

# Furniture labels for grounded_sam (comma-separated)
FURNITURE_LABELS="${FURNITURE_LABELS:-sofa,armchair,chair,dining chair,table,coffee table,bed,wardrobe,cabinet,lamp,floor lamp,curtain,rug,mirror}"

PYTHON=${PYTHON:-python3}
# ─────────────────────────────────────────────────────────────

echo "=== Step 1: Split raw images into train / val ==="
$PYTHON src/prepare_data.py split \
    --source_dir "$SOURCE_DIR" \
    --output_dir "$DATASET_DIR" \
    --val_ratio  "$VAL_RATIO"

echo ""
echo "=== Step 2: Generate captions with ${CAPTIONER} ==="
$PYTHON src/prepare_data.py caption \
    --dataset_dir "$DATASET_DIR" \
    --captioner   "$CAPTIONER" \
    --device      "$DEVICE"

echo ""
echo "=== Step 3: Generate masks with ${MASKER} ==="
case "$MASKER" in
  sam)
    if [ ! -f "$SAM_CHECKPOINT" ]; then
      echo "  SAM checkpoint not found at '$SAM_CHECKPOINT'."
      echo "  Downloading to checkpoints/ …"
      mkdir -p checkpoints
      wget --progress=bar -P checkpoints \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        || { echo "ERROR: SAM checkpoint download failed."; exit 1; }
    fi
    $PYTHON src/prepare_data.py mask \
        --dataset_dir    "$DATASET_DIR" \
        --masker         sam \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --device         "$DEVICE"
    ;;
  grounded_sam)
    if [ ! -f "$SAM_CHECKPOINT" ]; then
      echo "  SAM checkpoint not found at '$SAM_CHECKPOINT'. Downloading …"
      mkdir -p checkpoints
      wget --progress=bar -P checkpoints \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        || { echo "ERROR: SAM checkpoint download failed."; exit 1; }
    fi
    $PYTHON src/prepare_data.py mask \
        --dataset_dir      "$DATASET_DIR" \
        --masker           grounded_sam \
        --sam_checkpoint   "$SAM_CHECKPOINT" \
        --furniture_labels "$FURNITURE_LABELS" \
        --device           "$DEVICE"
    ;;
  sam2|oneformer)
    $PYTHON src/prepare_data.py mask \
        --dataset_dir "$DATASET_DIR" \
        --masker      "$MASKER" \
        --device      "$DEVICE"
    ;;
  *)
    echo "Unknown MASKER='$MASKER'. Set to: sam | sam2 | grounded_sam | oneformer"
    exit 1
    ;;
esac

echo ""
echo "Data preparation complete. Dataset ready at: $DATASET_DIR"

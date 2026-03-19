#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# reannotate.sh  –  Re-generate captions and masks for an existing dataset
#
# Use this script when you want to reprocess an existing train/val split
# without re-splitting raw images (e.g. after updating the captioner or
# switching to a new masking backend like SAM3).
#
# Stages run:
#   1  Caption  – regenerate captions.json for train + val
#   2  Mask     – regenerate masks/ for train + val
#
# ── Quick start ──────────────────────────────────────────────────────────
#   bash scripts/reannotate.sh
#
# ── Override settings at runtime ─────────────────────────────────────────
#   CAPTIONER=florence2 MASKER=sam3 bash scripts/reannotate.sh
#   CAPTIONER=blip2     MASKER=grounded_sam bash scripts/reannotate.sh
#   SKIP_CAPTION=1 bash scripts/reannotate.sh   # only redo masks
#   SKIP_MASK=1    bash scripts/reannotate.sh   # only redo captions
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── tmux guard ─────────────────────────────────────────────────────────────
TMUX_SESSION="${TMUX_SESSION:-sd-reannotate}"
if [ "${NO_TMUX:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
    if command -v tmux &>/dev/null; then
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            echo "tmux session '$TMUX_SESSION' already exists."
            echo "  Attach : tmux attach -t $TMUX_SESSION"
            echo "  Kill & restart: tmux kill-session -t $TMUX_SESSION && bash scripts/reannotate.sh"
            exit 0
        fi
        echo "=== Launching reannotate inside tmux session '$TMUX_SESSION' ==="
        echo "    Will keep running even if SSH disconnects."
        echo ""
        echo "  Attach later  : tmux attach -t $TMUX_SESSION"
        echo "  Detach        : Ctrl+B then D"
        echo ""
        tmux new-session -d -s "$TMUX_SESSION" \
            "cd $(pwd) && NO_TMUX=1 bash scripts/reannotate.sh $(printf '%q ' "$@"); echo ''; echo '=== Reannotate finished. Press any key to close ==='; read -n1"
        tmux attach -t "$TMUX_SESSION"
        exit 0
    else
        echo "WARNING: tmux not found – running in current shell."
    fi
fi
# ───────────────────────────────────────────────────────────────────────────

# ── Configuration ──────────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-data/interior}"
DEVICE="${DEVICE:-cuda}"

# Captioner: blip (6 GB) | florence2 (8 GB, recommended) | blip2 (15 GB)
CAPTIONER="${CAPTIONER:-florence2}"

# Masker: sam3 (recommended, text-prompted) | grounded_sam | sam2 | oneformer | sam
MASKER="${MASKER:-sam3}"

FURNITURE_LABELS="${FURNITURE_LABELS:-sofa,armchair,chair,dining chair,table,coffee table,bed,wardrobe,cabinet,lamp,floor lamp,curtain,rug,mirror}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"

PYTHON="${PYTHON:-python3}"
# ── End of Configuration ───────────────────────────────────────────────────


_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf  "║  %-60s║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

# ── Sanity check ───────────────────────────────────────────────────────────
_header "Reannotate – Environment"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python not found. Set the PYTHON env var."
    exit 1
fi

if [ ! -d "$DATASET_DIR/train/images" ] || [ ! -d "$DATASET_DIR/val/images" ]; then
    echo "ERROR: Dataset not found at '$DATASET_DIR'."
    echo "  Run 'bash scripts/run_pipeline.sh' first to create the initial split,"
    echo "  or set DATASET_DIR to the correct path."
    exit 1
fi

TRAIN_COUNT=$(find "$DATASET_DIR/train/images" -name "*.jpg" -o -name "*.png" -o -name "*.webp" | wc -l)
VAL_COUNT=$(find  "$DATASET_DIR/val/images"   -name "*.jpg" -o -name "*.png" -o -name "*.webp" | wc -l)

echo "  Python    : $($PYTHON --version)"
echo "  Device    : $DEVICE"
echo "  Captioner : $CAPTIONER"
echo "  Masker    : $MASKER"
echo "  Dataset   : $DATASET_DIR"
echo "  Train imgs: $TRAIN_COUNT"
echo "  Val imgs  : $VAL_COUNT"
echo ""
echo "  This script will OVERWRITE existing captions.json and masks/."


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 1  –  Re-generate captions
# ══════════════════════════════════════════════════════════════════════════
_header "STAGE 1 – Re-generate captions  (--captioner $CAPTIONER)"

if [ "${SKIP_CAPTION:-0}" = "1" ]; then
    echo "  ↳ Skipped (SKIP_CAPTION=1)"
else
    # Back up old captions before overwriting
    for split in train val; do
        old="$DATASET_DIR/$split/captions.json"
        if [ -f "$old" ]; then
            cp "$old" "${old%.json}_backup.json"
            echo "  Backed up $old → ${old%.json}_backup.json"
        fi
    done

    $PYTHON src/prepare_data.py caption \
        --dataset_dir "$DATASET_DIR" \
        --captioner   "$CAPTIONER" \
        --device      "$DEVICE"

    echo "  ✓ Captions written to $DATASET_DIR/{train,val}/captions.json"
fi


# ══════════════════════════════════════════════════════════════════════════
#  STAGE 2  –  Re-generate masks
# ══════════════════════════════════════════════════════════════════════════
_header "STAGE 2 – Re-generate masks  (--masker $MASKER)"

if [ "${SKIP_MASK:-0}" = "1" ]; then
    echo "  ↳ Skipped (SKIP_MASK=1)"
else
    # Back up old masks before overwriting
    for split in train val; do
        old_dir="$DATASET_DIR/$split/masks"
        if [ -d "$old_dir" ]; then
            bak_dir="${old_dir}_backup"
            rm -rf "$bak_dir"
            cp -r "$old_dir" "$bak_dir"
            echo "  Backed up $old_dir → $bak_dir"
        fi
    done

    case "$MASKER" in
      sam3)
        $PYTHON src/prepare_data.py mask \
            --dataset_dir      "$DATASET_DIR" \
            --masker           sam3 \
            --furniture_labels "$FURNITURE_LABELS" \
            --device           "$DEVICE"
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
      sam)
        if [ ! -f "$SAM_CHECKPOINT" ]; then
            echo "  SAM checkpoint not found at '$SAM_CHECKPOINT'. Downloading …"
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
      sam2|oneformer)
        $PYTHON src/prepare_data.py mask \
            --dataset_dir "$DATASET_DIR" \
            --masker      "$MASKER" \
            --device      "$DEVICE"
        ;;
      *)
        echo "ERROR: Unknown MASKER='$MASKER'. Valid options: sam | sam2 | sam3 | grounded_sam | oneformer"
        exit 1
        ;;
    esac

    echo "  ✓ Masks written to $DATASET_DIR/{train,val}/masks/"
fi


# ══════════════════════════════════════════════════════════════════════════
#  Done
# ══════════════════════════════════════════════════════════════════════════
_header "Reannotate complete"
echo "  Dataset : $DATASET_DIR"
echo ""
echo "  Next steps:"
echo "    Train  : bash scripts/train.sh"
echo "    Full pipeline (retrain): bash scripts/run_pipeline.sh"

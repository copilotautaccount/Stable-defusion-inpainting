#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_pipeline.sh  –  Complete end-to-end pipeline
#
# Runs every stage in the correct order:
#   Stage 0  Environment check
#   Stage 1  Install dependencies          (scripts/setup.sh)
#   Stage 2  Split raw images → train/val  (src/prepare_data.py split)
#   Stage 3  Generate captions             (src/prepare_data.py caption)
#   Stage 4  Generate furniture masks      (src/prepare_data.py mask)
#   Stage 5  Configure accelerate          (accelerate config)
#   Stage 6  Fine-tune the model           (scripts/train.sh)
#   Stage 7  Run inference (optional)      (src/inference.py)
#
# ── Quick start ──────────────────────────────────────────────────────────
#   1. Copy your interior images into data/raw/
#   2. Edit the Configuration section below (or export env vars)
#   3. Run:  bash scripts/run_pipeline.sh
#
# ── Skip individual stages ───────────────────────────────────────────────
#   Export SKIP_<STAGE>=1 before running to bypass a stage, e.g.:
#     SKIP_INSTALL=1 SKIP_ACCELERATE=1 bash scripts/run_pipeline.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── tmux guard ─────────────────────────────────────────────────────────────
# Re-launch the entire pipeline inside a tmux session so it survives SSH
# disconnects and laptop sleep.  Skip with NO_TMUX=1.
TMUX_SESSION="${TMUX_SESSION:-sd-pipeline}"
if [ "${NO_TMUX:-0}" != "1" ] && [ -z "${TMUX:-}" ]; then
    if command -v tmux &>/dev/null; then
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            echo "tmux session '$TMUX_SESSION' already exists."
            echo "  Attach : tmux attach -t $TMUX_SESSION"
            echo "  Kill & restart: tmux kill-session -t $TMUX_SESSION && bash scripts/run_pipeline.sh"
            exit 0
        fi
        echo "=== Launching full pipeline inside tmux session '$TMUX_SESSION' ==="
        echo "    All stages will continue even if SSH disconnects or laptop sleeps."
        echo ""
        echo "  Attach later  : tmux attach -t $TMUX_SESSION"
        echo "  Detach (keep running): Ctrl+B then D"
        echo ""
        # Pass all original env vars + NO_TMUX=1 so the re-spawned script skips this block
        tmux new-session -d -s "$TMUX_SESSION" \
            "cd $(pwd) && NO_TMUX=1 bash scripts/run_pipeline.sh $(printf '%q ' "$@"); echo ''; echo '=== Pipeline finished. Press any key to close ==='; read -n1"
        tmux attach -t "$TMUX_SESSION"
        exit 0
    else
        echo "WARNING: tmux not found – running in current shell (will stop on SSH disconnect)."
    fi
fi
# ───────────────────────────────────────────────────────────────────────────

# ── Configuration ──────────────────────────────────────────────────────────
# Paths
SOURCE_DIR="${SOURCE_DIR:-data/raw}"            # directory with your raw images
DATASET_DIR="${DATASET_DIR:-data/interior}"     # output: split + annotated dataset
CONFIG="${CONFIG:-configs/train_config.yaml}"   # training config
OUTPUT_DIR="${OUTPUT_DIR:-outputs/interior-inpainting}"

# Hardware
DEVICE="${DEVICE:-cuda}"                        # "cuda" or "cpu"
NUM_GPUS="${NUM_GPUS:-1}"                       # number of GPUs for training

# Data split
VAL_RATIO="${VAL_RATIO:-0.1}"                   # fraction of images used for validation
SEED="${SEED:-42}"

# Captioner  →  blip (6 GB) | florence2 (8 GB) | blip2 (15 GB)
CAPTIONER="${CAPTIONER:-florence2}"

# Masker     →  grounded_sam (recommended) | oneformer | sam2 | sam
MASKER="${MASKER:-grounded_sam}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_h_4b8939.pth}"
FURNITURE_LABELS="${FURNITURE_LABELS:-sofa,armchair,chair,dining chair,table,coffee table,bed,wardrobe,cabinet,lamp,floor lamp,curtain,rug,mirror}"

# Inference (Stage 7 – skipped by default; set SKIP_INFERENCE=0 to enable)
INFERENCE_IMAGE="${INFERENCE_IMAGE:-}"          # also set this and INFERENCE_MASK to enable
INFERENCE_MASK="${INFERENCE_MASK:-}"
INFERENCE_PROMPT="${INFERENCE_PROMPT:-a modern living room with a white linen sofa}"
INFERENCE_OUTPUT="${INFERENCE_OUTPUT:-results/result.png}"

PYTHON="${PYTHON:-python3}"
# ── End of Configuration ───────────────────────────────────────────────────


# ─── Utility helpers ───────────────────────────────────────────────────────
_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf  "║  %-60s║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

_skip() { echo "  ↳ Skipped (SKIP_${1}=1)"; }
# ───────────────────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 0  –  Environment check                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 0 – Environment check"

if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: Python not found. Set the PYTHON env var, e.g.: PYTHON=python3.10"
    exit 1
fi
echo "  Python  : $($PYTHON --version)"
echo "  Device  : $DEVICE"
echo "  Captioner: $CAPTIONER"
echo "  Masker  : $MASKER"
echo ""
echo "  Source images : $SOURCE_DIR"
echo "  Dataset dir   : $DATASET_DIR"
echo "  Config        : $CONFIG"
echo "  Output dir    : $OUTPUT_DIR"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 1  –  Install dependencies                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 1 – Install dependencies  (bash scripts/setup.sh)"

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    _skip "INSTALL"
else
    bash scripts/setup.sh
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2  –  Split raw images into train / val                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 2 – Split raw images  (src/prepare_data.py split)"
#
#  Input : $SOURCE_DIR/  – your raw interior images (any folder structure)
#  Output: $DATASET_DIR/
#            ├── train/images/
#            └── val/images/

if [ "${SKIP_SPLIT:-0}" = "1" ]; then
    _skip "SPLIT"
else
    if [ ! -d "$SOURCE_DIR" ]; then
        echo "ERROR: Source directory '$SOURCE_DIR' not found."
        echo "  Place your interior images there first, then re-run."
        exit 1
    fi
    $PYTHON src/prepare_data.py split \
        --source_dir "$SOURCE_DIR" \
        --output_dir "$DATASET_DIR" \
        --val_ratio  "$VAL_RATIO" \
        --seed       "$SEED"
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 3  –  Generate captions                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 3 – Generate captions  (src/prepare_data.py caption --captioner $CAPTIONER)"
#
#  Input : $DATASET_DIR/{train,val}/images/
#  Output: $DATASET_DIR/{train,val}/captions.json
#
#  Captioner options and approximate VRAM requirements:
#    blip      →  Salesforce/blip-image-captioning-large   (~6 GB)  fast
#    florence2 →  microsoft/Florence-2-large               (~8 GB)  detailed  ← default
#    blip2     →  Salesforce/blip2-opt-2.7b                (~15 GB) best quality

if [ "${SKIP_CAPTION:-0}" = "1" ]; then
    _skip "CAPTION"
else
    $PYTHON src/prepare_data.py caption \
        --dataset_dir "$DATASET_DIR" \
        --captioner   "$CAPTIONER" \
        --device      "$DEVICE"
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 4  –  Generate furniture masks                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 4 – Generate masks  (src/prepare_data.py mask --masker $MASKER)"
#
#  Input : $DATASET_DIR/{train,val}/images/
#  Output: $DATASET_DIR/{train,val}/masks/   (*.png, white = region to inpaint)
#
#  Masker options:
#    grounded_sam  →  GroundingDINO + SAM, text-prompted     (~10 GB) ← default / recommended
#    oneformer     →  panoptic segmentation (ADE20K 150 cls) (~12 GB)
#    sam2          →  Meta SAM 2, HuggingFace, no download   (~8 GB)
#    sam           →  Meta SAM v1, needs local .pth           (~7 GB)

if [ "${SKIP_MASK:-0}" = "1" ]; then
    _skip "MASK"
else
    case "$MASKER" in
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
        echo "ERROR: Unknown MASKER='$MASKER'. Valid options: sam | sam2 | grounded_sam | oneformer"
        exit 1
        ;;
    esac
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 5  –  Configure accelerate (interactive, run once)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 5 – Configure accelerate  (accelerate config)"
#
#  This step is interactive.  Run it once on a new machine.
#  Skip with:  SKIP_ACCELERATE=1 bash scripts/run_pipeline.sh

ACCELERATE_CFG="${HOME}/.cache/huggingface/accelerate/default_config.yaml"
if [ "${SKIP_ACCELERATE:-0}" = "1" ]; then
    _skip "ACCELERATE"
elif [ -f "$ACCELERATE_CFG" ]; then
    echo "  Accelerate config already exists at $ACCELERATE_CFG — skipping wizard."
    echo "  Mixed precision : $(grep mixed_precision "$ACCELERATE_CFG" | awk '{print $2}')"
    echo "  Num processes   : $(grep num_processes   "$ACCELERATE_CFG" | awk '{print $2}')"
else
    echo "  Starting 'accelerate config' …"
    echo "  (Set SKIP_ACCELERATE=1 to skip this step on subsequent runs.)"
    accelerate config
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 6  –  Fine-tune the model                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 6 – Fine-tune  (scripts/train.sh)"
#
#  Input : $DATASET_DIR/   (images + masks + captions.json)
#          $CONFIG          (configs/train_config.yaml)
#  Output: $OUTPUT_DIR/    (checkpoints, final LoRA weights)
#
#  Single GPU by default.  Set NUM_GPUS=4 for multi-GPU training.

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    _skip "TRAIN"
else
    NUM_GPUS="$NUM_GPUS" CONFIG="$CONFIG" bash scripts/train.sh
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 7  –  Inference (optional)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
_header "STAGE 7 – Inference  (src/inference.py)"
#
#  Inference is skipped by default.  To enable it, set:
#    SKIP_INFERENCE=0  INFERENCE_IMAGE=path/to/room.jpg  INFERENCE_MASK=path/to/mask.png
#
#  After training completes, run inference like this:
#    python src/inference.py \
#        --model_dir "$OUTPUT_DIR" \
#        --image     path/to/room.jpg \
#        --mask      path/to/mask.png \
#        --prompt    "a modern living room with a white linen sofa" \
#        --output    results/result.png

if [ "${SKIP_INFERENCE:-1}" = "1" ]; then
    _skip "INFERENCE"
    echo "  To run inference manually:"
    echo "    python src/inference.py \\"
    echo "        --model_dir \"$OUTPUT_DIR\" \\"
    echo "        --image     <path/to/room.jpg> \\"
    echo "        --mask      <path/to/mask.png> \\"
    echo "        --prompt    \"$INFERENCE_PROMPT\" \\"
    echo "        --output    \"$INFERENCE_OUTPUT\""
elif [ -n "$INFERENCE_IMAGE" ] && [ -n "$INFERENCE_MASK" ]; then
    mkdir -p "$(dirname "$INFERENCE_OUTPUT")"
    $PYTHON src/inference.py \
        --model_dir "$OUTPUT_DIR" \
        --image     "$INFERENCE_IMAGE" \
        --mask      "$INFERENCE_MASK" \
        --prompt    "$INFERENCE_PROMPT" \
        --output    "$INFERENCE_OUTPUT"
    echo "  Result saved → $INFERENCE_OUTPUT"
else
    echo "  INFERENCE_IMAGE / INFERENCE_MASK not set – skipping."
    echo "  Set both env vars and re-run Stage 7 to generate a result."
fi


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Done                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  Dataset    : $DATASET_DIR"
echo "  Checkpoints: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════════"

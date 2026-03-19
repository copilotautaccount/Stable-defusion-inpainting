#!/usr/bin/env bash
# =============================================================================
# evaluate.sh — Compare Base vs Fine-tuned SDXL Inpainting on validation set
# =============================================================================
# Usage:
#   bash scripts/evaluate.sh                    # full val set (1024 px, 30 steps)
#   NUM_SAMPLES=20 bash scripts/evaluate.sh      # quick test with 20 images
#   RESOLUTION=512 STEPS=20 bash scripts/evaluate.sh   # faster smoke-test
# =============================================================================
set -euo pipefail

# ── Configurable via environment variables ────────────────────────────────────
BASE_MODEL="${BASE_MODEL:-diffusers/stable-diffusion-xl-1.0-inpainting-0.1}"
LORA_DIR="${LORA_DIR:-outputs/interior-inpainting-sdxl/stage2/unet_lora}"
IMAGE_DIR="${IMAGE_DIR:-data/interior/val/images}"
MASK_DIR="${MASK_DIR:-data/interior/val/masks}"
CAPTIONS="${CAPTIONS:-data/interior/val/captions.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation}"
NUM_SAMPLES="${NUM_SAMPLES:-}"       # empty = all validation images
RESOLUTION="${RESOLUTION:-1024}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-7.5}"
SEED="${SEED:-42}"

# ── Move to repo root ─────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

echo "================================================================="
echo " Inpainting Evaluation: Base vs Fine-tuned"
echo "================================================================="
echo "  Base model  : $BASE_MODEL"
echo "  LoRA dir    : $LORA_DIR"
echo "  Val images  : $IMAGE_DIR"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Resolution  : ${RESOLUTION}px   Steps: $STEPS   Guidance: $GUIDANCE"
[[ -n "$NUM_SAMPLES" ]] && echo "  Num samples : $NUM_SAMPLES" || echo "  Num samples : all"
echo "================================================================="
echo ""

# ── Check LoRA weights exist ──────────────────────────────────────────────────
if [[ ! -f "${LORA_DIR}/adapter_model.safetensors" ]]; then
    echo "[ERROR] LoRA weights not found at: ${LORA_DIR}/adapter_model.safetensors"
    echo "        Make sure training has completed (stage 2)."
    exit 1
fi

# ── Install optional dependency: lpips ───────────────────────────────────────
echo "[INFO] Checking optional dependency: lpips ..."
pip install -q lpips 2>/dev/null && echo "[INFO] lpips ready." \
  || echo "[WARN] Could not install lpips; LPIPS metric will be skipped."

# ── Build python argument list ────────────────────────────────────────────────
PYTHON_ARGS=(
    --base_model  "$BASE_MODEL"
    --lora_dir    "$LORA_DIR"
    --image_dir   "$IMAGE_DIR"
    --mask_dir    "$MASK_DIR"
    --output_dir  "$OUTPUT_DIR"
    --resolution  "$RESOLUTION"
    --steps       "$STEPS"
    --guidance    "$GUIDANCE"
    --seed        "$SEED"
)

[[ -f "$CAPTIONS" ]] && PYTHON_ARGS+=(--captions "$CAPTIONS")
[[ -n "$NUM_SAMPLES" ]] && PYTHON_ARGS+=(--num_samples "$NUM_SAMPLES")

# ── Run evaluation ────────────────────────────────────────────────────────────
python src/evaluate.py "${PYTHON_ARGS[@]}"

echo ""
echo "================================================================="
echo " Done! Results saved to: ${OUTPUT_DIR}/"
echo "   base/         — base model inpainted images"
echo "   finetuned/    — fine-tuned model inpainted images"
echo "   comparisons/  — side-by-side grids (Original|Mask|Base|FT)"
echo "   metrics.csv   — per-image numeric metrics"
echo "   report.html   — visual HTML report (open in browser)"
echo "================================================================="

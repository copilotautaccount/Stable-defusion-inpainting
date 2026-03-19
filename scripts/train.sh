#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# train.sh – Launch fine-tuning (single GPU or multi-GPU)
#
# Usage:
#   # Multi-GPU 4 GPUs (default)
#   bash scripts/train.sh
#
#   # Single GPU
#   NUM_GPUS=1 bash scripts/train.sh
#
#   # Skip tmux (run in current shell)
#   NO_TMUX=1 bash scripts/train.sh
#
# Attach to the running session later:
#   tmux attach -t sd-train
#
# Detach without stopping (inside tmux):
#   Ctrl+B  then  D
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Ensure we run with the correct conda environment ─────────
CONDA_BASE="${CONDA_BASE:-/home/diffusion/miniconda3}"
CONDA_ENV="${CONDA_ENV:-base}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

CONFIG="${CONFIG:-configs/train_config.yaml}"
# Default: use all 4 GPUs
NUM_GPUS="${NUM_GPUS:-4}"
PYTHON=${PYTHON:-python3}
TMUX_SESSION="${TMUX_SESSION:-sd-train}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config.yaml}"

# ── Compose the actual training command ──────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="accelerate launch --config_file $ACCELERATE_CONFIG --num_processes $NUM_GPUS src/train.py --config $CONFIG"
else
    TRAIN_CMD="$PYTHON src/train.py --config $CONFIG"
fi

# ── Run inside tmux unless already inside one or NO_TMUX=1 ──
if [ "${NO_TMUX:-0}" = "1" ] || [ -n "${TMUX:-}" ]; then
    # Already inside tmux, or user explicitly skipped it → run directly
    echo "=== Launching training in current shell ==="
    eval "$TRAIN_CMD"
else
    if ! command -v tmux &>/dev/null; then
        echo "WARNING: tmux not found – running in current shell (will stop on SSH disconnect)."
        eval "$TRAIN_CMD"
        exit 0
    fi

    # If session already exists, ask user what to do
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "tmux session '$TMUX_SESSION' already exists."
        echo "  • Attach to it : tmux attach -t $TMUX_SESSION"
        echo "  • Kill and restart: tmux kill-session -t $TMUX_SESSION && bash scripts/train.sh"
        exit 0
    fi

    echo "=== Launching training inside tmux session '$TMUX_SESSION' ==="
    echo "    Training will continue even if you close your terminal or SSH disconnects."
    echo ""
    echo "  Attach later  : tmux attach -t $TMUX_SESSION"
    echo "  Detach (keep running): Ctrl+B then D"
    echo ""

    # Start a new detached tmux session that runs the training command,
    # then keeps the pane open so you can read the final output.
    tmux new-session -d -s "$TMUX_SESSION" \
        "cd $(pwd) && $TRAIN_CMD; echo ''; echo '=== Training finished. Press any key to close ==='; read -n1"

    # Attach automatically so the user sees the output right away
    tmux attach -t "$TMUX_SESSION"
fi

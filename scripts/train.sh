#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# train.sh – Launch fine-tuning (single GPU or multi-GPU)
#
# Usage:
#   # Single GPU  (runs inside tmux so it survives SSH disconnect)
#   bash scripts/train.sh
#
#   # Multi-GPU (4 GPUs)
#   NUM_GPUS=4 bash scripts/train.sh
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

CONFIG="${CONFIG:-configs/train_config.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
PYTHON=${PYTHON:-python3}
TMUX_SESSION="${TMUX_SESSION:-sd-train}"

# ── Compose the actual training command ──────────────────────
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="accelerate launch --num_processes $NUM_GPUS --mixed_precision fp16 src/train.py --config $CONFIG"
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

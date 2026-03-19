#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# setup.sh – Install all Python dependencies for the project
# Usage: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON:-python3}

# ── Redirect pip tmp/cache to the large data partition ─────────────────────
# The root filesystem (/) is small (~8 GB); NVIDIA wheels are huge (several GB).
# Storing pip downloads + temp files on /home/diffusion avoids "No space left"
# errors on root-mounted /tmp.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/home/diffusion/.cache/pip}"
export TMPDIR="${TMPDIR:-/home/diffusion/tmp_pip}"
mkdir -p "$TMPDIR"

# ── Skip install if key packages already present ────────────────────────────
if $PYTHON -c "import torch, diffusers, accelerate, peft" 2>/dev/null; then
    echo "=== Key packages already installed – skipping full install ==="
    echo "    (Set FORCE_INSTALL=1 to reinstall anyway)"
    if [ "${FORCE_INSTALL:-0}" != "1" ]; then
        echo ""
        echo "Setup complete (skipped)."
        exit 0
    fi
fi

echo "=== Installing Python dependencies ==="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8 (adjust the index URL for your CUDA version)
# Comment out if you already have the correct torch installed.
# $PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "=== Configuring accelerate ==="
echo "Run the following to set up accelerate for your hardware:"
echo "  accelerate config"

echo ""
echo "=== (Optional) Download SAM checkpoint ==="
echo "For automatic mask generation, download the SAM ViT-H checkpoint:"
echo "  mkdir -p checkpoints"
echo "  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

echo ""
echo "Setup complete."

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# setup.sh – Install all Python dependencies for the project
# Usage: bash scripts/setup.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON:-python3}

echo "=== Installing Python dependencies ==="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

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

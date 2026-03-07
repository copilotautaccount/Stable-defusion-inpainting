# Stable Diffusion Inpainting – Interior Design Fine-tuning

Fine-tune **Stable Diffusion Inpainting** on an interior-design dataset to intelligently edit furniture and room layouts using text prompts.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Full Pipeline – Execution Order](#full-pipeline--execution-order)
4. [Quick Start](#quick-start)
5. [Dataset Preparation](#dataset-preparation)
6. [Training](#training)
7. [Inference](#inference)
8. [Configuration Reference](#configuration-reference)
9. [Hardware Requirements](#hardware-requirements)
10. [Recommended Datasets](#recommended-datasets)

---

## Requirements

### Software

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| PyTorch | ≥ 2.0 | Deep learning framework |
| `diffusers` | ≥ 0.27 | Stable Diffusion pipeline |
| `transformers` | ≥ 4.38 | CLIP text encoder, BLIP-2 captioning |
| `accelerate` | ≥ 0.28 | Distributed / mixed-precision training |
| `peft` | ≥ 0.10 | LoRA adapters |
| `xformers` | ≥ 0.0.24 | Memory-efficient attention (optional) |
| `segment-anything` | ≥ 1.0 | Automatic mask generation (optional) |
| `opencv-python` | ≥ 4.9 | Image processing |
| `albumentations` | ≥ 1.4 | Data augmentation |
| `wandb` | ≥ 0.16 | Experiment tracking (optional) |

Install all dependencies:

```bash
bash scripts/setup.sh
# or manually:
pip install -r requirements.txt
```

### Hardware

| Mode | GPU VRAM | Notes |
|------|----------|-------|
| LoRA fine-tuning (fp16) | **≥ 12 GB** | Recommended for consumer GPUs (RTX 3090/4090) |
| Full fine-tuning (fp16) | **≥ 24 GB** | A100 / H100 recommended |
| Full fine-tuning + gradient checkpointing | **≥ 16 GB** | Slower but feasible on 16 GB GPUs |

---

## Project Structure

```
Stable-defusion-inpainting/
├── configs/
│   ├── train_config.yaml      # All training hyper-parameters
│   └── annotation_config.yaml # Model options for captioning & masking
├── data/
│   └── interior/              # Processed dataset (created by prepare_data.sh)
│       ├── train/
│       │   ├── images/        # *.jpg / *.png
│       │   ├── masks/         # *.png (optional – auto-generated if absent)
│       │   └── captions.json  # {"filename.jpg": "caption text"}
│       └── val/
│           ├── images/
│           ├── masks/
│           └── captions.json
├── scripts/
│   ├── setup.sh               # Install dependencies
│   ├── prepare_data.sh        # Run full data pipeline
│   ├── run_pipeline.sh        # ← Run EVERYTHING in order (setup → data → train)
│   └── train.sh               # Launch training
├── src/
│   ├── dataset.py             # PyTorch Dataset + mask generators
│   ├── prepare_data.py        # Split / caption / mask CLI tool
│   ├── train.py               # Fine-tuning script (LoRA or full)
│   └── inference.py           # Inference script (single or batch)
├── tests/
│   └── test_dataset.py        # Unit tests
├── outputs/                   # Training checkpoints & final model
├── requirements.txt
└── README.md
```

---

## Full Pipeline – Execution Order

> **Short answer:** copy images → run one command.
>
> ```bash
> # Place your images in data/raw/, then:
> bash scripts/run_pipeline.sh
> ```
>
> The script walks through every stage in the correct order and can be
> re-run at any time; individual stages can be skipped with
> `SKIP_<STAGE>=1` env vars.

The table below shows the **mandatory sequence** for going from raw images to
a working inpainting model:

| # | Stage | Script / Command | Input | Output |
|---|-------|-----------------|-------|--------|
| 0 | Check environment | *(automatic)* | — | — |
| **1** | **Install dependencies** | `bash scripts/setup.sh` | `requirements.txt` | Python packages installed |
| **2** | **Split images** | `python src/prepare_data.py split` | `data/raw/` | `data/interior/{train,val}/images/` |
| **3** | **Generate captions** | `python src/prepare_data.py caption` | split images | `data/interior/{train,val}/captions.json` |
| **4** | **Generate masks** | `python src/prepare_data.py mask` | split images | `data/interior/{train,val}/masks/` |
| **5** | **Configure accelerate** | `accelerate config` *(once)* | — | `~/.cache/huggingface/accelerate/default_config.yaml` |
| **6** | **Fine-tune model** | `bash scripts/train.sh` | dataset + config | `outputs/interior-inpainting/` |
| 7 | *(optional)* Run inference | `python src/inference.py` | trained model + image + mask | inpainted image |

### Why this order matters

```
Raw images
    │
    ▼  Stage 2: split
data/interior/
├── train/images/  ─┐
└── val/images/    ─┤  Stage 3: caption  →  captions.json
                    └  Stage 4: mask     →  masks/*.png
                              │
                              ▼  Stage 5: accelerate config (once)
                              │
                              ▼  Stage 6: train.py
                        outputs/interior-inpainting/
                              │
                              ▼  Stage 7: inference.py
                          result.png
```

### One-command run (all stages)

```bash
# Copy your interior images first
cp -r /path/to/your/photos  data/raw/

# Run the full pipeline (florence2 captions + grounded_sam masks by default)
bash scripts/run_pipeline.sh
```

### Run with custom model choices

```bash
# Use BLIP-2 for captions and SAM 2 for masks
CAPTIONER=blip2 MASKER=sam2 bash scripts/run_pipeline.sh

# Use BLIP + OneFormer on a machine with limited VRAM
CAPTIONER=blip MASKER=oneformer bash scripts/run_pipeline.sh
```

### Re-run only specific stages

```bash
# Skip install and accelerate config (already done), re-run caption + mask + train
SKIP_INSTALL=1 SKIP_ACCELERATE=1 bash scripts/run_pipeline.sh

# Only retrain (captions & masks already exist)
SKIP_INSTALL=1 SKIP_SPLIT=1 SKIP_CAPTION=1 SKIP_MASK=1 SKIP_ACCELERATE=1 \
    bash scripts/run_pipeline.sh
```

### Step-by-step (manual)

If you prefer to run each step individually:

```bash
# ── Step 1: Install ───────────────────────────────────────────
bash scripts/setup.sh

# ── Step 2: Split raw images ──────────────────────────────────
python src/prepare_data.py split \
    --source_dir data/raw \
    --output_dir data/interior \
    --val_ratio  0.1

# ── Step 3: Generate captions ─────────────────────────────────
# Pick ONE of: blip (6 GB) | florence2 (8 GB, recommended) | blip2 (15 GB)
python src/prepare_data.py caption \
    --dataset_dir data/interior \
    --captioner   florence2 \
    --device      cuda

# ── Step 4: Generate masks ────────────────────────────────────
# Pick ONE of: grounded_sam (recommended) | oneformer | sam2 | sam
python src/prepare_data.py mask \
    --dataset_dir      data/interior \
    --masker           grounded_sam \
    --sam_checkpoint   checkpoints/sam_vit_h_4b8939.pth \
    --furniture_labels "sofa,chair,table,bed,cabinet,lamp" \
    --device           cuda

# ── Step 5: Configure accelerate (once per machine) ───────────
accelerate config

# ── Step 6: Fine-tune ─────────────────────────────────────────
bash scripts/train.sh
# or: python src/train.py --config configs/train_config.yaml

# ── Step 7: Inference (optional) ──────────────────────────────
python src/inference.py \
    --model_dir outputs/interior-inpainting \
    --image     data/interior/val/images/room_001.jpg \
    --mask      data/interior/val/masks/room_001.png \
    --prompt    "a modern living room with a white linen sofa" \
    --output    results/result.png
```

---

## Quick Start

See [Full Pipeline – Execution Order](#full-pipeline--execution-order) for the
complete step-by-step guide.  The minimal path is:

```bash
# 1. Install dependencies
bash scripts/setup.sh

# 2. Place your interior images in data/raw/
#    (JPG / PNG / WebP, any resolution)

# 3. Prepare dataset: split → caption → mask  (all-in-one)
bash scripts/prepare_data.sh

# 4. Configure accelerate for your hardware (interactive, run once)
accelerate config

# 5. Start training
bash scripts/train.sh
```

Or run **all stages in one command**:

```bash
cp -r /path/to/photos  data/raw/
bash scripts/run_pipeline.sh
```

---

## Dataset Preparation

Since the raw dataset contains **only images** (no masks and no captions), you
must run the preparation pipeline below to annotate it before training.

See [`configs/annotation_config.yaml`](configs/annotation_config.yaml) for a
complete reference of all model options and their hardware requirements.

### Step 1 – Organise images

Place all raw interior images into `data/raw/`. Any folder structure is supported.

```
data/raw/
├── living_room_001.jpg
├── bedroom_042.jpg
└── kitchen_015.png
```

### Step 2 – Split into train / val

```bash
python src/prepare_data.py split \
    --source_dir data/raw \
    --output_dir data/interior \
    --val_ratio  0.1
```

---

### Step 3 – Generate captions

Three captioning models are available. Choose based on your available GPU memory:

| Model | Flag | HuggingFace ID | VRAM | Speed | Notes |
|-------|------|----------------|------|-------|-------|
| **BLIP** | `--captioner blip` | `Salesforce/blip-image-captioning-large` | ~6 GB | Fast | Good quality, no prefix |
| **BLIP-2** | `--captioner blip2` | `Salesforce/blip2-opt-2.7b` | ~15 GB | Medium | Best quality, supports text prefix |
| **Florence-2** | `--captioner florence2` | `microsoft/Florence-2-large` | ~8 GB | Fast | Dense region-aware captions – **recommended for complex scenes** |

**Quick recommendation:**
- ≤ 8 GB VRAM → use `blip` or `florence2`
- ≥ 16 GB VRAM → use `blip2` for richer, longer captions
- Multi-object / cluttered interiors → use `florence2`

```bash
# Fast & memory-efficient
python src/prepare_data.py caption \
    --dataset_dir data/interior \
    --captioner   blip

# Best quality
python src/prepare_data.py caption \
    --dataset_dir data/interior \
    --captioner   blip2

# Detailed region-aware captions (recommended)
python src/prepare_data.py caption \
    --dataset_dir data/interior \
    --captioner   florence2
```

**Alternatively**, write `captions.json` manually:

```json
{
  "living_room_001.jpg": "A modern living room with grey sectional sofa and floor lamp",
  "bedroom_042.jpg": "A minimalist Scandinavian bedroom with white linen and oak furniture"
}
```

---

### Step 4 – Generate masks

Four masking models are available. Choose based on your requirements:

| Model | Flag | Approach | VRAM | Accuracy | Notes |
|-------|------|----------|------|----------|-------|
| **Grounded-SAM** | `--masker grounded_sam` | Text-prompted (GroundingDINO + SAM) | ~10 GB | ⭐⭐⭐⭐⭐ | **Recommended** – specify which furniture to mask |
| **OneFormer** | `--masker oneformer` | Panoptic segmentation (ADE20K) | ~12 GB | ⭐⭐⭐⭐ | Category-level masks for 150+ indoor classes |
| **SAM 2** | `--masker sam2` | Automatic (no labels needed) | ~8 GB | ⭐⭐⭐⭐ | Improved SAM, no download needed |
| **SAM** | `--masker sam` | Automatic (centre heuristic) | ~7 GB | ⭐⭐⭐ | Requires manual checkpoint download |

#### Grounded-SAM (recommended)

Detects furniture by name and produces pixel-precise masks. Most useful when you
know which object categories should be inpainted.

```bash
# Install dependencies
pip install groundingdino-py

# Download SAM checkpoint
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Run (specify the furniture categories you want to mask)
python src/prepare_data.py mask \
    --dataset_dir      data/interior \
    --masker           grounded_sam \
    --sam_checkpoint   checkpoints/sam_vit_h_4b8939.pth \
    --furniture_labels "sofa,armchair,chair,table,coffee table,bed,cabinet,lamp,curtain"
```

#### OneFormer – semantic category masks

Uses ADE20K panoptic segmentation (150 categories including sofa, chair, bed,
table, cabinet, wardrobe, lamp, curtain, rug, mirror, etc.).

```bash
python src/prepare_data.py mask \
    --dataset_dir data/interior \
    --masker      oneformer
```

#### SAM 2 – improved automatic masking

No manual checkpoint download required (loaded from HuggingFace).

```bash
pip install 'git+https://github.com/facebookresearch/sam2.git'

python src/prepare_data.py mask \
    --dataset_dir data/interior \
    --masker      sam2
```

#### SAM v1 – classic automatic masking

```bash
# Download checkpoint
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

python src/prepare_data.py mask \
    --dataset_dir    data/interior \
    --masker         sam \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type     vit_h
```

If no masks are provided, the training script generates **random masks** on-the-fly
(bounding boxes, irregular strokes, or mixed – controlled by `data.mask_type` in
`configs/train_config.yaml`).

---

## Training

### LoRA Fine-tuning (recommended)

LoRA fine-tunes only a small set of adapter weights (~1–4% of total parameters),
making it feasible on a single consumer GPU with 12+ GB of VRAM.

```bash
# Single GPU
python src/train.py --config configs/train_config.yaml

# Multi-GPU (example: 4 × A100)
accelerate launch --num_processes=4 src/train.py --config configs/train_config.yaml
```

### Full Fine-tuning

```bash
python src/train.py --config configs/train_config.yaml --no_lora
```

### Monitoring

Training metrics and validation images are logged to **W&B** by default.
Change `logging.report_to` in `configs/train_config.yaml` to `"tensorboard"` or `"none"`.

```bash
wandb login   # first time only
```

### Checkpoints

Checkpoints are saved every `training.checkpointing_steps` steps under
`outputs/interior-inpainting/checkpoint-<step>/`.
Only the 3 most recent checkpoints are retained.

Resume training:

```bash
# Automatically load the latest checkpoint
python src/train.py --config configs/train_config.yaml
# (set resume_from_checkpoint: "latest" in the config)
```

---

## Inference

### Single image

```bash
python src/inference.py \
    --model_dir outputs/interior-inpainting \
    --image     path/to/living_room.jpg \
    --mask      path/to/mask.png \
    --prompt    "a modern living room with a white linen sofa and oak coffee table" \
    --output    result.png
```

### Using LoRA adapters

```bash
python src/inference.py \
    --model_dir runwayml/stable-diffusion-inpainting \
    --lora_dir  outputs/interior-inpainting/unet_lora \
    --image     room.jpg \
    --mask      mask.png \
    --prompt    "a cosy Japandi bedroom with natural textures" \
    --output    result.png
```

### Batch inference

```bash
python src/inference.py \
    --model_dir  outputs/interior-inpainting \
    --image_dir  data/interior/val/images \
    --mask_dir   data/interior/val/masks \
    --prompt     "a bright Scandinavian living room with natural wood accents" \
    --output_dir results/
```

### Key inference parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--num_inference_steps` | 50 | More steps → higher quality, slower |
| `--guidance_scale` | 7.5 | Higher → more prompt-adherent, less diverse |
| `--strength` | 0.99 | 1.0 = full inpaint; lower = preserve original texture |
| `--seed` | 42 | Set for reproducibility |

---

## Configuration Reference

Edit `configs/train_config.yaml` to adjust the training run.

```yaml
model:
  pretrained_model_name_or_path: "runwayml/stable-diffusion-inpainting"

lora:
  enabled: true    # false = full fine-tune
  rank: 16         # higher rank = more capacity
  alpha: 32

data:
  dataset_dir: "data/interior"
  image_size: 512
  mask_type: "mixed"   # "bbox" | "irregular" | "mixed"

training:
  train_batch_size: 2
  gradient_accumulation_steps: 4   # effective batch = 8
  num_train_epochs: 50
  learning_rate: 1.0e-4
  mixed_precision: "fp16"
```

---

## Hardware Requirements

| Setup | GPU | VRAM | Training Speed |
|-------|-----|------|----------------|
| LoRA, fp16, batch=2, 512px | RTX 3090 | 12 GB | ~1 it/s |
| LoRA, fp16, batch=4, 512px | A100 40GB | 20 GB | ~3 it/s |
| Full, fp16, batch=2, 512px | A100 80GB | 35 GB | ~0.8 it/s |

Enable `gradient_checkpointing: true` and `use_8bit_adam: true` (requires `bitsandbytes`) to
reduce VRAM usage by ~20–30% at the cost of slightly slower training.

---

## Recommended Datasets

For interior design inpainting, the following public datasets work well:

| Dataset | Images | Notes |
|---------|--------|-------|
| [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) | 27 000+ | Indoor scenes with semantic labels |
| [LSUN Bedroom/Living Room](https://www.yf.io/p/lsun) | 3M+ | High-resolution interior photos |
| [InteriorNet](https://interiornet.org/) | 20M+ | Synthetic rendered interiors |
| [Structured3D](https://structured3d-dataset.org/) | 196 500 | Photo-realistic panoramic rooms |
| [SUN RGB-D](https://rgbd.cs.princeton.edu/) | 10 000+ | Real indoor RGBD images |
| Scraped Pinterest/Houzz images | Custom | High-quality real-world interiors |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## License

This project is released under the [MIT License](LICENSE).
The base model (`runwayml/stable-diffusion-inpainting`) is subject to the
[CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
# Stable Diffusion Inpainting – Interior Design Fine-tuning

Fine-tune **Stable Diffusion Inpainting** on an interior-design dataset to intelligently edit furniture and room layouts using text prompts.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Inference](#inference)
7. [Configuration Reference](#configuration-reference)
8. [Hardware Requirements](#hardware-requirements)
9. [Recommended Datasets](#recommended-datasets)

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
│   └── train_config.yaml      # All training hyper-parameters
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

## Quick Start

```bash
# 1. Install dependencies
bash scripts/setup.sh

# 2. Place your interior images in data/raw/
#    (JPG / PNG / WebP, any resolution)

# 3. Prepare dataset (split + caption + mask)
bash scripts/prepare_data.sh

# 4. Configure accelerate for your hardware
accelerate config

# 5. Start training
bash scripts/train.sh
```

---

## Dataset Preparation

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

### Step 3 – Auto-generate captions (recommended)

Uses **BLIP-2** (Salesforce/blip2-opt-2.7b) to generate interior-specific captions.
Requires ~15 GB of GPU memory for the 2.7B model.

```bash
python src/prepare_data.py caption \
    --dataset_dir data/interior \
    --device      cuda
```

**Alternatively**, create `captions.json` manually:

```json
{
  "living_room_001.jpg": "A modern living room with grey sectional sofa and floor lamp",
  "bedroom_042.jpg": "A minimalist Scandinavian bedroom with white linen and oak furniture"
}
```

### Step 4 – Auto-generate masks (optional)

Uses **SAM** (Meta's Segment Anything) to detect and mask furniture objects.
Download the SAM checkpoint first:

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Then run:

```bash
python src/prepare_data.py mask \
    --dataset_dir    data/interior \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type     vit_h \
    --device         cuda
```

If no masks are provided, the training script generates **random masks** on-the-fly
(bounding boxes, irregular strokes, or mixed – controlled by `data.mask_type` in the config).

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
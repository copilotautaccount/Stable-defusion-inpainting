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
11. [Fine-tuning Deep Dive](#fine-tuning-deep-dive)
12. [Getting Updates](#getting-updates)
13. [Running Tests](#running-tests)

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

### Step-by-step manual – `data/interior_v2`

Dùng các lệnh dưới đây để chạy từng bước trực tiếp trên thư mục `data/interior_v2`
(images đã được split sẵn, chỉ cần chạy caption → mask → train):

```bash
# ── Bước 1: Sinh captions (Florence-2, ~8 GB VRAM) ───────────
python src/prepare_data.py caption \
    --dataset_dir data/interior_v2 \
    --captioner   florence2 \
    --device      cuda

# ── Bước 2: Sinh masks (Grounded-SAM, ~10 GB VRAM) ──────────
python src/prepare_data.py mask \
    --dataset_dir      data/interior_v2 \
    --masker           grounded_sam \
    --sam_checkpoint   checkpoints/sam_vit_h_4b8939.pth \
    --furniture_labels "sofa,armchair,chair,dining chair,table,coffee table,bed,wardrobe,cabinet,lamp,floor lamp,curtain,rug,mirror" \
    --device           cuda

# ── Bước 3: Cấu hình accelerate (chạy 1 lần / máy) ──────────
accelerate config

# ── Bước 4: Train SDXL – Stage 1 (noise MSE only) ────────────
python src/train.py \
    --config configs/train_config.yaml \
    --stage  1

# ── Bước 5: Train SDXL – Stage 2 (multi-loss từ Stage 1) ─────
python src/train.py \
    --config configs/train_config.yaml \
    --stage  2

# ── Bước 6: Inference (tuỳ chọn) ────────────────────────────
python src/inference.py \
    --model_dir outputs/interior-inpainting-sdxl/stage2 \
    --lora_dir  outputs/interior-inpainting-sdxl/stage2/unet_lora \
    --image     data/interior_v2/val/images/<tên_ảnh>.jpg \
    --mask      data/interior_v2/val/masks/<tên_ảnh>.png \
    --prompt    "a modern living room with a white linen sofa" \
    --output    outputs/inference_results/result.png
```

> **Chạy nhanh toàn bộ pipeline cho `interior_v2`:**
> ```bash
> DATASET_DIR=data/interior_v2 \
> SKIP_SPLIT=1 \
> SKIP_INSTALL=1 \
> bash scripts/run_pipeline.sh
> ```
> `SKIP_SPLIT=1` vì `interior_v2` đã có sẵn thư mục `train/` và `val/`.

### Step-by-step (manual – generic)

If you prefer to run each step individually with a custom dataset:

```bash
# ── Step 1: Install ───────────────────────────────────────────
bash scripts/setup.sh

# ── Step 2: Split raw images ──────────────────────────────────
python src/prepare_data.py split \
    --source_dir data/raw \
    --output_dir data/interior_v2 \
    --val_ratio  0.1

# ── Step 3: Generate captions ─────────────────────────────────
# Pick ONE of: blip (6 GB) | florence2 (8 GB, recommended) | blip2 (15 GB)
python src/prepare_data.py caption \
    --dataset_dir data/interior_v2 \
    --captioner   florence2 \
    --device      cuda

# ── Step 4: Generate masks ────────────────────────────────────
# Pick ONE of: grounded_sam (recommended) | oneformer | sam2 | sam
python src/prepare_data.py mask \
    --dataset_dir      data/interior_v2 \
    --masker           grounded_sam \
    --sam_checkpoint   checkpoints/sam_vit_h_4b8939.pth \
    --furniture_labels "sofa,armchair,chair,dining chair,table,coffee table,bed,wardrobe,cabinet,lamp,floor lamp,curtain,rug,mirror" \
    --device           cuda

# ── Step 5: Configure accelerate (once per machine) ───────────
accelerate config

# ── Step 6a: Train – Stage 1 (SDXL LoRA, noise MSE only) ─────
python src/train.py --config configs/train_config.yaml --stage 1

# ── Step 6b: Train – Stage 2 (multi-loss từ Stage 1) ─────────
python src/train.py --config configs/train_config.yaml --stage 2

# ── Step 7: Inference (optional) ──────────────────────────────
python src/inference.py \
    --model_dir outputs/interior-inpainting-sdxl/stage2 \
    --lora_dir  outputs/interior-inpainting-sdxl/stage2/unet_lora \
    --image     data/interior_v2/val/images/room_001.jpg \
    --mask      data/interior_v2/val/masks/room_001.png \
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

## Fine-tuning Deep Dive

> **Tiếng Việt / Vietnamese** — phần này trình bày chi tiết từng bước trong quá trình
> fine-tune model mà script `src/train.py` thực hiện.
> An English summary follows each Vietnamese block.

---

### Tổng quan kiến trúc (Architecture Overview)

Stable Diffusion Inpainting gồm **4 thành phần** chính. Trong quá trình fine-tune
chỉ **UNet** được cập nhật trọng số; các thành phần còn lại bị đóng băng (frozen):

| Thành phần | Vai trò | Được huấn luyện? |
|------------|---------|-----------------|
| **VAE** (`AutoencoderKL`) | Nén ảnh RGB 512×512 → không gian latent 64×64×4 (encode) và ngược lại (decode) | ❌ Frozen |
| **Text Encoder** (`CLIPTextModel`) | Chuyển text prompt thành vector 768 chiều để hướng dẫn UNet | ❌ Frozen |
| **UNet** (`UNet2DConditionModel`) | Mô hình chính dự đoán nhiễu tại mỗi bước khuếch tán | ✅ Trainable |
| **Noise Scheduler** (`DDPMScheduler`) | Quản lý quá trình thêm/xoá nhiễu, không có trọng số | — |

> **EN:** Only the UNet is trained. VAE and CLIP text encoder are frozen — their
> weights never change. The noise scheduler has no learnable parameters.

---

### Bước 1 – Khởi tạo và cấu hình (Initialisation)

```python
# src/train.py  lines 210-215
tokenizer    = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder")
vae          = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
unet         = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
scheduler    = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
```

**Điều xảy ra:**
- Tải trọng số đã được pre-train từ `runwayml/stable-diffusion-inpainting`
  (HuggingFace Hub).
- VAE và text encoder bị đóng băng (`requires_grad_(False)`) — chúng không
  thay đổi trong suốt quá trình huấn luyện.
- UNet (hoặc chỉ các adapter LoRA của nó) được đánh dấu là trainable.

> **EN:** All four components are loaded from the pre-trained checkpoint.
> VAE and text encoder are immediately frozen. Only the UNet (or its LoRA
> adapters) will accumulate gradients.

---

### Bước 2 – Cài LoRA vào UNet (Inject LoRA Adapters)

```python
# src/train.py  lines 95-111
lora_cfg = LoraConfig(
    r=16,           # rank — số chiều ẩn của ma trận low-rank
    lora_alpha=32,  # scaling: weight_scale = alpha / rank = 2.0
    target_modules=["to_q","to_k","to_v","to_out.0",
                    "proj_in","proj_out","ff.net.0.proj","ff.net.2"],
    lora_dropout=0.05,
    bias="none",
)
unet = get_peft_model(unet, lora_cfg)
```

**Điều xảy ra:**
- Thay vì cập nhật **toàn bộ** ~860 triệu tham số của UNet, LoRA chèn thêm
  **2 ma trận nhỏ** (A và B) vào từng lớp attention/projection được chọn.
- Số tham số có thể huấn luyện giảm xuống còn khoảng **~8–15 triệu** (< 2%).
- Ma trận A khởi tạo ngẫu nhiên (Gaussian), ma trận B khởi tạo bằng 0 →
  lúc đầu LoRA không thay đổi output của lớp đó.
- Công thức: `W' = W₀ + (alpha/rank) × B×A`

```
Lớp attention gốc:          W₀  (frozen, không đổi)
                              ↓
LoRA thêm vào:         + (α/r) · B · A   ← chỉ A và B được huấn luyện
```

> **EN:** Instead of updating all ~860 M UNet parameters, LoRA injects tiny
> rank-16 matrices (A, B) into the eight named attention/projection modules.
> Only ~8–15 M parameters are trainable. The effective weight update is
> `ΔW = (alpha/rank) × B × A`.

---

### Bước 3 – Chuẩn bị dữ liệu (Dataset & DataLoader)

Mỗi batch gồm các tensor sau (xem `src/dataset.py`):

| Tensor | Shape | Ý nghĩa |
|--------|-------|---------|
| `pixel_values` | `(B, 3, 512, 512)` | Ảnh gốc chuẩn hoá `[-1, 1]` |
| `masked_image` | `(B, 3, 512, 512)` | Ảnh gốc với vùng mask bị tô đen (× 0) |
| `mask` | `(B, 1, 512, 512)` | 0 = giữ nguyên, 1 = vùng cần inpaint |
| `input_ids` | `(B, 77)` | Token IDs của text caption (CLIP tokenizer) |

**Augmentation (chỉ khi training):**
- Random horizontal flip (50%)
- Color jitter nhẹ (brightness ±10%, contrast ±10%, saturation ±10%, hue ±5%)

**Mask generation** (nếu không có mask sẵn):
- `"bbox"` – hình chữ nhật ngẫu nhiên
- `"irregular"` – nét cọ tự do (free-form brush strokes)
- `"mixed"` (mặc định) – 50% bbox, 50% irregular

> **EN:** Each DataLoader batch contains the original image, the masked image
> (inpaint region zeroed out), the binary mask, and tokenised caption IDs.
> Light augmentations (flip + colour jitter) are applied during training.

---

### Bước 4 – Vòng lặp huấn luyện từng step (Per-step Training Loop)

Đây là trái tim của quá trình fine-tune. Mỗi step thực hiện 8 micro-step sau:

#### 4a. Encode ảnh → latent space (VAE Encoding)

```python
# src/train.py  lines 357-363
latents        = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
masked_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor
```

- VAE nén ảnh `512×512×3` → `64×64×4` (hệ số nén 64×).
- `scaling_factor ≈ 0.18215` — chuẩn hoá phân phối latent.
- `masked_latents` là latent của ảnh đã xoá vùng inpaint.

> **EN:** The VAE encodes both the original and the masked image into 64×64×4
> latent tensors. All diffusion math runs in this compressed latent space.

#### 4b. Resize mask → latent resolution

```python
# src/train.py  lines 365-370
mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
# 512×512×1  →  64×64×1
```

> **EN:** The pixel-space mask is downsampled to match the 64×64 latent grid.

#### 4c. Thêm nhiễu ngẫu nhiên (Forward Diffusion / Add Noise)

```python
# src/train.py  lines 372-382
noise      = torch.randn_like(latents)           # Gaussian noise ε ~ N(0,I)
timesteps  = torch.randint(0, 1000, (B,))        # t ~ Uniform[0, 999]
noisy_latents = scheduler.add_noise(latents, noise, timesteps)
# = √ᾱₜ · latents  +  √(1-ᾱₜ) · noise    (DDPM formula)
```

- `t` được lấy ngẫu nhiên từ 0–999 (1000 timesteps).
- Timestep càng lớn → ảnh nhiễu càng nhiều.
- Mục tiêu: model học cách **dự đoán phần nhiễu** đã được thêm vào.

> **EN:** A random Gaussian noise tensor `ε` is sampled, a random timestep
> `t` is drawn, and the DDPM formula mixes the clean latent with noise to
> produce `noisy_latents`. The model's job is to un-mix this.

#### 4d. Encode text caption (CLIP Text Encoding)

```python
# src/train.py  lines 384-387
encoder_hidden_states = text_encoder(input_ids)[0]
# shape: (B, 77, 768) — 77 token positions, 768-dim embeddings
```

> **EN:** The frozen CLIP text encoder converts the tokenised caption into
> 768-dimensional contextual embeddings that the UNet will cross-attend to.

#### 4e. Tạo input 9 channel cho UNet (9-channel UNet Input)

```python
# src/train.py  lines 389-391
unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)
# Channels:    [4 channels]  [1 channel] [4 channels]  =  9 channels total
```

**Đây là điểm khác biệt quan trọng** giữa SD Inpainting và SD gốc:
- SD gốc: UNet nhận input **4 channels** (chỉ noisy latent).
- SD Inpainting: UNet nhận **9 channels** = noisy latent + mask + masked image latent.
- Cấu trúc này giúp UNet "nhìn thấy" vùng nào cần inpaint và phần ảnh
  nào cần giữ nguyên.

```
Noisy latent   [4ch]  ┐
Mask           [1ch]  ├──→  UNet (9-channel input conv)  →  noise prediction [4ch]
Masked latent  [4ch]  ┘
```

> **EN:** This is the inpainting-specific design. The standard SD UNet takes
> 4 channels; the inpainting variant takes 9 — appending the resized mask and
> the masked-image latent so the UNet "sees" what must be filled in.

#### 4f. UNet dự đoán nhiễu (Noise Prediction)

```python
# src/train.py  lines 393-396
model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
# shape: (B, 4, 64, 64)  — predicted noise ε̂
```

- UNet là kiến trúc **U-Net với Transformer blocks** (cross-attention với text).
- Với mỗi timestep `t` và mỗi text condition, UNet dự đoán lượng nhiễu `ε̂`.

> **EN:** The UNet processes the 9-channel input, the timestep embedding, and
> the CLIP text context via cross-attention, and outputs a 4-channel predicted
> noise tensor.

#### 4g. Tính hàm mất mát (Loss Computation)

```python
# src/train.py  lines 398-408
target = noise                              # prediction_type = "epsilon"
loss   = F.mse_loss(model_pred, target)    # MSE( ε̂, ε )
```

- Dùng **MSE loss** (Mean Squared Error) giữa nhiễu dự đoán và nhiễu thực.
- Đây là tiêu chuẩn DDPM / LDM: model tối thiểu hoá `E[‖ε - ε̂‖²]`.

> **EN:** The loss is the mean squared error between the predicted noise and
> the actual noise that was added. Minimising this is equivalent to maximising
> the ELBO of the diffusion model's variational lower bound.

#### 4h. Backward & cập nhật trọng số (Backprop + Optimiser Step)

```python
# src/train.py  lines 414-419
accelerator.backward(loss)
accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)   # gradient clipping
optimizer.step()       # AdamW update  (chỉ LoRA A, B matrices)
lr_scheduler.step()    # cosine LR decay
optimizer.zero_grad()
```

- **Gradient accumulation = 4** → gradient được tích luỹ qua 4 micro-batch
  trước khi update một lần. Effective batch size = `2 × 4 = 8`.
- **Gradient clipping** tại `max_norm=1.0` giúp ổn định training.
- **Cosine LR scheduler** với warmup 200 steps: LR tăng dần rồi giảm dần
  theo đường cong cosine.

> **EN:** Gradients flow only into the LoRA A and B matrices (the frozen
> weights receive zero gradient). After 4 accumulation steps, AdamW updates
> the parameters. A cosine schedule with 200-step linear warmup governs
> the learning rate.

---

### Bước 5 – Checkpoint và Validation (Checkpointing & Validation)

**Checkpoint** (mỗi 500 steps):
```python
# src/train.py  lines 434-448
accelerator.save_state(f"outputs/interior-inpainting/checkpoint-{global_step}/")
# Giữ tối đa 3 checkpoint gần nhất, xoá checkpoint cũ
```

**Validation** (mỗi 500 steps):
- Tạo ảnh trắng (dummy image) + mask trung tâm.
- Chạy inference 20 steps với 4 prompt validation.
- Log kết quả lên W&B để theo dõi chất lượng trực quan.

> **EN:** Every 500 steps the full accelerator state is saved (keeping only
> the 3 most recent checkpoints). The validation loop runs 20-step inference
> on four fixed prompts and uploads the results to W&B for qualitative
> monitoring.

---

### Bước 6 – Lưu model cuối (Save Final Model)

```python
# src/train.py  lines 470-483
if lora_enabled:
    unet.save_pretrained("outputs/interior-inpainting/unet_lora/")
else:
    pipeline.save_pretrained("outputs/interior-inpainting/")
```

**LoRA mode (mặc định):**
- Chỉ lưu các adapter nhỏ (`unet_lora/`) — kích thước ~50–200 MB.
- Khi inference: tải SD Inpainting gốc + nạp adapter lên trên.

**Full fine-tune mode:**
- Lưu toàn bộ pipeline (~4–7 GB).

> **EN:** In LoRA mode only the small adapter weights (~50–200 MB) are saved.
> At inference time the base model is loaded and the adapters are applied on
> top. Full fine-tune mode saves the entire pipeline.

---

### Tóm tắt luồng dữ liệu qua một training step

```
┌─────────────────────────────────────────────────────────────┐
│  Input: (image, mask, caption)                              │
│                                                             │
│  1. VAE.encode(image)       → latents        [B,4,64,64]   │
│  2. VAE.encode(masked_img)  → masked_latents [B,4,64,64]   │
│  3. Interpolate(mask)       → mask_64        [B,1,64,64]   │
│  4. randn_like(latents)     → ε (noise)      [B,4,64,64]   │
│  5. randint(0,1000)         → t (timestep)   [B]           │
│  6. add_noise(latents,ε,t)  → noisy_latents  [B,4,64,64]   │
│  7. CLIP.encode(caption)    → text_embeds    [B,77,768]    │
│                                                             │
│  8. UNet([noisy_latents | mask_64 | masked_latents],        │
│          t, text_embeds)    → ε̂ (pred noise) [B,4,64,64]   │
│                                                             │
│  9. loss = MSE(ε̂, ε)                                       │
│ 10. loss.backward() → update only LoRA A, B                │
└─────────────────────────────────────────────────────────────┘
```

---

### Tham số cấu hình quan trọng nhất

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|-----------------|---------|
| `lora.rank` | 16 | Số chiều của low-rank matrices — tăng → chất lượng cao hơn nhưng tốn VRAM |
| `lora.alpha` | 32 | Hệ số scale = alpha/rank = 2.0 |
| `training.num_train_epochs` | 50 | Số epoch — thường cần 20–100 tuỳ dataset |
| `training.learning_rate` | 1e-4 | LR ban đầu — quá cao → training không ổn định |
| `training.train_batch_size` | 2 | Training batch size per GPU |
| `training.gradient_accumulation_steps` | 4 | Effective batch = 2 × 4 = 8 |
| `training.mixed_precision` | fp16 | Giảm ~50% VRAM, tốc độ tăng đáng kể |
| `data.mask_type` | mixed | Loại mask ngẫu nhiên dùng khi không có mask sẵn |
| `data.image_size` | 512 | Độ phân giải training — 768 cần nhiều VRAM hơn |

---

## Getting Updates

This section explains how to pull the latest bug-fixes and improvements into
a copy of the repository that you have already cloned on your server.

### Scenario A – you cloned from `main` (the default)

```bash
cd /path/to/Stable-defusion-inpainting   # enter your local clone

git fetch origin          # download all remote changes without touching your files
git pull origin main      # merge the latest main branch into your local copy
```

### Scenario B – you want the Florence-2 fix branch specifically

The fix for the `"model of type florence2 to instantiate model of type ``"` crash
(and the `torch_dtype` deprecation warning) lives on the
`copilot/fine-tune-stable-diffusion-inpainting` branch.

#### Option 1 – switch to the fix branch directly

```bash
cd /path/to/Stable-defusion-inpainting

git fetch origin
git checkout copilot/fine-tune-stable-diffusion-inpainting
```

Everything in `src/prepare_data.py` now includes the fix.  Run as normal:

```bash
python src/prepare_data.py caption \
    --dataset_dir data/interior \
    --captioner   florence2 \
    --device      cuda
```

#### Option 2 – cherry-pick just the fix commit into your current branch

If you want to stay on your own branch but apply only the Florence-2 fix:

```bash
git fetch origin

# The commit SHA for the Florence-2 fix:
git cherry-pick fe4a26fbae0baed63c44b99d71904d66a2ec57e6
```

#### Option 3 – merge the fix branch into your working branch

```bash
git fetch origin
git merge origin/copilot/fine-tune-stable-diffusion-inpainting
```

### Verifying the update was applied

After updating, confirm the version-aware Florence-2 loader is present:

```bash
grep "_transformers_version" src/prepare_data.py
# Should print two lines (the helper function definition and its call site)
```

Then run the test suite to make sure everything is healthy:

```bash
pip install pytest
pytest tests/test_prepare_data.py -v
# All tests should pass (54 passed)
```

### Keeping up with future changes

```bash
# Run this any time you want the latest version:
git pull origin main
```

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
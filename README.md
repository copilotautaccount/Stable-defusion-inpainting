# Stable Diffusion Inpainting

Fine-tune and run inference with a **Stable Diffusion Inpainting** model on your own dataset.

---

## Table of Contents

- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
  - [Single-image inference](#single-image-inference)
  - [Batch inference](#batch-inference)
  - [Using the base model without fine-tuning](#using-the-base-model-without-fine-tuning)
- [Project Structure](#project-structure)

---

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (≥ 16 GB VRAM for 512×512 training).

---

## Dataset Preparation

Organise your data as follows:

```
data/
├── images/          # original RGB images  (*.png / *.jpg)
├── masks/           # binary masks          (*.png / *.jpg)
│                    #   white (255) = region to fill
│                    #   black  (0) = region to keep
└── prompts.txt      # (optional) one text prompt per line,
                     # in the same order as the sorted image list
```

Image and mask filenames do **not** need to share the same name – they are matched by sort order.  
If `prompts.txt` is omitted every image will be trained with an empty prompt.

---

## Training

```bash
python train.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
    --data_dir ./data \
    --output_dir ./output \
    --num_train_epochs 10 \
    --train_batch_size 2 \
    --learning_rate 1e-5 \
    --image_size 512 \
    --save_steps 500 \
    --mixed_precision fp16
```

| Argument | Default | Description |
|---|---|---|
| `--pretrained_model_name_or_path` | `runwayml/stable-diffusion-inpainting` | Base model (HF id or local path) |
| `--data_dir` | *(required)* | Dataset root directory |
| `--output_dir` | `./output` | Where to save checkpoints & final model |
| `--image_size` | `512` | Training resolution |
| `--train_batch_size` | `2` | Per-device batch size |
| `--num_train_epochs` | `10` | Number of training epochs |
| `--learning_rate` | `1e-5` | Initial learning rate |
| `--save_steps` | `500` | Save a checkpoint every N steps |
| `--mixed_precision` | `no` | `fp16` / `bf16` / `no` |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation |
| `--use_8bit_adam` | `False` | Enable 8-bit Adam (requires `bitsandbytes`) |

After training the full pipeline (tokeniser, VAE, text encoder, fine-tuned UNet) is
saved to `--output_dir` in the standard Diffusers format and can be loaded directly
for inference.

---

## Inference

### Single-image inference

```bash
python inference.py \
    --model_path ./output \
    --image ./data/images/photo.png \
    --mask  ./data/masks/photo.png \
    --prompt "a beautiful garden" \
    --output_dir ./results
```

### Batch inference

```bash
python inference.py \
    --model_path ./output \
    --image_dir ./data/images \
    --mask_dir  ./data/masks \
    --prompt "a beautiful garden" \
    --output_dir ./results
```

Files are matched by **filename stem** (e.g. `photo.png` ↔ `photo.png`).

### Using the base model without fine-tuning

```bash
python inference.py \
    --model_path runwayml/stable-diffusion-inpainting \
    --image ./photo.png \
    --mask  ./mask.png \
    --prompt "a cozy living room"
```

### Full list of inference arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `./output` | Fine-tuned model dir or HF model id |
| `--lora_weights` | `None` | Optional LoRA adapter directory |
| `--image` | `None` | Single source image (single-image mode) |
| `--mask` | `None` | Single mask image (single-image mode) |
| `--image_dir` | `None` | Directory of images (batch mode) |
| `--mask_dir` | `None` | Directory of masks (batch mode) |
| `--prompt` | `""` | Text prompt |
| `--negative_prompt` | `"low quality, blurry, distorted"` | Negative prompt |
| `--num_inference_steps` | `50` | Denoising steps |
| `--guidance_scale` | `7.5` | Classifier-free guidance scale |
| `--strength` | `1.0` | Inpainting strength (0–1) |
| `--num_images_per_prompt` | `1` | Images to generate per input |
| `--seed` | `None` | Random seed for reproducibility |
| `--output_dir` | `./results` | Where to save generated images |
| `--image_size` | `512` | Inference resolution |
| `--device` | auto | `cuda` / `cpu` |
| `--mixed_precision` | `fp16` | `fp16` / `bf16` / `no` |

---

## Project Structure

```
.
├── dataset.py       # PyTorch Dataset for inpainting data
├── train.py         # Fine-tuning script
├── inference.py     # Inference script
├── requirements.txt # Python dependencies
└── README.md
```

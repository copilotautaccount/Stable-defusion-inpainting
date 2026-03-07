"""
Inference script for the fine-tuned SD Inpainting model.

Usage
-----
# Basic inpainting with a single image and mask
python src/inference.py \\
    --model_dir outputs/interior-inpainting \\
    --image path/to/room.jpg \\
    --mask path/to/mask.png \\
    --prompt "a modern living room with a white sofa" \\
    --output result.png

# Batch inference from a directory
python src/inference.py \\
    --model_dir outputs/interior-inpainting \\
    --image_dir data/interior/val/images \\
    --mask_dir  data/interior/val/masks \\
    --prompt "a bright, airy Scandinavian living room" \\
    --output_dir results/

# LoRA adapter on top of base model
python src/inference.py \\
    --model_dir runwayml/stable-diffusion-inpainting \\
    --lora_dir  outputs/interior-inpainting/unet_lora \\
    --image path/to/room.jpg \\
    --mask  path/to/mask.png \\
    --prompt "a cosy Japanese-style bedroom" \\
    --output result.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str, size: Optional[int] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def _load_mask(path: str, size: Optional[int] = None) -> Image.Image:
    mask = Image.open(path).convert("L")
    if size:
        mask = mask.resize((size, size), Image.NEAREST)
    return mask


def _build_pipeline(
    model_dir: str,
    lora_dir: Optional[str],
    device: str,
    dtype: torch.dtype,
    enable_xformers: bool,
):
    """Load the inpainting pipeline (optionally with LoRA adapters)."""
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        safety_checker=None,
    )

    if lora_dir:
        pipe.unet.load_attn_procs(lora_dir)

    if enable_xformers:
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def inpaint_single(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
    num_images: int,
) -> List[Image.Image]:
    """Run inpainting on a single image + mask."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    outputs = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images
    return outputs


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def inpaint_batch(
    pipe,
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
    resolution: int,
) -> None:
    """Run inpainting for every image/mask pair in a directory."""
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(p for p in image_root.iterdir() if p.suffix.lower() in extensions)

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    from tqdm import tqdm

    for img_path in tqdm(image_paths, desc="Inpainting"):
        mask_path = mask_root / (img_path.stem + ".png")
        if not mask_path.exists():
            # Try same extension
            mask_path = mask_root / img_path.name
        if not mask_path.exists():
            print(f"  [WARN] Mask not found for {img_path.name}, skipping.")
            continue

        image = _load_image(str(img_path), size=resolution)
        mask = _load_mask(str(mask_path), size=resolution)

        results = inpaint_single(
            pipe, image, mask, prompt, negative_prompt,
            num_inference_steps, guidance_scale, strength, seed, num_images=1,
        )
        save_path = out_root / img_path.name
        results[0].save(save_path)

    print(f"Results saved to {out_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference with fine-tuned SD Inpainting for interior design"
    )

    # Model
    p.add_argument("--model_dir", required=True,
                   help="Path to saved pipeline or HF model ID")
    p.add_argument("--lora_dir", default=None,
                   help="Path to LoRA adapter directory (unet_lora/)")

    # Input – single image
    p.add_argument("--image", default=None, help="Path to input image")
    p.add_argument("--mask",  default=None, help="Path to binary mask (255=inpaint)")

    # Input – batch
    p.add_argument("--image_dir",  default=None, help="Directory of input images (batch mode)")
    p.add_argument("--mask_dir",   default=None, help="Directory of masks (batch mode)")
    p.add_argument("--output_dir", default="results", help="Output directory (batch mode)")

    # Output – single
    p.add_argument("--output", default="output.png", help="Output path (single mode)")

    # Generation
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--negative_prompt", default=(
        "blurry, low quality, distorted, ugly, out of focus, "
        "bad anatomy, watermark, text, oversaturated"
    ), help="Negative prompt")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--strength", type=float, default=0.99,
                   help="Inpainting strength; 1.0 = full noise, lower = preserve more")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_images", type=int, default=1,
                   help="Number of images to generate per prompt (single mode)")
    p.add_argument("--resolution", type=int, default=512)

    # Hardware
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use fp16 precision (default: True)")
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--enable_xformers", action="store_true", default=False)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float16 if args.fp16 else torch.float32

    print(f"Loading pipeline from: {args.model_dir}")
    pipe = _build_pipeline(
        model_dir=args.model_dir,
        lora_dir=args.lora_dir,
        device=args.device,
        dtype=dtype,
        enable_xformers=args.enable_xformers,
    )

    # Batch mode
    if args.image_dir and args.mask_dir:
        inpaint_batch(
            pipe=pipe,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed,
            resolution=args.resolution,
        )
        return

    # Single-image mode
    if not (args.image and args.mask):
        raise ValueError(
            "Provide either (--image + --mask) for single mode "
            "or (--image_dir + --mask_dir) for batch mode."
        )

    image = _load_image(args.image, size=args.resolution)
    mask = _load_mask(args.mask, size=args.resolution)

    results = inpaint_single(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
        num_images=args.num_images,
    )

    if args.num_images == 1:
        results[0].save(args.output)
        print(f"Saved: {args.output}")
    else:
        out_path = Path(args.output)
        for i, img in enumerate(results):
            p = out_path.with_stem(f"{out_path.stem}_{i}")
            img.save(p)
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()

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

# Manual interactive inference – pick a random image, draw a mask, type a prompt
python src/inference.py \\
    --model_dir outputs/interior-inpainting \\
    --manual \\
    --image_dir data/interior/val/images
"""

from __future__ import annotations

import argparse
import os
import random
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


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _pick_random_image(image_dir: str) -> Path:
    """Return a random image path from *image_dir*.

    Raises ``FileNotFoundError`` if the directory does not exist and
    ``RuntimeError`` if no supported images are found.
    """
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(
        p for p in root.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {root}")
    return random.choice(paths)


def _draw_mask_interactive(image: Image.Image) -> Image.Image:
    """Open a matplotlib window and let the user paint a binary mask.

    The user draws white strokes on a transparent overlay.  Close the window
    (or press **q**) to finish.  Returns a single-channel ``"L"`` PIL image
    with 255 = inpaint region and 0 = keep.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Draw mask (left-click & drag). Close window when done.")
    ax.axis("off")

    w, h = image.size
    mask_array = np.zeros((h, w), dtype=np.uint8)

    state = {"drawing": False, "brush": 20}

    def _on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        state["drawing"] = True
        _paint(event)

    def _on_release(event):
        state["drawing"] = False

    def _on_motion(event):
        if state["drawing"] and event.inaxes == ax:
            _paint(event)

    def _paint(event):
        x, y = int(round(event.xdata)), int(round(event.ydata))
        r = state["brush"]
        y0, y1 = max(0, y - r), min(h, y + r)
        x0, x1 = max(0, x - r), min(w, x + r)
        mask_array[y0:y1, x0:x1] = 255
        # Visual overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[mask_array > 0] = [255, 0, 0, 120]
        if state.get("_overlay") is not None:
            state["_overlay"].remove()
        state["_overlay"] = ax.imshow(overlay)
        fig.canvas.draw_idle()

    def _on_scroll(event):
        if event.button == "up":
            state["brush"] = min(state["brush"] + 5, 100)
        else:
            state["brush"] = max(state["brush"] - 5, 3)

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    plt.show()

    return Image.fromarray(mask_array, mode="L")


def _show_results(
    original: Image.Image,
    mask: Image.Image,
    results: List[Image.Image],
    save_path: Optional[str] = None,
) -> None:
    """Display original / mask / inpainted result(s) side-by-side.

    If *save_path* is given the comparison figure is also saved to disk.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    n_results = len(results)
    ncols = 2 + n_results
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    for i, img in enumerate(results):
        axes[2 + i].imshow(img)
        axes[2 + i].set_title(f"Result {i + 1}" if n_results > 1 else "Result")
        axes[2 + i].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison saved: {save_path}")
    plt.show()


def manual_inference(
    pipe,
    image_dir: str,
    resolution: int,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
    num_images: int,
    output: str,
) -> None:
    """Interactive manual inference workflow.

    1. Pick a random image from *image_dir*.
    2. Let the user draw a mask interactively.
    3. Ask the user for an inpainting prompt.
    4. Run inference and display results.
    """
    # Step 1 – random image
    img_path = _pick_random_image(image_dir)
    print(f"Selected image: {img_path}")
    image = _load_image(str(img_path), size=resolution)

    # Step 2 – interactive mask
    print("Draw mask on the image. Close the window when done.")
    mask = _draw_mask_interactive(image)
    mask = mask.resize(image.size, Image.NEAREST)

    if np.array(mask).sum() == 0:
        print("[WARN] Empty mask – nothing to inpaint. Exiting.")
        return

    # Step 3 – prompt
    prompt = input("Enter inpainting prompt: ").strip()
    if not prompt:
        prompt = "A high-quality interior design photo"
        print(f"  Using default prompt: {prompt}")

    # Step 4 – inference
    print("Running inpainting …")
    results = inpaint_single(
        pipe, image, mask, prompt, negative_prompt,
        num_inference_steps, guidance_scale, strength, seed, num_images,
    )

    # Save
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if num_images == 1:
        results[0].save(out_path)
        print(f"Saved: {out_path}")
    else:
        for i, img in enumerate(results):
            p = out_path.with_stem(f"{out_path.stem}_{i}")
            img.save(p)
            print(f"Saved: {p}")

    # Step 5 – show comparison
    comparison_path = str(out_path.with_stem(f"{out_path.stem}_comparison"))
    _show_results(image, mask, results, save_path=comparison_path)


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

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference with fine-tuned SD Inpainting for interior design"
    )

    # Model
    p.add_argument("--model_dir", required=True,
                   help="Path to saved pipeline or HF model ID")
    p.add_argument("--lora_dir", default=None,
                   help="Path to LoRA adapter directory (unet_lora/)")

    # Manual interactive mode
    p.add_argument("--manual", action="store_true", default=False,
                   help="Manual mode: pick random image, draw mask, type prompt")

    # Input – single image
    p.add_argument("--image", default=None, help="Path to input image")
    p.add_argument("--mask",  default=None, help="Path to binary mask (255=inpaint)")

    # Input – batch / manual image source
    p.add_argument("--image_dir",  default=None,
                   help="Directory of input images (batch or manual mode)")
    p.add_argument("--mask_dir",   default=None, help="Directory of masks (batch mode)")
    p.add_argument("--output_dir", default="results", help="Output directory (batch mode)")

    # Output – single / manual
    p.add_argument("--output", default="output.png", help="Output path (single/manual mode)")

    # Generation
    p.add_argument("--prompt", default=None,
                   help="Text prompt (required for single/batch mode; "
                        "entered interactively in manual mode)")
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
                   help="Number of images to generate per prompt (single/manual mode)")
    p.add_argument("--resolution", type=int, default=512)

    # Hardware
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use fp16 precision (default: True)")
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--enable_xformers", action="store_true", default=False)

    args = p.parse_args(argv)

    # Validate: prompt is required unless manual mode
    if not args.manual and args.prompt is None:
        p.error("--prompt is required for single/batch mode (or use --manual)")

    return args


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

    # Manual interactive mode
    if args.manual:
        if not args.image_dir:
            raise ValueError("--image_dir is required for --manual mode")
        manual_inference(
            pipe=pipe,
            image_dir=args.image_dir,
            resolution=args.resolution,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed,
            num_images=args.num_images,
            output=args.output,
        )
        return

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
            "Provide either (--image + --mask) for single mode, "
            "(--image_dir + --mask_dir) for batch mode, "
            "or --manual with --image_dir for interactive mode."
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

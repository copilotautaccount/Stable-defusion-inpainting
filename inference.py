"""
inference.py — Stable Diffusion Inpainting Inference Script
============================================================
This script loads a Stable Diffusion inpainting model and uses it to
fill (inpaint) a masked region of an input image based on a text prompt.

Requirements:
    pip install torch torchvision diffusers transformers accelerate Pillow

Usage:
    python inference.py \
        --image path/to/input.png \
        --mask  path/to/mask.png  \
        --prompt "a cozy living room" \
        --output result.png

Mask convention:
    White pixels (255) = area to inpaint (will be replaced).
    Black pixels  (0)  = area to keep   (will be preserved).
"""

import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    """Load an image from *path* and convert it to RGB."""
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> Image.Image:
    """Load a mask image from *path* and convert it to greyscale (L)."""
    return Image.open(path).convert("L")


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str, device: str) -> StableDiffusionInpaintPipeline:
    """
    Download (or load from cache) the inpainting pipeline.

    Parameters
    ----------
    model_id : str
        Hugging Face model repository ID, e.g.
        "runwayml/stable-diffusion-inpainting".
    device : str
        Target device — "cuda" or "cpu".

    Returns
    -------
    StableDiffusionInpaintPipeline
        The ready-to-use pipeline moved to *device*.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # Speed optimisation: reduce VRAM usage with attention slicing.
    if device == "cuda":
        pipe.enable_attention_slicing()

    return pipe


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inpainting(
    pipe: StableDiffusionInpaintPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> Image.Image:
    """
    Run the inpainting pipeline on *image* using *mask*.

    Parameters
    ----------
    pipe : StableDiffusionInpaintPipeline
        Loaded pipeline.
    image : PIL.Image.Image
        Original input image (RGB).
    mask : PIL.Image.Image
        Mask image (L / greyscale).  White = inpaint, black = keep.
    prompt : str
        Text describing what to generate inside the masked region.
    negative_prompt : str
        Text describing what to *avoid* generating.
    width, height : int
        Output resolution (must be multiples of 8; 512 recommended).
    num_inference_steps : int
        Number of denoising steps.  More steps → higher quality but slower.
    guidance_scale : float
        How strongly the model follows the prompt (typical range 5–15).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PIL.Image.Image
        The inpainted result image.
    """
    generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

    # Resize to the target dimensions expected by the model.
    image_resized = image.resize((width, height))
    mask_resized = mask.resize((width, height))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_resized,
        mask_image=mask_resized,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    return result.images[0]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Inpainting — fill a masked image region with AI-generated content.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument("--image",  required=True,  help="Path to the input image (PNG / JPEG).")
    parser.add_argument("--mask",   required=True,  help="Path to the mask image (white = inpaint, black = keep).")
    parser.add_argument("--prompt", required=True,  help="Text prompt describing what to generate in the masked area.")

    # --- Optional ---
    parser.add_argument("--output",           default="output.png",
                        help="Path for the saved result image.")
    parser.add_argument("--model",            default="runwayml/stable-diffusion-inpainting",
                        help="Hugging Face model ID.")
    parser.add_argument("--negative-prompt",  default="",
                        help="Text describing what to avoid in the output.")
    parser.add_argument("--width",            type=int,   default=512,
                        help="Output width  (multiple of 8).")
    parser.add_argument("--height",           type=int,   default=512,
                        help="Output height (multiple of 8).")
    parser.add_argument("--steps",            type=int,   default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--guidance-scale",   type=float, default=7.5,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--seed",             type=int,   default=42,
                        help="Random seed for reproducibility.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()
    print(f"[info] Using device: {device}")

    print(f"[info] Loading model: {args.model}")
    pipe = load_pipeline(args.model, device)

    print(f"[info] Loading image : {args.image}")
    image = load_image(args.image)

    print(f"[info] Loading mask  : {args.mask}")
    mask = load_mask(args.mask)

    print(f"[info] Running inpainting …")
    result = run_inpainting(
        pipe=pipe,
        image=image,
        mask=mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    result.save(args.output)
    print(f"[info] Result saved to: {args.output}")


if __name__ == "__main__":
    main()

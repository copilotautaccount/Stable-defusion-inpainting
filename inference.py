"""
Run inference with a fine-tuned (or base) Stable Diffusion Inpainting model.

The script supports:
  * Loading a model saved by ``train.py`` (local directory or HuggingFace repo)
  * Single-image mode     – provide ``--image`` and ``--mask``
  * Batch directory mode  – provide ``--image_dir`` and ``--mask_dir``
  * Optional LoRA weights via ``--lora_weights``

Usage examples
--------------
# Single image
python inference.py \
    --model_path ./output \
    --image ./data/images/photo.png \
    --mask  ./data/masks/photo.png \
    --prompt "a beautiful garden" \
    --output_dir ./results

# Batch directory
python inference.py \
    --model_path ./output \
    --image_dir ./data/images \
    --mask_dir  ./data/masks \
    --prompt "a beautiful garden" \
    --output_dir ./results

# Using the base model without fine-tuning
python inference.py \
    --model_path runwayml/stable-diffusion-inpainting \
    --image ./photo.png \
    --mask  ./mask.png \
    --prompt "a cozy living room"
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm.auto import tqdm


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_pipeline(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    lora_weights: Optional[str] = None,
) -> StableDiffusionInpaintPipeline:
    """Load the inpainting pipeline from *model_path*.

    Args:
        model_path:    Local directory produced by ``train.py`` or a HuggingFace
                       model id (e.g. ``runwayml/stable-diffusion-inpainting``).
        device:        Target device string (``"cuda"``, ``"cpu"``, etc.).
        dtype:         Torch dtype for model weights.
        lora_weights:  Optional path to a LoRA weights directory.

    Returns:
        A ready-to-use :class:`StableDiffusionInpaintPipeline`.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
    )

    if lora_weights:
        pipe.load_lora_weights(lora_weights)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def prepare_inputs(
    image_path: str,
    mask_path: str,
    image_size: Tuple[int, int],
) -> Tuple[Image.Image, Image.Image]:
    """Load and resize an image/mask pair.

    The mask is binarised: pixels > 127 are treated as the region to inpaint.

    Args:
        image_path: Path to the source RGB image.
        mask_path:  Path to the binary mask image.
        image_size: (width, height) to resize both images.

    Returns:
        Tuple of (image, mask) as PIL Images.
    """
    import numpy as np

    image = Image.open(image_path).convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize(image_size, Image.Resampling.NEAREST)

    mask_arr = np.array(mask)
    mask_arr = (mask_arr > 127).astype("uint8") * 255
    mask = Image.fromarray(mask_arr, mode="L")

    return image, mask


def collect_image_mask_pairs(
    image_dir: str,
    mask_dir: str,
) -> List[Tuple[Path, Path]]:
    """Collect matching (image, mask) path pairs from two directories.

    Files are matched by **stem** (filename without extension).  Both
    directories must contain the same set of stems.

    Args:
        image_dir: Directory containing source images.
        mask_dir:  Directory containing corresponding masks.

    Returns:
        Sorted list of (image_path, mask_path) pairs.

    Raises:
        FileNotFoundError: If no images are found or stems do not match.
    """
    image_dir_path = Path(image_dir)
    mask_dir_path = Path(mask_dir)

    image_map = {
        p.stem: p
        for p in image_dir_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    mask_map = {
        p.stem: p
        for p in mask_dir_path.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    if not image_map:
        raise FileNotFoundError(f"No image files found in {image_dir}")

    common_stems = sorted(set(image_map) & set(mask_map))
    missing_masks = sorted(set(image_map) - set(mask_map))

    if missing_masks:
        print(
            f"Warning: {len(missing_masks)} image(s) have no matching mask and "
            "will be skipped: " + ", ".join(missing_masks[:5])
            + (" ..." if len(missing_masks) > 5 else "")
        )

    if not common_stems:
        raise FileNotFoundError(
            "No matching image/mask pairs found. "
            "Make sure filenames (without extension) match between "
            f"{image_dir} and {mask_dir}."
        )

    return [(image_map[s], mask_map[s]) for s in common_stems]


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference with a fine-tuned Stable Diffusion Inpainting model"
    )

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output",
        help=(
            "Path to the fine-tuned model directory saved by train.py, or a "
            "HuggingFace model id such as 'runwayml/stable-diffusion-inpainting'."
        ),
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Optional path to a directory containing LoRA adapter weights.",
    )

    # Input – single image mode
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single source image (use with --mask).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to a single mask image (use with --image).",
    )

    # Input – batch directory mode
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory of source images for batch inference (use with --mask_dir).",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory of mask images for batch inference (use with --image_dir).",
    )

    # Generation
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt describing the desired inpainting result.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted",
        help="Negative text prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="How strongly to transform the masked region (0–1).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of output images to generate per input.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible outputs.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory where generated images are saved.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Resolution (square) used for inference.",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to run inference on (e.g. 'cuda', 'cpu'). "
            "Defaults to 'cuda' if available, otherwise 'cpu'."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Use half-precision inference to reduce memory usage.",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_inference(
    pipe: StableDiffusionInpaintPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    num_images_per_prompt: int,
    generator: Optional[torch.Generator],
) -> List[Image.Image]:
    """Run the inpainting pipeline on a single image/mask pair.

    Args:
        pipe:                   The loaded :class:`StableDiffusionInpaintPipeline`.
        image:                  Source RGB image.
        mask:                   Binary mask (white = region to inpaint).
        prompt:                 Text prompt.
        negative_prompt:        Negative text prompt.
        num_inference_steps:    Denoising steps.
        guidance_scale:         CFG scale.
        strength:               Inpainting strength (0–1).
        num_images_per_prompt:  Number of output images to generate.
        generator:              Optional :class:`torch.Generator` for seeding.

    Returns:
        List of generated PIL images.
    """
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )
    return result.images


def main():
    args = parse_args()

    # Validate input arguments
    single_mode = args.image is not None and args.mask is not None
    batch_mode = args.image_dir is not None and args.mask_dir is not None

    if not single_mode and not batch_mode:
        raise ValueError(
            "Provide either --image + --mask for single-image mode, "
            "or --image_dir + --mask_dir for batch mode."
        )
    if single_mode and batch_mode:
        raise ValueError(
            "Specify either single-image mode (--image / --mask) "
            "or batch mode (--image_dir / --mask_dir), not both."
        )

    # Device & dtype
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.mixed_precision]

    if args.device == "cpu" and args.mixed_precision != "no":
        print(
            "Warning: mixed precision is not recommended on CPU. "
            "Falling back to float32."
        )
        dtype = torch.float32

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pipeline
    print(f"Loading model from: {args.model_path}")
    pipe = load_pipeline(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        lora_weights=args.lora_weights,
    )

    # Random seed
    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    image_size = (args.image_size, args.image_size)

    # ------------------------------------------------------------------ #
    # Build the list of (image_path, mask_path) pairs to process
    # ------------------------------------------------------------------ #
    if single_mode:
        pairs = [(Path(args.image), Path(args.mask))]
    else:
        pairs = collect_image_mask_pairs(args.image_dir, args.mask_dir)

    print(f"Running inference on {len(pairs)} image(s)…")

    for image_path, mask_path in tqdm(pairs, desc="Generating"):
        image, mask = prepare_inputs(
            str(image_path), str(mask_path), image_size
        )

        generated_images = run_inference(
            pipe=pipe,
            image=image,
            mask=mask,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            num_images_per_prompt=args.num_images_per_prompt,
            generator=generator,
        )

        stem = image_path.stem
        for idx, gen_image in enumerate(generated_images):
            if args.num_images_per_prompt == 1:
                out_filename = f"{stem}_inpainted.png"
            else:
                out_filename = f"{stem}_inpainted_{idx}.png"
            out_path = os.path.join(args.output_dir, out_filename)
            gen_image.save(out_path)
            print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()

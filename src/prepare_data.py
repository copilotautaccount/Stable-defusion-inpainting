"""
Data-preparation utilities for interior inpainting fine-tuning.

Usage
-----
# 1. Organise raw images into train/val splits
python src/prepare_data.py split \\
    --source_dir data/raw \\
    --output_dir data/interior \\
    --val_ratio 0.1

# 2. Auto-generate captions with BLIP-2
python src/prepare_data.py caption \\
    --dataset_dir data/interior

# 3. Auto-generate object masks with SAM
python src/prepare_data.py mask \\
    --dataset_dir data/interior \\
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \\
    --model_type vit_h
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Split raw images into train / val
# ---------------------------------------------------------------------------

def split_dataset(
    source_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
) -> None:
    """Shuffle and split raw images into ``train/`` and ``val/`` subsets.

    Args:
        source_dir: Directory containing raw images (flat or recursive).
        output_dir: Root output directory; ``train/images`` and ``val/images``
            will be created inside.
        val_ratio: Fraction of images reserved for validation.
        seed: Random seed for reproducibility.
        extensions: Accepted image file extensions.
    """
    source = Path(source_dir)
    all_images: List[Path] = sorted(
        p for p in source.rglob("*") if p.suffix.lower() in extensions
    )
    if not all_images:
        raise RuntimeError(f"No images found in {source_dir}")

    random.seed(seed)
    random.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_ratio))
    splits = {
        "val": all_images[:n_val],
        "train": all_images[n_val:],
    }

    for split, paths in splits.items():
        out_dir = Path(output_dir) / split / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in tqdm(paths, desc=f"Copying {split}"):
            shutil.copy2(src, out_dir / src.name)

    print(f"Split complete – train: {len(splits['train'])}, val: {len(splits['val'])}")


# ---------------------------------------------------------------------------
# Auto-caption with BLIP-2
# ---------------------------------------------------------------------------

def generate_captions(
    dataset_dir: str,
    splits: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 60,
    interior_prefix: str = "An interior design photo of",
) -> None:
    """Generate image captions using BLIP-2 and save them as
    ``captions.json`` inside each split directory.

    Args:
        dataset_dir: Root dataset directory containing split sub-directories.
        splits: Splits to process (default: ``["train", "val"]``).
        device: PyTorch device string.
        batch_size: Number of images per inference batch.
        max_new_tokens: Maximum tokens to generate per caption.
        interior_prefix: Prefix prepended to BLIP-2 conditional generation.
    """
    try:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for caption generation. "
            "Install them with: pip install transformers torch"
        ) from exc

    splits = splits or ["train", "val"]
    root = Path(dataset_dir)

    print("Loading BLIP-2 processor and model …")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()

    for split in splits:
        images_dir = root / split / "images"
        if not images_dir.exists():
            print(f"  Skipping {split} – directory not found.")
            continue

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)
        captions: dict = {}

        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Captioning {split}"):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            prompts = [interior_prefix] * len(images)

            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            for path, output in zip(batch_paths, outputs):
                caption = processor.decode(output, skip_special_tokens=True).strip()
                captions[path.name] = caption

        out_path = root / split / "captions.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(captions, fh, ensure_ascii=False, indent=2)
        print(f"  Saved {len(captions)} captions → {out_path}")


# ---------------------------------------------------------------------------
# Auto-generate object masks with SAM
# ---------------------------------------------------------------------------

def generate_masks(
    dataset_dir: str,
    sam_checkpoint: str,
    model_type: str = "vit_h",
    splits: Optional[List[str]] = None,
    device: str = "cuda",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 2000,
) -> None:
    """Use Segment Anything Model (SAM) to generate per-image object masks
    and save them as PNG files in ``masks/``.

    The function runs SAM in *automatic* mode, picks the largest segment
    that overlaps the image centre (proxy for the main furniture/object),
    and saves it as the inpainting mask.

    Args:
        dataset_dir: Root dataset directory.
        sam_checkpoint: Path to downloaded SAM checkpoint (.pth).
        model_type: SAM model type – ``"vit_h"``, ``"vit_l"``, or ``"vit_b"``.
        splits: Splits to process (default: ``["train", "val"]``).
        device: PyTorch device string.
        points_per_side: SAM grid density; higher → more segments, slower.
        pred_iou_thresh: SAM predicted IoU threshold.
        stability_score_thresh: SAM stability score threshold.
        min_mask_region_area: Minimum mask area in pixels to keep.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "segment-anything is required. Install with: pip install segment-anything"
        ) from exc

    splits = splits or ["train", "val"]
    root = Path(dataset_dir)

    print("Loading SAM model …")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    for split in splits:
        images_dir = root / split / "images"
        masks_dir = root / split / "masks"
        if not images_dir.exists():
            print(f"  Skipping {split} – images directory not found.")
            continue
        masks_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)

        for img_path in tqdm(image_paths, desc=f"Masking {split}"):
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            cx, cy = w // 2, h // 2

            annotations = mask_generator.generate(image_rgb)
            if not annotations:
                continue

            # Pick the segment whose bounding box is closest to image centre
            def dist_to_centre(ann: dict) -> float:
                x, y, bw, bh = ann["bbox"]
                mx, my = x + bw / 2, y + bh / 2
                return (mx - cx) ** 2 + (my - cy) ** 2

            annotations.sort(key=dist_to_centre)
            best_mask = annotations[0]["segmentation"].astype(np.uint8) * 255

            mask_path = masks_dir / (img_path.stem + ".png")
            cv2.imwrite(str(mask_path), best_mask)

        print(f"  Saved masks → {masks_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data preparation utilities for interior inpainting fine-tuning"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── split ──
    p_split = sub.add_parser("split", help="Split raw images into train/val")
    p_split.add_argument("--source_dir", required=True, help="Directory with raw images")
    p_split.add_argument("--output_dir", required=True, help="Root output directory")
    p_split.add_argument("--val_ratio", type=float, default=0.1, help="Validation fraction")
    p_split.add_argument("--seed", type=int, default=42)

    # ── caption ──
    p_cap = sub.add_parser("caption", help="Auto-generate BLIP-2 captions")
    p_cap.add_argument("--dataset_dir", required=True)
    p_cap.add_argument("--splits", nargs="+", default=["train", "val"])
    p_cap.add_argument("--device", default="cuda")
    p_cap.add_argument("--batch_size", type=int, default=8)

    # ── mask ──
    p_mask = sub.add_parser("mask", help="Auto-generate SAM masks")
    p_mask.add_argument("--dataset_dir", required=True)
    p_mask.add_argument("--sam_checkpoint", required=True, help="Path to SAM .pth checkpoint")
    p_mask.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p_mask.add_argument("--splits", nargs="+", default=["train", "val"])
    p_mask.add_argument("--device", default="cuda")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "split":
        split_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    elif args.command == "caption":
        generate_captions(
            dataset_dir=args.dataset_dir,
            splits=args.splits,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "mask":
        generate_masks(
            dataset_dir=args.dataset_dir,
            sam_checkpoint=args.sam_checkpoint,
            model_type=args.model_type,
            splits=args.splits,
            device=args.device,
        )


if __name__ == "__main__":
    main()

"""
Interior Inpainting Dataset
============================
Expects the following directory layout inside ``dataset_dir``:

    dataset_dir/
    ├── train/
    │   ├── images/          # original RGB images  (*.jpg / *.png)
    │   ├── masks/           # binary masks 0=keep 255=inpaint (*.png)
    │   └── captions.json    # {"image_name.jpg": "caption text", ...}
    └── val/
        ├── images/
        ├── masks/
        └── captions.json

If ``masks/`` does not exist the dataset generates random masks on-the-fly.
If ``captions.json`` does not exist a default caption is used.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Mask generators
# ---------------------------------------------------------------------------

def _random_bbox_mask(
    height: int,
    width: int,
    min_area: float = 0.05,
    max_area: float = 0.50,
) -> np.ndarray:
    """Return a binary mask (H×W uint8) with a single rectangular hole."""
    mask = np.zeros((height, width), dtype=np.uint8)
    area = random.uniform(min_area, max_area) * height * width
    aspect = random.uniform(0.3, 3.0)
    h = int(np.sqrt(area / aspect))
    w = int(aspect * h)
    h, w = min(h, height - 1), min(w, width - 1)
    top = random.randint(0, height - h)
    left = random.randint(0, width - w)
    mask[top : top + h, left : left + w] = 255
    return mask


def _random_irregular_mask(
    height: int,
    width: int,
    min_area: float = 0.05,
    max_area: float = 0.50,
    num_strokes: int = 10,
) -> np.ndarray:
    """Return an irregular free-form mask drawn with random brush strokes."""
    mask = np.zeros((height, width), dtype=np.uint8)
    target_area = random.uniform(min_area, max_area) * height * width

    painted = 0
    attempts = 0
    while painted < target_area and attempts < 200:
        attempts += 1
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        length = random.randint(20, min(height, width) // 3)
        angle = random.uniform(0, 2 * np.pi)
        brush = random.randint(max(1, min(height, width) // 30), min(height, width) // 10)
        for _ in range(length):
            dx = int(brush * np.cos(angle))
            dy = int(brush * np.sin(angle))
            x1, y1 = max(0, x - dx), max(0, y - dy)
            x2, y2 = min(width - 1, x + dx), min(height - 1, y + dy)
            cv2.line(mask, (x1, y1), (x2, y2), 255, brush)
            angle += random.uniform(-0.4, 0.4)
            x = np.clip(x + int(np.cos(angle) * 5), 0, width - 1)
            y = np.clip(y + int(np.sin(angle) * 5), 0, height - 1)
        painted = np.sum(mask > 0)

    return mask


def generate_mask(
    height: int,
    width: int,
    mask_type: str = "mixed",
    min_area: float = 0.05,
    max_area: float = 0.50,
) -> np.ndarray:
    """Generate a random inpainting mask.

    Args:
        height: Image height.
        width: Image width.
        mask_type: One of ``"bbox"``, ``"irregular"``, or ``"mixed"``.
        min_area: Minimum fraction of pixels to mask.
        max_area: Maximum fraction of pixels to mask.

    Returns:
        Binary mask as a uint8 numpy array (H×W), values in {0, 255}.
    """
    if mask_type == "bbox":
        return _random_bbox_mask(height, width, min_area, max_area)
    if mask_type == "irregular":
        return _random_irregular_mask(height, width, min_area, max_area)
    # mixed: 50/50 chance
    if random.random() < 0.5:
        return _random_bbox_mask(height, width, min_area, max_area)
    return _random_irregular_mask(height, width, min_area, max_area)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InteriorInpaintingDataset(Dataset):
    """PyTorch Dataset for Stable Diffusion inpainting fine-tuning on
    interior-design images.

    Each sample returns a dict with:
        pixel_values   – (3, H, W) float32, range [-1, 1]  – original image
        masked_image   – (3, H, W) float32, range [-1, 1]  – image ⊙ (1-mask)
        mask           – (1, H, W) float32, range [0, 1]   – 1 = inpaint area
        input_ids      – (77,) int64                       – tokenised caption
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        tokenizer=None,
        size: int = 512,
        mask_type: str = "mixed",
        mask_min_area: float = 0.05,
        mask_max_area: float = 0.50,
        center_crop: bool = False,
        random_flip: bool = True,
        default_caption: str = "A high-quality interior design photo",
        augment: bool = True,
    ) -> None:
        self.size = size
        self.tokenizer = tokenizer
        self.mask_type = mask_type
        self.mask_min_area = mask_min_area
        self.mask_max_area = mask_max_area
        self.random_flip = random_flip and (split == "train")
        self.augment = augment and (split == "train")
        self.default_caption = default_caption

        root = Path(dataset_dir) / split
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        self.captions_path = root / "captions.json"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.image_paths: List[Path] = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in extensions
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

        # Optional: pre-computed masks
        self.use_precomputed_masks = self.masks_dir.exists()

        # Optional: per-image captions
        self.captions: Dict[str, str] = {}
        if self.captions_path.exists():
            with open(self.captions_path, "r", encoding="utf-8") as fh:
                self.captions = json.load(fh)

        # Image transforms (no normalisation yet – applied later)
        resize_crop: List[Callable] = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            resize_crop.append(transforms.CenterCrop(size))
        else:
            resize_crop.append(transforms.RandomCrop(size))
        self.spatial_transform = transforms.Compose(resize_crop)

        self.to_tensor = transforms.ToTensor()   # [0,1]
        self.normalise = transforms.Normalize([0.5], [0.5])  # → [-1,1]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return self.spatial_transform(img)

    def _load_mask(self, image_path: Path, size: Tuple[int, int]) -> np.ndarray:
        """Return a (H, W) uint8 mask – 255 = region to inpaint."""
        if self.use_precomputed_masks:
            mask_path = self.masks_dir / (image_path.stem + ".png")
            if mask_path.exists():
                m = np.array(Image.open(mask_path).convert("L").resize((size[1], size[0]), Image.NEAREST))
                _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
                return m
        return generate_mask(
            size[0], size[1],
            mask_type=self.mask_type,
            min_area=self.mask_min_area,
            max_area=self.mask_max_area,
        )

    def _get_caption(self, image_path: Path) -> str:
        caption = self.captions.get(image_path.name) or self.captions.get(image_path.stem)
        return caption if caption else self.default_caption

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]

        # ---- Image ----
        pil_image = self._load_image(image_path)

        # Optional colour jitter augmentation
        if self.augment:
            pil_image = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )(pil_image)

        # Optional horizontal flip (applied to both image and mask)
        do_flip = self.random_flip and random.random() < 0.5

        if do_flip:
            pil_image = transforms.functional.hflip(pil_image)

        image_np = np.array(pil_image)          # (H, W, 3) uint8

        # ---- Mask ----
        h, w = image_np.shape[:2]
        mask_np = self._load_mask(image_path, (h, w))   # (H, W) uint8

        if do_flip:
            mask_np = np.fliplr(mask_np).copy()

        # ---- Masked image (grey-out the inpaint region) ----
        mask_3c = mask_np[:, :, None] / 255.0             # (H, W, 1) float
        masked_image_np = (image_np * (1.0 - mask_3c)).astype(np.uint8)

        # ---- To tensors ----
        pixel_values = self.normalise(self.to_tensor(pil_image))          # [-1,1]
        masked_image = self.normalise(self.to_tensor(Image.fromarray(masked_image_np)))

        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0) / 255.0  # [0,1]

        # ---- Caption → token IDs ----
        caption = self._get_caption(image_path)
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)
        else:
            input_ids = torch.zeros(77, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "masked_image": masked_image,
            "mask": mask_tensor,
            "input_ids": input_ids,
            "caption": caption,
        }

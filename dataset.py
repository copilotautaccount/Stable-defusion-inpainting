"""
Dataset utilities for Stable Diffusion Inpainting fine-tuning.

Expected directory layout:
    data_dir/
        images/       – original RGB images  (*.png / *.jpg)
        masks/        – binary masks          (*.png / *.jpg)
        prompts.txt   – one text prompt per line, aligned with sorted image list
                        (optional – falls back to an empty string when absent)
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_images(folder: Path) -> List[Path]:
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {folder}")
    return paths


class InpaintingDataset(Dataset):
    """
    Pairs original images with binary masks and optional text prompts for
    inpainting training.

    Args:
        data_dir:   Root directory containing ``images/``, ``masks/``, and
                    optionally ``prompts.txt``.
        image_size: Target (width, height) to resize images and masks.
        tokenizer:  HuggingFace tokenizer used to encode prompts.
        augment:    Whether to apply random horizontal flips.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        tokenizer=None,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.augment = augment

        self.image_paths = _collect_images(self.data_dir / "images")

        mask_dir = self.data_dir / "masks"
        self.mask_paths = _collect_images(mask_dir)

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) does not match "
                f"number of masks ({len(self.mask_paths)})"
            )

        prompts_file = self.data_dir / "prompts.txt"
        if prompts_file.exists():
            with prompts_file.open("r", encoding="utf-8") as f:
                self.prompts: List[str] = [line.rstrip("\n") for line in f]
            if len(self.prompts) != len(self.image_paths):
                raise ValueError(
                    f"prompts.txt has {len(self.prompts)} lines but there are "
                    f"{len(self.image_paths)} images"
                )
        else:
            self.prompts = [""] * len(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize(self.image_size, Image.LANCZOS)

    def _load_mask(self, path: Path) -> Image.Image:
        mask = Image.open(path).convert("L").resize(self.image_size, Image.NEAREST)
        arr = np.array(mask)
        arr = (arr > 127).astype(np.uint8) * 255
        return Image.fromarray(arr, mode="L")

    def __getitem__(self, idx: int):
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        prompt = self.prompts[idx]

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_np = np.array(image).astype(np.float32) / 127.5 - 1.0
        mask_np = np.array(mask).astype(np.float32) / 255.0

        masked_image_np = image_np * (1 - mask_np[..., np.newaxis])

        sample = {
            "pixel_values": torch.from_numpy(image_np).permute(2, 0, 1),
            "mask_values": torch.from_numpy(mask_np).unsqueeze(0),
            "masked_pixel_values": torch.from_numpy(masked_image_np).permute(2, 0, 1),
            "prompt": prompt,
        }

        if self.tokenizer is not None:
            tokens = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            sample["input_ids"] = tokens.input_ids.squeeze(0)

        return sample

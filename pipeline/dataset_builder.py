"""Orchestrate the full dataset-building pipeline."""

import json
import logging
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from pipeline.caption_generator import CaptionGenerator
from pipeline.mask_generator import MaskGenerator

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class DatasetBuilder:
    """Build an inpainting dataset from paired background / object folders.

    Expected input layout::

        <dataset_root>/
            background/   # images *without* the added object
            object/       # images *with* the added object (same filenames)

    Output layout::

        <output_dir>/
            images/       # copies of the object images
            masks/        # binary masks (white = inpaint region)
            captions/     # per-image .txt caption files
            metadata.json # full index of the dataset
    """

    def __init__(
        self,
        dataset_root: str,
        output_dir: str,
        caption_mode: str = "simple",
        mask_kwargs: dict | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.bg_dir = self.dataset_root / "background"
        self.obj_dir = self.dataset_root / "object"

        self.mask_gen = MaskGenerator(**(mask_kwargs or {}))
        self.caption_gen = CaptionGenerator(mode=caption_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> list[dict]:
        """Run the full pipeline and return metadata records.

        Returns:
            A list of dicts, one per image pair, with keys
            ``image``, ``mask``, ``caption``, ``source_background``,
            ``source_object``.
        """
        self._validate_dirs()
        pairs = self._match_pairs()
        logger.info("Found %d image pairs.", len(pairs))
        if not pairs:
            logger.warning("No matching image pairs found – nothing to do.")
            return []

        # Prepare output dirs
        imgs_dir = self.output_dir / "images"
        masks_dir = self.output_dir / "masks"
        caps_dir = self.output_dir / "captions"
        for d in (imgs_dir, masks_dir, caps_dir):
            d.mkdir(parents=True, exist_ok=True)

        metadata: list[dict] = []

        for bg_path, obj_path in tqdm(pairs, desc="Processing pairs"):
            stem = bg_path.stem

            # 1. Generate mask
            mask = self.mask_gen.generate_mask_from_paths(
                str(bg_path), str(obj_path)
            )

            # 2. Generate caption
            obj_img = cv2.imread(str(obj_path))
            caption = self.caption_gen.generate_caption(obj_img, mask)

            # 3. Save outputs
            out_img = imgs_dir / obj_path.name
            out_mask = masks_dir / f"{stem}.png"
            out_cap = caps_dir / f"{stem}.txt"

            cv2.imwrite(str(out_img), obj_img)
            cv2.imwrite(str(out_mask), mask)
            out_cap.write_text(caption, encoding="utf-8")

            record = {
                "image": str(out_img.relative_to(self.output_dir)),
                "mask": str(out_mask.relative_to(self.output_dir)),
                "caption": caption,
                "source_background": str(bg_path),
                "source_object": str(obj_path),
            }
            metadata.append(record)
            logger.debug("Processed: %s", stem)

        # 4. Write metadata
        meta_path = self.output_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Dataset built successfully – %d samples in %s",
            len(metadata),
            self.output_dir,
        )
        return metadata

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dirs(self):
        if not self.bg_dir.is_dir():
            raise FileNotFoundError(
                f"Background directory not found: {self.bg_dir}"
            )
        if not self.obj_dir.is_dir():
            raise FileNotFoundError(
                f"Object directory not found: {self.obj_dir}"
            )

    def _match_pairs(self) -> list[tuple[Path, Path]]:
        """Return sorted list of (background, object) path pairs."""
        bg_files = {
            f.stem: f
            for f in self.bg_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        }
        obj_files = {
            f.stem: f
            for f in self.obj_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        }

        common = sorted(set(bg_files) & set(obj_files))
        if not common:
            logger.warning(
                "No matching filenames between %s and %s", self.bg_dir, self.obj_dir
            )
        return [(bg_files[name], obj_files[name]) for name in common]

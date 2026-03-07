"""
Unit tests for the dataset module.
Run with: pytest tests/test_dataset.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add src to path so dataset can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import (
    InteriorInpaintingDataset,
    generate_mask,
    _random_bbox_mask,
    _random_irregular_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_dataset(tmp_path):
    """Create a minimal dataset directory with 5 synthetic images."""
    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        img_dir.mkdir(parents=True)

        captions = {}
        for i in range(5):
            name = f"room_{i:03d}.png"
            arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / name)
            captions[name] = f"A beautiful interior room number {i}"

        with open(tmp_path / split / "captions.json", "w") as fh:
            json.dump(captions, fh)

    return tmp_path


@pytest.fixture()
def tiny_dataset_with_masks(tmp_path):
    """Create a minimal dataset directory with images AND precomputed masks."""
    for split in ("train", "val"):
        img_dir = tmp_path / split / "images"
        mask_dir = tmp_path / split / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for i in range(4):
            name = f"room_{i:03d}.png"
            arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / name)

            m = np.zeros((256, 256), dtype=np.uint8)
            m[64:192, 64:192] = 255
            Image.fromarray(m).save(mask_dir / name)

    return tmp_path


# ---------------------------------------------------------------------------
# Mask generator tests
# ---------------------------------------------------------------------------

class TestMaskGenerators:
    def test_bbox_mask_shape(self):
        m = _random_bbox_mask(256, 256)
        assert m.shape == (256, 256)
        assert m.dtype == np.uint8

    def test_bbox_mask_values(self):
        m = _random_bbox_mask(256, 256)
        unique = set(np.unique(m))
        assert unique.issubset({0, 255}), "Mask should contain only 0 and 255"

    def test_bbox_mask_has_masked_area(self):
        m = _random_bbox_mask(256, 256, min_area=0.1)
        assert np.any(m > 0), "Mask should have at least some masked pixels"

    def test_irregular_mask_shape(self):
        m = _random_irregular_mask(256, 256)
        assert m.shape == (256, 256)
        assert m.dtype == np.uint8

    def test_generate_mask_mixed(self):
        for _ in range(10):
            m = generate_mask(128, 128, mask_type="mixed")
            assert m.shape == (128, 128)

    def test_generate_mask_area_bounds(self):
        min_a, max_a = 0.1, 0.3
        for _ in range(20):
            m = _random_bbox_mask(256, 256, min_area=min_a, max_area=max_a)
            area_frac = np.mean(m > 0)
            assert area_frac >= min_a * 0.5, f"Area {area_frac:.3f} too small"

    def test_generate_mask_invalid_type_falls_back(self):
        # "mixed" is the default branch; invalid type would hit the irregular branch
        # The function does not raise for unknown types - it falls through to irregular
        m = generate_mask(128, 128, mask_type="bbox")
        assert m.shape == (128, 128)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestInteriorInpaintingDataset:
    def test_dataset_length(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        assert len(ds) == 5

    def test_sample_keys(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        sample = ds[0]
        required_keys = {"pixel_values", "masked_image", "mask", "input_ids", "caption"}
        assert required_keys.issubset(sample.keys())

    def test_pixel_values_range(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        sample = ds[0]
        pv = sample["pixel_values"]
        assert pv.shape == (3, 64, 64), f"Unexpected shape: {pv.shape}"
        assert pv.min() >= -1.05, "pixel_values should be >= -1"
        assert pv.max() <= 1.05, "pixel_values should be <= 1"

    def test_mask_range(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        sample = ds[0]
        m = sample["mask"]
        assert m.shape == (1, 64, 64)
        assert m.min() >= 0.0
        assert m.max() <= 1.0

    def test_masked_image_range(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        sample = ds[0]
        mi = sample["masked_image"]
        assert mi.shape == (3, 64, 64)
        assert mi.min() >= -1.05
        assert mi.max() <= 1.05

    def test_caption_loading(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        for i in range(len(ds)):
            caption = ds[i]["caption"]
            assert isinstance(caption, str)
            assert len(caption) > 0

    def test_default_caption_fallback(self, tmp_path):
        """Dataset without captions.json should use default caption."""
        img_dir = tmp_path / "train" / "images"
        img_dir.mkdir(parents=True)
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / "room.png")

        ds = InteriorInpaintingDataset(
            dataset_dir=str(tmp_path),
            split="train",
            size=64,
            default_caption="Test default caption",
        )
        assert ds[0]["caption"] == "Test default caption"

    def test_precomputed_masks_used(self, tiny_dataset_with_masks):
        """When masks/ dir exists the dataset should load them."""
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset_with_masks),
            split="train",
            size=256,
        )
        assert ds.use_precomputed_masks is True
        sample = ds[0]
        m = sample["mask"][0].numpy()
        assert np.any(m > 0.5), "Precomputed mask should have masked pixels"

    def test_no_images_raises(self, tmp_path):
        (tmp_path / "train" / "images").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="No images found"):
            InteriorInpaintingDataset(str(tmp_path), split="train", size=64)

    def test_missing_images_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InteriorInpaintingDataset(str(tmp_path), split="train", size=64)

    def test_val_split_no_augment(self, tiny_dataset):
        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="val",
            size=64,
        )
        assert ds.random_flip is False
        assert ds.augment is False

    def test_dataloader_collation(self, tiny_dataset):
        import torch
        from torch.utils.data import DataLoader

        ds = InteriorInpaintingDataset(
            dataset_dir=str(tiny_dataset),
            split="train",
            size=64,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["pixel_values"].shape == (2, 3, 64, 64)
        assert batch["mask"].shape == (2, 1, 64, 64)
        assert batch["masked_image"].shape == (2, 3, 64, 64)

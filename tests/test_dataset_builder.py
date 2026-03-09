"""Unit tests for the dataset_builder module."""

import json

import cv2
import numpy as np
import pytest

from pipeline.dataset_builder import DatasetBuilder


def _create_test_dataset(root, pairs=3, h=200, w=300):
    """Create a tiny synthetic dataset for testing."""
    bg_dir = root / "background"
    obj_dir = root / "object"
    bg_dir.mkdir(parents=True)
    obj_dir.mkdir(parents=True)

    for i in range(pairs):
        bg = np.full((h, w, 3), 120, dtype=np.uint8)
        obj = bg.copy()
        x1, y1 = 50 + i * 10, 30 + i * 10
        cv2.rectangle(obj, (x1, y1), (x1 + 60, y1 + 60), (255, 255, 255), cv2.FILLED)

        cv2.imwrite(str(bg_dir / f"img_{i:03d}.png"), bg)
        cv2.imwrite(str(obj_dir / f"img_{i:03d}.png"), obj)


class TestDatasetBuilder:
    def test_build_produces_outputs(self, tmp_path):
        ds_root = tmp_path / "dataset"
        out_dir = tmp_path / "output"
        _create_test_dataset(ds_root, pairs=2)

        builder = DatasetBuilder(str(ds_root), str(out_dir))
        metadata = builder.build()

        assert len(metadata) == 2
        assert (out_dir / "images").is_dir()
        assert (out_dir / "masks").is_dir()
        assert (out_dir / "captions").is_dir()
        assert (out_dir / "metadata.json").is_file()

    def test_metadata_json_valid(self, tmp_path):
        ds_root = tmp_path / "dataset"
        out_dir = tmp_path / "output"
        _create_test_dataset(ds_root, pairs=1)

        builder = DatasetBuilder(str(ds_root), str(out_dir))
        builder.build()

        with open(out_dir / "metadata.json") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        record = data[0]
        assert "image" in record
        assert "mask" in record
        assert "caption" in record

    def test_masks_are_binary(self, tmp_path):
        ds_root = tmp_path / "dataset"
        out_dir = tmp_path / "output"
        _create_test_dataset(ds_root, pairs=1)

        builder = DatasetBuilder(str(ds_root), str(out_dir))
        builder.build()

        mask_files = list((out_dir / "masks").glob("*.png"))
        assert len(mask_files) == 1
        mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}

    def test_missing_background_dir(self, tmp_path):
        ds_root = tmp_path / "dataset"
        out_dir = tmp_path / "output"
        (ds_root / "object").mkdir(parents=True)

        builder = DatasetBuilder(str(ds_root), str(out_dir))
        with pytest.raises(FileNotFoundError):
            builder.build()

    def test_no_matching_pairs(self, tmp_path):
        ds_root = tmp_path / "dataset"
        out_dir = tmp_path / "output"
        bg_dir = ds_root / "background"
        obj_dir = ds_root / "object"
        bg_dir.mkdir(parents=True)
        obj_dir.mkdir(parents=True)

        # Different filenames
        bg = np.full((100, 100, 3), 120, dtype=np.uint8)
        cv2.imwrite(str(bg_dir / "a.png"), bg)
        cv2.imwrite(str(obj_dir / "b.png"), bg)

        builder = DatasetBuilder(str(ds_root), str(out_dir))
        metadata = builder.build()
        assert len(metadata) == 0

    def test_caption_files_created(self, tmp_path):
        ds_root = tmp_path / "dataset"
        out_dir = tmp_path / "output"
        _create_test_dataset(ds_root, pairs=2)

        builder = DatasetBuilder(str(ds_root), str(out_dir))
        builder.build()

        cap_files = list((out_dir / "captions").glob("*.txt"))
        assert len(cap_files) == 2
        for f in cap_files:
            text = f.read_text()
            assert len(text) > 0

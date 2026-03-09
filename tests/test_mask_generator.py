"""Unit tests for the mask_generator module."""

import cv2
import numpy as np
import pytest

from pipeline.mask_generator import MaskGenerator


@pytest.fixture
def mask_gen():
    return MaskGenerator()


def _make_pair(h=200, w=300):
    """Create a synthetic background/object pair with a known rectangle."""
    bg = np.full((h, w, 3), 120, dtype=np.uint8)  # uniform gray
    obj = bg.copy()
    # Draw a white rectangle as the "added object"
    cv2.rectangle(obj, (100, 50), (200, 150), (255, 255, 255), cv2.FILLED)
    return bg, obj


class TestMaskGenerator:
    def test_mask_shape_matches_input(self, mask_gen):
        bg, obj = _make_pair()
        mask = mask_gen.generate_mask(bg, obj)
        assert mask.shape == (200, 300)

    def test_mask_is_binary(self, mask_gen):
        bg, obj = _make_pair()
        mask = mask_gen.generate_mask(bg, obj)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}

    def test_mask_detects_added_object(self, mask_gen):
        bg, obj = _make_pair()
        mask = mask_gen.generate_mask(bg, obj)
        # The rectangle region (100:200, 50:150) should be largely white
        roi = mask[50:150, 100:200]
        white_ratio = np.count_nonzero(roi) / roi.size
        assert white_ratio > 0.7, f"Expected most of the ROI to be white, got {white_ratio:.2f}"

    def test_identical_images_produce_empty_mask(self, mask_gen):
        bg, _ = _make_pair()
        mask = mask_gen.generate_mask(bg, bg.copy())
        assert np.count_nonzero(mask) == 0

    def test_size_mismatch_is_handled(self, mask_gen):
        bg = np.full((200, 300, 3), 120, dtype=np.uint8)
        obj = np.full((210, 310, 3), 120, dtype=np.uint8)
        cv2.rectangle(obj, (100, 50), (200, 150), (255, 255, 255), cv2.FILLED)
        mask = mask_gen.generate_mask(bg, obj)
        assert mask.shape == (200, 300)

    def test_custom_parameters(self):
        gen = MaskGenerator(blur_kernel=7, threshold=30, morph_kernel=7, min_area=50)
        bg, obj = _make_pair()
        mask = gen.generate_mask(bg, obj)
        assert mask.shape == (200, 300)

    def test_generate_mask_from_paths_missing_file(self, mask_gen, tmp_path):
        with pytest.raises(FileNotFoundError):
            mask_gen.generate_mask_from_paths(
                str(tmp_path / "missing.png"), str(tmp_path / "also_missing.png")
            )

    def test_generate_mask_from_paths(self, mask_gen, tmp_path):
        bg, obj = _make_pair()
        bg_path = str(tmp_path / "bg.png")
        obj_path = str(tmp_path / "obj.png")
        cv2.imwrite(bg_path, bg)
        cv2.imwrite(obj_path, obj)

        mask = mask_gen.generate_mask_from_paths(bg_path, obj_path)
        assert mask.shape == (200, 300)
        assert np.count_nonzero(mask) > 0

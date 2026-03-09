"""Unit tests for the caption_generator module."""

import cv2
import numpy as np
import pytest

from pipeline.caption_generator import CaptionGenerator, _dominant_colour


@pytest.fixture
def caption_gen():
    return CaptionGenerator(mode="simple")


def _make_image_and_mask(h=300, w=400, rect=(150, 100, 250, 200), colour=(0, 0, 255)):
    """Create an image with a coloured rectangle and the matching mask."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), colour, cv2.FILLED)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (rect[0], rect[1]), (rect[2], rect[3]), 255, cv2.FILLED)
    return img, mask


class TestCaptionGenerator:
    def test_simple_returns_string(self, caption_gen):
        img, mask = _make_image_and_mask()
        caption = caption_gen.generate_caption(img, mask)
        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_simple_contains_position(self, caption_gen):
        # Object in the center
        img, mask = _make_image_and_mask(rect=(150, 100, 250, 200))
        caption = caption_gen.generate_caption(img, mask)
        assert "center" in caption

    def test_simple_small_object(self, caption_gen):
        img, mask = _make_image_and_mask(h=500, w=500, rect=(200, 200, 220, 220))
        caption = caption_gen.generate_caption(img, mask)
        assert "small" in caption

    def test_empty_mask(self, caption_gen):
        img = np.full((200, 300, 3), 120, dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        caption = caption_gen.generate_caption(img, mask)
        assert isinstance(caption, str)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown caption mode"):
            CaptionGenerator(mode="invalid")


class TestDominantColour:
    def test_red_object(self):
        img = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)  # BGR red
        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert _dominant_colour(img, mask) == "red"

    def test_green_object(self):
        img = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)  # BGR green
        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert _dominant_colour(img, mask) == "green"

    def test_blue_object(self):
        img = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)  # BGR blue
        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert _dominant_colour(img, mask) == "blue"

    def test_dark_object(self):
        img = np.full((100, 100, 3), (10, 10, 10), dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert _dominant_colour(img, mask) == "dark"

    def test_white_object(self):
        img = np.full((100, 100, 3), (255, 255, 255), dtype=np.uint8)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        assert _dominant_colour(img, mask) == "white"

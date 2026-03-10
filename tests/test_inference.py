"""
Unit tests for the inference module.
Run with: pytest tests/test_inference.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Add src to path so inference can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import (
    _IMAGE_EXTENSIONS,
    _load_image,
    _load_mask,
    _pick_random_image,
    _show_results,
    manual_inference,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def image_dir(tmp_path):
    """Create a temporary directory with 5 synthetic images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"room_{i:03d}.png")
    return img_dir


@pytest.fixture()
def single_image(tmp_path):
    """Create a single synthetic image and return its path."""
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    path = tmp_path / "test_image.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture()
def single_mask(tmp_path):
    """Create a single synthetic mask and return its path."""
    arr = np.zeros((128, 128), dtype=np.uint8)
    arr[32:96, 32:96] = 255
    path = tmp_path / "test_mask.png"
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# _load_image tests
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_loads_rgb(self, single_image):
        img = _load_image(str(single_image))
        assert img.mode == "RGB"

    def test_resizes(self, single_image):
        img = _load_image(str(single_image), size=64)
        assert img.size == (64, 64)

    def test_no_resize(self, single_image):
        img = _load_image(str(single_image))
        assert img.size == (128, 128)


# ---------------------------------------------------------------------------
# _load_mask tests
# ---------------------------------------------------------------------------

class TestLoadMask:
    def test_loads_grayscale(self, single_mask):
        m = _load_mask(str(single_mask))
        assert m.mode == "L"

    def test_resizes(self, single_mask):
        m = _load_mask(str(single_mask), size=64)
        assert m.size == (64, 64)


# ---------------------------------------------------------------------------
# _pick_random_image tests
# ---------------------------------------------------------------------------

class TestPickRandomImage:
    def test_returns_valid_image(self, image_dir):
        result = _pick_random_image(str(image_dir))
        assert result.exists()
        assert result.suffix.lower() in _IMAGE_EXTENSIONS

    def test_picks_from_directory(self, image_dir):
        result = _pick_random_image(str(image_dir))
        assert result.parent == image_dir

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _pick_random_image(str(tmp_path / "nonexistent"))

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="No images found"):
            _pick_random_image(str(empty))

    def test_ignores_non_image_files(self, tmp_path):
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "readme.txt").write_text("hello")
        (d / "data.csv").write_text("a,b,c")
        with pytest.raises(RuntimeError, match="No images found"):
            _pick_random_image(str(d))

    def test_random_selection(self, image_dir):
        """Picking multiple times should eventually yield different images."""
        picks = {_pick_random_image(str(image_dir)).name for _ in range(50)}
        assert len(picks) > 1, "Should pick different images (probabilistic)"


# ---------------------------------------------------------------------------
# _show_results tests (non-interactive – save only)
# ---------------------------------------------------------------------------

class TestShowResults:
    def test_show_results_saves_file(self, tmp_path):
        """_show_results saves a comparison figure to disk."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        original = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        mask = Image.fromarray(
            np.zeros((64, 64), dtype=np.uint8), mode="L"
        )
        result = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )

        save_path = str(tmp_path / "comparison.png")

        # Patch plt.show to prevent window and matplotlib.use to keep Agg
        with patch("matplotlib.pyplot.show"):
            _show_results(original, mask, [result], save_path=save_path)

        assert Path(save_path).exists()

    def test_show_results_multiple_results(self, tmp_path):
        """_show_results handles multiple result images."""
        import matplotlib
        matplotlib.use("Agg")

        original = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        mask = Image.fromarray(
            np.zeros((64, 64), dtype=np.uint8), mode="L"
        )
        results = [
            Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            for _ in range(3)
        ]

        save_path = str(tmp_path / "multi.png")
        with patch("matplotlib.pyplot.show"):
            _show_results(original, mask, results, save_path=save_path)

        assert Path(save_path).exists()


# ---------------------------------------------------------------------------
# parse_args tests
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_manual_mode_no_prompt_required(self):
        args = parse_args([
            "--model_dir", "some/model",
            "--manual",
            "--image_dir", "some/dir",
        ])
        assert args.manual is True
        assert args.prompt is None

    def test_single_mode_requires_prompt(self):
        with pytest.raises(SystemExit):
            parse_args([
                "--model_dir", "some/model",
                "--image", "img.png",
                "--mask", "mask.png",
            ])

    def test_single_mode_with_prompt(self):
        args = parse_args([
            "--model_dir", "some/model",
            "--image", "img.png",
            "--mask", "mask.png",
            "--prompt", "a nice room",
        ])
        assert args.manual is False
        assert args.prompt == "a nice room"

    def test_batch_mode_with_prompt(self):
        args = parse_args([
            "--model_dir", "some/model",
            "--image_dir", "imgs/",
            "--mask_dir", "masks/",
            "--prompt", "a nice room",
        ])
        assert args.image_dir == "imgs/"
        assert args.mask_dir == "masks/"

    def test_defaults(self):
        args = parse_args([
            "--model_dir", "model",
            "--manual",
            "--image_dir", "dir",
        ])
        assert args.seed == 42
        assert args.num_inference_steps == 50
        assert args.guidance_scale == 7.5
        assert args.strength == 0.99
        assert args.resolution == 512
        assert args.num_images == 1
        assert args.output == "output.png"

    def test_manual_flag_default_is_false(self):
        args = parse_args([
            "--model_dir", "model",
            "--prompt", "test",
            "--image", "img.png",
            "--mask", "mask.png",
        ])
        assert args.manual is False


# ---------------------------------------------------------------------------
# manual_inference tests (mocked pipeline)
# ---------------------------------------------------------------------------

class TestManualInference:
    def test_empty_mask_exits_early(self, image_dir, tmp_path):
        """If user draws nothing the function should return without error."""
        mock_pipe = MagicMock()

        # Mock _draw_mask_interactive to return an empty mask
        empty_mask = Image.fromarray(
            np.zeros((512, 512), dtype=np.uint8), mode="L"
        )
        with patch("inference._draw_mask_interactive", return_value=empty_mask), \
             patch("builtins.input", return_value="some prompt"):
            manual_inference(
                pipe=mock_pipe,
                image_dir=str(image_dir),
                resolution=512,
                negative_prompt="bad",
                num_inference_steps=1,
                guidance_scale=7.5,
                strength=0.99,
                seed=42,
                num_images=1,
                output=str(tmp_path / "out.png"),
            )
        # Pipeline should NOT have been called
        mock_pipe.assert_not_called()

    def test_runs_full_pipeline(self, image_dir, tmp_path):
        """With a valid mask and prompt, should call pipeline and save."""
        # Mock pipeline to return a dummy image
        result_img = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        mock_pipe = MagicMock()
        mock_pipe.device = "cpu"
        mock_pipe.return_value.images = [result_img]

        # Non-empty mask
        mask_arr = np.zeros((512, 512), dtype=np.uint8)
        mask_arr[100:400, 100:400] = 255
        drawn_mask = Image.fromarray(mask_arr, mode="L")

        out_path = tmp_path / "result.png"

        with patch("inference._draw_mask_interactive", return_value=drawn_mask), \
             patch("inference._show_results"), \
             patch("builtins.input", return_value="a bright room"):
            manual_inference(
                pipe=mock_pipe,
                image_dir=str(image_dir),
                resolution=512,
                negative_prompt="bad",
                num_inference_steps=1,
                guidance_scale=7.5,
                strength=0.99,
                seed=42,
                num_images=1,
                output=str(out_path),
            )

        # Pipeline called once
        mock_pipe.assert_called_once()
        # Output saved
        assert out_path.exists()

    def test_default_prompt_on_empty_input(self, image_dir, tmp_path):
        """If user enters empty prompt, default should be used."""
        result_img = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        mock_pipe = MagicMock()
        mock_pipe.device = "cpu"
        mock_pipe.return_value.images = [result_img]

        mask_arr = np.zeros((512, 512), dtype=np.uint8)
        mask_arr[100:400, 100:400] = 255
        drawn_mask = Image.fromarray(mask_arr, mode="L")

        with patch("inference._draw_mask_interactive", return_value=drawn_mask), \
             patch("inference._show_results"), \
             patch("builtins.input", return_value=""):
            manual_inference(
                pipe=mock_pipe,
                image_dir=str(image_dir),
                resolution=512,
                negative_prompt="bad",
                num_inference_steps=1,
                guidance_scale=7.5,
                strength=0.99,
                seed=42,
                num_images=1,
                output=str(tmp_path / "out.png"),
            )

        # Check the prompt passed to the pipeline
        call_kwargs = mock_pipe.call_args
        assert call_kwargs is not None
        # The prompt should be the default
        prompt_used = call_kwargs[1].get("prompt") or call_kwargs[0][2]
        assert "interior" in prompt_used.lower() or "high-quality" in prompt_used.lower()

    def test_missing_image_dir_raises(self, tmp_path):
        """manual_inference should fail if image_dir doesn't exist."""
        mock_pipe = MagicMock()
        with pytest.raises(FileNotFoundError):
            manual_inference(
                pipe=mock_pipe,
                image_dir=str(tmp_path / "nonexistent"),
                resolution=512,
                negative_prompt="bad",
                num_inference_steps=1,
                guidance_scale=7.5,
                strength=0.99,
                seed=42,
                num_images=1,
                output=str(tmp_path / "out.png"),
            )

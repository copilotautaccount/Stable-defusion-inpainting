"""
Unit tests for prepare_data.py – model selection CLI, split utility, and
the _pick_best_mask / generate_captions / generate_masks public API.

Run with: pytest tests/test_prepare_data.py -v
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import prepare_data as pd_mod
from prepare_data import (
    _DEFAULT_FURNITURE_LABELS,
    _pick_best_mask,
    _torch_dtype_kwarg,
    _transformers_version,
    generate_captions,
    generate_masks,
    split_dataset,
    _build_parser,
    _image_paths,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_random_image(path: Path, size: int = 64) -> None:
    """Save a synthetic random RGB PNG to *path*."""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_images(root: Path, split: str, n: int = 4) -> Path:
    """Create n synthetic 64×64 RGB PNG images inside root/split/images/."""
    img_dir = root / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        _make_random_image(img_dir / f"room_{i:03d}.png")
    return img_dir


@pytest.fixture()
def tiny_raw(tmp_path):
    """A flat directory of raw images to test split_dataset."""
    for i in range(10):
        _make_random_image(tmp_path / f"img_{i:03d}.png")
    return tmp_path


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Pre-split dataset (train + val) without masks or captions."""
    _make_images(tmp_path, "train", n=6)
    _make_images(tmp_path, "val", n=2)
    return tmp_path


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_creates_train_and_val(self, tiny_raw, tmp_path):
        out = tmp_path / "out"
        split_dataset(str(tiny_raw), str(out), val_ratio=0.2, seed=0)
        assert (out / "train" / "images").exists()
        assert (out / "val" / "images").exists()

    def test_correct_split_sizes(self, tiny_raw, tmp_path):
        out = tmp_path / "out"
        split_dataset(str(tiny_raw), str(out), val_ratio=0.2, seed=42)
        train_imgs = list((out / "train" / "images").iterdir())
        val_imgs = list((out / "val" / "images").iterdir())
        total = len(train_imgs) + len(val_imgs)
        assert total == 10
        assert len(val_imgs) == 2

    def test_empty_source_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(RuntimeError, match="No images found"):
            split_dataset(str(tmp_path / "empty"), str(tmp_path / "out"))

    def test_reproducible_with_seed(self, tmp_path):
        # Use a dedicated source dir to prevent rglob from picking up output files
        src = tmp_path / "src"
        src.mkdir()
        for i in range(10):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(src / f"img_{i:03d}.png")

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        split_dataset(str(src), str(out1), seed=7)
        split_dataset(str(src), str(out2), seed=7)
        names1 = sorted(p.name for p in (out1 / "val" / "images").iterdir())
        names2 = sorted(p.name for p in (out2 / "val" / "images").iterdir())
        assert names1 == names2


# ---------------------------------------------------------------------------
# _image_paths helper
# ---------------------------------------------------------------------------

class TestImagePaths:
    def test_returns_sorted_list(self, tiny_dataset):
        paths = _image_paths(tiny_dataset / "train" / "images")
        assert len(paths) == 6
        assert paths == sorted(paths)

    def test_filters_non_image_files(self, tiny_dataset):
        img_dir = tiny_dataset / "train" / "images"
        (img_dir / "notes.txt").write_text("ignore me")
        paths = _image_paths(img_dir)
        assert all(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} for p in paths)


# ---------------------------------------------------------------------------
# _pick_best_mask
# ---------------------------------------------------------------------------

class TestPickBestMask:
    def _make_ann(self, cx: int, cy: int, w: int = 20, h: int = 20, size: int = 64) -> dict:
        seg = np.zeros((size, size), dtype=bool)
        seg[cy : cy + h, cx : cx + w] = True
        return {
            "bbox": [cx, cy, w, h],
            "segmentation": seg,
        }

    def test_returns_closest_to_centre(self):
        # Centre is (32, 32).  ann_near is closer.
        ann_far = self._make_ann(0, 0)
        ann_near = self._make_ann(28, 28)
        result = _pick_best_mask([ann_far, ann_near], cx=32, cy=32)
        assert result is not None
        # The near annotation was picked: its mask region is True around (28,28)
        assert result[28, 28] == 255

    def test_returns_none_for_empty(self):
        assert _pick_best_mask([], cx=32, cy=32) is None

    def test_mask_dtype_and_values(self):
        ann = self._make_ann(10, 10)
        result = _pick_best_mask([ann], cx=20, cy=20)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255})


# ---------------------------------------------------------------------------
# generate_captions – argument validation (no GPU required)
# ---------------------------------------------------------------------------

class TestGenerateCaptionsArgValidation:
    def test_unknown_captioner_raises(self, tiny_dataset):
        with pytest.raises(ValueError, match="Unknown captioner"):
            generate_captions(str(tiny_dataset), captioner="gpt4o")

    def test_supported_captioners_accepted(self):
        """Check that the supported names don't raise ValueError before loading."""
        for name in ("blip", "blip2", "florence2"):
            try:
                # Will fail at import/model-load stage, not at validation
                generate_captions("/nonexistent", captioner=name, splits=[])
            except (FileNotFoundError, RuntimeError, ImportError):
                pass
            except ValueError as exc:
                pytest.fail(f"ValueError raised for supported captioner '{name}': {exc}")

    def test_empty_splits_skips_gracefully(self, tiny_dataset):
        # splits=[] means "process nothing" – should complete without error
        generate_captions(str(tiny_dataset), captioner="blip2", splits=[])

    def test_missing_split_dir_is_skipped(self, tmp_path):
        # Only "train" exists; "val" does not
        _make_images(tmp_path, "train", n=2)
        # Should not raise; val is silently skipped
        generate_captions(str(tmp_path), captioner="blip2", splits=["val"])


# ---------------------------------------------------------------------------
# generate_captions – BLIP mocked end-to-end
# ---------------------------------------------------------------------------

class TestGenerateCaptionsMocked:
    @patch("prepare_data._caption_blip")
    def test_blip_captions_saved(self, mock_fn, tiny_dataset):
        mock_fn.return_value = {"room_000.png": "sofa caption", "room_001.png": "bed caption"}

        # Patch the per-split call so it writes the file
        root = tiny_dataset
        images_dir = root / "train" / "images"
        paths = _image_paths(images_dir)

        mock_fn.return_value = {p.name: f"caption for {p.stem}" for p in paths}

        with patch.object(pd_mod, "_caption_blip", mock_fn):
            generate_captions(str(root), captioner="blip", splits=["train"])

        caption_file = root / "train" / "captions.json"
        assert caption_file.exists()
        data = json.loads(caption_file.read_text())
        assert len(data) == len(paths)
        for name in data:
            assert isinstance(data[name], str) and len(data[name]) > 0

    @patch("prepare_data._caption_florence2")
    def test_florence2_captions_saved(self, mock_fn, tiny_dataset):
        root = tiny_dataset
        paths = _image_paths(root / "train" / "images")
        mock_fn.return_value = {p.name: f"florence caption {i}" for i, p in enumerate(paths)}

        with patch.object(pd_mod, "_caption_florence2", mock_fn):
            generate_captions(str(root), captioner="florence2", splits=["train"])

        caption_file = root / "train" / "captions.json"
        assert caption_file.exists()
        data = json.loads(caption_file.read_text())
        assert len(data) == len(paths)


# ---------------------------------------------------------------------------
# generate_masks – argument validation
# ---------------------------------------------------------------------------

class TestGenerateMasksArgValidation:
    def test_unknown_masker_raises(self, tiny_dataset):
        with pytest.raises(ValueError, match="Unknown masker"):
            generate_masks(str(tiny_dataset), masker="magical_ai")

    def test_supported_maskers_accepted(self):
        for name in ("sam", "sam2", "grounded_sam", "oneformer"):
            try:
                generate_masks("/nonexistent", masker=name, splits=[])
            except (FileNotFoundError, RuntimeError, ImportError):
                pass
            except ValueError as exc:
                pytest.fail(f"ValueError for supported masker '{name}': {exc}")

    def test_empty_splits_skips_gracefully(self, tiny_dataset):
        generate_masks(str(tiny_dataset), masker="sam2", splits=[])

    def test_missing_split_dir_is_skipped(self, tmp_path):
        generate_masks(str(tmp_path), masker="sam", splits=["train"])

    def test_sam_missing_checkpoint_raises(self, tiny_dataset):
        _make_images(tiny_dataset, "extra", n=1)
        # When segment-anything is not installed ImportError fires first;
        # when installed but file missing, FileNotFoundError is raised.
        with pytest.raises((FileNotFoundError, ImportError)):
            generate_masks(
                str(tiny_dataset),
                masker="sam",
                splits=["extra"],
                sam_checkpoint="/no/such/file.pth",
            )


# ---------------------------------------------------------------------------
# generate_masks – SAM mocked end-to-end
# ---------------------------------------------------------------------------

class TestGenerateMasksMocked:
    @patch("prepare_data._mask_sam")
    def test_sam_masks_dir_created(self, mock_fn, tiny_dataset):
        generate_masks(str(tiny_dataset), masker="sam", splits=["train"])
        assert (tiny_dataset / "train" / "masks").exists()
        mock_fn.assert_called_once()

    @patch("prepare_data._mask_sam2")
    def test_sam2_masks_dir_created(self, mock_fn, tiny_dataset):
        generate_masks(str(tiny_dataset), masker="sam2", splits=["train"])
        assert (tiny_dataset / "train" / "masks").exists()
        mock_fn.assert_called_once()

    @patch("prepare_data._mask_grounded_sam")
    def test_grounded_sam_masks_dir_created(self, mock_fn, tiny_dataset):
        generate_masks(str(tiny_dataset), masker="grounded_sam", splits=["train"])
        assert (tiny_dataset / "train" / "masks").exists()
        mock_fn.assert_called_once()

    @patch("prepare_data._mask_oneformer")
    def test_oneformer_masks_dir_created(self, mock_fn, tiny_dataset):
        generate_masks(str(tiny_dataset), masker="oneformer", splits=["train"])
        assert (tiny_dataset / "train" / "masks").exists()
        mock_fn.assert_called_once()

    @patch("prepare_data._mask_grounded_sam")
    def test_grounded_sam_receives_furniture_labels(self, mock_fn, tiny_dataset):
        custom_labels = "sofa,chair,table"
        generate_masks(
            str(tiny_dataset),
            masker="grounded_sam",
            splits=["train"],
            furniture_labels=custom_labels,
        )
        args, kwargs = mock_fn.call_args
        # furniture_labels is the 4th positional arg (index 3): paths, masks_dir, device, furniture_labels
        passed = kwargs.get("furniture_labels") or args[3]
        assert passed == custom_labels

    @patch("prepare_data._mask_oneformer")
    def test_oneformer_receives_target_labels(self, mock_fn, tiny_dataset):
        target = ["sofa", "chair"]
        generate_masks(
            str(tiny_dataset),
            masker="oneformer",
            splits=["train"],
            target_labels=target,
        )
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args
        # target_labels is passed as the 4th positional argument
        args, kwargs = call_kwargs
        passed_labels = kwargs.get("target_labels") or args[3]
        assert passed_labels == target


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

class TestCLIParser:
    def test_caption_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["caption", "--dataset_dir", "data/interior"])
        assert args.captioner == "blip2"
        assert args.splits == ["train", "val"]

    def test_caption_blip_selected(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["caption", "--dataset_dir", "data/interior", "--captioner", "blip"]
        )
        assert args.captioner == "blip"

    def test_caption_florence2_selected(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["caption", "--dataset_dir", "data/interior", "--captioner", "florence2"]
        )
        assert args.captioner == "florence2"

    def test_caption_invalid_choice_raises(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["caption", "--dataset_dir", "data/interior", "--captioner", "gpt4"]
            )

    def test_mask_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["mask", "--dataset_dir", "data/interior"])
        assert args.masker == "grounded_sam"
        assert args.model_type == "vit_h"

    def test_mask_sam2_selected(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["mask", "--dataset_dir", "data/interior", "--masker", "sam2"]
        )
        assert args.masker == "sam2"

    def test_mask_oneformer_selected(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["mask", "--dataset_dir", "data/interior", "--masker", "oneformer"]
        )
        assert args.masker == "oneformer"

    def test_mask_grounded_sam_with_labels(self):
        parser = _build_parser()
        args = parser.parse_args([
            "mask",
            "--dataset_dir", "data/interior",
            "--masker", "grounded_sam",
            "--furniture_labels", "sofa,chair",
        ])
        assert args.furniture_labels == "sofa,chair"

    def test_mask_sam_with_checkpoint(self):
        parser = _build_parser()
        args = parser.parse_args([
            "mask",
            "--dataset_dir", "data/interior",
            "--masker", "sam",
            "--sam_checkpoint", "checkpoints/sam.pth",
        ])
        assert args.sam_checkpoint == "checkpoints/sam.pth"

    def test_mask_invalid_masker_raises(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["mask", "--dataset_dir", "data/interior", "--masker", "magic"]
            )

    def test_split_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["split", "--source_dir", "data/raw", "--output_dir", "data/out"])
        assert args.val_ratio == 0.1
        assert args.seed == 42

    def test_split_custom_ratio(self):
        parser = _build_parser()
        args = parser.parse_args([
            "split", "--source_dir", "data/raw", "--output_dir", "data/out",
            "--val_ratio", "0.2",
        ])
        assert args.val_ratio == 0.2


# ---------------------------------------------------------------------------
# Default furniture labels constant
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_furniture_labels_non_empty(self):
        assert len(_DEFAULT_FURNITURE_LABELS) > 0

    def test_furniture_labels_comma_separated(self):
        labels = [l.strip() for l in _DEFAULT_FURNITURE_LABELS.split(",") if l.strip()]
        assert len(labels) >= 5  # at minimum: sofa, chair, table, bed, lamp

    def test_furniture_labels_contains_key_items(self):
        labels_lower = _DEFAULT_FURNITURE_LABELS.lower()
        for item in ("sofa", "chair", "table", "bed", "lamp"):
            assert item in labels_lower, f"'{item}' missing from default labels"


# ---------------------------------------------------------------------------
# _transformers_version and _torch_dtype_kwarg helpers
# ---------------------------------------------------------------------------

class TestTransformersVersionHelper:
    def test_returns_tuple(self):
        ver = _transformers_version()
        assert isinstance(ver, tuple)
        assert len(ver) == 2
        assert all(isinstance(x, int) for x in ver)

    def test_returns_zero_zero_on_import_failure(self):
        """When transformers is not importable the helper returns (0, 0)."""
        import sys
        with patch.dict(sys.modules, {"transformers": None}):
            ver = _transformers_version()
        assert ver == (0, 0)

    def test_real_version_is_positive(self):
        """The installed transformers should have a positive major version."""
        ver = _transformers_version()
        # transformers has been at major version 4 for a long time
        assert ver[0] >= 4 or ver == (0, 0), (
            "Expected major version >= 4 (or (0,0) fallback)"
        )


class TestTorchDtypeKwarg:
    """Verify _torch_dtype_kwarg returns the correct key for the given version."""

    def _patch_version(self, major: int, minor: int):
        return patch("prepare_data._transformers_version", return_value=(major, minor))

    def test_old_transformers_uses_torch_dtype(self):
        import torch
        with self._patch_version(4, 44):
            kwarg = _torch_dtype_kwarg(torch.float16)
        assert "torch_dtype" in kwarg
        assert "dtype" not in kwarg
        assert kwarg["torch_dtype"] == torch.float16

    def test_new_transformers_uses_dtype(self):
        import torch
        with self._patch_version(4, 48):
            kwarg = _torch_dtype_kwarg(torch.float16)
        assert "dtype" in kwarg
        assert "torch_dtype" not in kwarg
        assert kwarg["dtype"] == torch.float16

    def test_boundary_version_4_47_uses_torch_dtype(self):
        """4.47 is still below the 4.48 threshold → torch_dtype."""
        import torch
        with self._patch_version(4, 47):
            kwarg = _torch_dtype_kwarg(torch.float32)
        assert "torch_dtype" in kwarg

    def test_boundary_version_4_48_uses_dtype(self):
        import torch
        with self._patch_version(4, 48):
            kwarg = _torch_dtype_kwarg(torch.float32)
        assert "dtype" in kwarg


# ---------------------------------------------------------------------------
# Florence-2 version-aware loading path
# ---------------------------------------------------------------------------

class TestFlorence2Loading:
    """Verify that _caption_florence2 passes the correct kwargs to
    AutoModelForCausalLM.from_pretrained depending on the transformers version.

    Strategy: patch sys.modules["transformers"] so the lazy-import inside
    _caption_florence2 picks up our mock, then abort early (via StopIteration)
    after loading to avoid having to simulate a full inference pass.
    """

    def _capture_model_kwargs(self, tf_version: tuple, tiny_dataset) -> dict:
        """Return the kwargs that the selected model class's from_pretrained was
        called with for the given simulated *tf_version*.

        For tf_version < (4, 45) the legacy AutoModelForCausalLM path is used;
        for >= (4, 45) the native Florence2ForConditionalGeneration path is used.
        Both mocks raise StopIteration to abort after the load call so we capture
        the kwargs without running inference.
        """
        import sys
        captured: dict = {}

        mock_processor_inst = MagicMock()
        mock_processor_cls = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor_inst

        def fake_from_pretrained(model_id, **kwargs):
            captured.update(kwargs)
            raise StopIteration("abort-after-load")  # skip inference

        # Mock both possible model classes; whichever the code picks will fire.
        mock_auto_model_cls = MagicMock()
        mock_auto_model_cls.from_pretrained.side_effect = fake_from_pretrained

        mock_native_model_cls = MagicMock()
        mock_native_model_cls.from_pretrained.side_effect = fake_from_pretrained

        fake_tf = MagicMock()
        fake_tf.AutoModelForCausalLM = mock_auto_model_cls
        fake_tf.Florence2ForConditionalGeneration = mock_native_model_cls
        fake_tf.AutoProcessor = mock_processor_cls

        paths = _image_paths(tiny_dataset / "train" / "images")
        with patch("prepare_data._transformers_version", return_value=tf_version), \
             patch.dict(sys.modules, {"transformers": fake_tf}):
            with pytest.raises(StopIteration):
                pd_mod._caption_florence2(paths[:1], "cpu", batch_size=1)

        return captured

    def test_old_transformers_passes_trust_remote_code(self, tiny_dataset):
        """transformers < 4.45 → trust_remote_code=True must be passed."""
        kwargs = self._capture_model_kwargs((4, 44), tiny_dataset)
        assert kwargs.get("trust_remote_code") is True

    def test_new_transformers_omits_trust_remote_code(self, tiny_dataset):
        """transformers >= 4.45 → trust_remote_code must NOT be in kwargs.

        Passing it causes the 'model of type florence2 to instantiate model of
        type ``' architecture-class mismatch crash.
        """
        kwargs = self._capture_model_kwargs((4, 45), tiny_dataset)
        assert "trust_remote_code" not in kwargs

    def test_new_transformers_uses_dtype_kwarg(self, tiny_dataset):
        """transformers >= 4.48 → dtype kwarg, not torch_dtype."""
        kwargs = self._capture_model_kwargs((4, 48), tiny_dataset)
        assert "dtype" in kwargs
        assert "torch_dtype" not in kwargs

    def test_old_transformers_uses_torch_dtype_kwarg(self, tiny_dataset):
        """transformers < 4.48 → torch_dtype kwarg (not yet deprecated)."""
        kwargs = self._capture_model_kwargs((4, 44), tiny_dataset)
        assert "torch_dtype" in kwargs
        assert "dtype" not in kwargs

    def test_processor_always_trust_remote_code_on_new_transformers(self, tiny_dataset):
        """AutoProcessor.from_pretrained should be called WITH trust_remote_code=True
        even on transformers >= 4.45.

        The native Florence2Processor.__init__ accesses tokenizer.image_token which
        does not exist on RobertaTokenizer (the underlying tokenizer of Florence-2),
        raising:
            AttributeError: RobertaTokenizer has no attribute image_token

        Passing trust_remote_code=True forces the cached remote processor code that
        avoids this attribute access, fixing the crash on transformers 5.x.
        """
        import sys

        mock_processor_cls = MagicMock()
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        mock_auto_model_cls = MagicMock()
        mock_auto_model_cls.from_pretrained.side_effect = StopIteration("abort")

        mock_native_model_cls = MagicMock()
        mock_native_model_cls.from_pretrained.side_effect = StopIteration("abort")

        fake_tf = MagicMock()
        fake_tf.AutoModelForCausalLM = mock_auto_model_cls
        fake_tf.Florence2ForConditionalGeneration = mock_native_model_cls
        fake_tf.AutoProcessor = mock_processor_cls

        paths = _image_paths(tiny_dataset / "train" / "images")
        with patch("prepare_data._transformers_version", return_value=(4, 45)), \
             patch.dict(sys.modules, {"transformers": fake_tf}):
            with pytest.raises(StopIteration):
                pd_mod._caption_florence2(paths[:1], "cpu", batch_size=1)

        _, proc_kwargs = mock_processor_cls.from_pretrained.call_args
        assert proc_kwargs.get("trust_remote_code") is True

    def test_processor_trust_remote_code_on_old_transformers(self, tiny_dataset):
        """AutoProcessor.from_pretrained should be called WITH
        trust_remote_code=True on transformers < 4.45."""
        import sys

        mock_processor_cls = MagicMock()
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        mock_auto_model_cls = MagicMock()
        mock_auto_model_cls.from_pretrained.side_effect = StopIteration("abort")

        mock_native_model_cls = MagicMock()
        mock_native_model_cls.from_pretrained.side_effect = StopIteration("abort")

        fake_tf = MagicMock()
        fake_tf.AutoModelForCausalLM = mock_auto_model_cls
        fake_tf.Florence2ForConditionalGeneration = mock_native_model_cls
        fake_tf.AutoProcessor = mock_processor_cls

        paths = _image_paths(tiny_dataset / "train" / "images")
        with patch("prepare_data._transformers_version", return_value=(4, 44)), \
             patch.dict(sys.modules, {"transformers": fake_tf}):
            with pytest.raises(StopIteration):
                pd_mod._caption_florence2(paths[:1], "cpu", batch_size=1)

        _, proc_kwargs = mock_processor_cls.from_pretrained.call_args
        assert proc_kwargs.get("trust_remote_code") is True

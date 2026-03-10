"""
Unit tests for inference.py helper functions.
Run with: pytest tests/test_inference.py -v
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path so inference can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diffusers_mock():
    """Return a minimal mock of the diffusers module."""
    diffusers_mod = types.ModuleType("diffusers")
    pipeline_mock = MagicMock()
    diffusers_mod.StableDiffusionInpaintPipeline = pipeline_mock
    return diffusers_mod, pipeline_mock


# ---------------------------------------------------------------------------
# _build_pipeline – LoRA-directory detection
# ---------------------------------------------------------------------------

class TestBuildPipelineLoraDetection:
    """Tests for the LoRA-output-directory auto-detection in _build_pipeline."""

    def test_lora_dir_with_metadata_resolves_base_model(self, tmp_path):
        """model_dir containing training_metadata.json should auto-detect base model."""
        metadata = {
            "base_model": "runwayml/stable-diffusion-inpainting",
            "lora_dir": "unet_lora",
        }
        (tmp_path / "training_metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "unet_lora").mkdir()

        diffusers_mod, pipeline_cls = _make_diffusers_mock()
        fake_pipe = MagicMock()
        fake_pipe.to.return_value = fake_pipe
        pipeline_cls.from_pretrained.return_value = fake_pipe

        import torch

        with patch.dict(sys.modules, {"diffusers": diffusers_mod}):
            # Re-import so the patched module is used
            if "inference" in sys.modules:
                del sys.modules["inference"]
            from inference import _build_pipeline

            _build_pipeline(
                model_dir=str(tmp_path),
                lora_dir=None,
                device="cpu",
                dtype=torch.float32,
                enable_xformers=False,
            )

        pipeline_cls.from_pretrained.assert_called_once()
        call_args = pipeline_cls.from_pretrained.call_args
        assert call_args[0][0] == "runwayml/stable-diffusion-inpainting"

    def test_lora_dir_with_metadata_sets_lora_dir_automatically(self, tmp_path):
        """Auto-detected lora_dir should be passed to load_attn_procs."""
        metadata = {
            "base_model": "runwayml/stable-diffusion-inpainting",
            "lora_dir": "unet_lora",
        }
        (tmp_path / "training_metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "unet_lora").mkdir()

        diffusers_mod, pipeline_cls = _make_diffusers_mock()
        fake_pipe = MagicMock()
        fake_pipe.to.return_value = fake_pipe
        pipeline_cls.from_pretrained.return_value = fake_pipe

        import torch

        with patch.dict(sys.modules, {"diffusers": diffusers_mod}):
            if "inference" in sys.modules:
                del sys.modules["inference"]
            from inference import _build_pipeline

            _build_pipeline(
                model_dir=str(tmp_path),
                lora_dir=None,
                device="cpu",
                dtype=torch.float32,
                enable_xformers=False,
            )

        fake_pipe.unet.load_attn_procs.assert_called_once_with(
            str(tmp_path / "unet_lora")
        )

    def test_explicit_lora_dir_not_overridden_by_metadata(self, tmp_path):
        """Explicit --lora_dir must take precedence over metadata."""
        metadata = {
            "base_model": "runwayml/stable-diffusion-inpainting",
            "lora_dir": "unet_lora",
        }
        (tmp_path / "training_metadata.json").write_text(json.dumps(metadata))
        (tmp_path / "unet_lora").mkdir()
        explicit_lora = str(tmp_path / "my_custom_lora")

        diffusers_mod, pipeline_cls = _make_diffusers_mock()
        fake_pipe = MagicMock()
        fake_pipe.to.return_value = fake_pipe
        pipeline_cls.from_pretrained.return_value = fake_pipe

        import torch

        with patch.dict(sys.modules, {"diffusers": diffusers_mod}):
            if "inference" in sys.modules:
                del sys.modules["inference"]
            from inference import _build_pipeline

            _build_pipeline(
                model_dir=str(tmp_path),
                lora_dir=explicit_lora,
                device="cpu",
                dtype=torch.float32,
                enable_xformers=False,
            )

        fake_pipe.unet.load_attn_procs.assert_called_once_with(explicit_lora)

    def test_local_dir_without_model_index_or_metadata_raises(self, tmp_path):
        """Local directory with neither model_index.json nor training_metadata.json
        must raise OSError with a helpful message."""
        import torch

        # The OSError is raised before the diffusers import, so no mock needed.
        if "inference" in sys.modules:
            del sys.modules["inference"]
        from inference import _build_pipeline

        with pytest.raises(OSError, match="model_index.json"):
            _build_pipeline(
                model_dir=str(tmp_path),
                lora_dir=None,
                device="cpu",
                dtype=torch.float32,
                enable_xformers=False,
            )

    def test_local_dir_with_model_index_loads_normally(self, tmp_path):
        """A directory that has model_index.json should be loaded without LoRA detection."""
        (tmp_path / "model_index.json").write_text("{}")

        diffusers_mod, pipeline_cls = _make_diffusers_mock()
        fake_pipe = MagicMock()
        fake_pipe.to.return_value = fake_pipe
        pipeline_cls.from_pretrained.return_value = fake_pipe

        import torch

        with patch.dict(sys.modules, {"diffusers": diffusers_mod}):
            if "inference" in sys.modules:
                del sys.modules["inference"]
            from inference import _build_pipeline

            _build_pipeline(
                model_dir=str(tmp_path),
                lora_dir=None,
                device="cpu",
                dtype=torch.float32,
                enable_xformers=False,
            )

        pipeline_cls.from_pretrained.assert_called_once()
        call_args = pipeline_cls.from_pretrained.call_args
        assert call_args[0][0] == str(tmp_path)

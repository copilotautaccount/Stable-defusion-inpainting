#!/usr/bin/env python
# coding=utf-8
"""
DreamBooth LoRA fine-tuning for Stable Diffusion XL Inpainting.

Kết hợp các kỹ thuật từ:
  - DreamBooth LoRA (SDXL): prior preservation, EDM-style training, SNR gamma,
    text encoder LoRA, DoRA, Prodigy optimizer, Kohya export
  - SDXL Inpainting: 9-channel UNet input, mask/masked-image conditioning,
    dual text encoders, SDXL micro-conditioning (time_ids)

Usage
-----
# Single GPU
python train_dreambooth_lora_sdxl_inpaint.py --config configs/train_config.yaml

# Multi-GPU
accelerate launch --num_processes=4 train_dreambooth_lora_sdxl_inpaint.py \
    --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from omegaconf import OmegaConf
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_peft_version,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


# ──────────────────────────────────────────────────────────────────────────────
# Config & helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str):
    return OmegaConf.load(path)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: str | None,
    subfolder: str = "text_encoder",
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    """Encode prompt with both SDXL text encoders."""
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            text_input_ids = tokenize_prompt(tokenizers[i], prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds_list.append(prompt_embeds.view(bs_embed, seq_len, -1))

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


# ──────────────────────────────────────────────────────────────────────────────
# LoRA / DoRA config
# ──────────────────────────────────────────────────────────────────────────────

def get_lora_config(rank: int, alpha: int, dropout: float, use_dora: bool, target_modules: list):
    base = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if use_dora:
        if is_peft_version("<", "0.9.0"):
            raise ValueError("DoRA requires peft >= 0.9.0. Run: pip install -U peft")
        base["use_dora"] = True
    return LoraConfig(**base)


# ──────────────────────────────────────────────────────────────────────────────
# Model card
# ──────────────────────────────────────────────────────────────────────────────

def save_model_card(output_dir: str, base_model: str, dataset_dir: str,
                    lora_enabled: bool, use_dora: bool,
                    instance_prompt: str | None = None,
                    validation_prompt: str | None = None) -> None:
    adapter_type = "DoRA" if use_dora else ("LoRA" if lora_enabled else "Full fine-tune")
    card = f"""---
base_model: {base_model}
tags:
  - stable-diffusion-xl
  - inpainting
  - interior-design
  - dreambooth
  - {adapter_type.lower()}
license: openrail++
---

# SDXL Inpainting DreamBooth {adapter_type} – Interior Design

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) using
DreamBooth with {adapter_type} adapters on an interior-design inpainting dataset.

Training dataset: `{dataset_dir}`

{"**Trigger word:** `" + instance_prompt + "`" if instance_prompt else ""}
{"**Validation prompt:** `" + validation_prompt + "`" if validation_prompt else ""}
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)


# ──────────────────────────────────────────────────────────────────────────────
# DreamBooth Inpainting Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DreamBoothInpaintingDataset(Dataset):
    """
    Dataset for DreamBooth LoRA Inpainting.

    Supports:
      - instance_data_dir  : folder with images (+ optional per-image .txt captions)
      - class_data_dir     : folder with class images for prior preservation
      - mask_dir           : optional folder with pre-computed masks
                             (falls back to random box masks when absent)
    """

    def __init__(
        self,
        instance_data_dir: str,
        instance_prompt: str,
        class_prompt: str | None = None,
        class_data_dir: str | None = None,
        class_num: int | None = None,
        size: int = 1024,
        repeats: int = 1,
        center_crop: bool = False,
        random_flip: bool = False,
        mask_dir: str | None = None,
        mask_min_area: float = 0.1,
        mask_max_area: float = 0.5,
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.mask_min_area = mask_min_area
        self.mask_max_area = mask_max_area

        # ── Instance images ──────────────────────────────────────────────
        instance_data_root = Path(instance_data_dir)
        if not instance_data_root.exists():
            raise ValueError(f"Instance data dir does not exist: {instance_data_dir}")

        _exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        instance_paths = sorted(p for p in instance_data_root.iterdir() if p.suffix.lower() in _exts)
        if not instance_paths:
            raise ValueError(f"No images found in {instance_data_dir}")

        # Repeat for DreamBooth-style oversampling
        self.instance_paths: list[Path] = list(itertools.chain.from_iterable(
            itertools.repeat(p, repeats) for p in instance_paths
        ))

        # Optional per-image captions (file.jpg → file.txt)
        self.custom_instance_prompts: list[str | None] = []
        for p in self.instance_paths:
            cap_path = p.with_suffix(".txt")
            if cap_path.exists():
                self.custom_instance_prompts.append(cap_path.read_text().strip())
            else:
                self.custom_instance_prompts.append(None)

        # ── Mask dir ─────────────────────────────────────────────────────
        self.mask_dir = Path(mask_dir) if mask_dir else None

        # ── Image transforms ─────────────────────────────────────────────
        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS)
        self.train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_to_tensor = transforms.Compose([
            transforms.ToTensor(),          # [0,1]
        ])

        # Pre-process & cache instance images
        self._preprocess_instances()

        self.num_instance_images = len(self.instance_paths)
        self._length = self.num_instance_images

        # ── Class images (prior preservation) ────────────────────────────
        self.class_data_root = None
        if class_data_dir is not None:
            self.class_data_root = Path(class_data_dir)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            class_paths = sorted(p for p in self.class_data_root.iterdir() if p.suffix.lower() in _exts)
            self.num_class_images = min(len(class_paths), class_num) if class_num else len(class_paths)
            self.class_paths = class_paths[: self.num_class_images]
            self._length = max(self.num_class_images, self.num_instance_images)

        self.class_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    # ── Pre-processing ────────────────────────────────────────────────────

    def _preprocess_instances(self):
        self.pixel_values: list[torch.Tensor] = []
        self.mask_values: list[torch.Tensor] = []
        self.masked_image_values: list[torch.Tensor] = []
        self.original_sizes: list[tuple[int, int]] = []
        self.crop_top_lefts: list[tuple[int, int]] = []

        for img_path in self.instance_paths:
            image = exif_transpose(Image.open(img_path))
            if image.mode != "RGB":
                image = image.convert("RGB")

            self.original_sizes.append((image.height, image.width))
            image = self.train_resize(image)

            # Flip
            if self.random_flip and random.random() < 0.5:
                image = self.train_flip(image)

            # Crop
            if self.center_crop:
                y1 = max(0, int(round((image.height - self.size) / 2.0)))
                x1 = max(0, int(round((image.width - self.size) / 2.0)))
                image = self.train_crop(image)
            else:
                y1, x1, h, w = self.train_crop.get_params(image, (self.size, self.size))
                image = crop(image, y1, x1, h, w)

            self.crop_top_lefts.append((y1, x1))

            # Pixel values (normalised)
            pv = self.to_tensor_norm(image)
            self.pixel_values.append(pv)

            # Mask
            mask = self._load_or_generate_mask(img_path, image)  # [1, H, W], values 0/1
            self.mask_values.append(mask)

            # Masked image = image * (1 - mask)  → erased regions
            masked_pv = pv * (1 - mask)
            self.masked_image_values.append(masked_pv)

    def _load_or_generate_mask(self, img_path: Path, image: Image.Image) -> torch.Tensor:
        """Return a binary mask tensor [1, H, W] with 1 = inpaint region."""
        size = self.size
        if self.mask_dir is not None:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            if mask_path.exists():
                m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
                m_arr = np.array(m, dtype=np.float32) / 255.0
                m_arr = (m_arr > 0.5).astype(np.float32)
                # Fall back to random box if mask is blank
                if m_arr.max() == 0:
                    return self._random_box_mask(size)
                return torch.from_numpy(m_arr).unsqueeze(0)

        return self._random_box_mask(size)

    def _random_box_mask(self, size: int) -> torch.Tensor:
        """Generate a random rectangular mask."""
        area = size * size
        mask_area = random.uniform(self.mask_min_area, self.mask_max_area) * area
        h = int(random.uniform(0.2, 0.8) * size)
        w = int(mask_area / max(h, 1))
        w = min(w, size)
        y0 = random.randint(0, size - h)
        x0 = random.randint(0, size - w)
        mask = np.zeros((size, size), dtype=np.float32)
        mask[y0: y0 + h, x0: x0 + w] = 1.0
        return torch.from_numpy(mask).unsqueeze(0)

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        i = index % self.num_instance_images
        example = {
            "instance_images":        self.pixel_values[i],
            "instance_masks":         self.mask_values[i],
            "instance_masked_images": self.masked_image_values[i],
            "original_size":          self.original_sizes[i],
            "crop_top_left":          self.crop_top_lefts[i],
            "instance_prompt":        self.custom_instance_prompts[i] or self.instance_prompt,
        }

        if self.class_data_root:
            j = index % self.num_class_images
            cls_img = exif_transpose(Image.open(self.class_paths[j]))
            if cls_img.mode != "RGB":
                cls_img = cls_img.convert("RGB")
            cls_pv = self.class_transform(cls_img)
            cls_mask = self._random_box_mask(self.size)
            example["class_images"]        = cls_pv
            example["class_masks"]         = cls_mask
            example["class_masked_images"] = cls_pv * (1 - cls_mask)
            example["class_prompt"]        = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation: bool = False):
    pixel_values       = [e["instance_images"] for e in examples]
    masks              = [e["instance_masks"] for e in examples]
    masked_images      = [e["instance_masked_images"] for e in examples]
    prompts            = [e["instance_prompt"] for e in examples]
    original_sizes     = [e["original_size"] for e in examples]
    crop_top_lefts     = [e["crop_top_left"] for e in examples]

    if with_prior_preservation:
        pixel_values   += [e["class_images"] for e in examples]
        masks          += [e["class_masks"] for e in examples]
        masked_images  += [e["class_masked_images"] for e in examples]
        prompts        += [e["class_prompt"] for e in examples]
        original_sizes += [e["original_size"] for e in examples]
        crop_top_lefts += [e["crop_top_left"] for e in examples]

    return {
        "pixel_values":   torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
        "masks":          torch.stack(masks).float(),
        "masked_images":  torch.stack(masked_images).to(memory_format=torch.contiguous_format).float(),
        "prompts":        prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prior-preservation class image generation
# ──────────────────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


def generate_class_images(cfg, accelerator):
    """Generate class images for prior preservation if not enough exist."""
    class_images_dir = Path(cfg.dreambooth.class_data_dir)
    class_images_dir.mkdir(parents=True, exist_ok=True)
    cur = len(list(class_images_dir.iterdir()))
    target = cfg.dreambooth.num_class_images

    if cur >= target:
        return

    logger.info(f"Generating {target - cur} class images in {class_images_dir} …")

    has_fp16 = torch.cuda.is_available() or torch.backends.mps.is_available()
    torch_dtype = torch.float16 if has_fp16 else torch.float32

    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(accelerator.device)

    # Dummy all-white image + centre-square mask for class generation
    res = cfg.training.resolution
    dummy_img  = Image.fromarray(np.full((res, res, 3), 255, dtype=np.uint8))
    mask_arr   = np.zeros((res, res), dtype=np.uint8)
    mask_arr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
    dummy_mask = Image.fromarray(mask_arr)

    ds = PromptDataset(cfg.dreambooth.class_prompt, target - cur)
    dl = DataLoader(ds, batch_size=cfg.training.sample_batch_size)
    dl = accelerator.prepare(dl)

    for batch in tqdm(dl, desc="Generating class images",
                      disable=not accelerator.is_local_main_process):
        images = pipeline(
            prompt=list(batch["prompt"]),
            image=[dummy_img] * len(batch["prompt"]),
            mask_image=[dummy_mask] * len(batch["prompt"]),
            height=res, width=res,
        ).images
        for i, img in enumerate(images):
            h = insecure_hashlib.sha1(img.tobytes()).hexdigest()
            img.save(class_images_dir / f"{batch['index'][i].item() + cur}-{h}.jpg")

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def log_validation(pipeline, cfg, accelerator, epoch, step, weight_dtype):
    if not cfg.validation.validation_prompts:
        return []

    logger.info(f"Running validation (epoch {epoch}, step {step}) …")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
        if cfg.training.seed is not None else None
    )

    res = cfg.training.resolution
    val_dir = Path(cfg.training.output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    _img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    val_images_dir = Path(cfg.data.dataset_dir) / cfg.data.val_split / "images"
    val_masks_dir  = Path(cfg.data.dataset_dir) / cfg.data.val_split / "masks"
    val_img_paths  = sorted(
        p for p in val_images_dir.iterdir() if p.suffix.lower() in _img_exts
    ) if val_images_dir.exists() else []

    autocast_ctx = (
        torch.autocast(accelerator.device.type, dtype=weight_dtype)
        if torch.cuda.is_available() else nullcontext()
    )

    images = []
    num_to_show = min(cfg.validation.num_validation_images, len(cfg.validation.validation_prompts))
    for i, prompt in enumerate(cfg.validation.validation_prompts[:num_to_show]):
        if val_img_paths:
            ip = val_img_paths[i % len(val_img_paths)]
            pil_img = Image.open(ip).convert("RGB").resize((res, res), Image.LANCZOS)
            mp = val_masks_dir / (ip.stem + ".png")
            if mp.exists():
                marr = np.array(Image.open(mp).convert("L").resize((res, res), Image.NEAREST))
            else:
                marr = np.zeros((res, res), dtype=np.uint8)
            if marr.max() == 0:
                marr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
            pil_mask = Image.fromarray(marr)
        else:
            pil_img  = Image.fromarray(np.full((res, res, 3), 255, dtype=np.uint8))
            marr     = np.zeros((res, res), dtype=np.uint8)
            marr[res // 4: 3 * res // 4, res // 4: 3 * res // 4] = 255
            pil_mask = Image.fromarray(marr)

        with autocast_ctx:
            out = pipeline(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                height=res, width=res,
                num_inference_steps=25,
                generator=generator,
            ).images[0]

        out.save(val_dir / f"epoch{epoch:04d}_step{step:07d}_{i}.png")
        images.append(out)

    if accelerator.is_main_process and cfg.logging.report_to == "wandb" and is_wandb_available():
        import wandb
        accelerator.log(
            {"validation": [
                wandb.Image(img, caption=p)
                for img, p in zip(images, cfg.validation.validation_prompts[:len(images)])
            ]},
            step=step,
        )

    return images


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (save / load hooks)
# ──────────────────────────────────────────────────────────────────────────────

def build_save_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg):
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        unet_lora_layers = te1_lora_layers = te2_lora_layers = None

        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                te1_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            elif text_encoder_two is not None and isinstance(
                model, type(accelerator.unwrap_model(text_encoder_two))
            ):
                te2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            else:
                raise ValueError(f"Unexpected model type: {model.__class__}")
            weights.pop()

        StableDiffusionXLInpaintPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=te1_lora_layers,
            text_encoder_2_lora_layers=te2_lora_layers,
        )

    return save_model_hook


def build_load_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg):
    def load_model_hook(models, input_dir):
        unet_ = te1_ = te2_ = None
        while models:
            m = models.pop()
            if isinstance(m, type(accelerator.unwrap_model(unet))):
                unet_ = m
            elif isinstance(m, type(accelerator.unwrap_model(text_encoder_one))):
                te1_ = m
            elif text_encoder_two is not None and isinstance(
                m, type(accelerator.unwrap_model(text_encoder_two))
            ):
                te2_ = m
            else:
                raise ValueError(f"Unexpected model type: {m.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_sd = {k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_sd = convert_unet_state_dict_to_peft(unet_sd)
        incompatible = set_peft_model_state_dict(unet_, unet_sd, adapter_name="default")
        if incompatible and getattr(incompatible, "unexpected_keys", None):
            logger.warning(f"Unexpected keys when loading LoRA: {incompatible.unexpected_keys}")

        if cfg.dreambooth.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=te1_)
            if te2_ is not None:
                _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder_2.", text_encoder=te2_)

        if cfg.training.mixed_precision == "fp16":
            ms = [unet_]
            if cfg.dreambooth.train_text_encoder:
                ms += [te1_, te2_]
            cast_training_params([m for m in ms if m is not None])

    return load_model_hook


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ── Accelerator ───────────────────────────────────────────────────────
    project_config = ProjectConfiguration(
        project_dir=cfg.training.output_dir,
        logging_dir=os.path.join(cfg.training.output_dir, "logs"),
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to if cfg.logging.report_to != "none" else None,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if accelerator.is_main_process:
        os.makedirs(cfg.training.output_dir, exist_ok=True)

    set_seed(cfg.training.seed)

    # ── Generate class images if prior preservation enabled ───────────────
    if cfg.dreambooth.with_prior_preservation:
        if accelerator.is_main_process:
            generate_class_images(cfg, accelerator)
        accelerator.wait_for_everyone()

    # ── Models ────────────────────────────────────────────────────────────
    pretrained = cfg.model.pretrained_model_name_or_path

    tokenizer_one = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer_2", use_fast=False)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(pretrained, None, "text_encoder_2")

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    # EDM-style training detection
    do_edm = getattr(cfg.training, "do_edm_style_training", False)
    if do_edm:
        try:
            noise_scheduler = EDMEulerScheduler.from_pretrained(pretrained, subfolder="scheduler")
        except Exception:
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained, subfolder="scheduler")
        logger.info("EDM-style training enabled.")

    text_encoder_one = text_encoder_cls_one.from_pretrained(pretrained, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(pretrained, subfolder="text_encoder_2")

    vae_path = getattr(cfg.model, "pretrained_vae_model_name_or_path", None) or pretrained
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if not getattr(cfg.model, "pretrained_vae_model_name_or_path", None) else None,
    )

    # SDXL Inpainting UNet (9 input channels)
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")

    # Freeze non-trainable weights
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Weight dtype ──────────────────────────────────────────────────────
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)   # VAE always fp32
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # ── xFormers ─────────────────────────────────────────────────────────
    if getattr(cfg.training, "enable_xformers", False):
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers enabled.")
        else:
            logger.warning("xFormers requested but not available.")

    # ── Gradient checkpointing ────────────────────────────────────────────
    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── LoRA / DoRA on UNet ───────────────────────────────────────────────
    rank     = cfg.lora.rank
    alpha    = getattr(cfg.lora, "alpha", rank)
    dropout  = getattr(cfg.lora, "dropout", 0.0)
    use_dora = getattr(cfg.lora, "use_dora", False)

    unet_lora_cfg = get_lora_config(
        rank=rank, alpha=alpha, dropout=dropout, use_dora=use_dora,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_cfg)

    # ── LoRA on text encoders (optional) ─────────────────────────────────
    train_te = cfg.dreambooth.train_text_encoder
    if train_te:
        te_lora_cfg = get_lora_config(
            rank=rank, alpha=alpha, dropout=dropout, use_dora=use_dora,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(te_lora_cfg)
        text_encoder_two.add_adapter(te_lora_cfg)
        if cfg.training.gradient_checkpointing:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # ── Mixed-precision: upcast LoRA params to fp32 ───────────────────────
    if cfg.training.mixed_precision == "fp16":
        models_to_cast = [unet]
        if train_te:
            models_to_cast += [text_encoder_one, text_encoder_two]
        cast_training_params(models_to_cast, dtype=torch.float32)

    # ── Save / Load hooks ─────────────────────────────────────────────────
    accelerator.register_save_state_pre_hook(
        build_save_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg)
    )
    accelerator.register_load_state_pre_hook(
        build_load_hook(accelerator, unet, text_encoder_one, text_encoder_two, cfg)
    )

    # ── Optimiser ─────────────────────────────────────────────────────────
    unet_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize = [{"params": unet_params, "lr": cfg.training.learning_rate}]

    if train_te:
        te1_params = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        te2_params = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        te_lr = getattr(cfg.training, "text_encoder_lr", cfg.training.learning_rate)
        params_to_optimize += [
            {"params": te1_params, "lr": te_lr, "weight_decay": cfg.training.adam_weight_decay},
            {"params": te2_params, "lr": te_lr, "weight_decay": cfg.training.adam_weight_decay},
        ]

    optimizer_name = getattr(cfg.training, "optimizer", "adamw").lower()

    if optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("Install prodigyopt: pip install prodigyopt")
        optimizer = prodigyopt.Prodigy(
            params_to_optimize,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            weight_decay=cfg.training.adam_weight_decay,
            eps=cfg.training.adam_epsilon,
            decouple=getattr(cfg.training, "prodigy_decouple", True),
            use_bias_correction=getattr(cfg.training, "prodigy_use_bias_correction", True),
            safeguard_warmup=getattr(cfg.training, "prodigy_safeguard_warmup", True),
        )
    else:
        if getattr(cfg.training, "use_8bit_adam", False):
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                logger.warning("bitsandbytes not available – using standard AdamW.")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            params_to_optimize,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            weight_decay=cfg.training.adam_weight_decay,
            eps=cfg.training.adam_epsilon,
        )

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    train_dataset = DreamBoothInpaintingDataset(
        instance_data_dir=cfg.data.instance_data_dir,
        instance_prompt=cfg.dreambooth.instance_prompt,
        class_prompt=cfg.dreambooth.class_prompt if cfg.dreambooth.with_prior_preservation else None,
        class_data_dir=cfg.dreambooth.class_data_dir if cfg.dreambooth.with_prior_preservation else None,
        class_num=cfg.dreambooth.num_class_images,
        size=cfg.training.resolution,
        repeats=getattr(cfg.data, "repeats", 1),
        center_crop=cfg.data.center_crop,
        random_flip=cfg.data.random_flip,
        mask_dir=getattr(cfg.data, "mask_dir", None),
        mask_min_area=getattr(cfg.data, "mask_min_area", 0.1),
        mask_max_area=getattr(cfg.data, "mask_max_area", 0.5),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        collate_fn=lambda ex: collate_fn(ex, cfg.dreambooth.with_prior_preservation),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    max_train_steps = (
        cfg.training.max_train_steps
        or cfg.training.num_train_epochs * num_update_steps_per_epoch
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=getattr(cfg.training, "lr_num_cycles", 1),
        power=getattr(cfg.training, "lr_power", 1.0),
    )

    # ── Prepare with Accelerator ──────────────────────────────────────────
    if train_te:
        unet, text_encoder_one, text_encoder_two, optimizer, train_loader, lr_scheduler = (
            accelerator.prepare(
                unet, text_encoder_one, text_encoder_two, optimizer, train_loader, lr_scheduler
            )
        )
    else:
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )

    # Recompute steps after potential loader resizing
    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    if not cfg.training.max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # ── Pre-compute text embeddings (when text encoder is frozen) ─────────
    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = (cfg.training.resolution, cfg.training.resolution)
        ids = list(original_size + crops_coords_top_left + target_size)
        return torch.tensor([ids], device=accelerator.device, dtype=weight_dtype)

    if not train_te:
        tokenizers   = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                emb, pooled = encode_prompt(text_encoders, tokenizers, prompt)
            return emb.to(accelerator.device), pooled.to(accelerator.device)

        # Pre-compute instance embeddings (skip if per-image captions)
        has_custom_prompts = any(p is not None for p in train_dataset.custom_instance_prompts
                                  if p != cfg.dreambooth.instance_prompt)
        if not has_custom_prompts:
            instance_prompt_embeds, instance_pooled_embeds = compute_text_embeddings(
                cfg.dreambooth.instance_prompt
            )
            if cfg.dreambooth.with_prior_preservation:
                class_prompt_embeds, class_pooled_embeds = compute_text_embeddings(
                    cfg.dreambooth.class_prompt
                )
                prompt_embeds      = torch.cat([instance_prompt_embeds, class_prompt_embeds], dim=0)
                pooled_embeds      = torch.cat([instance_pooled_embeds, class_pooled_embeds], dim=0)
            else:
                prompt_embeds = instance_prompt_embeds
                pooled_embeds = instance_pooled_embeds
        else:
            has_custom_prompts = True   # will encode per-batch

    # ── Trackers ──────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        accelerator.init_trackers(
            cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info("***** Starting DreamBooth LoRA Inpainting training *****")
    logger.info(f"  Instances     = {len(train_dataset)}")
    logger.info(f"  Epochs        = {num_train_epochs}")
    logger.info(f"  Batch size    = {cfg.training.train_batch_size}")
    logger.info(f"  Grad. accum   = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total steps   = {max_train_steps}")
    logger.info(f"  LoRA rank     = {rank}  |  alpha = {alpha}  |  DoRA = {use_dora}")
    logger.info(f"  Prior preserv = {cfg.dreambooth.with_prior_preservation}")
    logger.info(f"  Train TE      = {train_te}")
    logger.info(f"  EDM-style     = {do_edm}")

    global_step  = 0
    first_epoch  = 0
    resume_step  = 0

    # ── Resume ────────────────────────────────────────────────────────────
    if getattr(cfg.training, "resume_from_checkpoint", None):
        ckpt = cfg.training.resume_from_checkpoint
        if ckpt == "latest":
            dirs = sorted(
                [d for d in Path(cfg.training.output_dir).iterdir() if d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            ckpt = str(dirs[-1]) if dirs else None

        if ckpt:
            accelerator.load_state(ckpt)
            state_file = Path(ckpt) / "training_state.json"
            if state_file.exists():
                saved = json.loads(state_file.read_text())
                global_step = saved["global_step"]
                first_epoch = saved["epoch"]
            else:
                global_step = int(Path(ckpt).name.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step - first_epoch * num_update_steps_per_epoch
            logger.info(f"Resumed from {ckpt}  (global_step={global_step})")

    # ── EDM sigma helper ──────────────────────────────────────────────────
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_ts = noise_scheduler.timesteps.to(accelerator.device)
        timesteps   = timesteps.to(accelerator.device)
        step_indices = [(schedule_ts == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ── SNR gamma ─────────────────────────────────────────────────────────
    snr_gamma = getattr(cfg.training, "snr_gamma", None)

    # ── Training loop ─────────────────────────────────────────────────────
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if train_te:
            text_encoder_one.train()
            text_encoder_two.train()
            # Ensure embeddings get gradients for gradient checkpointing
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        # Skip already-processed batches on resume
        batches_to_skip = resume_step * cfg.training.gradient_accumulation_steps if epoch == first_epoch else 0
        if batches_to_skip > 0:
            logger.info(f"  Epoch {epoch}: skipping {batches_to_skip} batches …")
            if hasattr(accelerator, "skip_first_batches"):
                active_loader = accelerator.skip_first_batches(train_loader, batches_to_skip)
            else:
                active_loader = iter(train_loader)
                for _ in range(batches_to_skip):
                    next(active_loader, None)
        else:
            active_loader = train_loader

        for step, batch in enumerate(active_loader):
            with accelerator.accumulate(unet):
                # ── Encode images → latents ───────────────────────────────
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["masked_images"].to(dtype=vae.dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                # Resize mask to latent spatial dimensions
                mask = F.interpolate(
                    batch["masks"].to(dtype=weight_dtype),
                    size=latents.shape[-2:],
                    mode="nearest",
                )

                latents        = latents.to(weight_dtype)
                masked_latents = masked_latents.to(weight_dtype)

                # ── Noise & timesteps ─────────────────────────────────────
                noise = torch.randn_like(latents)
                bsz   = latents.shape[0]

                if not do_edm:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device,
                    ).long()
                else:
                    indices   = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # EDM preconditioning
                if do_edm:
                    sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    inp_noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)
                else:
                    inp_noisy_latents = noisy_latents

                # ── SDXL micro-conditioning (time_ids) ────────────────────
                add_time_ids = torch.cat([
                    compute_time_ids(s, c)
                    for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                ])

                # ── Text embeddings ───────────────────────────────────────
                if not train_te:
                    if has_custom_prompts:
                        cur_embeds, cur_pooled = compute_text_embeddings(batch["prompts"])
                    else:
                        repeat_n = bsz // 2 if cfg.dreambooth.with_prior_preservation else bsz
                        cur_embeds = prompt_embeds.repeat(repeat_n, 1, 1)
                        cur_pooled = pooled_embeds.repeat(repeat_n, 1)
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, batch["prompts"])
                    tokens_two = tokenize_prompt(tokenizer_two, batch["prompts"])
                    cur_embeds, cur_pooled = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )

                # ── 9-channel UNet input (inpainting) ────────────────────
                # [noisy_latents(4) | mask(1) | masked_image_latents(4)]
                unet_input = torch.cat([inp_noisy_latents, mask, masked_latents], dim=1)

                added_cond_kwargs = {
                    "time_ids":    add_time_ids,
                    "text_embeds": cur_pooled,
                }

                model_pred = unet(
                    unet_input,
                    timesteps,
                    cur_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # ── EDM output postconditioning ───────────────────────────
                weighting = None
                if do_edm:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        model_pred = model_pred * (-sigmas) + noisy_latents
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        model_pred = model_pred * (-sigmas / (sigmas ** 2 + 1) ** 0.5) + (
                            noisy_latents / (sigmas ** 2 + 1)
                        )
                    weighting = (sigmas ** -2.0).float()

                # ── Target ────────────────────────────────────────────────
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = latents if do_edm else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        latents if do_edm
                        else noise_scheduler.get_velocity(latents, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction_type: {noise_scheduler.config.prediction_type}")

                # ── Prior preservation split ───────────────────────────────
                if cfg.dreambooth.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target,     target_prior     = torch.chunk(target, 2, dim=0)

                    # Prior loss
                    if weighting is not None:
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2)
                            .reshape(target_prior.shape[0], -1), 1
                        ).mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # ── Instance loss ─────────────────────────────────────────
                if snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2)
                            .reshape(target.shape[0], -1), 1
                        ).mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Min-SNR weighting (Section 3.4 of https://arxiv.org/abs/2303.09556)
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = base_weight + 1
                    else:
                        mse_loss_weights = base_weight
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = (loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights).mean()

                # Add prior loss
                if cfg.dreambooth.with_prior_preservation:
                    loss = loss + cfg.dreambooth.prior_loss_weight * prior_loss

                # ── Backward ──────────────────────────────────────────────
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    all_params = unet_params
                    if train_te:
                        all_params = itertools.chain(all_params, te1_params, te2_params)
                    accelerator.clip_grad_norm_(all_params, cfg.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── Sync & checkpoint ─────────────────────────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.logging.log_every_n_steps == 0:
                        accelerator.log(
                            {"train_loss": loss.detach().item(),
                             "lr": lr_scheduler.get_last_lr()[0]},
                            step=global_step,
                        )

                    # Save checkpoint
                    if global_step % cfg.training.checkpointing_steps == 0:
                        ckpt_dir = os.path.join(cfg.training.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(ckpt_dir)
                        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as fh:
                            json.dump({"global_step": global_step, "epoch": epoch}, fh)
                        # Prune old checkpoints
                        ckpts = sorted(
                            [d for d in Path(cfg.training.output_dir).iterdir()
                             if d.name.startswith("checkpoint-")],
                            key=lambda d: int(d.name.split("-")[1]),
                        )
                        keep = getattr(cfg.training, "checkpoints_total_limit", 3) or 3
                        for old in ckpts[:-keep]:
                            shutil.rmtree(old)
                        logger.info(f"Saved checkpoint: {ckpt_dir}")

                    # Validation
                    if global_step % cfg.validation.validation_steps == 0:
                        unet_unwrapped = accelerator.unwrap_model(unet)
                        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                            pretrained,
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            unet=unet_unwrapped,
                            safety_checker=None,
                            torch_dtype=weight_dtype,
                        )
                        log_validation(pipeline, cfg, accelerator, epoch, global_step, weight_dtype)
                        del pipeline
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

            if global_step >= max_train_steps:
                break

    # ── Save final LoRA weights ───────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet).to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_unwrapped))

        te1_lora_layers = te2_lora_layers = None
        if train_te:
            te1 = accelerator.unwrap_model(text_encoder_one).to(torch.float32)
            te2 = accelerator.unwrap_model(text_encoder_two).to(torch.float32)
            te1_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(te1))
            te2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(te2))

        StableDiffusionXLInpaintPipeline.save_lora_weights(
            save_directory=cfg.training.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=te1_lora_layers,
            text_encoder_2_lora_layers=te2_lora_layers,
        )

        # ── Optional: Kohya-compatible export ────────────────────────────
        if getattr(cfg.training, "output_kohya_format", False):
            lora_sd = load_file(f"{cfg.training.output_dir}/pytorch_lora_weights.safetensors")
            peft_sd = convert_all_state_dict_to_peft(lora_sd)
            kohya_sd = convert_state_dict_to_kohya(peft_sd)
            save_file(kohya_sd, f"{cfg.training.output_dir}/pytorch_lora_weights_kohya.safetensors")
            logger.info("Saved Kohya-format LoRA weights.")

        # ── Final validation run ──────────────────────────────────────────
        vae_final = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if not getattr(cfg.model, "pretrained_vae_model_name_or_path", None) else None,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            pretrained, vae=vae_final, torch_dtype=weight_dtype, safety_checker=None
        )
        pipeline.load_lora_weights(cfg.training.output_dir)
        log_validation(pipeline, cfg, accelerator, num_train_epochs, global_step, weight_dtype)
        del pipeline

        # ── Model card ────────────────────────────────────────────────────
        save_model_card(
            cfg.training.output_dir,
            base_model=pretrained,
            dataset_dir=cfg.data.instance_data_dir,
            lora_enabled=True,
            use_dora=use_dora,
            instance_prompt=cfg.dreambooth.instance_prompt,
            validation_prompt=(cfg.validation.validation_prompts[0]
                               if cfg.validation.validation_prompts else None),
        )

        # ── Push to Hub ───────────────────────────────────────────────────
        if getattr(cfg, "push_to_hub", None) and cfg.push_to_hub.enabled:
            from huggingface_hub import HfApi
            api = HfApi(token=cfg.push_to_hub.hub_token)
            api.upload_folder(
                folder_path=cfg.training.output_dir,
                repo_id=cfg.push_to_hub.hub_model_id,
                repo_type="model",
                commit_message="End of DreamBooth LoRA Inpainting training",
                ignore_patterns=["step_*", "epoch_*", "checkpoint-*"],
            )

    accelerator.end_training()
    logger.info("Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamBooth LoRA for SDXL Inpainting")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
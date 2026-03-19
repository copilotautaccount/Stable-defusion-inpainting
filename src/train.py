"""
Fine-tuning script for Stable Diffusion XL Inpainting on interior-design images.

Supports:
  • Two-stage LoRA fine-tuning (default)
      Stage 1 – noise MSE loss only  (builds a solid denoising foundation)
      Stage 2 – full multi-loss from stage-1 checkpoint
  • Single-stage training (--stage 1 or --stage 2 independently)
  • LoRA fine-tuning (default, memory-efficient)
  • Full fine-tuning (--no_lora flag)

Usage
-----
# Full two-stage run (sequential)
python src/train.py --config configs/train_config.yaml

# Train only stage 1
python src/train.py --config configs/train_config.yaml --stage 1

# Train only stage 2 (stage-1 checkpoint must exist)
python src/train.py --config configs/train_config.yaml --stage 2

# Multi-GPU with accelerate
accelerate launch --num_processes=4 src/train.py --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from dataset import InteriorInpaintingDataset

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> DictConfig:
    return OmegaConf.load(path)


def save_model_card(output_dir: str, base_model: str, dataset_dir: str, lora_enabled: bool, stage: int) -> None:
    card = f"""---
base_model: {base_model}
tags:
  - stable-diffusion-xl
  - inpainting
  - interior-design
  - fine-tuned
  - {"lora" if lora_enabled else "full-fine-tune"}
  - two-stage-training
  - stage-{stage}
license: creativeml-openrail-m
---

# Stable Diffusion XL Inpainting – Interior Design (Stage {stage})

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) on an interior-design
inpainting dataset using **Two-stage LoRA training**.

Training dataset: `{dataset_dir}`

{"This checkpoint uses **LoRA** adapters." if lora_enabled else "This is a **full fine-tune**."}

## Training stages

| Stage | Loss functions |
|-------|---------------|
| 1 | `noise_mse` only |
| 2 | `noise_mse` + `pixel` + `perceptual` + `clip` + `boundary` + `depth` + `semantic` |
"""
    with open(os.path.join(output_dir, "README.md"), "w") as fh:
        fh.write(card)


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def enable_lora(unet: UNet2DConditionModel, cfg: DictConfig) -> UNet2DConditionModel:
    """Inject LoRA adapters into UNet attention/projection layers."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install peft: pip install peft") from exc

    lora_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()
    return unet


# ---------------------------------------------------------------------------
# Multi-loss components (Stage 2)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """VGG-16 feature-matching perceptual loss (relu3_3 features)."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features[:16]
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.to(device, dtype=dtype).eval()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n   = (pred   * 0.5 + 0.5 - self.mean) / self.std
        target_n = (target * 0.5 + 0.5 - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))


class CLIPImageLoss(nn.Module):
    """CLIP image-embedding cosine similarity loss."""

    def __init__(self, device: torch.device, dtype: torch.dtype,
                 model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model = self.model.to(device, dtype=dtype).eval()
        self.device = device
        self.dtype = dtype

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _to_numpy(t):
            return [
                (t[i].detach().clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).cpu().float().numpy()
                for i in range(t.shape[0])
            ]

        pred_inputs   = self.processor(images=_to_numpy(pred),   return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)
        target_inputs = self.processor(images=_to_numpy(target), return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

        pred_feat   = self.model(pixel_values=pred_inputs).pooler_output
        target_feat = self.model(pixel_values=target_inputs).pooler_output
        return 1.0 - F.cosine_similarity(pred_feat, target_feat).mean()


class DepthConsistencyLoss(nn.Module):
    """Depth-map consistency loss using MiDaS small."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        import torch.hub
        self.midas = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True
        )
        self.midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        ).small_transform
        for p in self.midas.parameters():
            p.requires_grad_(False)
        self.midas = self.midas.to(device).eval()
        self.device = device

    def _depth(self, t: torch.Tensor) -> torch.Tensor:
        imgs = (t.detach().clamp(-1, 1) * 0.5 + 0.5)
        depths = []
        for i in range(imgs.shape[0]):
            img_np = (imgs[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8")
            inp = self.midas_transforms(img_np).to(self.device)
            if inp.dim() == 3:
                inp = inp.unsqueeze(0)
            with torch.no_grad():
                d = self.midas(inp)[0]
            depths.append(d)
        return torch.stack(depths)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._depth(pred), self._depth(target))


class BoundaryLoss(nn.Module):
    """Edge-boundary consistency loss using Sobel gradients."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device
        ).view(1, 1, 3, 3)

        def edges(t: torch.Tensor) -> torch.Tensor:
            gray = t.float().mean(dim=1, keepdim=True)
            ex = F.conv2d(gray, sobel_x, padding=1)
            ey = F.conv2d(gray, sobel_y, padding=1)
            return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

        return F.l1_loss(edges(pred), edges(target))


class SemanticSegmentationLoss(nn.Module):
    """Semantic consistency loss using a lightweight DeepLab segmentation model."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        import torchvision.models.segmentation as tseg
        self.model = tseg.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model = self.model.to(device=device, dtype=torch.float32).eval()
        self.device = device
        self.dtype = dtype

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inp_pred   = F.interpolate((pred.clamp(-1, 1)   * 0.5 + 0.5), size=(520, 520), mode="bilinear", align_corners=False).float()
        inp_target = F.interpolate((target.clamp(-1, 1) * 0.5 + 0.5), size=(520, 520), mode="bilinear", align_corners=False).float()
        with torch.no_grad():
            feat_pred   = self.model(inp_pred)["out"]
            feat_target = self.model(inp_target)["out"]
        return F.mse_loss(feat_pred, feat_target)


class StageTwoLoss(nn.Module):
    """
    Combined multi-loss for stage 2.

    Default weights (balanced for ~3500 interior images, SDXL LoRA):
        noise_mse   1.00   (~35%)  – primary diffusion objective
        pixel       0.15   (~10%)  – latent-space L1
        perceptual  0.03   (~20%)  – VGG texture matching
        clip        0.00           – disabled (VRAM-heavy)
        boundary    0.50   ( ~3%)  – Sobel edge sharpness
        depth       0.000  (~15%)  – MiDaS depth consistency
        semantic    0.05   ( ~2%)  – DeepLab layout consistency
    """

    def __init__(
        self,
        weights: DictConfig,
        device: torch.device,
        dtype: torch.dtype,
        vae: AutoencoderKL | None = None,
        t_max_aux: int = 600,
    ) -> None:
        super().__init__()
        self.w = weights
        self.vae = vae
        self.t_max_aux = t_max_aux
        self.vae_scale = vae.config.scaling_factor if vae is not None else 0.18215
        self.perceptual = PerceptualLoss(device, dtype)           if weights.get("perceptual", 0) > 0 else None
        self.clip_loss  = CLIPImageLoss(device, dtype)            if weights.get("clip",       0) > 0 else None
        self.boundary   = BoundaryLoss()                          if weights.get("boundary",   0) > 0 else None
        self.depth      = DepthConsistencyLoss(device, dtype)     if weights.get("depth",      0) > 0 else None
        self.semantic   = SemanticSegmentationLoss(device, dtype) if weights.get("semantic",   0) > 0 else None

    def _decode_small(
        self, latents: torch.Tensor, with_grad: bool = True
    ) -> torch.Tensor:
        """Decode latent → RGB pixel space at 1/4 resolution to avoid OOM.

        1024-px SDXL latents are 128×128×4.  Downsampling to 32×32 before
        decode keeps the VAE activation footprint ~16× smaller while still
        giving VGG meaningful 256-px texture features.
        """
        h, w = latents.shape[-2:]
        # Cast to VAE dtype (bf16/fp16) to avoid dtype mismatch during decode.
        # x0_pred from _predict_x0 may be float32 due to mixed-precision arithmetic.
        vae_dtype = next(self.vae.parameters()).dtype
        small = F.interpolate(
            latents.to(dtype=vae_dtype), size=(max(8, h // 4), max(8, w // 4)),
            mode="bilinear", align_corners=False,
        ) / self.vae_scale
        if with_grad:
            imgs = self.vae.decode(small).sample
        else:
            with torch.no_grad():
                imgs = self.vae.decode(small).sample
        return imgs.clamp(-1, 1)

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ):
        losses = {}
        # Mask aux losses for high-noise timesteps (unreliable x0 estimate)
        use_aux = (timesteps is None) or (timesteps < self.t_max_aux).all()

        losses["noise_mse"] = F.mse_loss(noise_pred.float(), noise_target.float(), reduction="mean")

        if use_aux:
            # Latent-space losses (operate directly on 4-ch latent tensors)
            if self.w.get("pixel", 0) > 0:
                losses["pixel"] = F.l1_loss(pred_image, target_image)

            if self.boundary is not None:
                losses["boundary"] = self.boundary(pred_image, target_image)

            # Pixel-space losses: decode latents → 3-ch RGB once and share.
            # pred decoded with grad; target decoded without grad to save memory.
            needs_decode = any([self.perceptual, self.clip_loss, self.depth, self.semantic])
            if needs_decode:
                pred_px = self._decode_small(pred_image,   with_grad=True)
                tgt_px  = self._decode_small(target_image, with_grad=False)

                if self.perceptual is not None:
                    losses["perceptual"] = self.perceptual(pred_px, tgt_px)

                if self.clip_loss is not None:
                    losses["clip"] = self.clip_loss(pred_px, tgt_px)

                if self.depth is not None:
                    losses["depth"] = self.depth(pred_px, tgt_px)

                if self.semantic is not None:
                    losses["semantic"] = self.semantic(pred_px, tgt_px)

        total = sum(self.w.get(k, 0.0) * v for k, v in losses.items())
        return total, losses


# ---------------------------------------------------------------------------
# SDXL text encoding
# ---------------------------------------------------------------------------

def encode_prompt_sdxl(
    batch: dict,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    device: torch.device,
    dtype: torch.dtype,
):
    """Returns (prompt_embeds [B,77,2048], pooled_prompt_embeds [B,1280])."""
    captions = list(batch["caption"]) if isinstance(batch.get("caption"), (list, tuple)) \
               else [batch["caption"]] * batch["pixel_values"].shape[0]

    tokens_1 = tokenizer_1(
        captions, padding="max_length", max_length=tokenizer_1.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    enc1_out = text_encoder_1(tokens_1, output_hidden_states=True)
    prompt_embeds_1 = enc1_out.hidden_states[-2]

    tokens_2 = tokenizer_2(
        captions, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    enc2_out = text_encoder_2(tokens_2, output_hidden_states=True)
    prompt_embeds_2 = enc2_out.hidden_states[-2]
    pooled_embeds   = enc2_out.text_embeds

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    return prompt_embeds.to(dtype=dtype), pooled_embeds.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def log_validation(
    pipeline: StableDiffusionXLInpaintPipeline,
    cfg: DictConfig,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    output_dir: str,
    weight_dtype: torch.dtype = torch.float32,
) -> None:
    if not cfg.validation.validation_prompts:
        return

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
    val_dir = Path(output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image as PILImage
    import numpy as np

    _img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    val_images_dir = Path(cfg.data.dataset_dir) / cfg.data.val_split / "images"
    val_masks_dir  = Path(cfg.data.dataset_dir) / cfg.data.val_split / "masks"
    val_img_paths = sorted(
        p for p in val_images_dir.iterdir() if p.suffix.lower() in _img_exts
    ) if val_images_dir.exists() else []

    images = []
    num_to_show = min(cfg.validation.num_validation_images, len(cfg.validation.validation_prompts))
    res = cfg.training.resolution

    for i, prompt in enumerate(cfg.validation.validation_prompts[:num_to_show]):
        if val_img_paths:
            img_path = val_img_paths[i % len(val_img_paths)]
            pil_img  = PILImage.open(img_path).convert("RGB").resize((res, res), PILImage.LANCZOS)
            mask_path = val_masks_dir / (img_path.stem + ".png")
            if mask_path.exists():
                mask_arr = np.array(
                    PILImage.open(mask_path).convert("L").resize((res, res), PILImage.NEAREST)
                )
            else:
                mask_arr = np.zeros((res, res), dtype=np.uint8)
            if mask_arr.max() == 0:
                mask_arr[res // 4 : 3 * res // 4, res // 4 : 3 * res // 4] = 255
            pil_mask = PILImage.fromarray(mask_arr)
        else:
            pil_img  = PILImage.fromarray(np.full((res, res, 3), 255, dtype=np.uint8))
            mask_arr = np.zeros((res, res), dtype=np.uint8)
            mask_arr[res // 4 : 3 * res // 4, res // 4 : 3 * res // 4] = 255
            pil_mask = PILImage.fromarray(mask_arr)

        with torch.autocast(accelerator.device.type, dtype=weight_dtype):
            out = pipeline(
                prompt=prompt,
                image=pil_img,
                mask_image=pil_mask,
                height=res,
                width=res,
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        save_path = val_dir / f"epoch{epoch:04d}_step{step:07d}_{i}.png"
        out.save(save_path)
        images.append(out)

    if accelerator.is_main_process and cfg.logging.report_to == "wandb" and is_wandb_available():
        import wandb
        accelerator.log(
            {"validation": [wandb.Image(img, caption=p)
                            for img, p in zip(images, cfg.validation.validation_prompts[:len(images)])]},
            step=step,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _decode_latents(vae: AutoencoderKL, latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode latents without gradient (use for fixed reference targets)."""
    latents = latents.to(dtype=dtype) / vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(latents).sample
    return images.clamp(-1, 1)


def _predict_x0(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """Estimate clean latent x₀ from noise/v-prediction (retains gradient through noise_pred)."""
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps][:, None, None, None]
    if scheduler.config.prediction_type == "epsilon":
        return (noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    elif scheduler.config.prediction_type == "v_prediction":
        return alpha_t.sqrt() * noisy_latents - (1.0 - alpha_t).sqrt() * noise_pred
    return noisy_latents


def _build_add_time_ids(
    original_size: tuple,
    crops_coords_top_left: tuple,
    target_size: tuple,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)
    t = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return t.repeat(batch_size, 1)


def _load_stage1_weights(
    unet: UNet2DConditionModel, cfg: DictConfig, stage2_cfg: DictConfig
) -> UNet2DConditionModel:
    """Load stage-1 LoRA via PeftModel.from_pretrained. Returns a trainable PeftModel."""
    from peft import PeftModel
    stage1_path = stage2_cfg.get("stage1_lora_path", "auto")
    if stage1_path == "auto":
        stage1_lora_dir = Path(cfg.two_stage.stage1.output_dir) / "unet_lora"
        if not stage1_lora_dir.exists():
            ckpts = sorted(
                [d for d in Path(cfg.two_stage.stage1.output_dir).iterdir()
                 if d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if ckpts:
                stage1_lora_dir = ckpts[-1]
        stage1_path = str(stage1_lora_dir)

    if not Path(stage1_path).exists():
        logger.warning(
            f"Stage-1 LoRA path not found: {stage1_path}. "
            "Stage 2 will initialise from a fresh LoRA instead."
        )
        return enable_lora(unet, cfg)
    try:
        peft_unet = PeftModel.from_pretrained(unet, stage1_path, is_trainable=True)
        peft_unet.print_trainable_parameters()
        logger.info(f"Loaded stage-1 LoRA from {stage1_path}")
        return peft_unet
    except Exception as exc:
        logger.warning(f"Could not load stage-1 LoRA: {exc}. Falling back to fresh LoRA.")
        return enable_lora(unet, cfg)


def _build_pipeline(
    accelerator, unet, vae,
    text_encoder_1, text_encoder_2,
    tokenizer_1, tokenizer_2,
    noise_scheduler, pretrained, weight_dtype, cfg,
) -> StableDiffusionXLInpaintPipeline:
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
        pretrained,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder_1,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


# ---------------------------------------------------------------------------
# Core training loop (single stage)
# ---------------------------------------------------------------------------

def train_one_stage(
    cfg: DictConfig,
    stage: int,
    stage_cfg: DictConfig,
    accelerator: Accelerator,
    pretrained: str,
    weight_dtype: torch.dtype,
) -> None:
    output_dir = stage_cfg.output_dir
    resolution = cfg.training.resolution

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    set_seed(cfg.training.seed)

    # ── Models ────────────────────────────────────────────────────────────
    tokenizer_1     = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    tokenizer_2     = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer_2")
    text_encoder_1  = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder")
    text_encoder_2  = CLIPTextModelWithProjection.from_pretrained(pretrained, subfolder="text_encoder_2")
    vae             = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
    unet            = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # ── LoRA ──────────────────────────────────────────────────────────────
    if cfg.lora.enabled:
        if stage == 2:
            # _load_stage1_weights wraps unet with PEFT and loads stage-1 weights.
            # Must be done BEFORE any other PEFT call; returns a PeftModel.
            unet = _load_stage1_weights(unet, cfg, stage_cfg)
        else:
            unet = enable_lora(unet, cfg)
    else:
        unet.requires_grad_(True)

    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if cfg.training.enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("xFormers memory-efficient attention enabled.")
        except Exception:
            logger.warning("xFormers not available – using standard attention.")

    # ── Stage-2 multi-loss ─────────────────────────────────────────────────
    stage2_loss_fn = None
    if stage == 2:
        stage2_loss_fn = StageTwoLoss(
            weights=stage_cfg.loss_weights,
            device=accelerator.device,
            dtype=weight_dtype,
            vae=vae,
            t_max_aux=int(OmegaConf.select(stage_cfg, "t_max_aux", default=600)),
        )

    # ── Optimiser ─────────────────────────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    logger.info(f"[Stage {stage}] Trainable params: {sum(p.numel() for p in trainable_params):,}")

    if cfg.training.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not available – using standard AdamW.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    # Per-stage overrides take priority over shared training config.
    _beta2     = float(OmegaConf.select(stage_cfg, "adam_beta2",   default=cfg.training.adam_beta2))
    _grad_norm = float(OmegaConf.select(stage_cfg, "max_grad_norm", default=cfg.training.max_grad_norm))
    _snr_gamma = float(OmegaConf.select(stage_cfg, "snr_gamma",    default=0.0))
    logger.info(f"[Stage {stage}] adam_beta2={_beta2}  max_grad_norm={_grad_norm}  snr_gamma={_snr_gamma}")

    optimizer = optimizer_cls(
        trainable_params,
        lr=stage_cfg.learning_rate,
        betas=(cfg.training.adam_beta1, _beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    train_dataset = InteriorInpaintingDataset(
        dataset_dir=cfg.data.dataset_dir,
        split=cfg.data.train_split,
        tokenizer=None,          # SDXL uses dual tokenizer; raw captions returned by dataset
        size=resolution,
        mask_type=cfg.data.mask_type,
        mask_min_area=cfg.data.mask_min_area,
        mask_max_area=cfg.data.mask_max_area,
        center_crop=cfg.data.center_crop,
        random_flip=cfg.data.random_flip,
        default_caption=cfg.data.default_caption,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── LR scheduler ───────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    max_train_steps = stage_cfg.get("max_train_steps") or (
        stage_cfg.num_train_epochs * num_update_steps_per_epoch
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=stage_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=stage_cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # ── Prepare with Accelerator ───────────────────────────────────────────
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # ── Logging ────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        accelerator.init_trackers(
            f"{cfg.logging.run_name}-stage{stage}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info(f"***** Stage {stage} training *****")
    logger.info(f"  Examples  = {len(train_dataset)}")
    logger.info(f"  Epochs    = {num_train_epochs}")
    logger.info(f"  Batch     = {cfg.training.train_batch_size}")
    logger.info(f"  Grad acc  = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Max steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    # ── Resume ─────────────────────────────────────────────────────────────
    _REQUIRED_CKPT_FILES = {"optimizer.bin", "scheduler.bin", "random_states_0.pkl"}

    def _is_valid_checkpoint(path: str) -> bool:
        """Return True only if the checkpoint directory has all required files."""
        p = Path(path)
        if not p.is_dir():
            return False
        missing = _REQUIRED_CKPT_FILES - {f.name for f in p.iterdir()}
        if missing:
            logger.warning(f"Incomplete checkpoint {path} – missing: {missing}. Skipping.")
            return False
        return True

    resume_ckpt = stage_cfg.get("resume_from_checkpoint")
    if resume_ckpt:
        ckpt = resume_ckpt
        if ckpt == "latest":
            dirs = sorted(
                [d for d in Path(output_dir).iterdir() if d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
                reverse=True,  # newest first – iterate until we find a valid one
            )
            ckpt = None
            for candidate in dirs:
                if _is_valid_checkpoint(str(candidate)):
                    ckpt = str(candidate)
                    break
            if ckpt is None and dirs:
                logger.warning("No valid checkpoint found – starting from scratch.")

        if ckpt and _is_valid_checkpoint(ckpt):
            accelerator.load_state(ckpt)
            state_file = Path(ckpt) / "training_state.json"
            if state_file.exists():
                with open(state_file) as fh:
                    saved = json.load(fh)
                global_step = saved["global_step"]
                first_epoch = saved["epoch"]
            else:
                global_step = int(Path(ckpt).name.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step - first_epoch * num_update_steps_per_epoch
            logger.info(f"Resumed from {ckpt} (step={global_step}, epoch={first_epoch})")
        elif ckpt:
            logger.warning(f"Checkpoint {ckpt} is invalid – starting from scratch.")

    # ── Training loop ───────────────────────────────────────────────────────
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc=f"Stage {stage}",
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        loss_acum: dict = {}

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
                # ── Encode to latents ──────────────────────────────────
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["masked_image"].to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                mask = F.interpolate(
                    batch["mask"].to(dtype=weight_dtype),
                    size=latents.shape[-2:],
                    mode="nearest",
                )

                # ── Noise & timesteps ──────────────────────────────────
                noise     = torch.randn_like(latents)
                bsz       = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # ── SDXL text conditioning ─────────────────────────────
                prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
                    batch, text_encoder_1, text_encoder_2,
                    tokenizer_1, tokenizer_2,
                    accelerator.device, weight_dtype,
                )
                add_time_ids = _build_add_time_ids(
                    original_size=(resolution, resolution),
                    crops_coords_top_left=(0, 0),
                    target_size=(resolution, resolution),
                    dtype=weight_dtype,
                    device=accelerator.device,
                    batch_size=bsz,
                )
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }

                # SDXL inpainting UNet: 9-channel input
                unet_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                model_pred = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # ── Target ─────────────────────────────────────────────
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                # ── Loss ───────────────────────────────────────────────
                if stage == 1 or stage2_loss_fn is None:
                    if _snr_gamma > 0:
                        # Min-SNR weighting: prevents high-noise timesteps from
                        # dominating the gradient signal.  Ref: Hang et al. 2023.
                        alphas_cp = noise_scheduler.alphas_cumprod.to(latents.device, dtype=torch.float32)
                        snr = alphas_cp[timesteps] / (1.0 - alphas_cp[timesteps])
                        mse_weights = (torch.stack([snr, _snr_gamma * torch.ones_like(snr)], dim=1)
                                       .min(dim=1)[0] / snr)
                        raw = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        raw = raw.mean(dim=list(range(1, raw.ndim)))  # [B]
                        loss = (raw * mse_weights).mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    step_comps = {"noise_mse": loss.item()}
                else:
                    # Compute x₀ estimate in latent space.
                    # Clamped to [-4, 4] to prevent blow-up at high timesteps.
                    # Aux losses are further masked by timestep inside stage2_loss_fn.
                    x0_pred = _predict_x0(noisy_latents, model_pred, timesteps, noise_scheduler)
                    loss, step_comps = stage2_loss_fn(model_pred, target, x0_pred, latents, timesteps)
                    step_comps = {k: v.item() if torch.is_tensor(v) else v for k, v in step_comps.items()}

                avg_loss = accelerator.gather(loss.repeat(cfg.training.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.training.gradient_accumulation_steps
                for k, v in step_comps.items():
                    loss_acum[k] = loss_acum.get(k, 0.0) + v / cfg.training.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, _grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.logging.log_every_n_steps == 0:
                        log_dict = {"train/loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
                        log_dict.update({f"train/{k}": v for k, v in loss_acum.items()})
                        accelerator.log(log_dict, step=global_step)
                        train_loss = 0.0
                        loss_acum = {}

                    if global_step % stage_cfg.checkpointing_steps == 0:
                        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(ckpt_dir)
                        with open(os.path.join(ckpt_dir, "training_state.json"), "w") as fh:
                            json.dump({"global_step": global_step, "epoch": epoch, "stage": stage}, fh)
                        ckpts = sorted(
                            [d for d in Path(output_dir).iterdir() if d.name.startswith("checkpoint-")],
                            key=lambda d: int(d.name.split("-")[1]),
                        )
                        for old in ckpts[:-3]:
                            shutil.rmtree(old)

                    if global_step % cfg.validation.validation_steps == 0:
                        pipeline = _build_pipeline(
                            accelerator, unet, vae,
                            text_encoder_1, text_encoder_2,
                            tokenizer_1, tokenizer_2,
                            noise_scheduler, pretrained, weight_dtype, cfg
                        )
                        log_validation(pipeline, cfg, accelerator, epoch, global_step, output_dir, weight_dtype)
                        del pipeline

            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])

            if global_step >= max_train_steps:
                break

    # ── Save final model ────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)

        if cfg.lora.enabled:
            lora_out = os.path.join(output_dir, "unet_lora")
            unet_unwrapped.save_pretrained(lora_out)
            logger.info(f"[Stage {stage}] LoRA weights saved to {lora_out}")
        else:
            pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                pretrained,
                unet=unet_unwrapped,
                vae=vae,
                text_encoder=text_encoder_1,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer_1,
                tokenizer_2=tokenizer_2,
                torch_dtype=weight_dtype,
            )
            pipeline.save_pretrained(output_dir)

        save_model_card(output_dir, pretrained, cfg.data.dataset_dir, cfg.lora.enabled, stage)

    accelerator.end_training()
    logger.info(f"[Stage {stage}] Complete.")


# ---------------------------------------------------------------------------
# Accelerator factory
# ---------------------------------------------------------------------------

def _make_accelerator(cfg: DictConfig, stage: int) -> Accelerator:
    stage_cfg = cfg.two_stage.stage1 if stage == 1 else cfg.two_stage.stage2
    output_dir = stage_cfg.output_dir if OmegaConf.select(cfg, "two_stage.enabled", default=False) \
                 else cfg.training.output_dir
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
    )
    return Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to if cfg.logging.report_to != "none" else None,
        project_config=project_config,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg: DictConfig, run_stage: int | None = None) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    pretrained = cfg.model.pretrained_model_name_or_path
    weight_dtype = torch.float32
    if cfg.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    two_stage_enabled = OmegaConf.select(cfg, "two_stage.enabled", default=False)

    if two_stage_enabled:
        stages_to_run = [run_stage] if run_stage else [1, 2]
        for s in stages_to_run:
            stage_cfg  = cfg.two_stage.stage1 if s == 1 else cfg.two_stage.stage2
            accelerator = _make_accelerator(cfg, s)
            train_one_stage(cfg, s, stage_cfg, accelerator, pretrained, weight_dtype)
    else:
        # Single-stage fallback using top-level training config
        stage_cfg = OmegaConf.create({
            "output_dir":             cfg.training.output_dir,
            "num_train_epochs":       cfg.training.num_train_epochs,
            "max_train_steps":        cfg.training.max_train_steps,
            "checkpointing_steps":    cfg.training.checkpointing_steps,
            "resume_from_checkpoint": cfg.training.resume_from_checkpoint,
            "learning_rate":          cfg.training.learning_rate,
            "lr_scheduler":           cfg.training.lr_scheduler,
            "lr_warmup_steps":        cfg.training.lr_warmup_steps,
        })
        accelerator = _make_accelerator(cfg, 1)
        train_one_stage(cfg, run_stage or 1, stage_cfg, accelerator, pretrained, weight_dtype)

    if OmegaConf.select(cfg, "push_to_hub.enabled", default=False) and cfg.push_to_hub.hub_model_id:
        final_dir = cfg.two_stage.stage2.output_dir if two_stage_enabled else cfg.training.output_dir
        from huggingface_hub import HfApi
        api = HfApi(token=cfg.push_to_hub.hub_token)
        api.upload_folder(
            folder_path=final_dir,
            repo_id=cfg.push_to_hub.hub_model_id,
            repo_type="model",
        )

    logger.info("All training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SDXL inpainting – Two-stage LoRA")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--no_lora", action="store_true",
        help="Disable LoRA and perform full fine-tuning",
    )
    parser.add_argument(
        "--stage", type=int, choices=[1, 2], default=None,
        help="Run only stage 1 or stage 2 (default: run both sequentially)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_lora:
        cfg.lora.enabled = False

    main(cfg, run_stage=args.stage)

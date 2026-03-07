"""
Fine-tuning script for Stable Diffusion Inpainting on interior-design images.

Supports:
  • LoRA fine-tuning (default, memory-efficient)
  • Full fine-tuning (--no_lora flag)

Usage
-----
# Single GPU
python src/train.py --config configs/train_config.yaml

# Multi-GPU with accelerate
accelerate launch --num_processes=4 src/train.py --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import is_wandb_available
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import InteriorInpaintingDataset

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load and return an OmegaConf DictConfig from a YAML file."""
    return OmegaConf.load(path)


def save_model_card(
    output_dir: str,
    base_model: str,
    dataset_dir: str,
    lora_enabled: bool,
) -> None:
    """Write a simple model card to the output directory."""
    card = f"""---
base_model: {base_model}
tags:
  - stable-diffusion
  - inpainting
  - interior-design
  - fine-tuned
  - {"lora" if lora_enabled else "full-fine-tune"}
license: creativeml-openrail-m
---

# Stable Diffusion Inpainting – Interior Design

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) on an interior-design
inpainting dataset.

Training dataset: `{dataset_dir}`

{"This checkpoint uses **LoRA** adapters." if lora_enabled else "This is a **full fine-tune**."}
"""
    with open(os.path.join(output_dir, "README.md"), "w") as fh:
        fh.write(card)


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def enable_lora(unet: UNet2DConditionModel, cfg) -> None:
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
# Validation
# ---------------------------------------------------------------------------

def log_validation(
    pipeline: StableDiffusionInpaintPipeline,
    cfg,
    accelerator: Accelerator,
    epoch: int,
    step: int,
) -> None:
    """Run the pipeline on fixed validation prompts and log images."""
    if not cfg.validation.validation_prompts:
        return

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed)
    val_dir = Path(cfg.training.output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for i, prompt in enumerate(cfg.validation.validation_prompts[: cfg.validation.num_validation_images]):
        # Use a white dummy image + centre-rectangle mask for visual tracking
        dummy_img = torch.ones(1, 3, cfg.training.resolution, cfg.training.resolution)
        dummy_mask = torch.zeros(1, 1, cfg.training.resolution, cfg.training.resolution)
        h, w = cfg.training.resolution, cfg.training.resolution
        dummy_mask[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0

        from PIL import Image as PILImage
        import numpy as np

        pil_img = PILImage.fromarray(
            ((dummy_img[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        )
        pil_mask = PILImage.fromarray(
            (dummy_mask[0, 0].numpy() * 255).astype(np.uint8)
        )

        out = pipeline(
            prompt=prompt,
            image=pil_img,
            mask_image=pil_mask,
            height=cfg.training.resolution,
            width=cfg.training.resolution,
            num_inference_steps=20,
            generator=generator,
        ).images[0]

        save_path = val_dir / f"epoch{epoch:04d}_step{step:07d}_{i}.png"
        out.save(save_path)
        images.append(out)

    if accelerator.is_main_process and cfg.logging.report_to == "wandb" and is_wandb_available():
        import wandb

        accelerator.log(
            {
                "validation": [
                    wandb.Image(img, caption=prompt)
                    for img, prompt in zip(
                        images, cfg.validation.validation_prompts[: len(images)]
                    )
                ]
            },
            step=step,
        )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ── Accelerator ──────────────────────────────────────────────────────
    project_config = ProjectConfiguration(
        project_dir=cfg.training.output_dir,
        logging_dir=os.path.join(cfg.training.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.logging.report_to if cfg.logging.report_to != "none" else None,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.training.output_dir, exist_ok=True)

    set_seed(cfg.training.seed)

    # ── Models ───────────────────────────────────────────────────────────
    pretrained = cfg.model.pretrained_model_name_or_path
    tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if cfg.lora.enabled:
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

    # ── Optimiser ────────────────────────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    if cfg.training.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not available – using standard AdamW.")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params,
        lr=cfg.training.learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    train_dataset = InteriorInpaintingDataset(
        dataset_dir=cfg.data.dataset_dir,
        split=cfg.data.train_split,
        tokenizer=tokenizer,
        size=cfg.data.image_size,
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

    # ── LR Scheduler ─────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    max_train_steps = cfg.training.max_train_steps or (
        cfg.training.num_train_epochs * num_update_steps_per_epoch
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # ── Prepare with Accelerator ─────────────────────────────────────────
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # ── Logging ──────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        accelerator.init_trackers(
            cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {num_train_epochs}")
    logger.info(f"  Batch size (per device) = {cfg.training.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # ── Resume ───────────────────────────────────────────────────────────
    if cfg.training.resume_from_checkpoint:
        ckpt = cfg.training.resume_from_checkpoint
        if ckpt == "latest":
            dirs = sorted(
                [d for d in Path(cfg.training.output_dir).iterdir() if d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            ckpt = str(dirs[-1]) if dirs else None

        if ckpt:
            accelerator.load_state(ckpt)
            global_step = int(Path(ckpt).name.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed from checkpoint: {ckpt} (global step {global_step})")

    # ── Training loop ────────────────────────────────────────────────────
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(unet):
                # Encode images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["masked_image"].to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                # Resize mask to latent resolution
                mask = F.interpolate(
                    batch["mask"].to(dtype=weight_dtype),
                    size=latents.shape[-2:],
                    mode="nearest",
                )

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device)
                )[0]

                # Concatenate noisy latents, mask, and masked image latents
                # along channel dim – as required by SD inpainting UNet (9 channels)
                unet_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                # Predict noise
                model_pred = unet(
                    unet_input, timesteps, encoder_hidden_states
                ).sample

                # Compute loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather for logging
                avg_loss = accelerator.gather(loss.repeat(cfg.training.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.training.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.logging.log_every_n_steps == 0:
                        accelerator.log(
                            {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]},
                            step=global_step,
                        )
                        train_loss = 0.0

                    # Checkpointing
                    if global_step % cfg.training.checkpointing_steps == 0:
                        ckpt_dir = os.path.join(
                            cfg.training.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(ckpt_dir)
                        # Keep only the 3 most recent checkpoints
                        ckpts = sorted(
                            [
                                d for d in Path(cfg.training.output_dir).iterdir()
                                if d.name.startswith("checkpoint-")
                            ],
                            key=lambda d: int(d.name.split("-")[1]),
                        )
                        for old in ckpts[:-3]:
                            shutil.rmtree(old)

                    # Validation
                    if global_step % cfg.validation.validation_steps == 0:
                        pipeline = _build_pipeline(
                            accelerator, unet, vae, text_encoder, tokenizer,
                            noise_scheduler, pretrained, weight_dtype, cfg
                        )
                        log_validation(pipeline, cfg, accelerator, epoch, global_step)
                        del pipeline

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # ── Save final model ─────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)

        if cfg.lora.enabled:
            unet_unwrapped.save_pretrained(
                os.path.join(cfg.training.output_dir, "unet_lora")
            )
        else:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                pretrained,
                unet=unet_unwrapped,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                safety_checker=None,
            )
            pipeline.save_pretrained(cfg.training.output_dir)

        save_model_card(
            cfg.training.output_dir,
            pretrained,
            cfg.data.dataset_dir,
            cfg.lora.enabled,
        )

        if cfg.push_to_hub.enabled and cfg.push_to_hub.hub_model_id:
            from huggingface_hub import HfApi

            api = HfApi(token=cfg.push_to_hub.hub_token)
            api.upload_folder(
                folder_path=cfg.training.output_dir,
                repo_id=cfg.push_to_hub.hub_model_id,
                repo_type="model",
            )

    accelerator.end_training()
    logger.info("Training complete.")


def _build_pipeline(
    accelerator, unet, vae, text_encoder, tokenizer, noise_scheduler,
    pretrained, weight_dtype, cfg
) -> StableDiffusionInpaintPipeline:
    """Assemble a pipeline for validation inference."""
    unet_unwrapped = accelerator.unwrap_model(unet)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained,
        unet=unet_unwrapped,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SD inpainting on interior images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA and perform full fine-tuning",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_lora:
        cfg.lora.enabled = False

    main(cfg)

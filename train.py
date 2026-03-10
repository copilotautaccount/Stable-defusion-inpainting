"""
Fine-tune a Stable Diffusion Inpainting model on a custom dataset.

Usage example
-------------
python train.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
    --data_dir ./data \
    --output_dir ./output \
    --num_train_epochs 10 \
    --train_batch_size 2 \
    --learning_rate 1e-5 \
    --image_size 512 \
    --save_steps 500 \
    --mixed_precision fp16

After training the fine-tuned weights are saved under ``output_dir`` and can be
loaded directly with ``inference.py``.
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import InpaintingDataset

logger = get_logger(__name__, log_level="INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion for inpainting"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="HuggingFace model id or path to local pretrained model directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (must contain images/ and masks/ sub-dirs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory where checkpoints and the final model are saved.",
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Training image resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size for training."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10, help="Total number of epochs."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Overrides num_train_epochs when set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after warmup).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of warmup steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save a checkpoint every N global steps.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer (requires bitsandbytes).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model components
    # ------------------------------------------------------------------ #
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Freeze VAE and text encoder – only UNet is trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for --use_8bit_adam. "
                "Install with: pip install bitsandbytes"
            )
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # ------------------------------------------------------------------ #
    # Dataset & DataLoader
    # ------------------------------------------------------------------ #
    train_dataset = InpaintingDataset(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        tokenizer=tokenizer,
        augment=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # Scheduler & step counts
    # ------------------------------------------------------------------ #
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ------------------------------------------------------------------ #
    # Prepare with Accelerate
    # ------------------------------------------------------------------ #
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    accelerator.init_trackers("stable_diffusion_inpainting")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples            = {len(train_dataset)}")
    logger.info(f"  Num epochs              = {args.num_train_epochs}")
    logger.info(f"  Batch size per device   = {args.train_batch_size}")
    logger.info(f"  Total batch size        = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(weight_dtype)
                mask_values = batch["mask_values"].to(weight_dtype)
                masked_pixel_values = batch["masked_pixel_values"].to(weight_dtype)

                # Encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                masked_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                # Resize mask to match latent dimensions
                mask_latent = F.interpolate(
                    mask_values,
                    size=(latents.shape[2], latents.shape[3]),
                    mode="nearest",
                )

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate latents with mask and masked image for inpainting
                latent_model_input = torch.cat(
                    [noisy_latents, mask_latent, masked_latents], dim=1
                )

                # Encode text
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                noise_pred = unet(
                    latent_model_input, timesteps, encoder_hidden_states
                ).sample

                # Compute loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction_type "
                        f"{noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if (
                    accelerator.is_main_process
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    checkpoint_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    accelerator.save_state(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # ------------------------------------------------------------------ #
    # Save final pipeline
    # ------------------------------------------------------------------ #
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet_unwrapped,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Saved final pipeline to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

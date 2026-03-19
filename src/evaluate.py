"""
Evaluation & Comparison: Base Model vs Fine-tuned Model
========================================================
Runs inference on the validation set with **both** the base SDXL inpainting
model and the LoRA fine-tuned model, computes quality metrics, renders
side-by-side comparison grids, and writes a summary CSV + HTML report.

Metrics computed
----------------
* SSIM  (Structural Similarity Index) — full image, higher ↑ is better
* PSNR  (Peak Signal-to-Noise Ratio) — full image, higher ↑ is better (dB)
* LPIPS (Learned Perceptual Image Patch Similarity) — lower ↓ is better
         requires:  pip install lpips
* CLIP Score — cosine similarity between generated image and text prompt
               higher ↑ is better (uses openai/clip-vit-base-patch32)
* Masked L1 / L2 — pixel error restricted to the inpainted region, lower ↓

Memory strategy
---------------
To avoid holding two SDXL pipelines in VRAM simultaneously, inference is
run in two sequential passes — base first, fine-tuned second — then metrics
are computed from the saved images.

Usage
-----
# Quick smoke-test (10 images, 512 px)
python src/evaluate.py \\
    --num_samples 10 \\
    --resolution  512 \\
    --steps       20

# Full validation run (default args match train_config.yaml)
python src/evaluate.py

# Custom paths
python src/evaluate.py \\
    --base_model  diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \\
    --lora_dir    outputs/interior-inpainting-sdxl/stage2/unet_lora \\
    --image_dir   data/interior/val/images \\
    --mask_dir    data/interior/val/masks \\
    --captions    data/interior/val/captions.json \\
    --output_dir  outputs/evaluation \\
    --num_samples 50 \\
    --resolution  1024
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    """Convert PIL image to [1, 3, H, W] float32 tensor in [0, 1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    from skimage.metrics import structural_similarity as _ssim
    a = np.array(img_a.convert("RGB"))
    b = np.array(img_b.convert("RGB"))
    return float(_ssim(a, b, data_range=255, channel_axis=2))


def compute_psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    a = np.array(img_a.convert("RGB"))
    b = np.array(img_b.convert("RGB"))
    return float(_psnr(a, b, data_range=255))


def compute_lpips(
    img_a: Image.Image,
    img_b: Image.Image,
    lpips_fn,
    device: str,
) -> float:
    """LPIPS (lower = more similar). Inputs scaled to [-1, 1]."""
    a = _pil_to_tensor(img_a, device) * 2.0 - 1.0
    b = _pil_to_tensor(img_b, device) * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_fn(a, b).item())


def compute_clip_score(
    img: Image.Image,
    prompt: str,
    clip_model,
    clip_processor,
    device: str,
) -> float:
    """CLIP image–text cosine similarity (higher is better)."""
    inputs = clip_processor(
        text=[prompt], images=img, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    return float(out.logits_per_image.item())


def compute_masked_metrics(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Dict[str, float]:
    """
    Compute L1 / L2 pixel error restricted to the inpainted region.
    mask: L-mode image where 255 = inpaint region.
    """
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    gen  = np.array(generated.convert("RGB"), dtype=np.float32)
    m    = np.array(mask.convert("L")) > 127        # H×W boolean

    if m.sum() == 0:
        return {"masked_l1": 0.0, "masked_l2": 0.0}

    flat_orig = orig[m]
    flat_gen  = gen[m]
    l1 = float(np.mean(np.abs(flat_orig - flat_gen)) / 255.0)
    l2 = float(np.sqrt(np.mean((flat_orig - flat_gen) ** 2)) / 255.0)
    return {"masked_l1": l1, "masked_l2": l2}


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _is_sdxl(model_path: str) -> bool:
    keywords = ["xl", "sdxl", "stable-diffusion-xl"]
    return any(k in model_path.lower() for k in keywords)


def _build_pipeline(model_id: str, lora_dir: Optional[str], device: str, dtype: torch.dtype):
    """Build an (optionally LoRA-enhanced) inpainting pipeline."""
    if _is_sdxl(model_id):
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    else:
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )

    if lora_dir:
        pipe.load_lora_weights(lora_dir)

    pipe.enable_attention_slicing()
    return pipe.to(device)


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

def _run_one(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
    is_sdxl: bool,
) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    kwargs: Dict = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    )
    if is_sdxl:
        kwargs["height"] = image.height
        kwargs["width"]  = image.width
    return pipe(**kwargs).images[0]


# ---------------------------------------------------------------------------
# Inference pass (one model at a time to save VRAM)
# ---------------------------------------------------------------------------

def _inference_pass(
    model_id: str,
    lora_dir: Optional[str],
    image_paths: List[Path],
    mask_root: Path,
    captions: Dict[str, str],
    default_prompt: str,
    out_dir: Path,
    args,
) -> None:
    """Run inference for all images and save results to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    dtype  = torch.bfloat16 if "cuda" in device else torch.float32
    is_sdxl = _is_sdxl(model_id)

    label = "Fine-tuned" if lora_dir else "Base"
    print(f"\n[INFO] Loading {label} pipeline ...")
    pipe = _build_pipeline(model_id, lora_dir, device, dtype)

    neg_prompt = (
        "blurry, low quality, distorted, ugly, out of focus, "
        "bad anatomy, watermark, text, oversaturated"
    )

    for img_path in tqdm(image_paths, desc=f"Inferring ({label})"):
        out_path = out_dir / img_path.name
        if out_path.exists() and not args.overwrite:
            continue  # resume-friendly

        mask_path = mask_root / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = mask_root / img_path.name
        if not mask_path.exists():
            tqdm.write(f"  [WARN] No mask for {img_path.name}, skipping.")
            continue

        image = Image.open(img_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS
        )
        mask = Image.open(mask_path).convert("L").resize(
            (args.resolution, args.resolution), Image.NEAREST
        )
        prompt = captions.get(img_path.name, captions.get(img_path.stem, default_prompt))

        result = _run_one(
            pipe, image, mask, prompt, neg_prompt,
            args.steps, args.guidance, args.strength, args.seed, is_sdxl,
        )
        result.save(out_path)

    # Free VRAM before loading next model
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Comparison grid visualisation
# ---------------------------------------------------------------------------

def _make_grid(
    original: Image.Image,
    mask: Image.Image,
    base_out: Image.Image,
    ft_out: Image.Image,
    metrics_base: Dict,
    metrics_ft: Dict,
) -> Image.Image:
    """4-panel grid: Original | Mask | Base | Fine-tuned with metric overlay."""
    W, H   = original.size
    pad    = 6
    lbl_h  = 28
    n_cols = 4
    cw     = n_cols * W + (n_cols + 1) * pad
    ch     = H + lbl_h + 2 * pad

    canvas = Image.new("RGB", (cw, ch), color=(20, 20, 20))
    draw   = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
    except Exception:
        font = ImageFont.load_default()

    def _metric_str(m: Dict) -> str:
        parts = []
        if "ssim" in m:
            parts.append(f"SSIM {m['ssim']:.3f}")
        if "psnr" in m:
            parts.append(f"PSNR {m['psnr']:.1f}")
        if "lpips" in m:
            parts.append(f"LPIPS {m['lpips']:.3f}")
        return "  ".join(parts)

    panels = [
        ("Original", original.convert("RGB"), {}),
        ("Mask",     mask.convert("RGB"),     {}),
        ("Base",     base_out.convert("RGB"), metrics_base),
        ("Fine-tuned", ft_out.convert("RGB"), metrics_ft),
    ]

    for i, (label, img, mets) in enumerate(panels):
        x0 = pad + i * (W + pad)
        y0 = pad + lbl_h
        canvas.paste(img.resize((W, H)), (x0, y0))

        # Label header
        full_label = f"{label}  {_metric_str(mets)}" if mets else label
        draw.text((x0 + 4, pad + 6), full_label, fill=(255, 230, 60), font=font)

    return canvas


# ---------------------------------------------------------------------------
# Metric computation pass (no model loaded)
# ---------------------------------------------------------------------------

def _compute_all_metrics(
    image_paths: List[Path],
    mask_root: Path,
    captions: Dict[str, str],
    default_prompt: str,
    base_dir: Path,
    ft_dir: Path,
    args,
) -> List[Dict]:

    device = args.device

    # -- optional LPIPS ---------------------------------------------------
    lpips_fn = None
    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net="alex").to(device).eval()
        print("[INFO] LPIPS (alex) loaded.")
    except ImportError:
        warnings.warn(
            "lpips not installed → LPIPS skipped.  Install with: pip install lpips"
        )

    # -- optional CLIP ----------------------------------------------------
    clip_model = clip_proc = None
    if not args.skip_clip:
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("[INFO] CLIP (ViT-B/32) loaded.")
        except Exception as e:
            warnings.warn(f"CLIP not available: {e}")

    records: List[Dict] = []
    grid_dir = Path(args.output_dir) / "comparisons"
    grid_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Computing metrics"):
        base_path = base_dir / img_path.name
        ft_path   = ft_dir   / img_path.name
        if not base_path.exists() or not ft_path.exists():
            tqdm.write(f"  [WARN] Missing output for {img_path.name}, skipping metrics.")
            continue

        mask_path = mask_root / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = mask_root / img_path.name

        original = Image.open(img_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS
        )
        mask = Image.open(mask_path).convert("L").resize(
            (args.resolution, args.resolution), Image.NEAREST
        ) if mask_path.exists() else Image.new("L", original.size, 255)

        base_out = Image.open(base_path).convert("RGB")
        ft_out   = Image.open(ft_path).convert("RGB")

        prompt = captions.get(img_path.name, captions.get(img_path.stem, default_prompt))

        row: Dict = {"image": img_path.name, "prompt": prompt[:100]}

        # SSIM
        row["base_ssim"] = compute_ssim(original, base_out)
        row["ft_ssim"]   = compute_ssim(original, ft_out)

        # PSNR
        row["base_psnr"] = compute_psnr(original, base_out)
        row["ft_psnr"]   = compute_psnr(original, ft_out)

        # Masked L1/L2
        base_m = compute_masked_metrics(original, base_out, mask)
        ft_m   = compute_masked_metrics(original, ft_out,   mask)
        row["base_masked_l1"] = base_m["masked_l1"]
        row["ft_masked_l1"]   = ft_m["masked_l1"]
        row["base_masked_l2"] = base_m["masked_l2"]
        row["ft_masked_l2"]   = ft_m["masked_l2"]

        # LPIPS
        if lpips_fn is not None:
            row["base_lpips"] = compute_lpips(original, base_out, lpips_fn, device)
            row["ft_lpips"]   = compute_lpips(original, ft_out,   lpips_fn, device)

        # CLIP
        if clip_model is not None:
            row["base_clip"] = compute_clip_score(base_out, prompt, clip_model, clip_proc, device)
            row["ft_clip"]   = compute_clip_score(ft_out,   prompt, clip_model, clip_proc, device)

        records.append(row)

        # Comparison grid
        grid = _make_grid(
            original, mask, base_out, ft_out,
            metrics_base={k.replace("base_", ""): row[k] for k in ("base_ssim", "base_psnr") if k in row},
            metrics_ft  ={k.replace("ft_", ""):   row[k] for k in ("ft_ssim",   "ft_psnr")   if k in row},
        )
        grid.save(grid_dir / img_path.name)

    return records


# ---------------------------------------------------------------------------
# Summary printing & HTML report
# ---------------------------------------------------------------------------

_METRIC_DEFS = [
    ("ssim",       "SSIM",         "↑"),
    ("psnr",       "PSNR (dB)",    "↑"),
    ("lpips",      "LPIPS",        "↓"),
    ("clip",       "CLIP Score",   "↑"),
    ("masked_l1",  "Masked L1",    "↓"),
    ("masked_l2",  "Masked L2",    "↓"),
]


def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    w = 62
    print("\n" + "=" * w)
    print(f"  Evaluation Summary  ({n} images)")
    print("=" * w)
    print(f"  {'Metric':<18} {'Base':>12} {'Fine-tuned':>12} {'Δ':>10}  {'Better?':>8}")
    print("-" * w)
    for key, label, direction in _METRIC_DEFS:
        bc, fc = f"base_{key}", f"ft_{key}"
        if bc not in df.columns:
            continue
        bm = df[bc].mean()
        fm = df[fc].mean()
        delta = fm - bm
        better = (delta > 0) if direction == "↑" else (delta < 0)
        indicator = "✓ FT wins" if better else "✗ base wins"
        print(f"  {label+' '+direction:<18} {bm:>12.4f} {fm:>12.4f} {delta:>+10.4f}  {indicator}")
    print("=" * w + "\n")


def _save_html_report(df: pd.DataFrame, out_root: Path) -> None:
    """Render an HTML page with per-image grids and a summary table."""
    metric_cols = [c for c in df.columns if c not in ("image", "prompt")]
    summary_html = df[metric_cols].mean().to_frame("Mean").T.to_html(
        float_format="%.4f", classes="summary-table", border=0
    )

    rows_html = []
    for _, row in df.iterrows():
        grid_rel = f"comparisons/{row['image']}"
        metric_spans = "".join(
            f"<span class='metric'><b>{c}</b>: {row[c]:.4f}</span> "
            for c in metric_cols
            if c in row and not isinstance(row[c], str)
        )
        rows_html.append(
            f"""<div class="card">
  <img src="{grid_rel}" loading="lazy">
  <div class="meta">
    <b>{row['image']}</b><br>
    <em>{row.get('prompt','')}</em><br>
    {metric_spans}
  </div>
</div>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Inpainting Evaluation Report</title>
  <style>
    body  {{ font-family: "Segoe UI", Arial, sans-serif; background:#1a1a2e; color:#eee;
             max-width:1300px; margin:0 auto; padding:24px; }}
    h1,h2 {{ color:#e0b3ff; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:24px; }}
    td,th {{ border:1px solid #444; padding:8px 12px; font-size:13px; }}
    th    {{ background:#2d2d4e; }}
    tr:nth-child(even) {{ background:#242438; }}
    .summary-table td, .summary-table th {{ min-width:110px; text-align:right; }}
    .card {{ background:#242438; border-radius:8px; margin-bottom:20px;
             padding:12px; display:flex; gap:16px; align-items:flex-start; }}
    .card img {{ max-width:820px; border-radius:4px; flex-shrink:0; }}
    .meta  {{ font-size:12px; line-height:1.7; }}
    .metric {{ display:inline-block; background:#1a1a3e; border-radius:4px;
               padding:2px 6px; margin:2px; }}
    em     {{ color:#aaa; }}
  </style>
</head>
<body>
<h1>&#x1F3A8; Inpainting Evaluation: Base vs Fine-tuned</h1>
<h2>Summary Metrics</h2>
{summary_html}
<h2>Per-image Comparisons</h2>
<p style="font-size:13px;color:#aaa">Grid layout: <b>Original | Mask | Base | Fine-tuned</b></p>
{"".join(rows_html)}
</body>
</html>"""

    report_path = out_root / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[INFO] HTML report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args) -> None:
    is_sdxl = _is_sdxl(args.base_model)
    print(f"[INFO] Architecture : {'SDXL' if is_sdxl else 'SD 1.x'}")
    print(f"[INFO] Base model   : {args.base_model}")
    print(f"[INFO] LoRA weights : {args.lora_dir}")
    print(f"[INFO] Resolution   : {args.resolution}  Steps: {args.steps}")

    # -- Discover validation images ----------------------------------------
    image_root = Path(args.image_dir)
    mask_root  = Path(args.mask_dir)
    exts       = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in exts)

    if not all_images:
        raise FileNotFoundError(f"No images found in {image_root}")

    if args.num_samples and args.num_samples < len(all_images):
        rng    = np.random.default_rng(args.seed)
        idxs   = rng.choice(len(all_images), size=args.num_samples, replace=False)
        all_images = [all_images[i] for i in sorted(idxs)]

    print(f"[INFO] Evaluating on {len(all_images)} images")

    # -- Load captions -------------------------------------------------------
    captions: Dict[str, str] = {}
    if args.captions and Path(args.captions).exists():
        with open(args.captions) as f:
            captions = json.load(f)

    default_prompt = args.prompt or "A high-quality interior design photo"

    # -- Output directories --------------------------------------------------
    out_root = Path(args.output_dir)
    base_dir = out_root / "base"
    ft_dir   = out_root / "finetuned"

    # -- Pass 1: base model inference ----------------------------------------
    _inference_pass(
        model_id=args.base_model,
        lora_dir=None,
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        out_dir=base_dir,
        args=args,
    )

    # -- Pass 2: fine-tuned model inference ----------------------------------
    _inference_pass(
        model_id=args.base_model,
        lora_dir=args.lora_dir,
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        out_dir=ft_dir,
        args=args,
    )

    # -- Metric computation (no GPU model loaded) ----------------------------
    print("\n[INFO] Computing metrics ...")
    records = _compute_all_metrics(
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        base_dir=base_dir,
        ft_dir=ft_dir,
        args=args,
    )

    if not records:
        print("[WARN] No records collected. Check that inference outputs exist.")
        return

    df = pd.DataFrame(records)
    csv_path = out_root / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Per-image metrics → {csv_path}")

    _print_summary(df)
    _save_html_report(df, out_root)

    print(f"\n[INFO] All results saved to: {out_root}/")
    print(f"       ├── base/        — base model outputs")
    print(f"       ├── finetuned/   — fine-tuned outputs")
    print(f"       ├── comparisons/ — side-by-side grids")
    print(f"       ├── metrics.csv  — per-image metrics")
    print(f"       └── report.html  — visual report")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and compare Base vs Fine-tuned SDXL inpainting model"
    )
    # Paths
    p.add_argument(
        "--base_model",
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Base model HuggingFace ID or local path",
    )
    p.add_argument(
        "--lora_dir",
        default="outputs/interior-inpainting-sdxl/stage2/unet_lora",
        help="Directory containing fine-tuned LoRA weights (adapter_model.safetensors)",
    )
    p.add_argument("--image_dir", default="data/interior/val/images",
                   help="Validation image directory")
    p.add_argument("--mask_dir",  default="data/interior/val/masks",
                   help="Validation mask directory (255 = inpaint region)")
    p.add_argument("--captions",  default="data/interior/val/captions.json",
                   help="JSON caption file {filename: caption}")
    p.add_argument("--output_dir", default="outputs/evaluation",
                   help="Root directory for all outputs")

    # Generation
    p.add_argument("--prompt", default=None,
                   help="Override prompt for all images (ignores captions.json)")
    p.add_argument("--steps",    type=int,   default=30,
                   help="Number of diffusion steps (default 30)")
    p.add_argument("--guidance", type=float, default=7.5,
                   help="Guidance scale (default 7.5)")
    p.add_argument("--strength", type=float, default=0.99,
                   help="Inpainting strength (default 0.99)")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--resolution", type=int, default=1024,
                   help="Image resolution (default 1024 for SDXL)")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Randomly sample N images from val set (default: all)")

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_clip",  action="store_true",
                   help="Skip CLIP score computation (faster, less VRAM)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run inference even if outputs already exist")

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

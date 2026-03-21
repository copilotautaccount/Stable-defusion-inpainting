"""
3-Way Evaluation: SDXL Base  vs  Stage-1 Fine-tuned  vs  Stage-2 Fine-tuned
=============================================================================
Runs inference sequentially (one model at a time to conserve VRAM), computes
quality metrics, renders 5-panel comparison grids, and writes a summary CSV +
HTML report.

Models compared
---------------
  1. SDXL Base  — diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (no LoRA)
  2. Stage 1    — base model + stage-1 LoRA adapter
  3. Stage 2    — base model + stage-2 (final) LoRA adapter

Metrics
-------
  • SSIM        — Structural Similarity Index          ↑ higher is better
  • PSNR (dB)   — Peak Signal-to-Noise Ratio           ↑ higher is better
  • LPIPS       — Learned Perceptual Image Patch Sim.  ↓ lower  is better
  • CLIP Score  — image-text cosine similarity          ↑ higher is better
  • Masked L1   — mean abs pixel error in masked area   ↓ lower  is better
  • Masked L2   — RMS pixel error in masked area        ↓ lower  is better

Usage
-----
# Quick test (20 images, 512 px)
python src/evaluate_3way.py --num_samples 20 --resolution 512 --steps 20

# Full run (defaults match configs/train_config.yaml)
python src/evaluate_3way.py

# Custom paths
python src/evaluate_3way.py \\
    --base_model   diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \\
    --stage1_lora  outputs/interior-inpainting-sdxl/stage1/unet_lora \\
    --stage2_lora  outputs/interior-inpainting-sdxl/stage2/unet_lora \\
    --image_dir    data/interior/val/images \\
    --mask_dir     data/interior/val/masks \\
    --captions     data/interior/val/captions.json \\
    --output_dir   outputs/evaluation_3way \\
    --num_samples  50
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
# Metric helpers  (identical to evaluate.py)
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: Image.Image, device: str) -> torch.Tensor:
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
    inputs = clip_processor(
        text=[prompt], images=img, return_tensors="pt", padding=True,
        truncation=True, max_length=77,
    ).to(device)
    with torch.no_grad():
        out = clip_model(**inputs)
    return float(out.logits_per_image.item())


def compute_masked_metrics(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Dict[str, float]:
    orig = np.array(original.convert("RGB"), dtype=np.float32)
    gen  = np.array(generated.convert("RGB"), dtype=np.float32)
    m    = np.array(mask.convert("L")) > 127

    if m.sum() == 0:
        return {"masked_l1": 0.0, "masked_l2": 0.0}

    flat_orig = orig[m]
    flat_gen  = gen[m]
    l1 = float(np.mean(np.abs(flat_orig - flat_gen)) / 255.0)
    l2 = float(np.sqrt(np.mean((flat_orig - flat_gen) ** 2)) / 255.0)
    return {"masked_l1": l1, "masked_l2": l2}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(model_id: str, lora_dir: Optional[str], device: str, dtype: torch.dtype):
    from diffusers import StableDiffusionXLInpaintPipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if lora_dir:
        pipe.load_lora_weights(lora_dir)
    pipe.enable_attention_slicing()
    try:
        pipe.unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
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
) -> Image.Image:
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        mask_image=mask,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        height=image.height,
        width=image.width,
        generator=generator,
    ).images[0]


# ---------------------------------------------------------------------------
# Inference pass  (one model at a time)
# ---------------------------------------------------------------------------

_NEG_PROMPT = (
    "blurry, low quality, distorted, ugly, out of focus, "
    "bad anatomy, watermark, text, oversaturated"
)


def _inference_pass(
    label: str,
    model_id: str,
    lora_dir: Optional[str],
    image_paths: List[Path],
    mask_root: Path,
    captions: Dict[str, str],
    default_prompt: str,
    out_dir: Path,
    args,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    dtype  = torch.float16 if "cuda" in device else torch.float32

    print(f"\n{'='*60}")
    print(f"  [{label}]  Loading pipeline …")
    print(f"  base_model : {model_id}")
    if lora_dir:
        print(f"  lora_dir   : {lora_dir}")
    print(f"{'='*60}")

    pipe = _build_pipeline(model_id, lora_dir, device, dtype)

    for img_path in tqdm(image_paths, desc=f"Infer ({label})"):
        out_path = out_dir / img_path.name
        if out_path.exists() and not args.overwrite:
            continue

        # locate mask
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
            pipe, image, mask, prompt, _NEG_PROMPT,
            args.steps, args.guidance, args.strength, args.seed,
        )
        result.save(out_path)

    # Free VRAM before loading the next model
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 5-panel comparison grid
# ---------------------------------------------------------------------------

def _make_grid_5(
    original: Image.Image,
    mask: Image.Image,
    base_out: Image.Image,
    stage1_out: Image.Image,
    stage2_out: Image.Image,
    metrics_base: Dict,
    metrics_s1: Dict,
    metrics_s2: Dict,
) -> Image.Image:
    """5-panel grid: Original | Mask | Base | Stage-1 | Stage-2"""
    W, H   = original.size
    pad    = 6
    lbl_h  = 30
    n_cols = 5
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
        ("Original",      original.convert("RGB"),  {}),
        ("Mask",          mask.convert("RGB"),       {}),
        ("SDXL Base",     base_out.convert("RGB"),   metrics_base),
        ("Stage-1 LoRA",  stage1_out.convert("RGB"), metrics_s1),
        ("Stage-2 LoRA",  stage2_out.convert("RGB"), metrics_s2),
    ]

    colors = [(255, 255, 255), (180, 180, 180), (255, 200, 80), (100, 220, 255), (120, 255, 160)]

    for i, ((label, img, mets), color) in enumerate(zip(panels, colors)):
        x0 = pad + i * (W + pad)
        y0 = pad + lbl_h
        canvas.paste(img.resize((W, H)), (x0, y0))
        full_label = f"{label}  {_metric_str(mets)}" if mets else label
        draw.text((x0 + 4, pad + 6), full_label, fill=color, font=font)

    return canvas


# ---------------------------------------------------------------------------
# Metric computation pass  (no model in memory)
# ---------------------------------------------------------------------------

def _compute_all_metrics(
    image_paths: List[Path],
    mask_root: Path,
    captions: Dict[str, str],
    default_prompt: str,
    base_dir: Path,
    stage1_dir: Path,
    stage2_dir: Path,
    args,
) -> List[Dict]:
    device = args.device

    # LPIPS
    lpips_fn = None
    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net="alex").to(device).eval()
        print("[INFO] LPIPS (alex) loaded.")
    except ImportError:
        warnings.warn("lpips not installed → LPIPS skipped.  pip install lpips")

    # CLIP
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
        bp  = base_dir   / img_path.name
        s1p = stage1_dir / img_path.name
        s2p = stage2_dir / img_path.name

        if not (bp.exists() and s1p.exists() and s2p.exists()):
            tqdm.write(f"  [WARN] Missing output for {img_path.name}, skipping metrics.")
            continue

        mask_path = mask_root / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = mask_root / img_path.name

        original = Image.open(img_path).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS
        )
        mask = (
            Image.open(mask_path).convert("L").resize(
                (args.resolution, args.resolution), Image.NEAREST
            ) if mask_path.exists() else Image.new("L", original.size, 255)
        )

        base_out   = Image.open(bp).convert("RGB")
        stage1_out = Image.open(s1p).convert("RGB")
        stage2_out = Image.open(s2p).convert("RGB")

        prompt = captions.get(img_path.name, captions.get(img_path.stem, default_prompt))
        row: Dict = {"image": img_path.name, "prompt": prompt[:100]}

        # ---- SSIM ----
        row["base_ssim"]   = compute_ssim(original, base_out)
        row["stage1_ssim"] = compute_ssim(original, stage1_out)
        row["stage2_ssim"] = compute_ssim(original, stage2_out)

        # ---- PSNR ----
        row["base_psnr"]   = compute_psnr(original, base_out)
        row["stage1_psnr"] = compute_psnr(original, stage1_out)
        row["stage2_psnr"] = compute_psnr(original, stage2_out)

        # ---- Masked L1/L2 ----
        for prefix, gen in [("base", base_out), ("stage1", stage1_out), ("stage2", stage2_out)]:
            mm = compute_masked_metrics(original, gen, mask)
            row[f"{prefix}_masked_l1"] = mm["masked_l1"]
            row[f"{prefix}_masked_l2"] = mm["masked_l2"]

        # ---- LPIPS ----
        if lpips_fn is not None:
            row["base_lpips"]   = compute_lpips(original, base_out,   lpips_fn, device)
            row["stage1_lpips"] = compute_lpips(original, stage1_out, lpips_fn, device)
            row["stage2_lpips"] = compute_lpips(original, stage2_out, lpips_fn, device)

        # ---- CLIP ----
        if clip_model is not None:
            row["base_clip"]   = compute_clip_score(base_out,   prompt, clip_model, clip_proc, device)
            row["stage1_clip"] = compute_clip_score(stage1_out, prompt, clip_model, clip_proc, device)
            row["stage2_clip"] = compute_clip_score(stage2_out, prompt, clip_model, clip_proc, device)

        records.append(row)

        # ---- 5-panel grid ----
        grid = _make_grid_5(
            original, mask, base_out, stage1_out, stage2_out,
            metrics_base={"ssim": row["base_ssim"],   "psnr": row["base_psnr"]},
            metrics_s1  ={"ssim": row["stage1_ssim"], "psnr": row["stage1_psnr"]},
            metrics_s2  ={"ssim": row["stage2_ssim"], "psnr": row["stage2_psnr"]},
        )
        grid.save(grid_dir / img_path.name)

    return records


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

_METRIC_DEFS = [
    ("ssim",       "SSIM",         "↑"),
    ("psnr",       "PSNR (dB)",    "↑"),
    ("lpips",      "LPIPS",        "↓"),
    ("clip",       "CLIP Score",   "↑"),
    ("masked_l1",  "Masked L1",    "↓"),
    ("masked_l2",  "Masked L2",    "↓"),
]


def _winner_symbol(vals: List[float], direction: str) -> List[str]:
    """Return symbol list where best value gets 🏆, others get '' """
    if direction == "↑":
        best = max(vals)
    else:
        best = min(vals)
    return ["🏆" if abs(v - best) < 1e-9 else "  " for v in vals]


def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    w = 76
    print("\n" + "=" * w)
    print(f"  3-Way Evaluation Summary  ({n} images)")
    print("=" * w)
    header = f"  {'Metric':<18} {'Base':>11} {'Stage-1':>11} {'Stage-2':>11}  {'Winner':>10}"
    print(header)
    print("-" * w)
    for key, label, direction in _METRIC_DEFS:
        bc  = f"base_{key}"
        s1c = f"stage1_{key}"
        s2c = f"stage2_{key}"
        if bc not in df.columns:
            continue
        bm  = df[bc].mean()
        s1m = df[s1c].mean()
        s2m = df[s2c].mean()
        syms = _winner_symbol([bm, s1m, s2m], direction)
        winners = ["Base", "Stage-1", "Stage-2"]
        best_name = winners[([bm, s1m, s2m].index(max([bm, s1m, s2m]) if direction == "↑" else min([bm, s1m, s2m])))]
        print(
            f"  {label+' '+direction:<18} {bm:>11.4f}{syms[0]} {s1m:>11.4f}{syms[1]} {s2m:>11.4f}{syms[2]}  {best_name}"
        )
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _save_html_report(df: pd.DataFrame, out_root: Path) -> None:
    metric_cols = [c for c in df.columns if c not in ("image", "prompt")]
    summary_html = df[metric_cols].mean().to_frame("Mean").T.to_html(
        float_format="%.4f", classes="summary-table", border=0
    )

    # Build per-metric 3-way summary table
    rows_3way = []
    for key, label, direction in _METRIC_DEFS:
        bc, s1c, s2c = f"base_{key}", f"stage1_{key}", f"stage2_{key}"
        if bc not in df.columns:
            continue
        bm  = df[bc].mean()
        s1m = df[s1c].mean()
        s2m = df[s2c].mean()
        syms = _winner_symbol([bm, s1m, s2m], direction)
        rows_3way.append(
            f"<tr><td>{label} {direction}</td>"
            f"<td>{bm:.4f}{syms[0]}</td>"
            f"<td>{s1m:.4f}{syms[1]}</td>"
            f"<td>{s2m:.4f}{syms[2]}</td></tr>"
        )
    table_3way = f"""
<table><thead>
  <tr><th>Metric</th><th>SDXL Base</th><th>Stage-1 LoRA</th><th>Stage-2 LoRA ✨</th></tr>
</thead><tbody>
{"".join(rows_3way)}
</tbody></table>"""

    cards_html = []
    for _, row in df.iterrows():
        grid_rel = f"comparisons/{row['image']}"
        metric_spans = "".join(
            f"<span class='metric'><b>{c}</b>: {row[c]:.4f}</span> "
            for c in metric_cols
            if c in row and not isinstance(row[c], str)
        )
        cards_html.append(f"""
<div class="card">
  <img src="{grid_rel}" loading="lazy">
  <div class="meta">
    <b>{row['image']}</b><br>
    <em>{row.get('prompt','')}</em><br>
    {metric_spans}
  </div>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>3-Way Inpainting Evaluation</title>
  <style>
    body  {{ font-family: "Segoe UI", Arial, sans-serif; background:#1a1a2e;
             color:#eee; max-width:1400px; margin:0 auto; padding:24px; }}
    h1,h2 {{ color:#e0b3ff; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:24px; }}
    td,th {{ border:1px solid #444; padding:8px 12px; font-size:13px; }}
    th    {{ background:#2d2d4e; }}
    tr:nth-child(even) {{ background:#242438; }}
    .card {{ background:#242438; border-radius:8px; margin-bottom:20px;
             padding:12px; display:flex; gap:16px; align-items:flex-start; }}
    .card img {{ max-width:1100px; border-radius:4px; flex-shrink:0; }}
    .meta  {{ font-size:12px; line-height:1.9; }}
    .metric {{ display:inline-block; background:#1a1a3e; border-radius:4px;
               padding:2px 6px; margin:2px; }}
    em {{ color:#aaa; }}
    .legend {{ font-size:13px; color:#ccc; background:#242438;
               border-radius:6px; padding:10px 16px; margin-bottom:20px; }}
  </style>
</head>
<body>
<h1>&#x1F3A8; 3-Way Inpainting Evaluation</h1>
<div class="legend">
  Grid columns (left → right):
  <b style="color:#fff">Original</b> &nbsp;|&nbsp;
  <b style="color:#bbb">Mask</b> &nbsp;|&nbsp;
  <b style="color:#ffc850">SDXL Base</b> &nbsp;|&nbsp;
  <b style="color:#64dcff">Stage-1 LoRA</b> &nbsp;|&nbsp;
  <b style="color:#78ffa0">Stage-2 LoRA ✨</b>
</div>
<h2>Summary</h2>
{table_3way}
<h2>Per-image Comparisons</h2>
{"".join(cards_html)}
</body>
</html>"""

    report_path = out_root / "report_3way.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[INFO] HTML report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args) -> None:
    print("\n" + "="*60)
    print("  3-Way SDXL Inpainting Evaluation")
    print("="*60)
    print(f"  Base model  : {args.base_model}")
    print(f"  Stage-1 LoRA: {args.stage1_lora}")
    print(f"  Stage-2 LoRA: {args.stage2_lora}")
    print(f"  Resolution  : {args.resolution}   Steps: {args.steps}")
    print("="*60)

    # Validate LoRA paths
    for name, path in [("Stage-1 LoRA", args.stage1_lora), ("Stage-2 LoRA", args.stage2_lora)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} path not found: {path}")

    # Discover validation images
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

    print(f"\n[INFO] Evaluating on {len(all_images)} images")

    # Load captions
    captions: Dict[str, str] = {}
    if args.captions and Path(args.captions).exists():
        with open(args.captions) as f:
            raw = json.load(f)
        # Support both list-of-dicts and dict formats
        if isinstance(raw, list):
            for item in raw:
                if "filename" in item and "prompt" in item:
                    captions[item["filename"]] = item["prompt"]
                    captions[Path(item["filename"]).stem] = item["prompt"]
        elif isinstance(raw, dict):
            captions = raw

    default_prompt = args.prompt or "A high-quality interior design photo"

    # Output subdirectories
    out_root   = Path(args.output_dir)
    base_dir   = out_root / "base"
    stage1_dir = out_root / "stage1"
    stage2_dir = out_root / "stage2"

    # ── Pass 1: SDXL Base ──────────────────────────────────────────────────
    _inference_pass(
        label="SDXL Base",
        model_id=args.base_model,
        lora_dir=None,
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        out_dir=base_dir,
        args=args,
    )

    # ── Pass 2: Stage-1 LoRA ───────────────────────────────────────────────
    _inference_pass(
        label="Stage-1 LoRA",
        model_id=args.base_model,
        lora_dir=args.stage1_lora,
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        out_dir=stage1_dir,
        args=args,
    )

    # ── Pass 3: Stage-2 LoRA (final) ───────────────────────────────────────
    _inference_pass(
        label="Stage-2 LoRA (final)",
        model_id=args.base_model,
        lora_dir=args.stage2_lora,
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        out_dir=stage2_dir,
        args=args,
    )

    # ── Metrics (no GPU model loaded) ──────────────────────────────────────
    print("\n[INFO] Computing metrics across 3 models …")
    records = _compute_all_metrics(
        image_paths=all_images,
        mask_root=mask_root,
        captions=captions,
        default_prompt=default_prompt,
        base_dir=base_dir,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        args=args,
    )

    if not records:
        print("[WARN] No records collected. Check that inference outputs exist.")
        return

    df = pd.DataFrame(records)
    csv_path = out_root / "metrics_3way.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Per-image metrics → {csv_path}")

    _print_summary(df)
    _save_html_report(df, out_root)

    print(f"\n[INFO] All results saved to: {out_root}/")
    print(f"       ├── base/              — SDXL Base outputs")
    print(f"       ├── stage1/            — Stage-1 LoRA outputs")
    print(f"       ├── stage2/            — Stage-2 LoRA outputs")
    print(f"       ├── comparisons/       — 5-panel grids")
    print(f"       ├── metrics_3way.csv   — per-image metrics")
    print(f"       └── report_3way.html   — visual HTML report")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-Way evaluation: SDXL Base vs Stage-1 LoRA vs Stage-2 LoRA"
    )
    # Model paths
    p.add_argument(
        "--base_model",
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Base SDXL inpainting model (HuggingFace ID or local path)",
    )
    p.add_argument(
        "--stage1_lora",
        default="outputs/interior-inpainting-sdxl/stage1/unet_lora",
        help="Stage-1 LoRA adapter directory",
    )
    p.add_argument(
        "--stage2_lora",
        default="outputs/interior-inpainting-sdxl/stage2/unet_lora",
        help="Stage-2 (final) LoRA adapter directory",
    )
    # Data
    p.add_argument("--image_dir",  default="data/interior/val/images")
    p.add_argument("--mask_dir",   default="data/interior/val/masks")
    p.add_argument("--captions",   default="data/interior/val/captions.json")
    p.add_argument("--output_dir", default="outputs/evaluation_3way")

    # Generation
    p.add_argument("--prompt",     default=None,
                   help="Override prompt for all images (ignores captions.json)")
    p.add_argument("--steps",      type=int,   default=30)
    p.add_argument("--guidance",   type=float, default=7.5)
    p.add_argument("--strength",   type=float, default=0.99)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--resolution", type=int,   default=1024,
                   help="Inference resolution (1024 recommended for SDXL)")
    p.add_argument("--num_samples", type=int,  default=None,
                   help="Randomly sample N images from the val set (default: all)")

    # Misc
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_clip",  action="store_true",
                   help="Skip CLIP score (faster)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-run inference even if outputs already exist")

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

"""
Data-preparation utilities for interior inpainting fine-tuning.

Captioning models
-----------------
  blip        Salesforce/blip-image-captioning-large  ~6 GB VRAM  fast, good quality
  blip2       Salesforce/blip2-opt-2.7b               ~15 GB VRAM best quality (default)
  florence2   microsoft/Florence-2-large              ~8 GB VRAM  detailed region captions

Masking models
--------------
  sam          Meta SAM ViT-H (local .pth checkpoint)   ~7 GB VRAM  automatic segments
  sam2         facebook/sam2-hiera-large (HuggingFace)  ~8 GB VRAM  improved SAM, no download
  sam3         facebook/sam3 (text-prompted)             ~10 GB VRAM concept-aware segmentation
  grounded_sam GroundingDINO + SAM (text-prompted)      ~10 GB VRAM furniture-aware (recommended)
  oneformer    shi-labs/oneformer_ade20k_swin_large      ~12 GB VRAM semantic segmentation

Usage
-----
# 1. Organise raw images into train/val splits
python src/prepare_data.py split \\
    --source_dir data/raw \\
    --output_dir data/interior \\
    --val_ratio 0.1

# 2. Auto-generate captions  (choose --captioner blip | blip2 | florence2)
python src/prepare_data.py caption \\
    --dataset_dir data/interior \\
    --captioner   blip2

python src/prepare_data.py caption \\
    --dataset_dir data/interior \\
    --captioner   florence2

# 3. Auto-generate object masks  (choose --masker sam | sam2 | sam3 | grounded_sam | oneformer)
python src/prepare_data.py mask \\
    --dataset_dir data/interior \\
    --masker      grounded_sam \\
    --furniture_labels "sofa,chair,table,bed,cabinet,lamp"

python src/prepare_data.py mask \\
    --dataset_dir    data/interior \\
    --masker         sam \\
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth

python src/prepare_data.py mask \\
    --dataset_dir data/interior \\
    --masker      sam2

python src/prepare_data.py mask \\
    --dataset_dir data/interior \\
    --masker      sam3 \\
    --furniture_labels "sofa,chair,table,bed,cabinet,lamp"

python src/prepare_data.py mask \\
    --dataset_dir data/interior \\
    --masker      oneformer
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Default furniture categories used by grounded_sam and oneformer
_DEFAULT_FURNITURE_LABELS = (
    "sofa,armchair,chair,dining chair,table,coffee table,dining table,"
    "bed,wardrobe,cabinet,bookshelf,desk,lamp,floor lamp,curtain,rug,mirror"
)

# ADE20K category names that correspond to interior furniture/objects
# (used to filter OneFormer predictions)
_ADE20K_FURNITURE_IDS = {
    "sofa", "chair", "armchair", "swivel chair", "bed", "table",
    "coffee table", "dining table", "desk", "cabinet", "wardrobe",
    "bookcase", "shelf", "lamp", "floor lamp", "chandelier", "mirror",
    "curtain", "rug", "mat", "ottoman", "bench", "stool",
    "television", "monitor", "refrigerator",
}


# Regex matching tokenizer special-token artefacts that leak into decoded text
# when batched generation is used with padding (e.g. BLIP, BLIP-2).
_SPECIAL_TOKEN_RE = re.compile(r"(<pad>|</s>|<s>|<unk>|<sep>|<bos>|<eos>)+", re.IGNORECASE)


def _clean_caption(text: str) -> str:
    """Remove tokenizer padding/special-token artefacts from a decoded caption."""
    return _SPECIAL_TOKEN_RE.sub(" ", text).strip()


def _image_paths(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTENSIONS)


# ---------------------------------------------------------------------------
# Transformers version helpers
# ---------------------------------------------------------------------------

def _transformers_version() -> tuple:
    """Return the installed ``transformers`` version as a ``(major, minor)`` tuple.

    Falls back to ``(0, 0)`` if the package is not importable (e.g. in tests).
    """
    try:
        import transformers as _tf
        parts = _tf.__version__.split(".")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)


def _torch_dtype_kwarg(dtype) -> dict:
    """Return the correct kwarg for specifying dtype when loading a HuggingFace model.

    * ``transformers < 4.48`` – uses the original ``torch_dtype`` parameter.
    * ``transformers >= 4.48`` – ``torch_dtype`` is deprecated; use ``dtype``.
    """
    if _transformers_version() >= (4, 48):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


# ---------------------------------------------------------------------------
# Split raw images into train / val
# ---------------------------------------------------------------------------

def split_dataset(
    source_dir: str,
    output_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
) -> None:
    """Shuffle and split raw images into ``train/`` and ``val/`` subsets.

    Args:
        source_dir: Directory containing raw images (flat or recursive).
        output_dir: Root output directory; ``train/images`` and ``val/images``
            will be created inside.
        val_ratio: Fraction of images reserved for validation.
        seed: Random seed for reproducibility.
        extensions: Accepted image file extensions.
    """
    source = Path(source_dir)
    all_images: List[Path] = sorted(
        p for p in source.rglob("*") if p.suffix.lower() in extensions
    )
    if not all_images:
        raise RuntimeError(f"No images found in {source_dir}")

    random.seed(seed)
    random.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_ratio))
    splits = {
        "val": all_images[:n_val],
        "train": all_images[n_val:],
    }

    for split, paths in splits.items():
        out_dir = Path(output_dir) / split / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        to_copy = [src for src in paths if not (out_dir / src.name).exists()]
        skipped = len(paths) - len(to_copy)
        if skipped:
            print(f"  [{split}] Skipping {skipped} already-copied images.")
        for src in tqdm(to_copy, desc=f"Copying {split}"):
            shutil.copy2(src, out_dir / src.name)

    print(f"Split complete – train: {len(splits['train'])}, val: {len(splits['val'])}")


# ---------------------------------------------------------------------------
# Captioning backends
# ---------------------------------------------------------------------------

def _caption_blip(
    image_paths: List[Path],
    device: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    """Caption images with BLIP (Salesforce/blip-image-captioning-large).

    Lighter than BLIP-2: ~6 GB VRAM, faster, good quality.
    """
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc

    print("Loading BLIP processor and model (Salesforce/blip-image-captioning-large) …")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        **_torch_dtype_kwarg(torch.float16 if device != "cpu" else torch.float32),
    ).to(device)
    model.eval()

    captions: dict = {}
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning (BLIP)"):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        for path, out_ids in zip(batch, ids):
            captions[path.name] = _clean_caption(
                processor.decode(out_ids, skip_special_tokens=True)
            )

    return captions


def _caption_blip2(
    image_paths: List[Path],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    interior_prefix: str,
) -> dict:
    """Caption images with BLIP-2 (Salesforce/blip2-opt-2.7b).

    Best caption quality; ~15 GB VRAM. Supports a conditional text prefix.
    """
    try:
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc

    print("Loading BLIP-2 processor and model (Salesforce/blip2-opt-2.7b) …")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        **_torch_dtype_kwarg(torch.float16 if device != "cpu" else torch.float32),
        device_map=device,
    )
    model.eval()

    captions: dict = {}
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning (BLIP-2)"):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        prompts = [interior_prefix] * len(images)
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        for path, out_ids in zip(batch, ids):
            captions[path.name] = _clean_caption(
                processor.decode(out_ids, skip_special_tokens=True)
            )

    return captions


def _caption_florence2(
    image_paths: List[Path],
    device: str,
    batch_size: int,
) -> dict:
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("pip install transformers torch einops timm") from exc

    model_id = "microsoft/Florence-2-large"
    print(f"Loading Florence-2 from {model_id} …")

    _dtype = torch.float16 if device == "cuda" else torch.float32

    # Processor always needs trust_remote_code=True to avoid the
    # ``RobertaTokenizer has no attribute image_token`` crash.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Florence-2 is always loaded via trust_remote_code=True; there is no
    # native Florence2ForConditionalGeneration class in the transformers package.
    # Always use `torch_dtype` (not `dtype`) because the remote custom __init__
    # does not accept the newer `dtype` kwarg introduced in transformers >= 4.48.
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    task_token = "<MORE_DETAILED_CAPTION>"
    captions: dict = {}

    skipped: list = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning (Florence-2)"):
        raw_batch = image_paths[i : i + batch_size]

        # Load images, skipping any that PIL cannot read (corrupt / truncated files)
        valid_paths: list = []
        valid_images: list = []
        for p in raw_batch:
            try:
                img = Image.open(p).convert("RGB")
                img.load()          # force full decode so truncated files surface here
                valid_paths.append(p)
                valid_images.append(img)
            except Exception as exc:
                print(f"\n  WARNING: skipping unreadable image {p.name}: {exc}")
                skipped.append(p)

        if not valid_paths:
            continue

        inputs = processor(
            text=[task_token] * len(valid_images),
            images=valid_images,
            return_tensors="pt",
            padding=True,
        )

        # Move tensors to device / dtype
        inputs = {k: v.to(device).to(_dtype) if v.dtype == torch.float32 else v.to(device)
                  for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        for path, out_ids, image in zip(valid_paths, generated_ids, valid_images):
            generated_text = processor.batch_decode(out_ids.unsqueeze(0), skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_token,
                image_size=(image.width, image.height),
            )
            captions[path.name] = _clean_caption(parsed_answer[task_token])

    if skipped:
        print(f"\n  Skipped {len(skipped)} unreadable image(s):")
        for p in skipped:
            print(f"    {p}")

    return captions


def generate_captions(
    dataset_dir: str,
    captioner: str = "blip2",
    splits: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 8,
    max_new_tokens: int = 60,
    interior_prefix: str = "An interior design photo of",
) -> None:
    """Generate image captions and save them as ``captions.json``.

    Three captioning backends are available via the ``captioner`` argument:

    * ``"blip"``     – Salesforce/blip-image-captioning-large  (~6 GB VRAM, fast)
    * ``"blip2"``    – Salesforce/blip2-opt-2.7b               (~15 GB VRAM, best quality)
    * ``"florence2"``– microsoft/Florence-2-large              (~8 GB VRAM, detailed)

    Args:
        dataset_dir: Root dataset directory containing split sub-directories.
        captioner: Which captioning model to use (``"blip"``, ``"blip2"``, or
            ``"florence2"``).
        splits: Splits to process (default: ``["train", "val"]``).
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
        batch_size: Number of images per inference batch.
        max_new_tokens: Maximum new tokens to generate (blip / blip2 only).
        interior_prefix: Conditional prompt prefix (blip2 only).
    """
    _SUPPORTED = ("blip", "blip2", "florence2")
    if captioner not in _SUPPORTED:
        raise ValueError(
            f"Unknown captioner '{captioner}'. Choose one of: {_SUPPORTED}"
        )

    splits = splits if splits is not None else ["train", "val"]
    root = Path(dataset_dir)

    for split in splits:
        images_dir = root / split / "images"
        if not images_dir.exists():
            print(f"  Skipping {split} – directory not found.")
            continue

        paths = _image_paths(images_dir)
        out_path = root / split / "captions.json"

        # Resume: load existing captions and skip already-processed images
        existing_captions: dict = {}
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as fh:
                existing_captions = json.load(fh)
        pending = [p for p in paths if p.name not in existing_captions]
        skipped = len(paths) - len(pending)
        print(f"\n[{split}] {len(paths)} images – captioner: {captioner}")
        if skipped:
            print(f"  Resuming: skipping {skipped} already-captioned images; {len(pending)} remaining.")

        new_captions: dict = {}
        if pending:
            if captioner == "blip":
                new_captions = _caption_blip(pending, device, batch_size, max_new_tokens)
            elif captioner == "blip2":
                new_captions = _caption_blip2(pending, device, batch_size, max_new_tokens, interior_prefix)
            else:
                new_captions = _caption_florence2(pending, device, batch_size)

        captions = {**existing_captions, **new_captions}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(captions, fh, ensure_ascii=False, indent=2)
        print(f"  Saved {len(captions)} captions → {out_path}")


# ---------------------------------------------------------------------------
# Masking backends
# ---------------------------------------------------------------------------

def _pick_best_mask(
    annotations: list,
    cx: int,
    cy: int,
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.85,
) -> Optional[np.ndarray]:
    """Combine the top SAM annotation masks into a single inpainting mask.

    Instead of returning only the single mask closest to the image centre,
    this function merges all annotations whose area is between
    ``min_area_frac`` and ``max_area_frac`` of the total image area. Very
    small segments (noise) and very large segments (full-image background)
    are discarded.  Among the remaining candidates the five with the
    highest ``predicted_iou`` (falling back to centre-distance ranking when
    IoU is unavailable) are combined via bitwise OR.

    Args:
        annotations: List of SAM annotation dicts (each has ``bbox``,
            ``segmentation``, and optionally ``predicted_iou`` / ``area``).
        cx, cy: Image centre coordinates used as a tie-breaker.
        min_area_frac: Minimum segment area as a fraction of image area.
        max_area_frac: Maximum segment area as a fraction of image area.

    Returns:
        A ``(H, W)`` ``uint8`` mask with 0 / 255 values, or ``None`` when
        *annotations* is empty or all masks are zero after combination.
    """
    if not annotations:
        return None

    # Determine image dimensions from the first annotation's segmentation
    seg0 = annotations[0]["segmentation"]
    total_pixels = seg0.shape[0] * seg0.shape[1]

    # Filter by area fraction to drop noise and full-background segments
    filtered = []
    for ann in annotations:
        seg = ann["segmentation"]
        area = int(ann.get("area", np.sum(seg)))
        frac = area / total_pixels
        if min_area_frac <= frac <= max_area_frac:
            filtered.append(ann)

    # Fall back to the original list when filtering removes everything
    if not filtered:
        filtered = annotations

    # Rank candidates: prefer high predicted_iou, break ties by distance to centre
    def score(ann: dict) -> tuple:
        iou = ann.get("predicted_iou", 0.0)
        x, y, bw, bh = ann["bbox"]
        dist_sq = (x + bw / 2 - cx) ** 2 + (y + bh / 2 - cy) ** 2
        # Higher iou first (negate for ascending sort), then closer to centre
        return (-iou, dist_sq)

    filtered.sort(key=score)

    # Combine up to 5 top candidates
    combined = np.zeros_like(seg0, dtype=np.uint8)
    for ann in filtered[:5]:
        combined |= ann["segmentation"].astype(np.uint8) * 255

    return combined if np.any(combined) else None


def _mask_sam(
    image_paths: List[Path],
    masks_dir: Path,
    device: str,
    sam_checkpoint: str,
    model_type: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    min_mask_region_area: int,
) -> None:
    """Generate masks with SAM (Segment Anything Model) – requires a local
    checkpoint downloaded from Meta.

    Download:
        mkdir -p checkpoints
        wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    Checkpoint URLs:
        ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
        ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ImportError("pip install segment-anything") from exc

    if not sam_checkpoint or not Path(sam_checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: '{sam_checkpoint}'. "
            "Download with: wget -P checkpoints "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    print(f"Loading SAM ({model_type}) from {sam_checkpoint} …")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    for img_path in tqdm(image_paths, desc="Masking (SAM)"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        anns = gen.generate(rgb)
        mask = _pick_best_mask(anns, w // 2, h // 2)
        if mask is not None:
            cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), mask)
        else:
            # Save an empty mask so the resume logic won't reprocess this image
            cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), np.zeros((h, w), dtype=np.uint8))


def _mask_sam2(
    image_paths: List[Path],
    masks_dir: Path,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
) -> None:
    """Generate masks with SAM 2 loaded directly from HuggingFace Hub.

    No manual checkpoint download required.
    Model: facebook/sam2-hiera-large (~8 GB VRAM).

    SAM 2 improves over SAM with better object boundaries and supports
    video – useful when the dataset contains image sequences.
    """
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf
    except ImportError as exc:
        raise ImportError(
            "SAM 2 is required. Install with:\n"
            "  pip install 'git+https://github.com/facebookresearch/sam2.git'"
        ) from exc

    print("Loading SAM 2 (facebook/sam2-hiera-large) from HuggingFace …")
    sam2_model = build_sam2_hf("facebook/sam2-hiera-large", device=device)
    gen = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
    )

    for img_path in tqdm(image_paths, desc="Masking (SAM 2)"):
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        anns = gen.generate(image)
        mask = _pick_best_mask(anns, w // 2, h // 2)
        if mask is not None:
            cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), mask)
        else:
            # Save an empty mask so the resume logic won't reprocess this image
            cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), np.zeros((h, w), dtype=np.uint8))


def _mask_sam3(
    image_paths: List[Path],
    masks_dir: Path,
    device: str,
    furniture_labels: str,
) -> None:
    """Generate masks with SAM 3 (Segment Anything Model 3) using text prompts.

    SAM 3 is Meta's third-generation segmentation model supporting
    open-vocabulary, text-prompted concept segmentation.  Unlike SAM / SAM 2
    (which rely on a centre-heuristic), SAM 3 accepts a natural-language
    description of the objects to segment, producing masks for every instance
    that matches the concept.

    Model: ``facebook/sam3`` (~10 GB VRAM).

    Requires:
        pip install sam3

    The model checkpoint is downloaded automatically from HuggingFace on
    first use (authentication via ``huggingface-cli login`` may be required).

    Args:
        furniture_labels: Comma-separated list of furniture categories, e.g.
            ``"sofa,chair,table,bed,cabinet,lamp"``.  Each label is used as
            a text prompt; the resulting masks for all labels are merged
            (bitwise OR) into a single inpainting mask per image.
    """
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "SAM 3 is not installed.  Install it with:\n"
            "  pip install sam3\n"
            "or follow https://github.com/facebookresearch/sam3 for source install.\n"
            "Alternatively use '--masker grounded_sam' which relies only on "
            "groundingdino-py + segment-anything."
        ) from exc

    labels = [lbl.strip() for lbl in furniture_labels.split(",") if lbl.strip()]
    if not labels:
        raise ValueError("furniture_labels must contain at least one label for SAM 3")

    # ── Resolve the BPE vocab file ──────────────────────────────────────────
    # sam3's model_builder defaults to  <sam3_package>/../assets/bpe_simple_vocab_16e6.txt.gz
    # which resolves to  site-packages/assets/  – a directory that is NOT created
    # by the sam3 installer.  We locate the file from other known locations and,
    # if still not found, download it so the user never has to intervene manually.
    import importlib.util as _ilu
    import urllib.request as _urlreq

    _BPE_FNAME = "bpe_simple_vocab_16e6.txt.gz"
    _BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"

    def _find_bpe() -> str:
        # 1. sam3's expected location (may already be fixed by user / installer)
        _sam3_dir = Path(_ilu.find_spec("sam3").origin).parent
        candidate = _sam3_dir.parent / "assets" / _BPE_FNAME
        if candidate.exists():
            return str(candidate)
        # 2. open_clip package (ships the same file)
        for _pkg in ("open_clip", "clip"):
            spec = _ilu.find_spec(_pkg)
            if spec is not None:
                c = Path(spec.origin).parent / _BPE_FNAME
                if c.exists():
                    return str(c)
        # 3. Project checkpoints/ directory
        c = Path(__file__).parent.parent / "checkpoints" / _BPE_FNAME
        if c.exists():
            return str(c)
        # 4. Download to checkpoints/ as a last resort
        c.parent.mkdir(parents=True, exist_ok=True)
        print(f"  BPE vocab not found locally – downloading to {c} …")
        _urlreq.urlretrieve(_BPE_URL, c)
        print("  Download complete.")
        # Also mirror to sam3's expected path so future runs don't need to download
        _expected = _sam3_dir.parent / "assets" / _BPE_FNAME
        try:
            _expected.parent.mkdir(parents=True, exist_ok=True)
            import shutil as _sh
            _sh.copy2(str(c), str(_expected))
        except OSError:
            pass  # non-fatal; checkpoints/ copy is sufficient
        return str(c)

    _bpe_path = _find_bpe()
    print(f"  BPE vocab: {_bpe_path}")

    print("Loading SAM 3 (facebook/sam3) …")
    try:
        model = build_sam3_image_model(device=device, bpe_path=_bpe_path)
    except Exception as _exc:
        _msg = str(_exc)
        if "GatedRepo" in type(_exc).__name__ or "403" in _msg or "gated" in _msg.lower() or "not in the authorized list" in _msg:
            raise RuntimeError(
                "\n"
                "╔═══════════════════════════════════════════════════════════╗\n"
                "║  facebook/sam3 is a GATED model – access not granted yet  ║\n"
                "╚═══════════════════════════════════════════════════════════╝\n"
                "\n"
                "Steps to fix:\n"
                "  1. Request access at  https://huggingface.co/facebook/sam3\n"
                "     (click 'Agree and access repository')\n"
                "  2. After approval, login:\n"
                "       huggingface-cli login\n"
                "  3. Re-run the pipeline.\n"
                "\n"
                "Alternatively, switch to grounded_sam (already installed, same quality):\n"
                "  MASKER=grounded_sam bash scripts/run_pipeline.sh\n"
            ) from _exc
        raise
    processor = Sam3Processor(model)

    for img_path in tqdm(image_paths, desc="Masking (SAM 3)"):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Accumulate masks for every furniture label via text prompts
        combined = np.zeros((h, w), dtype=np.uint8)
        state = processor.set_image(image)

        for label in labels:
            output = processor.set_text_prompt(state=state, prompt=label)
            masks = output.get("masks")
            if masks is None:
                continue
            for m in masks:
                mask_arr = np.asarray(m, dtype=bool)
                # Flatten to 2-D: SAM3 may return (1, H, W) or (H, W).
                while mask_arr.ndim > 2:
                    mask_arr = mask_arr[0]
                if mask_arr.ndim != 2:
                    continue
                combined |= mask_arr.astype(np.uint8) * 255

        # Always write the mask (empty = all-black when no furniture detected),
        # so the resume logic won't reprocess this image on the next run.
        cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), combined)


def _apply_max_mask_ratio(
    masks_dir: Path,
    image_paths: List[Path],
    max_ratio: float,
) -> None:
    """Post-process written masks so that no mask covers more than *max_ratio* of
    the image area.

    When a mask's white-pixel fraction exceeds *max_ratio*, the function keeps
    only the largest connected components (sorted by area, descending) that fit
    within the budget.  Any excess components are discarded.  If even the single
    largest component exceeds the budget, the mask is zeroed out entirely.

    Args:
        masks_dir:   Directory where mask PNGs have already been written.
        image_paths: The image paths processed in this run (used to derive mask names).
        max_ratio:   Maximum allowed fraction of white pixels (e.g. 0.70 for 70%).
    """
    if max_ratio >= 1.0:
        return  # no-op: no cap requested

    clipped_count = 0
    for img_path in image_paths:
        mask_path = masks_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape
        total_pixels = h * w
        white_pixels = int(np.count_nonzero(mask))
        if white_pixels / total_pixels <= max_ratio:
            continue  # already within budget – skip

        # Decompose into connected components and greedily keep the largest
        # ones until the budget is exhausted.
        binary = (mask > 127).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        # Build list of (area, label_id) excluding background (label 0)
        component_areas = [
            (int(stats[i, cv2.CC_STAT_AREA]), i)
            for i in range(1, num_labels)
        ]
        component_areas.sort(reverse=True)

        budget = int(total_pixels * max_ratio)
        clipped = np.zeros_like(binary)
        filled = 0
        for area, label_id in component_areas:
            if filled + area > budget:
                break
            clipped[labels == label_id] = 1
            filled += area

        cv2.imwrite(str(mask_path), (clipped * 255).astype(np.uint8))
        clipped_count += 1

    if clipped_count:
        print(f"  Clipped {clipped_count} mask(s) that exceeded {max_ratio:.0%} coverage.")


def _mask_grounded_sam(
    image_paths: List[Path],
    masks_dir: Path,
    device: str,
    furniture_labels: str,
    box_threshold: float,
    text_threshold: float,
    sam_checkpoint: str,
    sam_model_type: str,
) -> None:
    """Generate masks with GroundingDINO + SAM (Grounded-SAM).

    This is the **recommended** masking approach for interior design datasets
    because it allows you to specify which furniture categories to mask
    (e.g. ``"sofa,chair,table"``), rather than picking the centre object.

    Requires:
        pip install groundingdino-py segment-anything

    GroundingDINO model weights are downloaded automatically on first use.
    SAM checkpoint must be downloaded manually (see ``--sam_checkpoint``).

    Args:
        furniture_labels: Comma-separated list of furniture categories, e.g.
            ``"sofa,chair,table,bed,cabinet,lamp"``.
        box_threshold: GroundingDINO confidence threshold for bounding boxes.
        text_threshold: GroundingDINO token probability threshold.
    """
    try:
        import torch
        from groundingdino.util.inference import Model as GroundingDINO
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "Install with: pip install groundingdino-py segment-anything"
        ) from exc

    if not sam_checkpoint or not Path(sam_checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: '{sam_checkpoint}'. "
            "Download: wget -P checkpoints "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )

    labels = [lbl.strip() for lbl in furniture_labels.split(",") if lbl.strip()]
    caption = " . ".join(labels) + " ."

    print("Loading GroundingDINO model …")
    # Resolve config from the installed groundingdino-py package (no local clone needed).
    import importlib.util as _ilu
    _gd_pkg = Path(_ilu.find_spec("groundingdino").origin).parent
    gd_config = str(_gd_pkg / "config" / "GroundingDINO_SwinT_OGC.py")

    # Weights: look in project-local weights/ directory; auto-download if missing.
    _weights_dir = Path(__file__).parent.parent / "weights"
    gd_weights = str(_weights_dir / "groundingdino_swint_ogc.pth")
    if not Path(gd_weights).exists():
        import urllib.request
        _weights_dir.mkdir(parents=True, exist_ok=True)
        _url = (
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )
        print(f"  GroundingDINO weights not found. Downloading to {gd_weights} …")
        urllib.request.urlretrieve(_url, gd_weights)
        print("  Download complete.")

    gdino = GroundingDINO(
        model_config_path=gd_config,
        model_checkpoint_path=gd_weights,
        device=device,
    )

    print(f"Loading SAM ({sam_model_type}) for Grounded-SAM …")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    for img_path in tqdm(image_paths, desc="Masking (Grounded-SAM)"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Detect furniture bounding boxes with GroundingDINO
        # predict_with_caption returns (Detections, phrases) tuple
        detections, _phrases = gdino.predict_with_caption(
            image=image_bgr,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if len(detections.xyxy) == 0:
            # Save an empty mask so the resume logic won't reprocess this image
            empty_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), empty_mask)
            continue

        # Predict SAM masks for all detected boxes
        predictor.set_image(image_rgb)
        boxes = torch.tensor(detections.xyxy, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            boxes, image_rgb.shape[:2]
        )
        masks_pred, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Union all detected furniture masks
        combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for m in masks_pred.cpu().numpy():
            combined |= (m[0] * 255).astype(np.uint8)

        cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), combined)


def _mask_oneformer(
    image_paths: List[Path],
    masks_dir: Path,
    device: str,
    target_labels: Optional[List[str]] = None,
) -> None:
    """Generate masks with OneFormer panoptic segmentation.

    Model: ``shi-labs/oneformer_ade20k_swin_large`` (~12 GB VRAM).
    ADE20K covers 150 interior categories including sofa, chair, bed, table,
    cabinet, wardrobe, lamp, curtain, rug, mirror, and more.

    The function finds all predicted segments whose category name is in
    ``target_labels`` and combines them into a single inpainting mask.

    Args:
        target_labels: Set of ADE20K category names to include in the mask.
            Defaults to a broad set of furniture/object categories.
    """
    try:
        import torch
        from transformers import AutoProcessor, OneFormerForUniversalSegmentation
    except ImportError as exc:
        raise ImportError("pip install transformers torch") from exc

    if target_labels is None:
        target_labels = list(_ADE20K_FURNITURE_IDS)

    target_set = {lbl.lower() for lbl in target_labels}

    model_id = "shi-labs/oneformer_ade20k_swin_large"
    print(f"Loading OneFormer processor and model ({model_id}) …")
    processor = AutoProcessor.from_pretrained(model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_id,
        **_torch_dtype_kwarg(torch.float16 if device != "cpu" else torch.float32),
    ).to(device)
    model.eval()

    id2label: dict = model.config.id2label  # int → label string

    for img_path in tqdm(image_paths, desc="Masking (OneFormer)"):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        seg_map = result["segmentation"].cpu().numpy()  # (H, W) int with segment IDs
        combined = np.zeros_like(seg_map, dtype=np.uint8)

        for seg_info in result["segments_info"]:
            label_id = seg_info["label_id"]
            label_name = id2label.get(label_id, "").lower()
            # Bidirectional substring match so that e.g. "chair" matches both
            # "chair" and "armchair", and "coffee table" matches "table".
            if any(t in label_name or label_name in t for t in target_set):
                combined[seg_map == seg_info["id"]] = 255

        # Always write the mask (empty = all-black when no furniture detected),
        # so the resume logic won't reprocess this image on the next run.
        cv2.imwrite(str(masks_dir / (img_path.stem + ".png")), combined)


def generate_masks(
    dataset_dir: str,
    masker: str = "sam",
    splits: Optional[List[str]] = None,
    device: str = "cuda",
    # SAM / SAM 2 shared
    sam_checkpoint: str = "",
    # SAM specific
    model_type: str = "vit_h",
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.86,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 2000,
    # Grounded-SAM specific
    furniture_labels: str = _DEFAULT_FURNITURE_LABELS,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    # OneFormer specific
    target_labels: Optional[List[str]] = None,
    # Mask coverage cap
    max_mask_ratio: float = 0.70,
) -> None:
    """Generate per-image furniture masks and save them to ``masks/``.

    Five masking backends are available via the ``masker`` argument:

    * ``"sam"``          – Meta SAM ViT-H; automatic segments, centre heuristic.
                           Requires a local ``.pth`` checkpoint (``sam_checkpoint``).
    * ``"sam2"``         – Meta SAM 2; improved boundaries, no download needed.
    * ``"sam3"``         – Meta SAM 3; text-prompted concept segmentation,
                           no checkpoint download needed (auto from HuggingFace).
    * ``"grounded_sam"`` – GroundingDINO + SAM; text-prompted furniture detection
                           (most accurate for interior design – **recommended**).
    * ``"oneformer"``    – Panoptic segmentation on ADE20K 150 categories;
                           produces category-aware masks.

    Args:
        dataset_dir: Root dataset directory.
        masker: Which masking backend to use.
        splits: Splits to process (default: ``["train", "val"]``).
        device: PyTorch device string.
        sam_checkpoint: Path to SAM ``.pth`` checkpoint (required for ``"sam"``
            and ``"grounded_sam"``).
        model_type: SAM model variant (``"vit_h"``, ``"vit_l"``, ``"vit_b"``).
        points_per_side: Grid density for SAM / SAM 2 automatic mode.
        pred_iou_thresh: SAM predicted IoU threshold.
        stability_score_thresh: SAM stability score threshold.
        min_mask_region_area: Minimum mask area (pixels) to keep.
        furniture_labels: Comma-separated furniture categories for
            ``"grounded_sam"`` and ``"sam3"``.
        box_threshold: GroundingDINO box confidence threshold.
        text_threshold: GroundingDINO text probability threshold.
        target_labels: ADE20K category names to mask with ``"oneformer"``.
        max_mask_ratio: Maximum fraction of image pixels the mask may cover
            (default 0.70 = 70%).  Masks exceeding this threshold are clipped
            by discarding the smallest connected components.
    """
    _SUPPORTED = ("sam", "sam2", "sam3", "grounded_sam", "oneformer")
    if masker not in _SUPPORTED:
        raise ValueError(
            f"Unknown masker '{masker}'. Choose one of: {_SUPPORTED}"
        )

    splits = splits if splits is not None else ["train", "val"]
    root = Path(dataset_dir)

    for split in splits:
        images_dir = root / split / "images"
        masks_dir = root / split / "masks"
        if not images_dir.exists():
            print(f"  Skipping {split} – images directory not found.")
            continue
        masks_dir.mkdir(parents=True, exist_ok=True)

        paths = _image_paths(images_dir)

        # Resume: skip images whose mask file already exists
        pending = [p for p in paths if not (masks_dir / (p.stem + ".png")).exists()]
        skipped = len(paths) - len(pending)
        print(f"\n[{split}] {len(paths)} images – masker: {masker}")
        if skipped:
            print(f"  Resuming: skipping {skipped} already-masked images; {len(pending)} remaining.")
        if not pending:
            print(f"  All masks already exist, nothing to do.")
            continue
        paths = pending

        if masker == "sam":
            _mask_sam(
                paths, masks_dir, device, sam_checkpoint, model_type,
                points_per_side, pred_iou_thresh, stability_score_thresh,
                min_mask_region_area,
            )
        elif masker == "sam2":
            _mask_sam2(paths, masks_dir, device, points_per_side, pred_iou_thresh)
        elif masker == "sam3":
            _mask_sam3(paths, masks_dir, device, furniture_labels)
        elif masker == "grounded_sam":
            _mask_grounded_sam(
                paths, masks_dir, device, furniture_labels,
                box_threshold, text_threshold, sam_checkpoint, model_type,
            )
        elif masker == "oneformer":
            _mask_oneformer(paths, masks_dir, device, target_labels)

        _apply_max_mask_ratio(masks_dir, paths, max_mask_ratio)
        print(f"  Saved masks → {masks_dir}")


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset_dir: str,
    splits: Optional[List[str]] = None,
    min_mask_area_frac: float = 0.01,
    remove_empty: bool = False,
) -> dict:
    """Validate a prepared dataset and report quality statistics.

    Checks every split for:

    * images without a corresponding mask file,
    * masks that are empty or below ``min_mask_area_frac``,
    * images without a caption entry in ``captions.json``,
    * captions that reference non-existent images.

    When ``remove_empty`` is *True* empty/too-small mask files are deleted so
    that the masking step can be re-run for those images (the resume logic
    skips images whose mask already exists).

    Args:
        dataset_dir: Root dataset directory.
        splits: Splits to validate (default: ``["train", "val"]``).
        min_mask_area_frac: Minimum fraction of non-zero pixels for a mask
            to be considered valid.
        remove_empty: If *True*, delete mask files that are empty or below
            the area threshold so they can be re-generated.

    Returns:
        A dict mapping each split to its validation summary.
    """
    splits = splits if splits is not None else ["train", "val"]
    root = Path(dataset_dir)
    report: dict = {}

    for split in splits:
        images_dir = root / split / "images"
        masks_dir = root / split / "masks"
        captions_path = root / split / "captions.json"

        if not images_dir.exists():
            print(f"  [{split}] Skipping – images directory not found.")
            continue

        image_files = _image_paths(images_dir)
        image_names = {p.name for p in image_files}
        image_stems = {p.stem for p in image_files}

        # --- Masks ----------------------------------------------------------
        missing_masks: List[str] = []
        empty_masks: List[str] = []
        small_masks: List[str] = []
        valid_masks = 0

        if masks_dir.exists():
            for img in image_files:
                mask_path = masks_dir / (img.stem + ".png")
                if not mask_path.exists():
                    missing_masks.append(img.name)
                    continue
                m = np.array(Image.open(mask_path).convert("L"))
                area_frac = np.mean(m > 127)
                if area_frac == 0:
                    empty_masks.append(img.name)
                elif area_frac < min_mask_area_frac:
                    small_masks.append(img.name)
                else:
                    valid_masks += 1
        else:
            missing_masks = [p.name for p in image_files]

        # --- Captions -------------------------------------------------------
        captions: dict = {}
        if captions_path.exists():
            with open(captions_path, "r", encoding="utf-8") as fh:
                captions = json.load(fh)

        missing_captions = [
            name for name in image_names
            if name not in captions and name.rsplit(".", 1)[0] not in captions
        ]
        orphan_captions = [
            k for k in captions
            if k not in image_names and k not in image_stems
        ]

        # --- Report ---------------------------------------------------------
        summary = {
            "total_images": len(image_files),
            "valid_masks": valid_masks,
            "missing_masks": missing_masks,
            "empty_masks": empty_masks,
            "small_masks": small_masks,
            "missing_captions": missing_captions,
            "orphan_captions": orphan_captions,
        }
        report[split] = summary

        print(f"\n[{split}] Validation summary")
        print(f"  Images:           {len(image_files)}")
        print(f"  Valid masks:      {valid_masks}")
        print(f"  Missing masks:    {len(missing_masks)}")
        print(f"  Empty masks:      {len(empty_masks)}")
        print(f"  Small masks:      {len(small_masks)} (< {min_mask_area_frac:.2%} area)")
        print(f"  Missing captions: {len(missing_captions)}")
        print(f"  Orphan captions:  {len(orphan_captions)}")

        # --- Optional cleanup -----------------------------------------------
        if remove_empty and masks_dir.exists():
            removed = 0
            for name in empty_masks + small_masks:
                stem = name.rsplit(".", 1)[0]
                mask_file = masks_dir / (stem + ".png")
                if mask_file.exists():
                    mask_file.unlink()
                    removed += 1
            if removed:
                print(f"  Removed {removed} empty/small mask files for re-generation.")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data preparation utilities for interior inpainting fine-tuning"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── split ──────────────────────────────────────────────────────────────
    p_split = sub.add_parser("split", help="Split raw images into train/val")
    p_split.add_argument("--source_dir", required=True, help="Directory with raw images")
    p_split.add_argument("--output_dir", required=True, help="Root output directory")
    p_split.add_argument("--val_ratio", type=float, default=0.1)
    p_split.add_argument("--seed", type=int, default=42)

    # ── caption ────────────────────────────────────────────────────────────
    p_cap = sub.add_parser(
        "caption",
        help="Auto-generate captions (blip | blip2 | florence2)",
    )
    p_cap.add_argument("--dataset_dir", required=True)
    p_cap.add_argument(
        "--captioner",
        default="blip2",
        choices=["blip", "blip2", "florence2"],
        help=(
            "blip      – Salesforce/blip-image-captioning-large (~6 GB VRAM, fast)\n"
            "blip2     – Salesforce/blip2-opt-2.7b (~15 GB VRAM, best quality) [default]\n"
            "florence2 – microsoft/Florence-2-large (~8 GB VRAM, detailed)"
        ),
    )
    p_cap.add_argument("--splits", nargs="+", default=["train", "val"])
    p_cap.add_argument("--device", default="cuda")
    p_cap.add_argument("--batch_size", type=int, default=8)
    p_cap.add_argument("--max_new_tokens", type=int, default=60)
    p_cap.add_argument(
        "--interior_prefix",
        default="An interior design photo of",
        help="Conditional text prefix for BLIP-2 prompted generation (blip2 only)",
    )

    # ── mask ───────────────────────────────────────────────────────────────
    p_mask = sub.add_parser(
        "mask",
        help="Auto-generate furniture masks (sam | sam2 | sam3 | grounded_sam | oneformer)",
    )
    p_mask.add_argument("--dataset_dir", required=True)
    p_mask.add_argument(
        "--masker",
        default="grounded_sam",
        choices=["sam", "sam2", "sam3", "grounded_sam", "oneformer"],
        help=(
            "sam          – Meta SAM ViT-H, automatic mode, needs local .pth\n"
            "sam2         – Meta SAM 2, improved, no download needed\n"
            "sam3         – Meta SAM 3, text-prompted concept segmentation\n"
            "grounded_sam – GroundingDINO+SAM, text-prompted [default, recommended]\n"
            "oneformer    – Panoptic segmentation on ADE20K 150 categories"
        ),
    )
    p_mask.add_argument("--splits", nargs="+", default=["train", "val"])
    p_mask.add_argument("--device", default="cuda")
    # SAM / SAM 2 / Grounded-SAM
    p_mask.add_argument(
        "--sam_checkpoint",
        default="",
        help="Path to SAM .pth checkpoint (required for --masker sam or grounded_sam)",
    )
    p_mask.add_argument(
        "--model_type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant",
    )
    p_mask.add_argument("--points_per_side", type=int, default=32)
    p_mask.add_argument("--pred_iou_thresh", type=float, default=0.86)
    # Grounded-SAM
    p_mask.add_argument(
        "--furniture_labels",
        default=_DEFAULT_FURNITURE_LABELS,
        help="Comma-separated furniture categories for grounded_sam (default: broad set)",
    )
    p_mask.add_argument("--box_threshold", type=float, default=0.35)
    p_mask.add_argument("--text_threshold", type=float, default=0.25)
    # SAM specific
    p_mask.add_argument("--stability_score_thresh", type=float, default=0.92,
                        help="SAM stability score threshold (sam only, default: 0.92)")
    p_mask.add_argument("--min_mask_region_area", type=int, default=2000,
                        help="Minimum mask area in pixels (sam only, default: 2000)")
    # Mask coverage cap
    p_mask.add_argument(
        "--max_mask_ratio",
        type=float,
        default=0.70,
        help=(
            "Maximum fraction of image area the mask may cover (default: 0.70 = 70%%). "
            "Masks exceeding this are clipped by removing the smallest connected components."
        ),
    )

    # ── validate ──────────────────────────────────────────────────────────
    p_val = sub.add_parser(
        "validate",
        help="Validate dataset quality: check masks, captions, report issues",
    )
    p_val.add_argument("--dataset_dir", required=True)
    p_val.add_argument("--splits", nargs="+", default=["train", "val"])
    p_val.add_argument(
        "--min_mask_area_frac",
        type=float,
        default=0.01,
        help="Minimum fraction of non-zero pixels for a mask to be valid (default: 0.01)",
    )
    p_val.add_argument(
        "--remove_empty",
        action="store_true",
        help="Delete empty/small mask files so they can be re-generated",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "split":
        split_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    elif args.command == "caption":
        generate_captions(
            dataset_dir=args.dataset_dir,
            captioner=args.captioner,
            splits=args.splits,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            interior_prefix=args.interior_prefix,
        )

    elif args.command == "mask":
        generate_masks(
            dataset_dir=args.dataset_dir,
            masker=args.masker,
            splits=args.splits,
            device=args.device,
            sam_checkpoint=args.sam_checkpoint,
            model_type=args.model_type,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            min_mask_region_area=args.min_mask_region_area,
            furniture_labels=args.furniture_labels,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            max_mask_ratio=args.max_mask_ratio,
        )

    elif args.command == "validate":
        validate_dataset(
            dataset_dir=args.dataset_dir,
            splits=args.splits,
            min_mask_area_frac=args.min_mask_area_frac,
            remove_empty=args.remove_empty,
        )


if __name__ == "__main__":
    main()

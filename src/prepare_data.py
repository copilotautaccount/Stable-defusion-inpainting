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

# 3. Auto-generate object masks  (choose --masker sam | sam2 | grounded_sam | oneformer)
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
    --masker      oneformer
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
            captions[path.name] = processor.decode(out_ids, skip_special_tokens=True).strip()

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
            captions[path.name] = processor.decode(out_ids, skip_special_tokens=True).strip()

    return captions


def _caption_florence2(
    image_paths: List[Path],
    device: str,
    batch_size: int,
) -> dict:
    try:
        import torch
        # Sử dụng Auto class để linh hoạt hơn
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("pip install transformers torch einops timm") from exc

    model_id = "microsoft/Florence-2-large"
    print(f"Loading Florence-2 (An toàn) từ {model_id} …")

    _dtype = torch.float16 if device != "cuda" else torch.float32

    # SỬA LỖI TẠI ĐÂY:
    # Luôn sử dụng trust_remote_code=True cho cả Model và Processor 
    # để đảm bảo chúng dùng chung kiến trúc "custom" từ HuggingFace, 
    # tránh lỗi 'image_token' của bản native.
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    model_kwargs = _torch_dtype_kwarg(_dtype)
    model_kwargs["trust_remote_code"] = True
    
    # Florence-2 chạy tốt nhất với Flash Attention nếu có, 
    # nhưng "eager" là an toàn nhất cho mọi cấu hình.
    model_kwargs["attn_implementation"] = "eager" 

    model = AutoModelForCausalLM.from_pretrained(
        model_id, **model_kwargs
    ).to(device)
    model.eval()

    task_token = "<MORE_DETAILED_CAPTION>"
    captions: dict = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Captioning (Florence-2)"):
        batch = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        
        inputs = processor(
            text=[task_token] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        # Chuyển dữ liệu sang GPU/CPU
        inputs = {k: v.to(device).to(_dtype) if v.dtype == torch.float32 else v.to(device) 
                 for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            
        for path, out_ids, image in zip(batch, generated_ids, images):
            generated_text = processor.batch_decode(out_ids.unsqueeze(0), skip_special_tokens=False)[0]
            
            # Post-process để lấy nội dung text sạch
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_token,
                image_size=(image.width, image.height)
            )
            captions[path.name] = parsed_answer[task_token]

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

def _pick_best_mask(annotations: list, cx: int, cy: int) -> Optional[np.ndarray]:
    """Return the SAM annotation mask whose bounding-box centre is closest
    to (cx, cy) – a simple heuristic for the main furniture object."""
    if not annotations:
        return None

    def dist(ann: dict) -> float:
        x, y, bw, bh = ann["bbox"]
        return (x + bw / 2 - cx) ** 2 + (y + bh / 2 - cy) ** 2

    annotations.sort(key=dist)
    return annotations[0]["segmentation"].astype(np.uint8) * 255


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
) -> None:
    """Generate per-image furniture masks and save them to ``masks/``.

    Four masking backends are available via the ``masker`` argument:

    * ``"sam"``          – Meta SAM ViT-H; automatic segments, centre heuristic.
                           Requires a local ``.pth`` checkpoint (``sam_checkpoint``).
    * ``"sam2"``         – Meta SAM 2; improved boundaries, no download needed.
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
            ``"grounded_sam"``.
        box_threshold: GroundingDINO box confidence threshold.
        text_threshold: GroundingDINO text probability threshold.
        target_labels: ADE20K category names to mask with ``"oneformer"``.
    """
    _SUPPORTED = ("sam", "sam2", "grounded_sam", "oneformer")
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
        elif masker == "grounded_sam":
            _mask_grounded_sam(
                paths, masks_dir, device, furniture_labels,
                box_threshold, text_threshold, sam_checkpoint, model_type,
            )
        elif masker == "oneformer":
            _mask_oneformer(paths, masks_dir, device, target_labels)

        print(f"  Saved masks → {masks_dir}")


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
        help="Auto-generate furniture masks (sam | sam2 | grounded_sam | oneformer)",
    )
    p_mask.add_argument("--dataset_dir", required=True)
    p_mask.add_argument(
        "--masker",
        default="grounded_sam",
        choices=["sam", "sam2", "grounded_sam", "oneformer"],
        help=(
            "sam          – Meta SAM ViT-H, automatic mode, needs local .pth\n"
            "sam2         – Meta SAM 2, improved, no download needed\n"
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
            furniture_labels=args.furniture_labels,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )


if __name__ == "__main__":
    main()

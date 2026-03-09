# Stable Diffusion Inpainting – Dataset Pipeline

Pipeline to create training datasets for **Stable Diffusion inpainting** from paired
background / object photographs.

## How It Works

1. **Pair matching** – images in `background/` and `object/` are matched by filename.
2. **Mask generation** – pixel-level difference between each pair is computed, thresholded, and morphologically cleaned to produce a binary mask that highlights the added object.
3. **Caption generation** – a short text description of the masked region is generated (rule-based by default; optionally via the BLIP vision-language model).

### Output structure

```
output/
├── images/        # object images (copied)
├── masks/         # binary masks (white = inpaint region)
├── captions/      # per-image .txt caption files
└── metadata.json  # full dataset index
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/copilotautaccount/Stable-defusion-inpainting.git
cd Stable-defusion-inpainting

# Install dependencies
pip install -r requirements.txt

# Run the pipeline (rule-based captions)
python run_pipeline.py --input /path/to/dataset --output ./output

# Run with BLIP model captions (requires transformers + torch)
python run_pipeline.py --input /path/to/dataset --output ./output --caption-mode blip
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Root folder containing `background/` and `object/` |
| `--output` | `./output` | Output directory |
| `--caption-mode` | `simple` | `simple` (rule-based) or `blip` (model-based) |
| `--blur-kernel` | `5` | Gaussian blur kernel size for mask generation |
| `--threshold` | `25` | Pixel-difference threshold |
| `--morph-kernel` | `5` | Morphological operation kernel size |
| `--min-area` | `100` | Minimum contour area (pixels) to keep |

## Expected Input Layout

```
<dataset_root>/
├── background/   # images WITHOUT the added object
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── object/       # images WITH the added object (same filenames)
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Requirements

- Python ≥ 3.10
- OpenCV, NumPy, Pillow, tqdm
- *(optional)* `transformers` + `torch` for BLIP captions
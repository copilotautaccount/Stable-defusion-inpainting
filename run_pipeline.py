#!/usr/bin/env python3
"""CLI entry-point for the Stable Diffusion inpainting dataset pipeline.

Usage examples
--------------
# Simple (rule-based captions):
python run_pipeline.py --input /path/to/dataset --output ./output

# With BLIP model captions:
python run_pipeline.py --input /path/to/dataset --output ./output --caption-mode blip

# Custom mask parameters:
python run_pipeline.py --input /path/to/dataset --output ./output \
    --blur-kernel 7 --threshold 30 --morph-kernel 7 --min-area 200
"""

import argparse
import logging
import sys

from pipeline.dataset_builder import DatasetBuilder


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an inpainting dataset from paired background/object images.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Root directory of the dataset. Must contain 'background/' and "
            "'object/' sub-folders with matching filenames."
        ),
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Directory where the processed dataset will be written.",
    )
    parser.add_argument(
        "--caption-mode",
        choices=["simple", "blip"],
        default="simple",
        help="Caption generation mode (default: simple).",
    )
    # Mask-generator tunables
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=100)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mask_kwargs = {
        "blur_kernel": args.blur_kernel,
        "threshold": args.threshold,
        "morph_kernel": args.morph_kernel,
        "min_area": args.min_area,
    }

    builder = DatasetBuilder(
        dataset_root=args.input,
        output_dir=args.output,
        caption_mode=args.caption_mode,
        mask_kwargs=mask_kwargs,
    )
    metadata = builder.build()
    print(f"Done – processed {len(metadata)} image pairs.")


if __name__ == "__main__":
    main()

"""Prepare photos for labelling and training.

For every image in dataset/images:

    1. Apply the EXIF orientation to the pixel data. labelImg ignores the orientation tag
       while OpenCV and Ultralytics apply it, which would misalign the labels.
    2. Resize so that the longest side is at most --max-size pixels.
    3. Remove all metadata, including GPS coordinates.

The original files are moved to dataset/originals. Images that already have a label file
are skipped.

Usage:
    python training/prepare_photos.py [--max-size 1280]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "dataset" / "images"
LABELS = ROOT / "dataset" / "labels"
ORIGINALS = ROOT / "dataset" / "originals"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare photos for labelling and training.")
    parser.add_argument("--max-size", type=int, default=1280, help="maximum length of the longest side in pixels")
    args = parser.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    ORIGINALS.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found. Copy your photos to {IMAGES} and run this script again.")
        return 1

    processed = skipped = 0
    for src in files:
        if (ORIGINALS / src.name).exists() or (LABELS / f"{src.stem}.txt").exists():
            skipped += 1
            continue

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((args.max_size, args.max_size), Image.LANCZOS)
            tmp = src.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=92, optimize=True)  # saved without EXIF data

        shutil.move(src, ORIGINALS / src.name)
        tmp.rename(IMAGES / f"{src.stem}.jpg")
        processed += 1

    print(f"Processed {processed} image(s), skipped {skipped}. Originals moved to {ORIGINALS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

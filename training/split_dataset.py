"""Validate the labels in dataset/ and split the data into training and validation sets.

Output (Ultralytics layout):
    training/split/images/{train,val}
    training/split/labels/{train,val}

Usage:
    python training/split_dataset.py [--val 0.2] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGES = ROOT / "dataset" / "images"
SRC_LABELS = ROOT / "dataset" / "labels"
CLASSES_FILE = ROOT / "dataset" / "classes.txt"
OUT = ROOT / "training" / "split"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_label(path: Path, num_classes: int) -> list[str]:
    """Return a list of problems found in a YOLO label file."""
    errors = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{n}: expected 5 values, found {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{n}: values are not valid numbers")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{path.name}:{n}: unknown class {cls}")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w == 0 or h == 0:
            errors.append(f"{path.name}:{n}: coordinates must be between 0 and 1 and width/height must be > 0")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate labels and split the dataset.")
    parser.add_argument("--val", type=float, default=0.2, help="fraction of images used for validation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for a reproducible split")
    args = parser.parse_args()
    if not 0 < args.val < 1:
        parser.error("--val must be between 0 and 1")

    num_classes = len([c for c in CLASSES_FILE.read_text(encoding="utf-8").splitlines() if c.strip()])
    images = sorted(p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    labeled, unlabeled, errors = [], [], []
    for img in images:
        label = SRC_LABELS / f"{img.stem}.txt"
        if label.exists():
            errors += validate_label(label, num_classes)
            labeled.append(img)
        else:
            unlabeled.append(img)

    if errors:
        print("Invalid labels found:")
        for e in errors:
            print(f"  {e}")
        return 1

    # Unlabeled images would be treated as "no chicken here", so they are left out.
    if unlabeled:
        print(f"Skipping {len(unlabeled)} image(s) without a label file:")
        for img in unlabeled:
            print(f"  {img.name}")

    if len(labeled) < 2:
        print("At least two labeled images are required.")
        return 1

    random.Random(args.seed).shuffle(labeled)
    n_val = max(1, round(len(labeled) * args.val))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

    if OUT.exists():
        shutil.rmtree(OUT)

    for name, files in splits.items():
        img_dir = OUT / "images" / name
        lbl_dir = OUT / "labels" / name
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for img in files:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(SRC_LABELS / f"{img.stem}.txt", lbl_dir / f"{img.stem}.txt")

    print(f"Done: {len(splits['train'])} training / {len(splits['val'])} validation images in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Prueft die Labels in dataset/ und teilt den Datensatz in Train / Val auf (Standard 80 / 20).

Ergebnis (Ultralytics-Format, Labels werden ueber /images/ -> /labels/ gefunden):
    training/split/images/{train,val}
    training/split/labels/{train,val}

Aufruf (aus dem Projektordner):
    .venv\\Scripts\\python.exe training\\split_dataset.py [--val 0.2] [--seed 42]
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
    """Gibt eine Liste von Fehlermeldungen zurueck (leer = Label ist in Ordnung)."""
    errors = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{n}: erwartet 5 Werte, gefunden {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{n}: keine gueltigen Zahlen")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{path.name}:{n}: Klasse {cls} existiert nicht")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w == 0 or h == 0:
            errors.append(f"{path.name}:{n}: Koordinaten muessen zwischen 0 und 1 liegen (und w,h > 0)")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=float, default=0.2, help="Anteil Validierungsbilder (0..1)")
    ap.add_argument("--seed", type=int, default=42, help="Zufalls-Seed (gleicher Seed = gleiche Aufteilung)")
    args = ap.parse_args()
    if not 0 < args.val < 1:
        ap.error("--val muss zwischen 0 und 1 liegen")

    num_classes = len([c for c in CLASSES_FILE.read_text(encoding="utf-8").splitlines() if c.strip()])
    images = sorted(p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    labeled, unlabeled, errors = [], [], []
    for img in images:
        label = SRC_LABELS / f"{img.stem}.txt"
        if not label.exists():
            unlabeled.append(img)
            continue
        errors += validate_label(label, num_classes)
        labeled.append(img)

    if errors:
        print("FEHLER in den Labels - bitte korrigieren:")
        for e in errors:
            print("  -", e)
        return 1

    if unlabeled:
        # Bilder ohne Label wuerden als "hier ist kein Huhn" gelernt -> falsch, also weglassen.
        print("WARNUNG: Diese Bilder haben kein Label und werden uebersprungen:")
        for img in unlabeled:
            print("  -", img.name)

    if len(labeled) < 2:
        print("Zu wenige gelabelte Bilder.")
        return 1

    random.Random(args.seed).shuffle(labeled)
    n_val = max(1, round(len(labeled) * args.val))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

    # Alten Split loeschen, damit keine veralteten Dateien (und Ultralytics-*.cache) uebrig bleiben
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

    print(f"Fertig: {len(splits['train'])} Train / {len(splits['val'])} Val -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

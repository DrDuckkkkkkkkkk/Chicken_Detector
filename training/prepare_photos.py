"""Bereitet eigene Handyfotos fuers Labeln und Training vor.

Was passiert:
  1. EXIF-Drehung wird fest ins Bild eingerechnet. labelImg ignoriert die Markierung,
     OpenCV/Ultralytics beachten sie -> ohne diesen Schritt passen die Labels nicht.
  2. Bild wird auf maximal --max-size Pixel (laengste Seite) verkleinert.
  3. Alle Metadaten (GPS, Kamera, Datum, ...) werden entfernt.
  4. Ergebnis wird als JPG gespeichert.

Die Originale werden nach dataset/originals/ verschoben (nicht im Repository).

Aufruf (aus dem Projektordner):
    .venv\\Scripts\\python.exe training\\prepare_photos.py [--max-size 1280]
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
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=1280, help="laengste Bildseite in Pixel")
    args = ap.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"Keine Bilder gefunden. Fotos nach {IMAGES} kopieren und erneut ausfuehren.")
        return 1
    done = skipped = 0

    for src in files:
        backup = ORIGINALS / src.name
        if backup.exists():
            skipped += 1  # wurde schon verarbeitet
            continue
        if (LABELS / f"{src.stem}.txt").exists():
            # Schon gelabelt: Drehen/Verkleinern wuerde nichts kaputtmachen (Labels sind relativ),
            # aber eine Drehung schon. Sicherheitshalber nicht anfassen.
            print(f"Uebersprungen (hat schon ein Label): {src.name}")
            skipped += 1
            continue

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)  # Drehung anwenden, Orientation-Tag entfernen
            im = im.convert("RGB")
            im.thumbnail((args.max_size, args.max_size), Image.LANCZOS)
            tmp = src.with_suffix(".tmp.jpg")
            im.save(tmp, "JPEG", quality=92, optimize=True)  # ohne exif= -> keine Metadaten

        shutil.move(src, backup)
        tmp.rename(IMAGES / f"{src.stem}.jpg")
        done += 1

    print(f"Fertig: {done} Bilder aufbereitet, {skipped} uebersprungen. Originale in {ORIGINALS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

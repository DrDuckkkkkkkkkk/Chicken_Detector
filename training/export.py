"""Exportiert models/chicken.pt erneut nach ONNX, z.B. mit kleinerer Eingabegroesse fuer mehr FPS.

    .venv\\Scripts\\python.exe training\\export.py --imgsz 416
    -> models/chicken_416.onnx   (C++: --size 416)

Kleinere Groesse = deutlich schneller, aber kleine / weit entfernte Huehner werden schlechter erkannt.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    if args.imgsz % 32:
        ap.error("--imgsz muss ein Vielfaches von 32 sein")

    pt = MODELS / "chicken.pt"
    if not pt.exists():
        print("models/chicken.pt fehlt - zuerst training\\train.py ausfuehren.")
        return 1

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    target = MODELS / ("chicken.onnx" if args.imgsz == 640 else f"chicken_{args.imgsz}.onnx")
    # Ultralytics schreibt die .onnx neben die .pt -> in einem Temp-Ordner exportieren,
    # sonst wuerde models/chicken.onnx (640) ueberschrieben.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pt = Path(tmp) / "chicken.pt"
        shutil.copy2(pt, tmp_pt)
        onnx_path = YOLO(str(tmp_pt)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
        shutil.move(onnx_path, target)

    from sanitize_model import sanitize

    sanitize(target)  # lokale Pfade aus den Metadaten entfernen
    print(f"Gespeichert: {target}  (C++: --size {args.imgsz})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

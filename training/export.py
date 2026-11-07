"""Export models/chicken.pt to ONNX with a different input size.

Usage:
    python training/export.py --imgsz 416

Creates models/chicken_416.onnx, to be used with --size 416 in the C++ application.
Smaller sizes are faster but detect small or distant chickens less reliably.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from sanitize_model import sanitize

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the chicken model to ONNX.")
    parser.add_argument("--imgsz", type=int, default=640, help="input size, multiple of 32")
    args = parser.parse_args()
    if args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32")

    weights = MODELS / "chicken.pt"
    if not weights.exists():
        print("models/chicken.pt not found. Run training/train.py first.")
        return 1

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    target = MODELS / ("chicken.onnx" if args.imgsz == 640 else f"chicken_{args.imgsz}.onnx")

    # Ultralytics writes the .onnx next to the .pt file, so export from a temporary copy
    # to avoid overwriting models/chicken.onnx.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_weights = Path(tmp) / "chicken.pt"
        shutil.copy2(weights, tmp_weights)
        onnx_path = YOLO(str(tmp_weights)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True,
                                                  dynamic=False)
        shutil.move(onnx_path, target)

    sanitize(target)
    print(f"Saved {target} (use --size {args.imgsz} in the C++ application)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

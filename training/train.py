"""Train the chicken detector and export it to ONNX.

Usage:
    python training/train.py [--model yolo11n.pt] [--epochs 100] [--imgsz 640]

Use --model models/chicken.pt to continue training the existing model.

Output:
    models/chicken.pt     PyTorch weights (for further training)
    models/chicken.onnx   ONNX model for the C++ application
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from sanitize_model import sanitize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data.yaml"
SPLIT = ROOT / "training" / "split"
RUNS = ROOT / "training" / "runs"
MODELS = ROOT / "models"


def disable_telemetry() -> None:
    """Turn off Ultralytics usage analytics and automatic logger integrations."""
    from ultralytics import settings

    settings.update({"sync": False, "clearml": False, "comet": False, "dvc": False,
                     "mlflow": False, "raytune": False, "wandb": False})


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the chicken detector.")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="base model (yolo11n.pt, yolo11s.pt, ...) or models/chicken.pt to continue training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size, multiple of 32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30, help="stop after this many epochs without improvement")
    args = parser.parse_args()

    if not (SPLIT / "images" / "train").is_dir():
        print("training/split not found. Run training/split_dataset.py first.")
        return 1

    disable_telemetry()

    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training on {'GPU' if device == '0' else 'CPU'}")

    MODELS.mkdir(exist_ok=True)
    weights = args.model if Path(args.model).exists() else str(MODELS / args.model)

    YOLO(weights).train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        cache="ram",
        workers=4,
        seed=42,
        deterministic=True,
        plots=True,
        project=str(RUNS),
        name="chicken",
        exist_ok=True,
    )

    best = RUNS / "chicken" / "weights" / "best.pt"
    if not best.exists():
        print("Training finished without producing best.pt, see the output above.")
        return 1

    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(DATA), imgsz=args.imgsz, device=device, plots=False)
    print(f"Best model: mAP50 = {metrics.box.map50:.3f}, mAP50-95 = {metrics.box.map:.3f}")

    # Opset 12 with a static input size is reliably supported by OpenCV DNN.
    onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    shutil.copy2(best, MODELS / "chicken.pt")
    shutil.copy2(onnx_path, MODELS / "chicken.onnx")

    sanitize(MODELS / "chicken.pt")
    sanitize(MODELS / "chicken.onnx")
    print(f"Models saved to {MODELS} (use --size {args.imgsz} in the C++ application)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

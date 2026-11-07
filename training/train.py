"""Train the chicken detector and export it to ONNX.

Usage:
    python training/train.py [--model yolo11n.pt] [--epochs 100] [--imgsz 640]

Use --model models/chicken.pt to continue training the existing model. In that case a lower
learning rate without warm-up is used, so the model keeps what it has already learned.

If models/chicken.pt already exists, it is only replaced when the new model achieves a higher
mAP50-95 on the validation set. Use --force to replace it anyway.

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


def evaluate(weights: Path, imgsz: int, device: str) -> tuple[float, float]:
    """Return (mAP50, mAP50-95) of a model on the validation set."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(DATA), imgsz=imgsz, device=device, plots=False, verbose=False,
                                     project=str(RUNS), name="val", exist_ok=True)
    return metrics.box.map50, metrics.box.map


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the chicken detector.")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="base model (yolo11n.pt, yolo11s.pt, ...) or models/chicken.pt to continue training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="training image size, multiple of 32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30, help="stop after this many epochs without improvement")
    parser.add_argument("--force", action="store_true", help="replace the current model even if the new one is worse")
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
    weights = Path(args.model) if Path(args.model).exists() else MODELS / args.model

    current = MODELS / "chicken.pt"
    baseline = evaluate(current, args.imgsz, device) if current.exists() else None
    if baseline:
        print(f"Current model: mAP50 = {baseline[0]:.3f}, mAP50-95 = {baseline[1]:.3f}")

    # Pretrained base models (yolo11n.pt, ...) use the default schedule. An already trained
    # chicken model is fine-tuned gently; the default warm-up would undo much of what it learned.
    fine_tune = not weights.name.lower().startswith("yolo")
    extra = dict(warmup_epochs=0, optimizer="AdamW", lr0=0.0005, lrf=0.1) if fine_tune else {}
    if fine_tune:
        print(f"Continuing training of {weights.name}")

    YOLO(str(weights)).train(
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
        **extra,
    )

    best = RUNS / "chicken" / "weights" / "best.pt"
    if not best.exists():
        print("Training finished without producing best.pt, see the output above.")
        return 1

    new_map50, new_map = evaluate(best, args.imgsz, device)
    print(f"New model: mAP50 = {new_map50:.3f}, mAP50-95 = {new_map:.3f}")

    if baseline and new_map <= baseline[1] and not args.force:
        print("The new model is not better than the current one, so models/ was left unchanged.")
        print(f"The new model is still available at {best}. Use --force to replace the current model anyway.")
        return 0

    # Opset 12 with a static input size is reliably supported by OpenCV DNN.
    onnx_path = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    shutil.copy2(best, MODELS / "chicken.pt")
    shutil.copy2(onnx_path, MODELS / "chicken.onnx")

    sanitize(MODELS / "chicken.pt")
    sanitize(MODELS / "chicken.onnx")
    print(f"Models saved to {MODELS} (use --size {args.imgsz} in the C++ application)")
    print("Run training/export.py --imgsz 416 and --imgsz 320 to update the smaller models as well.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

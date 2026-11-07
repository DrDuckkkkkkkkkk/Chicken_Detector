"""Trainiert ein YOLO-Modell auf dem Huehner-Datensatz und exportiert es nach ONNX.

Aufruf (aus dem Projektordner):
    .venv\\Scripts\\python.exe training\\train.py [--epochs 100] [--imgsz 640] [--model yolo11n.pt]

Ergebnis:
    models/chicken.pt    PyTorch-Gewichte (fuer Python / Weitertrainieren)
    models/chicken.onnx  ONNX-Modell fuer das C++-Programm
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "training" / "data.yaml"
SPLIT = ROOT / "training" / "split"
RUNS = ROOT / "training" / "runs"
MODELS = ROOT / "models"


def disable_telemetry() -> None:
    """Ultralytics sendet standardmaessig anonyme Nutzungsdaten und Absturzberichte ("sync").
    Das wird hier dauerhaft abgeschaltet, ebenso automatische Logger-Integrationen."""
    from ultralytics import settings

    settings.update({"sync": False, "clearml": False, "comet": False, "dvc": False,
                     "mlflow": False, "raytune": False, "wandb": False})


def main() -> int:
    ap = argparse.ArgumentParser(description="Huehner-Detektor trainieren")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Startmodell: yolo11n/s/m.pt (neu) oder models/chicken.pt (vorhandenes Modell weitertrainieren)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="Bildgroesse (Vielfaches von 32)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30, help="frueh stoppen nach n Epochen ohne Verbesserung")
    args = ap.parse_args()

    if not (SPLIT / "images" / "train").is_dir():
        print("training/split fehlt - zuerst training\\split_dataset.py ausfuehren.")
        return 1

    disable_telemetry()

    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Trainiere auf: {'GPU' if device == '0' else 'CPU'}")

    # Vortrainierte Gewichte landen in models/ statt im aktuellen Arbeitsordner
    MODELS.mkdir(exist_ok=True)
    weights = args.model if Path(args.model).exists() else str(MODELS / args.model)

    model = YOLO(weights)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        cache="ram",       # Datensatz ist klein -> komplett im RAM halten, spart Lesezugriffe
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
        print("Training ohne best.pt beendet - siehe Ausgabe oben.")
        return 1

    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(DATA), imgsz=args.imgsz, device=device, plots=False)
    print(f"Bestes Modell: mAP50 = {metrics.box.map50:.3f}, mAP50-95 = {metrics.box.map:.3f}")

    shutil.copy2(best, MODELS / "chicken.pt")
    # opset 12 + simplify: wird von OpenCV DNN zuverlaessig geladen. Feste Eingabegroesse = schneller.
    onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)
    shutil.copy2(onnx_path, MODELS / "chicken.onnx")

    # Lokale Pfade (Benutzername) und Trainingsballast aus den Modelldateien entfernen
    from sanitize_model import sanitize

    sanitize(MODELS / "chicken.pt")
    sanitize(MODELS / "chicken.onnx")
    print(f"Modelle gespeichert in {MODELS}  (C++: --size {args.imgsz})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

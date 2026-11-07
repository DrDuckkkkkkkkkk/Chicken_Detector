"""Entfernt persoenliche Daten aus Modelldateien, bevor sie veroeffentlicht werden.

Ultralytics schreibt beim Training automatisch absolute Pfade (inkl. Windows-Benutzername),
Git-Informationen und den Optimizer-Zustand in die Modelldateien. Dieses Skript entfernt das:

  .pt    Optimizer, EMA, Git-Infos und Pfade werden entfernt (Datei wird auch kleiner).
         Die Gewichte bleiben vollstaendig -> zum Erkennen und Weitertrainieren geeignet.
  .onnx  Pfade in den Metadaten werden entfernt.

Wird von train.py und export.py automatisch aufgerufen. Manuell:
    .venv\\Scripts\\python.exe training\\sanitize_model.py models\\chicken.pt models\\chicken.onnx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Absolute Windows- oder Unix-Pfade erkennen
_ABS_PATH = re.compile(r"([A-Za-z]:[\\/]|\\\\|/(home|Users|root)/)")


def _clean_value(value):
    """Ersetzt absolute Pfade durch den reinen Datei-/Ordnernamen."""
    if isinstance(value, str) and _ABS_PATH.match(value):
        return Path(value.replace("\\", "/")).name
    return value


def sanitize_pt(path: Path) -> None:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("ema") is not None:
        ckpt["model"] = ckpt["ema"]  # EMA-Gewichte sind die besten, wie bei Ultralytics strip_optimizer
    for key in ("ema", "optimizer", "updates", "scaler", "git", "train_results"):
        ckpt.pop(key, None)
    ckpt["epoch"] = -1  # markiert "fertig trainiert" -> Weitertrainieren startet sauber neu

    if isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"] = {k: _clean_value(v) for k, v in ckpt["train_args"].items()}

    model = ckpt["model"]
    model.half()  # fp16 wie bei Ultralytics-Releases, halbiert die Groesse
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "args") and isinstance(model.args, dict):
        model.args = {k: _clean_value(v) for k, v in model.args.items()}
    # Ultralytics merkt sich am Modell, aus welcher Datei es geladen wurde
    for attr in ("pt_path", "yaml_file"):
        if isinstance(getattr(model, attr, None), str):
            setattr(model, attr, _clean_value(getattr(model, attr)))

    torch.save(ckpt, path)


def sanitize_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    for prop in model.metadata_props:
        if prop.key == "description":
            prop.value = "Ultralytics YOLO11 chicken detection model"
        elif _ABS_PATH.search(prop.value):
            prop.value = re.sub(r"[A-Za-z]:[\\/][^'\",}]*|/(home|Users|root)/[^'\",}]*", "", prop.value)
    onnx.save(model, str(path))


def check_clean(path: Path) -> bool:
    data = path.read_bytes()
    markers = [b":\\Users\\", b":/Users/", b"\\\\Users\\\\"]
    return not any(m in data for m in markers)


def sanitize(path: Path) -> None:
    if path.suffix == ".pt":
        sanitize_pt(path)
    elif path.suffix == ".onnx":
        sanitize_onnx(path)
    else:
        raise ValueError(f"Unbekannter Dateityp: {path}")
    if not check_clean(path):
        raise RuntimeError(f"{path.name} enthaelt nach der Bereinigung noch lokale Pfade")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        p = Path(arg)
        sanitize(p)
        print(f"Bereinigt: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

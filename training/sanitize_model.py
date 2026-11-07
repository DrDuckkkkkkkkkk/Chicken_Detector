"""Remove local file paths and training state from model files before sharing them.

Ultralytics stores absolute paths (including the user name), git information and the
optimizer state inside trained models. This script removes them:

    .pt     drops optimizer, EMA, git info and absolute paths; weights are kept, so the
            model can still be used for inference and further training
    .onnx   replaces paths in the metadata

train.py and export.py call this automatically. Manual use:

    python training/sanitize_model.py models/chicken.pt models/chicken.onnx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches absolute paths such as C:\Users\..., C:/Users/..., /home/... but not URLs (https://).
_ABS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?![\\/])[^'\",}]*|(?<![A-Za-z0-9:/])/(?:home|Users|root)/[^'\",}]*")
_LEAK_MARKERS = (b":\\Users\\", b":/Users/", b"\\\\Users\\\\", b"/home/")


def _strip_path(value):
    """Reduce an absolute path to its final component, leave everything else untouched."""
    if isinstance(value, str) and _ABS_PATH.fullmatch(value):
        return Path(value.replace("\\", "/")).name
    return value


def sanitize_pt(path: Path) -> None:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("ema") is not None:
        ckpt["model"] = ckpt["ema"]
    for key in ("ema", "optimizer", "updates", "scaler", "git", "train_results"):
        ckpt.pop(key, None)
    ckpt["epoch"] = -1  # marks the training as finished

    if isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"] = {k: _strip_path(v) for k, v in ckpt["train_args"].items()}

    model = ckpt["model"].half()
    for p in model.parameters():
        p.requires_grad = False
    if isinstance(getattr(model, "args", None), dict):
        model.args = {k: _strip_path(v) for k, v in model.args.items()}
    for attr in ("pt_path", "yaml_file"):
        if isinstance(getattr(model, attr, None), str):
            setattr(model, attr, _strip_path(getattr(model, attr)))

    torch.save(ckpt, path)


def sanitize_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path))
    for prop in model.metadata_props:
        if prop.key == "description":
            prop.value = "Ultralytics YOLO11 chicken detection model"
        else:
            prop.value = _ABS_PATH.sub("", prop.value)
    onnx.save(model, str(path))


def is_clean(path: Path) -> bool:
    data = path.read_bytes()
    return not any(marker in data for marker in _LEAK_MARKERS)


def sanitize(path: Path) -> None:
    path = Path(path)
    if path.suffix == ".pt":
        sanitize_pt(path)
    elif path.suffix == ".onnx":
        sanitize_onnx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if not is_clean(path):
        raise RuntimeError(f"{path.name} still contains local paths after sanitizing")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        sanitize(Path(arg))
        print(f"Sanitized {arg}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

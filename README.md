# ChickenDetector

ChickenDetector detects and counts chickens in images, videos and live webcam streams.

The project covers the full pipeline: preparing and labelling a custom dataset, fine-tuning a YOLO11 model with Ultralytics, exporting it to ONNX, and running inference from a standalone C++ application built on OpenCV. The C++ application has no Python dependency and runs entirely offline.

## Table of Contents

- [Features](#features)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Usage](#usage)
- [Performance](#performance)
- [Implementation Details](#implementation-details)
- [Privacy](#privacy)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Detection on single images, image folders, video files and webcams
- Bounding boxes with confidence scores and a live chicken count
- Optional saving of annotated images and videos
- Built-in benchmark mode
- Multiple model sizes to trade accuracy for speed
- Label validation and reproducible train/validation split
- Photo preprocessing that corrects EXIF rotation and removes metadata

## Pipeline Overview

```
Photos -> prepare_photos.py -> labelImg -> split_dataset.py -> train.py -> ONNX model -> ChickenDetector
```

| Step | Tool | Output |
|------|------|--------|
| 1. Preprocess photos | `prepare_photos.py` | Rotated, resized images without metadata |
| 2. Annotate | labelImg | YOLO label files |
| 3. Validate and split | `split_dataset.py` | Training and validation sets |
| 4. Train | `train.py` | `chicken.pt`, `chicken.onnx` |
| 5. Export (optional) | `export.py` | Models with smaller input sizes |
| 6. Inference | `ChickenDetector.exe` | Detections, counts, annotated output |

## Project Structure

```
ChickenDetector/
├── dataset/
│   ├── images/              Preprocessed images (not tracked)
│   ├── labels/              YOLO annotations, one file per image (not tracked)
│   ├── originals/           Original photos (not tracked)
│   └── classes.txt          Class names
├── training/
│   ├── prepare_photos.py    EXIF rotation, resizing, metadata removal
│   ├── split_dataset.py     Label validation and train/val split
│   ├── data.yaml            Ultralytics dataset configuration
│   ├── train.py             Training and ONNX export
│   └── export.py            ONNX export with custom input size
├── src/
│   ├── Detector.h           Preprocessing, inference, post-processing
│   └── main.cpp             Command-line interface
├── models/                  Trained models (generated, not tracked)
├── CMakeLists.txt
├── requirements.txt
└── README.md
```

The directories `training/split/`, `training/runs/`, `models/` and `results/` are generated during use and excluded from version control.

## Requirements

| Component | Version |
|-----------|---------|
| Operating system | Windows 10/11 (x64) |
| Compiler | Visual Studio 2022 or newer, "Desktop development with C++" workload |
| CMake | 3.20 or newer (included with Visual Studio) |
| OpenCV | 4.x with the `dnn`, `highgui` and `videoio` modules (via vcpkg) |
| Python | 3.10 or newer (training) |
| Python | 3.9 (labelImg only, separate environment) |

A GPU is optional. Training uses CUDA automatically if available. All results below were produced on a CPU.

## Installation

### 1. Clone the repository

```
git clone https://github.com/<username>/ChickenDetector.git
cd ChickenDetector
```

### 2. Install OpenCV

```
vcpkg install opencv4[dnn,highgui,ffmpeg]:x64-windows
```

### 3. Set up the Python environment

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Build the application

```
cmake -S . -B build
cmake --build build --config Release
```

CMake locates vcpkg via the `VCPKG_ROOT` environment variable and falls back to `%USERPROFILE%\vcpkg`. A custom location can be specified with `-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`.

The executable is written to `build\Release\ChickenDetector.exe`. The required OpenCV DLLs are copied to the same directory.

## Dataset Preparation

The current model was trained on 116 images with 461 annotated chickens, captured in an indoor coop under mostly low-light conditions. The dataset itself is not part of this repository. To train a model, provide your own images and annotations as described below.

### Preprocessing

Place the source images in `dataset/images/` and run:

```
.venv\Scripts\python.exe training\prepare_photos.py
```

The script performs the following steps:

1. **EXIF rotation.** Smartphone images are frequently stored in landscape orientation with an EXIF orientation tag. labelImg ignores this tag, whereas OpenCV and Ultralytics apply it, which results in misaligned annotations. The script applies the rotation to the pixel data and removes the tag.
2. **Resizing.** Images are scaled to a maximum of 1280 px on the longest side (configurable with `--max-size`).
3. **Metadata removal.** All metadata, including GPS coordinates, is removed.

Original files are moved to `dataset/originals/`. Images that already have an annotation are skipped, so the script can be run again after adding new images.

### Annotation

Annotation is done with [labelImg](https://github.com/HumanSignal/labelImg). labelImg is not compatible with Python 3.10 or newer, so a separate environment is required:

```
conda create -n label_env python=3.9
conda activate label_env
pip install labelImg
labelImg dataset\images dataset\classes.txt dataset\labels
```

Recommended settings:

- Set the annotation format to **YOLO** (toolbar button on the left).
- Enable *View > Auto Save mode*.
- Enable *Use default label* with the value `Chicken`.

Annotation guidelines:

- Annotate every visible chicken, including partially occluded and truncated ones.
- Draw tight boxes that include the entire bird.
- Images without chickens may be saved without boxes and serve as negative examples.

### Label format

Each label file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to the range 0 to 1. The only class is `0` (Chicken).

## Training

```
.venv\Scripts\python.exe training\split_dataset.py
.venv\Scripts\python.exe training\train.py
```

`split_dataset.py` validates all label files and aborts with a descriptive error if a file contains an invalid number of values, an unknown class, or coordinates outside the valid range. Images without a label file are excluded. The remaining images are split 80/20 into training and validation sets using a fixed seed.

| Option | Default | Description |
|--------|---------|-------------|
| `--val` | 0.2 | Fraction of images used for validation |
| `--seed` | 42 | Random seed for the split |

`train.py` fine-tunes a pretrained YOLO11 model and exports the best checkpoint to ONNX.

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `yolo11n.pt` | Base model (`yolo11s.pt` is more accurate but approx. 2.5x slower) |
| `--epochs` | 100 | Maximum number of epochs |
| `--patience` | 30 | Early stopping after this many epochs without improvement |
| `--batch` | 8 | Batch size |
| `--imgsz` | 640 | Input size (multiple of 32) |

The current model was trained with:

```
.venv\Scripts\python.exe training\train.py --epochs 250 --patience 60 --batch 16
```

Outputs:

- `models/chicken.pt`: PyTorch weights
- `models/chicken.onnx`: ONNX model for the C++ application
- `training/runs/chicken/`: training curves, confusion matrix, sample predictions and `results.csv` with per-epoch metrics

### Exporting smaller models

```
.venv\Scripts\python.exe training\export.py --imgsz 320
```

This produces `models/chicken_320.onnx`, which must be used with `--size 320`.

## Usage

```
ChickenDetector.exe <model.onnx> <source> [options]
```

`<source>` may be an image file, a directory of images, a video file, or a webcam index.

### Examples

```
build\Release\ChickenDetector.exe models\chicken_416.onnx image.jpg --size 416
build\Release\ChickenDetector.exe models\chicken.onnx images\
build\Release\ChickenDetector.exe models\chicken.onnx video.mp4 --out results
build\Release\ChickenDetector.exe models\chicken_320.onnx 0 --size 320
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--conf <0..1>` | 0.25 | Minimum confidence score |
| `--nms <0..1>` | 0.45 | IoU threshold for non-maximum suppression |
| `--size <n>` | 640 | Model input size; must match the export size |
| `--out <dir>` | `results` | Output directory; videos are saved only when specified |
| `--no-show` | | Disable the display window |
| `--bench <n>` | | Run detection n times per image and report timing |

### Controls

| Key | Action |
|-----|--------|
| Any key | Next image |
| `Esc` / `Q` | Exit |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime error (missing file, unreadable source, model error) |
| 2 | Invalid command-line arguments |

## Performance

Accuracy was measured on the 23 validation images. Inference time was measured on an 8-thread CPU without GPU acceleration, averaged over 100 runs (`--bench 100`).

| Model | Input size | Inference time | FPS | mAP50 | mAP50-95 |
|-------|------------|----------------|-----|-------|----------|
| `chicken.onnx` | 640 | 118 ms | 8.5 | 0.848 | 0.503 |
| `chicken_416.onnx` | 416 | 53 ms | 19 | 0.863 | 0.517 |
| `chicken_320.onnx` | 320 | 33 ms | 30 | 0.790 | 0.457 |

The 416 model offers the best overall trade-off and is the recommended default. On this dataset it is slightly more accurate than the 640 model while running more than twice as fast. The 320 model is suited for real-time webcam use where frame rate takes priority.

Training stopped early after 146 of 250 epochs; the best checkpoint was reached at epoch 86 (approx. 1.3 hours on CPU).

Profiling shows that nearly all processing time is spent in the neural network. Pre- and post-processing are negligible, so reducing the input size is the most effective way to improve throughput.

## Implementation Details

### Preprocessing

Each frame is letterboxed: scaled to fit the model input while preserving its aspect ratio and padded with gray (value 114). This matches the preprocessing used by Ultralytics during training. Pixel values are normalized to 0 to 1 and converted from BGR to RGB.

### Post-processing

YOLO11 produces 8400 candidate boxes per image. The decoder reads the output tensor directly and evaluates the confidence score first. Box coordinates are computed and mapped back to the original image only for candidates above the threshold. Overlapping boxes are then removed with non-maximum suppression.

Both standard (`[1, 4+C, N]`) and transposed (`[1, N, 4+C]`) output layouts are supported.

### Memory and allocation

All intermediate buffers (letterbox image, input blob, network output, candidate lists) are members of the `Detector` class and are reused across frames, so no memory is allocated per frame in steady state. Resource management relies exclusively on RAII; the code contains no manual `new` or `delete`.

### Warm-up

The constructor performs one inference on a blank image. This avoids the significantly higher latency of the first inference call in OpenCV DNN.

### Webcam capture

A dedicated capture thread continuously reads frames, and the detector always processes the most recent one. If inference is slower than the camera frame rate, older frames are discarded instead of being queued, which keeps latency constant. Video files are processed frame by frame without dropping frames.

### Error handling

Invalid arguments, missing files, unreadable sources and a mismatch between `--size` and the model's input size are reported with descriptive messages instead of unhandled exceptions.

## Privacy

- Ultralytics collects anonymous usage data by default. `train.py` and `export.py` disable this via the Ultralytics settings (`sync: False`). This change applies to the global Ultralytics configuration.
- `prepare_photos.py` removes all EXIF metadata, including GPS location data.
- The C++ application does not establish any network connections.
- `data.yaml` uses relative paths, so no local user paths are stored in the repository.

## Limitations

- Single class only; individual chickens are not distinguished.
- The training data originates from a single coop. Performance in substantially different environments may be lower.
- Densely grouped chickens, for example on a perch, may be merged into one detection or missed.
- Detection quality decreases in very dark or motion-blurred images.
- Currently tested on Windows only.

## Roadmap

- [ ] Extend the dataset with daylight and outdoor images
- [ ] Evaluate ONNX Runtime as an inference backend (approx. 1.7x faster in initial tests)
- [ ] Time-series logging of chicken counts
- [ ] Linux and Raspberry Pi support
- [ ] Publish trained models as GitHub releases

## License

This project is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for the full text.

The license was chosen for consistency with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), which is also licensed under AGPL-3.0. Models trained with this project are derived from Ultralytics weights and are subject to the same terms.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics): model architecture and training framework
- [OpenCV](https://opencv.org/): image processing and DNN inference
- [labelImg](https://github.com/HumanSignal/labelImg): annotation tool

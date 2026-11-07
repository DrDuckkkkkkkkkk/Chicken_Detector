# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-07

### Added

- C++ detector based on OpenCV DNN with support for images, image folders, video files and webcams
- Command-line options for confidence threshold, NMS threshold, input size, output directory and benchmarking
- Threaded webcam capture that always processes the most recent frame
- Photo preprocessing script (EXIF rotation, resizing, metadata removal)
- Label validation and reproducible train/validation split
- Training script with ONNX export and disabled Ultralytics telemetry
- Export script for models with smaller input sizes

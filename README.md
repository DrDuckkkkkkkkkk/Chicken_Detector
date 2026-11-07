# ChickenDetector

ChickenDetector finds and counts chickens in images, videos and live webcam streams.

The project contains everything needed to go from your own photos to a working detector: scripts to prepare and label a dataset, a training pipeline based on YOLO11, and a fast C++ application that runs the trained model with OpenCV. The C++ application works without Python and does not need an internet connection.

A trained model is included, so you can start detecting chickens right after building the application. The training photos themselves are not part of the repository.

The included model works best in conditions similar to the ones it was trained in: an indoor chicken coop, mostly in low light. For the best results in your own setup, you can continue training it with photos of your own chickens. The model then learns your coop, your lighting and your birds.


## Features

- Ready to use: a trained chicken model is included
- Detects chickens in single images, whole folders, video files and webcam streams
- Draws a box with a confidence score around every chicken and shows the total count
- Saves annotated images and videos
- Includes a benchmark mode to measure speed
- Supports several model sizes, so you can choose between accuracy and speed
- Checks your labels for errors before training
- Prepares phone photos automatically (rotation, size, removal of location data)
- The included model can be trained further with your own photos


## Quick start

If you only want to detect chickens, you do not need Python. Install OpenCV, build the application as described under Installation, and run it with the included model:

    build\Release\ChickenDetector.exe models\chicken_416.onnx photo.jpg --size 416

For a live webcam:

    build\Release\ChickenDetector.exe models\chicken_320.onnx 0 --size 320


## How it works

1. Phone photos are cleaned up with `prepare_photos.py`.
2. Every chicken in the photos is marked with a box in labelImg.
3. `split_dataset.py` checks the labels and splits the images into a training set and a validation set.
4. `train.py` fine-tunes a pretrained YOLO11 model on these images and exports it as an ONNX file.
5. The C++ application loads the ONNX file and detects chickens.


## Project layout

    dataset/
        images/             your prepared photos
        labels/             one label file per photo
        originals/          untouched original photos
        classes.txt         list of class names
    training/
        prepare_photos.py   fixes rotation, resizes, removes metadata
        split_dataset.py    checks labels, splits into training and validation
        data.yaml           dataset settings for training
        train.py            trains the model and exports it
        export.py           exports the model with a different input size
        sanitize_model.py   removes local paths from model files
    src/
        Detector.h          detection logic
        main.cpp            command line application
    models/
        chicken.pt          trained model, used for further training
        chicken.onnx        model for the application, input size 640
        chicken_416.onnx    model for the application, input size 416
        chicken_320.onnx    model for the application, input size 320

Images, labels and build output are not part of the repository.


## Requirements

- Windows 10 or 11, 64-bit
- Visual Studio 2022 or newer with the "Desktop development with C++" workload
- vcpkg for installing OpenCV
- Python 3.10 or newer, only needed for training
- A separate Python 3.9 environment for labelImg, only needed for labelling

A graphics card is not required. All results below were measured on a normal CPU. If an NVIDIA graphics card is available, training uses it automatically.


## Installation

Clone the repository:

    git clone https://github.com/DrDuckkkkkkkkkk/Chicken_Detector.git
    cd Chicken_Detector

Install OpenCV with vcpkg:

    vcpkg install opencv4[dnn,highgui,ffmpeg]:x64-windows

Set up Python (only needed if you want to train):

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

Build the application:

    cmake -S . -B build
    cmake --build build --config Release

The finished program is located at `build\Release\ChickenDetector.exe`.

CMake looks for vcpkg in the folder set in the `VCPKG_ROOT` environment variable. If that variable is not set, it uses `%USERPROFILE%\vcpkg`.


## Preparing your own dataset

To improve the model for your own chickens, you need your own photos. The included model was trained on 116 photos with 461 chickens. When training a new model from scratch, around 100 photos are a good starting point. When continuing to train the included model, fewer photos can already make a noticeable difference.

### Preparing the photos

Copy your photos into `dataset/images` and run:

    .venv\Scripts\python.exe training\prepare_photos.py

This step is important for photos taken with a phone. Phones often save pictures sideways and only store a note that the picture should be rotated. labelImg ignores this note, but the training does not. Without this step, the boxes you draw end up in the wrong place.

The script also reduces the photos to a maximum of 1280 pixels and removes all metadata, including the GPS location. The original photos are moved to `dataset/originals`.

### Labelling

The photos are labelled with labelImg. labelImg only works with Python 3.9 or older, so it needs its own environment:

    conda create -n label_env python=3.9
    conda activate label_env
    pip install labelImg
    labelImg dataset\images dataset\classes.txt dataset\labels

Before you start, switch the format in the left toolbar from PascalVOC to YOLO. It also helps to turn on Auto Save in the View menu and to set "Chicken" as the default label.

Useful keys: W draws a new box, D opens the next image, A goes back to the previous one.

Some tips for good results:

- Mark every chicken, including chickens that are partly hidden or cut off at the edge of the picture.
- Draw the box tightly around the whole bird.
- Photos without any chickens are also useful. Save them without boxes.


## Training

    .venv\Scripts\python.exe training\split_dataset.py
    .venv\Scripts\python.exe training\train.py

The first script checks every label file and stops with an error message if something is wrong. It then puts 80 percent of the images into the training set and 20 percent into the validation set. The split is the same every time you run it.

The second script trains the model. When it is finished, `chicken.pt` and `chicken.onnx` in the `models` folder are replaced with the new model. Charts and example results from the training are saved in `training/runs/chicken`.

There are two ways to train.

Continue training the included model. This is the recommended way if you want the detector to work better with your own chickens. The model keeps what it has already learned and adapts to your photos:

    .venv\Scripts\python.exe training\train.py --model models\chicken.pt --epochs 100

Train a new model from scratch, starting from the general YOLO11 model. The base model is downloaded automatically on the first run. The included model was trained this way:

    .venv\Scripts\python.exe training\train.py --epochs 250 --patience 60 --batch 16

Other useful options are `--model yolo11s.pt` for a larger and more accurate, but slower model, and `--imgsz` to change the image size used for training.

Tip: make a copy of the `models` folder before training, so you can go back to the previous model if the new one does not perform better.

A smaller and faster model can be exported afterwards:

    .venv\Scripts\python.exe training\export.py --imgsz 416

This creates `models/chicken_416.onnx`.


## Usage

    ChickenDetector.exe <model> <source> [options]

The source can be an image, a folder with images, a video file or the number of a webcam.

Examples:

    build\Release\ChickenDetector.exe models\chicken_416.onnx photo.jpg --size 416
    build\Release\ChickenDetector.exe models\chicken.onnx my_photos
    build\Release\ChickenDetector.exe models\chicken.onnx video.mp4 --out results
    build\Release\ChickenDetector.exe models\chicken_320.onnx 0 --size 320

Options:

- `--conf` sets how confident the model must be before something counts as a chicken. The value is between 0 and 1, the default is 0.25.
- `--nms` sets how much two boxes may overlap before one of them is removed. The default is 0.45.
- `--size` must match the size the model was exported with. The default is 640.
- `--out` sets the folder for the results. Images are always saved, videos only when this option is used.
- `--no-show` runs the program without opening a window.
- `--bench` followed by a number runs the detection that many times and prints the average time.

In the window, press any key to go to the next image. Press Esc or Q to quit.


## Results

These are the results of the included model, tested on 23 validation photos. Speed was measured on a CPU with 8 threads, without a graphics card. Results with your own photos and hardware may differ.

- Input size 640: 118 ms per image (about 8 frames per second), mAP50 of 0.85
- Input size 416: 53 ms per image (about 19 frames per second), mAP50 of 0.86
- Input size 320: 33 ms per image (about 30 frames per second), mAP50 of 0.79

mAP50 is a common measure of detection accuracy. A value of 1.0 would mean every chicken is found with no false detections.

The 416 model is the best choice for most uses. On this dataset it is even slightly more accurate than the 640 model and more than twice as fast. The 320 model is recommended for webcam use when a smooth picture matters most.


## Technical notes

Before detection, every image is scaled to the input size of the model without distorting it. The free space is filled with gray. This is exactly how the images were prepared during training.

The model returns 8400 possible boxes per image. The program first looks only at the confidence score of each box and only calculates the position for the few boxes above the threshold. Boxes that overlap too much are then removed.

All memory needed for detection is reserved once and reused for every frame. This keeps the program fast and stable during long video or webcam sessions.

When using a webcam, a separate thread reads the camera images. The detector always works on the newest image. If detection is slower than the camera, older images are skipped. This way the picture never falls behind. Video files are processed completely, frame by frame.

Almost all of the processing time is spent inside the neural network itself. The quickest way to make detection faster is therefore to use a smaller input size.


## Privacy

- The training scripts turn off the anonymous usage statistics that Ultralytics sends by default. This setting applies to your whole Ultralytics installation.
- `prepare_photos.py` removes all metadata from your photos, including GPS locations.
- The C++ application does not connect to the internet.
- Ultralytics stores local file paths, including your user name, inside trained models. `train.py` and `export.py` remove these automatically with `sanitize_model.py`, so model files can be shared safely. The models contain only the learned weights, not the training photos.


## Known limitations

- Only chickens as a whole are detected. Individual birds are not recognised.
- The included model was trained in a single indoor coop, mostly in low light. It may perform worse outdoors or in very different surroundings. Training it further with your own photos solves this.
- Chickens sitting closely together, for example on a perch, are sometimes detected as one or missed.
- Very dark or blurry images reduce accuracy.
- So far the project has only been tested on Windows.


## Planned improvements

- More photos, especially in daylight and outdoors
- Faster detection with ONNX Runtime
- Logging the number of chickens over time
- Support for Linux and Raspberry Pi


## Contributing

Improvements are welcome. If you have fixed a bug, made the detector faster or added a useful feature, feel free to open a pull request. For larger changes, please open an issue first so we can discuss the idea.

If you train the detector on your own chickens, I am also happy to hear how well it works in your setup.


## License

This project is licensed under the GNU Affero General Public License v3.0. The full text is in the LICENSE file.

This license was chosen because Ultralytics YOLO, which is used for training, uses the same license. The included models and all models trained with this project are based on Ultralytics models and fall under the same terms.


## Credits

- Ultralytics YOLO for the model and the training framework
- OpenCV for image processing and running the model
- labelImg for labelling the images

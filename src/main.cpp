// Copyright (C) 2026 DrDuckkkkkkkkkk
// SPDX-License-Identifier: AGPL-3.0-only
//
// Command-line front end: detects chickens in images, image folders, video files and webcam streams.

#include "Detector.h"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kWindowName = "ChickenDetector (Esc/Q to quit)";
const std::vector<std::string> kClassNames = {"Chicken"};

struct Options {
    std::string model;
    std::string source;
    DetectorConfig detector;
    std::optional<fs::path> outDir;
    bool show = true;
    int benchRuns = 0;
};

void printUsage(const char* exe) {
    std::cerr << "Usage: " << exe << " <model.onnx> <image|folder|video|camera-index> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --conf <0..1>   minimum confidence (default 0.25)\n"
              << "  --nms <0..1>    IoU threshold for non-maximum suppression (default 0.45)\n"
              << "  --size <n>      model input size, must match the export size (default 640)\n"
              << "  --out <dir>     output directory (default for images: results)\n"
              << "  --no-show       do not open a window\n"
              << "  --bench <n>     run detection n times per image and report timing\n";
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isImageFile(const fs::path& p) {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};
    return std::find(exts.begin(), exts.end(), toLower(p.extension().string())) != exts.end();
}

bool isCameraIndex(const std::string& s) {
    return !s.empty() && s.size() <= 3 &&
           std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

float parseFloat(const std::string& value, const std::string& option) {
    try {
        size_t pos = 0;
        const float f = std::stof(value, &pos);
        if (pos == value.size()) return f;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("Invalid value for " + option + ": " + value);
}

int parseInt(const std::string& value, const std::string& option) {
    try {
        size_t pos = 0;
        const int i = std::stoi(value, &pos);
        if (pos == value.size()) return i;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("Invalid value for " + option + ": " + value);
}

Options parseArgs(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("Missing arguments");

    Options o;
    o.model = argv[1];
    o.source = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("Missing value after " + arg);
            return argv[++i];
        };
        if (arg == "--conf")
            o.detector.confThreshold = parseFloat(next(), arg);
        else if (arg == "--nms")
            o.detector.nmsThreshold = parseFloat(next(), arg);
        else if (arg == "--size")
            o.detector.inputSize = parseInt(next(), arg);
        else if (arg == "--out")
            o.outDir = fs::path(next());
        else if (arg == "--no-show")
            o.show = false;
        else if (arg == "--bench")
            o.benchRuns = parseInt(next(), arg);
        else
            throw std::invalid_argument("Unknown option: " + arg);
    }
    if (o.benchRuns < 0) throw std::invalid_argument("--bench must not be negative");
    return o;
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& dets, double fps) {
    const cv::Scalar boxColor(0, 200, 255);
    char text[64];

    for (const auto& d : dets) {
        cv::rectangle(frame, d.box, boxColor, 2);

        const bool known = d.classId >= 0 && d.classId < static_cast<int>(kClassNames.size());
        std::snprintf(text, sizeof(text), "%s %.0f%%", known ? kClassNames[d.classId].c_str() : "?",
                      d.confidence * 100.f);

        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int y = d.box.y - ts.height - 6 >= 0 ? d.box.y - 4 : d.box.y + ts.height + 4;
        cv::rectangle(frame, {d.box.x, y - ts.height - 2}, {d.box.x + ts.width + 4, y + baseline}, boxColor,
                      cv::FILLED);
        cv::putText(frame, text, {d.box.x + 2, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                    cv::LINE_AA);
    }

    if (fps > 0)
        std::snprintf(text, sizeof(text), "Chickens: %zu | %.1f FPS", dets.size(), fps);
    else
        std::snprintf(text, sizeof(text), "Chickens: %zu", dets.size());
    cv::putText(frame, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, text, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool quitRequested(int delayMs) {
    const int key = cv::waitKey(delayMs);
    return key == 27 || key == 'q' || key == 'Q';
}

int runImages(Detector& detector, const Options& o, const std::vector<fs::path>& files) {
    const fs::path outDir = o.outDir.value_or("results");
    fs::create_directories(outDir);

    std::vector<Detection> dets;
    size_t total = 0;

    for (const auto& file : files) {
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Skipped (unreadable): " << file.string() << "\n";
            continue;
        }

        if (o.benchRuns > 0) {
            double sum = 0;
            double best = 1e9;
            for (int i = 0; i < o.benchRuns; ++i) {
                const auto t0 = Clock::now();
                detector.detect(img, dets);
                const double ms = elapsedMs(t0);
                sum += ms;
                best = std::min(best, ms);
            }
            std::printf("Benchmark %s: %d runs, average %.2f ms (%.1f FPS), best %.2f ms\n",
                        file.filename().string().c_str(), o.benchRuns, sum / o.benchRuns, 1000.0 * o.benchRuns / sum,
                        best);
        }

        const auto t0 = Clock::now();
        detector.detect(img, dets);
        const double ms = elapsedMs(t0);
        total += dets.size();
        std::printf("%-30s %3zu %s (%.1f ms)\n", file.filename().string().c_str(), dets.size(),
                    dets.size() == 1 ? "chicken" : "chickens", ms);

        drawDetections(img, dets, 0);
        const fs::path outPath = outDir / (file.stem().string() + "_result.jpg");
        if (!cv::imwrite(outPath.string(), img)) std::cerr << "Could not save " << outPath.string() << "\n";

        if (o.show) {
            cv::imshow(kWindowName, img);
            if (quitRequested(0)) break;
        }
    }

    std::printf("Total: %zu %s in %zu %s. Results saved to %s\n", total, total == 1 ? "chicken" : "chickens",
                files.size(), files.size() == 1 ? "image" : "images", outDir.string().c_str());
    return 0;
}

// Reads camera frames on a background thread and keeps only the newest one, so a slow detector
// never builds up latency.
class LatestFrameGrabber {
public:
    explicit LatestFrameGrabber(cv::VideoCapture& cap) : cap_(cap), thread_([this] { run(); }) {}

    ~LatestFrameGrabber() {
        running_ = false;
        cond_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    LatestFrameGrabber(const LatestFrameGrabber&) = delete;
    LatestFrameGrabber& operator=(const LatestFrameGrabber&) = delete;

    // Blocks until a new frame is available. Returns false when the stream has ended.
    bool next(cv::Mat& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return hasNew_ || !running_; });
        if (!hasNew_) return false;
        cv::swap(out, latest_);
        hasNew_ = false;
        return true;
    }

private:
    void run() {
        cv::Mat frame;
        while (running_) {
            if (!cap_.read(frame) || frame.empty()) {
                running_ = false;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                cv::swap(frame, latest_);
                hasNew_ = true;
            }
            cond_.notify_one();
        }
        cond_.notify_all();
    }

    cv::VideoCapture& cap_;
    std::mutex mutex_;
    std::condition_variable cond_;
    cv::Mat latest_;
    bool hasNew_ = false;
    std::atomic<bool> running_{true};
    std::thread thread_;  // declared last so all other members exist when the thread starts
};

int runStream(Detector& detector, const Options& o) {
    const bool camera = isCameraIndex(o.source);

    cv::VideoCapture cap;
    if (camera) {
        cap.open(std::stoi(o.source), cv::CAP_ANY);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    } else {
        cap.open(o.source);
    }
    if (!cap.isOpened()) throw std::runtime_error("Could not open source: " + o.source);

    cv::VideoWriter writer;
    std::vector<Detection> dets;
    cv::Mat frame;
    double fps = 0;
    size_t frames = 0;
    double totalMs = 0;

    auto process = [&](cv::Mat& f) -> bool {
        const auto t0 = Clock::now();
        detector.detect(f, dets);
        const double ms = elapsedMs(t0);
        totalMs += ms;
        ++frames;
        fps = fps == 0 ? 1000.0 / ms : 0.9 * fps + 0.1 * (1000.0 / ms);
        drawDetections(f, dets, fps);

        if (o.outDir) {
            if (!writer.isOpened()) {
                fs::create_directories(*o.outDir);
                const std::string name = camera ? "webcam_result.mp4" : fs::path(o.source).stem().string() + "_result.mp4";
                const fs::path outPath = *o.outDir / name;
                double srcFps = cap.get(cv::CAP_PROP_FPS);
                if (srcFps <= 0 || srcFps > 240) srcFps = 30;
                writer.open(outPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), srcFps, f.size());
                if (!writer.isOpened()) throw std::runtime_error("Could not write video: " + outPath.string());
                std::cout << "Saving to " << outPath.string() << "\n";
            }
            writer.write(f);
        }

        if (o.show) {
            cv::imshow(kWindowName, f);
            if (quitRequested(1)) return false;
        }
        return true;
    };

    if (camera) {
        LatestFrameGrabber grabber(cap);
        while (grabber.next(frame))
            if (!process(frame)) break;
    } else {
        // Video files: process every frame, nothing is dropped.
        while (cap.read(frame))
            if (!process(frame)) break;
    }

    if (frames > 0)
        std::printf("%zu frames, average detection time %.2f ms (%.1f FPS)\n", frames, totalMs / frames,
                    1000.0 * frames / totalMs);
    return 0;
}

std::vector<fs::path> collectImages(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir))
        if (entry.is_regular_file() && isImageFile(entry.path())) files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    return files;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parseArgs(argc, argv);
        if (!fs::is_regular_file(o.model)) throw std::runtime_error("Model file not found: " + o.model);

        cv::setNumThreads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        Detector detector(o.model, o.detector);

        int rc = 0;
        const fs::path source(o.source);
        if (fs::is_directory(source)) {
            const auto files = collectImages(source);
            if (files.empty()) throw std::runtime_error("No images found in " + o.source);
            rc = runImages(detector, o, files);
        } else if (isImageFile(source)) {
            if (!fs::is_regular_file(source)) throw std::runtime_error("Image not found: " + o.source);
            rc = runImages(detector, o, {source});
        } else {
            rc = runStream(detector, o);
        }

        if (o.show) cv::destroyAllWindows();
        return rc;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        printUsage(argc > 0 ? argv[0] : "ChickenDetector");
        return 2;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

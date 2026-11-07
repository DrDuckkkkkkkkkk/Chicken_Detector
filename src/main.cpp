// ChickenDetector: findet Huehner in Bildern, Ordnern, Videos oder im Webcam-Bild.
//
//   ChickenDetector <modell.onnx> <quelle> [optionen]
//
//   <quelle>  Bilddatei | Ordner mit Bildern | Videodatei | Webcam-Nummer (z.B. 0)
//
// Optionen siehe printUsage() bzw. README.md.

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

constexpr const char* kWindow = "ChickenDetector (ESC/Q = Ende)";
const std::vector<std::string> kClassNames = {"Huhn"};

struct Options {
    std::string model;
    std::string source;
    DetectorConfig det;
    std::optional<fs::path> outDir;  // nur speichern, wenn angegeben (bzw. Standard bei Einzelbildern)
    bool show = true;
    int benchRuns = 0;
};

void printUsage(const char* exe) {
    std::cerr << "Aufruf: " << exe << " <modell.onnx> <bild|ordner|video|webcam-nummer> [optionen]\n"
              << "  --conf <0..1>   minimale Sicherheit           (Standard 0.25)\n"
              << "  --nms  <0..1>   NMS-IoU-Schwelle              (Standard 0.45)\n"
              << "  --size <n>      Eingabegroesse des Modells    (Standard 640, muss zum Export passen)\n"
              << "  --out  <ordner> Ergebnisse speichern          (Standard bei Bildern: results)\n"
              << "  --no-show       kein Fenster anzeigen\n"
              << "  --bench <n>     Bild n-mal erkennen und Zeit messen\n";
}

std::string lower(std::string s) {
    // unsigned char-Cast: std::tolower mit negativen char-Werten (Umlaute) waere undefiniertes Verhalten
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool isImageFile(const fs::path& p) {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"};
    const std::string ext = lower(p.extension().string());
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

bool isCameraIndex(const std::string& s) {
    return !s.empty() && s.size() <= 3 &&
           std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

float parseFloat(const std::string& v, const char* name) {
    try {
        size_t pos = 0;
        const float f = std::stof(v, &pos);
        if (pos == v.size()) return f;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument(std::string("Ungueltiger Wert fuer ") + name + ": " + v);
}

int parseInt(const std::string& v, const char* name) {
    try {
        size_t pos = 0;
        const int i = std::stoi(v, &pos);
        if (pos == v.size()) return i;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument(std::string("Ungueltiger Wert fuer ") + name + ": " + v);
}

Options parseArgs(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("Zu wenige Argumente");
    Options o;
    o.model = argv[1];
    o.source = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("Wert fehlt nach " + a);
            return argv[++i];
        };
        if (a == "--conf") o.det.confThreshold = parseFloat(next(), "--conf");
        else if (a == "--nms") o.det.nmsThreshold = parseFloat(next(), "--nms");
        else if (a == "--size") o.det.inputSize = parseInt(next(), "--size");
        else if (a == "--out") o.outDir = fs::path(next());
        else if (a == "--no-show") o.show = false;
        else if (a == "--bench") o.benchRuns = parseInt(next(), "--bench");
        else throw std::invalid_argument("Unbekannte Option: " + a);
    }
    if (o.benchRuns < 0) throw std::invalid_argument("--bench muss >= 0 sein");
    return o;
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& dets, double fps) {
    const cv::Scalar boxColor(0, 200, 255);
    char buf[64];
    for (const auto& d : dets) {
        cv::rectangle(frame, d.box, boxColor, 2);
        const std::string& name =
            d.classId >= 0 && d.classId < static_cast<int>(kClassNames.size()) ? kClassNames[d.classId] : "?";
        std::snprintf(buf, sizeof(buf), "%s %.0f%%", name.c_str(), d.confidence * 100.f);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        // Label oberhalb der Box, bei Boxen am oberen Rand innerhalb
        const int y = d.box.y - ts.height - 6 >= 0 ? d.box.y - 4 : d.box.y + ts.height + 4;
        cv::rectangle(frame, {d.box.x, y - ts.height - 2}, {d.box.x + ts.width + 4, y + baseline}, boxColor,
                      cv::FILLED);
        cv::putText(frame, buf, {d.box.x + 2, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                    cv::LINE_AA);
    }
    if (fps > 0)
        std::snprintf(buf, sizeof(buf), "Huehner: %zu  |  %.1f FPS", dets.size(), fps);
    else
        std::snprintf(buf, sizeof(buf), "Huehner: %zu", dets.size());
    cv::putText(frame, buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(frame, buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

bool userQuit(int delayMs) {
    const int k = cv::waitKey(delayMs);
    return k == 27 || k == 'q' || k == 'Q';
}

// ---------------------------------------------------------------------------------------------
// Bilder (einzeln oder ganzer Ordner)
// ---------------------------------------------------------------------------------------------
int runImages(Detector& det, const Options& o, const std::vector<fs::path>& files) {
    const fs::path outDir = o.outDir.value_or("results");
    fs::create_directories(outDir);

    std::vector<Detection> dets;
    size_t total = 0;
    for (const auto& file : files) {
        cv::Mat img = cv::imread(file.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Uebersprungen (nicht lesbar): " << file.string() << "\n";
            continue;
        }

        if (o.benchRuns > 0) {
            double sum = 0, best = 1e9;
            for (int i = 0; i < o.benchRuns; ++i) {
                const auto t0 = Clock::now();
                det.detect(img, dets);
                const double ms = msSince(t0);
                sum += ms;
                best = std::min(best, ms);
            }
            std::printf("Benchmark %s: %d Laeufe, Schnitt %.2f ms (%.1f FPS), bester %.2f ms\n",
                        file.filename().string().c_str(), o.benchRuns, sum / o.benchRuns,
                        1000.0 * o.benchRuns / sum, best);
        }

        const auto t0 = Clock::now();
        det.detect(img, dets);
        const double ms = msSince(t0);
        total += dets.size();
        std::printf("%-30s %3zu Huehner  (%.1f ms)\n", file.filename().string().c_str(), dets.size(), ms);

        drawDetections(img, dets, 0);
        const fs::path outPath = outDir / (file.stem().string() + "_result.jpg");
        if (!cv::imwrite(outPath.string(), img)) std::cerr << "Speichern fehlgeschlagen: " << outPath.string() << "\n";

        if (o.show) {
            cv::imshow(kWindow, img);
            // Einzelbild: warten bis Taste; Ordner: Taste = naechstes Bild, ESC/Q = Ende
            if (userQuit(0)) break;
        }
    }
    std::printf("Gesamt: %zu Huehner in %zu Bild(ern). Ergebnisse in %s\n", total, files.size(),
                outDir.string().c_str());
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Webcam: ein eigener Thread liest staendig Frames, die Erkennung nimmt immer den NEUESTEN.
// So entsteht keine Verzoegerung (Lag), auch wenn die Erkennung langsamer als die Kamera ist.
// ---------------------------------------------------------------------------------------------
class LatestFrameGrabber {
public:
    explicit LatestFrameGrabber(cv::VideoCapture& cap) : cap_(cap), thread_([this] { loop(); }) {}
    ~LatestFrameGrabber() {
        running_ = false;
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();  // RAII: Thread wird immer sauber beendet
    }
    LatestFrameGrabber(const LatestFrameGrabber&) = delete;
    LatestFrameGrabber& operator=(const LatestFrameGrabber&) = delete;

    // Wartet auf einen neuen Frame und tauscht ihn in `out` (kein Kopieren der Pixel).
    bool next(cv::Mat& out) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return fresh_ || !running_; });
        if (!fresh_) return false;
        cv::swap(out, latest_);
        fresh_ = false;
        return true;
    }

private:
    void loop() {
        cv::Mat frame;
        while (running_) {
            if (!cap_.read(frame) || frame.empty()) {
                running_ = false;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(mtx_);
                cv::swap(frame, latest_);  // alter Frame wird beim naechsten read() ueberschrieben
                fresh_ = true;
            }
            cv_.notify_one();
        }
        cv_.notify_all();
    }

    cv::VideoCapture& cap_;
    std::mutex mtx_;
    std::condition_variable cv_;
    cv::Mat latest_;
    bool fresh_ = false;
    std::atomic<bool> running_{true};
    std::thread thread_;  // muss als letztes Member initialisiert werden
};

int runStream(Detector& det, const Options& o) {
    const bool camera = isCameraIndex(o.source);
    cv::VideoCapture cap;
    if (camera) {
        cap.open(std::stoi(o.source), cv::CAP_ANY);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // moeglichst wenig Puffer -> geringe Latenz
    } else {
        cap.open(o.source);
    }
    if (!cap.isOpened()) throw std::runtime_error("Quelle konnte nicht geoeffnet werden: " + o.source);

    cv::VideoWriter writer;
    std::vector<Detection> dets;
    cv::Mat frame;
    double fps = 0;
    size_t frames = 0;
    double totalMs = 0;

    auto process = [&](cv::Mat& f) -> bool {
        const auto t0 = Clock::now();
        det.detect(f, dets);
        const double ms = msSince(t0);
        totalMs += ms;
        ++frames;
        fps = fps == 0 ? 1000.0 / ms : 0.9 * fps + 0.1 * (1000.0 / ms);  // geglaettet
        drawDetections(f, dets, fps);

        if (o.outDir) {
            if (!writer.isOpened()) {
                fs::create_directories(*o.outDir);
                const fs::path outPath = *o.outDir / (camera ? "webcam_result.mp4"
                                                              : fs::path(o.source).stem().string() + "_result.mp4");
                double srcFps = cap.get(cv::CAP_PROP_FPS);
                if (srcFps <= 0 || srcFps > 240) srcFps = 30;
                writer.open(outPath.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), srcFps, f.size());
                if (!writer.isOpened()) throw std::runtime_error("Video konnte nicht geschrieben werden: " + outPath.string());
                std::cout << "Speichere nach " << outPath.string() << "\n";
            }
            writer.write(f);
        }
        if (o.show) {
            cv::imshow(kWindow, f);
            if (userQuit(1)) return false;
        }
        return true;
    };

    if (camera) {
        LatestFrameGrabber grabber(cap);
        while (grabber.next(frame))
            if (!process(frame)) break;
    } else {
        while (cap.read(frame))  // Videodatei: jeden Frame verarbeiten, nichts ueberspringen
            if (!process(frame)) break;
    }

    if (frames > 0)
        std::printf("%zu Frames, Erkennung im Schnitt %.2f ms (%.1f FPS)\n", frames, totalMs / frames,
                    1000.0 * frames / totalMs);
    return 0;
}

std::vector<fs::path> collectImages(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && isImageFile(e.path())) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    return files;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parseArgs(argc, argv);
        if (!fs::is_regular_file(o.model)) throw std::runtime_error("Modelldatei nicht gefunden: " + o.model);

        cv::setNumThreads(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        Detector det(o.model, o.det);

        int rc;
        const fs::path src(o.source);
        if (fs::is_directory(src)) {
            const auto files = collectImages(src);
            if (files.empty()) throw std::runtime_error("Keine Bilder im Ordner: " + o.source);
            rc = runImages(det, o, files);
        } else if (isImageFile(src)) {
            if (!fs::is_regular_file(src)) throw std::runtime_error("Bild nicht gefunden: " + o.source);
            rc = runImages(det, o, {src});
        } else {
            rc = runStream(det, o);
        }
        if (o.show) cv::destroyAllWindows();
        return rc;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Fehler: " << e.what() << "\n\n";
        printUsage(argc > 0 ? argv[0] : "ChickenDetector");
        return 2;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV-Fehler: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fehler: " << e.what() << "\n";
        return 1;
    }
}

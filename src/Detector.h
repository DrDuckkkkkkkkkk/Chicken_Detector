#pragma once
// Detector: laedt ein YOLO-Modell (ONNX, exportiert mit Ultralytics) ueber OpenCV DNN
// und findet Objekte (hier: Huehner) in BGR-Bildern.
//
// Performance-Entscheidungen:
//  - Alle Zwischenpuffer (Letterbox-Bild, Blob, Kandidatenlisten) sind Member und werden
//    zwischen Frames wiederverwendet -> keine Heap-Allokationen pro Frame im Normalfall.
//  - Die Netzausgabe wird ueber rohe Zeiger gelesen. Pro Vorschlag wird zuerst nur der
//    Score geprueft; die Box wird nur fuer die wenigen Vorschlaege ueber der Schwelle berechnet.
//  - Ein Warmup-Durchlauf im Konstruktor sorgt dafuer, dass der erste echte Frame nicht langsam ist.
//
// Speicher: ausschliesslich RAII (cv::Mat, std::vector, cv::dnn::Net) -> keine manuellen
// new/delete, keine Leaks. Eine Instanz ist NICHT thread-safe (ein Detector pro Thread).

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;      // Rahmen im Originalbild (Pixel)
    float confidence;  // 0..1
    int classId;
};

struct DetectorConfig {
    int inputSize = 640;          // muss zur Export-Groesse des ONNX-Modells passen
    float confThreshold = 0.25f;  // minimale Sicherheit
    float nmsThreshold = 0.45f;   // IoU-Schwelle fuer Non-Maximum-Suppression
    int maxDetections = 300;      // Obergrenze pro Bild
};

class Detector {
public:
    explicit Detector(const std::string& onnxPath, DetectorConfig cfg = {}) : cfg_(cfg) {
        if (cfg_.inputSize <= 0 || cfg_.inputSize % 32 != 0)
            throw std::invalid_argument("inputSize muss ein positives Vielfaches von 32 sein");
        if (cfg_.confThreshold < 0.f || cfg_.confThreshold > 1.f || cfg_.nmsThreshold < 0.f ||
            cfg_.nmsThreshold > 1.f)
            throw std::invalid_argument("Schwellwerte muessen zwischen 0 und 1 liegen");

        net_ = cv::dnn::readNetFromONNX(onnxPath);  // wirft cv::Exception bei defekter Datei
        if (net_.empty()) throw std::runtime_error("Modell konnte nicht geladen werden: " + onnxPath);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        letterbox_.create(cfg_.inputSize, cfg_.inputSize, CV_8UC3);
        warmup();
    }

    // Nicht kopierbar (das Netz ist teuer), aber verschiebbar.
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) = default;
    Detector& operator=(Detector&&) = default;

    const DetectorConfig& config() const { return cfg_; }

    // Fuehrt die Erkennung aus. Das Ergebnis wird in `out` geschrieben (wird vorher geleert),
    // damit der Aufrufer den Vektor zwischen Frames wiederverwenden kann.
    void detect(const cv::Mat& image, std::vector<Detection>& out) {
        out.clear();
        if (image.empty()) return;

        const cv::Mat& bgr = toBgr(image);
        const LetterboxInfo lb = makeLetterbox(bgr);

        // swapRB=true: OpenCV liefert BGR, YOLO erwartet RGB. crop=false: Groesse passt bereits.
        cv::dnn::blobFromImage(letterbox_, blob_, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false,
                               CV_32F);
        net_.setInput(blob_);
        net_.forward(outputs_);  // wiederverwendeter Ausgabepuffer
        decode(outputs_.front(), lb, bgr.size(), out);
    }

    std::vector<Detection> detect(const cv::Mat& image) {
        std::vector<Detection> out;
        detect(image, out);
        return out;
    }

private:
    struct LetterboxInfo {
        float scale;
        int padX, padY;
    };

    // Graustufen/BGRA -> BGR. Bei BGR wird nur eine Referenz zurueckgegeben (keine Kopie).
    const cv::Mat& toBgr(const cv::Mat& img) {
        if (img.type() == CV_8UC3) return img;
        if (img.depth() != CV_8U) throw std::invalid_argument("Nur 8-Bit-Bilder werden unterstuetzt");
        const int code = img.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR;
        cv::cvtColor(img, converted_, code);
        return converted_;
    }

    // Bild proportional auf inputSize x inputSize skalieren, Rand grau (114) auffuellen.
    // Schreibt direkt in den wiederverwendeten Puffer letterbox_.
    LetterboxInfo makeLetterbox(const cv::Mat& img) {
        const int s = cfg_.inputSize;
        const float scale = std::min(static_cast<float>(s) / img.cols, static_cast<float>(s) / img.rows);
        const int newW = std::max(1, static_cast<int>(std::lround(img.cols * scale)));
        const int newH = std::max(1, static_cast<int>(std::lround(img.rows * scale)));
        const int padX = (s - newW) / 2;
        const int padY = (s - newH) / 2;

        // Nur den Rand fuellen statt des ganzen Bildes
        const cv::Scalar gray(114, 114, 114);
        if (padY > 0) {
            letterbox_.rowRange(0, padY).setTo(gray);
            letterbox_.rowRange(padY + newH, s).setTo(gray);
        }
        if (padX > 0) {
            letterbox_.colRange(0, padX).setTo(gray);
            letterbox_.colRange(padX + newW, s).setTo(gray);
        }
        cv::Mat roi = letterbox_(cv::Rect(padX, padY, newW, newH));
        if (newW == img.cols && newH == img.rows)
            img.copyTo(roi);
        else
            cv::resize(img, roi, roi.size(), 0, 0, scale < 1.f ? cv::INTER_AREA : cv::INTER_LINEAR);
        return {scale, padX, padY};
    }

    // YOLOv8/11-Ausgabe: [1, 4 + Klassen, N] (Standard) oder transponiert [1, N, 4 + Klassen].
    void decode(const cv::Mat& raw, const LetterboxInfo& lb, cv::Size imgSize, std::vector<Detection>& out) {
        if (raw.dims != 3 || raw.size[0] != 1 || raw.type() != CV_32F)
            throw std::runtime_error("Unerwartete Modellausgabe (kein YOLOv8/11-Detect-Modell?)");

        // Die Achse mit den wenigen Eintraegen (4 + Klassen) ist die Attribut-Achse.
        const bool transposed = raw.size[1] > raw.size[2];
        const int numAttrs = transposed ? raw.size[2] : raw.size[1];
        const int numPreds = transposed ? raw.size[1] : raw.size[2];
        const int numClasses = numAttrs - 4;
        if (numClasses < 1) throw std::runtime_error("Modellausgabe hat keine Klassen-Scores");

        const float* data = raw.ptr<float>();
        // Zugriff auf Attribut a von Vorschlag i ohne Kopie/Transponieren
        const size_t strideAttr = transposed ? 1 : static_cast<size_t>(numPreds);
        const size_t stridePred = transposed ? static_cast<size_t>(numAttrs) : 1;
        auto at = [&](int a, int i) { return data[a * strideAttr + i * stridePred]; };

        boxes_.clear();
        scores_.clear();
        classIds_.clear();

        const float invScale = 1.f / lb.scale;
        const cv::Rect imgRect(0, 0, imgSize.width, imgSize.height);

        for (int i = 0; i < numPreds; ++i) {
            // Schneller Pfad: erst nur die Scores pruefen
            float best = at(4, i);
            int bestCls = 0;
            for (int c = 1; c < numClasses; ++c) {
                const float v = at(4 + c, i);
                if (v > best) {
                    best = v;
                    bestCls = c;
                }
            }
            if (best < cfg_.confThreshold) continue;

            // Koordinaten aus dem Letterbox-Raum ins Originalbild zurueckrechnen
            const float w = at(2, i) * invScale;
            const float h = at(3, i) * invScale;
            const float x = (at(0, i) - lb.padX) * invScale - 0.5f * w;
            const float y = (at(1, i) - lb.padY) * invScale - 0.5f * h;
            const cv::Rect r = cv::Rect(cvRound(x), cvRound(y), cvRound(w), cvRound(h)) & imgRect;
            if (r.area() <= 0) continue;

            boxes_.push_back(r);
            scores_.push_back(best);
            classIds_.push_back(bestCls);
        }

        // Klassenweise NMS (bei nur einer Klasse identisch zu normaler NMS)
        if (numClasses == 1)
            cv::dnn::NMSBoxes(boxes_, scores_, cfg_.confThreshold, cfg_.nmsThreshold, keep_, 1.f,
                              cfg_.maxDetections);
        else
            cv::dnn::NMSBoxesBatched(boxes_, scores_, classIds_, cfg_.confThreshold, cfg_.nmsThreshold,
                                     keep_, 1.f, cfg_.maxDetections);

        out.reserve(keep_.size());
        for (int idx : keep_) out.push_back({boxes_[idx], scores_[idx], classIds_[idx]});
    }

    void warmup() {
        std::vector<Detection> tmp;
        try {
            detect(cv::Mat(cfg_.inputSize, cfg_.inputSize, CV_8UC3, cv::Scalar(114, 114, 114)), tmp);
        } catch (const cv::Exception&) {
            throw std::runtime_error("Modell laesst sich mit Eingabegroesse " + std::to_string(cfg_.inputSize) +
                                     " nicht ausfuehren. Passt --size zur Export-Groesse des Modells?");
        }
    }

    DetectorConfig cfg_;
    cv::dnn::Net net_;

    // Wiederverwendete Puffer
    cv::Mat letterbox_, blob_, converted_;
    std::vector<cv::Mat> outputs_;
    std::vector<cv::Rect> boxes_;
    std::vector<float> scores_;
    std::vector<int> classIds_;
    std::vector<int> keep_;
};

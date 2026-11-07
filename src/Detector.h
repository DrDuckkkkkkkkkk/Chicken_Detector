// Copyright (C) 2026 DrDuckkkkkkkkkk
// SPDX-License-Identifier: AGPL-3.0-only
//
// YOLO11 object detector based on OpenCV DNN.
// Expects an ONNX model exported with Ultralytics (opset 12, static input size).

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

struct DetectorConfig {
    int inputSize = 640;  // must match the export size of the model
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    int maxDetections = 300;
};

// Not thread-safe: use one instance per thread.
class Detector {
public:
    explicit Detector(const std::string& onnxPath, DetectorConfig cfg = {}) : cfg_(cfg) {
        if (cfg_.inputSize <= 0 || cfg_.inputSize % 32 != 0)
            throw std::invalid_argument("Input size must be a positive multiple of 32");
        if (cfg_.confThreshold < 0.f || cfg_.confThreshold > 1.f || cfg_.nmsThreshold < 0.f ||
            cfg_.nmsThreshold > 1.f)
            throw std::invalid_argument("Thresholds must be between 0 and 1");

        net_ = cv::dnn::readNetFromONNX(onnxPath);
        if (net_.empty()) throw std::runtime_error("Could not load model: " + onnxPath);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        letterbox_.create(cfg_.inputSize, cfg_.inputSize, CV_8UC3);
        warmup();
    }

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) = default;
    Detector& operator=(Detector&&) = default;

    const DetectorConfig& config() const { return cfg_; }

    // Writes the detections into `out`. Reusing the same vector across frames avoids allocations.
    void detect(const cv::Mat& image, std::vector<Detection>& out) {
        out.clear();
        if (image.empty()) return;

        const cv::Mat& bgr = toBgr(image);
        const Letterbox lb = letterbox(bgr);

        cv::dnn::blobFromImage(letterbox_, blob_, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
        net_.setInput(blob_);
        net_.forward(outputs_);
        decode(outputs_.front(), lb, bgr.size(), out);
    }

    std::vector<Detection> detect(const cv::Mat& image) {
        std::vector<Detection> out;
        detect(image, out);
        return out;
    }

private:
    struct Letterbox {
        float scale;
        int padX;
        int padY;
    };

    const cv::Mat& toBgr(const cv::Mat& img) {
        if (img.type() == CV_8UC3) return img;
        if (img.depth() != CV_8U) throw std::invalid_argument("Only 8-bit images are supported");
        cv::cvtColor(img, converted_, img.channels() == 1 ? cv::COLOR_GRAY2BGR : cv::COLOR_BGRA2BGR);
        return converted_;
    }

    // Resize while keeping the aspect ratio and pad with gray, as done by Ultralytics during training.
    Letterbox letterbox(const cv::Mat& img) {
        const int s = cfg_.inputSize;
        const float scale = std::min(static_cast<float>(s) / img.cols, static_cast<float>(s) / img.rows);
        const int newW = std::max(1, static_cast<int>(std::lround(img.cols * scale)));
        const int newH = std::max(1, static_cast<int>(std::lround(img.rows * scale)));
        const int padX = (s - newW) / 2;
        const int padY = (s - newH) / 2;

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

    // Output layout is [1, 4 + classes, N]. The transposed layout [1, N, 4 + classes] is supported as well.
    void decode(const cv::Mat& raw, const Letterbox& lb, cv::Size imgSize, std::vector<Detection>& out) {
        if (raw.dims != 3 || raw.size[0] != 1 || raw.type() != CV_32F)
            throw std::runtime_error("Unexpected model output, expected a YOLOv8/YOLO11 detection model");

        const bool transposed = raw.size[1] > raw.size[2];
        const int numAttrs = transposed ? raw.size[2] : raw.size[1];
        const int numPreds = transposed ? raw.size[1] : raw.size[2];
        const int numClasses = numAttrs - 4;
        if (numClasses < 1) throw std::runtime_error("Model output contains no class scores");

        const float* data = raw.ptr<float>();
        const size_t strideAttr = transposed ? 1 : static_cast<size_t>(numPreds);
        const size_t stridePred = transposed ? static_cast<size_t>(numAttrs) : 1;
        auto at = [&](int attr, int pred) { return data[attr * strideAttr + pred * stridePred]; };

        boxes_.clear();
        scores_.clear();
        classIds_.clear();

        const float invScale = 1.f / lb.scale;
        const cv::Rect imgRect(0, 0, imgSize.width, imgSize.height);

        for (int i = 0; i < numPreds; ++i) {
            // Check the score first and skip the box math for the vast majority of candidates.
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

        if (numClasses == 1)
            cv::dnn::NMSBoxes(boxes_, scores_, cfg_.confThreshold, cfg_.nmsThreshold, keep_, 1.f, cfg_.maxDetections);
        else
            cv::dnn::NMSBoxesBatched(boxes_, scores_, classIds_, cfg_.confThreshold, cfg_.nmsThreshold, keep_, 1.f,
                                     cfg_.maxDetections);

        out.reserve(keep_.size());
        for (int idx : keep_) out.push_back({boxes_[idx], scores_[idx], classIds_[idx]});
    }

    // The first inference in OpenCV DNN is much slower than the following ones.
    void warmup() {
        std::vector<Detection> tmp;
        try {
            detect(cv::Mat(cfg_.inputSize, cfg_.inputSize, CV_8UC3, cv::Scalar(114, 114, 114)), tmp);
        } catch (const cv::Exception&) {
            throw std::runtime_error("Model cannot run with input size " + std::to_string(cfg_.inputSize) +
                                     ". Does --size match the export size of the model?");
        }
    }

    DetectorConfig cfg_;
    cv::dnn::Net net_;

    // Buffers reused across frames
    cv::Mat letterbox_;
    cv::Mat blob_;
    cv::Mat converted_;
    std::vector<cv::Mat> outputs_;
    std::vector<cv::Rect> boxes_;
    std::vector<float> scores_;
    std::vector<int> classIds_;
    std::vector<int> keep_;
};

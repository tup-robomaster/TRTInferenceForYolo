#pragma once

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <dirent.h>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "preprocess.h"
#include "./macros.h"

#include "NvOnnxParser.h"
using namespace nvonnxparser;

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
#define MAX_OUTPUT_BBOX_COUNT 1000

using namespace nvinfer1;

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };

    static constexpr int LOCATIONS = 8;
    struct alignas(float) Detection
    {
        float bbox[LOCATIONS];
        float conf; // bbox_conf * cls_conf
        float class_id;
        float color;
    };
}

cv::Rect get_rect(cv::Mat &img, float bbox[4], int input_H, int input_W);

float iou(float lbox[4], float rbox[4]);

bool cmp(const Yolo::Detection &a, const Yolo::Detection &b);

void nms(std::vector<Yolo::Detection> &res, float *output, float conf_thresh, float nms_thresh = 0.5);

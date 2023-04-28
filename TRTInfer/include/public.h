#pragma once

#include <opencv2/opencv.hpp>

#include <Eigen/Core>

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

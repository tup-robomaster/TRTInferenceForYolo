# TRTInferenceForYoloX

为沈阳航空航天大学TUP机器人实验室设计的Yolox Tensorrt高性能推理加速模块

# Author: INIF-FISH iratecat1@outlook.com

# 环境

Eigen3 3.3.7

Cuda 11.8

cudnn 8.9.0

OpenCV 4.6.0

Tensorrt 8.5.3

# 使用



```cpp
#include "TRTInferenceForYoloX/TRTInfer/include/Inference.h"
...
TRTInferV1::TRTInfer myInfer(const int device);
nvinfer1::IHostMemory *data = myInfer.createEngine(const std::stringonnx_path, unsigned int maxBatchSize, int input_h, int input_w); //[Optional]
myInfer.saveEngineFile(IHostMemory* data, const std::string engine_file_path); //[Optional]
myInfer.initMoudle(const std::string engine_file_path, const int batch_size, const int num_apex, const int num_classes, const int num_colors, const int topK);
std::vector<cv::Mat> frames;
...
while(...)
{
    ...
    std::vector<std::vector<TRTInferV1::DetectObject>> result = myInfer.doInference(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold);
    std::vector<std::vector<TRTInferV1::DetectObject>> result = myInfer.doInferenceLimitFPS(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold, const int limited_fps); //[Optional]
    ...
}
...
myInfer.unInitMoudle(); //[Optional]
```

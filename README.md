# TRTInferenceForYolov5

为沈阳航空航天大学TUP机器人实验室设计的Yolov5 Tensorrt高性能推理加速模块

# Author: INIF-FISH iratecat1@outlook.com

# 环境

[Recommend]

Eigen3 3.3.7

Cuda 11.8

cudnn 8.9.0

OpenCV 4.6.0

Tensorrt 8.5.3

# 使用

```cpp
#include "TRTInferenceForYolov5/TRTInfer/include/Inference.h"
...
TRTInferV1::TRTInfer myInfer(const int device);
nvinfer1::IHostMemory *data = myInfer.createEngine(const std::string onnx_path, unsigned int maxBatchSize, int input_h, int input_w); //[Optional]
myInfer.saveEngineFile(IHostMemory* data, const std::string engine_file_path); //[Optional]
myInfer.initModule(const std::string engine_file_path, const int batch_size, const int num_apex, const int num_classes, const int num_colors, const int topK);
std::vector<cv::Mat> frames;
myInfer.calculate_inter_frame_compensation(const int limited_fps); //[Optional]
...
while(...)
{
    ...
    std::vector<std::vector<TRTInferV1::DetectObject>> result = myInfer.doInference(std::vector<cv::Mat> &frames, float obj_threshold,float confidence_threshold, float nms_threshold);
    std::vector<std::vector<TRTInferV1::DetectObject>> result = myInfer.doInferenceLimitFPS(std::vector<cv::Mat> &frames, float obj_threshold, float confidence_threshold, float nms_threshold, const int limited_fps); //[Optional]
    ...
}
...
myInfer.unInitModule(); //[Optional]
```

注意：应根据sample中CMakeLists.txt作适当修改以适配不同环境/设备，特别注意显卡架构代码

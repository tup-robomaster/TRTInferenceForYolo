#pragma once
#include "public.h"

class TRTInfer
{
private:
    const char *INPUT_BLOB_NAME = "input_0";
    const char *OUTPUT_BLOB_NAME = "output_0";

private:
    Logger gLogger;
    IRuntime *runtime;
    IExecutionContext *context;
    void *buffers[2];
    int output_size = 1;
    int inputIndex;
    int outputIndex;
    Dims input_dims;
    Dims out_dims;
    uint8_t *img_host = nullptr;
    uint8_t *img_device = nullptr;
    float *output;

public:
    TRTInfer(const int device);
    ~TRTInfer();

    bool initMoudle(const std::string engine_file_path, const int max_batch, const int img_h, const int img_w);
    void unInitMoudle();

    std::vector<std::vector<Yolo::Detection>> doInference(std::shared_ptr<std::vector<cv::Mat>> frames, int batchSize, float confidence_threshold, float nms_threshold);
    ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config);
};
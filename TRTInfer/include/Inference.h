#pragma once
#include "public.h"

namespace TRTInferV1
{
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };

    struct Object
    {
        cv::Rect_<float> rect;
        int cls;
        int color;
        float prob;
        std::vector<cv::Point2f> pts;
    };

    struct ArmorObject : Object
    {
        int area;
        cv::Point2f apex[4];
    };

    class TRTInfer
    {
    private:
        const char *INPUT_BLOB_NAME = "images";
        const char *OUTPUT_BLOB_NAME = "output";

    private:
        Logger gLogger;
        IRuntime *runtime;
        ICudaEngine *engine;
        IExecutionContext *context;
        void *buffers[2];
        int output_size = 1;
        int inputIndex = -1;
        int outputIndex = -1;
        Dims input_dims;
        Dims out_dims;
        uint8_t *img_host = nullptr;
        uint8_t *img_device = nullptr;
        float *output;

    private:
        void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
        void decodeOutputs(const float *prob, std::vector<ArmorObject> &objects, Eigen::Matrix<float, 3, 3> &transform_matrix, float confidence_threshold, float nms_threshold);

    public:
        TRTInfer(const int device);
        ~TRTInfer();

        bool initMoudle(const std::string engine_file_path, const int max_batch, const int img_h, const int img_w);
        void unInitMoudle();

        void saveEngineFile(IHostMemory *data, const std::string engine_file_path);

        std::vector<std::vector<ArmorObject>> doInference(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold);
        IHostMemory *createEngine(const std::string onnx_path, unsigned int maxBatchSize);
    };
}

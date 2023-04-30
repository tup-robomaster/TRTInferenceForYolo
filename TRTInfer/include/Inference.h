#ifndef __INFERENCE_H
#define __INFERENCE_H

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

    /**
     * @brief TRT推理
     * 高性能TRT YOLOX推理模块
     */
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
        Dims output_dims;
        uint8_t *img_host = nullptr;
        uint8_t *img_device = nullptr;
        float *output;

    private:
        void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
        void decodeOutputs(const float *prob, std::vector<ArmorObject> &objects, Eigen::Matrix<float, 3, 3> &transform_matrix, float confidence_threshold, float nms_threshold);

    public:
        /**
         * @brief 构造函数
         * @param device
         * 使用的GPU索引
         */
        TRTInfer(const int device);
        ~TRTInfer();

        /**
         * @brief 初始化TRT模型
         * @param engine_file_path
         * engine路径
         * @param batch_size
         * 推理时使用的batch_size,输入图片数量不可大于此设置值，此设定值不可大于构建引擎时应用的maxBatchSize，最佳设定值为maxBatchSize/2
         */
        bool initMoudle(const std::string engine_file_path, const int batch_size);
        /**
         * @brief 反初始化TRT模型，释放显存
         */
        void unInitMoudle();
        /**
         * @brief 保存engine文件至指定路径
         * @param data
         * 由 createEngine() 构建的序列化模型
         * @param engine_file_path
         * engine文件保存路径
         */
        void saveEngineFile(IHostMemory *data, const std::string engine_file_path);
        /**
         * @brief 执行推理
         * @param frames
         * 需要推理的图像序列，图像数量决定推理时batch_size，不可大于初始化模型时指定的batch_size
         * @param confidence_threshold
         * 置信度阈值
         * @param nms_threshold
         * 非极大值抑制阈值
         */
        std::vector<std::vector<ArmorObject>> doInference(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold);
        /**
         * @brief 构建engine
         * @param onnx_path
         * 用于构建engine的onnx文件路径
         * @param maxBatchSize
         * 最大batch_size设定值
         * @param input_h
         * Tensor输入图像尺寸 h
         * @param input_w
         * Tensor输入图像尺寸 w
         */
        IHostMemory *createEngine(const std::string onnx_path, unsigned int maxBatchSize, int input_h, int input_w);
    };
}

#endif // __INFERENCE_H
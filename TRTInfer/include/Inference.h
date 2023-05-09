#ifndef __INFERENCE_H
#define __INFERENCE_H

#include "public.h"

namespace TRTInferV1
{
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

        static constexpr int LOCATIONS = 4;
        struct alignas(float) Detection
        {
            // center_x center_y w h -> xyxy
            float bbox[LOCATIONS];
            float conf; // bbox_conf * cls_conf
            float class_id;
        };
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
        int batch_size = 0;
        int num_classes = -1;

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
        int inter_frame_compensation = 0;

    private:
        float iou(float lbox[4], float rbox[4]);
        void nms(std::vector<Yolo::Detection> &res, float *output, float conf_thresh, float nms_thresh);
        void postprocess(std::vector<std::vector<Yolo::Detection>> &batch_res, std::vector<cv::Mat> &frames, float &confidence_threshold, float &nms_threshold);

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
         * @param num_classes
         * num_classes设定值，类别数量
         */
        bool initMoudle(const std::string engine_file_path, const int batch_size, const int num_classes);
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
        std::vector<std::vector<Yolo::Detection>> doInference(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold);
        /**
         * @brief 计算帧内时间补偿
         * @param limited_fps
         * 目标FPS设定值，将根据此设定值计算时间补偿，配合doInferenceLimitFPS使用
         */
        void calculate_inter_frame_compensation(const int limited_fps);
        /**
         * @brief 执行推理(帧限制)
         * @param frames
         * 需要推理的图像序列，图像数量决定推理时batch_size，不可大于初始化模型时指定的batch_size
         * @param confidence_threshold
         * 置信度阈值
         * @param nms_threshold
         * 非极大值抑制阈值
         * @param limited_fps
         * 目标FPS设定值，推理过程的帧数将尝试限定在目标值附近，若运行帧率大于设定值，实际帧数将会接近并稳定下来，指定帧数越高，实际帧数偏差越大，帧数稳定性为+1~-2FPS
         */
        std::vector<std::vector<Yolo::Detection>> doInferenceLimitFPS(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold, const int limited_fps);
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
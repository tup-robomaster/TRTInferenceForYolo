#ifndef __INFERENCE_H
#define __INFERENCE_H

#include "public.h"

namespace TRTInferV1
{
    struct DetectionObj
    {
        int classId;
        float confidence;
        float x1;
        float y1;
        float x2;
        float y2;
    };

    /**
     * @brief TRT推理
     * 高性能TRT YOLOv5推理模块
     */
    class TRTInfer
    {
    private:
        const char *INPUT_BLOB_NAME = "images";   // 输入Tensor名称
        const char *OUTPUT_BLOB_NAME = "output0"; // 输出Tensor名称
        int batch_size = 0;
        int num_classes = -1;
        int input_size = 0;

    private:
        TRTLogger gLogger;
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
        int num_stride = 0;
        const int num_stride_640 = 3;
        const int num_stride_1280 = 4;

        const float anchors_640[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0},
                                         {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},
                                         {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};

        const float anchors_1280[4][6] = {{19, 27, 44, 40, 38, 94},
                                          {96, 68, 86, 152, 180, 137},
                                          {140, 301, 303, 264, 238, 542},
                                          {436, 615, 739, 380, 925, 792}};

        float *anchors;

    private:
        int inter_frame_compensation = 0;
        bool _is_inited = false;

    private:
        void nms(std::vector<DetectionObj> &input_boxes, float &nms_threshold);
        void decode_output(std::vector<DetectionObj> &res, cv::Mat &frame, float *pdata, float &obj_threshold, float &confidence_threshold, float &nms_threshold);
        void postprocess(std::vector<std::vector<DetectionObj>> &batch_res, std::vector<cv::Mat> &frames, float &obj_threshold, float &confidence_threshold, float &nms_threshold);

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
        bool initModule(const std::string engine_file_path, const int batch_size, const int num_classes);
        /**
         * @brief 反初始化TRT模型，释放显存
         */
        void unInitModule();
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
         * @param obj_threshold
         * box置信度阈值
         * @param confidence_threshold
         * 置信度阈值
         * @param nms_threshold
         * 非极大值抑制阈值
         */
        std::vector<std::vector<DetectionObj>> doInference(std::vector<cv::Mat> &frames, float obj_threshold, float confidence_threshold, float nms_threshold);
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
         * @param obj_threshold
         * box置信度阈值
         * @param confidence_threshold
         * 置信度阈值
         * @param nms_threshold
         * 非极大值抑制阈值
         * @param limited_fps
         * 目标FPS设定值，推理过程的帧数将尝试限定在目标值附近，若运行帧率大于设定值，实际帧数将会接近并稳定下来，指定帧数越高，实际帧数偏差越大，帧数稳定性为+1~-2FPS
         */
        std::vector<std::vector<DetectionObj>> doInferenceLimitFPS(std::vector<cv::Mat> &frames, float obj_threshold, float confidence_threshold, float nms_threshold, const int limited_fps);
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
         * @param BuilderFlag
         * 计算精度 0:FP16 1:INT8 else:None
         */
        IHostMemory *createEngine(const std::string onnx_path, unsigned int maxBatchSize, int input_h, int input_w, int BuilderFlag);

        /**
         *@brief 获取输入大小W
         */
        int getInputW();
        /**
         *@brief 获取输入大小H
         */
        int getInputH();
    };
}

#endif // __INFERENCE_H

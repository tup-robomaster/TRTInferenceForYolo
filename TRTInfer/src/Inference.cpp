#include "../include/Inference.h"
namespace TRTInferV1
{
    static constexpr int NUM_CLASSES = 8; // Number of classes
    static constexpr int NUM_COLORS = 8;  // Number of color
    static constexpr int TOPK = 128;      // TopK
    static constexpr float MERGE_CONF_ERROR = 0.15;
    static constexpr float MERGE_MIN_IOU = 0.9;

    static inline int argmax(const float *ptr, int len)
    {
        int max_arg = 0;
        for (int i = 1; i < len; i++)
        {
            if (ptr[i] > ptr[max_arg])
                max_arg = i;
        }
        return max_arg;
    }

    static void qsort_descent_inplace(std::vector<ArmorObject> &faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);
                i++;
                j--;
            }
        }
        if (left < j)
            qsort_descent_inplace(faceobjects, left, j);
        if (i < right)
            qsort_descent_inplace(faceobjects, i, right);
    }

    static void qsort_descent_inplace(std::vector<ArmorObject> &objects)
    {
        if (objects.empty())
            return;

        qsort_descent_inplace(objects, 0, objects.size() - 1);
    }

    static inline float intersection_area(const ArmorObject &a, const ArmorObject &b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    static void nms_sorted_bboxes(std::vector<ArmorObject> &faceobjects, std::vector<int> &picked, float nms_threshold)
    {
        picked.clear();
        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            ArmorObject &a = faceobjects[i];
            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                ArmorObject &b = faceobjects[picked[j]];
                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                float iou = inter_area / union_area;
                if (iou > nms_threshold || isnan(iou))
                {
                    keep = 0;
                    // Stored for Merge
                    if (iou > MERGE_MIN_IOU && abs(a.prob - b.prob) < MERGE_CONF_ERROR && a.cls == b.cls && a.color == b.color)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            b.pts.emplace_back(a.apex[i]);
                        }
                    }
                }
            }
            if (keep)
                picked.emplace_back(i);
        }
    }

    float calcTriangleArea(cv::Point2f pts[3])
    {
        auto a = sqrt(pow((pts[0] - pts[1]).x, 2) + pow((pts[0] - pts[1]).y, 2));
        auto b = sqrt(pow((pts[1] - pts[2]).x, 2) + pow((pts[1] - pts[2]).y, 2));
        auto c = sqrt(pow((pts[2] - pts[0]).x, 2) + pow((pts[2] - pts[0]).y, 2));
        auto p = (a + b + c) / 2.f;
        return sqrt(p * (p - a) * (p - b) * (p - c));
    }

    float calcTetragonArea(cv::Point2f pts[4])
    {
        return calcTriangleArea(&pts[0]) + calcTriangleArea(&pts[1]);
    }

    void TRTInfer::generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
    {
        for (auto stride : strides)
        {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;

            for (int g1 = 0; g1 < num_grid_h; g1++)
            {
                for (int g0 = 0; g0 < num_grid_w; g0++)
                {
                    GridAndStride grid_stride = {g0, g1, stride};
                    grid_strides.emplace_back(grid_stride);
                }
            }
        }
    }

    void generateYoloxProposals(std::vector<GridAndStride> grid_strides, const float *feat_ptr,
                                Eigen::Matrix<float, 3, 3> &transform_matrix, float prob_threshold,
                                std::vector<ArmorObject> &objects)
    {

        const int num_anchors = grid_strides.size();
        // Travel all the anchors
        for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx)
        {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;
            const int basic_pos = anchor_idx * (9 + NUM_COLORS + NUM_CLASSES);

            float x_1 = (feat_ptr[basic_pos + 0] + grid0) * stride;
            float y_1 = (feat_ptr[basic_pos + 1] + grid1) * stride;
            float x_2 = (feat_ptr[basic_pos + 2] + grid0) * stride;
            float y_2 = (feat_ptr[basic_pos + 3] + grid1) * stride;
            float x_3 = (feat_ptr[basic_pos + 4] + grid0) * stride;
            float y_3 = (feat_ptr[basic_pos + 5] + grid1) * stride;
            float x_4 = (feat_ptr[basic_pos + 6] + grid0) * stride;
            float y_4 = (feat_ptr[basic_pos + 7] + grid1) * stride;

            int box_color = argmax(feat_ptr + basic_pos + 9, NUM_COLORS);
            int box_class = argmax(feat_ptr + basic_pos + 9 + NUM_COLORS, NUM_CLASSES);
            float box_objectness = (feat_ptr[basic_pos + 8]);
            float box_prob = box_objectness;

            if (box_prob >= prob_threshold)
            {
                ArmorObject obj;

                Eigen::Matrix<float, 3, 4> apex_norm;
                Eigen::Matrix<float, 3, 4> apex_dst;

                apex_norm << x_1, x_2, x_3, x_4,
                    y_1, y_2, y_3, y_4,
                    1, 1, 1, 1;

                apex_dst = transform_matrix * apex_norm;

                for (int i = 0; i < 4; i++)
                {
                    obj.apex[i] = cv::Point2f(apex_dst(0, i), apex_dst(1, i));
                    obj.pts.emplace_back(obj.apex[i]);
                }

                std::vector<cv::Point2f> tmp(obj.apex, obj.apex + 4);
                obj.rect = cv::boundingRect(tmp);
                obj.cls = box_class;
                obj.color = box_color;
                obj.prob = box_prob;

                objects.emplace_back(obj);
            }
        } // point anchor loop
    }

    void TRTInfer::decodeOutputs(const float *prob, std::vector<ArmorObject> &objects, Eigen::Matrix<float, 3, 3> &transform_matrix, float confidence_threshold, float nms_threshold)
    {
        std::vector<ArmorObject> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;

        generate_grids_and_stride(this->input_dims.d[3], this->input_dims.d[2], strides, grid_strides);
        generateYoloxProposals(grid_strides, prob, transform_matrix, confidence_threshold, proposals);
        qsort_descent_inplace(proposals);
        if (proposals.size() >= TOPK)
            proposals.resize(TOPK);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);
        int count = picked.size();
        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];
        }
    }

    TRTInfer::TRTInfer(const int device)
    {
        cudaSetDevice(device);
    }

    TRTInfer::~TRTInfer()
    {
    }

    bool TRTInfer::initMoudle(const std::string engine_file_path, const int batch_size)
    {
        char *trtModelStream{nullptr};
        size_t size{0};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        else
        {
            this->gLogger.log(ILogger::Severity::kERROR, "Engine bad file");
            return false;
        }
        this->runtime = createInferRuntime(this->gLogger);
        assert(runtime != nullptr);
        this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
        assert(this->engine != nullptr);
        this->context = this->engine->createExecutionContext();
        assert(context != nullptr);
        delete trtModelStream;
        this->input_dims = this->engine->getTensorShape(INPUT_BLOB_NAME);
        this->input_dims.d[0] = batch_size;
        this->output_dims = this->engine->getTensorShape(OUTPUT_BLOB_NAME);
        this->context->setInputShape(INPUT_BLOB_NAME, input_dims);
        this->output_size = output_dims.d[1] * output_dims.d[2];
        int IOtensorsNum = engine->getNbIOTensors();
        assert(IOtensorsNum == 2);
        for (int i = 0; i < IOtensorsNum; ++i)
        {
            if (strcmp(this->engine->getIOTensorName(i), INPUT_BLOB_NAME))
            {
                this->inputIndex = i;
                assert(this->engine->getTensorDataType(INPUT_BLOB_NAME) == nvinfer1::DataType::kFLOAT);
            }
            else if (strcmp(this->engine->getIOTensorName(i), OUTPUT_BLOB_NAME))
            {
                this->outputIndex = i;
                assert(this->engine->getTensorDataType(OUTPUT_BLOB_NAME) == nvinfer1::DataType::kFLOAT);
            }
        }
        if (this->inputIndex == -1 || this->outputIndex == -1)
        {
            this->gLogger.log(ILogger::Severity::kERROR, "Uncorrect Input/Output tensor name");
            delete context;
            delete engine;
            return false;
        }
        CHECK(cudaMalloc(&buffers[inputIndex], batch_size * this->input_dims.d[1] * this->input_dims.d[2] * this->input_dims.d[3] * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batch_size * this->output_size * sizeof(float)));
        CHECK(cudaMallocHost((void **)&this->img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3 * sizeof(float)));
        CHECK(cudaMalloc((void **)&this->img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3 * sizeof(float)));
        this->output = (float *)malloc(batch_size * this->output_size * sizeof(float));
        return true;
    }

    void TRTInfer::unInitMoudle()
    {
        delete this->context;
        delete this->runtime;
        CHECK(cudaFree(img_device));
        CHECK(cudaFreeHost(img_host));
        CHECK(cudaFree(this->buffers[this->inputIndex]));
        CHECK(cudaFree(this->buffers[this->outputIndex]));
    }

    void TRTInfer::saveEngineFile(IHostMemory *data, const std::string engine_file_path)
    {
        std::string serialize_str;
        std::ofstream serialize_output_stream;
        serialize_str.resize(data->size());
        memcpy((void *)serialize_str.data(), data->data(), data->size());
        serialize_output_stream.open(engine_file_path);
        serialize_output_stream << serialize_str;
        serialize_output_stream.close();
    }

    std::vector<std::vector<ArmorObject>> TRTInfer::doInference(std::vector<cv::Mat> &frames, float confidence_threshold, float nms_threshold)
    {
        if (frames.size() == 0 || int(frames.size()) > this->input_dims.d[0])
        {
            this->gLogger.log(ILogger::Severity::kWARNING, "Invalid frames size");
            return {};
        }
        std::vector<std::vector<ArmorObject>> batch_res(frames.size());
        cudaStream_t stream = nullptr;
        CHECK(cudaStreamCreate(&stream));
        float *buffer_idx = (float *)buffers[this->inputIndex];
        for (size_t b = 0; b < frames.size(); ++b)
        {
            cv::Mat &img = frames[b];
            if (img.empty())
                continue;
            size_t size_image = img.cols * img.rows * 3;
            size_t size_image_dst = this->input_dims.d[3] * this->input_dims.d[2] * 3;
            memcpy(img_host, img.data, size_image);
            CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, this->input_dims.d[3], this->input_dims.d[2], stream);
            buffer_idx += size_image_dst;
        }
        this->context->setOptimizationProfileAsync(0, stream);
        this->context->setTensorAddress(INPUT_BLOB_NAME, this->buffers[this->inputIndex]);
        this->context->setTensorAddress(OUTPUT_BLOB_NAME, this->buffers[this->outputIndex]);
        bool success = this->context->enqueueV3(stream);
        if (!success)
        {
            this->gLogger.log(ILogger::Severity::kERROR, "DoInference failed");
            CHECK(cudaStreamDestroy(stream));
            return {};
        }
        CHECK(cudaMemcpyAsync(this->output, buffers[this->outputIndex], frames.size() * this->output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaStreamDestroy(stream));
        for (int b = 0; b < int(frames.size()); ++b)
        {
            auto &res = batch_res[b];
            float r = std::min(this->input_dims.d[3] / (frames[b].cols * 1.0), this->input_dims.d[2] / (frames[b].rows * 1.0));
            int unpad_w = r * frames[b].cols;
            int unpad_h = r * frames[b].rows;

            int dw = this->input_dims.d[3] - unpad_w;
            int dh = this->input_dims.d[2] - unpad_h;

            dw /= 2;
            dh /= 2;

            Eigen::Matrix3f transform_matrix;
            transform_matrix << 1.0 / r, 0, -dw / r,
                0, 1.0 / r, -dh / r,
                0, 0, 1;
            this->decodeOutputs(&this->output[b * this->output_size], res, transform_matrix, confidence_threshold, nms_threshold);
            for (auto object = res.begin(); object != res.end(); ++object)
            {
                // 对候选框预测角点进行平均,降低误差
                if ((*object).pts.size() >= 8)
                {
                    auto N = (*object).pts.size();
                    cv::Point2f pts_final[4];
                    for (int i = 0; i < (int)N; i++)
                    {
                        pts_final[i % 4] += (*object).pts[i];
                    }

                    for (int i = 0; i < 4; i++)
                    {
                        pts_final[i].x = pts_final[i].x / (N / 4);
                        pts_final[i].y = pts_final[i].y / (N / 4);
                    }

                    (*object).apex[0] = pts_final[0];
                    (*object).apex[1] = pts_final[1];
                    (*object).apex[2] = pts_final[2];
                    (*object).apex[3] = pts_final[3];
                }

                (*object).area = (int)(calcTetragonArea((*object).apex));
            }
        }
        return batch_res;
    }

    IHostMemory *TRTInfer::createEngine(const std::string onnx_path, unsigned int maxBatchSize, int input_h, int input_w)
    {
        IBuilder *builder = createInferBuilder(this->gLogger);
        uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(flag);

        IParser *parser = createParser(*network, gLogger);
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(ILogger::Severity::kWARNING)))
        {
            this->gLogger.log(ILogger::Severity::kINTERNAL_ERROR, "failed parse the onnx mode");
        }
        // 解析有错误将返回
        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cout << parser->getError(i)->desc() << std::endl;
        }
        this->gLogger.log(ILogger::Severity::kINFO, "successfully parse the onnx mode");
        IBuilderConfig *config = builder->createBuilderConfig();
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        // 这里有个OptProfileSelector，这个用来设置优化的参数,比如（Tensor的形状或者动态尺寸），

        profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, Dims4(1, 3, input_h, input_w));
        profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, Dims4(int(maxBatchSize / 2), 3, input_h, input_w));
        profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, Dims4(maxBatchSize, 3, input_h, input_w));

        // Build engine
        config->addOptimizationProfile(profile);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 10 << 20);
        config->setFlag(nvinfer1::BuilderFlag::kFP16); // 设置精度计算
        // config->setFlag(nvinfer1::BuilderFlag::kINT8);
        IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);
        this->gLogger.log(ILogger::Severity::kINFO, "successfully convert onnx to engine");

        // 销毁
        delete network;
        delete parser;
        delete config;
        delete builder;

        return serializedModel;
    }
}
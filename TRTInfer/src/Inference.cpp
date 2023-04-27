#include "../include/Inference.h"

cv::Rect get_rect(cv::Mat &img, float bbox[4], int input_H, int input_W)
{
    float l, r, t, b;
    float r_w = input_W / (img.cols * 1.0);
    float r_h = input_H / (img.rows * 1.0);
    if (r_h > r_w)
    {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (input_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (input_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else
    {
        l = bbox[0] - bbox[2] / 2.f - (input_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (input_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

float iou(float lbox[4], float rbox[4])
{
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
        (std::min)(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
        (std::max)(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
        (std::min)(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection &a, const Yolo::Detection &b)
{
    return a.conf > b.conf;
}

void nms(std::vector<Yolo::Detection> &res, float *output, float conf_thresh, float nms_thresh)
{
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BBOX_COUNT; ++i)
    {
        if (output[1 + det_size * i + 4] <= conf_thresh)
            continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0)
            m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); ++it)
    {
        // std::cout << it->second[0].class_id << " --- " << std::endl;
        auto &dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m)
        {
            auto &item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n)
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh)
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

TRTInfer::TRTInfer(const int device)
{
    cudaSetDevice(device);
}

TRTInfer::~TRTInfer()
{
}

bool TRTInfer::initMoudle(const std::string engine_file_path, const int max_batch, const int img_h, const int img_w)
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
    ICudaEngine *engine;
    engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    this->input_dims = engine->getTensorShape(INPUT_BLOB_NAME);
    this->out_dims = engine->getTensorShape(OUTPUT_BLOB_NAME);
    for (int i = 0; i < this->out_dims.nbDims; ++i)
    {
        this->output_size *= this->out_dims.d[i];
    }
    int IOtensorsNum = engine->getNbIOTensors();
    assert(IOtensorsNum == 2);
    bool IOtensorsNameCheck = true;
    for (int i = 0; i < IOtensorsNum; ++i)
    {
        if (engine->getIOTensorName(i) == INPUT_BLOB_NAME)
        {
            inputIndex = i;
            assert(engine->getTensorDataType(INPUT_BLOB_NAME) == nvinfer1::DataType::kFLOAT);
        }
        else if (engine->getIOTensorName(i) == OUTPUT_BLOB_NAME)
        {
            outputIndex = i;
            assert(engine->getTensorDataType(OUTPUT_BLOB_NAME) == nvinfer1::DataType::kFLOAT);
        }
        else
        {
            IOtensorsNameCheck = false;
        }
    }
    if (!IOtensorsNameCheck)
    {
        this->gLogger.log(ILogger::Severity::kERROR, "Uncorrect Input/Output tensor name");
        delete[] engine;
        return false;
    }
    CHECK(cudaMalloc(&buffers[inputIndex], max_batch * 3 * img_h * img_w * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], max_batch * this->output_size * sizeof(float)));
    CHECK(cudaMallocHost((void **)&this->img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc((void **)&this->img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    this->output = (float *)malloc(max_batch * this->output_size * sizeof(float));
    delete[] engine;
    return true;
}

void TRTInfer::unInitMoudle()
{
    delete[] this->context;
    delete[] this->runtime;
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(this->buffers[this->inputIndex]));
    CHECK(cudaFree(this->buffers[this->outputIndex]));
}

std::vector<std::vector<Yolo::Detection>> TRTInfer::doInference(std::shared_ptr<std::vector<cv::Mat>> frames, int batchSize, float confidence_threshold, float nms_threshold)
{
    std::vector<std::vector<Yolo::Detection>> batch_res(batchSize);
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreate(&stream));
    float *buffer_idx = (float *)buffers[this->inputIndex];
    for (size_t b = 0; b < frames->size(); ++b)
    {
        cv::Mat img = frames->at(b);
        if (img.empty())
            continue;
        size_t size_image = img.cols * img.rows * 3;
        size_t size_image_dst = this->input_dims.d[2] * this->input_dims.d[3] * 3;
        memcpy(img_host, img.data, size_image);
        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, this->input_dims.d[2], this->input_dims.d[3], stream);
        buffer_idx += size_image_dst;
    }
    bool success = this->context->enqueueV3(stream);
    if (!success)
    {
        this->gLogger.log(ILogger::Severity::kERROR, "DoInference failed");
        CHECK(cudaStreamDestroy(stream));
        return {};
    }
    CHECK(cudaMemcpyAsync(this->output, buffers[1], batchSize * this->output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamDestroy(stream));
    for (int b = 0; b < int(frames->size()); ++b)
    {
        auto &res = batch_res[b];
        nms(res, &this->output[b * this->output_size], confidence_threshold, nms_threshold);
    }
    return batch_res;
}

ICudaEngine *TRTInfer::createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config)
{
    const char *onnx_path = "./best.onnx";

    INetworkDefinition *network = builder->createNetworkV2(1U); // 此处重点1U为OU就有问题

    IParser *parser = createParser(*network, gLogger);
    parser->parseFromFile(onnx_path, static_cast<int32_t>(ILogger::Severity::kWARNING));
    // 解析有错误将返回
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    this->gLogger.log(ILogger::Severity::kINFO, "successfully parse the onnx mode");

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(nvinfer1::BuilderFlag::kFP16); // 设置精度计算
    // config->setFlag(nvinfer1::BuilderFlag::kINT8);
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    this->gLogger.log(ILogger::Severity::kINFO, "successfully  convert onnx to  engine");

    // 销毁
    delete[] network;
    delete[] parser;

    return engine;
}

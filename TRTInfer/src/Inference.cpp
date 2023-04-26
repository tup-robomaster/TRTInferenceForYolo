#include "../include/Inference.h"

TRTInfer::TRTInfer()
{
    cudaSetDevice(DEVICE);
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
        std::cout << "[ERR] bad file" << std::endl;
        return false;
    }
    this->runtime = createInferRuntime(this->gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    for (int j = 0; j < out_dims.nbDims; j++)
    {
        this->output_size *= out_dims.d[j];
    }
    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    mBatchSize = engine->getMaxBatchSize();
    engine->destroy();
    CHECK(cudaMalloc(&buffers[inputIndex], max_batch * 3 * img_h * img_w * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], this->output_size * sizeof(float)));
    return true;
}

void TRTInfer::unInitMoudle()
{
    context->destroy();
    runtime->destroy();
    CHECK(cudaFree(buffers[this->inputIndex]));
    CHECK(cudaFree(buffers[this->outputIndex]));
}
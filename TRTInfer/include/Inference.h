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
    int mBatchSize;

public:
    TRTInfer();
    ~TRTInfer();

    bool initMoudle(const std::string engine_file_path, const int max_batch, const int img_h, const int img_w);
    void unInitMoudle();
};
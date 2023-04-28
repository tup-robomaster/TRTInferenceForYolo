#include "./include/main.h"

int main()
{
    TRTInferV1::TRTInfer myInfer(0);
    myInfer.initMoudle("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/build/TRTInferSample", 1, 416, 416);
    std::shared_ptr<std::vector<cv::Mat>> frames;
    frames->emplace_back(cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/include/main.h"));
    myInfer.doInference(frames,1,0.7,0.3);
    return 0;
}
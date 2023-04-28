#include "./include/main.h"

int main()
{
    cv::namedWindow("Test",cv::WINDOW_NORMAL);
    TRTInferV1::TRTInfer myInfer(0);
    // nvinfer1::IHostMemory *data = myInfer.createEngine("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/build/best.onnx", 5);
    // myInfer.saveEngineFile(data, "/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/engines/model_trt.engine");
    myInfer.initMoudle("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/engines/model_trt.engine", 1, 416, 416);
    std::vector<cv::Mat> frames;
    cv::Mat img = cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/46.jpg");
    assert(!img.empty());
    frames.push_back(img);
    std::vector<std::vector<TRTInferV1::ArmorObject>> result = myInfer.doInference(frames, 0.9, 0.5);
    std::vector<TRTInferV1::ArmorObject> _batch0 = result[0];
    for (int i(0); i < int(_batch0.size()); ++i)
    {
        cv::line(img, _batch0[i].apex[0], _batch0[i].apex[1], cv::Scalar(255, 255, 255), 2);
        cv::line(img, _batch0[i].apex[1], _batch0[i].apex[2], cv::Scalar(255, 255, 255), 2);
        cv::line(img, _batch0[i].apex[2], _batch0[i].apex[3], cv::Scalar(255, 255, 255), 2);
        cv::line(img, _batch0[i].apex[3], _batch0[i].apex[0], cv::Scalar(255, 255, 255), 2);
        std::cout << _batch0[0].cls << " " << _batch0[i].color << " " << _batch0[i].prob << std::endl;
    }
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}
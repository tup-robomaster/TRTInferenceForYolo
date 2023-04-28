#include "./include/main.h"

int main()
{
    cv::namedWindow("Test", cv::WINDOW_NORMAL);
    TRTInferV1::TRTInfer myInfer(0);
    // nvinfer1::IHostMemory *data = myInfer.createEngine("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/build/best.onnx", 5);
    // myInfer.saveEngineFile(data, "/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/engines/model_trt.engine");
    myInfer.initMoudle("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/engines/model_trt.engine", 1, 416, 416);

    cv::VideoCapture cap(0);
    std::vector<cv::Mat> frames;

    while (true)
    {
        frames.clear();
        // cv::Mat img = cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/46.jpg");
        if (!cap.isOpened())
        {
            continue;
        }
        cv::Mat img;
        cap.read(img);
        assert(!img.empty());
        frames.push_back(img);
        auto start_t = std::chrono::system_clock::now().time_since_epoch();
        std::vector<std::vector<TRTInferV1::ArmorObject>> result = myInfer.doInference(frames, 0.9, 0.5);
        auto end_t = std::chrono::system_clock::now().time_since_epoch();
        std::vector<TRTInferV1::ArmorObject> _batch0 = result[0];
        for (int i(0); i < int(_batch0.size()); ++i)
        {
            cv::line(img, _batch0[i].apex[0], _batch0[i].apex[1], cv::Scalar(255, 255, 255), 2);
            cv::line(img, _batch0[i].apex[1], _batch0[i].apex[2], cv::Scalar(255, 255, 255), 2);
            cv::line(img, _batch0[i].apex[2], _batch0[i].apex[3], cv::Scalar(255, 255, 255), 2);
            cv::line(img, _batch0[i].apex[3], _batch0[i].apex[0], cv::Scalar(255, 255, 255), 2);
            char ch[10];
            sprintf(ch, "%d", int(std::chrono::nanoseconds(1000000000).count() / (end_t - start_t).count()));
            std::string fps_str = ch;
            cv::putText(img, fps_str, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 1, {0,255,0});
        }
        cv::imshow("Test", img);
        cv::waitKey(1);
    }

    return 0;
}
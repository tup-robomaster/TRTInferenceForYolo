#include "./include/main.h"

int main()
{
    cv::namedWindow("Test", cv::WINDOW_NORMAL);
    cv::namedWindow("Test2", cv::WINDOW_NORMAL);
    cv::namedWindow("Test3", cv::WINDOW_NORMAL);
    cv::namedWindow("Test4", cv::WINDOW_NORMAL);
    TRTInferV1::TRTInfer myInfer(0);
    // nvinfer1::IHostMemory *data = myInfer.createEngine("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/build/best.onnx", 8, 416, 416);
    // myInfer.saveEngineFile(data, "/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/engines/model_trt.engine");
    myInfer.initMoudle("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/engines/model_trt.engine", 4, 416, 416);

    // cv::VideoCapture cap(0);
    std::vector<cv::Mat> frames;

    while (true)
    {
        frames.clear();
        cv::Mat img = cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/46.jpg");
        cv::Mat img2 = cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/SAU0076.jpg");
        cv::Mat img3 = cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/1674.jpg");
        cv::Mat img4 = cv::imread("/home/ninefish/nine-fish/TRTInferenceForYoloX/sample/SAU0830.jpg");
        // if (!cap.isOpened())
        // {
        //     continue;
        // }
        // cv::Mat img;
        // cap.read(img);
        assert(!img.empty());
        frames.push_back(img);
        frames.push_back(img2);
        frames.push_back(img3);
        frames.push_back(img4);
        auto start_t = std::chrono::system_clock::now().time_since_epoch();
        std::vector<std::vector<TRTInferV1::ArmorObject>> result = myInfer.doInference(frames, 0.8, 0.5);
        auto end_t = std::chrono::system_clock::now().time_since_epoch();
        for (int i(0); i < int(frames.size()); ++i)
        {
            for (int j(0); j < int(result[i].size()); ++j)
            {
                cv::line(frames[i], result[i][j].apex[0], result[i][j].apex[1], cv::Scalar(255, 255, 255), 1);
                cv::line(frames[i], result[i][j].apex[1], result[i][j].apex[2], cv::Scalar(255, 255, 255), 1);
                cv::line(frames[i], result[i][j].apex[2], result[i][j].apex[3], cv::Scalar(255, 255, 255), 1);
                cv::line(frames[i], result[i][j].apex[3], result[i][j].apex[0], cv::Scalar(255, 255, 255), 1);
            }
            char ch[255];
            sprintf(ch, "FPS %d", int(std::chrono::nanoseconds(1000000000).count() / (end_t - start_t).count()));
            std::string fps_str = ch;
            cv::putText(frames[i], fps_str, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 1, {0, 255, 0});
        }
        std::cout << "FPS :" << int(std::chrono::nanoseconds(1000000000).count() / (end_t - start_t).count()) << std::endl;
        cv::imshow("Test", frames[0]);
        cv::imshow("Test2", frames[1]);
        cv::imshow("Test3", frames[2]);
        cv::imshow("Test4", frames[3]);
        cv::waitKey(1);
    }

    return 0;
}
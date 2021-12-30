#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "torch/script.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include  "detector.h"
using namespace cv;
using namespace std;
/* main */


int main(int argc, char *argv[])
{
    // 默认参数
    Config config = {0.25f, 0.45f, "yolov5s.torchscript", "data/coco.names"};
    cout<<"Load Model"<<endl;
    Detector detector(config);
    cout << "Read Image" << endl;
    string img_path = argv[1];
    Mat img = imread(img_path, IMREAD_COLOR);
    auto detection = detector.detect(img);
    detector.letterBoxImage(img);
    detector.postProcess(img,detection);
    imwrite("assets/output.jpg",img);
    cout << "detect Image" << endl;
    return 0;
}

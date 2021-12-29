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

#define DEFAULT_FONT_PATH "fonts/NotoSansCJKsc-Regular.otf"


int main(int argc, char *argv[])
{
    // 默认参数
    Config config = {0.25f, 0.45f, "yolov5s.torchscript", "data/coco.names"};
    Detector detector(config);
    string img_path = "data/images/zidane.jpg";
    Mat img = imread(img_path, IMREAD_COLOR);
    detector.detect(img);
    return 0;
}

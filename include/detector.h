#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "torch/torch.h"
#include "torch/script.h"

using namespace cv;
using namespace std;

struct Config{
    float confThreshold;
    float nmsThreshold;
    string weightPath;
    string classNamePath;
};

class Detector{
    public:
        Detector(Config&config);
        void detect(Mat&img);
        void postProcess(torch::Tensor&detection);
    private:
        float nmsThreshold = 0.45;
        float confThreshold = 0.25;
        int inWidth = 640;
        int inHeight = 640;
        vector<string> classNames;
        torch::jit::script::Module model;
        vector<float> letterBoxImage(Mat &img);
};

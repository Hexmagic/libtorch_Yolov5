#include "torch/script.h"
#include "ATen/ATen.h"
#include "torch/library.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace cv;
using namespace std;

vector<int> nms(torch::Tensor& boxes, torch::Tensor& scores, torch::Tensor& labels, double conf_thresh, double nms_thresh){
    vector<int> keep;
    if (boxes.numel() == 0)
        return keep;
    auto conf_mask = scores > conf_thresh;
    cout << "Score Size:" << scores.sizes() << endl;
    scores = scores.masked_select(conf_mask);
    cout << "Conf Thresh Score Size:" << scores.sizes() << endl;
    conf_mask = conf_mask.unsqueeze(1).expand_as(boxes);
    boxes = boxes.masked_select(conf_mask).reshape({-1,4});
    cout << "Conf Thresh Boxes Size:" << boxes.sizes() << endl;
    auto x1 = boxes.select(1, 0).contiguous();
    auto y1 = boxes.select(1, 1).contiguous();
    auto x2 = boxes.select(1, 2).contiguous();
    auto y2 = boxes.select(1, 3).contiguous();
    auto areas = (x2 - x1) * (y2 - y1);
    auto ordered_index = torch::argsort(scores);
    cout << ordered_index << endl;
    int box_num = scores.numel();
    cout << "Box Num " << box_num << endl;
    int suppressed[box_num];
    memset(suppressed, 0, sizeof(suppressed));


    int cnt = 0;
    for (int i = 0;i < box_num;i++){
        int index = ordered_index[i].item<int>();
        if (suppressed[index] == 1)
            continue;
        keep.push_back(index);
        auto ix1 = x1[index];
        auto iy1 = y1[index];
        auto ix2 = x2[index];
        auto iy2 = y2[index];
        auto areai = areas[index].item<double>();
        cout << "Box " << index << " Area:" << areai << endl;

        for (int j = i + 1;j < box_num;j++){
            auto index_ = ordered_index[j].item<int>();
            if (suppressed[index_])
                continue;
            auto jx1 = x1[index_];
            auto jy1 = y1[index_];
            auto jx2 = x2[index_];
            auto jy2 = y2[index_];
            auto inter_tx = max(jx1, ix1);
            auto inter_ty = max(jy1, iy1);
            auto inter_bx = min(jx2, ix2);
            auto inter_by = min(jy2, iy2);
            auto h = max(torch::zeros({1}), inter_by - inter_ty);
            auto w = max(torch::zeros({1}), inter_bx - inter_tx);
            auto inter_area = h * w;
            auto areaj = areas[index_].item<double>() ;
            auto area_inter = inter_area.item<double>();
            cout << "Box " << index_ << "  Area:" << areaj << endl;
            cout << "Box " << index << " Box " << index_ << " Inter:" << area_inter << endl;
            auto iou = inter_area / (areai + areaj - inter_area);
            cout << "Box " << index << " Box " << index_ << " IOU:" << iou << endl;
            if (iou.item<double>() > nms_thresh){
                suppressed[index_] = 1;
            }
        }
    }
    return keep;
}

int main(){
    auto model = torch::jit::load("yolov3.pt");
    vector<torch::IValue> inputs;
    Mat img = imread("test.jpg");
    copyMakeBorder(img, img, 420, 420, 0, 0, BORDER_CONSTANT, Scalar(0));
    resize(img, img, {416,416});
    cv::cvtColor(img, img, COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1 / 255.0, 0);
    torch::Tensor img_tensor = torch::from_blob(img.data, {1,416,416,3});
    img_tensor = torch::permute(img_tensor, {0,3,1,2});
    inputs.push_back(img_tensor);
    auto output = model(inputs).toTensor();
    output = output[0];
    cout << output.sizes() << endl;
    double nms_thresh = 0.7;
    double conf_thresh = 0.6;
    auto boxes = output.slice(1, 0, 4);
    auto w = boxes.select(1, 2).clone();
    auto h = boxes.select(1, 3).clone();
    boxes.select(1, 2) = boxes.select(1, 0) + w;
    boxes.select(1, 3) = boxes.select(1, 1) + h;
    auto top10 = boxes.slice(0, 1, 10);
    cout << "Top 10:" << top10 << endl;
    cout << "BoxNum: " << boxes.sizes() << endl;
    auto scores = output.select(1, 4);
    auto labels = output.slice(1, 5, 8);
    auto keep = nms(boxes, scores, labels, conf_thresh, nms_thresh);
    cout << "Keeped Index: " << endl;
    for (auto& index : keep){
        cout << index << " ";
    }
    cout << "Boxes Size" << boxes.sizes() << endl;

}
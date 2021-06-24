#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "torch/script.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <opencv2/freetype.hpp>
using namespace std;
/* main */

#define DEFAULT_FONT_PATH "fonts/NotoSansCJKsc-Regular.otf"

vector<int> nms(torch::Tensor& boxes, torch::Tensor& scores, torch::Tensor& labels, double conf_thresh, double nms_thresh){
    vector<int> keep;
    if (boxes.numel() == 0)
        return keep;
    auto conf_mask = scores > conf_thresh;

    scores = scores.masked_select(conf_mask);
    labels = labels.masked_select(conf_mask);
    boxes = boxes.masked_select(conf_mask.unsqueeze(1).expand_as(boxes)).reshape({-1,4});

    auto x1 = boxes.select(1, 0).contiguous();// 左上角X 坐标
    auto y1 = boxes.select(1, 1).contiguous();// 左上角Y 坐标
    auto x2 = boxes.select(1, 2).contiguous();// 右下角 X
    auto y2 = boxes.select(1, 3).contiguous();// 右下角 Y
    auto areas = (x2 - x1) * (y2 - y1);// 每个预测框的面积
    auto ordered_index = torch::argsort(scores); // 按score进行排序
    int box_num = scores.numel();

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
            auto iou = inter_area / (areai + areaj - inter_area);// IOU
            if (iou.item<double>() > nms_thresh){
                suppressed[index_] = 1;
            }
        }
    }
    return keep;
}
// 对比YOLOv3_Pytorch的yolo_loss进行复刻
class YoloLoss{
    public:
    vector<vector<int>> anchors;
    int num_classes = 80;
    int num_anchors = 3;
    int WIDTH = 416;
    int bbox_atts = 85;
    int HEIGHT = 416;    
    

    YoloLoss(vector<vector<int>>& anchors) :anchors(anchors){

    }
    torch::Tensor forward(torch::Tensor& input){
        auto bs = input.size(0);
        auto in_h = input.size(2);
        auto in_w = input.size(3);
        double stride_h = this->HEIGHT / in_h;
        double stride_w = this->WIDTH / in_w;
        vector<double> scaled_w;
        vector<double> scaled_h;
        for (auto& anchor : this->anchors){
            scaled_w.push_back(anchor[0] / stride_w);
            scaled_h.push_back(anchor[1] / stride_h);
        }
        cout << "Input Size" << input.sizes() << endl;
        auto prediction = input.view({bs,this->num_anchors,this->bbox_atts,in_h,in_w}).contiguous();
        auto tmp = prediction.permute({2,0,1,3,4});
        auto x = torch::sigmoid(tmp.select(0, 0));
        auto y = torch::sigmoid(tmp.select(0, 1));
        auto w = tmp.select(0, 2);
        auto h = tmp.select(0, 3);
        auto conf = torch::sigmoid(tmp.select(0, 4).contiguous());
        auto pred_cls = torch::sigmoid(tmp.slice(0, 5, this->bbox_atts)).contiguous();
        auto grid_x = torch::linspace(0, in_w - 1, in_w).repeat({in_w,1}).repeat({bs * this->num_anchors,1,1}).view(x.sizes()).to(torch::kFloat32);
        auto grid_y = torch::linspace(0, in_h - 1, in_h).repeat({in_h,1}).t().repeat({bs * this->num_anchors,1,1}).view(y.sizes()).to(torch::kFloat32);
        auto anchor_w = torch::tensor(scaled_w).reshape({this->num_anchors,1});
        auto anchor_h = torch::tensor(scaled_h).reshape({this->num_anchors,1});
        anchor_w = anchor_w.repeat({bs,1}).repeat({1,1,in_h * in_w}).view(w.sizes());
        anchor_h = anchor_h.repeat({bs,1}).repeat({1,1,in_h * in_w}).view(h.sizes());
        auto pred_boxes = torch::zeros({4,bs,this->num_anchors,in_h,in_w});
        cout << pred_boxes.sizes() << endl;
        pred_boxes[0] = x.data() + grid_x;
        pred_boxes[1] = y.data() + grid_y;
        pred_boxes[2] = torch::exp(w.data()) * anchor_w;
        pred_boxes[3] = torch::exp(h.data()) * anchor_h;
        pred_boxes = pred_boxes.permute({1,2,3,4,0});
        vector<double> stride_arr = {stride_w,stride_h,stride_w,stride_h};
        auto _scale = torch::tensor(stride_arr);
        conf = conf.view({bs,-1 ,1});
        cout << pred_cls.sizes() << endl;
        pred_cls = pred_cls.permute({1,2,3,4,0}).view({bs,-1,this->num_classes});
        pred_boxes = pred_boxes.contiguous().view({bs,-1,4}) * _scale;
        cout << "Cat Tensor: Box Size " << pred_boxes.sizes() << " Conf size: " << conf.sizes() << " Cls size: " << pred_cls.sizes() << endl;
        auto output = torch::cat({
           pred_boxes,
           conf,
           pred_cls
            }, -1);
        return output;
    }
};

vector<cv::Scalar> get_colors(int num_classes){
    vector<cv::Scalar> colors;
    for (int i = 0;i < num_classes;i++){
        auto color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        colors.push_back(color);
    }
    return colors;
}
void loadClassNames(string path, vector<string>& classnames){
    ifstream coconame(path);
    string row;
    while (getline(coconame, row))
    {
        classnames.push_back(row);
    }
}
static void drawText(cv::Mat& img, const std::string& text, int fontHeight, const cv::Scalar& fgColor,
    const cv::Scalar& bgColor, const cv::Point& leftTopShift) {
    if (text.empty()) {
        printf("text cannot be empty!\n");
        return;
    }

    cv::Ptr<cv::freetype::FreeType2> ft2;
    int baseline = 0;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData(DEFAULT_FONT_PATH, 0);
    cv::Size textSize = ft2->getTextSize(text, fontHeight, -1, &baseline);
    cv::Point textLeftBottom(0, textSize.height);
    textLeftBottom -= cv::Point(0, baseline);   // (left, bottom) of text
    cv::Point textLeftTop(textLeftBottom.x, textLeftBottom.y - textSize.height);    // (left, top) of text
    // Draw text background
    textLeftTop += leftTopShift;
    cv::rectangle(img, textLeftTop, textLeftTop + cv::Point(textSize.width, textSize.height + baseline), bgColor,
        cv::FILLED);
    textLeftBottom += leftTopShift;
    ft2->putText(img, text, textLeftBottom, fontHeight, fgColor, -1, cv::LINE_AA, true);
}

int main(int argc,char *argv[]){
    // 默认参数
    string img_path = argv[1];
    vector<string>classNames;
    loadClassNames("data/coco.names", classNames);
    double conf_thresh = 0.5;
    double nms_thresh = 0.45;
    int PRED_WIDTH = 416;
    int PRED_HEIGHT = 416;
    vector<vector<vector<int>>> anchors = {
        {{116, 90}, {156, 198}, {373, 326}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{10, 13}, {16, 30}, {33, 23}}
    };
    vector<YoloLoss> yolo_losses;
    for (int i = 0;i < 3;i++){
        yolo_losses.push_back(YoloLoss(anchors[i]));
    }
    // 加载模型
    auto module = torch::jit::load("weights/model.pt");
    vector<cv::Scalar> colors = get_colors(80);
    // 加载处理图片
    auto oimg = cv::imread(img_path, cv::IMREAD_COLOR);
    int HEIGHT = oimg.rows;
    int WIDTH = oimg.cols;
    cv::Mat img;
    cv::resize(oimg, img, {PRED_WIDTH,PRED_HEIGHT}, cv::INTER_LINEAR);
    cv::Mat cimg;
    cv::cvtColor(img, cimg, cv::COLOR_BGR2RGB);
    cv::Mat img_float;
    cimg.convertTo(img_float, CV_32F, 1 / 255.0, 0);
    torch::Tensor img_tensor = torch::from_blob(img_float.data, {1,PRED_WIDTH,PRED_HEIGHT,3});
    img_tensor = img_tensor.permute({0,3,1,2});
    std::vector<torch::IValue> inputs;
    inputs.push_back(img_tensor);
    auto outputs = module(inputs).toTuple();

    // 处理输出 
    std::vector<torch::Tensor> output_list;
    auto first = outputs->elements()[0].toTensor();

    for (int i = 0;i < 3;i++){
        auto loss = yolo_losses[i];
        auto elem = outputs->elements()[i].toTensor();
        auto out = loss.forward(elem);
        output_list.push_back(out);
    }
    auto output = torch::cat(output_list, 1);
    output = output[0];
    auto boxes = output.slice(1, 0, 4);
    auto copy_box = boxes.clone();
    copy_box.select(1, 0) = boxes.select(1, 0) - boxes.select(1, 2) / 2;
    copy_box.select(1, 1) = boxes.select(1, 1) - boxes.select(1, 3) / 2;
    copy_box.select(1, 2) = boxes.select(1, 0) + boxes.select(1, 2) / 2;
    copy_box.select(1, 3) = boxes.select(1, 1) + boxes.select(1, 3) / 2;
    copy_box = copy_box.clamp(0, INT_MAX);
    auto scores = output.select(1, 4);
    auto labels = output.slice(1, 5, 85);
    labels = torch::argmax(labels, -1);

    auto keep = nms(copy_box, scores, labels, conf_thresh, nms_thresh);
    // 可视化
    for (auto& index : keep){
        auto box = copy_box[index];
        auto score = scores[index];
        std::vector<float> box_vec(box.data_ptr<float>(), box.data_ptr<float>() + box.numel());

        int org_w = (box_vec[2] - box_vec[0]) / PRED_WIDTH * WIDTH;
        int org_h = (box_vec[3] - box_vec[1]) / PRED_HEIGHT * HEIGHT;
        int org_x = box_vec[0]/PRED_WIDTH*WIDTH;
        int org_y = box_vec[1]/PRED_HEIGHT*HEIGHT;
        cv::Rect rect(org_x,org_y,org_w,org_h);
        auto cls = labels[index].item<int>();
        //cout<<colors[cls]<<endl;
        cv::rectangle(oimg, rect, colors[cls], 1);
        drawText(oimg, classNames[cls], 14, colors[cls], cv::Scalar(255, 255, 255), cv::Point(org_x, org_y));
    }
    cv::imshow("Test", oimg);
    cv::waitKey(0);
    cv::imwrite("assets/output.jpg", oimg);
}

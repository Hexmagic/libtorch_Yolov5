#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "torch/script.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nms.hpp"

torch::Tensor xywh2xyxy(torch::Tensor& box){
  auto new_box = torch::zeros_like(box);
  auto x = box.select(1, 0);
  auto y = box.select(1, 1);
  auto w = box.select(1, 2);
  auto h = box.select(1, 3);
  new_box.select(1, 0) = x - w / 2;
  new_box.select(1, 1) = y - h / 2;
  new_box.select(1, 2) = x + w / 2;
  new_box.select(1, 3) = y + h / 2;
  return new_box;
}

/* main */
int main(int argc, const char* argv[]) {

  torch::DeviceType device_type = torch::kCPU;


  auto module = torch::jit::load("../model.pt");
  std::cout << "load model" << std::endl;
  std::vector<torch::jit::IValue> inputs;

  cv::Mat img = cv::imread("../test.jpg");
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::Mat img_float;
  img.convertTo(img_float, 5, 1.0 / 255, 0);
  cv::resize(img_float, img_float, cv::Size(416, 416));

  auto img_tensor = torch::from_blob(img_float.data, {1,416,416,3});
  img_tensor = img_tensor.permute({0,3,1,2});
  auto img_var = torch::autograd::make_variable(img_tensor, false);
  inputs.push_back(img_var);
  at::Tensor output = module.forward(inputs).toTensor();
  output = output[0];
  auto old_box = output.slice(1, 0, 4);
  auto box = xywh2xyxy(old_box);
  auto score = output.slice(1, 4, 5).flatten();
  auto cls = output.slice(1, 5, 8);
  torch::Tensor keep = torch::zeros({score.size(0)}).to(torch::kLong).to(score.device());
  int count;
  std::cout << box[0] << std::endl;
  nms(box, score, keep, count, 0.5, 10);
  auto nonzero = torch::nonzero(keep).squeeze();
  std::cout<<keep[nonzero[0]]<<std::endl;
  std::cout << box[0] << std::endl;
  return 0;
}


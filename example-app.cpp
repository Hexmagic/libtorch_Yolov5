#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "torch/script.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* main */
int main(int argc, const char* argv[]) {

  auto module = torch::jit::load("../model.pt");
  std::cout<<"load model"<<std::endl;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1,3,416,416}));
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout<<output.slice(1,0,5)<<std::endl;
  return 0;
}


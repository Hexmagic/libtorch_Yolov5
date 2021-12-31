#include "detector.h"

Detector::Detector(Config &config)
{
    this->nmsThreshold = config.nmsThreshold;
    this->confThreshold = config.confThreshold;
    ifstream ifs(config.classNamePath);
    string line;
    while (getline(ifs, line))
        this->classNames.push_back(line);
    ifs.close();
    this->model = torch::jit::load(config.weightPath);
    this->model.to(torch::kCPU);
    this->model.eval();
}

PadInfo Detector::letterBoxImage(Mat &image)
{
    float row = image.rows*1.0f;
    float col = image.cols*1.0f;
    float scale = max(row / this->inHeight, col / this->inWidth);
    int dst_col = col / scale;
    int dst_row = row / scale;
    resize(image, image, Size(dst_col, dst_row));
    int left = (this->inWidth - dst_col) / 2;
    int top = (this->inHeight - dst_row) / 2;
    int right = (this->inWidth - dst_col + 1) / 2;
    int bottom = (this->inHeight - dst_row + 1) / 2;
    cv::copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return {scale, top, left};
}

Detection Detector::detect(Mat &img)
{
    // 预处理 添加border
    Mat im;
    img.copyTo(im);
    cout<<im.size()<<endl;
    PadInfo padInfo = letterBoxImage(im);

    cvtColor(im, im, COLOR_BGR2RGB);
    im.convertTo(im, CV_32FC3, 1.0 / 255.0);
    // 转换成tensor
    torch::Tensor tensor = torch::from_blob(im.data, {1, this->inHeight, this->inWidth, 3}).to(torch::kCPU);
    tensor = tensor.permute({0,3,1,2}).contiguous();
    // 获取输出 ,因为可以处理批，这里只用一张图片测试，所以获取第一个输出即可
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    auto output = this->model.forward(inputs);
    auto detection = output.toTuple()->elements()[0].toTensor();
    return {padInfo,detection};
}
void Detector::postProcess(Mat&img,Detection &detection)
{
    // 1 Num 85
    PadInfo padInfo = detection.info;
    // output{batchSize,Num,5+numClasses}
    torch::Tensor output = detection.detection;
    int classNums = output.size(2) - 5;
    int nums = output.size(1);
    auto output_0 = output.index({0});
    cout << output.size(0)<<"," << output.size(1)<<"," << output.size(2) << endl;
    std::vector<Rect> boxes;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<int> classIndexList;
    cout << output_0.size(0)<<"," << output_0.size(1)<<endl;
    for (int i = 0; i < output_0.size(0); i++)
    {
        auto row = output_0.index({i});
        scores.push_back(row.index({4}).item().toFloat());
        float cx = row.index({0}).item().toFloat();
        float cy = row.index({1}).item().toFloat();
        float w = row.index({2}).item().toFloat();
        float h = row.index({3}).item().toFloat();
        cv::Rect rect(cx-w/2, cy-h/2, w, h);
        boxes.push_back(rect);
        indices.push_back(i);
        int classIndex =torch::argmax(row.index({torch::indexing::Slice(5,torch::indexing::None)})).item().toInt();
        classIndexList.push_back(classIndex);
    }
    dnn::NMSBoxes(boxes, scores,this->confThreshold,this->nmsThreshold,indices);
    std::vector<string> rstNames;
    for(int i=0;i<indices.size();i++){
        string name = this->classNames[classIndexList[indices[i]]];
        rstNames.push_back(name);
    }
    drawPredection(img,boxes,scores,rstNames,indices);
}

void Detector::drawPredection(Mat&img,std::vector<Rect>&boxes,std::vector<float>&scores,std::vector<string>&clsNames,std::vector<int>&ind){
    for(int i=0;i<ind.size();i++){
        Rect rect = boxes[ind[i]];
        float score = scores[ind[i]];
        string name = clsNames[i];
        rectangle(img,rect,Scalar(0,0,255));
        cout<<name<<endl;
        putText(img,name,Point(rect.x,rect.y),FONT_HERSHEY_PLAIN,1.2 ,Scalar(0,255,0));
    }
    //imshow("rst",img);
    //waitKey(0);

}

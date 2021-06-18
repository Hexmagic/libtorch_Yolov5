#include "torch/script.h"

bool nms(const torch::Tensor& boxes, const torch::Tensor& scores, torch::Tensor& keep, int& count, float overlap, int top_k)
{
    count = 0;    
    if (0 == boxes.numel())
    {
        return false;
    }

    torch::Tensor x1 = boxes.select(1, 0).clone();
    torch::Tensor y1 = boxes.select(1, 1).clone();
    torch::Tensor x2 = boxes.select(1, 2).clone();
    torch::Tensor y2 = boxes.select(1, 3).clone();
    torch::Tensor area = (x2 - x1) * (y2 - y1);
    //    std::cout<<area<<std::endl;

    std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(scores.unsqueeze(1), 0, 0);
    torch::Tensor v = std::get<0>(sort_ret).squeeze(1).to(scores.device());
    torch::Tensor idx = std::get<1>(sort_ret).squeeze(1).to(scores.device());

    int num_ = idx.size(0);
    if (num_ > top_k) //python:idx = idx[-top_k:]
    {
        idx = idx.slice(0, num_ - top_k, num_).clone();
    }
    torch::Tensor xx1, yy1, xx2, yy2, w, h;
    while (idx.numel() > 0)
    {
        auto i = idx[-1];
        keep[count] = i;
        count += 1;
        if (1 == idx.size(0))
        {
            break;
        }
        idx = idx.slice(0, 0, idx.size(0) - 1).clone();

        xx1 = x1.index_select(0, idx);
        yy1 = y1.index_select(0, idx);
        xx2 = x2.index_select(0, idx);
        yy2 = y2.index_select(0, idx);

        xx1 = xx1.clamp(x1[i].item().toFloat(), INT_MAX * 1.0);
        yy1 = yy1.clamp(y1[i].item().toFloat(), INT_MAX * 1.0);
        xx2 = xx2.clamp(INT_MIN * 1.0, x2[i].item().toFloat());
        yy2 = yy2.clamp(INT_MIN * 1.0, y2[i].item().toFloat());

        w = xx2 - xx1;
        h = yy2 - yy1;

        w = w.clamp(0, INT_MAX);
        h = h.clamp(0, INT_MAX);

        torch::Tensor inter = w * h;
        torch::Tensor rem_areas = area.index_select(0, idx);

        torch::Tensor union_ = (rem_areas - inter) + area[i];
        torch::Tensor Iou = inter * 1.0 / union_;
        torch::Tensor index_small = Iou < overlap;
        auto mask_idx = torch::nonzero(index_small).squeeze();
        idx = idx.index_select(0, mask_idx);//pthon: idx = idx[IoU.le(overlap)]
    }
    return true;
}
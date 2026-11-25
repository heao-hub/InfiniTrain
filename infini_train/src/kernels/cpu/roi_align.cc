#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include "glog/logging.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
namespace infini_train::kernels::cpu {
namespace {
void Run(const float *input, float *output, const float *boxes, int64_t count,
         const std::vector<int64_t> &dims, const std::vector<int64_t> &size,
         float scale, int64_t ratio, bool aligned, bool backward) {

  const auto channels = dims[1], height = dims[2], width = dims[3];
  for (int64_t r = 0; r < count; ++r) {
    const float *box = boxes + r * 5;
    CHECK(std::isfinite(box[0]));
    CHECK_EQ(box[0], std::floor(box[0]));
    const auto batch = static_cast<int64_t>(box[0]);
    CHECK_GE(batch, 0); 
    CHECK_LT(batch, dims[0]);

    for (int i = 1; i < 5; ++i) CHECK(std::isfinite(box[i]));
    const float offset = aligned ? 0.5f : 0.0f;
    const float x0 = box[1] * scale - offset;
    const float y0 = box[2] * scale - offset;
    float rw = box[3] * scale - offset - x0;
    float rh = box[4] * scale - offset - y0;
    if (!aligned) { 
      rw = std::max(rw, 1.0f);
      rh = std::max(rh, 1.0f); 
    }

    CHECK_GT(rw, 0);
    CHECK_GT(rh, 0);
    
    // 计算每个输出格内采样点个数
    const int64_t nx = ratio > 0 ? ratio : static_cast<int64_t>(std::ceil(rw / size[1]));
    const int64_t ny = ratio > 0 ? ratio : static_cast<int64_t>(std::ceil(rh / size[0]));
    
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t py = 0; py < size[0]; ++py) {
        for (int64_t px = 0; px < size[1]; ++px) {
          // [r,c,py,px]  [count, channels, size[0], size[1]]
          const int64_t outidx = ((r * channels + c) * size[0] + py) * size[1] + px;
          float sum = 0;
          for (int64_t iy = 0; iy < ny; ++iy) {
            for (int64_t ix = 0; ix < nx; ++ix) {
              // 计算采样点坐标
              float y = y0 + (py + (iy + 0.5f) / ny) * rh / size[0];
              float x = x0 + (px + (ix + 0.5f) / nx) * rw / size[1];
              if (y < -1 || y > height || x < -1 || x > width) continue;
              
              y = std::max(y, 0.0f); x = std::max(x, 0.0f);
              const auto yl = std::min(static_cast<int64_t>(y), height-1);
              const auto xl = std::min(static_cast<int64_t>(x), width-1);
              const auto yh = std::min(yl+1, height-1), xh = std::min(xl+1, width-1);
              y = std::min(y, static_cast<float>(height-1));
              x = std::min(x, static_cast<float>(width-1));
              
              const float dy = y-yl, dx = x-xl;
              const int64_t base = (batch*channels+c)*height*width;
              // 相邻四个输入位置
              const int64_t index[] = {base+yl*width+xl, base+yl*width+xh, base+yh*width+xl, base+yh*width+xh};
              // 四个插值权重
              const float weight[] = {(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx};

              for (int i=0; i<4; ++i) {
                if (backward) {
                  output[index[i]] += input[outidx] * weight[i] / (nx*ny);
                } else {
                  sum += input[index[i]] * weight[i] / (nx*ny);
                }
              }
            }
          }
          if (!backward) output[outidx] = sum;
        }
      }
    }
  }
}
}  // namespace
std::shared_ptr<Tensor> RoiAlignForward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> boxes,
    std::vector<int64_t> size, float scale, int64_t ratio, bool aligned) {
  CHECK(input->Dtype() == DataType::kFLOAT32); 
  CHECK(boxes->Dtype() == DataType::kFLOAT32);
  
  auto d = input->Dims(); 
  CHECK_GT(d[2],0); 
  CHECK_GT(d[3],0);

  auto result = std::make_shared<Tensor>(std::vector<int64_t>{boxes->Dims()[0],d[1],size[0],size[1]}, input->Dtype(), input->GetDevice());
  auto x = input->Contiguous(); 
  auto b = boxes->Contiguous();
  Run(static_cast<const float *>(x->DataPtr()), static_cast<float *>(result->DataPtr()), static_cast<const float *>(b->DataPtr()), boxes->Dims()[0], d, size, scale, ratio, aligned, false);
  return result;
}
std::shared_ptr<Tensor> RoiAlignBackward(std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> boxes,
    std::vector<int64_t> dims, std::vector<int64_t> size, float scale, int64_t ratio, bool aligned) {
  CHECK(grad->Dtype() == DataType::kFLOAT32); CHECK(boxes->Dtype() == DataType::kFLOAT32);
  auto result = std::make_shared<Tensor>(dims, grad->Dtype(), grad->GetDevice());
  auto *out = static_cast<float *>(result->DataPtr()); 
  std::fill(out, out + result->NumElements(), 0.0f);
  
  auto g = grad->Contiguous(); 
  auto b = boxes->Contiguous();
  Run(static_cast<const float *>(g->DataPtr()), out, static_cast<const float *>(b->DataPtr()), boxes->Dims()[0], dims, size, scale, ratio, aligned, true);
  return result;
}
}  // namespace infini_train::kernels::cpu
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, RoiAlignForward, infini_train::kernels::cpu::RoiAlignForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, RoiAlignBackward, infini_train::kernels::cpu::RoiAlignBackward)

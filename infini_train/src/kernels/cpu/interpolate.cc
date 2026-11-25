#include <algorithm>
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

float SourceCoordinate(int64_t index, int64_t input_size, int64_t output_size,
                       bool align_corners) {
  if (align_corners) {
    return output_size > 1 ? static_cast<float>(index) * (input_size - 1) /
                                 (output_size - 1)
                           : 0.0f;
  }
  return std::max(0.0f, (index + 0.5f) * input_size / output_size - 0.5f);
}

void Resize(
  const float *source, float *target,     // 输入、输出数据指针
  int64_t planes, int64_t ih, int64_t iw, // N*C 输入高、宽
  int64_t oh, int64_t ow,                 // 输出高、宽
  bool backward, bool align_corners) {

  for (int64_t p = 0; p < planes; ++p) {
    for (int64_t y = 0; y < oh; ++y) {
      const float fy = SourceCoordinate(y, ih, oh, align_corners);
      const int64_t y0 = std::min(static_cast<int64_t>(fy), ih - 1);
      const int64_t y1 = std::min(y0 + 1, ih - 1);
      const float dy = fy - y0;
      for (int64_t x = 0; x < ow; ++x) {
        const float fx = SourceCoordinate(x, iw, ow, align_corners);
        const int64_t x0 = std::min(static_cast<int64_t>(fx), iw - 1);
        const int64_t x1 = std::min(x0 + 1, iw - 1);
        const float dx = fx - x0;
        const int64_t indices[] = {p * ih * iw + y0 * iw + x0,
                                   p * ih * iw + y0 * iw + x1,
                                   p * ih * iw + y1 * iw + x0,
                                   p * ih * iw + y1 * iw + x1};
        const float weights[] = {(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx};
        const int64_t out = (p * oh + y) * ow + x;
        if (backward) {
          for (int i = 0; i < 4; ++i) target[indices[i]] += source[out] * weights[i];
        } else {
          target[out] = 0;
          for (int i = 0; i < 4; ++i) target[out] += source[indices[i]] * weights[i];
        }
      }
    }
  }
}
}  // namespace

std::shared_ptr<Tensor> InterpolateBilinear2dForward(
    std::shared_ptr<Tensor> input, std::vector<int64_t> size, bool align_corners) {
  CHECK(input->Dtype() == DataType::kFLOAT32);
  auto d = input->Dims();
  CHECK_GT(d[2], 0); CHECK_GT(d[3], 0);
  // 创建输出tensor [ N, C, target_H, target_W] 
  auto output = std::make_shared<Tensor>(std::vector<int64_t>{d[0], d[1], size[0], size[1]}, input->Dtype(), input->GetDevice());
  auto contiguous = input->Contiguous();
  Resize(static_cast<const float *>(contiguous->DataPtr()), static_cast<float *>(output->DataPtr()), d[0]*d[1], d[2], d[3], size[0], size[1], false, align_corners);
  return output;
}
std::shared_ptr<Tensor> InterpolateBilinear2dBackward(
    std::shared_ptr<Tensor> grad, std::vector<int64_t> dims, bool align_corners) {
  CHECK(grad->Dtype() == DataType::kFLOAT32);
  auto output = std::make_shared<Tensor>(dims, grad->Dtype(), grad->GetDevice());
  auto *data = static_cast<float *>(output->DataPtr());
  std::fill(data, data + output->NumElements(), 0.0f);
  auto contiguous = grad->Contiguous();
  Resize(static_cast<const float *>(contiguous->DataPtr()), data, dims[0]*dims[1], dims[2], dims[3], grad->Dims()[2], grad->Dims()[3], true, align_corners);
  return output;
}
}  // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, InterpolateBilinear2dForward, infini_train::kernels::cpu::InterpolateBilinear2dForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, InterpolateBilinear2dBackward, infini_train::kernels::cpu::InterpolateBilinear2dBackward)

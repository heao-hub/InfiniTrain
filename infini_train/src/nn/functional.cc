#include "infini_train/include/nn/functional.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/device.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/interpolate.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/roi_align.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::function {
std::shared_ptr<Tensor> Tril(const std::shared_ptr<Tensor> &input,
                             int64_t diagonal) {
  return std::make_shared<autograd::Tril>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Triu(const std::shared_ptr<Tensor> &input,
                             int64_t diagonal) {
  return std::make_shared<autograd::Triu>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size) {
  auto ones = std::make_shared<Tensor>(size, DataType::kFLOAT32);
  return init::Ones(ones);
}

std::shared_ptr<Tensor> Reciprocal(const std::shared_ptr<Tensor> &input) {
  return input->Reciprocal();
}

std::shared_ptr<Tensor> Sin(const std::shared_ptr<Tensor> &input) {
  return input->Sin();
}

std::shared_ptr<Tensor> Cos(const std::shared_ptr<Tensor> &input) {
  return input->Cos();
}

std::shared_ptr<Tensor> Tanh(const std::shared_ptr<Tensor> &input) {
  return input->Tanh();
}

std::shared_ptr<Tensor> Pow(const std::shared_ptr<Tensor> &input,
                            float exponent) {
  return input->Pow(exponent);
}

std::shared_ptr<Tensor> Pow(float base, const std::shared_ptr<Tensor> &input) {
  return std::make_shared<autograd::Pow>(base, true)->Apply({input})[0];
}

std::shared_ptr<Tensor> Rsqrt(const std::shared_ptr<Tensor> &input) {
  return input->Rsqrt();
}

std::shared_ptr<Tensor> Mean(const std::shared_ptr<Tensor> &input, int64_t dim,
                             bool keep_dim) {
  return std::make_shared<autograd::Mean>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sum(const std::shared_ptr<Tensor> &input, int64_t dim,
                            bool keep_dim) {
  return std::make_shared<autograd::Sum>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Min(const std::shared_ptr<Tensor> &input, int64_t dim,
                            bool keep_dim) {
  return std::make_shared<autograd::Min>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Max(const std::shared_ptr<Tensor> &input, int64_t dim,
                            bool keep_dim) {
  return std::make_shared<autograd::Max>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Slice(const std::shared_ptr<Tensor> &input,
                              const std::vector<int64_t> &starts,
                              const std::vector<int64_t> &ends,
                              const std::vector<int64_t> &steps) {
  return input->Slice(starts, ends, steps);
}

std::shared_ptr<Tensor> Stack(
    const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
  return std::make_shared<autograd::Stack>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Concat(
    const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
  return std::make_shared<autograd::Concat>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Softmax(const std::shared_ptr<Tensor> &input,
                                int64_t dim) {
  return std::make_shared<autograd::Softmax>(dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sigmoid(const std::shared_ptr<Tensor> &input) {
  return std::make_shared<autograd::Sigmoid>()->Apply({input})[0];
}

std::shared_ptr<Tensor> InterpolateBilinear2d(
    const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &size,
    bool align_corners) {
  return std::make_shared<autograd::InterpolateBilinear>(size, align_corners)
      ->Apply({input})[0];
}

std::shared_ptr<Tensor> RoiAlign(
    const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &boxes,
    const std::vector<int64_t> &size, float scale, int64_t ratio, bool aligned) {
  return std::make_shared<autograd::RoiAlign>(size, scale, ratio, aligned)
      ->Apply({input, boxes})[0];
}

// base[B, T, D]  已占位的文本embedding B个文本样本，每个样本有T个位置， 每个位置是一个D维向量  
// replacement[N, Q, D]  图像单元对应的embedding向量 N个图像单元 每个图像单元内Q个视觉向量 
// spans[N, 3]  替换位置 第k行写作（b, s, e）表示用第k个图像替换第b个文本样本中的[start, end)位置
std::shared_ptr<Tensor> ReplaceSequenceSpans(
    const std::shared_ptr<Tensor> &base,
    const std::shared_ptr<Tensor> &replacements,
    const std::shared_ptr<Tensor> &spans) {

  const auto &dims = base->Dims();
  const auto &src = replacements->Dims();
  CHECK_EQ(dims.size(), 3);
  CHECK_EQ(src.size(), 3);
  CHECK_EQ(src[2], dims[2]);
  CHECK(base->Dtype() == replacements->Dtype());
  CHECK(base->GetDevice() == replacements->GetDevice());
  CHECK(spans->Dtype() == DataType::kINT64);
  CHECK((spans->Dims() == std::vector<int64_t>{src[0], 3}));
  if (src[0] == 0) return base;
  CHECK_GT(src[1], 0);

  auto host = spans->GetDevice().type() == Device::DeviceType::kCPU
                  ? spans
                  : std::make_shared<Tensor>(spans->To(Device()));
  const auto *bounds = static_cast<const int64_t *>(host->DataPtr());
  // [B, T, D] --> [ B*T, D]
  const int64_t rows = dims[0] * dims[1];
  auto text = base->View({rows, dims[2]});

  std::vector<std::shared_ptr<Tensor>> parts;
  int64_t cursor = 0;
  for (int64_t i = 0; i < src[0]; ++i) {
    const auto *span = bounds + 3 * i;
    CHECK_GE(span[0], 0);
    CHECK_LT(span[0], dims[0]);
    CHECK_GE(span[1], 0);
    CHECK_LE(span[2], dims[1]);
    CHECK_GT(span[2], span[1]);
    CHECK_EQ(span[2] - span[1], src[1]);
    const int64_t begin = span[0] * dims[1] + span[1];
    CHECK_GE(begin, cursor) << "Sequence spans must be sorted and disjoint";
    
    if (cursor < begin) parts.push_back(text->Slice(0, cursor, begin));
    parts.push_back(replacements->Slice(0, i, i + 1)->View({src[1], src[2]}));
    cursor = begin + src[1];
  }
  if (cursor < rows) parts.push_back(text->Slice(0, cursor, rows));
  auto fused = parts.size() == 1 ? parts.front() : Concat(parts, 0);
  return fused->View(dims);
}


}  // namespace infini_train::nn::function

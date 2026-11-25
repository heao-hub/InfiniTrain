#include "infini_train/include/autograd/interpolate.h"

#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
InterpolateBilinear::InterpolateBilinear(std::vector<int64_t> size, bool align_corners)
    : Function("InterpolateBilinear2d"), size_(std::move(size)), align_corners_(align_corners) {}

std::vector<std::shared_ptr<Tensor>> InterpolateBilinear::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(inputs[0]->Dims().size(), 4);
  CHECK_EQ(size_.size(), 2);
  CHECK_GT(size_[0], 0);
  CHECK_GT(size_[1], 0);
  input_dims_ = inputs[0]->Dims();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {inputs[0]->GetDevice().type(), "InterpolateBilinear2dForward"},
      inputs[0], size_, align_corners_)};
}

void InterpolateBilinear::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &,
    const std::vector<std::shared_ptr<Tensor>> &) {}

std::vector<std::shared_ptr<Tensor>> InterpolateBilinear::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad) {
  CHECK_EQ(grad.size(), 1);
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {grad[0]->GetDevice().type(), "InterpolateBilinear2dBackward"},
      grad[0], input_dims_, align_corners_)};
}
}  // namespace infini_train::autograd

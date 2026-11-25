#include "infini_train/include/autograd/roi_align.h"

#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
RoiAlign::RoiAlign(std::vector<int64_t> size, float scale, int64_t ratio,
                   bool aligned)
    : Function("RoiAlign"), size_(std::move(size)), scale_(scale),
      ratio_(ratio), aligned_(aligned) {}

std::vector<std::shared_ptr<Tensor>> RoiAlign::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(inputs[0]->Dims().size(), 4);
  CHECK_EQ(inputs[1]->Dims().size(), 2);
  CHECK_EQ(inputs[1]->Dims()[1], 5);
  CHECK_EQ(size_.size(), 2);
  CHECK_GT(size_[0], 0);
  CHECK_GT(size_[1], 0);
  CHECK_GT(scale_, 0);
  CHECK(inputs[0]->GetDevice() == inputs[1]->GetDevice());
  dims_ = inputs[0]->Dims();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {inputs[0]->GetDevice().type(), "RoiAlignForward"}, inputs[0], inputs[1],
      size_, scale_, ratio_, aligned_)};
}

void RoiAlign::SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                            const std::vector<std::shared_ptr<Tensor>> &) {
  ctx_.SaveForBackward({inputs[1]});
}

std::vector<std::shared_ptr<Tensor>> RoiAlign::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad) {
  auto boxes = ctx_.GetSavedTensors()[0];
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
              {grad[0]->GetDevice().type(), "RoiAlignBackward"}, grad[0],
              boxes, dims_, size_, scale_, ratio_, aligned_),
          nullptr};
}
}  // namespace infini_train::autograd

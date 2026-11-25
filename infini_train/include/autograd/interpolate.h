#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
namespace autograd {
class InterpolateBilinear final : public Function {
 public:
  InterpolateBilinear(std::vector<int64_t> size, bool align_corners);
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override;
  void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                    const std::vector<std::shared_ptr<Tensor>> &outputs) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad) override;

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> input_dims_;
  bool align_corners_;
};
}  // namespace autograd
}  // namespace infini_train

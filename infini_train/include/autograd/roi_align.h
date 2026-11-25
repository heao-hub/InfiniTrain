#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
namespace autograd {
class RoiAlign final : public Function {
 public:
  RoiAlign(std::vector<int64_t> size, float scale, int64_t ratio, bool aligned);
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override;
  void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                    const std::vector<std::shared_ptr<Tensor>> &outputs) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad) override;

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> dims_;
  float scale_;
  int64_t ratio_;
  bool aligned_;
};
}  // namespace autograd
}  // namespace infini_train

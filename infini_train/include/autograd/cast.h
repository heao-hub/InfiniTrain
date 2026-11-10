#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/datatype.h"

namespace infini_train::autograd {

class Cast : public Function {
 public:
  explicit Cast(DataType dtype) : Function("CastFunction"), dtype_(dtype) {}
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override;
  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  DataType dtype_;
  DataType input_dtype_;
};

}  // namespace infini_train::autograd

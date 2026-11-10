#include "infini_train/include/autograd/cast.h"

#include "glog/logging.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Cast::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
  CHECK_EQ(inputs.size(), 1);
  input_dtype_ = inputs[0]->Dtype();
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {inputs[0]->GetDevice().type(), "Cast"}, inputs[0], dtype_)};
}

std::vector<std::shared_ptr<Tensor>> Cast::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  CHECK_EQ(grad_outputs.size(), 1);
  return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
      {grad_outputs[0]->GetDevice().type(), "Cast"}, grad_outputs[0],
      input_dtype_)};
}

}  // namespace infini_train::autograd

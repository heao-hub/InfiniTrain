#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/nn/modules/transformer/cross_attention.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

namespace {
using namespace infini_train;

std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &shape,
                                   const std::vector<float> &values) {
  auto tensor = std::make_shared<Tensor>(shape, DataType::kFLOAT32);
  std::copy(values.begin(), values.end(), static_cast<float *>(tensor->DataPtr()));
  return tensor;
}

TEST(CrossAttentionCPU, IndependentInputsAndValueGradient) {
  auto attention = std::make_shared<nn::CrossAttention>(2, 1);
  for (auto &parameter : attention->Parameters()) {
    auto *data = static_cast<float *>(parameter->DataPtr());
    std::fill(data, data + parameter->NumElements(), 0.0f);
  }
  // Zero Q and K yield uniform attention; V and output projections are identity.
  for (const auto *name : {"v_proj", "out_proj"}) {
    auto weight = attention->module(name).parameter("weight");
    auto *data = static_cast<float *>(weight->DataPtr());
    data[0] = data[3] = 1.0f;
  }
  auto query = MakeTensor({1, 1, 2}, {7, 8});
  auto key = MakeTensor({1, 3, 2}, {0, 0, 0, 0, 0, 0});
  auto value = MakeTensor({1, 3, 2}, {1, 2, 3, 4, 8, 9})->RequiresGrad();
  auto output = (*attention)({query, key, value})[0];
  test::ExpectTensorNear(output, {4, 5}, 1e-5f);
  output->Sum(2)->Sum(1)->Sum(0)->Backward();
  test::ExpectTensorNear(value->grad(),
                         {1.0f / 3, 1.0f / 3, 1.0f / 3, 1.0f / 3,
                          1.0f / 3, 1.0f / 3},
                         1e-5f);
}

TEST(CrossAttentionCPU, NonUniformWeightsAndMask) {
  auto attention = std::make_shared<nn::CrossAttention>(2, 1);
  for (auto &parameter : attention->Parameters()) {
    auto *data = static_cast<float *>(parameter->DataPtr());
    std::fill(data, data + parameter->NumElements(), 0.0f);
  }
  for (const auto *name : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
    auto weight = attention->module(name).parameter("weight");
    auto *data = static_cast<float *>(weight->DataPtr());
    data[0] = data[3] = 1.0f;
  }
  // Scores are 0 and log(3), so softmax weights are 1/4 and 3/4.
  auto query = MakeTensor({1, 1, 2}, {1, 0});
  auto key = MakeTensor({1, 2, 2}, {0, 0, std::log(3.0f) * std::sqrt(2.0f), 0});
  auto value = MakeTensor({1, 2, 2}, {2, 4, 6, 8})->RequiresGrad();
  auto output = (*attention)({query, key, value})[0];
  test::ExpectTensorNear(output, {5, 7}, 1e-5f);
  output->Sum(2)->Sum(1)->Sum(0)->Backward();
  test::ExpectTensorNear(value->grad(), {0.25f, 0.25f, 0.75f, 0.75f}, 1e-5f);

  auto mask = MakeTensor({1, 1, 1, 2}, {0, -std::numeric_limits<float>::infinity()});
  auto masked = (*attention)({query, key, value, mask})[0];
  test::ExpectTensorNear(masked, {2, 4}, 1e-5f);
}
}  // namespace

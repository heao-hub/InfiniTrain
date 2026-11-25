#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "example/fm9g/resampler.h"
#include "infini_train/include/tensor.h"

TEST(FM9GResamplerCPU, ForwardAndBackward) {
  using namespace infini_train;
  auto model = std::make_shared<fm9g::Resampler>(4, 8, 2, 4);
  auto features = std::make_shared<Tensor>(std::vector<int64_t>{1,4,4}, DataType::kFLOAT32);
  auto *x = static_cast<float *>(features->DataPtr());
  for (int i=0; i<16; ++i) x[i] = std::sin(static_cast<float>(i));
  features->RequiresGrad();
  auto sizes = std::make_shared<Tensor>(std::vector<int64_t>{1,2}, DataType::kINT64);
  auto *p = static_cast<int64_t *>(sizes->DataPtr()); p[0]=2; p[1]=2;
  auto output = (*model)({features, sizes})[0];
  ASSERT_EQ(output->Dims(), (std::vector<int64_t>{1,4,8}));
  output->Sum(2)->Sum(1)->Sum(0)->Backward();
  ASSERT_NE(features->grad(), nullptr);
  auto *grad = static_cast<float *>(features->grad()->DataPtr());
  for (int i=0; i<16; ++i) EXPECT_TRUE(std::isfinite(grad[i]));
  auto proj_grad = model->parameter("proj")->grad();
  ASSERT_NE(proj_grad, nullptr);
  ASSERT_EQ(proj_grad->Dims(), (std::vector<int64_t>{8, 8}));
  const auto *proj_data = static_cast<const float *>(proj_grad->DataPtr());
  for (size_t i = 0; i < proj_grad->NumElements(); ++i) {
    EXPECT_TRUE(std::isfinite(proj_data[i]));
  }
}

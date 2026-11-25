#include <algorithm>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

TEST(RoiAlignCPU, ForwardAndInputGradient) {
  using namespace infini_train;
  auto x = std::make_shared<Tensor>(std::vector<int64_t>{1,1,2,2}, DataType::kFLOAT32);
  const std::vector<float> values{1,2,3,4};
  std::copy(values.begin(), values.end(), static_cast<float *>(x->DataPtr()));
  x->RequiresGrad();
  auto boxes = std::make_shared<Tensor>(std::vector<int64_t>{1,5}, DataType::kFLOAT32);
  const std::vector<float> roi{0,0,0,1,1};
  std::copy(roi.begin(), roi.end(), static_cast<float *>(boxes->DataPtr()));
  auto y = nn::function::RoiAlign(x, boxes, {1,1}, 1.0f, 1, false);
  test::ExpectTensorNear(y, {2.5f}, 1e-6f);
  y->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
  test::ExpectTensorNear(x->grad(), {0.25f,0.25f,0.25f,0.25f}, 1e-6f);
}

TEST(RoiAlignCPU, OverlappingRegionsAccumulateGradients) {
  using namespace infini_train;
  auto x = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32);
  const std::vector<float> values{1, 2, 3, 4};
  std::copy(values.begin(), values.end(), static_cast<float *>(x->DataPtr()));
  x->RequiresGrad();
  auto boxes = std::make_shared<Tensor>(std::vector<int64_t>{2, 5}, DataType::kFLOAT32);
  const std::vector<float> rois{0, 0, 0, 1, 1, 0, 0, 0, 1, 1};
  std::copy(rois.begin(), rois.end(), static_cast<float *>(boxes->DataPtr()));
  auto y = nn::function::RoiAlign(x, boxes, {1, 1}, 1.0f, 1, false);
  test::ExpectTensorNear(y, {2.5f, 2.5f}, 1e-6f);
  y->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
  test::ExpectTensorNear(x->grad(), {0.5f, 0.5f, 0.5f, 0.5f}, 1e-6f);
}

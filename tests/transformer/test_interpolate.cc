#include <algorithm>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

TEST(InterpolateCPU, BilinearForwardAndBackward) {
  using namespace infini_train;
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32);
  const std::vector<float> data{0, 4, 8, 12};
  std::copy(data.begin(), data.end(), static_cast<float *>(input->DataPtr()));
  input->RequiresGrad();
  auto output = nn::function::InterpolateBilinear2d(input, {3, 3});
  test::ExpectTensorNear(output, {0, 2, 4, 4, 6, 8, 8, 10, 12}, 1e-5f);
  output->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
  test::ExpectTensorNear(input->grad(), {2.25f, 2.25f, 2.25f, 2.25f}, 1e-5f);
}

TEST(InterpolateCPU, AlignCornersControlsForwardAndBackward) {
  using namespace infini_train;
  for (bool align_corners : {false, true}) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 2},
                                          DataType::kFLOAT32);
    auto *data = static_cast<float *>(input->DataPtr());
    data[0] = 0;
    data[1] = 4;
    input->RequiresGrad();
    auto output = nn::function::InterpolateBilinear2d(input, {1, 4},
                                                       align_corners);
    if (align_corners) {
      test::ExpectTensorNear(output, {0, 4.0f / 3, 8.0f / 3, 4}, 1e-5f);
    } else {
      test::ExpectTensorNear(output, {0, 1, 3, 4}, 1e-5f);
    }
    output->Slice(3, 1, 2)->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    if (align_corners) {
      test::ExpectTensorNear(input->grad(), {2.0f / 3, 1.0f / 3}, 1e-5f);
    } else {
      test::ExpectTensorNear(input->grad(), {0.75f, 0.25f}, 1e-5f);
    }
  }
}

TEST(InterpolateCPU, AlignCornersWithSingleOutput) {
  using namespace infini_train;
  for (bool align_corners : {false, true}) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2},
                                          DataType::kFLOAT32);
    const std::vector<float> data{0, 4, 8, 12};
    std::copy(data.begin(), data.end(), static_cast<float *>(input->DataPtr()));
    input->RequiresGrad();
    auto output = nn::function::InterpolateBilinear2d(input, {1, 1},
                                                       align_corners);
    test::ExpectTensorNear(output, {align_corners ? 0.0f : 6.0f}, 1e-5f);
    output->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    if (align_corners) {
      test::ExpectTensorNear(input->grad(), {1, 0, 0, 0}, 1e-5f);
    } else {
      test::ExpectTensorNear(input->grad(), {0.25f, 0.25f, 0.25f, 0.25f},
                             1e-5f);
    }
  }
}

TEST(InterpolateCPU, DownsampleRespectsCoordinateMode) {
  using namespace infini_train;
  for (bool align_corners : {false, true}) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 4}, DataType::kFLOAT32);
    const std::vector<float> data{0, 2, 4, 6};
    std::copy(data.begin(), data.end(), static_cast<float *>(input->DataPtr()));
    input->RequiresGrad();
    auto output = nn::function::InterpolateBilinear2d(input, {1, 2}, align_corners);
    if (align_corners) {
      test::ExpectTensorNear(output, {0, 6}, 1e-5f);
    } else {
      test::ExpectTensorNear(output, {1, 5}, 1e-5f);
    }
    output->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    if (align_corners) {
      test::ExpectTensorNear(input->grad(), {1, 0, 0, 1}, 1e-5f);
    } else {
      test::ExpectTensorNear(input->grad(), {0.5f, 0.5f, 0.5f, 0.5f}, 1e-5f);
    }
  }
}

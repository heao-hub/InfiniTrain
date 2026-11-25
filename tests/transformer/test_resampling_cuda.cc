#include <algorithm>
#include <memory>
#include <vector>

#ifdef USE_CUDA
#include <cuda.h>
#endif

#include "gtest/gtest.h"
#include "example/fm9g/resampler.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/transformer/cross_attention.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

namespace {
using namespace infini_train;
class ResamplingDeviceTest : public test::InfiniTrainTest {
 protected:
  void SetUp() override {
#ifdef USE_CUDA
    if (GetDevice().type() == Device::DeviceType::kCUDA) {
      ASSERT_EQ(cuInit(0), CUDA_SUCCESS) << "CUDA driver initialization failed";
    }
#endif
  }
};

std::shared_ptr<Tensor> FloatTensor(const std::vector<int64_t> &dims,
                                    const std::vector<float> &values) {
  auto t = std::make_shared<Tensor>(dims, DataType::kFLOAT32);
  std::copy(values.begin(), values.end(), static_cast<float *>(t->DataPtr()));
  return t;
}

std::vector<float> Values(const std::shared_ptr<Tensor> &tensor) {
  auto cpu = tensor->To(Device());
  const auto *data = static_cast<const float *>(cpu.DataPtr());
  return {data, data + cpu.NumElements()};
}

TEST_P(ResamplingDeviceTest, InterpolateMatchesCpuForwardAndBackward) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) GTEST_SKIP() << "CUDA only";
  REQUIRE_MIN_DEVICES(1);
  for (bool aligned : {false, true}) {
    auto cpu = FloatTensor({1, 1, 2, 3}, {1, 2, 3, 4, 5, 6})->RequiresGrad();
    auto gpu = std::make_shared<Tensor>(cpu->Detach()->To(GetDevice()))->RequiresGrad();
    auto a = nn::function::InterpolateBilinear2d(cpu, {5, 4}, aligned);
    auto b = nn::function::InterpolateBilinear2d(gpu, {5, 4}, aligned);
    test::ExpectTensorNear(b, Values(a), 1e-5f);
    a->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    b->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    test::ExpectTensorNear(gpu->grad(), Values(cpu->grad()), 1e-4f);
  }
}

TEST_P(ResamplingDeviceTest, RoiAlignMatchesCpuWithOverlappingBoxes) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) GTEST_SKIP() << "CUDA only";
  REQUIRE_MIN_DEVICES(1);
  for (bool aligned : {false, true}) for (int64_t ratio : {-1, 2}) {
    auto cpu = FloatTensor({1, 1, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8})->RequiresGrad();
    auto gpu = std::make_shared<Tensor>(cpu->Detach()->To(GetDevice()))->RequiresGrad();
    auto boxes = FloatTensor({2, 5}, {0, 0, 0, 2, 2, 0, 0.5f, 0.5f, 2.5f, 2.5f});
    auto gpu_boxes = std::make_shared<Tensor>(boxes->To(GetDevice()));
    auto a = nn::function::RoiAlign(cpu, boxes, {2, 2}, 1.0f, ratio, aligned);
    auto b = nn::function::RoiAlign(gpu, gpu_boxes, {2, 2}, 1.0f, ratio, aligned);
    test::ExpectTensorNear(b, Values(a), 1e-5f);
    a->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    b->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    test::ExpectTensorNear(gpu->grad(), Values(cpu->grad()), 1e-4f);
  }
}

TEST_P(ResamplingDeviceTest, LowPrecisionForwardAndBackward) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) GTEST_SKIP() << "CUDA only";
  REQUIRE_MIN_DEVICES(1);
  for (auto dtype : {DataType::kFLOAT16, DataType::kBFLOAT16}) {
    auto cpu = FloatTensor({1, 1, 2, 2}, {1, 2, 3, 4});
    auto device_fp32 = std::make_shared<Tensor>(cpu->To(GetDevice()));
    auto device_input = std::make_shared<Tensor>(device_fp32->To(dtype))->RequiresGrad();
    auto resized = nn::function::InterpolateBilinear2d(device_input, {3, 3}, false);
    ASSERT_EQ(resized->Dtype(), dtype);
    auto resized_fp32 = std::make_shared<Tensor>(resized->To(DataType::kFLOAT32));
    test::ExpectTensorNear(resized_fp32, {1, 1.5f, 2, 2, 2.5f, 3, 3, 3.5f, 4}, 0.02f);
    resized_fp32->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    ASSERT_NE(device_input->grad(), nullptr);
    auto grad_fp32 = std::make_shared<Tensor>(device_input->grad()->To(DataType::kFLOAT32));
    test::ExpectTensorNear(grad_fp32, {2.25f, 2.25f, 2.25f, 2.25f}, 0.05f);

    auto roi_input = std::make_shared<Tensor>(device_fp32->To(dtype))->RequiresGrad();
    auto boxes_cpu = FloatTensor({1, 5}, {0, 0, 0, 1, 1});
    auto boxes = std::make_shared<Tensor>(boxes_cpu->To(GetDevice()));
    auto roi = nn::function::RoiAlign(roi_input, boxes, {1, 1}, 1.0f, 1, false);
    ASSERT_EQ(roi->Dtype(), dtype);
    auto roi_fp32 = std::make_shared<Tensor>(roi->To(DataType::kFLOAT32));
    test::ExpectTensorNear(roi_fp32, {2.5f}, 0.02f);
    roi_fp32->Sum(3)->Sum(2)->Sum(1)->Sum(0)->Backward();
    ASSERT_NE(roi_input->grad(), nullptr);
    auto roi_grad = std::make_shared<Tensor>(roi_input->grad()->To(DataType::kFLOAT32));
    test::ExpectTensorNear(roi_grad, {0.25f, 0.25f, 0.25f, 0.25f}, 0.05f);
  }
}

TEST_P(ResamplingDeviceTest, CrossAttentionIndependentInputs) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) GTEST_SKIP() << "CUDA only";
  REQUIRE_MIN_DEVICES(1);
  auto attention = std::make_shared<nn::CrossAttention>(2, 1);
  for (auto &parameter : attention->Parameters()) {
    auto *data = static_cast<float *>(parameter->DataPtr());
    std::fill(data, data + parameter->NumElements(), 0.0f);
  }
  for (const auto *name : {"v_proj", "out_proj"}) {
    auto weight = attention->module(name).parameter("weight");
    auto *data = static_cast<float *>(weight->DataPtr());
    data[0] = data[3] = 1.0f;
  }
  attention->To(GetDevice());
  auto q = std::make_shared<Tensor>(FloatTensor({1, 1, 2}, {7, 8})->To(GetDevice()));
  auto k = std::make_shared<Tensor>(FloatTensor({1, 3, 2}, {0, 0, 0, 0, 0, 0})->To(GetDevice()));
  auto v = std::make_shared<Tensor>(FloatTensor({1, 3, 2}, {1, 2, 3, 4, 8, 9})->To(GetDevice()))->RequiresGrad();
  auto output = (*attention)({q, k, v})[0];
  test::ExpectTensorNear(output, {4, 5}, 1e-5f);
  output->Sum(2)->Sum(1)->Sum(0)->Backward();
  ASSERT_NE(v->grad(), nullptr);
  test::ExpectTensorNear(v->grad(), {1.0f/3, 1.0f/3, 1.0f/3, 1.0f/3, 1.0f/3, 1.0f/3}, 1e-4f);
}

TEST_P(ResamplingDeviceTest, ResamplerMatchesCpuForwardAndBackward) {
  if (GetDevice().type() != Device::DeviceType::kCUDA) GTEST_SKIP() << "CUDA only";
  REQUIRE_MIN_DEVICES(1);
  auto cpu = std::make_shared<fm9g::Resampler>(4, 8, 2, 4);
  auto gpu = std::make_shared<fm9g::Resampler>(4, 8, 2, 4);
  gpu->LoadStateDict(cpu->StateDict());
  gpu->To(GetDevice());
  auto x = FloatTensor({1, 4, 4}, {0.1f,0.2f,0.3f,0.4f, 0.5f,0.6f,0.7f,0.8f,
                                   0.9f,1.0f,1.1f,1.2f, 1.3f,1.4f,1.5f,1.6f})->RequiresGrad();
  auto gx = std::make_shared<Tensor>(x->Detach()->To(GetDevice()))->RequiresGrad();
  auto sizes = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kINT64);
  auto *p = static_cast<int64_t *>(sizes->DataPtr()); p[0] = 2; p[1] = 2;
  auto a = (*cpu)({x, sizes})[0];
  auto b = (*gpu)({gx, sizes})[0];
  ASSERT_EQ(b->Dims(), (std::vector<int64_t>{1, 4, 8}));
  test::ExpectTensorNear(b, Values(a), 2e-3f);
  a->Sum(2)->Sum(1)->Sum(0)->Backward();
  b->Sum(2)->Sum(1)->Sum(0)->Backward();
  ASSERT_NE(x->grad(), nullptr);
  ASSERT_NE(gx->grad(), nullptr);
  test::ExpectTensorNear(gx->grad(), Values(x->grad()), 5e-3f);
  ASSERT_NE(cpu->parameter("proj")->grad(), nullptr);
  ASSERT_NE(gpu->parameter("proj")->grad(), nullptr);
  test::ExpectTensorNear(gpu->parameter("proj")->grad(), Values(cpu->parameter("proj")->grad()), 5e-3f);
}
INFINI_TRAIN_REGISTER_TEST(ResamplingDeviceTest);
}  // namespace

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class FM9GAttentionPrerequisitesTest : public test::InfiniTrainTest {};

TEST_P(FM9GAttentionPrerequisitesTest, SliceConcatTransposeViewForwardBackward) {
  constexpr int B = 2, T = 3, H = 40, D = 96;
  std::vector<float> input(B * T * H * D);
  std::iota(input.begin(), input.end(), 0.0f);
  auto x = std::make_shared<Tensor>(input.data(),
      std::vector<int64_t>{B, T, H, D}, DataType::kFLOAT32, GetDevice());
  x->RequiresGrad();

  auto left = x->Slice(3, 0, 64);
  auto right = x->Slice(3, 64, 96);
  auto joined = nn::function::Concat({left, right}, 3);
  ASSERT_EQ(joined->Dims(), (std::vector<int64_t>{B, T, H, D}));
  test::ExpectTensorFloatEqual(joined, input);

  auto output = joined->Transpose(1, 2)->Contiguous()->View({B, H, T * D});
  ASSERT_EQ(output->Dims(), (std::vector<int64_t>{B, H, T * D}));
  std::vector<float> expected(input.size()), upstream(input.size()), expected_grad(input.size());
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
          const int src = ((b * T + t) * H + h) * D + d;
          const int dst = ((b * H + h) * T + t) * D + d;
          expected[dst] = input[src];
          upstream[dst] = static_cast<float>(dst + 1);
          expected_grad[src] = upstream[dst];
        }
      }
    }
  }
  test::ExpectTensorFloatEqual(output, expected);
  auto grad = std::make_shared<Tensor>(upstream.data(), output->Dims(),
                                       DataType::kFLOAT32, GetDevice());
  output->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  test::ExpectTensorFloatEqual(x->grad(), expected_grad);
}

TEST_P(FM9GAttentionPrerequisitesTest, SharedKeyPositionGradientAccumulates) {
  constexpr int B = 2, H = 40, T = 3, D = 32;
  std::vector<float> input(B * T * D);
  std::iota(input.begin(), input.end(), 0.0f);
  auto x = std::make_shared<Tensor>(input.data(),
      std::vector<int64_t>{B, 1, T, D}, DataType::kFLOAT32, GetDevice());
  x->RequiresGrad();
  auto output = x->RepeatInterleave(H, 1);
  ASSERT_EQ(output->Dims(), (std::vector<int64_t>{B, H, T, D}));

  std::vector<float> expected(B * H * T * D), upstream(expected.size());
  std::vector<float> expected_grad(input.size(), 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
          const int src = (b * T + t) * D + d;
          const int dst = ((b * H + h) * T + t) * D + d;
          expected[dst] = input[src];
          upstream[dst] = static_cast<float>((h + 1) + (t * D + d) % 7);
          expected_grad[src] += upstream[dst];
        }
      }
    }
  }
  test::ExpectTensorFloatEqual(output, expected);
  auto grad = std::make_shared<Tensor>(upstream.data(), output->Dims(),
                                       DataType::kFLOAT32, GetDevice());
  output->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  test::ExpectTensorFloatEqual(x->grad(), expected_grad);
}

TEST_P(FM9GAttentionPrerequisitesTest, BF16CastToFP32PreservesGradient) {
  const std::vector<float> values{1.0f, 2.0f, 3.0f};
  auto source = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{3},
                                         DataType::kFLOAT32, GetDevice());
  auto x = std::make_shared<Tensor>(source->To(DataType::kBFLOAT16));
  x->RequiresGrad();
  auto output = std::make_shared<Tensor>(x->To(DataType::kFLOAT32));
  test::ExpectTensorFloatEqual(output, values);
  ASSERT_NE(output->grad_fn(), nullptr)
      << "BF16 -> FP32 cast did not connect the autograd graph";
  auto grad = std::make_shared<Tensor>(values.data(), output->Dims(),
                                       DataType::kFLOAT32, GetDevice());
  output->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  auto actual = std::make_shared<Tensor>(x->grad()->To(DataType::kFLOAT32));
  test::ExpectTensorFloatEqual(actual, values);
}

TEST_P(FM9GAttentionPrerequisitesTest, BF16FP32RoundTripPreservesGradient) {
  const std::vector<float> values{1.0f, 2.0f, 3.0f};
  auto source = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{3},
                                         DataType::kFLOAT32, GetDevice());
  auto x = std::make_shared<Tensor>(source->To(DataType::kBFLOAT16));
  x->RequiresGrad();
  auto fp32 = std::make_shared<Tensor>(x->To(DataType::kFLOAT32));
  auto scaled = fp32->Mul(2.0f);
  auto output = std::make_shared<Tensor>(scaled->To(DataType::kBFLOAT16));
  auto output_fp32 = std::make_shared<Tensor>(output->To(DataType::kFLOAT32));
  test::ExpectTensorFloatEqual(output_fp32, {2.0f, 4.0f, 6.0f});
  ASSERT_NE(fp32->grad_fn(), nullptr) << "BF16 -> FP32 cast disconnected";
  ASSERT_NE(output->grad_fn(), nullptr) << "FP32 -> BF16 cast disconnected";
  auto upstream_fp32 = std::make_shared<Tensor>(values.data(), output->Dims(),
                                                DataType::kFLOAT32, GetDevice());
  auto grad = std::make_shared<Tensor>(upstream_fp32->To(DataType::kBFLOAT16));
  output->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  auto actual = std::make_shared<Tensor>(x->grad()->To(DataType::kFLOAT32));
  test::ExpectTensorFloatEqual(actual, {2.0f, 4.0f, 6.0f});
}

TEST_P(FM9GAttentionPrerequisitesTest, FP32CastToBF16PreservesGradient) {
  const std::vector<float> values{1.0f, 2.0f, 3.0f};
  auto x = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{3},
                                    DataType::kFLOAT32, GetDevice());
  x->RequiresGrad();
  auto output = std::make_shared<Tensor>(x->To(DataType::kBFLOAT16));
  ASSERT_NE(output->grad_fn(), nullptr);
  auto upstream = std::make_shared<Tensor>(values.data(), x->Dims(),
                                           DataType::kFLOAT32, GetDevice());
  auto grad = std::make_shared<Tensor>(upstream->To(DataType::kBFLOAT16));
  output->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  EXPECT_EQ(x->grad()->Dtype(), DataType::kFLOAT32);
  test::ExpectTensorFloatEqual(x->grad(), values);
}

TEST_P(FM9GAttentionPrerequisitesTest, SameDtypePreservesGraph) {
  const std::vector<float> values{1.0f, 2.0f, 3.0f};
  auto x = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{3},
                                    DataType::kFLOAT32, GetDevice());
  x->RequiresGrad();
  auto intermediate = x->Mul(2.0f);
  auto output = std::make_shared<Tensor>(intermediate->To(DataType::kFLOAT32));
  ASSERT_NE(output->grad_fn(), nullptr);
  auto grad = std::make_shared<Tensor>(values.data(), output->Dims(),
                                       DataType::kFLOAT32, GetDevice());
  output->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  test::ExpectTensorFloatEqual(x->grad(), {2.0f, 4.0f, 6.0f});
}

TEST_P(FM9GAttentionPrerequisitesTest, NonGradAndIntegerConversions) {
  auto x = std::make_shared<Tensor>(std::vector<int64_t>{3},
                                    DataType::kFLOAT32, GetDevice());
  x->Fill(1.0f);
  auto frozen = std::make_shared<Tensor>(x->To(DataType::kBFLOAT16));
  EXPECT_FALSE(frozen->requires_grad());
  EXPECT_EQ(frozen->grad_fn(), nullptr);
  x->RequiresGrad();
  auto integer = std::make_shared<Tensor>(x->To(DataType::kINT32));
  EXPECT_FALSE(integer->requires_grad());
  EXPECT_EQ(integer->grad_fn(), nullptr);
  {
    autograd::NoGradGuard guard;
    auto converted = std::make_shared<Tensor>(x->To(DataType::kBFLOAT16));
    EXPECT_EQ(converted->grad_fn(), nullptr);
  }
}

TEST_P(FM9GAttentionPrerequisitesTest, ModuleConversionKeepsLeafParameters) {
  auto layer = std::make_shared<nn::Linear>(2, 3, false, GetDevice());
  layer->To(DataType::kBFLOAT16);
  auto weight = layer->parameter(nn::Linear::kParamWeightName);
  ASSERT_NE(weight, nullptr);
  EXPECT_EQ(weight->Dtype(), DataType::kBFLOAT16);
  EXPECT_TRUE(weight->requires_grad());
  EXPECT_TRUE(weight->is_leaf());
  EXPECT_EQ(weight->grad_fn(), nullptr);
}

INFINI_TRAIN_REGISTER_TEST(FM9GAttentionPrerequisitesTest);

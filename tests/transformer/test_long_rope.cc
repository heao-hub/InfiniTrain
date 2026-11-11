#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/nn/modules/transformer/long_rope.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class LongRoPETest : public test::InfiniTrainTest {};

TEST_P(LongRoPETest, BatchPositionsForwardAndBackward) {
  LongRoPEConfig config{4, 100.0f, 8, 8, {1.0f, 2.0f}, {3.0f, 4.0f}};
  LongRoPE rope(config);
  const std::vector<float> q_values{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  const std::vector<float> k_values{1, 2, 3, 4, 5, 6, 7, 8,
                                    9, 10, 11, 12, 13, 14, 15, 16};
  const std::vector<float> ids_float{0, 1, 1, 0};
  auto q = std::make_shared<Tensor>(q_values.data(), std::vector<int64_t>{2, 2, 2, 4},
                                     DataType::kFLOAT32, GetDevice());
  auto k = std::make_shared<Tensor>(k_values.data(), std::vector<int64_t>{2, 2, 1, 4},
                                     DataType::kFLOAT32, GetDevice());
  auto ids = std::make_shared<Tensor>(ids_float.data(), std::vector<int64_t>{2, 2},
                                       DataType::kFLOAT32, GetDevice());
  auto pos = std::make_shared<Tensor>(ids->To(DataType::kINT64));
  q->RequiresGrad(); k->RequiresGrad();
  auto [qo, ko] = rope.Forward(q, k, pos);
  EXPECT_EQ(qo->Dims(), q->Dims()); EXPECT_EQ(ko->Dims(), k->Dims());

  auto check = [&](const std::shared_ptr<Tensor> &input,
                   const std::shared_ptr<Tensor> &output,
                   const std::vector<float> &values, int heads) {
    std::vector<float> expected(values.size()), weights(values.size()), gradient(values.size());
    for (int b = 0; b < 2; ++b) for (int t = 0; t < 2; ++t)
      for (int h = 0; h < heads; ++h) for (int j = 0; j < 2; ++j) {
        int left = ((b * 2 + t) * heads + h) * 4 + j;
        int right = left + 2;
        float p = ids_float[b * 2 + t];
        float angle = p / (config.short_factors[j] * std::pow(config.base, float(j) / 2));
        float c = std::cos(angle), s = std::sin(angle);
        expected[left] = values[left] * c - values[right] * s;
        expected[right] = values[left] * s + values[right] * c;
        weights[left] = float((left % 7) + 1);
        weights[right] = float((right % 7) + 1);
        gradient[left] = weights[left] * c + weights[right] * s;
        gradient[right] = -weights[left] * s + weights[right] * c;
      }
    test::ExpectTensorNear(output, expected, 1e-5f);
    auto upstream = std::make_shared<Tensor>(weights.data(), output->Dims(),
                                               DataType::kFLOAT32, GetDevice());
    output->Backward(upstream);
    ASSERT_NE(input->grad(), nullptr);
    test::ExpectTensorNear(input->grad(), gradient, 1e-5f);
  };
  check(q, qo, q_values, 2);
  check(k, ko, k_values, 1);
}

TEST_P(LongRoPETest, ExtendedCacheRemainsActiveForShorterSequence) {
  LongRoPEConfig config{2, 100.0f, 2, 2, {1.0f}, {2.0f}};
  LongRoPE rope(config);
  const std::vector<float> ids_data{0, 1, 2};
  auto ids = std::make_shared<Tensor>(ids_data.data(), std::vector<int64_t>{1, 3},
                                      DataType::kFLOAT32, GetDevice());
  auto pos = std::make_shared<Tensor>(ids->To(DataType::kINT64));
  const std::vector<float> data{1, 0, 1, 0, 1, 0};
  auto x = std::make_shared<Tensor>(data.data(), std::vector<int64_t>{1, 3, 1, 2},
                                    DataType::kFLOAT32, GetDevice());
  auto [long_q, long_k] = rope.Forward(x, x, pos);
  test::ExpectTensorNear(long_q, {1, 0, std::cos(0.5f), std::sin(0.5f),
                                std::cos(1.0f), std::sin(1.0f)}, 1e-6f);
  auto short_x = x->Slice(1, 0, 2);
  const std::vector<float> short_ids_data{0, 1};
  auto short_ids = std::make_shared<Tensor>(short_ids_data.data(),
      std::vector<int64_t>{1, 2}, DataType::kFLOAT32, GetDevice());
  auto short_pos = std::make_shared<Tensor>(short_ids->To(DataType::kINT64));
  auto [short_q, short_k] = rope.Forward(short_x, short_x, short_pos);
  test::ExpectTensorNear(short_q, {1, 0, std::cos(0.5f), std::sin(0.5f)}, 1e-6f);
}

TEST_P(LongRoPETest, BF16RoundTripBackward) {
  LongRoPEConfig config{2, 100.0f, 2, 2, {1.0f}, {1.0f}};
  LongRoPE rope(config);
  const std::vector<float> values{1, 2, 3, 4};
  const std::vector<float> positions{0, 1};
  auto source = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{1, 2, 1, 2},
                                         DataType::kFLOAT32, GetDevice());
  auto x = std::make_shared<Tensor>(source->To(DataType::kBFLOAT16));
  x->RequiresGrad();
  auto ids = std::make_shared<Tensor>(positions.data(), std::vector<int64_t>{1, 2},
                                      DataType::kFLOAT32, GetDevice());
  auto pos = std::make_shared<Tensor>(ids->To(DataType::kINT64));
  auto key = std::make_shared<Tensor>(source->To(DataType::kBFLOAT16));
  auto [q, k] = rope.Forward(x, key, pos);
  {
    autograd::NoGradGuard no_grad;
    auto q_fp32 = std::make_shared<Tensor>(q->To(DataType::kFLOAT32));
    test::ExpectTensorFloatEqual(q_fp32, {1.0f, 2.0f, -1.7421875f, 4.6875f});
  }
  auto grad_fp32 = std::make_shared<Tensor>(values.data(), q->Dims(),
                                              DataType::kFLOAT32, GetDevice());
  auto grad = std::make_shared<Tensor>(grad_fp32->To(DataType::kBFLOAT16));
  q->Backward(grad);
  ASSERT_NE(x->grad(), nullptr);
  auto actual = std::make_shared<Tensor>(x->grad()->To(DataType::kFLOAT32));
  test::ExpectTensorFloatEqual(actual, {1.0f, 2.0f, 4.96875f, -0.36328125f});
}

TEST_P(LongRoPETest, ConfiguresInterleavingAmplitudeAndPositionLimit) {
  LongRoPEConfig config{4, 100.0f, 8, 8, {1.0f, 1.0f}, {1.0f, 1.0f}};
  config.rotary_interleaved = true;
  config.position_limit = LongRoPEPositionLimit::kCacheLength;
  config.attention_factor = 2.0f;
  LongRoPE rope(config);
  const std::vector<float> values{1, 2, 3, 4};
  const std::vector<float> position{2};
  auto q = std::make_shared<Tensor>(values.data(), std::vector<int64_t>{1, 1, 1, 4},
                                     DataType::kFLOAT32, GetDevice());
  auto k = std::make_shared<Tensor>(values.data(), q->Dims(), DataType::kFLOAT32, GetDevice());
  auto ids = std::make_shared<Tensor>(position.data(), std::vector<int64_t>{1, 1},
                                       DataType::kFLOAT32, GetDevice());
  auto pos = std::make_shared<Tensor>(ids->To(DataType::kINT64));
  q->RequiresGrad();
  auto [out, key] = rope.Forward(q, k, pos);
  const float c0 = std::cos(2.0f), s0 = std::sin(2.0f);
  const float c1 = std::cos(0.2f), s1 = std::sin(0.2f);
  test::ExpectTensorNear(out, {2 * (c0 - 2 * s0), 2 * (s0 + 2 * c0),
                               2 * (3 * c1 - 4 * s1), 2 * (3 * s1 + 4 * c1)}, 1e-5f);
  auto upstream = std::make_shared<Tensor>(values.data(), q->Dims(),
                                            DataType::kFLOAT32, GetDevice());
  out->Backward(upstream);
  ASSERT_NE(q->grad(), nullptr);
  test::ExpectTensorNear(q->grad(), {2 * (c0 + 2 * s0), 2 * (-s0 + 2 * c0),
                                     2 * (3 * c1 + 4 * s1), 2 * (-3 * s1 + 4 * c1)}, 1e-5f);
}

INFINI_TRAIN_REGISTER_TEST(LongRoPETest);

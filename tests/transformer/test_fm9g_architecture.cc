#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "example/fm9g/config.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

TEST(FM9GConfigTest, TinyKeepsDerivedScaleParameters) {
  auto config = fm9g::TinyTextConfig();
  fm9g::SanitizeTextConfig(config);
  EXPECT_EQ(config.n_layer, 2);
  EXPECT_EQ(config.n_embd, 128);
  EXPECT_FLOAT_EQ(config.scale_depth.value(), 1.4f);
  EXPECT_EQ(config.dim_model_base, 256);
  EXPECT_EQ(config.rope_factors.size(), 4);
}

class FM9GArchitectureTest : public infini_train::test::InfiniTrainTest {};

TEST_P(FM9GArchitectureTest, FactorScaledFrequencies) {
  auto freqs = PrecomputeFreqsCis(4, 2, 10000.0f, false, GetDevice(), {2.0f, 4.0f});
  test::ExpectTensorNear(freqs, {1.0f, 0.0f, 1.0f, 0.0f,
                                std::cos(0.5f), std::sin(0.5f),
                                std::cos(0.0025f), std::sin(0.0025f)}, 1e-5f);
}

TEST_P(FM9GArchitectureTest, TinyTextForwardAndBackward) {
  SKIP_CPU();
  auto config = fm9g::TinyTextConfig();
  config.vocab_size = 32;
  config.original_vocab_size = 32;
  config.block_size = 4;
  auto model = std::make_shared<nn::TransformerModel>(config);
  model->To(GetDevice());
  const float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto ids_float = std::make_shared<Tensor>(values, std::vector<int64_t>{1, 4},
                                            DataType::kFLOAT32, GetDevice());
  auto ids = std::make_shared<Tensor>(ids_float->To(DataType::kINT64));
  auto result = (*model)({ids})[0];
  EXPECT_EQ(result->Dims(), (std::vector<int64_t>{1, 4, 32}));
  result->Sum(2)->Sum(1)->Sum(0)->Backward();
  auto state = model->StateDict();
  auto it = state.find("transformer.h.0.attn.q_a_proj.weight");
  ASSERT_NE(it, state.end());
  EXPECT_NE(it->second->grad(), nullptr);
}

INFINI_TRAIN_REGISTER_TEST(FM9GArchitectureTest);

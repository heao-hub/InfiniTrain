#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/transformer/multimodal_model.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/tensor.h"

#include "example/fm9g/config.h"
#include "example/fm9g/resampler.h"
#include "tests/common/test_utils.h"

namespace infini_train {
TEST(MultimodalFusionCPU, MixedBatchMultipleSpansAndGradient) {
  const float text[] = {0, 1, 2, 3, 4, 5, 6, 7};
  const float visual[] = {20, 30};
  const int64_t spans[] = {1, 0, 1, 1, 2, 3};
  const std::vector<int64_t> shape{2, 4, 1};
  auto base =
      std::make_shared<Tensor>(text, shape, DataType::kFLOAT32)->RequiresGrad();
  auto source = std::make_shared<Tensor>(visual, std::vector<int64_t>{2, 1, 1},
                                         DataType::kFLOAT32)
                    ->RequiresGrad();
  auto bounds =
      std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kINT64);
  std::copy_n(spans, 6, static_cast<int64_t *>(bounds->DataPtr()));
  auto fused = nn::function::ReplaceSequenceSpans(base, source, bounds);
  test::ExpectTensorNear(fused, {0, 1, 2, 3, 20, 5, 30, 7}, 0);
  fused->Backward(std::make_shared<Tensor>(text, shape, DataType::kFLOAT32));
  test::ExpectTensorNear(base->grad(), {0, 1, 2, 3, 0, 5, 0, 7}, 0);
  test::ExpectTensorNear(source->grad(), {4, 6}, 0);
}
class MultimodalTrainingTest : public test::InfiniTrainTest {};

TEST_P(MultimodalTrainingTest, FrozenDecoderPropagatesToVisionModules) {
  SKIP_CPU();  // Same decoder backend coverage as TinyTextForwardAndBackward.
  auto config = fm9g::TinyTextConfig();
  config.vocab_size = config.original_vocab_size = 16;
  auto decoder = std::make_shared<nn::TransformerModel>(config);
  nn::VisionConfig vision_config;
  vision_config.hidden_size = 4;
  vision_config.intermediate_size = 8;
  vision_config.num_hidden_layers = vision_config.num_attention_heads = 1;
  vision_config.patch_size = vision_config.image_size = 2;
  auto encoder = std::make_shared<nn::VisionTransformer>(vision_config);
  auto resampler = std::make_shared<fm9g::Resampler>(1, config.n_embd, 1, 4);
  nn::MultimodalModel model(decoder, encoder, resampler);
  model.To(GetDevice());
  for (const auto &parameter : decoder->Parameters())
    parameter->set_requires_grad(false);
  auto ids = std::make_shared<Tensor>(std::vector<int64_t>{1, 3},
                                      DataType::kINT64, GetDevice());
  ids->Fill(1);
  auto pixels = std::make_shared<Tensor>(std::vector<int64_t>{1, 3, 2, 2},
                                         DataType::kFLOAT32, GetDevice());
  pixels->Fill(1);
  auto sizes =
      std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kINT64);
  sizes->Fill(1);
  auto spans =
      std::make_shared<Tensor>(std::vector<int64_t>{1, 3}, DataType::kINT64);
  const int64_t bounds[] = {0, 1, 2};
  std::copy_n(bounds, 3, static_cast<int64_t *>(spans->DataPtr()));
  auto logits = model({ids, pixels, sizes, spans})[0];
  EXPECT_EQ(logits->Dims(), (std::vector<int64_t>{1, 3, 16}));
  logits->Sum(2)->Sum(1)->Sum(0)->Backward();
  EXPECT_NE(resampler->parameter("proj")->grad(), nullptr);
  EXPECT_NE(encoder->module("embeddings")
                .module("patch_embedding")
                .parameter("weight")
                ->grad(),
            nullptr);
}

INFINI_TRAIN_REGISTER_TEST(MultimodalTrainingTest);
}  // namespace infini_train

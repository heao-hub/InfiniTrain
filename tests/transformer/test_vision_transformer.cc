#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/nn/modules/transformer/bidirectional_self_attention.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

namespace {
using namespace infini_train;

std::shared_ptr<Tensor> Floats(const std::vector<int64_t> &shape, const std::vector<float> &values) {
    auto tensor = std::make_shared<Tensor>(shape, DataType::kFLOAT32);
    std::copy(values.begin(), values.end(), static_cast<float *>(tensor->DataPtr()));
    return tensor;
}

void ZeroParameters(nn::Module &module) {
    for (auto &p : module.Parameters()) {
        auto *data = static_cast<float *>(p->DataPtr());
        std::fill(data, data + p->NumElements(), 0.0f);
    }
}

TEST(VisionTransformerCPU, BidirectionalMaskAndInputGradient) {
    nn::TransformerConfig config;
    config.n_embd = 2;
    config.n_head = 1;
    config.position_embedding_type = nn::PositionEmbeddingType::kNone;
    auto attention = std::make_shared<nn::BidirectionalSelfAttention>(config);
    ZeroParameters(*attention);
    // Q=K=0 gives uniform attention. V and output projection are identities.
    for (const auto *name : {"v_proj", "out_proj"}) {
        auto weight = attention->module(name).parameter("weight");
        auto *data = static_cast<float *>(weight->DataPtr());
        data[0] = data[3] = 1.0f;
    }
    auto input = Floats({1, 3, 2}, {1, 2, 3, 4, 50, 60})->RequiresGrad();
    {
        autograd::NoGradGuard no_grad;
        test::ExpectTensorNear((*attention)({input})[0], {18, 22, 18, 22, 18, 22}, 1e-5f);
    }
    const float blocked = -std::numeric_limits<float>::infinity();
    auto mask = Floats({1, 1, 3, 3}, {0, 0, blocked, 0, 0, blocked, 0, 0, blocked});
    auto output = (*attention)({input, mask})[0];
    test::ExpectTensorNear(output, {2, 3, 2, 3, 2, 3}, 1e-5f);
    output->Sum(2)->Sum(1)->Sum(0)->Backward();
    test::ExpectTensorNear(input->grad(), {1.5f, 1.5f, 1.5f, 1.5f, 0, 0}, 1e-5f);
}

TEST(VisionTransformerCPU, PatchOrderAndBucketPositions) {
    nn::VisionConfig config;
    config.num_channels = 1;
    config.patch_size = 2;
    config.image_size = 4;
    config.hidden_size = 1;
    auto embeddings = std::make_shared<nn::VisionEmbeddings>(config);
    ZeroParameters(*embeddings);
    auto weight = embeddings->module("patch_embedding").parameter("weight");
    const std::vector<float> coefficients{1, 10, 100, 1000};
    std::copy(coefficients.begin(), coefficients.end(), static_cast<float *>(weight->DataPtr()));
    auto position = embeddings->module("position_embedding").parameter("weight");
    auto *positions = static_cast<float *>(position->DataPtr());
    positions[0] = 0; positions[1] = 100; positions[2] = 200; positions[3] = 300;
    auto sizes = std::make_shared<Tensor>(std::vector<int64_t>{1, 2}, DataType::kINT64);
    auto *layout = static_cast<int64_t *>(sizes->DataPtr());
    layout[0] = 1; layout[1] = 2;
    auto pixels = Floats({1, 1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    test::ExpectTensorFloatEqual((*embeddings)({pixels, sizes})[0], {6521, 8843});
    // Same physical patch stream, different logical layout; invalidates position cache.
    layout[0] = 2; layout[1] = 1;
    test::ExpectTensorFloatEqual((*embeddings)({pixels, sizes})[0], {6521, 8943});
}

TEST(VisionTransformerCPU, SharedLayerUsesConfiguredEpsilon) {
    nn::TransformerConfig config;
    config.n_embd = 2;
    config.n_head = 1;
    config.ffn_hidden_size = 4;
    config.position_embedding_type = nn::PositionEmbeddingType::kNone;
    config.attention_direction = nn::AttentionDirection::kBidirectional;
    config.norm_eps = 0.25f;
    nn::TransformerLayer layer(config);
    auto norm = layer.mutable_module(nn::TransformerLayer::kLn1LayerName);
    auto result = (*norm)({Floats({1, 1, 2}, {0, 2})})[0];
    const float expected = 1.0f / std::sqrt(1.25f);
    test::ExpectTensorNear(result, {-expected, expected}, 1e-6f);
}

TEST(VisionTransformerCPU, SharedStackPaddingIsolationAndGradient) {
    nn::VisionConfig config;
    config.hidden_size = 4;
    config.intermediate_size = 6;
    config.num_attention_heads = 2;
    config.num_hidden_layers = 2;
    config.num_channels = 1;
    config.image_size = 2;
    config.patch_size = 1;
    auto model = std::make_shared<nn::VisionTransformer>(config);
    auto state = model->StateDict();
    ASSERT_TRUE(state.contains("encoder.h.0.mlp.c_fc.weight"));
    ASSERT_TRUE(state.contains("encoder.h.1.attn.q_proj.weight"));
    for (const auto &[name, p] : state) {
        auto *data = static_cast<float *>(p->DataPtr());
        for (size_t i = 0; i < p->NumElements(); ++i) {
            data[i] = 0.1f * std::sin(static_cast<float>(i + 1));
            if (name.find("ln_") != std::string::npos || name.find("layernorm") != std::string::npos) {
                if (name.ends_with("weight")) { data[i] += 1.0f; }
            }
        }
    }
    auto sizes = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kINT64);
    auto *layout = static_cast<int64_t *>(sizes->DataPtr());
    layout[0] = 1; layout[1] = 3; layout[2] = 1; layout[3] = 2;
    auto input = Floats({2, 1, 1, 3}, {0.2f, 0.4f, 0.7f, 0.3f, 0.5f, 10})->RequiresGrad();
    auto output = (*model)({input, sizes})[0];
    ASSERT_EQ(output->Dims(), (std::vector<int64_t>{2, 3, 4}));
    auto changed = Floats({2, 1, 1, 3}, {0.2f, 0.4f, 0.7f, 0.3f, 0.5f, -100});
    std::shared_ptr<Tensor> changed_output;
    {
        autograd::NoGradGuard no_grad;
        changed_output = (*model)({changed, sizes})[0];
    }
    auto *actual = static_cast<float *>(output->DataPtr());
    auto *other = static_cast<float *>(changed_output->DataPtr());
    for (int i = 0; i < 20; ++i) { EXPECT_NEAR(actual[i], other[i], 1e-5f); }
    // Loss on channel 0 of valid tokens in sample 1, avoiding LayerNorm's zero-sum direction.
    output->Slice(0, 1, 2)->Slice(1, 0, 2)->Slice(2, 0, 1)->Sum(2)->Sum(1)->Sum(0)->Backward();
    ASSERT_NE(input->grad(), nullptr);
    auto *gradient = static_cast<float *>(input->grad()->DataPtr());
    EXPECT_NEAR(gradient[5], 0.0f, 1e-6f);
    const float epsilon = 1e-3f;
    auto evaluate = [&](float delta) {
        autograd::NoGradGuard no_grad;
        auto perturbed = Floats({2, 1, 1, 3}, {0.2f, 0.4f, 0.7f, 0.3f + delta, 0.5f, 10});
        auto y = (*model)({perturbed, sizes})[0];
        auto *values = static_cast<float *>(y->DataPtr());
        return values[12] + values[16];
    };
    EXPECT_NEAR(gradient[3], (evaluate(epsilon) - evaluate(-epsilon)) / (2 * epsilon), 3e-3f);
    for (const auto &[name, p] : state) {
        ASSERT_NE(p->grad(), nullptr) << name;
        auto *data = static_cast<float *>(p->grad()->DataPtr());
        for (size_t i = 0; i < p->NumElements(); ++i) { EXPECT_TRUE(std::isfinite(data[i])) << name; }
    }
}
} // namespace

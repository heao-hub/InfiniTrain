#include "infini_train/include/nn/modules/transformer/bidirectional_self_attention.h"
#include <cmath>

#include "glog/logging.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

BidirectionalSelfAttention::BidirectionalSelfAttention(const TransformerConfig &config)
    : CloneableModule(kType), config_(config) {
    CHECK_GT(config.n_head, 0);
    CHECK_GT(config.n_embd, 0);
    CHECK_EQ(config.n_embd % config.n_head, 0);
    CHECK_EQ(parallel::global::GetTensorParallelSize(), 1) << "Bidirectional attention currently requires TP=1";
    CHECK(!parallel::global::GetSequenceParallelEnabled()) << "Bidirectional attention does not support SP yet";
    CHECK(config.position_embedding_type == PositionEmbeddingType::kNone);
    modules_[kKProjName] = std::make_shared<nn::Linear>(config.n_embd, config.n_embd, config.add_bias_linear);
    modules_[kVProjName] = std::make_shared<nn::Linear>(config.n_embd, config.n_embd, config.add_bias_linear);
    modules_[kQProjName] = std::make_shared<nn::Linear>(config.n_embd, config.n_embd, config.add_bias_linear);
    modules_[kOutProjName] = std::make_shared<nn::Linear>(config.n_embd, config.n_embd, config.add_bias_linear);
}

std::vector<std::shared_ptr<Tensor>>
BidirectionalSelfAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK(inputs.size() == 1 || inputs.size() == 2);
    CHECK_EQ(inputs[0]->Dims().size(), 3);
    CHECK_EQ(inputs[0]->Dims()[2], config_.n_embd);
    const int64_t batch = inputs[0]->Dims()[0];
    const int64_t sequence = inputs[0]->Dims()[1];
    const int64_t head_dim = config_.n_embd / config_.n_head;
    auto query = (*modules_[kQProjName])({inputs[0]})[0]
                     ->View({batch, sequence, config_.n_head, head_dim})
                     ->Transpose(1, 2);
    auto key = (*modules_[kKProjName])({inputs[0]})[0]
                   ->View({batch, sequence, config_.n_head, head_dim})
                   ->Transpose(1, 2);
    auto value = (*modules_[kVProjName])({inputs[0]})[0]
                     ->View({batch, sequence, config_.n_head, head_dim})
                     ->Transpose(1, 2);
    auto attention = query->Matmul(key->Transpose(-2, -1)) * (1.0f / std::sqrt(static_cast<float>(head_dim)));
    // inputs[1]：可选的 attention mask
    if (inputs.size() == 2 && inputs[1]) {
        CHECK(inputs[1]->Dims() == (std::vector<int64_t>{batch, 1, sequence, sequence}));
        CHECK(inputs[1]->GetDevice() == inputs[0]->GetDevice());
        CHECK(inputs[1]->Dtype() == inputs[0]->Dtype());
        // 加性 mask，当前视觉编码器生成的值为：可以关注的 key：0 需要屏蔽的 key：-∞
        attention = attention + inputs[1];
    }
    attention = nn::function::Softmax(attention, -1);
    auto output = attention->Matmul(value)->Transpose(1, 2)->Contiguous()->View({batch, sequence, config_.n_embd});
    return (*modules_[kOutProjName])({output});
}

} // namespace infini_train::nn

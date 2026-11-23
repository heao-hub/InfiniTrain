#pragma once

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
namespace infini_train::nn {

// Forward: {hidden[B,S,C], optional additive mask[B,1,S,S]}.
// Zero mask entries allow attention; -infinity blocks keys. No causal mask.
class BidirectionalSelfAttention : public CloneableModule<BidirectionalSelfAttention> {
public:
    static constexpr char kType[] = "BidirectionalSelfAttention";
    static constexpr char kQProjName[] = "q_proj";
    static constexpr char kKProjName[] = "k_proj";
    static constexpr char kVProjName[] = "v_proj";
    static constexpr char kOutProjName[] = "out_proj";
    explicit BidirectionalSelfAttention(const TransformerConfig &config);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

private:
    TransformerConfig config_;
};

} // namespace infini_train::nn

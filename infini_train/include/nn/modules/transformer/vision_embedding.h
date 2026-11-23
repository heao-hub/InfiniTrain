#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

struct VisionConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_channels = 3;
    int64_t image_size = 0;
    int64_t patch_size = 0;
    float layer_norm_eps = 1e-6f;
};

// Patch tokens use row-major patch order. Valid tokens occupy a contiguous prefix;
// target_sizes [B,2] describes their original 2-D layout in patch units.
class VisionEmbeddings : public CloneableModule<VisionEmbeddings> {
public:
    static constexpr char kType[] = "VisionEmbeddings";
    static constexpr char kPatchEmbeddingName[] = "patch_embedding";
    static constexpr char kPositionEmbeddingName[] = "position_embedding";
    explicit VisionEmbeddings(const VisionConfig &config);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

private:
    VisionConfig config_;
    std::vector<int64_t> cached_target_sizes_;
    std::vector<int64_t> cached_position_shape_;
    std::shared_ptr<infini_train::Tensor> cached_position_ids_;
};

} // namespace infini_train::nn

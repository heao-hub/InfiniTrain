#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace fm9g {

class Resampler : public infini_train::nn::CloneableModule<Resampler> {
public:
    static constexpr char kType[] = "FM9GVResampler";

    Resampler(int64_t num_queries = 64, int64_t embed_dim = 2560, int64_t num_heads = 16, int64_t kv_dim = 1152,
              bool align_corners = false);

    int64_t num_heads() const;

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

private:
    int64_t num_queries_;
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t kv_dim_;
    int64_t grid_size_;
    bool align_corners_;
};

} // namespace fm9g

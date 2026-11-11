#pragma once

#include <cstdint>
#include <memory>
#include <tuple>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
// RoPE helper method
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end,
                                           float theta = 10000.0f,
                                           bool use_scaled = false,
                                           Device device = Device());

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ApplyRotaryEmbedding(const std::shared_ptr<Tensor> &xq,
                     const std::shared_ptr<Tensor> &xk,
                     const std::shared_ptr<Tensor> &freqs_cis,
                     bool rotary_interleaved);

// x: [B,T,H,D], cos/sin: broadcastable to [B,T,H,D/2].
std::shared_ptr<Tensor> ApplyRotaryCoefficients(
    const std::shared_ptr<Tensor> &x, 
    const std::shared_ptr<Tensor> &cos,
    const std::shared_ptr<Tensor> &sin, 
    bool rotary_interleaved);
}  // namespace infini_train

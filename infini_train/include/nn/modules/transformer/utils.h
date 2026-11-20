#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
// RoPE helper method
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end,
                                           float theta = 10000.0f,
                                           bool use_scaled = false,
                                           Device device = Device(),
                                           const std::vector<float> &freq_factors = {});

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ApplyRotaryEmbedding(const std::shared_ptr<Tensor> &xq,
                     const std::shared_ptr<Tensor> &xk,
                     const std::shared_ptr<Tensor> &freqs_cis,
                     bool rotary_interleaved);
}  // namespace infini_train

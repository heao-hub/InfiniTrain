#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "infini_train/include/datatype.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

enum class LongRoPEPositionLimit {
  kSequenceLength,
  kCacheLength,
};

struct LongRoPEConfig {
  int64_t rotary_dim;
  float base;
  int64_t max_position_embeddings;
  int64_t original_max_position_embeddings;
  std::vector<float> short_factors;
  std::vector<float> long_factors;
  // Dtype of inverse frequencies after model conversion, used on cache rebuild.
  DataType inverse_frequency_dtype = DataType::kFLOAT32;
  bool rotary_interleaved = false;
  LongRoPEPositionLimit position_limit = LongRoPEPositionLimit::kSequenceLength;
  // If absent, compute sqrt(1 + log(max/original) / log(original)).
  std::optional<float> attention_factor = std::nullopt;
};

// Single-threaded, host-backed reference implementation. Inputs are [B,T,H,D]
// and position_ids is INT64 [B,T] or [1,T]. Q and K may have different H.
// Cache expansion uses T and retains the expanded cache for later calls.
class LongRoPE {
 public:
  explicit LongRoPE(LongRoPEConfig config);

  std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> Forward(
      const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
      const std::shared_ptr<Tensor> &position_ids);

 private:
  void BuildCache(int64_t seq_len, DataType cache_dtype, bool initial = false);

  LongRoPEConfig config_;
  float amplitude_;
  int64_t cached_length_ = 0;
  std::vector<float> cos_cache_;
  std::vector<float> sin_cache_;
};

}  // namespace infini_train

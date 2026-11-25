#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

// Batch-first multi-head cross attention. Inputs are query [B,Tq,C], key
// [B,Tk,C], value [B,Tk,C], and optionally an additive mask [B,1,Tq,Tk].
class CrossAttention : public CloneableModule<CrossAttention> {
 public:
  static constexpr char kType[] = "CrossAttention";
  CrossAttention(int64_t embed_dim, int64_t num_heads, bool bias = true);
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override;

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
};

}  // namespace infini_train::nn

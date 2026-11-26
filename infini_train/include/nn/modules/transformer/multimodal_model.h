#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {
class TransformerModel;

// Decoder-only vision-language model, using the decoder's contiguous positions
// and causal mask. Use unpadded or right-padded inputs with ignored pad labels.
class MultimodalModel : public CloneableModule<MultimodalModel> {
 public:
  MultimodalModel(std::shared_ptr<TransformerModel> decoder,
                  std::shared_ptr<Module> vision_encoder,
                  std::shared_ptr<Module> resampler);

  // {ids}, {ids, visual_tokens, spans}, or {ids, pixels, target_sizes, spans}.
  // Each vision unit maps to one (batch, begin, end) row in int64 spans.
  // Encoder and resampler both accept {features, target_sizes} and return
  // their result as the first tensor. Visual tokens must match text dtype.
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override;
};
}  // namespace infini_train::nn

#include "infini_train/include/nn/modules/transformer/multimodal_model.h"

#include <utility>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn {
MultimodalModel::MultimodalModel(std::shared_ptr<TransformerModel> decoder,
                                 std::shared_ptr<Module> vision_encoder,
                                 std::shared_ptr<Module> resampler)
    : CloneableModule("MultimodalModel") {
  CHECK(decoder && vision_encoder && resampler);
  CHECK_EQ(parallel::global::GetPipelineParallelSize(), 1);
  CHECK_EQ(parallel::global::GetVirtualPipelineParallelSize(), 1);
  CHECK_EQ(parallel::global::GetTensorParallelSize(), 1);
  modules_["decoder"] = std::move(decoder);
  modules_["vision_encoder"] = std::move(vision_encoder);
  modules_["resampler"] = std::move(resampler);
}

std::vector<std::shared_ptr<Tensor>> MultimodalModel::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
  CHECK(inputs.size() == 1 || inputs.size() == 3 || inputs.size() == 4);
  auto decoder =
      std::static_pointer_cast<TransformerModel>(modules_.at("decoder"));
  if (inputs.size() == 1) return (*decoder)(inputs);
  auto visual = inputs[1];
  if (inputs.size() == 4) {
    auto patches =
        (*modules_.at("vision_encoder"))({inputs[1], inputs[2]}).at(0);
    visual = (*modules_.at("resampler"))({patches, inputs[2]}).at(0);
  }
  auto text = decoder->PrepareInputEmbeddings(inputs[0]);
  return decoder->ForwardFromEmbeddings(
      function::ReplaceSequenceSpans(text, visual, inputs.back()));
}
}  // namespace infini_train::nn

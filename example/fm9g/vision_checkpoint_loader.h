#pragma once
#include <string>
#include "infini_train/include/nn/modules/transformer/transformer.h"
namespace fm9g {
void LoadVisionFromBin(infini_train::nn::VisionTransformer &model, const std::string &path);
}  // namespace fm9g

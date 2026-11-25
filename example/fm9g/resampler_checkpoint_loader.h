#pragma once
#include <string>
namespace fm9g {
class Resampler;
// Load the converted FP32 weights before moving the model to another device.
void LoadResamplerFromBin(Resampler &model, const std::string &path);
}

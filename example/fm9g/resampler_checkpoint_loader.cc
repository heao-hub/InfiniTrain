#include "example/fm9g/resampler_checkpoint_loader.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "example/fm9g/resampler.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace fm9g {
void LoadResamplerFromBin(Resampler &model, const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  CHECK(stream) << "Cannot open resampler checkpoint: " << path;
  char magic[8];
  int64_t dims[4];
  stream.read(magic, sizeof(magic));
  stream.read(reinterpret_cast<char *>(dims), sizeof(dims));
  CHECK(stream);
  CHECK_EQ(std::memcmp(magic, "FM9RS002", 8), 0);
  auto state = model.StateDict();
  CHECK_EQ(state.at("query")->Dims()[0], dims[0]);
  CHECK_EQ(state.at("proj")->Dims()[0], dims[1]);
  CHECK_EQ(state.at("query")->Dims()[1], dims[3]);
  CHECK_EQ(model.num_heads(), dims[2]);
  const std::vector<std::string> names = {
      "pos_embed", "feature_1x_embedding", "feature_4x_embedding", "query",
      "features_1x_projector.weight", "features_1x_projector.bias",
      "features_4x_projector.weight", "features_4x_projector.bias",
      "attention.q_proj.weight", "attention.q_proj.bias", "attention.k_proj.weight", "attention.k_proj.bias",
      "attention.v_proj.weight", "attention.v_proj.bias", "attention.out_proj.weight", "attention.out_proj.bias",
      "ln_q.weight", "ln_q.bias", "ln_kv.weight", "ln_kv.bias", "ln_proj.weight", "ln_proj.bias",
      "ln_post.weight", "ln_post.bias", "out_mlp_fc1.weight", "out_mlp_fc1.bias",
      "out_mlp_fc2.weight", "out_mlp_fc2.bias", "proj"};
  for (const auto &name : names) {
    auto tensor = state.at(name);
    CHECK(tensor->Dtype() == infini_train::DataType::kFLOAT32);
    CHECK(tensor->GetDevice().type() == infini_train::Device::DeviceType::kCPU);
    stream.read(static_cast<char *>(tensor->DataPtr()), static_cast<std::streamsize>(tensor->SizeInBytes()));
    CHECK(stream) << "Cannot read resampler tensor " << name;
  }
  CHECK_EQ(stream.peek(), std::ifstream::traits_type::eof());
}
}  // namespace fm9g
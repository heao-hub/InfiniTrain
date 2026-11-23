#include "example/fm9g/vision_checkpoint_loader.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "glog/logging.h"
#include "infini_train/include/tensor.h"

namespace fm9g {

using infini_train::DataType;
using infini_train::Tensor;

namespace {
constexpr char kMagic[8] = {'F', 'M', '9', 'V', 'E', '0', '0', '1'};

void ReadExact(std::ifstream &stream, void *data, size_t bytes, const std::string &name) {
    stream.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
    CHECK(stream) << "Failed to read Vision Transformer tensor " << name;
}

void ReadTensor(std::ifstream &stream, const std::shared_ptr<Tensor> &tensor, const std::string &name) {
    CHECK(tensor->Dtype() == DataType::kFLOAT32);
    CHECK(tensor->GetDevice().type() == infini_train::Device::DeviceType::kCPU)
        << "Load vision weights before moving the model to an accelerator";
    ReadExact(stream, tensor->DataPtr(), tensor->SizeInBytes(), name);
}
} // namespace

void LoadVisionFromBin(infini_train::nn::VisionTransformer &model, const std::string &path) {
    CHECK(std::filesystem::exists(path)) << "Vision Transformer checkpoint not found: " << path;
    std::ifstream stream(path, std::ios::binary);
    char magic[8];
    ReadExact(stream, magic, sizeof(magic), "header");
    CHECK_EQ(std::memcmp(magic, kMagic, sizeof(kMagic)), 0);
    int64_t dims[7];
    float layer_norm_eps;
    ReadExact(stream, dims, sizeof(dims), "configuration");
    ReadExact(stream, &layer_norm_eps, sizeof(layer_norm_eps), "layer_norm_eps");
    CHECK_EQ(dims[0], model.Config().hidden_size);
    CHECK_EQ(dims[1], model.Config().intermediate_size);
    CHECK_EQ(dims[2], model.Config().num_hidden_layers);
    CHECK_EQ(dims[3], model.Config().num_attention_heads);
    CHECK_EQ(dims[4], model.Config().num_channels);
    CHECK_EQ(dims[5], model.Config().image_size);
    CHECK_EQ(dims[6], model.Config().patch_size);
    CHECK_EQ(layer_norm_eps, model.Config().layer_norm_eps);

    auto state = model.StateDict();
    const auto read = [&](const std::string &name) {
        auto iter = state.find(name);
        CHECK(iter != state.end()) << "Missing InfiniTrain VPM parameter " << name;
        ReadTensor(stream, iter->second, name);
    };
    read("embeddings.patch_embedding.weight");
    read("embeddings.patch_embedding.bias");
    read("embeddings.position_embedding.weight");
    for (int64_t i = 0; i < model.Config().num_hidden_layers; ++i) {
        const std::string prefix = "encoder.h." + std::to_string(i) + ".";
        for (const char *projection : {"k_proj", "v_proj", "q_proj", "out_proj"}) {
            read(prefix + "attn." + projection + ".weight");
            read(prefix + "attn." + projection + ".bias");
        }
        read(prefix + "ln_1.weight");
        read(prefix + "ln_1.bias");
        read(prefix + "mlp.c_fc.weight");
        read(prefix + "mlp.c_fc.bias");
        read(prefix + "mlp.c_proj.weight");
        read(prefix + "mlp.c_proj.bias");
        read(prefix + "ln_2.weight");
        read(prefix + "ln_2.bias");
    }
    read("post_layernorm.weight");
    read("post_layernorm.bias");
    CHECK_EQ(stream.peek(), std::ifstream::traits_type::eof()) << "Unexpected trailing VPM checkpoint data";
}

} // namespace fm9g

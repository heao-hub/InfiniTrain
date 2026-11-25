#include "example/fm9g/resampler.h"

#include <cmath>
#include <cstdint>

#include "glog/logging.h"


#include "infini_train/include/nn/modules/transformer/cross_attention.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/tensor.h"

namespace fm9g {

using infini_train::DataType;
using infini_train::Tensor;
namespace nn = infini_train::nn;

namespace {
std::shared_ptr<Tensor> ToNchw(const std::shared_ptr<Tensor> &input) {
    return input->Transpose(1, 3)->Transpose(2, 3)->Contiguous();
}

std::shared_ptr<Tensor> ToNhwc(const std::shared_ptr<Tensor> &input) {
    return input->Transpose(1, 2)->Transpose(2, 3)->Contiguous();
}

std::shared_ptr<Tensor> InterpolateNhwc(const std::shared_ptr<Tensor> &input, int64_t output_height,
                                        int64_t output_width, bool align_corners) {
    return ToNhwc(
        nn::function::InterpolateBilinear2d(ToNchw(input), {output_height, output_width}, align_corners));
}

std::shared_ptr<Tensor> PoolRegularGrid(const std::shared_ptr<Tensor> &input, int64_t grid_size, int64_t pooled_size) {
    const int64_t batch = input->Dims()[0];
    const int64_t height = input->Dims()[1];
    const int64_t width = input->Dims()[2];
    const int64_t regions = grid_size * grid_size;
    auto boxes = std::make_shared<Tensor>(std::vector<int64_t>{batch * regions, 5}, DataType::kFLOAT32);
    auto *box_data = static_cast<float *>(boxes->DataPtr());
    const float roi_height = static_cast<float>(height) / grid_size;
    const float roi_width = static_cast<float>(width) / grid_size;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t row = 0; row < grid_size; ++row) {
            for (int64_t col = 0; col < grid_size; ++col) {
                const int64_t offset = ((b * grid_size + row) * grid_size + col) * 5;
                box_data[offset] = static_cast<float>(b);
                box_data[offset + 1] = col * roi_width;
                box_data[offset + 2] = row * roi_height;
                box_data[offset + 3] = (col + 1) * roi_width;
                box_data[offset + 4] = (row + 1) * roi_height;
            }
        }
    }
    boxes = std::make_shared<Tensor>(boxes->To(input->GetDevice()));
    auto pooled = nn::function::RoiAlign(ToNchw(input), boxes, {pooled_size, pooled_size});
    return pooled->View({batch * regions, input->Dims()[3], pooled_size * pooled_size})->Transpose(1, 2)->Contiguous();
}
} // namespace

Resampler::Resampler(int64_t num_queries, int64_t embed_dim, int64_t num_heads, int64_t kv_dim, bool align_corners)
    : CloneableModule(kType), num_queries_(num_queries), embed_dim_(embed_dim), num_heads_(num_heads), kv_dim_(kv_dim),
      grid_size_(static_cast<int64_t>(std::sqrt(num_queries))), align_corners_(align_corners) {
    CHECK_GT(num_queries_, 0);
    CHECK_GT(num_heads_, 0);
    CHECK_GT(kv_dim_, 0);
    CHECK_EQ(kv_dim_ % 4, 0);
    CHECK_EQ(grid_size_ * grid_size_, num_queries_);
    CHECK_EQ(kv_dim_ % num_heads_, 0);

    parameters_["pos_embed"]
        = std::make_shared<Tensor>(std::vector<int64_t>{num_queries_, kv_dim_}, DataType::kFLOAT32)->RequiresGrad();
    parameters_["feature_1x_embedding"]
        = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, kv_dim_}, DataType::kFLOAT32)->RequiresGrad();
    parameters_["feature_4x_embedding"]
        = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, kv_dim_}, DataType::kFLOAT32)->RequiresGrad();
    parameters_["query"]
        = std::make_shared<Tensor>(std::vector<int64_t>{num_queries_, kv_dim_}, DataType::kFLOAT32)->RequiresGrad();
    modules_["features_1x_projector"] = std::make_shared<nn::Linear>(kv_dim_, kv_dim_);
    modules_["features_4x_projector"] = std::make_shared<nn::Linear>(kv_dim_, kv_dim_);
    modules_["attention"] = std::make_shared<nn::CrossAttention>(kv_dim_, num_heads_);
    modules_["ln_q"] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{kv_dim_}, 1e-6f);
    modules_["ln_kv"] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{kv_dim_}, 1e-6f);
    modules_["ln_proj"] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{kv_dim_}, 1e-6f);
    modules_["ln_post"] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{kv_dim_}, 1e-6f);
    modules_["out_mlp_fc1"] = std::make_shared<nn::Linear>(kv_dim_, 2 * kv_dim_);
    modules_["gelu"] = std::make_shared<nn::NewGELU>();
    modules_["out_mlp_fc2"] = std::make_shared<nn::Linear>(2 * kv_dim_, embed_dim_);
    parameters_["proj"] = std::make_shared<Tensor>(std::vector<int64_t>{embed_dim_, embed_dim_}, DataType::kFLOAT32)->RequiresGrad();
    nn::init::Normal(parameters_["query"], 0, 0.02f);
    nn::init::Zeros(parameters_["feature_1x_embedding"]);
    nn::init::Zeros(parameters_["feature_4x_embedding"]);
    nn::init::Normal(parameters_["proj"], 0, 1.0f / std::sqrt(static_cast<float>(embed_dim_)));
    // Match the reference 2D sin/cos position initialization (x followed by y).
    auto *pos_data = static_cast<float *>(parameters_["pos_embed"]->DataPtr());
    const int64_t half = kv_dim_ / 2, quarter = kv_dim_ / 4;
    for (int64_t y = 0; y < grid_size_; ++y) {
        for (int64_t x = 0; x < grid_size_; ++x) {
            for (int64_t j = 0; j < quarter; ++j) {
                const float freq = std::pow(10000.0f, -static_cast<float>(j) / quarter);
                const auto base = (y * grid_size_ + x) * kv_dim_;
                pos_data[base + j] = std::sin(x * freq);
                pos_data[base + quarter + j] = std::cos(x * freq);
                pos_data[base + half + j] = std::sin(y * freq);
                pos_data[base + half + quarter + j] = std::cos(y * freq);
            }
        }
    }
}

int64_t Resampler::num_heads() const { return num_heads_; }

std::vector<std::shared_ptr<Tensor>> Resampler::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 2) << "Resampler expects {features, patch_sizes}";
    const auto &features = inputs[0];
    CHECK_EQ(features->Dims().size(), 3);
    auto patch_sizes_cpu = inputs[1]->GetDevice().type() == infini_train::Device::DeviceType::kCPU
                             ? inputs[1]
                             : std::make_shared<Tensor>(inputs[1]->To(infini_train::Device()));
    CHECK(patch_sizes_cpu->Dtype() == DataType::kINT64);
    const int64_t batch = features->Dims()[0];
    CHECK_GT(batch, 0);
    CHECK_EQ(patch_sizes_cpu->NumElements(), batch * 2);
    const auto *patch_sizes = static_cast<const int64_t *>(patch_sizes_cpu->DataPtr());
    if (batch > 1) {
        std::vector<std::shared_ptr<Tensor>> outputs;
        outputs.reserve(batch);
        for (int64_t index = 0; index < batch; ++index) {
            auto item_size = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kINT64);
            auto *item_size_data = static_cast<int64_t *>(item_size->DataPtr());
            item_size_data[0] = patch_sizes[2 * index];
            item_size_data[1] = patch_sizes[2 * index + 1];
            outputs.push_back(Forward({features->Slice(0, index, index + 1), item_size})[0]);
        }
        return {nn::function::Concat(outputs, 0)};
    }
    const int64_t height = patch_sizes[0];
    const int64_t width = patch_sizes[1];
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);
    CHECK_GE(features->Dims()[1], height * width);
    CHECK_EQ(features->Dims()[2], kv_dim_);

    auto feature_1x = features->Slice(1, 0, height * width)->View({batch, height, width, kv_dim_});
    auto feature_4x = InterpolateNhwc(feature_1x, height * 4, width * 4, align_corners_);

    auto value_1x = (*modules_["features_1x_projector"])({feature_1x})[0];
    auto value_4x = (*modules_["features_4x_projector"])({feature_4x})[0];
    value_1x = (*modules_["ln_kv"])({value_1x->View({batch, height * width, kv_dim_})})[0]->View(
        {batch, height, width, kv_dim_});
    value_4x = (*modules_["ln_kv"])({value_4x->View({batch, height * width * 16, kv_dim_})})[0]->View(
        {batch, height * 4, width * 4, kv_dim_});

    auto pos = parameters_["pos_embed"]->View({1, grid_size_, grid_size_, kv_dim_});
    auto pos_1x = InterpolateNhwc(pos, height, width, align_corners_);
    auto pos_4x = InterpolateNhwc(pos, height * 4, width * 4, align_corners_);
    auto key_1x = value_1x + pos_1x + parameters_["feature_1x_embedding"];
    auto key_4x = value_4x + pos_4x + parameters_["feature_4x_embedding"];

    key_1x = PoolRegularGrid(key_1x, grid_size_, 3);
    key_4x = PoolRegularGrid(key_4x, grid_size_, 3);
    value_1x = PoolRegularGrid(value_1x, grid_size_, 3);
    value_4x = PoolRegularGrid(value_4x, grid_size_, 3);
    auto keys = nn::function::Concat({key_1x, key_4x}, 1);
    auto values = nn::function::Concat({value_1x, value_4x}, 1);

    auto query = (*modules_["ln_q"])({parameters_["query"]->View({1, num_queries_, kv_dim_})})[0]
               + parameters_["pos_embed"]->View({1, num_queries_, kv_dim_});
    query = query->View({num_queries_, 1, kv_dim_});
    auto output = (*modules_["attention"])({query, keys, values})[0]->View({batch, num_queries_, kv_dim_});
    output = (*modules_["ln_proj"])({output})[0];
    output = (*modules_["ln_post"])({output})[0];
    output = (*modules_["out_mlp_fc1"])({output})[0];
    output = (*modules_["gelu"])({output})[0];
    output = (*modules_["out_mlp_fc2"])({output})[0];

    auto projected = output->Contiguous()
                         ->View({batch * num_queries_, embed_dim_})
                         ->Matmul(parameters_["proj"]);
    return {projected->View({batch, num_queries_, embed_dim_})};
    
}

} // namespace fm9g

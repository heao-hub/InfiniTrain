#include "infini_train/include/nn/modules/transformer/vision_embedding.h"
#include <algorithm>

#include "glog/logging.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

VisionEmbeddings::VisionEmbeddings(const VisionConfig &config) : CloneableModule(kType), config_(config) {
    CHECK_GT(config.patch_size, 0);
    CHECK_GT(config.num_channels, 0);
    CHECK_GT(config.hidden_size, 0);
    CHECK_GE(config.image_size, config.patch_size);
    CHECK_EQ(config.image_size % config.patch_size, 0);
    modules_[kPatchEmbeddingName]
        = std::make_shared<nn::Linear>(config.num_channels * config.patch_size * config.patch_size, config.hidden_size);
    const int64_t positions_per_side = config.image_size / config.patch_size;
    modules_[kPositionEmbeddingName]
        = std::make_shared<nn::Embedding>(positions_per_side * positions_per_side, config.hidden_size);
}

std::vector<std::shared_ptr<Tensor>> VisionEmbeddings::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 2);
    const auto &pixels = inputs[0];
    const auto &target_sizes = inputs[1];
    CHECK_EQ(pixels->Dims().size(), 4);
    CHECK_GT(pixels->Dims()[0], 0);
    CHECK_GT(pixels->Dims()[2], 0);
    CHECK_GT(pixels->Dims()[3], 0);
    CHECK_EQ(pixels->Dims()[1], config_.num_channels);
    CHECK(target_sizes->Dtype() == DataType::kINT64);
    CHECK_EQ(target_sizes->NumElements(), pixels->Dims()[0] * 2);

    const int64_t batch = pixels->Dims()[0];
    const int64_t height = pixels->Dims()[2] / config_.patch_size;
    const int64_t width = pixels->Dims()[3] / config_.patch_size;
    CHECK_EQ(height * config_.patch_size, pixels->Dims()[2]);
    CHECK_EQ(width * config_.patch_size, pixels->Dims()[3]);

    auto patches = pixels->View({batch, config_.num_channels, height, config_.patch_size, width, config_.patch_size});
    patches = patches->Transpose(1, 2)->Transpose(2, 4)->Transpose(3, 4)->Contiguous();
    patches = patches->View({batch, height * width, config_.num_channels * config_.patch_size * config_.patch_size});
    auto embeddings = (*modules_[kPatchEmbeddingName])({patches})[0];

    auto sizes_cpu = target_sizes->GetDevice().type() == Device::DeviceType::kCPU
                       ? target_sizes
                       : std::make_shared<Tensor>(target_sizes->To(Device()));
    const auto *sizes = static_cast<const int64_t *>(sizes_cpu->DataPtr());
    const std::vector<int64_t> target_size_values(sizes, sizes + target_sizes->NumElements());
    const std::vector<int64_t> position_shape{batch, height * width};
    if (!cached_position_ids_ || cached_target_sizes_ != target_size_values || cached_position_shape_ != position_shape
        || cached_position_ids_->GetDevice() != pixels->GetDevice()) {
        auto position_ids = std::make_shared<Tensor>(position_shape, DataType::kINT64);
        auto *ids = static_cast<int64_t *>(position_ids->DataPtr());
        std::fill(ids, ids + position_ids->NumElements(), int64_t{0});
        const int64_t positions_per_side = config_.image_size / config_.patch_size;
        for (int64_t b = 0; b < batch; ++b) {
            const int64_t target_height = sizes[2 * b];
            const int64_t target_width = sizes[2 * b + 1];
            CHECK_GT(target_height, 0);
            CHECK_GT(target_width, 0);
            CHECK_LE(target_height, height * width / target_width);
            CHECK_LE(target_height * target_width, height * width)
                << "Target patch layout exceeds the padded patch sequence";
            for (int64_t row = 0; row < target_height; ++row) {
                const int64_t bucket_row = row * positions_per_side / target_height;
                for (int64_t col = 0; col < target_width; ++col) {
                    const int64_t bucket_col = col * positions_per_side / target_width;
                    ids[b * height * width + row * target_width + col] = bucket_row * positions_per_side + bucket_col; 
                }
            }
        }
        cached_target_sizes_ = target_size_values;
        cached_position_shape_ = position_shape;
        cached_position_ids_ = std::make_shared<Tensor>(position_ids->To(pixels->GetDevice()));
    }
    auto positions = (*modules_[kPositionEmbeddingName])({cached_position_ids_})[0];
    return {embeddings + positions};
}

} // namespace infini_train::nn

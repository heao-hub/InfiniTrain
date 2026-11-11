#include "infini_train/include/nn/modules/transformer/long_rope.h"

#include <cmath>
#include <utility>

#include "glog/logging.h"
#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/modules/transformer/utils.h"

namespace infini_train {
namespace {

float RoundTo(float value, DataType dtype) {
  switch (dtype) {
    case DataType::kFLOAT32: return value;
    case DataType::kBFLOAT16: return static_cast<float>(BF16(value));
    case DataType::kFLOAT16: return static_cast<float>(FP16(value));
    default: LOG(FATAL) << "LongRoPE supports FP32, BF16 and FP16"; return 0;
  }
}

std::shared_ptr<Tensor> AsFloat(const std::shared_ptr<Tensor> &x) {
  return x->Dtype() == DataType::kFLOAT32
             ? x : std::make_shared<Tensor>(x->To(DataType::kFLOAT32));
}

void Sync(Device device) {
  core::DeviceGuard guard(device);
  auto *impl = core::GetDeviceGuardImpl(device.type());
  impl->SynchronizeStream(impl->GetStream(device));
}

}  // namespace

LongRoPE::LongRoPE(LongRoPEConfig config)
    : config_(std::move(config)), amplitude_(1.0f) {
  CHECK_GT(config_.rotary_dim, 0);
  CHECK_EQ(config_.rotary_dim % 2, 0);
  CHECK_GT(config_.base, 0.0f);
  CHECK(std::isfinite(config_.base));
  CHECK_GT(config_.max_position_embeddings, 0);
  CHECK_GT(config_.original_max_position_embeddings, 1);
  CHECK_EQ(config_.short_factors.size(), config_.rotary_dim / 2);
  CHECK_EQ(config_.long_factors.size(), config_.rotary_dim / 2);
  for (const auto &factors : {config_.short_factors, config_.long_factors}) {
    for (float f : factors) CHECK(std::isfinite(f) && f > 0.0f);
  }
  RoundTo(1.0f, config_.inverse_frequency_dtype);
  if (config_.attention_factor) {
    amplitude_ = *config_.attention_factor;
  } else {
    const double ratio = static_cast<double>(config_.max_position_embeddings) /
                         config_.original_max_position_embeddings;
    const double magnitude = 1.0 + std::log(ratio) /
        std::log(static_cast<double>(config_.original_max_position_embeddings));
    CHECK(std::isfinite(magnitude) && magnitude >= 0.0);
    amplitude_ = static_cast<float>(std::sqrt(magnitude));
  }
  CHECK(std::isfinite(amplitude_) && amplitude_ > 0.0f);
  // Initial cache is built in FP32; cache rebuilds use the input dtype.
  BuildCache(config_.max_position_embeddings, DataType::kFLOAT32, true);
}

void LongRoPE::BuildCache(int64_t seq_len, DataType cache_dtype, bool initial) {
  const auto half = config_.rotary_dim / 2;
  const auto &factors = seq_len > config_.original_max_position_embeddings
                            ? config_.long_factors : config_.short_factors;
  cos_cache_.resize(seq_len * half);
  sin_cache_.resize(seq_len * half);
  for (int64_t p = 0; p < seq_len; ++p) {
    for (int64_t j = 0; j < half; ++j) {
      const float inv = 1.0f / std::pow(config_.base,
          static_cast<float>(2 * j) / static_cast<float>(config_.rotary_dim));
      // Cache rebuilding rounds inverse frequencies to the cache dtype.
      const float stored_inv = RoundTo(inv, initial ? DataType::kFLOAT32
                                                   : config_.inverse_frequency_dtype);
      const float angle = (static_cast<float>(p) * (1.0f / factors[j])) *
                          RoundTo(stored_inv, cache_dtype);
      const auto idx = p * half + j;
      cos_cache_[idx] = RoundTo(RoundTo(std::cos(angle), cache_dtype) * amplitude_, cache_dtype);
      sin_cache_[idx] = RoundTo(RoundTo(std::sin(angle), cache_dtype) * amplitude_, cache_dtype);
    }
  }
  cached_length_ = seq_len;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LongRoPE::Forward(const std::shared_ptr<Tensor> &q,
                       const std::shared_ptr<Tensor> &k,
                       const std::shared_ptr<Tensor> &position_ids) {
  CHECK(q && k && position_ids);
  CHECK_EQ(q->Dims().size(), 4);
  CHECK_EQ(k->Dims().size(), 4);
  CHECK_EQ(position_ids->Dims().size(), 2);
  const auto &s = q->Dims();
  const int64_t batch = s[0], seq = s[1], dim = config_.rotary_dim;
  CHECK_GT(batch, 0); CHECK_GT(seq, 0); CHECK_GT(s[2], 0);
  CHECK_EQ(s[3], dim);
  CHECK_EQ(k->Dims()[0], batch); CHECK_EQ(k->Dims()[1], seq);
  CHECK_GT(k->Dims()[2], 0); CHECK_EQ(k->Dims()[3], dim);
  CHECK_EQ(static_cast<int>(q->Dtype()), static_cast<int>(k->Dtype()))
      << "LongRoPE Q and K must have the same dtype";
  CHECK(q->Dtype() == DataType::kFLOAT32 || q->Dtype() == DataType::kBFLOAT16 ||
        q->Dtype() == DataType::kFLOAT16);
  CHECK(q->GetDevice() == k->GetDevice());
  CHECK(position_ids->GetDevice() == q->GetDevice());
  CHECK(position_ids->Dtype() == DataType::kINT64);
  CHECK(position_ids->Dims()[0] == batch || position_ids->Dims()[0] == 1);
  CHECK_EQ(position_ids->Dims()[1], seq);
  if (seq > cached_length_) BuildCache(seq, q->Dtype());

  std::vector<float> cos_values(batch * seq * dim / 2);
  std::vector<float> sin_values(cos_values.size());
  {
    autograd::NoGradGuard no_grad;
    auto positions = position_ids->To(Device());
    Sync(q->GetDevice());
    const auto *ids = static_cast<const int64_t *>(positions.DataPtr());
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t t = 0; t < seq; ++t) {
        const int64_t p = ids[(position_ids->Dims()[0] == 1 ? 0 : b) * seq + t];
        const auto limit = config_.position_limit == LongRoPEPositionLimit::kSequenceLength
                               ? seq : cached_length_;
        CHECK_GE(p, 0); CHECK_LT(p, limit);
        for (int64_t j = 0; j < dim / 2; ++j) {
          const auto out = (b * seq + t) * (dim / 2) + j;
          cos_values[out] = RoundTo(cos_cache_[p * (dim / 2) + j], q->Dtype());
          sin_values[out] = RoundTo(sin_cache_[p * (dim / 2) + j], q->Dtype());
        }
      }
    }
  }
  const std::vector<int64_t> coeff_shape{batch, seq, 1, dim / 2};
  auto cos = std::make_shared<Tensor>(cos_values.data(), coeff_shape,
                                      DataType::kFLOAT32, q->GetDevice());
  auto sin = std::make_shared<Tensor>(sin_values.data(), coeff_shape,
                                      DataType::kFLOAT32, q->GetDevice());
  // The constructor copies from host asynchronously on accelerators.
  Sync(q->GetDevice());
  auto autocast = GetCurrentAutocastContext();
  autocast.enabled = false;
  AutocastGuard autocast_guard(autocast);
  auto q_rotated = ApplyRotaryCoefficients(AsFloat(q), cos, sin, config_.rotary_interleaved);
  auto k_rotated = ApplyRotaryCoefficients(AsFloat(k), cos, sin, config_.rotary_interleaved);
  if (q->Dtype() != DataType::kFLOAT32) {
    q_rotated = std::make_shared<Tensor>(q_rotated->To(q->Dtype()));
    k_rotated = std::make_shared<Tensor>(k_rotated->To(k->Dtype()));
  }
  return {q_rotated, k_rotated};
}

}  // namespace infini_train

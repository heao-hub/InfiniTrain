#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/lora/lora_parallel_linear.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/model_parallel_config.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

CausalSelfAttention::CausalSelfAttention(const TransformerConfig &config)
    : CloneableModule(kType), config_(config) {
  SetupAttention(config);

  if (config_.attention_variant == AttentionVariant::kLowRankQKV) {
    CHECK_EQ(parallel::global::GetTensorParallelSize(), 1)
        << "Low-rank attention currently supports TP=1 only";
    CHECK(!parallel::global::GetSequenceParallelEnabled());
    CHECK_GT(config.q_lora_rank, 0);
    CHECK_GT(config.kv_lora_rank, 0);
    CHECK_GT(config.qk_nope_head_dim, 0);
    CHECK_GT(config.qk_rope_head_dim, 0);
    CHECK_EQ(config.qk_rope_head_dim % 2, 0);
    CHECK_GT(config.v_head_dim, 0);
    CHECK(config.position_embedding_type == PositionEmbeddingType::kRoPE);
    CHECK(!config.qk_layernorm);
    CHECK(!config.add_bias_linear);
    CHECK_EQ(config.n_kv_head, config.n_head);
    CHECK(!config.rotary_interleaved);
    const auto qdim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    modules_["q_a_proj"] = std::make_shared<Linear>(n_embd_, config.q_lora_rank, false);
    modules_["q_a_layernorm"] = std::make_shared<RMSNorm>(config.q_lora_rank, config.norm_eps);
    modules_["q_b_proj"] = std::make_shared<Linear>(config.q_lora_rank, n_head_ * qdim, false);
    modules_["kv_a_proj_with_mqa"] = std::make_shared<Linear>(n_embd_, config.kv_lora_rank + config.qk_rope_head_dim, false);
    modules_["kv_a_layernorm"] = std::make_shared<RMSNorm>(config.kv_lora_rank, config.norm_eps);
    modules_["kv_b_proj"] = std::make_shared<Linear>(config.kv_lora_rank, n_head_ * (config.qk_nope_head_dim + config.v_head_dim), false);
    modules_["o_proj"] = std::make_shared<Linear>(n_head_ * config.v_head_dim, n_embd_, false);
    return;
  }
  if (config_.qk_layernorm) {
    modules_[kQNormLayerName] =
        std::make_shared<nn::RMSNorm>(head_dim_, config_.norm_eps);
    modules_[kKNormLayerName] =
        std::make_shared<nn::RMSNorm>(head_dim_, config_.norm_eps);
    // NOTE(zbl): In Qwen3-8B, Q/K norm sees full sequences and TP-local heads.
    //            So we only need to finalize its replicated weights across TP
    //            regardless of SP.
    modules_[kQNormLayerName]
        ->parameter(RMSNorm::kParamWeightName)
        ->set_sequence_parallel(false);
    modules_[kKNormLayerName]
        ->parameter(RMSNorm::kParamWeightName)
        ->set_sequence_parallel(false);
  }

  int64_t qkv_dim = (config.n_head + 2 * n_kv_head_) * head_dim_;
  // qkv: ColumnParallel (do not gather output)
  modules_[kCAttnLayerName] = std::make_shared<
      nn::parallel::ColumnParallelLinear>(
      /*in_features=*/n_embd_,
      /*out_features=*/qkv_dim,
      /*bias=*/config_.add_bias_linear,
      /*gather_output=*/false,
      /*input_is_parallel=*/false,
      /*skip_bias_add=*/false,
      /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled(),
      /*disable_tp_comm_pipeline=*/
      nn::parallel::global::GetModelParallelConfig()
          .tp_comm_overlap_disable_qkv);

  // proj: RowParallel (input is parallel and output is full)
  modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
      /*in_features=*/n_embd_,
      /*out_features=*/n_embd_,
      /*bias=*/config_.add_bias_linear,
      /*reduce_output=*/true,
      /*input_is_parallel=*/true,
      /*skip_bias_add=*/false,
      /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

  // FIXME(zbl): Decouple causal-mask ownership from position embedding. For
  // now, only learned-absolute models use
  //             this precomputed buffer; RoPE callers provide a runtime-sized
  //             mask.
  if (config_.position_embedding_type ==
      PositionEmbeddingType::kLearnedAbsolute) {
    buffers_[kParamBiasName] =
        function::Tril(
            nn::function::Ones({config_.block_size, config_.block_size}))
            ->View({1, 1, config_.block_size, config_.block_size});
  }
}

void CausalSelfAttention::SetupAttention(const TransformerConfig &config) {
  auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

  CHECK_GT(config.n_head, 0);
  CHECK_GT(config.n_embd, 0);
  CHECK_EQ(config.n_embd % config.n_head, 0)
      << "n_embd must be divisible by n_head";
  CHECK_EQ(config.n_head % tp_world_size, 0)
      << "n_head must be divisible by TP world size";

  n_head_ = config.n_head;
  n_embd_ = config.n_embd;
  head_dim_ = config.n_embd / config.n_head;
  local_n_head_ = n_head_ / tp_world_size;

  // For GQA, set n_kv_head and n_rep
  if (config.UseGQA()) {
    CHECK_EQ(config.n_head % config.n_kv_head, 0)
        << "n_head must be divisible by n_kv_head for GQA";
    CHECK_EQ(config.n_kv_head % tp_world_size, 0)
        << "n_kv_head must be divisible by TP world size for GQA";

    n_kv_head_ = config.n_kv_head;
    n_rep_ = n_head_ / n_kv_head_;
  } else {
    n_kv_head_ = n_head_;
    n_rep_ = 1;
  }
}

ShardedStateDict CausalSelfAttention::BuildShardedStateDict(
    const ShardedStateDictBuildContext &context) const {
  auto state = Module::BuildShardedStateDict(context);
  if (config_.attention_variant == AttentionVariant::kLowRankQKV) { return state; }
  const int tp_size = parallel::global::GetTensorParallelSize();
  const int rank = parallel::tp_rank;
  const int64_t q_global = n_head_ * head_dim_;
  const int64_t kv_global = n_kv_head_ * head_dim_;
  const int64_t q_local = q_global / tp_size;
  const int64_t kv_local = kv_global / tp_size;

  // FIXME(jym): Transformer should not depend on LoRA just to identify lora_B.
  // Move the packed-QKV segment layout into a dedicated sharding abstraction
  // shared by the base attention weight and LoRA parameters.
  const auto c_attn_prefix =
      context.logical_prefix.empty()
          ? kCAttnLayerName
          : context.logical_prefix + "." + kCAttnLayerName;
  auto set_qkv_segments = [&](const std::string &parameter_name) {
    const auto key = c_attn_prefix + "." + parameter_name;
    auto it = state.tensors.find(key);
    if (it == state.tensors.end()) {
      return;
    }
    auto &tensor = it->second;
    tensor.global_offset.assign(tensor.global_shape.size(), 0);
    tensor.segments = {
        {.global_offset = rank * q_local, .local_offset = 0, .length = q_local},
        {.global_offset = q_global + rank * kv_local,
         .local_offset = q_local,
         .length = kv_local},
        {.global_offset = q_global + kv_global + rank * kv_local,
         .local_offset = q_local + kv_local,
         .length = kv_local},
    };
  };

  set_qkv_segments(parallel::ColumnParallelLinear::kParamWeightName);
  if (config_.add_bias_linear) {
    set_qkv_segments(parallel::ColumnParallelLinear::kParamBiasName);
  }
  set_qkv_segments(lora::LoRAColumnParallelLinear::kParamLoraBName);
  return state;
}

std::shared_ptr<infini_train::Tensor> CausalSelfAttention::RepeatKV(
    const std::shared_ptr<infini_train::Tensor> &x, int64_t n_rep) {
  const auto &shape = x->Dims();
  const int64_t B = shape[0], T = shape[1], H = shape[2], D = shape[3];

  if (n_rep == 1) {
    return x;
  }

  return x->View({B, T, H, 1, D})
      ->RepeatInterleave(n_rep, 3)
      ->Contiguous()
      ->View({B, T, H * n_rep, D});
}

std::vector<std::shared_ptr<infini_train::Tensor>> CausalSelfAttention::Forward(
    const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
  if (config_.attention_variant == AttentionVariant::kLowRankQKV) { return ForwardLowRank(x); }
  const auto B = x[0]->Dims()[0];  // bs
  const auto C = x[0]->Dims()[2];  // n_embd

  const auto tp_size = nn::parallel::global::GetTensorParallelSize();

  const auto C_local = C / tp_size;
  const auto H_local = local_n_head_;
  const auto KV_local = n_kv_head_ / tp_size;
  const auto D = head_dim_;  // n_embd / n_head

  const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
  const auto start_pos = x.size() > 2 ? x[2] : nullptr;
  const auto mask = x.size() > 3 ? x[3] : nullptr;
  if (config_.position_embedding_type == PositionEmbeddingType::kRoPE) {
    CHECK(freqs_cis != nullptr) << "freqs_cis is null.";
  }

  // (B, T, C) -> (B, T, (H + 2 * n_kv_head) * D)
  auto qkv = (*modules_[kCAttnLayerName])({x[0]})[0];
  // NOTE(zbl): Acquire full T after AllGather is performed in
  // ColumnParallelLinear
  const auto T = qkv->Dims()[1];
  // NOTE(zbl): torch script uses torch.split({...}, dim) to split tensors into
  // sub-tensors in different sizes
  //            use Slice() to work around here
  const int64_t q_size_local = H_local * D;
  const int64_t kv_size_local = KV_local * D;
  // -> Split into q, k, v
  // q: (B, T, H_local, D)
  auto q = qkv->Slice(2, 0, q_size_local)->View({B, T, H_local, D});
  // k: (B, T, KV_local, D)
  auto k = qkv->Slice(2, q_size_local, q_size_local + kv_size_local)
               ->View({B, T, KV_local, D});
  // v: (B, T, KV_local, D)
  auto v = qkv->Slice(2, q_size_local + kv_size_local,
                      q_size_local + 2 * kv_size_local)
               ->View({B, T, KV_local, D});
  if (config_.qk_layernorm) {
    auto q_shape = q->Dims();
    q = (*modules_[kQNormLayerName])({q->View({B * T * H_local, D})})[0]->View(
        q_shape);

    auto k_shape = k->Dims();
    k = (*modules_[kKNormLayerName])({k->View({B * T * KV_local, D})})[0]->View(
        k_shape);
  }

  if (config_.position_embedding_type == PositionEmbeddingType::kRoPE) {
    // q: (B, T, H_local, D), k: (B, T, KV_local, D)
    std::tie(q, k) =
        ApplyRotaryEmbedding(q, k, freqs_cis, config_.rotary_interleaved);
  }

  // TODO(zbl): use kv cache during inference
  // if (use_kv_) { ... }

  // align n_head in GQA
  // (B, T, KV_local, D) -> (B, T, H_local, D) via RepeatKV
  k = RepeatKV(k, n_rep_);
  v = RepeatKV(v, n_rep_);

  // (B, T, H_local, D) -> (B, H_local, T, D)
  q = q->Transpose(1, 2);
  k = k->Transpose(1, 2);
  v = v->Transpose(1, 2);

  // TODO(zbl): support flash attention later
  // if (flash_) { ... }

  // manual implementation of attention
  // this materializes the large (T,T) matrix for all the queries and keys

  // q: (B, H_local, T, D)
  // k: (B, H_local, T, D) -> (B, H_local, D, T)
  // q @ k.T: (B, H_local, T, T) -> mul 1.0 / sqrt(D) -> (B, H_local, T, T)
  auto att = q->Matmul(k->Transpose(-2, -1)) *
             (1.0 / std::sqrt(static_cast<float>(D)));
  if (mask) {
    // mask: (1, 1, T, T)
    att = att->MaskedFill(mask, std::numeric_limits<float>::lowest());
  } else {
    // fallback causal mask: (1, 1, T, T)
    auto causal_mask = buffers_[kParamBiasName]->Slice(
        {0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
    att = att->MaskedFill(causal_mask == 0,
                          -std::numeric_limits<float>::infinity());
  }
  // (B, H_local, T, T)
  att = nn::function::Softmax(att, -1);
  // att: (B, H_local, T, T) @ v: (B, H_local, T, D) -> y: (B, H_local, T, D)
  auto y = att->Matmul(v);
  // (B, H_local, T, D) -> Transpose(1, 2) -> (B, T, H_local, D) -> (B, T,
  // C_local)
  y = y->Transpose(1, 2)->Contiguous()->View({B, T, C_local});
  // output projection
  // (B, T, C_local) -> RowParallelLinear(C, C) -> (B, T, C)
  y = (*modules_[kCProjLayerName])({y})[0];
  // (B, H, C) == (bs, seq_len, n_embd)
  return {y};
}

std::vector<std::shared_ptr<Tensor>> CausalSelfAttention::ForwardLowRank(
    const std::vector<std::shared_ptr<Tensor>> &x) {
  CHECK_GT(x.size(), 3);
  CHECK(x[1]);
  CHECK(x[3]);
  const auto batch = x[0]->Dims()[0];
  const auto seq = x[0]->Dims()[1];
  const auto qdim = config_.qk_nope_head_dim + config_.qk_rope_head_dim;

  auto q = (*modules_["q_a_proj"])({x[0]})[0];
  q = (*modules_["q_a_layernorm"])({q})[0];
  q = (*modules_["q_b_proj"])({q})[0]->View({batch, seq, n_head_, qdim});
  auto q_nope = q->Slice(-1, 0, config_.qk_nope_head_dim);
  auto q_rope = q->Slice(-1, config_.qk_nope_head_dim, qdim);

  auto kv = (*modules_["kv_a_proj_with_mqa"])({x[0]})[0];
  auto k_rope = kv->Slice(-1, config_.kv_lora_rank, config_.kv_lora_rank + config_.qk_rope_head_dim)
                    ->View({batch, seq, 1, config_.qk_rope_head_dim});
  kv = kv->Slice(-1, 0, config_.kv_lora_rank);
  kv = (*modules_["kv_a_layernorm"])({kv})[0];
  kv = (*modules_["kv_b_proj"])({kv})[0]->View(
      {batch, seq, n_head_, config_.qk_nope_head_dim + config_.v_head_dim});
  auto k_nope = kv->Slice(-1, 0, config_.qk_nope_head_dim);
  auto v = kv->Slice(-1, config_.qk_nope_head_dim, config_.qk_nope_head_dim + config_.v_head_dim);

  CHECK(q_rope->Dtype() == k_rope->Dtype());
  auto freqs = std::make_shared<Tensor>(x[1]->To(q_rope->Dtype()));
  std::tie(q_rope, k_rope) = ApplyRotaryEmbedding(q_rope, k_rope, freqs, false);
  k_rope = RepeatKV(k_rope, n_head_);

  q = function::Concat({q_nope, q_rope}, -1)->Transpose(1, 2);
  auto k = function::Concat({k_nope, k_rope}, -1)->Transpose(1, 2);
  v = v->Transpose(1, 2);

  auto scores = q->Matmul(k->Transpose(-2, -1)) * (1.0f / std::sqrt(static_cast<float>(qdim)));
  scores = scores->MaskedFill(x[3], std::numeric_limits<float>::lowest());
  auto output = function::Softmax(scores, -1)->Matmul(v)->Transpose(1, 2)->Contiguous()
                    ->View({batch, seq, n_head_ * config_.v_head_dim});
                    
  return (*modules_["o_proj"])({output});
}

}  // namespace infini_train::nn

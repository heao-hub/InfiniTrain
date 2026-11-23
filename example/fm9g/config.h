#pragma once

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/vision_embedding.h"

namespace fm9g {
inline infini_train::nn::TransformerConfig TextConfig() {
  infini_train::nn::TransformerConfig config;
  config.block_size = 32768;
  config.vocab_size = 73448;
  config.original_vocab_size = 73448;
  config.n_layer = 62;
  config.n_head = 40;
  config.n_kv_head = 40;
  config.n_embd = 2560;
  config.position_embedding_type = infini_train::nn::PositionEmbeddingType::kRoPE;
  config.activation_type = infini_train::nn::MLPType::kSwiGLU;
  config.norm_type = infini_train::nn::NormType::kRMSNorm;
  config.add_bias_linear = false;
  config.add_bias_lm_head = false;
  config.tie_weights = false;
  config.ffn_hidden_size = 6400;
  config.rope_theta = 10000.0f;
  config.attention_variant = infini_train::nn::AttentionVariant::kLowRankQKV;
  config.q_lora_rank = 768;
  config.kv_lora_rank = 256;
  config.qk_nope_head_dim = 64;
  config.qk_rope_head_dim = 32;
  config.v_head_dim = 64;
  config.embedding_scale = 12.0f;
  config.scale_depth = 1.4f;
  config.dim_model_base = 256;
  config.norm_eps = 1e-5f;
  config.rope_factors = {1.0591234f, 1.1241891f, 1.2596936f, 1.5380380f,
                         2.0939825f, 3.1446936f, 4.9379526f, 7.5245420f,
                         10.4754580f, 13.0620474f, 14.8553066f, 15.9060173f,
                         16.4619617f, 16.7403069f, 16.8758106f, 16.9408760f};
  return config;
}
inline infini_train::nn::TransformerConfig TinyTextConfig() {
  auto config = TextConfig();
  config.block_size = 128;
  config.n_layer = 2;
  config.n_head = 4;
  config.n_kv_head = 4;
  config.n_embd = 128;
  config.ffn_hidden_size = 256;
  config.q_lora_rank = 32;
  config.kv_lora_rank = 16;
  config.qk_nope_head_dim = 16;
  config.qk_rope_head_dim = 8;
  config.v_head_dim = 32;
  config.rope_factors.resize(config.qk_rope_head_dim / 2);
  return config;
}

inline infini_train::nn::VisionConfig VisionConfig() {
  infini_train::nn::VisionConfig config;
  config.hidden_size = 1152;
  config.intermediate_size = 4304;
  config.num_hidden_layers = 27;
  config.num_attention_heads = 16;
  config.num_channels = 3;
  config.image_size = 980;
  config.patch_size = 14;
  config.layer_norm_eps = 1e-6f;
  return config;
}

inline void SanitizeTextConfig(const infini_train::nn::TransformerConfig &c) {
  CHECK_GT(c.n_layer, 0);
  CHECK_GT(c.n_head, 0);
  CHECK_GT(c.n_embd, 0);
  CHECK_EQ(c.n_embd % c.n_head, 0);
  CHECK_EQ(c.n_head, c.n_kv_head);
  CHECK(c.position_embedding_type == infini_train::nn::PositionEmbeddingType::kRoPE);
  CHECK(c.attention_variant == infini_train::nn::AttentionVariant::kLowRankQKV);
  CHECK(c.activation_type == infini_train::nn::MLPType::kSwiGLU);
  CHECK(c.norm_type == infini_train::nn::NormType::kRMSNorm);
  CHECK(c.ffn_hidden_size.has_value());
  CHECK_GT(c.ffn_hidden_size.value(), 0);
  CHECK_GT(c.q_lora_rank, 0);
  CHECK_GT(c.kv_lora_rank, 0);
  CHECK_GT(c.qk_nope_head_dim, 0);
  CHECK_GT(c.qk_rope_head_dim, 0);
  CHECK_EQ(c.qk_rope_head_dim % 2, 0);
  CHECK_GT(c.v_head_dim, 0);
  CHECK_EQ(c.rope_factors.size(), c.qk_rope_head_dim / 2);
  CHECK_GT(c.dim_model_base, 0);
  CHECK(c.scale_depth.has_value());
  CHECK_GT(c.scale_depth.value(), 0.0f);
  CHECK(!c.add_bias_linear);
  CHECK(!c.tie_weights);
}
}  // namespace fm9g

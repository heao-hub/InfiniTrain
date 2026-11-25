#include "infini_train/include/nn/modules/transformer/cross_attention.h"

#include <cmath>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

CrossAttention::CrossAttention(int64_t embed_dim, int64_t num_heads, bool bias)
    : CloneableModule(kType), embed_dim_(embed_dim), num_heads_(num_heads) {
  CHECK_GT(embed_dim_, 0);
  CHECK_GT(num_heads_, 0);
  CHECK_EQ(embed_dim_ % num_heads_, 0);
  modules_["q_proj"] = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
  modules_["k_proj"] = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
  modules_["v_proj"] = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
  modules_["out_proj"] = std::make_shared<Linear>(embed_dim_, embed_dim_, bias);
}

std::vector<std::shared_ptr<Tensor>> CrossAttention::Forward(
    const std::vector<std::shared_ptr<Tensor>> &inputs) {
  CHECK(inputs.size() == 3 || inputs.size() == 4);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(inputs[i]);
    CHECK_EQ(inputs[i]->Dims().size(), 3);
    CHECK_EQ(inputs[i]->Dims()[2], embed_dim_);
    CHECK_EQ(inputs[i]->Dims()[0], inputs[0]->Dims()[0]);
    CHECK(inputs[i]->GetDevice() == inputs[0]->GetDevice());
    CHECK(inputs[i]->Dtype() == inputs[0]->Dtype());
  }
  const auto batch = inputs[0]->Dims()[0];
  const auto query_length = inputs[0]->Dims()[1];
  const auto key_length = inputs[1]->Dims()[1];
  CHECK_EQ(inputs[2]->Dims()[1], key_length);
  const auto head_dim = embed_dim_ / num_heads_;
  auto q = (*modules_["q_proj"])({inputs[0]})[0]
               ->View({batch, query_length, num_heads_, head_dim})
               ->Transpose(1, 2);
  auto k = (*modules_["k_proj"])({inputs[1]})[0]
               ->View({batch, key_length, num_heads_, head_dim})
               ->Transpose(1, 2);
  auto v = (*modules_["v_proj"])({inputs[2]})[0]
               ->View({batch, key_length, num_heads_, head_dim})
               ->Transpose(1, 2);
  auto scores = q->Matmul(k->Transpose(-2, -1)) *
                (1.0f / std::sqrt(static_cast<float>(head_dim)));
  if (inputs.size() == 4 && inputs[3]) {
    CHECK(inputs[3]->Dims() ==
          (std::vector<int64_t>{batch, 1, query_length, key_length}));
    CHECK(inputs[3]->GetDevice() == inputs[0]->GetDevice());
    CHECK(inputs[3]->Dtype() == inputs[0]->Dtype());
    scores = scores + inputs[3];
  }
  auto weights = function::Softmax(scores, -1);
  auto output = weights->Matmul(v)
                    ->Transpose(1, 2)
                    ->Contiguous()
                    ->View({batch, query_length, embed_dim_});
  return (*modules_["out_proj"])({output});
}

}  // namespace infini_train::nn

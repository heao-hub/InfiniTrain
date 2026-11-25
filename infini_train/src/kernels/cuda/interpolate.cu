#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"
#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {
constexpr int kThreads = 256;

cudaStream_t Stream(Device device) {
  return dynamic_cast<core::cuda::CudaStream *>(
             core::GetDeviceGuardImpl(device.type())->GetStream(device))
      ->cuda_stream();
}

__device__ float Coordinate(int64_t index, int64_t in, int64_t out,
                            bool align) {
  if (align) return out > 1 ? static_cast<float>(index) * (in - 1) / (out - 1) : 0.0f;
  return fmaxf(0.0f, (index + 0.5f) * in / out - 0.5f);
}

__global__ void InterpolateKernel(const float *src, float *dst, int64_t n,
                                  int64_t ih, int64_t iw, int64_t oh,
                                  int64_t ow, bool align, bool backward) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const int64_t x = idx % ow, y = (idx / ow) % oh, plane = idx / (ow * oh);
  const float fy = Coordinate(y, ih, oh, align), fx = Coordinate(x, iw, ow, align);
  const int64_t y0 = min(static_cast<int64_t>(fy), ih - 1), y1 = min(y0 + 1, ih - 1);
  const int64_t x0 = min(static_cast<int64_t>(fx), iw - 1), x1 = min(x0 + 1, iw - 1);
  const float dy = fy - y0, dx = fx - x0;
  const int64_t base = plane * ih * iw;
  const int64_t ids[4] = {base + y0 * iw + x0, base + y0 * iw + x1,
                          base + y1 * iw + x0, base + y1 * iw + x1};
  const float w[4] = {(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx};
  if (backward) {
    for (int i = 0; i < 4; ++i) atomicAdd(dst + ids[i], src[idx] * w[i]);
  } else {
    float v = 0;
    for (int i = 0; i < 4; ++i) v += src[ids[i]] * w[i];
    dst[idx] = v;
  }
}

}  // namespace

std::shared_ptr<Tensor> InterpolateBilinear2dForward(std::shared_ptr<Tensor> input,
    std::vector<int64_t> size, bool align) {
  core::DeviceGuard guard(input->GetDevice());
  if (input->Dtype() == DataType::kBFLOAT16 || input->Dtype() == DataType::kFLOAT16) {
    const auto dtype = input->Dtype();
    auto fp32 = std::make_shared<Tensor>(input->Contiguous()->To(DataType::kFLOAT32));
    auto result = InterpolateBilinear2dForward(fp32, size, align);
    return std::make_shared<Tensor>(result->To(dtype));
  }
  CHECK(input->Dtype() == DataType::kFLOAT32);
  const auto d = input->Dims(); CHECK_EQ(d.size(), 4); CHECK_EQ(size.size(), 2);
  CHECK_GT(d[2], 0); CHECK_GT(d[3], 0); CHECK_GT(size[0], 0); CHECK_GT(size[1], 0);
  auto x = input->Contiguous();
  auto out = std::make_shared<Tensor>(std::vector<int64_t>{d[0],d[1],size[0],size[1]}, input->Dtype(), input->GetDevice());
  const auto n = out->NumElements();
  if (n) { InterpolateKernel<<<(n+kThreads-1)/kThreads,kThreads,0,Stream(input->GetDevice())>>>(static_cast<const float *>(x->DataPtr()),static_cast<float *>(out->DataPtr()),n,d[2],d[3],size[0],size[1],align,false); CUDA_CHECK(cudaGetLastError()); }
  return out;
}
std::shared_ptr<Tensor> InterpolateBilinear2dBackward(std::shared_ptr<Tensor> grad,
    std::vector<int64_t> dims, bool align) {
  core::DeviceGuard guard(grad->GetDevice());
  if (grad->Dtype() == DataType::kBFLOAT16 || grad->Dtype() == DataType::kFLOAT16) {
    const auto dtype = grad->Dtype();
    auto fp32 = std::make_shared<Tensor>(grad->Contiguous()->To(DataType::kFLOAT32));
    auto result = InterpolateBilinear2dBackward(fp32, dims, align);
    return std::make_shared<Tensor>(result->To(dtype));
  }
  CHECK(grad->Dtype() == DataType::kFLOAT32); CHECK_EQ(dims.size(), 4);
  auto g = grad->Contiguous(); auto out = std::make_shared<Tensor>(dims,grad->Dtype(),grad->GetDevice());
  auto stream = Stream(grad->GetDevice());
  if (out->SizeInBytes()) CUDA_CHECK(cudaMemsetAsync(out->DataPtr(),0,out->SizeInBytes(),stream));
  const auto n = g->NumElements();
  if (n) { InterpolateKernel<<<(n+kThreads-1)/kThreads,kThreads,0,stream>>>(static_cast<const float *>(g->DataPtr()),static_cast<float *>(out->DataPtr()),n,dims[2],dims[3],g->Dims()[2],g->Dims()[3],align,true); CUDA_CHECK(cudaGetLastError()); }
  return out;
}
}  // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, InterpolateBilinear2dForward, infini_train::kernels::cuda::InterpolateBilinear2dForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, InterpolateBilinear2dBackward, infini_train::kernels::cuda::InterpolateBilinear2dBackward)

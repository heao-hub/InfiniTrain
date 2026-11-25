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

__global__ void RoiKernel(const float *src, float *dst, const float *boxes,
                          int64_t total, int64_t batches, int64_t channels, int64_t height,
                          int64_t width, int64_t ph, int64_t pw, float scale,
                          int64_t ratio, bool aligned, bool backward) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t px = idx % pw, py = (idx / pw) % ph;
  const int64_t c = (idx / pw / ph) % channels, r = idx / pw / ph / channels;
  const float *box = boxes + r * 5;
  const bool valid = isfinite(box[0]) && box[0] >= 0 && box[0] < batches &&
                     floorf(box[0]) == box[0] && isfinite(box[1]) &&
                     isfinite(box[2]) && isfinite(box[3]) && isfinite(box[4]);
  assert(valid);
  if (!valid) { if (!backward) dst[idx] = nanf(""); return; }
  const int64_t batch = static_cast<int64_t>(box[0]);
  const float off = aligned ? 0.5f : 0.0f;
  const float x0 = box[1] * scale - off, y0 = box[2] * scale - off;
  float rw = box[3] * scale - off - x0, rh = box[4] * scale - off - y0;
  if (!aligned) { rw = fmaxf(rw, 1.0f); rh = fmaxf(rh, 1.0f); }
  assert(rw > 0 && rh > 0);
  const int64_t nx = ratio > 0 ? ratio : static_cast<int64_t>(ceilf(rw / pw));
  const int64_t ny = ratio > 0 ? ratio : static_cast<int64_t>(ceilf(rh / ph));
  if (nx <= 0 || ny <= 0) { if (!backward) dst[idx] = 0; return; }
  const int64_t base = (batch * channels + c) * height * width;
  float sum = 0;
  for (int64_t iy = 0; iy < ny; ++iy) for (int64_t ix = 0; ix < nx; ++ix) {
    float y = y0 + (py + (iy + 0.5f) / ny) * rh / ph;
    float x = x0 + (px + (ix + 0.5f) / nx) * rw / pw;
    if (y < -1 || y > height || x < -1 || x > width) continue;
    y = fminf(fmaxf(y, 0), height - 1); x = fminf(fmaxf(x, 0), width - 1);
    const int64_t yl = static_cast<int64_t>(y), xl = static_cast<int64_t>(x);
    const int64_t yh = min(yl + 1, height - 1), xh = min(xl + 1, width - 1);
    const float dy = y - yl, dx = x - xl;
    const int64_t ids[4] = {base+yl*width+xl, base+yl*width+xh, base+yh*width+xl, base+yh*width+xh};
    const float w[4] = {(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx};
    for (int i = 0; i < 4; ++i) {
      if (backward) atomicAdd(dst + ids[i], src[idx] * w[i] / (nx * ny));
      else sum += src[ids[i]] * w[i] / (nx * ny);
    }
  }
  if (!backward) dst[idx] = sum;
}
}  // namespace

std::shared_ptr<Tensor> RoiAlignForward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> boxes,
    std::vector<int64_t> size, float scale, int64_t ratio, bool aligned) {
  core::DeviceGuard guard(input->GetDevice());
  if (input->Dtype() == DataType::kBFLOAT16 || input->Dtype() == DataType::kFLOAT16) {
    const auto dtype = input->Dtype();
    auto fp32 = std::make_shared<Tensor>(input->Contiguous()->To(DataType::kFLOAT32));
    auto result = RoiAlignForward(fp32, boxes, size, scale, ratio, aligned);
    return std::make_shared<Tensor>(result->To(dtype));
  }
  CHECK(input->Dtype() == DataType::kFLOAT32); CHECK(boxes->Dtype() == DataType::kFLOAT32);
  CHECK(input->GetDevice() == boxes->GetDevice());
  auto x=input->Contiguous(); auto b=boxes->Contiguous(); const auto d=input->Dims();
  CHECK_EQ(d.size(), 4); CHECK_GT(d[2], 0); CHECK_GT(d[3], 0);
  CHECK_EQ(boxes->Dims().size(), 2); CHECK_EQ(boxes->Dims()[1], 5);
  CHECK_EQ(size.size(), 2); CHECK_GT(size[0], 0); CHECK_GT(size[1], 0);
  auto out=std::make_shared<Tensor>(std::vector<int64_t>{boxes->Dims()[0],d[1],size[0],size[1]},input->Dtype(),input->GetDevice());
  const auto n=out->NumElements();
  if (n) { RoiKernel<<<(n+kThreads-1)/kThreads,kThreads,0,Stream(input->GetDevice())>>>(static_cast<const float *>(x->DataPtr()),static_cast<float *>(out->DataPtr()),static_cast<const float *>(b->DataPtr()),n,d[0],d[1],d[2],d[3],size[0],size[1],scale,ratio,aligned,false); CUDA_CHECK(cudaGetLastError()); }
  return out;
}
std::shared_ptr<Tensor> RoiAlignBackward(std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> boxes,
    std::vector<int64_t> dims, std::vector<int64_t> size, float scale, int64_t ratio, bool aligned) {
  core::DeviceGuard guard(grad->GetDevice());
  if (grad->Dtype() == DataType::kBFLOAT16 || grad->Dtype() == DataType::kFLOAT16) {
    const auto dtype = grad->Dtype();
    auto fp32 = std::make_shared<Tensor>(grad->Contiguous()->To(DataType::kFLOAT32));
    auto result = RoiAlignBackward(fp32, boxes, dims, size, scale, ratio, aligned);
    return std::make_shared<Tensor>(result->To(dtype));
  }
  CHECK(grad->Dtype() == DataType::kFLOAT32); CHECK(boxes->Dtype() == DataType::kFLOAT32);
  CHECK(grad->GetDevice() == boxes->GetDevice());
  CHECK_EQ(dims.size(), 4); CHECK_GT(dims[2], 0); CHECK_GT(dims[3], 0);
  CHECK_EQ(boxes->Dims().size(), 2); CHECK_EQ(boxes->Dims()[1], 5);
  CHECK_EQ(size.size(), 2); CHECK_GT(size[0], 0); CHECK_GT(size[1], 0);
  CHECK_EQ(grad->Dims().size(), 4);
  CHECK_EQ(grad->Dims()[0], boxes->Dims()[0]);
  CHECK_EQ(grad->Dims()[1], dims[1]);
  CHECK_EQ(grad->Dims()[2], size[0]); CHECK_EQ(grad->Dims()[3], size[1]);
  auto g=grad->Contiguous(); auto b=boxes->Contiguous(); auto out=std::make_shared<Tensor>(dims,grad->Dtype(),grad->GetDevice());
  auto stream=Stream(grad->GetDevice()); if (out->SizeInBytes()) CUDA_CHECK(cudaMemsetAsync(out->DataPtr(),0,out->SizeInBytes(),stream));
  const auto n=g->NumElements();
  if (n) { RoiKernel<<<(n+kThreads-1)/kThreads,kThreads,0,stream>>>(static_cast<const float *>(g->DataPtr()),static_cast<float *>(out->DataPtr()),static_cast<const float *>(b->DataPtr()),n,dims[0],dims[1],dims[2],dims[3],size[0],size[1],scale,ratio,aligned,true); CUDA_CHECK(cudaGetLastError()); }
  return out;
}
}  // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, RoiAlignForward, infini_train::kernels::cuda::RoiAlignForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, RoiAlignBackward, infini_train::kernels::cuda::RoiAlignBackward)

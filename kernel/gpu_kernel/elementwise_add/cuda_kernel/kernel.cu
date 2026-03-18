#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename scalar_t>
__global__ void add_kernel(const scalar_t* __restrict__ a,
                           const scalar_t* __restrict__ b,
                           scalar_t* __restrict__ y,
                           int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

}  // namespace

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "kc_add: expected CUDA tensors");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "kc_add: expected contiguous tensors");
  TORCH_CHECK(a.sizes() == b.sizes(), "kc_add: shape mismatch");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "kc_add: dtype mismatch");

  const auto n = a.numel();
  auto y = torch::empty_like(a);

  const c10::cuda::CUDAGuard device_guard(a.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int BLOCK = 256;
  const int64_t grid = (n + BLOCK - 1) / BLOCK;

  if (a.scalar_type() == at::kFloat) {
    add_kernel<float><<<grid, BLOCK, 0, stream>>>(
        (const float*)a.data_ptr<float>(), (const float*)b.data_ptr<float>(), (float*)y.data_ptr<float>(), n);
  } else if (a.scalar_type() == at::kHalf) {
    add_kernel<at::Half><<<grid, BLOCK, 0, stream>>>(
        (const at::Half*)a.data_ptr<at::Half>(), (const at::Half*)b.data_ptr<at::Half>(), (at::Half*)y.data_ptr<at::Half>(), n);
  } else {
    TORCH_CHECK(false, "kc_add: only supports float16/float32");
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kc_add: CUDA kernel failed: ", cudaGetErrorString(err));
  return y;
}


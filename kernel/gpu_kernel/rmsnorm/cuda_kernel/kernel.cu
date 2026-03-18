#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

namespace {

template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[BLOCK];
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride >= 1; stride >>= 1) {
    if (tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  return smem[0];
}

template <typename scalar_t, int BLOCK>
__global__ void rmsnorm_lastdim_kernel(const scalar_t* __restrict__ x,
                                       const scalar_t* __restrict__ w,
                                       scalar_t* __restrict__ y,
                                       int rows,
                                       int cols,
                                       float eps) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const int base = row * cols;

  float sumsq = 0.0f;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float v = (float)x[base + j];
    sumsq += v * v;
  }
  const float ss = block_reduce_sum<BLOCK>(sumsq);
  const float inv = rsqrtf(ss / (float)cols + eps);

  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float v = (float)x[base + j];
    const float g = (float)w[j];
    y[base + j] = (scalar_t)(v * inv * g);
  }
}

}  // namespace

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "kc_rmsnorm: expected CUDA tensors");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "kc_rmsnorm: expected contiguous tensors");
  TORCH_CHECK(x.dim() >= 1, "kc_rmsnorm: expected x.dim() >= 1");
  TORCH_CHECK(weight.dim() == 1, "kc_rmsnorm: expected weight 1D");
  TORCH_CHECK(x.size(x.dim() - 1) == weight.numel(), "kc_rmsnorm: weight length mismatch");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "kc_rmsnorm: dtype mismatch");
  const int cols = (int)weight.numel();
  TORCH_CHECK(cols > 0, "kc_rmsnorm: cols must be > 0");
  int64_t rows64 = x.numel() / cols;
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "kc_rmsnorm: invalid rows");
  const int rows = (int)rows64;

  auto y = torch::empty_like(x);
  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int BLOCK = 256;
  dim3 grid(rows);
  dim3 block(BLOCK);

  const float eps_f = (float)eps;
  if (x.scalar_type() == at::kFloat) {
    rmsnorm_lastdim_kernel<float, BLOCK><<<grid, block, 0, stream>>>(
        (const float*)x.data_ptr<float>(),
        (const float*)weight.data_ptr<float>(),
        (float*)y.data_ptr<float>(),
        rows,
        cols,
        eps_f);
  } else if (x.scalar_type() == at::kHalf) {
    rmsnorm_lastdim_kernel<at::Half, BLOCK><<<grid, block, 0, stream>>>(
        (const at::Half*)x.data_ptr<at::Half>(),
        (const at::Half*)weight.data_ptr<at::Half>(),
        (at::Half*)y.data_ptr<at::Half>(),
        rows,
        cols,
        eps_f);
  } else {
    TORCH_CHECK(false, "kc_rmsnorm: only supports float16/float32");
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kc_rmsnorm: CUDA kernel failed: ", cudaGetErrorString(err));
  return y;
}


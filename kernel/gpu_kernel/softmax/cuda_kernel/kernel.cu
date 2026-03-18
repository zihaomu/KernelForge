#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

namespace {

__device__ __forceinline__ float fmaxf2(float a, float b) { return a > b ? a : b; }

template <int BLOCK>
__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float smem[BLOCK];
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride >= 1; stride >>= 1) {
    if (tid < stride) smem[tid] = fmaxf2(smem[tid], smem[tid + stride]);
    __syncthreads();
  }
  return smem[0];
}

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
__global__ void softmax_lastdim_kernel(const scalar_t* __restrict__ x,
                                       scalar_t* __restrict__ y,
                                       int rows,
                                       int cols) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;

  // 1) max
  float vmax = -INFINITY;
  const int base = row * cols;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    vmax = fmaxf2(vmax, (float)x[base + j]);
  }
  const float maxv = block_reduce_max<BLOCK>(vmax);

  // 2) sum(exp(x-max))
  float vsum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    vsum += expf((float)x[base + j] - maxv);
  }
  const float sumv = block_reduce_sum<BLOCK>(vsum);

  // 3) write
  const float inv = 1.0f / sumv;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float e = expf((float)x[base + j] - maxv) * inv;
    y[base + j] = (scalar_t)e;
  }
}

template <int BLOCK>
__global__ void softmax_lastdim_kernel_half(const at::Half* __restrict__ x,
                                            at::Half* __restrict__ y,
                                            int rows,
                                            int cols) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;

  float vmax = -INFINITY;
  const int base = row * cols;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    vmax = fmaxf2(vmax, (float)x[base + j]);
  }
  const float maxv = block_reduce_max<BLOCK>(vmax);

  float vsum = 0.0f;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    vsum += expf((float)x[base + j] - maxv);
  }
  const float sumv = block_reduce_sum<BLOCK>(vsum);

  const float inv = 1.0f / sumv;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float e = expf((float)x[base + j] - maxv) * inv;
    y[base + j] = (at::Half)e;
  }
}

}  // namespace

torch::Tensor softmax_forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "kc_softmax: expected CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "kc_softmax: expected contiguous tensor");
  TORCH_CHECK(x.dim() >= 1, "kc_softmax: expected dim >= 1");

  const auto cols = (int)x.size(x.dim() - 1);
  TORCH_CHECK(cols > 0, "kc_softmax: last dim must be > 0");

  // Flatten all leading dims into rows.
  int64_t rows64 = 1;
  for (int i = 0; i < x.dim() - 1; ++i) rows64 *= x.size(i);
  TORCH_CHECK(rows64 > 0, "kc_softmax: invalid rows");
  TORCH_CHECK(rows64 <= INT32_MAX, "kc_softmax: rows too large");
  const int rows = (int)rows64;

  auto y = torch::empty_like(x);

  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int BLOCK = 256;
  dim3 grid(rows);
  dim3 block(BLOCK);

  if (x.scalar_type() == at::kFloat) {
    softmax_lastdim_kernel<float, BLOCK><<<grid, block, 0, stream>>>(
        (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), rows, cols);
  } else if (x.scalar_type() == at::kHalf) {
    softmax_lastdim_kernel_half<BLOCK><<<grid, block, 0, stream>>>(
        (const at::Half*)x.data_ptr<at::Half>(), (at::Half*)y.data_ptr<at::Half>(), rows, cols);
  } else {
    TORCH_CHECK(false, "kc_softmax: only supports float16/float32");
  }

  // Best-effort error check to surface launch issues early.
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kc_softmax: CUDA kernel failed: ", cudaGetErrorString(err));
  return y;
}


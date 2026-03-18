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

template <int BLOCK>
__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float smem[BLOCK];
  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = BLOCK / 2; stride >= 1; stride >>= 1) {
    if (tid < stride) smem[tid] = smem[tid] > smem[tid + stride] ? smem[tid] : smem[tid + stride];
    __syncthreads();
  }
  return smem[0];
}

template <typename scalar_t, int BLOCK>
__global__ void reduce_sum_lastdim_kernel(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ y,
                                          int rows,
                                          int cols) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const int base = row * cols;
  float acc = 0.0f;
  for (int j = threadIdx.x; j < cols; j += BLOCK) acc += (float)x[base + j];
  const float sum = block_reduce_sum<BLOCK>(acc);
  if (threadIdx.x == 0) y[row] = (scalar_t)sum;
}

template <int BLOCK>
__global__ void reduce_sum_lastdim_kernel_half(const at::Half* __restrict__ x,
                                               at::Half* __restrict__ y,
                                               int rows,
                                               int cols) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const int base = row * cols;
  float acc = 0.0f;
  for (int j = threadIdx.x; j < cols; j += BLOCK) acc += (float)x[base + j];
  const float sum = block_reduce_sum<BLOCK>(acc);
  if (threadIdx.x == 0) y[row] = (at::Half)sum;
}

template <typename scalar_t, int BLOCK>
__global__ void reduce_max_lastdim_kernel(const scalar_t* __restrict__ x,
                                          scalar_t* __restrict__ y,
                                          int rows,
                                          int cols) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const int base = row * cols;
  float vmax = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float v = (float)x[base + j];
    vmax = vmax > v ? vmax : v;
  }
  const float m = block_reduce_max<BLOCK>(vmax);
  if (threadIdx.x == 0) y[row] = (scalar_t)m;
}

template <int BLOCK>
__global__ void reduce_max_lastdim_kernel_half(const at::Half* __restrict__ x,
                                               at::Half* __restrict__ y,
                                               int rows,
                                               int cols) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const int base = row * cols;
  float vmax = -INFINITY;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float v = (float)x[base + j];
    vmax = vmax > v ? vmax : v;
  }
  const float m = block_reduce_max<BLOCK>(vmax);
  if (threadIdx.x == 0) y[row] = (at::Half)m;
}

static std::vector<int64_t> out_sizes_lastdim_reduced(const torch::Tensor& x) {
  std::vector<int64_t> out_sizes;
  out_sizes.reserve(x.dim() > 0 ? x.dim() - 1 : 0);
  for (int i = 0; i < x.dim() - 1; ++i) out_sizes.push_back(x.size(i));
  return out_sizes;
}

}  // namespace

torch::Tensor reduce_sum_lastdim(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "kc_reduce_sum: expected CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "kc_reduce_sum: expected contiguous tensor");
  TORCH_CHECK(x.dim() >= 1, "kc_reduce_sum: expected dim >= 1");
  const int cols = (int)x.size(x.dim() - 1);
  TORCH_CHECK(cols > 0, "kc_reduce_sum: last dim must be > 0");
  int64_t rows64 = x.numel() / cols;
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "kc_reduce_sum: invalid rows");
  const int rows = (int)rows64;

  auto y = torch::empty(out_sizes_lastdim_reduced(x), x.options());

  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int BLOCK = 256;
  dim3 grid(rows);
  dim3 block(BLOCK);
  if (x.scalar_type() == at::kFloat) {
    reduce_sum_lastdim_kernel<float, BLOCK><<<grid, block, 0, stream>>>(
        (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), rows, cols);
  } else if (x.scalar_type() == at::kHalf) {
    reduce_sum_lastdim_kernel_half<BLOCK><<<grid, block, 0, stream>>>(
        (const at::Half*)x.data_ptr<at::Half>(), (at::Half*)y.data_ptr<at::Half>(), rows, cols);
  } else {
    TORCH_CHECK(false, "kc_reduce_sum: only supports float16/float32");
  }
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kc_reduce_sum: CUDA kernel failed: ", cudaGetErrorString(err));
  return y;
}

torch::Tensor reduce_max_lastdim(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "kc_reduce_max: expected CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "kc_reduce_max: expected contiguous tensor");
  TORCH_CHECK(x.dim() >= 1, "kc_reduce_max: expected dim >= 1");
  const int cols = (int)x.size(x.dim() - 1);
  TORCH_CHECK(cols > 0, "kc_reduce_max: last dim must be > 0");
  int64_t rows64 = x.numel() / cols;
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "kc_reduce_max: invalid rows");
  const int rows = (int)rows64;

  auto y = torch::empty(out_sizes_lastdim_reduced(x), x.options());

  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int BLOCK = 256;
  dim3 grid(rows);
  dim3 block(BLOCK);
  if (x.scalar_type() == at::kFloat) {
    reduce_max_lastdim_kernel<float, BLOCK><<<grid, block, 0, stream>>>(
        (const float*)x.data_ptr<float>(), (float*)y.data_ptr<float>(), rows, cols);
  } else if (x.scalar_type() == at::kHalf) {
    reduce_max_lastdim_kernel_half<BLOCK><<<grid, block, 0, stream>>>(
        (const at::Half*)x.data_ptr<at::Half>(), (at::Half*)y.data_ptr<at::Half>(), rows, cols);
  } else {
    TORCH_CHECK(false, "kc_reduce_max: only supports float16/float32");
  }
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kc_reduce_max: CUDA kernel failed: ", cudaGetErrorString(err));
  return y;
}


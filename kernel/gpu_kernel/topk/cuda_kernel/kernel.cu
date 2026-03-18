#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

namespace {

__device__ __forceinline__ bool better_f(float a, float b, bool largest) { return largest ? (a > b) : (a < b); }

template <int BLOCK>
__device__ __forceinline__ void __syncthreads_if() {
  __syncthreads();
}

template <typename scalar_t, int BLOCK>
__global__ void topk_lastdim_kernel(const scalar_t* __restrict__ x,
                                    scalar_t* __restrict__ out_vals,
                                    int64_t* __restrict__ out_idx,
                                    int rows,
                                    int cols,
                                    int k,
                                    bool largest) {
  const int row = (int)blockIdx.x;
  if (row >= rows) return;

  // Fixed-size local buffer (k <= 32).
  float vbuf[32];
  int ibuf[32];

  const float worst_init = largest ? -INFINITY : INFINITY;
  for (int i = 0; i < 32; ++i) {
    vbuf[i] = worst_init;
    ibuf[i] = -1;
  }

  const int base = row * cols;
  for (int j = threadIdx.x; j < cols; j += BLOCK) {
    const float v = (float)x[base + j];
    if (!better_f(v, vbuf[k - 1], largest)) continue;

    int pos = k - 1;
    while (pos > 0 && better_f(v, vbuf[pos - 1], largest)) {
      vbuf[pos] = vbuf[pos - 1];
      ibuf[pos] = ibuf[pos - 1];
      --pos;
    }
    vbuf[pos] = v;
    ibuf[pos] = j;
  }

  // Write local candidates to shared memory for a simple merge.
  extern __shared__ char smem_raw[];
  float* s_vals = reinterpret_cast<float*>(smem_raw);
  int* s_idx = reinterpret_cast<int*>(s_vals + BLOCK * k);
  const int tid = threadIdx.x;
  for (int i = 0; i < k; ++i) {
    s_vals[tid * k + i] = vbuf[i];
    s_idx[tid * k + i] = ibuf[i];
  }
  __syncthreads();

  if (tid == 0) {
    float gvals[32];
    int gidx[32];
    for (int i = 0; i < k; ++i) {
      gvals[i] = worst_init;
      gidx[i] = -1;
    }

    for (int t = 0; t < BLOCK; ++t) {
      for (int i = 0; i < k; ++i) {
        const float v = s_vals[t * k + i];
        const int idx = s_idx[t * k + i];
        if (idx < 0) continue;
        if (!better_f(v, gvals[k - 1], largest)) continue;
        int pos = k - 1;
        while (pos > 0 && better_f(v, gvals[pos - 1], largest)) {
          gvals[pos] = gvals[pos - 1];
          gidx[pos] = gidx[pos - 1];
          --pos;
        }
        gvals[pos] = v;
        gidx[pos] = idx;
      }
    }

    // Write outputs.
    for (int i = 0; i < k; ++i) {
      out_vals[row * k + i] = (scalar_t)gvals[i];
      out_idx[row * k + i] = (int64_t)gidx[i];
    }
  }
}

static std::vector<int64_t> out_sizes_topk(const torch::Tensor& x, int64_t k) {
  std::vector<int64_t> out_sizes;
  out_sizes.reserve(x.dim());
  for (int i = 0; i < x.dim() - 1; ++i) out_sizes.push_back(x.size(i));
  out_sizes.push_back(k);
  return out_sizes;
}

}  // namespace

std::vector<torch::Tensor> topk_lastdim(torch::Tensor x, int64_t k, bool largest) {
  TORCH_CHECK(x.is_cuda(), "kc_topk: expected CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "kc_topk: expected contiguous tensor");
  TORCH_CHECK(x.dim() >= 1, "kc_topk: expected dim >= 1");
  TORCH_CHECK(k > 0, "kc_topk: k must be > 0");
  TORCH_CHECK(k <= 32, "kc_topk(v1): k must be <= 32");
  const int cols = (int)x.size(x.dim() - 1);
  TORCH_CHECK(cols > 0, "kc_topk: last dim must be > 0");
  TORCH_CHECK(k <= cols, "kc_topk: k must be <= last dim");

  int64_t rows64 = x.numel() / cols;
  TORCH_CHECK(rows64 > 0 && rows64 <= INT32_MAX, "kc_topk: invalid rows");
  const int rows = (int)rows64;

  auto values = torch::empty(out_sizes_topk(x, k), x.options());
  auto indices = torch::empty(out_sizes_topk(x, k), x.options().dtype(torch::kInt64));

  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  constexpr int BLOCK = 256;
  dim3 grid(rows);
  dim3 block(BLOCK);
  const size_t shmem = (size_t)BLOCK * (size_t)k * (sizeof(float) + sizeof(int));

  if (x.scalar_type() == at::kFloat) {
    topk_lastdim_kernel<float, BLOCK><<<grid, block, shmem, stream>>>(
        (const float*)x.data_ptr<float>(),
        (float*)values.data_ptr<float>(),
        (int64_t*)indices.data_ptr<int64_t>(),
        rows,
        cols,
        (int)k,
        largest);
  } else if (x.scalar_type() == at::kHalf) {
    topk_lastdim_kernel<at::Half, BLOCK><<<grid, block, shmem, stream>>>(
        (const at::Half*)x.data_ptr<at::Half>(),
        (at::Half*)values.data_ptr<at::Half>(),
        (int64_t*)indices.data_ptr<int64_t>(),
        rows,
        cols,
        (int)k,
        largest);
  } else {
    TORCH_CHECK(false, "kc_topk: only supports float16/float32");
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kc_topk: CUDA kernel failed: ", cudaGetErrorString(err));
  return {values, indices};
}

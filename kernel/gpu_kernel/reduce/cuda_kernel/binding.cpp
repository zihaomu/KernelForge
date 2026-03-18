#include <torch/extension.h>

torch::Tensor reduce_sum_lastdim(torch::Tensor x);
torch::Tensor reduce_max_lastdim(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_sum_lastdim", &reduce_sum_lastdim, "kc reduce sum (CUDA, dim=-1)");
  m.def("reduce_max_lastdim", &reduce_max_lastdim, "kc reduce max (CUDA, dim=-1)");
}


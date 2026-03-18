#include <torch/extension.h>

std::vector<torch::Tensor> topk_lastdim(torch::Tensor x, int64_t k, bool largest);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_lastdim", &topk_lastdim, "kc topk lastdim (CUDA)");
}


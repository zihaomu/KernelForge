#include <torch/extension.h>

#include <vector>

torch::Tensor softmax_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_forward", &softmax_forward, "kc softmax forward (CUDA, dim=-1)");
}


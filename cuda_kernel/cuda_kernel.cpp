// grouped_cumprod/grouped_cumprod.cpp
#include <torch/extension.h>

// CUDA関数を宣言
void grouped_cumprod_forward(torch::Tensor x, torch::Tensor key, torch::Tensor y);
void grouped_cumprod_backward(
    torch::Tensor param,
    torch::Tensor param_cumprod,
    torch::Tensor grad_out,
    torch::Tensor inv,
    torch::Tensor grad_in,
    torch::Tensor inv_len
);

// PyTorchに登録
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_cumprod_forward", &grouped_cumprod_forward, "Grouped Cumulative Product (CUDA)");
    m.def("grouped_cumprod_backward", &grouped_cumprod_backward,
          "Grouped Cumulative Product (CUDA backward)");
}


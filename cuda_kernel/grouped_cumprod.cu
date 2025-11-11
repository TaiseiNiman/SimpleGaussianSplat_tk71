#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

void grouped_cumprod(torch::Tensor unti_opacity, torch::Tensor pixel_index, torch::Tensor out) {

    float* u = unti_opacity.data_ptr<float>();
    long* p = pixel_index.data_ptr<long>();
    float* o = out.data_ptr<float>();

    auto p_begin = thrust::device_pointer_cast(p);
    auto p_end   = thrust::device_pointer_cast(p + pixel_index.numel());
    auto u_begin = thrust::device_pointer_cast(u);
    auto o_begin = thrust::device_pointer_cast(o);

    thrust::inclusive_scan_by_key(
        p_begin, p_end,
        u_begin,
        o_begin,
        thrust::equal_to<long>(),
        thrust::multiplies<float>()
    );
}

// Pythonからアクセスできるよう登録
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_cumprod", &grouped_cumprod, "Grouped Cumulative Product (CUDA)");
}

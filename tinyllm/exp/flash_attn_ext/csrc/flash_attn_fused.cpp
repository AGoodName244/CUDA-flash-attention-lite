// flash_attn.cpp

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor flash_attn_forward_online(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "flash_attn_forward_online",
        &flash_attn_forward_online,
        "FlashAttention-style online softmax forward, "
        "tiled over (Tq, Tkv) with warp-level reduction (float32 only)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("causal")
    );
}

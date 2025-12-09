// flash_attn.cpp
//
// Split version: QK/AV via at::matmul (cuBLAS + Tensor Core),

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

// softmax launchers implemented in flash_attn_cuda.cu
extern "C" void softmax_cuda_launcher_fp32(
    float* scores,
    int B, int H, int Tq, int Tkv,
    bool causal
);

extern "C" void softmax_cuda_launcher_bf16(
    __nv_bfloat16* scores,
    int B, int H, int Tq, int Tkv,
    bool causal
);

torch::Tensor flash_attn_forward_cuda(
    torch::Tensor q, // [B,H,Tq,D]
    torch::Tensor k, // [B,H,Tkv,D]
    torch::Tensor v, // [B,H,Tkv,D]
    double scale,
    bool causal
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");

    auto dtype = q.scalar_type();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kBFloat16,
        "flash_attn_forward_cuda supports only float32 and bfloat16 (got ",
        static_cast<int>(dtype), ")"
    );
    TORCH_CHECK(
        k.scalar_type() == dtype && v.scalar_type() == dtype,
        "q/k/v must have the same dtype"
    );

    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Tq,D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B,H,Tkv,D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B,H,Tkv,D]");

    int64_t B   = q.size(0);
    int64_t H   = q.size(1);
    int64_t Tq  = q.size(2);
    int64_t D   = q.size(3);
    int64_t Tkv = k.size(2);

    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "k/v batch mismatch");
    TORCH_CHECK(k.size(1) == H && v.size(1) == H, "k/v head mismatch");
    TORCH_CHECK(k.size(3) == D && v.size(3) == D, "k/v D mismatch");

    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    // ===== 1) QK^T * scale =====
    auto k_t    = k.transpose(2, 3); // [B,H,Tkv,D] -> [B,H,D,Tkv]
    auto scores = at::matmul(q, k_t); // [B,H,Tq,Tkv], dtype same as q
    scores.mul_(static_cast<float>(scale));
    scores = scores.contiguous();
    if (dtype == torch::kFloat32) {
        float* scores_ptr = scores.data_ptr<float>();
        softmax_cuda_launcher_fp32(
            scores_ptr,
            static_cast<int>(B),
            static_cast<int>(H),
            static_cast<int>(Tq),
            static_cast<int>(Tkv),
            causal
        );
    } else { // bfloat16
        auto* scores_ptr = reinterpret_cast<__nv_bfloat16*>(
            scores.data_ptr<at::BFloat16>()
        );
        softmax_cuda_launcher_bf16(
            scores_ptr,
            static_cast<int>(B),
            static_cast<int>(H),
            static_cast<int>(Tq),
            static_cast<int>(Tkv),
            causal
        );
    }

    // ===== 3) AV = scores @ V =====
    auto out = at::matmul(scores, v); // [B,H,Tq,D], dtype same as q/k/v

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "flash_attn_forward",
        &flash_attn_forward_cuda,
        "QK/AV via at::matmul + custom CUDA softmax (fp32/bf16)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("scale"),
        py::arg("causal")
    );
}

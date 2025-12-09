#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <math.h>
#include <limits>

#define WARP_SIZE 32
#define TR 16
#define TK 64

__device__ __forceinline__ int idx_qkv(int bh, int t, int d, int T, int D) {
    return (bh * T + t) * D + d;
}

__global__ void flash_attn_online_tiled_kernel(
    const float* __restrict__ q, // [BH, Tq, D]
    const float* __restrict__ k, // [BH, Tkv, D]
    const float* __restrict__ v, // [BH, Tkv, D]
    float* __restrict__ out, // [BH, Tq, D]
    int BH,
    int Tq,
    int Tkv,
    int D,
    bool causal
) {
    int bh = blockIdx.x;
    if (bh >= BH) return;

    int tq_tile_start = blockIdx.y * TR;

    int lane = threadIdx.x;  // 0..WARP_SIZE-1
    int row  = threadIdx.y;  // 0..TR-1

    int tq_idx = tq_tile_start + row;

    extern __shared__ float shmem[];

    // [TR, D] for q and acc
    float* q_sh   = shmem; // TR * D
    float* acc_sh = q_sh + TR * D; // TR * D

    // [TK, D] for K/V tiles
    float* k_tile = acc_sh + TR * D; // TK * D
    float* v_tile = k_tile + TK * D; // TK * D

    // [TR] for m, l
    float* m_sh   = v_tile + TK * D; // TR
    float* l_sh   = m_sh + TR; // TR

    if (tq_idx < Tq) {
        int q_base = (bh * Tq + tq_idx) * D;
        for (int d = lane; d < D; d += WARP_SIZE) {
            q_sh[row * D + d]   = q[q_base + d];
            acc_sh[row * D + d] = 0.0f;
        }
        if (lane == 0) {
            m_sh[row] = -INFINITY;
            l_sh[row] = 0.0f;
        }
    }
    __syncthreads();

    float inv_sqrt_D = rsqrtf((float)D);

    for (int kv_start = 0; kv_start < Tkv; kv_start += TK) {
        for (int j = row; j < TK; j += TR) {
            int tkv_idx = kv_start + j;
            if (tkv_idx < Tkv) {
                int base = (bh * Tkv + tkv_idx) * D;
                for (int d = lane; d < D; d += WARP_SIZE) {
                    float kval = k[base + d];
                    float vval = v[base + d];
                    k_tile[j * D + d] = kval;
                    v_tile[j * D + d] = vval;
                }
            } else {
                for (int d = lane; d < D; d += WARP_SIZE) {
                    k_tile[j * D + d] = 0.0f;
                    v_tile[j * D + d] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int j = 0; j < TK; ++j) {
            int tkv_idx = kv_start + j;
            if (tkv_idx >= Tkv) break;

            if (tq_idx >= Tq) {
                continue;
            }
            if (causal && tkv_idx > tq_idx) {
                continue;
            }

            float qk_partial = 0.0f;
            for (int d = lane; d < D; d += WARP_SIZE) {
                float q_val = q_sh[row * D + d];
                float k_val = k_tile[j * D + d];
                qk_partial += q_val * k_val;
            }

            // warp-level reduce
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                qk_partial += __shfl_down_sync(0xffffffff, qk_partial, offset);
            }

            float old_scale = 0.0f;
            float p         = 0.0f;

            if (lane == 0) {
                float dot = qk_partial * inv_sqrt_D;

                float m_old = m_sh[row];
                float m_new;
                if (kv_start == 0 && j == 0) {
                    m_new = dot;
                } else {
                    m_new = fmaxf(m_old, dot);
                }

                float exp_scale_old =
                    (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
                float exp_new = expf(dot - m_new);

                float l_old = l_sh[row];
                float l_new = l_old * exp_scale_old + exp_new;

                m_sh[row] = m_new;
                l_sh[row] = l_new;

                old_scale = exp_scale_old;
                p         = exp_new;
            }

            old_scale = __shfl_sync(0xffffffff, old_scale, 0);
            p         = __shfl_sync(0xffffffff, p, 0);

            for (int d = lane; d < D; d += WARP_SIZE) {
                float acc_val = acc_sh[row * D + d];
                float v_val   = v_tile[j * D + d];
                acc_val = acc_val * old_scale + p * v_val;
                acc_sh[row * D + d] = acc_val;
            }
        }
        __syncthreads();
    }

    if (tq_idx < Tq) {
        float l = l_sh[row];
        float inv_l = 1.0f / fmaxf(l, 1e-6f);
        int out_base = (bh * Tq + tq_idx) * D;
        for (int d = lane; d < D; d += WARP_SIZE) {
            out[out_base + d] = acc_sh[row * D + d] * inv_l;
        }
    }
}

// ======================= C++ wrapper =======================
at::Tensor flash_attn_forward_online(
    const at::Tensor& q, // [B,H,Tq,D]
    const at::Tensor& k, // [B,H,Tkv,D]
    const at::Tensor& v, // [B,H,Tkv,D]
    bool causal
) {
    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Tq,D]");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have same shape");
    TORCH_CHECK(q.scalar_type() == at::kFloat, "only float32 is supported for now");
    TORCH_CHECK(k.scalar_type() == at::kFloat, "only float32 is supported for now");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be CUDA");

    auto B   = q.size(0);
    auto H   = q.size(1);
    auto Tq  = q.size(2);
    auto D   = q.size(3);
    auto Tkv = k.size(2);

    TORCH_CHECK(D <= 1024, "D too large for this simple kernel (<=1024)");

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();
    auto v_contig = v.contiguous();

    auto out = at::empty_like(q_contig);

    int BH = static_cast<int>(B * H);
    int Tq_int  = static_cast<int>(Tq);
    int Tkv_int = static_cast<int>(Tkv);
    int D_int   = static_cast<int>(D);

    // block: (WARP_SIZE, TR)
    dim3 block(WARP_SIZE, TR);
    // grid: (BH, ceil(Tq / TR))
    dim3 grid(BH, (Tq_int + TR - 1) / TR);

    size_t shmem_floats = (2 * TR + 2 * TK) * static_cast<size_t>(D_int) + 2 * TR;
    size_t shmem_bytes  = shmem_floats * sizeof(float);


    TORCH_CHECK(
        shmem_bytes <= 96 * 1024,
        "Requested shared memory (", shmem_bytes,
        " bytes) too large; try smaller TR/TK or smaller D."
    );

    auto stream = at::cuda::getCurrentCUDAStream();

    flash_attn_online_tiled_kernel<<<grid, block, shmem_bytes, stream>>>(
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        BH,
        Tq_int,
        Tkv_int,
        D_int,
        causal
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

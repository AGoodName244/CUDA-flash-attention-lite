#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__device__ __forceinline__ int idx4_scores(
    int b, int h, int tq, int tkv,
    int H, int Tq, int Tkv
) {
    // layout: (((b * H + h) * Tq + tq) * Tkv + tkv)
    return (((b * H + h) * Tq + tq) * Tkv + tkv);
}

__global__ void softmax_kernel_parallel_fp32(
    float* __restrict__ scores, // [B,H,Tq,Tkv] in-place
    int B, int H, int Tq, int Tkv,
    bool causal
) {
    int row = blockIdx.x;
    int total_rows = B * H * Tq;
    if (row >= total_rows) return;

    int tmp = row;
    int tq_idx = tmp % Tq;
    tmp /= Tq;
    int h = tmp % H;
    int b = tmp / H;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_max = -INFINITY;
    for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
        if (causal && tkv_idx > tq_idx) continue;
        int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);
        float v = scores[s_idx];
        if (v > local_max) local_max = v;
    }

    __shared__ float shm_max[512];
    shm_max[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm_max[tid] = fmaxf(shm_max[tid], shm_max[tid + offset]);
        }
        __syncthreads();
    }
    float row_max = shm_max[0];

    float local_sum = 0.0f;
    for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
        int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);
        float v;
        if (causal && tkv_idx > tq_idx) {
            v = 0.0f;
        } else {
            v = expf(scores[s_idx] - row_max);
        }
        scores[s_idx] = v;
        local_sum += v;
    }

    __shared__ float shm_sum[512];
    shm_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm_sum[tid] += shm_sum[tid + offset];
        }
        __syncthreads();
    }

    float row_sum = shm_sum[0];
    if (row_sum <= 0.0f) {
        for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
            int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);
            scores[s_idx] = 0.0f;
        }
        return;
    }

    float inv_sum = 1.0f / row_sum;

    for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
        int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);
        scores[s_idx] *= inv_sum;
    }
}

__global__ void softmax_kernel_parallel_bf16(
    __nv_bfloat16* __restrict__ scores, // [B,H,Tq,Tkv] bf16 in-place
    int B, int H, int Tq, int Tkv,
    bool causal
) {
    int row = blockIdx.x;
    int total_rows = B * H * Tq;
    if (row >= total_rows) return;

    int tmp = row;
    int tq_idx = tmp % Tq;
    tmp /= Tq;
    int h = tmp % H;
    int b = tmp / H;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_max = -INFINITY;
    for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
        if (causal && tkv_idx > tq_idx) continue;
        int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);

        __nv_bfloat16 sh = scores[s_idx];
        float v = __bfloat162float(sh);
        if (v > local_max) local_max = v;
    }

    __shared__ float shm_max[512];
    shm_max[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm_max[tid] = fmaxf(shm_max[tid], shm_max[tid + offset]);
        }
        __syncthreads();
    }
    float row_max = shm_max[0];

    float local_sum = 0.0f;
    for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
        int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);
        float v;
        if (causal && tkv_idx > tq_idx) {
            v = 0.0f;
        } else {
            __nv_bfloat16 sh = scores[s_idx];
            float raw = __bfloat162float(sh);
            v = expf(raw - row_max);
        }
        scores[s_idx] = __float2bfloat16(v);
        local_sum += v;
    }

    __shared__ float shm_sum[512];
    shm_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm_sum[tid] += shm_sum[tid + offset];
        }
        __syncthreads();
    }

    float row_sum = shm_sum[0];
    if (row_sum <= 0.0f) {
        for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
            int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);
            scores[s_idx] = __float2bfloat16(0.0f);
        }
        return;
    }

    float inv_sum = 1.0f / row_sum;

    for (int tkv_idx = tid; tkv_idx < Tkv; tkv_idx += stride) {
        int s_idx = idx4_scores(b, h, tq_idx, tkv_idx, H, Tq, Tkv);

        __nv_bfloat16 sh = scores[s_idx];
        float v = __bfloat162float(sh);
        v = v * inv_sum;
        scores[s_idx] = __float2bfloat16(v);
    }
}

extern "C"
void softmax_cuda_launcher_fp32(
    float* scores,
    int B, int H, int Tq, int Tkv,
    bool causal
) {
    int total_rows = B * H * Tq;
    dim3 grid(total_rows);
    dim3 block(512);

    softmax_kernel_parallel_fp32<<<grid, block>>>(
        scores,
        B, H, Tq, Tkv,
        causal
    );
}

extern "C"
void softmax_cuda_launcher_bf16(
    __nv_bfloat16* scores,
    int B, int H, int Tq, int Tkv,
    bool causal
) {
    int total_rows = B * H * Tq;
    dim3 grid(total_rows);
    dim3 block(512);

    softmax_kernel_parallel_bf16<<<grid, block>>>(
        scores,
        B, H, Tq, Tkv,
        causal
    );
}

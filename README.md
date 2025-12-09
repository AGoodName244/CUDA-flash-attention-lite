# CUDA-flash-attention-lite
This repository contains a minimal, self-contained CUDA-based inference engine focused on accelerating the scaled dot-product attention used in Transformer models.  
The project includes:

- A PyTorch naive baseline implementation  
- Custom C++/CUDA attention kernels  
- Optimizations for softmax and memory access  
- Benchmarks covering both prefill and decode workloads  
- Integration with a modified Llama-3 model for end-to-end validation  

This work was developed as part of the *High-Performance Computing for Machine Learning* coursework.

---

## Repository Structure

```
CUDA-flash-attention-lite/
│
└── tinyllm/
    └── exp/
        ├── naive_attention.py                 # Main benchmark driver (baseline + CUDA kernels)
        ├── plot_attention_sweep.py            # Script for visualizing sweep results
        │
        ├── 4090/                              # Experiments on RTX 4090
        │    ├── *.xlsx                        # Sweep result tables
        │    └── *.png                         # Generated plots
        │
        ├── flash_attn_ext/
        │   └── csrc/
        │        ├── flash_attn.cpp                   # Split-operator PyTorch binding
        │        ├── flash_attn_cuda.cu               # CUDA kernel: split operator
        │        ├── flash_attn_fused.cpp             # Fused operator PyTorch binding
        │        ├── flash_attn_cuda_fused.cu         # CUDA kernel: optimized fused attention
        │        ├── flash_attn_cuda_fused_ori.cu     # CUDA kernel: original fused version
        │
        └── llama-models/
             └── models/llama3/                       # Modified Llama-3 attention and benchmark code
```

---

## Implemented Attention Variants

### 1. Naive attention (PyTorch baseline)

A reference implementation using standard PyTorch operations:

```
scores = q @ k^T
scores = causal_mask(scores)
attn = softmax(scores)
out = attn @ v
```

This serves as the correctness baseline for all custom kernels.

---

### 2. Split CUDA kernels (`flash_attn_ext`)

This version implements the QKᵀ computation, masking, softmax, and AV multiplication using custom softmax kernels.  
Compared to the PyTorch baseline, this reduces some overhead

Files:

- `flash_attn.cpp`
- `flash_attn_cuda.cu`

---

### 3. Fused online-softmax kernel (FlashAttention-style)

A more optimized version where:

- The scores tensor is never fully materialized  
- Softmax is computed in an online manner  
- Only essential intermediate values are kept in registers/shared memory  
- Kernel launch overhead is minimized  
- Peak memory usage is reduced  

Files:

- `flash_attn_fused.cpp`
- `flash_attn_cuda_fused.cu`

The original fused implementation (`flash_attn_cuda_fused_ori.cu`) is kept for comparison.

---

## Benchmarking

All benchmark logic is implemented in:

### `tinyllm/exp/naive_attention.py`

It supports multiple modes:

---

### Run baseline attention

```
python naive_attention.py \
    --B 1 --H 32 --Tq 2048 --Tkv 2048 --D 64 \
    --dtype float16 --device cuda --causal
```

---

### Select implementation

```
python naive_attention.py --impl naive
python naive_attention.py --impl ext
python naive_attention.py --impl online
```

---

### Run sweep experiments

```
python naive_attention.py --device cuda --dtype float16 --sweep
```

This generates:

```
attention_sweep_results.csv
```

---

### Visualize sweep results

```
python plot_attention_sweep.py
```

---

## Llama-3 Integration

The repository includes lightweight modifications to Meta’s Llama-3 implementation, allowing our custom attention kernels to be integrated and benchmarked as part of an end-to-end inference pipeline.

Modified files follow the same directory layout as upstream:

```
tinyllm/exp/llama-models/models/llama3/
```

Example usage (same interface as original Llama-3):

```
export PYTHONPATH=$(pwd)
NGPUS=1
CHECKPOINT_DIR=/path/to/Llama-3.2-1B-Instruct

torchrun --nproc_per_node=$NGPUS \
  -m models.llama3.scripts.interactive_chat \
  $CHECKPOINT_DIR \
  --world_size $NGPUS
```
Or you can run the benchmark script via
```
torchrun --nproc_per_node=$NGPUS \
  -m models.llama3.scripts.benchmark_flash_attn \
  $CHECKPOINT_DIR \
  --runs [iteration number]
  --prompt [prompt text]
```

Only the attention forward pass and benchmarking hooks are modified; training logic remains unchanged.

---

## Results Overview

A detailed evaluation is included in the associated coursework, but overall observations are:

- The fused kernel achieves noticeable speedups over the PyTorch baseline, especially for longer sequence lengths.  
- Peak memory usage is significantly reduced since the scores tensor is not stored.  
- Extending the kernels to Llama-3 inference validates correctness, with numerical deviations within acceptable tolerances.  
- For short sequences, kernel launch overhead can dominate; for long sequences, fused kernels become consistently advantageous.

Benchmark plots are available in the `4090/` directory.

---

## Requirements

```
Python 3.10+
PyTorch 2.x with CUDA
CUDA Toolkit 12.x
```

All kernels are compiled at runtime via PyTorch’s C++ extension mechanism.

---
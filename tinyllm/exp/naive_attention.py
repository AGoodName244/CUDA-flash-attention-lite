# Minimal PyTorch baseline for scaled dot-product attention
# Used as a reference & benchmark when integrating custom C++/CUDA kernels.
# python naive_attention.py --B 1 --H 32 --Tq 2048 --Tkv 2048 --D 64 --dtype float16 --device cuda --causal
# python naive_attention.py --B 1 --H 32 --Tq 2048 --Tkv 2048 --D 64 --dtype float32 --device cuda --causal --impl ext

# export PYTHONPATH=$(pwd)
# NGPUS=1
# CHECKPOINT_DIR=/root/9143/llama_ckpts/Llama-3.2-1B-Instruct/original

# torchrun --nproc_per_node=$NGPUS \
#   -m models.llama3.scripts.interactive_chat \
#   $CHECKPOINT_DIR \
#   --world_size $NGPUS

# python naive_attention.py --device cuda --dtype float32 --causal --sweep


import csv

from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
import os

_flash_attn_ext = None
_flash_attn_fused = None

def get_flash_attn_ext():
    global _flash_attn_ext
    if _flash_attn_ext is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        ext_dir = os.path.join(this_dir, "flash_attn_ext", "csrc")
        _flash_attn_ext = load(
            name="flash_attn_ext_split",
            sources=[
                os.path.join(ext_dir, "flash_attn.cpp"),
                os.path.join(ext_dir, "flash_attn_cuda.cu"),
            ],
            verbose=True,
        )
    return _flash_attn_ext

def get_flash_attn_fused():
    global _flash_attn_fused
    if _flash_attn_fused is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        ext_dir = os.path.join(this_dir, "flash_attn_ext", "csrc")
        _flash_attn_fused = load(
            name="flash_attn_ext_fused",
            sources=[
                os.path.join(ext_dir, "flash_attn_fused.cpp"),
                os.path.join(ext_dir, "flash_attn_cuda_fused.cu"),
            ],
            verbose=True,
        )
    return _flash_attn_fused

import argparse
import time
import math
import torch


def naive_attention(q, k, v, mask=None, is_causal=False, scale=None):
    """
    q, k, v: [B, H, T, D]
    mask: optional, broadcastable to [B, 1, Tq, Tkv] or [B, 1, 1, Tkv]
          values should be additive (e.g., 0 or -1e9)
    is_causal: if True, apply causal mask (no attending to future positions)
    scale: if None, use 1 / sqrt(D)
    """
    # Shapes
    B, H, Tq, D = q.shape
    _, _, Tkv, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # [B, H, Tq, Tkv]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        # causal mask: prevent attending to future tokens
        # upper triangular (strict) => True for positions to mask
        causal_mask = torch.triu(
            torch.ones(Tq, Tkv, device=scores.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if mask is not None:
        # mask is additive; should already be broadcastable
        scores = scores + mask

    # softmax along key dimension
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [B, H, Tq, Tkv]

    # [B, H, Tq, D]
    out = torch.matmul(attn, v)
    return out

def flash_attn_forward(q, k, v, is_causal=False):
    ext = get_flash_attn_ext()
    B, H, Tq, D = q.shape
    scale = 1.0 / math.sqrt(D)
    return ext.flash_attn_forward(q, k, v, float(scale), bool(is_causal))

def flash_attn_forward_online(q, k, v, is_causal: bool):
    ext = get_flash_attn_fused()
    return ext.flash_attn_forward_online(q, k, v, is_causal)

# def flash_attn_forawrd_tiled(q, k, v, is_causal: bool):
#     ext = get_flash_attn_ext()
#     return ext.flash_attn_forward_tiled(q, k, v, is_causal)

def benchmark(fn, name, device, warmup=10, iters=50):
    """Simple benchmark wrapper."""
    # warmup
    for _ in range(warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0 / iters

    print(f"{name}: {ms:.3f} ms/iter")
    return ms

def run_with_profiler(run_naive, run_flash, device):
    if device.type != "cuda":
        print("Profiler example only set up for CUDA now.")
        _ = run_naive()
        _ = run_flash()
        return
    
    _ = run_naive()
    _ = run_flash()
    torch.cuda.synchronize()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        with record_function("run_naive"):
            _ = run_naive()
        torch.cuda.synchronize()
        
        with record_function("run_flash"):
            _ = run_flash()
        torch.cuda.synchronize()
        
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    prof.export_chrome_trace("trace.json")
    
def profile_memory_once(fn, name, device):
    if device.type != "cuda":
        print("Memory profiler only set up for CUDA now.")
        _ = fn()
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _ = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"[{name}] peak cuda memory: {peak:.2f} MB")
    return peak

def attention_flops(B, H, Tq, Tkv, D):
    """
    Effective FLOPs of a *naive* scaled dot-product attention layer
    for a single forward pass with shapes:
      q: [B, H, Tq, D], k/v: [B, H, Tkv, D].
      - scores = q @ k^T          (2 * B * H * Tq * Tkv * D)
      - softmax over scores       (~5 * B * H * Tq * Tkv)
      - out = softmax(scores) @ v (2 * B * H * Tq * Tkv * D)

    This matches the "effective TFLOP/s" style used in FlashAttention.
    """
    flops_qk = 2.0 * B * H * Tq * Tkv * D
    flops_pv = 2.0 * B * H * Tq * Tkv * D
    flops_softmax = 5.0 * B * H * Tq * Tkv  # rough constant
    return flops_qk + flops_pv + flops_softmax




def parse_args():
    p = argparse.ArgumentParser(description="Naive scaled dot-product attention benchmark")

    # Tensor shape
    p.add_argument("--batch", "--B", type=int, default=1, dest="B")
    p.add_argument("--heads", "--H", type=int, default=32, dest="H")
    p.add_argument("--Tq", type=int, default=1024, help="Query sequence length")
    p.add_argument("--Tkv", type=int, default=1024, help="Key/Value sequence length")
    p.add_argument("--dim", "--D", type=int, default=64, dest="D", help="Head dimension")

    # Device / dtype
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float16",
                  choices=["float32", "float16", "bfloat16"])

    # Options
    p.add_argument("--causal", action="store_true", help="Use causal mask")
    p.add_argument("--no-causal", action="store_false", dest="causal")
    p.set_defaults(causal=False)

    p.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    
    p.add_argument("--impl", type=str, default="naive", choices=["naive", "ext", "online", "tiled"], help="attn approach")
    p.add_argument("--use-profiler", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="Run a built-in sweep of configs and save results to CSV")
    
    return p.parse_args()

def run_sweep(device, dtype, warmup=10, iters=50):
    prefill_exps = [
        ("prefill", 1, 32, 32,    32,    64),
        ("prefill", 1, 32, 64,    64,    64),
        ("prefill", 1, 32, 128,   128,   64),
        ("prefill", 1, 32, 256,   256,   64),
        ("prefill", 1, 32, 512,   512,   64),
        ("prefill", 1, 32, 1024,  1024,  64),
        ("prefill", 1, 32, 2048,  2048,  64),
        ("prefill", 1, 32, 4096,  4096,  64),
    ]

    decode_exps = [
        ("decode", 1, 32, 1,   128,   64),
        ("decode", 1, 32, 1,   256,   64),
        ("decode", 1, 32, 1,   512,   64),
        ("decode", 1, 32, 1,   1024,   64),
        ("decode", 1, 32, 1,   2048,  64),
        ("decode", 1, 32, 1,   4096,  64),
    ]

    d_sweep_exps = [
        ("D_sweep", 1, 32, 128, 128, 32),
        ("D_sweep", 1, 32, 128, 128, 64),
    ]

    exps = prefill_exps + decode_exps + d_sweep_exps

    results = []

    for tag, B, H, Tq, Tkv, D in exps:
        print(f"\n=== [sweep] {tag}: B={B}, H={H}, Tq={Tq}, Tkv={Tkv}, D={D} ===")

        q = torch.randn(B, H, Tq, D, device=device, dtype=dtype)
        k = torch.randn(B, H, Tkv, D, device=device, dtype=dtype)
        v = torch.randn(B, H, Tkv, D, device=device, dtype=dtype)

        def run_naive():
            return naive_attention(q, k, v, mask=None, is_causal=True)

        def run_ext():
            return flash_attn_forward(q, k, v, is_causal=True)

        def run_flash():
            return flash_attn_forward_online(q, k, v, is_causal=True)

        row_common = {
            "tag": tag,
            "B": B,
            "H": H,
            "Tq": Tq,
            "Tkv": Tkv,
            "D": D,
        }

        flops = attention_flops(B, H, Tq, Tkv, D)

        # ---- naive ----
        t_naive = benchmark(run_naive, "naive_attention (PyTorch)", device,
                            warmup=warmup, iters=iters)
        m_naive = profile_memory_once(run_naive, "naive_sweep", device)
        t_naive_s = t_naive / 1000.0 if t_naive is not None else None
        tflops_naive = flops / (t_naive_s * 1e12) if t_naive_s and t_naive_s > 0 else None

        results.append({
            **row_common,
            "impl": "naive",
            "time_ms": t_naive,
            "peak_mem_MB": m_naive,
            "tflops": tflops_naive,
            "status": "ok",
        })

        # ---- ext (split CUDA) ----
        try:
            t_ext = benchmark(run_ext, "flash_attn_ext (naive CUDA)", device,
                              warmup=warmup, iters=iters)
            m_ext = profile_memory_once(run_ext, "ext_sweep", device)
            t_ext_s = t_ext / 1000.0 if t_ext is not None else None
            tflops_ext = flops / (t_ext_s * 1e12) if t_ext_s and t_ext_s > 0 else None

            results.append({
                **row_common,
                "impl": "ext",
                "time_ms": t_ext,
                "peak_mem_MB": m_ext,
                "tflops": tflops_ext,
                "status": "ok",
            })
        except Exception as e:
            print(f"[sweep] ext failed: {e}")
            results.append({
                **row_common,
                "impl": "ext",
                "time_ms": None,
                "peak_mem_MB": None,
                "tflops": None,
                "status": f"error: {e}",
            })

        # ---- fused online ----
        try:
            t_onl = benchmark(run_flash, "flash_attn_online_softmax", device,
                              warmup=warmup, iters=iters)
            m_onl = profile_memory_once(run_flash, "online_sweep", device)
            t_onl_s = t_onl / 1000.0 if t_onl is not None else None
            tflops_onl = flops / (t_onl_s * 1e12) if t_onl_s and t_onl_s > 0 else None

            results.append({
                **row_common,
                "impl": "online",
                "time_ms": t_onl,
                "peak_mem_MB": m_onl,
                "tflops": tflops_onl,
                "status": "ok",
            })
        except Exception as e:
            print(f"[sweep] online failed: {e}")
            results.append({
                **row_common,
                "impl": "online",
                "time_ms": None,
                "peak_mem_MB": None,
                "tflops": None,
                "status": f"error: {e}",
            })


    out_path = os.path.join(os.path.dirname(__file__), "attention_sweep_results.csv")
    print(f"\n[sweep] writing results to {out_path}")
    fieldnames = [
        "tag", "impl", "B", "H", "Tq", "Tkv", "D",
        "time_ms", "peak_mem_MB", "tflops", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)



    print(f"[sweep] done, {len(results)} rows.")


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # dtype
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if args.sweep:
        run_sweep(device, dtype, warmup=args.warmup, iters=args.iters)
        return
    
    B, H, Tq, Tkv, D = args.B, args.H, args.Tq, args.Tkv, args.D

    print(f"=== naive_attention baseline ===")
    print(f"B={B}, H={H}, Tq={Tq}, Tkv={Tkv}, D={D}, dtype={dtype}, device={device}")
    print(f"causal={args.causal}")

    # Create random Q/K/V
    q = torch.randn(B, H, Tq, D, device=device, dtype=dtype)
    k = torch.randn(B, H, Tkv, D, device=device, dtype=dtype)
    v = torch.randn(B, H, Tkv, D, device=device, dtype=dtype)

    mask = None

    # def run():
    #     return naive_attention(q, k, v, mask=mask, is_causal=args.causal)
    def run_naive():
        return naive_attention(q, k, v, mask=mask, is_causal=args.causal)
    
    def run_ext():
        return flash_attn_forward(q, k, v, is_causal=args.causal)
    
    def run_flash():
        return flash_attn_forward_online(q, k, v, is_causal=args.causal)
    out_naive = run_naive()
    print("Naive output shape:", tuple(out_naive.shape))
    
    if args.impl == "ext":
        out_ext = run_ext()
        print("Ext output shape:", tuple(out_ext.shape))
        if out_ext.shape == out_naive.shape:
            print("Shapes match between naive and ext.")

        diff = (out_naive - out_ext).abs()
        print("max diff:", diff.max().item(), "mean diff:", diff.mean().item())

        benchmark(run_ext, "flash_attn_ext (naive CUDA)", device,
                  warmup=args.warmup, iters=args.iters)
        benchmark(run_naive, "naive_attention (PyTorch)", device,
                  warmup=args.warmup, iters=args.iters)
        if args.use_profiler:
            profile_memory_once(run_ext, "split cuda", device)
            profile_memory_once(run_naive, "naive_attention", device)
        
    elif args.impl == "online":
        print("=== flash_attn_ext: online softmax (no scores tensor) ===")
        out_ext = run_flash()
        print(f"Ext output shape: {tuple(out_ext.shape)}")
        print("Shapes match between naive and ext.",
            f"\nmax diff: {(out_ext - out_naive).abs().max().item()}",
            f"mean diff: {(out_ext - out_naive).abs().mean().item()}")
        benchmark(run_flash, "flash_attn_online_softmax", device,
                  warmup=args.warmup, iters=args.iters)
        benchmark(run_naive, "naive_attention (PyTorch)", device,
                  warmup=args.warmup, iters=args.iters)
        if args.use_profiler:
            run_with_profiler(run_naive, run_flash, device)
            profile_memory_once(run_flash, "flash_attn_online", device)
            profile_memory_once(run_naive, "naive_attention", device)
    else:
        benchmark(run_naive, "naive_attention (PyTorch)", device,
                  warmup=args.warmup, iters=args.iters)


    print("Done.")

if __name__ == "__main__":
    main()

import math
import os
import torch
import warnings
from torch.utils.cpp_extension import load

warnings.filterwarnings(
    "ignore",
    message="TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.*",
    category=UserWarning,
)

_split_ext = None
_fused_ext = None


def _get_ext_split():
    global _split_ext
    if _split_ext is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        ext_dir = os.path.join(this_dir, "..", "..", "..", "flash_attn_ext", "csrc")
        ext_dir = os.path.normpath(ext_dir)

        _split_ext = load(
            name="flash_attn_ext_split",
            sources=[
                os.path.join(ext_dir, "flash_attn.cpp"),
                os.path.join(ext_dir, "flash_attn_cuda.cu"),
            ],
            verbose=False,
        )
    return _split_ext


def _get_ext_fused():
    global _fused_ext
    if _fused_ext is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        ext_dir = os.path.join(this_dir, "..", "..", "..", "flash_attn_ext", "csrc")
        ext_dir = os.path.normpath(ext_dir)

        _fused_ext = load(
            name="flash_attn_ext_fused1",
            sources=[
                os.path.join(ext_dir, "flash_attn_fused.cpp"),
                os.path.join(ext_dir, "flash_attn_cuda_fused.cu"),
            ],
            verbose=False,
        )
    return _fused_ext


@torch.no_grad()
def flash_attn_split(q, k, v, is_causal: bool):
    """
    q, k, v: [B, H, T, D]
    """
    ext = _get_ext_split()
    orig_dtype = q.dtype
    if orig_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"flash_attn_split: unsupported dtype {orig_dtype}")

    # if orig_dtype not in (torch.float16, torch.bfloat16, torch.float32):
    #     raise RuntimeError(f"flash_attn_split: unsupported dtype {orig_dtype}")

    # q32 = q.contiguous().to(torch.float32)
    # k32 = k.contiguous().to(torch.float32)
    # v32 = v.contiguous().to(torch.float32)
    q_ = q.contiguous()
    k_ = k.contiguous()
    v_ = v.contiguous()

    B, H, Tq, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # out32 = ext.flash_attn_forward(q32, k32, v32, float(scale), bool(is_causal
    out = ext.flash_attn_forward(q_, k_, v_, float(scale), bool(is_causal))

    # return out32.to(orig_dtype).contiguous()
    return out.contiguous()


@torch.no_grad()
def flash_attn_online(q, k, v, is_causal: bool):
    ext = _get_ext_fused()
    orig_dtype = q.dtype

    if orig_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"flash_attn_online: unsupported dtype {orig_dtype}")

    q32 = q.contiguous().to(torch.float32)
    k32 = k.contiguous().to(torch.float32)
    v32 = v.contiguous().to(torch.float32)

    out32 = ext.flash_attn_forward_online(q32, k32, v32, bool(is_causal))

    return out32.to(orig_dtype).contiguous()

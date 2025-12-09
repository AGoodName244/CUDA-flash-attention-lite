import os
import time
from typing import Optional, List

import torch

from models.llama3.generation import Llama3
from models.datatypes import RawMessage


PREFILL_REPEAT = 16
PREFILL_MAX_GEN_LEN = 32

def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"

def build_generator(ckpt_dir: str,
                    use_custom_attn: bool,
                    max_seq_len: int = 2048,
                    max_batch_size: int = 1,
                    world_size: Optional[int] = None,
                    quantization_mode: Optional[str] = None) -> Llama3:

    os.environ["LLAMA_USE_CUSTOM_ATTN"] = "1" if use_custom_attn else "0"

    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )
    return generator

# def run_chat_once(generator: Llama3,
#                   prompt: str,
#                   max_gen_len: int = 256,
#                   temperature: float = 0.0,
#                   top_p: float = 1.0) -> int:

#     dialog: List[RawMessage] = [RawMessage(role="user", content=prompt)]
#     batch = [dialog]
#     total_output_tokens = 0

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t0 = time.time()

#     for token_results in generator.chat_completion(
#         batch,
#         temperature=temperature,
#         top_p=top_p,
#         max_gen_len=max_gen_len,
#     ):
#         result = token_results[0]
#         if result.finished:
#             break
#         total_output_tokens += len(result.text)

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t1 = time.time()

#     try:
#         prompt_ids = generator.tokenizer.encode(prompt, bos=True, eos=False)
#     except TypeError:
#         prompt_ids = generator.tokenizer.encode(prompt)

#     prompt_tokens = len(prompt_ids)

#     elapsed = t1 - t0
#     approx_total_tokens = prompt_tokens + total_output_tokens

#     print(f"    elapsed: {elapsed:.3f} s, "
#           f"prompt_tokens ≈ {prompt_tokens}, "
#           f"output_char_len = {total_output_tokens}, "
#           f"approx tokens/s ≈ {approx_total_tokens / max(elapsed, 1e-6):.2f}")

#     return approx_total_tokens, elapsed
def run_chat_once(generator: Llama3,
                  prompt: str,
                  max_gen_len: int = 256,
                  temperature: float = 0.0,
                  top_p: float = 1.0) -> int:

    dialog: List[RawMessage] = [RawMessage(role="user", content=prompt)]
    batch = [dialog]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    generated_text = ""
    full_text = ""
    use_delta = None

    for token_results in generator.chat_completion(
        batch,
        temperature=temperature,
        top_p=top_p,
        max_gen_len=max_gen_len,
    ):
        result = token_results[0]

        if use_delta is None:
            use_delta = hasattr(result, "delta_text")

        if use_delta:
            delta = getattr(result, "delta_text", "")
            generated_text += delta
            full_text = generated_text
        else:
            full_text = getattr(result, "text", "") or ""

        if result.finished:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    elapsed = t1 - t0

    try:
        prompt_ids = generator.tokenizer.encode(prompt, bos=True, eos=False)
    except TypeError:
        prompt_ids = generator.tokenizer.encode(prompt)
    prompt_tokens = len(prompt_ids)

    try:
        gen_ids = generator.tokenizer.encode(full_text, bos=False, eos=False)
    except TypeError:
        gen_ids = generator.tokenizer.encode(full_text)
    output_tokens = len(gen_ids)

    total_tokens = prompt_tokens + output_tokens
    tps = total_tokens / max(elapsed, 1e-6)

    print(
        f"    elapsed: {elapsed:.3f} s, "
        f"prompt_tokens = {prompt_tokens}, "
        f"output_tokens = {output_tokens}, "
        f"total_tokens = {total_tokens}, "
        f"tokens/s = {tps:.2f}"
    )

    return total_tokens, elapsed


def run_prefill_once(generator: Llama3, prompt: str):
    try:
        token_ids = generator.tokenizer.encode(prompt, bos=True, eos=False)
    except TypeError:
        token_ids = generator.tokenizer.encode(prompt)

    tokens = torch.tensor(
        token_ids,
        device=get_device(),
        dtype=torch.long
    ).unsqueeze(0)  # [1, T]
    B, T = tokens.shape
    print(f"[prefill] tokens shape: {tokens.shape}, total tokens = {T}")

    model = generator.model

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        _ = model(tokens, start_pos=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    elapsed = t1 - t0
    tps = T / max(elapsed, 1e-6)
    print(f"[prefill] elapsed = {elapsed:.4f} s, tokens = {T}, tokens/s = {tps:.2f}")
    return T, elapsed

def run_prefill_only(generator, prompt: str):
    tokens = generator.tokenizer.encode(prompt, bos=True, eos=False)
    x = torch.tensor(tokens, device="cuda").unsqueeze(0)  # [1, T]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = generator.model(x, start_pos=0)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[prefill only] peak allocated = {peak:.1f} MB")



def benchmark_mode(ckpt_dir: str,
                   use_custom_attn: bool,
                   prompt: str,
                   max_gen_len: int = 256,
                   runs: int = 3):
    mode_name = "CUSTOM_KERNEL" if use_custom_attn else "BASELINE_PYTORCH"
    print(f"\n==== Benchmark: {mode_name} ====")

    generator = build_generator(
        ckpt_dir=ckpt_dir,
        use_custom_attn=use_custom_attn,
        max_seq_len=max_gen_len * 4,
        max_batch_size=1,
        world_size=None,
        quantization_mode=None,
    )
    print("  Warming up...")
    _ = run_chat_once(generator, prompt, max_gen_len=max_gen_len)

    times = []
    token_counts = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print("  Running benchmark...")
    for i in range(runs):
        print(f"  Run {i+1}/{runs}:")
        tokens, t = run_chat_once(generator, prompt, max_gen_len=max_gen_len)
        token_counts.append(tokens)
        times.append(t)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tps = avg_tokens / avg_time

    print(f"\n---- {mode_name} Summary ----")
    print(f"  Avg elapsed: {avg_time:.3f} s")
    print(f"  Avg approx tokens: {avg_tokens:.1f}")
    print(f"  Avg approx tokens/s: {avg_tps:.2f}")

    if torch.cuda.is_available():
        max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"  Max GPU memory allocated: {max_alloc:.1f} MB")
        print(f"  Max GPU memory reserved : {max_reserved:.1f} MB")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str,
                        help="Path to Llama3.2 1B Instruct checkpoint dir")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument(
        "--prompt",
        type=str,
        default="hello " * 32,
        help="Prompt used for the benchmark (single-turn).",
    )

    args = parser.parse_args()

    # args.prompt = "Explain why transformers can model long-range dependencies more effectively than recurrent neural networks, and give one concrete example."

    # ========= 1) Prefill-only benchmark =========
    # print("\n==== Prefill-only: BASELINE_PYTORCH ====")
    # gen_base = build_generator(
    #     ckpt_dir=args.ckpt_dir,
    #     use_custom_attn=False,
    #     max_seq_len=args.max_gen_len * 4,
    #     max_batch_size=1,
    #     world_size=None,
    #     quantization_mode=None,
    # )
    # _ = run_prefill_once(gen_base, args.prompt)

    # print("\n==== Prefill-only: CUSTOM_KERNEL ====")
    # gen_custom = build_generator(
    #     ckpt_dir=args.ckpt_dir,
    #     use_custom_attn=True,
    #     max_seq_len=args.max_gen_len * 4,
    #     max_batch_size=1,
    #     world_size=None,
    #     quantization_mode=None,
    # )
    # _ = run_prefill_once(gen_custom, args.prompt)

    # print("\n==== Prefill-only memory occupy: BASELINE")
    # _ = run_prefill_only(gen_base, args.prompt)

    # print("\n==== Prefill-only memory occupy: CUSTOM")
    # _ = run_prefill_only(gen_custom, args.prompt)
    benchmark_mode(
        ckpt_dir=args.ckpt_dir,
        use_custom_attn=False,
        prompt=args.prompt,
        max_gen_len=args.max_gen_len,
        runs=args.runs,
    )

    benchmark_mode(
        ckpt_dir=args.ckpt_dir,
        use_custom_attn=True,
        prompt=args.prompt,
        max_gen_len=args.max_gen_len,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
            

    
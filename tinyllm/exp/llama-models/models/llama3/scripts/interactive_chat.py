from typing import Optional
from termcolor import cprint
from pathlib import Path
import fire
import os
import torch

from models.datatypes import RawMessage
from models.llama3.generation import Llama3


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
):
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )

    dialog = []  # List[RawMessage]
    print("=== Interactive Llama3.2-1B chat ===")
    print("Please enter your prompt，/reset clear history，/exit exit。\n")

    while True:
        try:
            user_text = input("User: ").strip()
        except EOFError:
            break

        if user_text.lower() in {"/exit", "/quit"}:
            print("Bye.")
            break
        if user_text.lower() == "/reset":
            dialog = []
            print("[history cleared]\n")
            continue
        if not user_text:
            continue

        dialog.append(RawMessage(role="user", content=user_text))

        batch = [dialog]
        assistant_text = ""

        print("Assistant: ", end="", flush=True)
        for token_results in generator.chat_completion(
            batch,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_seq_len,
        ):
            result = token_results[0]
            if result.finished:
                break

            new_text = result.text
            assistant_text += new_text
            print(new_text, end="", flush=True)
        print("\n")

        dialog.append(RawMessage(role="assistant", content=assistant_text))


def main():
    fire.Fire(run_main)


if __name__ == "__main__":
    main()

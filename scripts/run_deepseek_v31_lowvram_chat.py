#!/usr/bin/env python3
"""Interactive terminal chat for DeepSeek-V3.1 experimental low-VRAM mode.

This script intentionally enables expert capping through
AIRLLM_DEEPSEEK_MAX_LOCAL_EXPERTS so you can inspect quality degradation
on small GPUs.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive DeepSeek-V3.1 low-VRAM chat")
    parser.add_argument(
        "--model",
        default="unsloth/DeepSeek-V3.1-Terminus",
        help="Model repo id or local path",
    )
    parser.add_argument("--device", default="cuda:0", help="Inference device")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per prompt",
    )
    parser.add_argument(
        "--max-local-experts",
        type=int,
        default=8,
        help="Experimental cap for local experts per MoE layer",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "air_llm"))

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    os.environ["AIRLLM_DEEPSEEK_MAX_LOCAL_EXPERTS"] = str(args.max_local_experts)

    from airllm import AutoModel

    print("Loading model...")
    print(f"  model={args.model}")
    print(f"  device={args.device}")
    print(f"  AIRLLM_DEEPSEEK_MAX_LOCAL_EXPERTS={args.max_local_experts}")

    model = AutoModel.from_pretrained(
        args.model,
        device=args.device,
        max_seq_len=args.max_seq_len,
        prefetching=False,
        layers_per_batch=1,
    )

    print("\nModel ready. Type a prompt and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", ":q"}:
            print("Exiting.")
            break

        toks = model.tokenizer([prompt], return_tensors="pt")
        out = model.generate(
            toks["input_ids"].to(args.device),
            max_new_tokens=args.max_new_tokens,
            use_cache=False,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        text = model.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = text[len(prompt):].strip() if text.startswith(prompt) else text

        print("\nResponse:")
        print(answer)
        print()


if __name__ == "__main__":
    main()

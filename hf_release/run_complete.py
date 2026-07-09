#!/usr/bin/env python3
"""Prefix completion with QLLM V11 PAM checkpoint (pretrain base, HF release)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from modeling_qllm import load_model
from run_chat import get_chat_tokenizer

GPT2_EOS_ID = 50256


def complete_prefix(
    model,
    tokenizer,
    prefix: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    eos_token_id: int | None = GPT2_EOS_ID,
) -> tuple[str, str, int]:
    """Return (full_text, continuation_only, n_new_tokens)."""
    ids = tokenizer.encode(prefix)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        )
    n_new = out.shape[1] - x.shape[1]
    full = tokenizer.decode(out[0].tolist())
    continuation = tokenizer.decode(out[0, x.shape[1]:].tolist())
    return full, continuation.strip(), n_new


def main() -> None:
    p = argparse.ArgumentParser(description='Prefix completion (pretrain-style, HF release)')
    p.add_argument('--checkpoint', default='qllm_v11_e3k3_pretrain.pt')
    p.add_argument('--prompt', default='In 1923 , the University of')
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--repetition_penalty', type=float, default=1.2)
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f'Checkpoint not found: {ckpt}')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(str(ckpt), device=device)
    tokenizer = get_chat_tokenizer(model.config.vocab_size)

    full, continuation, n_new = complete_prefix(
        model,
        tokenizer,
        args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(f'Prompt: {args.prompt}')
    print(f'Generated: {full}')
    print(f'(continuation={n_new} tokens)')


if __name__ == '__main__':
    main()

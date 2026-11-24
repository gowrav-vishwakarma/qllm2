#!/usr/bin/env python3
"""Plain text continuation with SCI-1 (self-contained: torch + transformers only).

    python run_complete.py --prompt "In 1923, the University of"
    python run_complete.py --prompt "The theory of evolution" --temperature 0     # greedy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modeling_pam2 import TokenStreamer, load_model, load_tokenizer  # noqa: E402

EOT_ID = 50256   # <|endoftext|>: document separator


def pick_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def complete(model, tok, prompt: str, *, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9,
             repetition_penalty=1.2, seed=None, stream=False, device='cpu'):
    """Return (continuation text, new token ids, seconds)."""
    ids = torch.tensor([tok.encode(prompt)], device=device)
    gen = None
    if seed is not None and temperature > 0:
        gen = torch.Generator(device=device).manual_seed(seed)
    streamer = TokenStreamer(tok.decode, stop_ids=(EOT_ID,)) if stream else None
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=repetition_penalty, eos_token_id=EOT_ID,
                         generator=gen, streamer=streamer)
    dt = time.time() - t0
    new = out[0, ids.shape[1]:].tolist()
    if new and new[-1] == EOT_ID:
        new = new[:-1]
    return tok.decode(new), new, dt


def main() -> None:
    ap = argparse.ArgumentParser(description='SCI-1 text continuation')
    ap.add_argument('--checkpoint', default=str(HERE / 'sci1_chat_386m.pt'))
    ap.add_argument('--prompt', default='In 1923, the University of')
    ap.add_argument('--max_new_tokens', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=0.8, help='0 = greedy')
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--repetition_penalty', type=float, default=1.2)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--device', default='auto', help='auto | cpu | cuda | cuda:N')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    args = ap.parse_args()

    device = pick_device(args.device)
    t0 = time.time()
    model = load_model(args.checkpoint, device=device, dtype=getattr(torch, args.dtype))
    tok = load_tokenizer()
    print(f'[SCI-1 {model.count_parameters() / 1e6:.0f}M params, {args.checkpoint}, {device}, {args.dtype}, '
          f'loaded in {time.time() - t0:.1f}s]', file=sys.stderr)
    print(args.prompt, end='', flush=True)
    _, new, dt = complete(model, tok, args.prompt, max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                          repetition_penalty=args.repetition_penalty, seed=args.seed, stream=True,
                          device=device)
    print(flush=True)
    print(f'[{len(new)} new tokens in {dt:.1f}s, {len(new) / max(dt, 1e-9):.1f} tok/s]', file=sys.stderr)


if __name__ == '__main__':
    main()

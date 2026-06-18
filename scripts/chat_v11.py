#!/usr/bin/env python3
"""Interactive chat with a V11 checkpoint (post-SFT or base)."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from v11.model import V11LM, get_config


def build_prompt(user_text: str, system: str = "You are a helpful assistant.") -> str:
    return (
        f"### System:\n{system}\n"
        f"### User:\n{user_text.strip()}\n"
        f"### Assistant:\n"
    )


def main():
    p = argparse.ArgumentParser(description='Chat with a V11 checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_config(args.preset)
    model = V11LM(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print('V11 chat (empty line to quit)')
    while True:
        try:
            user = input('\nUser> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        prompt = build_prompt(user)
        ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = m.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
            )
        text = tokenizer.decode(out[0].tolist())
        if '### Assistant:' in text:
            reply = text.split('### Assistant:', 1)[1]
            reply = reply.split('### User:', 1)[0]
            reply = reply.split('<|endoftext|>', 1)[0]
        else:
            reply = text[len(prompt):]
        print(f"Assistant> {reply.strip()}")


if __name__ == '__main__':
    main()

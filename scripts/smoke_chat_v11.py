#!/usr/bin/env python3
"""Non-interactive chat smoke test on fixed prompts (V11 checkpoint)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v11.model import V11LM, get_config

DEFAULT_PROMPTS = [
    'What is the capital of France?',
    'Explain photosynthesis in one sentence.',
    'Write a Python function that adds two numbers.',
]


def build_prompt(user_text: str, system: str = 'You are a helpful assistant.') -> str:
    return (
        f'### System:\n{system}\n'
        f'### User:\n{user_text.strip()}\n'
        f'### Assistant:\n'
    )


def extract_reply(text: str, prompt: str) -> str:
    if '### Assistant:' in text:
        reply = text.split('### Assistant:', 1)[1]
        reply = reply.split('### User:', 1)[0]
        reply = reply.split('<|endoftext|>', 1)[0]
        return reply.strip()
    return text[len(prompt):].strip()


def main() -> None:
    p = argparse.ArgumentParser(description='Fixed-prompt chat smoke for V11')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--label', default='')
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--max_new_tokens', type=int, default=128)
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

    tag = args.label or Path(args.checkpoint).parent.name
    print('=' * 72)
    print(f'Smoke: {tag}  ckpt={args.checkpoint}')
    print('=' * 72)

    for i, user in enumerate(DEFAULT_PROMPTS, 1):
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
        reply = extract_reply(text, prompt)
        print(f'\n[{i}] User: {user}')
        print(f'Assistant: {reply}')
        print('-' * 72)


if __name__ == '__main__':
    main()

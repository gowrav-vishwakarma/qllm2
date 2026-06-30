#!/usr/bin/env python3
"""Chat with QLLM V11 PAM checkpoint (self-contained HF release)."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from modeling_qllm import load_model

IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
DEFAULT_SYSTEM = 'You are a helpful assistant.'


def get_chat_tokenizer():
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.add_special_tokens({'additional_special_tokens': [IM_START, IM_END]})
    tok.pad_token = tok.eos_token
    return tok


def format_chat_prompt(user_text: str, system: str = DEFAULT_SYSTEM) -> str:
    return (
        f'{IM_START}system\n{system}{IM_END}\n'
        f'{IM_START}user\n{user_text.strip()}{IM_END}\n'
        f'{IM_START}assistant\n'
    )


def generate_reply(
    model: V11LM,
    tokenizer,
    user_text: str,
    *,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, bool, int]:
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    prompt = format_chat_prompt(user_text)
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=im_end_id,
        )
    n_new = out.shape[1] - ids.shape[1]
    gen_ids = out[0, ids.shape[1]:].tolist()
    stopped = im_end_id in gen_ids
    reply = tokenizer.decode(gen_ids, skip_special_tokens=False)
    reply = reply.split(IM_END, 1)[0].strip()
    return reply, stopped, n_new


def main() -> None:
    p = argparse.ArgumentParser(description='Chat with QLLM V11 PAM (HF release)')
    p.add_argument('--checkpoint', default='qllm_v11_e3k3_chat.pt')
    p.add_argument('--config', default='config.json')
    p.add_argument('--prompt', default='')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device=device)
    tokenizer = get_chat_tokenizer()

    if args.prompt:
        reply, stopped, n_new = generate_reply(
            model,
            tokenizer,
            args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f'User> {args.prompt}')
        print(f'Assistant> {reply}')
        print(f'(new_tokens={n_new}, stopped_on_im_end={stopped})')
        return

    print('QLLM V11 PAM chat (empty line to quit)')
    while True:
        try:
            user = input('\nUser> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        reply, stopped, n_new = generate_reply(
            model,
            tokenizer,
            user,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f'Assistant> {reply}')
        if not stopped:
            print(f'  (warning: did not stop on {IM_END}, new_tokens={n_new})')


if __name__ == '__main__':
    main()

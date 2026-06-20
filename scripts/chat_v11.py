#!/usr/bin/env python3
"""Interactive chat with a V11 checkpoint (ChatML, post-SFT or base)."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v7.data import IM_END, format_chat_prompt, get_chat_tokenizer
from v11.model import V11LM, get_config


def main():
    p = argparse.ArgumentParser(description='Chat with a V11 checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--preset', default='v11_e3_k3')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_chat_tokenizer()
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)

    cfg = get_config(args.preset)
    cfg.vocab_size = len(tokenizer)
    model = V11LM(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    m = model._orig_mod if hasattr(model, '_orig_mod') else model

    print('V11 chat (empty line to quit)')
    while True:
        try:
            user = input('\nUser> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            break
        prompt = format_chat_prompt(user)
        ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = m.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.15,
                eos_token_id=im_end_id,
            )
        gen_ids = out[0, ids.shape[1]:].tolist()
        reply = tokenizer.decode(gen_ids, skip_special_tokens=False)
        reply = reply.split(IM_END, 1)[0].strip()
        print(f"Assistant> {reply}")


if __name__ == '__main__':
    main()

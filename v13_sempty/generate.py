"""Prefix completion from a v13_sempty checkpoint.

Usage:
    .venv/bin/python -m v13_sempty.generate \\
      --checkpoint checkpoints_v13_sempty/latest.pt \\
      --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v13_sempty.config import PAMConfig, get_config
from v13_sempty.model import LM


def _load_tokenizer(vocab_size: int):
    if vocab_size >= 50259:
        from v7.data import get_chat_tokenizer
        tok = get_chat_tokenizer()
        if len(tok) != vocab_size:
            raise SystemExit(
                f'Tokenizer vocab {len(tok)} != checkpoint vocab_size {vocab_size}'
            )
        return tok
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def _config_from_ckpt(ckpt: dict, preset_fallback: str) -> PAMConfig:
    raw = ckpt.get('config')
    if isinstance(raw, dict) and raw:
        fields = set(PAMConfig.__dataclass_fields__)
        return PAMConfig(**{k: v for k, v in raw.items() if k in fields})
    return get_config(preset_fallback)


def main() -> None:
    p = argparse.ArgumentParser(description='v13_sempty prefix completion')
    p.add_argument('--checkpoint', default='checkpoints_v13_sempty/latest.pt')
    p.add_argument('--preset', default='tiny',
                   help='Used only if checkpoint has no config dict')
    p.add_argument('--prompt', default='Once upon a time')
    p.add_argument('--max_tokens', type=int, default=40)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--repetition_penalty', type=float, default=1.2)
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f'Checkpoint not found: {ckpt_path}')

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f'Loading {ckpt_path} ...')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = _config_from_ckpt(ckpt, args.preset)
    tok = _load_tokenizer(cfg.vocab_size)
    cfg.vocab_size = len(tok)

    model = LM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    ids = tok.encode(args.prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    text = tok.decode(out[0].tolist())
    print(f'device={device}  vocab={cfg.vocab_size}')
    print(f'\nPrompt: {args.prompt}')
    print(f'Generated: {text}')


if __name__ == '__main__':
    main()

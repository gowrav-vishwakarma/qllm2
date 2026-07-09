"""
V11 prefix completion (same sampling as training ``gen_every``).

Usage:
    uv run python -m v11.generate \\
      --checkpoint checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt \\
      --prompt "In 1923 , the University of"

    uv run python -m v11.generate \\
      --checkpoint checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt \\
      --prompt "The capital of France is" \\
      --max_tokens 80 --temperature 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v7.data import get_chat_tokenizer
from v11.model import V11Config, V11LM, get_config


def _load_tokenizer(vocab_size: int):
    """Match training: chat presets use 50261; plain GPT-2 otherwise."""
    if vocab_size >= 50259:
        tok = get_chat_tokenizer()
        if len(tok) != vocab_size:
            raise SystemExit(
                f'Tokenizer vocab {len(tok)} != checkpoint vocab_size {vocab_size}'
            )
        return tok
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def _config_from_ckpt(ckpt: dict, preset_fallback: str) -> V11Config:
    raw = ckpt.get('config')
    if isinstance(raw, dict) and raw:
        fields = set(V11Config.__dataclass_fields__)
        return V11Config(**{k: v for k, v in raw.items() if k in fields})
    return get_config(preset_fallback)


def main() -> None:
    p = argparse.ArgumentParser(
        description='V11 prefix completion (gen_every-style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--checkpoint',
        default='checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt',
    )
    p.add_argument('--preset', default='v11_e3_k3_chat',
                   help='Used only if checkpoint has no config dict')
    p.add_argument('--prompt', default='In 1923 , the University of')
    p.add_argument('--max_tokens', type=int, default=100)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--repetition_penalty', type=float, default=1.2)
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f'Checkpoint not found: {ckpt_path}')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f'Loading {ckpt_path} ...')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = _config_from_ckpt(ckpt, args.preset)
    tok = _load_tokenizer(cfg.vocab_size)
    cfg.vocab_size = len(tok)

    model = V11LM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    print(f'device={device}  vocab={cfg.vocab_size}  '
          f'tokens={ckpt.get("global_tokens")}  '
          f'best_val_ppl={ckpt.get("best_val_ppl")}')
    print(f'\nPrompt: {args.prompt}')
    print(f'Generated: {text}')


if __name__ == '__main__':
    main()

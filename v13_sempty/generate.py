"""Prefix completion from a v13_sempty checkpoint.

One-shot:
    .venv/bin/python -m v13_sempty.generate \\
      --checkpoint checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt \\
      --prompt "In 1923, the University of"

Interactive (model loaded once, type prompts; blank line or Ctrl-D quits):
    .venv/bin/python -m v13_sempty.generate --checkpoint ... --interactive
  In the loop, `/set temperature=0.7 max_tokens=120 top_k=40` retunes sampling.
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


SAMPLING_KEYS = ('max_tokens', 'temperature', 'top_k', 'top_p', 'repetition_penalty')


def _complete(model, tok, device, prompt: str, s: dict) -> str:
    ids = tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=s['max_tokens'],
            temperature=s['temperature'],
            top_k=s['top_k'],
            top_p=s['top_p'],
            repetition_penalty=s['repetition_penalty'],
        )
    return tok.decode(out[0].tolist())


def _apply_set(line: str, s: dict) -> None:
    """`/set key=value ...` -- retune sampling in the interactive loop."""
    for kv in line.split()[1:]:
        if '=' not in kv:
            print(f'  ignored {kv!r} (use key=value)')
            continue
        k, v = kv.split('=', 1)
        if k not in SAMPLING_KEYS:
            print(f'  unknown key {k!r}; keys: {", ".join(SAMPLING_KEYS)}')
            continue
        s[k] = type(s[k])(float(v)) if k != 'max_tokens' else int(v)
    print('  sampling:', ' '.join(f'{k}={s[k]}' for k in SAMPLING_KEYS))


def _interactive(model, tok, device, s: dict) -> None:
    print('Interactive: type a prompt; `/set key=value` retunes; blank line / Ctrl-D quits.')
    print('  sampling:', ' '.join(f'{k}={s[k]}' for k in SAMPLING_KEYS))
    while True:
        try:
            line = input('\nprompt> ')
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line.strip():
            return
        if line.startswith('/set'):
            _apply_set(line, s)
            continue
        print(_complete(model, tok, device, line, s))


def main() -> None:
    p = argparse.ArgumentParser(description='v13_sempty prefix completion')
    p.add_argument('--checkpoint', default='checkpoints_v13_sempty/latest.pt')
    p.add_argument('--preset', default='tiny',
                   help='Used only if checkpoint has no config dict')
    p.add_argument('--prompt', default='Once upon a time')
    p.add_argument('--interactive', action='store_true',
                   help='load once, then read prompts from stdin in a loop')
    p.add_argument('--max_tokens', type=int, default=80)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--repetition_penalty', type=float, default=1.2)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
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
    cfg.gradient_checkpointing = False

    model = LM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    want = 'cuda' if args.device == 'auto' else args.device
    device = torch.device(want if want == 'cpu' or torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'device={device}  vocab={cfg.vocab_size}  params={n_params/1e6:.1f}M  '
          f'step={ckpt.get("global_step", ckpt.get("step", "?"))}  chrono={cfg.chrono}')

    s = {k: getattr(args, k) for k in SAMPLING_KEYS}
    if args.interactive:
        _interactive(model, tok, device, s)
        return
    text = _complete(model, tok, device, args.prompt, s)
    print(f'\nPrompt: {args.prompt}')
    print(f'Generated: {text}')


if __name__ == '__main__':
    main()

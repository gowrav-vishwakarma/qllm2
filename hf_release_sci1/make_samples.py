#!/usr/bin/env python3
"""Write SAMPLES.md: 2 plain continuations + a 3-turn chat, each greedy and sampled.

    python make_samples.py [--checkpoint sci1_386m_49b.pt] [--device cpu]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modeling_pam2 import load_model, load_tokenizer  # noqa: E402
from run_chat import GREEDY, SAMPLED, ChatSession  # noqa: E402
from run_complete import complete  # noqa: E402

COMPLETIONS = ['In 1923, the University of', 'Photosynthesis is the process by which']
CHAT = ['What is the capital of France?', 'What is it famous for?',
        'Write a short poem about it.']
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=str(HERE / 'sci1_386m_49b.pt'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=str(HERE / 'SAMPLES.md'))
    args = ap.parse_args()
    torch.set_num_threads(16)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device)
    tok = load_tokenizer()
    meta = getattr(model, 'meta', {})
    lines = ['# SCI-1 samples', '',
             f'Checkpoint: `{Path(args.checkpoint).name}` (step {meta.get("global_step")}, '
             f'{meta.get("global_tokens", 0) / 1e9:.2f}B tokens), {model.count_parameters():,} params, '
             f'generated with `modeling_pam2.py` only (no qllm2 repo), {device}, fp32, '
             f'{datetime.date.today().isoformat()}. Reproduce: `python make_samples.py`.', '',
             f'* greedy: temperature 0 (no repetition penalty)',
             f'* sampled: temperature {SAMPLED["temperature"]}, top-k {SAMPLED["top_k"]}, '
             f'top-p {SAMPLED["top_p"]}, repetition penalty {SAMPLED["repetition_penalty"]}, '
             f'seed {SEED} (torch.Generator on {device})', '']

    lines += ['## Plain continuation (max 80 new tokens, stop at `<|endoftext|>`)', '']
    for i, p in enumerate(COMPLETIONS, 1):
        for mode, kw in (('greedy', GREEDY), ('sampled', SAMPLED)):
            text, new, dt = complete(model, tok, p, max_new_tokens=80, seed=SEED, device=device, **kw)
            print(f'[complete {i} {mode}] {p!r} -> {text!r}', flush=True)
            lines += [f'### {i}. {mode}', '', '**Prompt:**', '', '```text', p, '```', '',
                      f'**Output** ({len(new)} tokens, {dt:.1f}s):', '', '```text', p + text, '```', '']

    lines += ['## Chat: one 3-turn ChatML session (no system message, max 160 new tokens per reply, '
              'stop at `<|im_end|>`, state carried between turns)', '']
    for mode, kw in (('greedy', GREEDY), ('sampled', SAMPLED)):
        s = ChatSession(model, tok, '', device)
        lines += [f'### {mode}', '']
        for j, msg in enumerate(CHAT, 1):
            reply, info = s.reply(msg, max_new_tokens=160, seed=SEED, **kw)
            print(f'[chat {mode} turn {j}] {msg!r} -> {reply!r}', flush=True)
            lines += [f'**Turn {j}, user:** {msg}', '', f'**Assistant** ({info["new_tokens"]} tokens, '
                      f'stopped_on_im_end={info["stopped_on_im_end"]}, {info["seconds"]:.1f}s):', '',
                      '```text', reply, '```', '']
        lines += ['<details><summary>Raw ChatML the model saw</summary>', '', '```text', s.transcript(),
                  '```', '', '</details>', '']
    Path(args.out).write_text('\n'.join(lines))
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

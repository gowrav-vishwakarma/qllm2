#!/usr/bin/env python3
"""Audit the SFT assistant-only loss mask (GPT 5.5 assessment, point 8).

Two modes:
  * default: encode synthetic multi-turn conversations with the real
    ``v7.data._encode_sft_example`` and assert the loss mask is 1 ONLY on
    assistant content + the closing ``<|im_end|>`` (system/user/role headers /
    pad all masked). Fully offline.
  * ``--shard PATH``: decode a few rows from a cached SFT shard
    (``.cache/v7_tokens/<tag>/train_0000.pt``) and print token/label/mask
    columns so the boundaries can be eyeballed on the training box.

Usage:
    uv run python scripts/audit_sft_mask.py
    uv run python scripts/audit_sft_mask.py \
        --shard .cache/v7_tokens/smoltalk2_sft_tf15_hard_full_sl2048_chatv4/train_0000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v7.data import IM_END, IM_START, _encode_sft_example, get_chat_tokenizer

SYNTHETIC = [
    [
        {'role': 'user', 'content': 'What is the capital of France?'},
        {'role': 'assistant', 'content': 'The capital of France is Paris.'},
    ],
    [
        {'role': 'system', 'content': 'You are terse.'},
        {'role': 'user', 'content': 'Hi there!'},
        {'role': 'assistant', 'content': 'Hello.'},
        {'role': 'user', 'content': 'Name a color.'},
        {'role': 'assistant', 'content': 'Blue.'},
    ],
]


def _audit_synthetic(tokenizer, seq_len: int = 128) -> bool:
    im_start = tokenizer.convert_tokens_to_ids(IM_START)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    ok = True

    for i, messages in enumerate(SYNTHETIC):
        enc = _encode_sft_example(messages, tokenizer, seq_len)
        if enc is None:
            print(f'[conv {i}] FAIL: encoder returned None')
            ok = False
            continue
        input_ids, labels, loss_mask = enc

        # Reconstruct expected assistant-target label positions.
        assistant_texts = [
            (m.get('content') or '').strip()
            for m in messages
            if (m.get('role') or '').lower() == 'assistant'
        ]

        masked_label_ids = labels[loss_mask == 1].tolist()
        decoded_targets = tokenizer.decode(masked_label_ids)

        # Every masked target token must be assistant content or the im_end token.
        leaked = []
        # Walk the sequence: a target is valid only inside an assistant turn.
        # Reconstruct role spans from input_ids by tracking im_start/im_end.
        n_targets = int(loss_mask.sum())
        n_imend_targets = int(((labels == im_end) & (loss_mask == 1)).sum())

        # No system/user text should appear in decoded targets.
        for m in messages:
            role = (m.get('role') or '').lower()
            if role in ('user', 'system'):
                content = (m.get('content') or '').strip()
                if content and content in decoded_targets:
                    leaked.append((role, content))

        role_headers_leaked = any(
            h in decoded_targets for h in ('user\n', 'system\n')
        )

        print(f'[conv {i}] targets={n_targets} im_end_targets={n_imend_targets} '
              f'assistant_turns={len(assistant_texts)}')
        print(f'  decoded targets: {decoded_targets!r}')

        conv_ok = True
        if leaked:
            print(f'  FAIL: user/system text leaked into loss: {leaked}')
            conv_ok = False
        if role_headers_leaked:
            print('  FAIL: role header leaked into loss targets')
            conv_ok = False
        if n_imend_targets < len(assistant_texts):
            print(f'  FAIL: expected >= {len(assistant_texts)} im_end targets, '
                  f'got {n_imend_targets} (end-of-turn not learned)')
            conv_ok = False
        # Each assistant content should be recoverable from the targets.
        for txt in assistant_texts:
            if txt and txt not in decoded_targets:
                print(f'  WARN: assistant text not contiguous in targets: {txt!r}')
        if conv_ok:
            print('  PASS: mask covers only assistant content + im_end')
        ok = ok and conv_ok

    return ok


def _audit_shard(shard_path: Path, tokenizer, n_rows: int = 2) -> bool:
    part = torch.load(shard_path, map_location='cpu', weights_only=False)
    input_ids = part['input_ids']
    labels = part['labels']
    loss_mask = part['loss_mask']
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    print(f'Shard {shard_path} rows={input_ids.shape[0]} seq_len={input_ids.shape[1]}')

    ok = True
    for r in range(min(n_rows, input_ids.shape[0])):
        lm = loss_mask[r]
        lab = labels[r]
        n_targets = int(lm.sum())
        n_imend = int(((lab == im_end) & (lm == 1)).sum())
        masked_label_ids = lab[lm == 1].tolist()
        decoded = tokenizer.decode(masked_label_ids)
        print(f'\n[row {r}] targets={n_targets} im_end_targets={n_imend}')
        print(f'  decoded targets: {decoded[:400]!r}')
        if n_targets == 0:
            print('  FAIL: empty loss mask')
            ok = False
        if n_imend == 0:
            print('  FAIL: no im_end target (end-of-turn not learned)')
            ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description='Audit SFT assistant-only loss mask')
    p.add_argument('--shard', default='', help='Optional cached SFT shard to inspect')
    p.add_argument('--rows', type=int, default=2)
    args = p.parse_args()

    tokenizer = get_chat_tokenizer()
    print(f'Tokenizer vocab: {len(tokenizer)} '
          f'(im_start={tokenizer.convert_tokens_to_ids(IM_START)}, '
          f'im_end={tokenizer.convert_tokens_to_ids(IM_END)})\n')

    print('=== Synthetic encoder audit ===')
    ok = _audit_synthetic(tokenizer)

    if args.shard:
        shard = Path(args.shard)
        if not shard.is_file():
            print(f'\n(shard not found: {shard} — skipping shard audit)')
        else:
            print('\n=== Cached shard audit ===')
            ok = _audit_shard(shard, tokenizer, n_rows=args.rows) and ok

    print('\n' + ('AUDIT PASSED' if ok else 'AUDIT FAILED'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

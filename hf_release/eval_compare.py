#!/usr/bin/env python3
"""Compare pretrain prefix completion vs SFT chat on the same questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from modeling_qllm import load_model
from run_chat import (
    DEFAULT_SYSTEM,
    NO_THINK_HINT,
    THINK_START,
    generate_reply,
    get_chat_tokenizer,
)
from run_complete import complete_prefix

SFT_SYSTEM = f'{DEFAULT_SYSTEM}\n\n{NO_THINK_HINT}'.strip()
TEMPS = (0.1, 0.7)
ANSWER_KEYS = ('pretrain@0.1', 'pretrain@0.7', 'sft@0.1', 'sft@0.7')


def _load_suite(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _answer_key(model: str, temp: float) -> str:
    return f'{model}@{temp:g}'


def _heuristic_note(prompt_id: str, text: str) -> str:
    blob = text.lower()
    if prompt_id in ('france', 'france_lower', 'france_title'):
        return 'ok' if 'paris' in blob else 'miss'
    if prompt_id == 'usa_capital':
        return 'ok' if 'washington' in blob else 'miss'
    if prompt_id == 'karl_marx':
        return 'ok' if 'marx' in blob or 'communist' in blob or 'philosopher' in blob else 'miss'
    if prompt_id == 'shakespeare':
        return 'ok' if 'shakespeare' in blob else 'miss'
    if prompt_id == 'largest_planet':
        return 'ok' if 'jupiter' in blob else 'miss'
    if prompt_id == 'math_2plus2':
        stripped = text.strip()
        if re.search(r'\b4\b', stripped[:20]):
            return 'ok'
        return 'fail'
    if prompt_id == 'math_multiply':
        return 'ok' if '391' in text.replace(',', '') else 'fail'
    if THINK_START in text:
        return 'think_block'
    if len(text.split()) > 80:
        return 'ramble'
    return ''


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_compare(
    *,
    pretrain_ckpt: Path,
    sft_ckpt: Path,
    suite: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temps = [float(t) for t in suite.get('temperatures', TEMPS)]
    max_new = int(suite.get('max_new_tokens', 128))

    print(f'Loading pretrain {pretrain_ckpt.name} ...')
    pretrain = load_model(str(pretrain_ckpt), device=device)
    print(f'Loading SFT {sft_ckpt.name} ...')
    sft = load_model(str(sft_ckpt), device=device)
    tokenizer = get_chat_tokenizer(pretrain.config.vocab_size)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    max_prompt_tokens = pretrain.config.max_seq_len - max_new

    questions_out: dict[str, Any] = {}

    for item in suite['questions']:
        qid = item['id']
        prompt = item['prompt'].strip()
        prefix = item['prefix'].strip()
        answers: dict[str, str] = {}
        heuristics: dict[str, str] = {}
        meta: dict[str, Any] = {}

        for temp in temps:
            _set_seed(seed)
            _, pre_cont, pre_n = complete_prefix(
                pretrain,
                tokenizer,
                prefix,
                device=device,
                max_new_tokens=max_new,
                temperature=temp,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            pre_key = _answer_key('pretrain', temp)
            answers[pre_key] = pre_cont
            heuristics[pre_key] = _heuristic_note(qid, pre_cont)
            meta[pre_key] = {'new_tokens': pre_n}

            _set_seed(seed)
            messages = [{'role': 'user', 'content': prompt}]
            reply, stopped, sft_n, raw = generate_reply(
                sft,
                tokenizer,
                messages,
                device=device,
                im_end_id=im_end_id,
                max_new_tokens=max_new,
                temperature=temp,
                max_prompt_tokens=max_prompt_tokens,
                default_system=SFT_SYSTEM,
                strip_thinking_blocks=False,
            )
            sft_key = _answer_key('sft', temp)
            answers[sft_key] = reply
            heuristics[sft_key] = _heuristic_note(qid, reply) or _heuristic_note(qid, raw)
            meta[sft_key] = {'new_tokens': sft_n, 'stopped_on_im_end': stopped}

            print(f'[{qid} T={temp}] pretrain: {pre_cont[:60]!r}...')
            print(f'[{qid} T={temp}] sft:      {reply[:60]!r}...')

        questions_out[qid] = {
            'category': item.get('category', ''),
            'prompt': prompt,
            'prefix': prefix,
            'notes': item.get('notes', ''),
            'answers': answers,
            'heuristic': heuristics,
            'meta': meta,
        }

    return {
        'round_tag': suite.get('round_tag', 'unknown'),
        'pretrain_checkpoint': str(pretrain_ckpt),
        'sft_checkpoint': str(sft_ckpt),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'device': str(device),
        'seed': seed,
        'temperatures': temps,
        'max_new_tokens': max_new,
        'sft_system': SFT_SYSTEM,
        'questions': questions_out,
    }


def _md_inline(text: str, max_len: int = 72) -> str:
    s = text.replace('\n', ' ').replace('|', '\\|')
    if len(s) > max_len:
        s = s[: max_len - 3] + '...'
    return s.replace('`', "'")


def _render_markdown(report: dict[str, Any]) -> str:
    tag = report['round_tag']
    lines: list[str] = [
        f'# Pretrain vs SFT — {tag}',
        '',
        f'Generated **{report["run_at"]}** on `{report["device"]}`.',
        '',
        'Same factual questions: **pretrain** completes the `prefix`; **SFT** answers in ChatML chat.',
        'Four answers per question: `pretrain@0.1`, `pretrain@0.7`, `sft@0.1`, `sft@0.7`.',
        'Full text: see the JSON log.',
        '',
        '## Summary',
        '',
        '| id | category | pretrain@0.1 | pretrain@0.7 | sft@0.1 | sft@0.7 |',
        '| -- | -------- | ------------ | ------------ | ------- | ------- |',
    ]

    tally: Counter[str] = Counter()
    for qid, row in report['questions'].items():
        heur = row.get('heuristic', {})
        ans = row['answers']
        cells = []
        for key in ANSWER_KEYS:
            h = heur.get(key, '')
            if h:
                tally[f'{key}:{h}'] += 1
            cells.append(_md_inline(ans.get(key, '')))
        lines.append(
            f'| {qid} | {row.get("category", "")} | '
            + ' | '.join(cells)
            + ' |'
        )

    lines.extend(['', '## Heuristic tally', ''])
    if tally:
        for label, count in sorted(tally.items()):
            lines.append(f'- `{label}`: {count}')
    else:
        lines.append('- (none)')

    lines.extend([
        '',
        '## Reproduce',
        '',
        '```bash',
        'cd hf_release',
        'uv run python eval_compare.py \\',
        f'  --pretrain {Path(report["pretrain_checkpoint"]).name} \\',
        f'  --sft {Path(report["sft_checkpoint"]).name} \\',
        '  --prompts eval_prompts_compare.yaml \\',
        f'  --out-md SAMPLES_{tag}-compare.md \\',
        f'  --out-json ../logs/v11/{tag}_compare.json',
        '```',
        '',
    ])
    return '\n'.join(lines).rstrip() + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description='Pretrain vs SFT comparison eval')
    p.add_argument('--pretrain', default='qllm_v11_e3k3_pretrain.pt')
    p.add_argument('--sft', default='qllm_v11_e3k3_chat.pt')
    p.add_argument('--prompts', default='eval_prompts_compare.yaml')
    p.add_argument('--out-md', default='SAMPLES_round-6b-gate-compare.md')
    p.add_argument('--out-json', default='../logs/v11/round-6b-gate_compare.json')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--round-tag', default='')
    args = p.parse_args()

    suite_path = Path(args.prompts)
    if not suite_path.is_file():
        raise SystemExit(f'Prompts file not found: {suite_path}')

    pretrain_ckpt = Path(args.pretrain)
    sft_ckpt = Path(args.sft)
    for path, label in ((pretrain_ckpt, 'pretrain'), (sft_ckpt, 'sft')):
        if not path.is_file():
            raise SystemExit(f'{label} checkpoint not found: {path}')

    suite = _load_suite(suite_path)
    seed = args.seed if args.seed is not None else int(suite.get('seed', 42))

    report = _run_compare(
        pretrain_ckpt=pretrain_ckpt,
        sft_ckpt=sft_ckpt,
        suite=suite,
        seed=seed,
    )
    if args.round_tag:
        report['round_tag'] = args.round_tag
    tag = report['round_tag']

    out_json = Path(args.out_json)
    if args.out_json == '../logs/v11/round-6b-gate_compare.json' and tag != 'round-6b-gate':
        out_json = Path(f'../logs/v11/{tag}_compare.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {out_json}')

    md = _render_markdown(report)
    out_md = Path(args.out_md)
    if args.out_md == 'SAMPLES_round-6b-gate-compare.md' and tag != 'round-6b-gate':
        out_md = Path(f'SAMPLES_{tag}-compare.md')
    out_md.write_text(md, encoding='utf-8')
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()

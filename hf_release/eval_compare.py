#!/usr/bin/env python3
"""Compare pretrain prefix completion vs SFT chat on the same questions.

Diagnostics (GPT 5.5 assessment):
  * ``--with-pretrain-chat`` also runs the PRETRAIN checkpoint through the SFT
    ChatML path (keys ``pretrain_chat@T``). If pretrain+ChatML also fails, the
    regression is prompt-format mismatch; if it still answers while SFT degrades,
    the SFT training is the dominant factor.
  * ``span_hit@20`` scores whether any gold span appears in the first 20 generated
    tokens — a clean factual-recall signal that ignores later rambling.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
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
SPAN_WINDOW_TOKENS = 20


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


def _span_hit(text: str, gold: list[str], tokenizer, n_tokens: int = SPAN_WINDOW_TOKENS) -> bool | None:
    """True if any gold span appears within the first ``n_tokens`` generated tokens."""
    if not gold:
        return None
    ids = tokenizer.encode(text)[:n_tokens]
    window = tokenizer.decode(ids).lower()
    return any(str(g).lower() in window for g in gold)


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
    with_pretrain_chat: bool,
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

    models = ['pretrain', 'sft']
    if with_pretrain_chat:
        models.append('pretrain_chat')
    answer_keys = [_answer_key(m, t) for t in temps for m in models]

    questions_out: dict[str, Any] = {}

    for item in suite['questions']:
        qid = item['id']
        prompt = item['prompt'].strip()
        prefix = item['prefix'].strip()
        gold = list(item.get('gold') or [])
        answers: dict[str, str] = {}
        heuristics: dict[str, str] = {}
        span_hits: dict[str, bool | None] = {}
        meta: dict[str, Any] = {}

        for temp in temps:
            # Pretrain prefix completion.
            _set_seed(seed)
            _, pre_cont, pre_n = complete_prefix(
                pretrain, tokenizer, prefix,
                device=device, max_new_tokens=max_new, temperature=temp,
                top_k=50, top_p=0.9, repetition_penalty=1.2,
            )
            pre_key = _answer_key('pretrain', temp)
            answers[pre_key] = pre_cont
            heuristics[pre_key] = _heuristic_note(qid, pre_cont)
            span_hits[pre_key] = _span_hit(pre_cont, gold, tokenizer)
            meta[pre_key] = {'new_tokens': pre_n}

            # SFT chat.
            _set_seed(seed)
            messages = [{'role': 'user', 'content': prompt}]
            reply, stopped, sft_n, raw = generate_reply(
                sft, tokenizer, messages,
                device=device, im_end_id=im_end_id, max_new_tokens=max_new,
                temperature=temp, max_prompt_tokens=max_prompt_tokens,
                default_system=SFT_SYSTEM, strip_thinking_blocks=False,
            )
            sft_key = _answer_key('sft', temp)
            answers[sft_key] = reply
            heuristics[sft_key] = _heuristic_note(qid, reply) or _heuristic_note(qid, raw)
            span_hits[sft_key] = _span_hit(reply, gold, tokenizer)
            meta[sft_key] = {'new_tokens': sft_n, 'stopped_on_im_end': stopped}

            print(f'[{qid} T={temp}] pretrain: {pre_cont[:60]!r}...')
            print(f'[{qid} T={temp}] sft:      {reply[:60]!r}...')

            # Diagnostic: pretrain checkpoint through the ChatML chat path.
            if with_pretrain_chat:
                _set_seed(seed)
                pc_reply, pc_stopped, pc_n, pc_raw = generate_reply(
                    pretrain, tokenizer, messages,
                    device=device, im_end_id=im_end_id, max_new_tokens=max_new,
                    temperature=temp, max_prompt_tokens=max_prompt_tokens,
                    default_system=SFT_SYSTEM, strip_thinking_blocks=False,
                )
                pc_key = _answer_key('pretrain_chat', temp)
                answers[pc_key] = pc_reply
                heuristics[pc_key] = _heuristic_note(qid, pc_reply) or _heuristic_note(qid, pc_raw)
                span_hits[pc_key] = _span_hit(pc_reply, gold, tokenizer)
                meta[pc_key] = {'new_tokens': pc_n, 'stopped_on_im_end': pc_stopped}
                print(f'[{qid} T={temp}] pre_chat: {pc_reply[:60]!r}...')

        questions_out[qid] = {
            'category': item.get('category', ''),
            'prompt': prompt,
            'prefix': prefix,
            'gold': gold,
            'notes': item.get('notes', ''),
            'answers': answers,
            'heuristic': heuristics,
            'span_hit': span_hits,
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
        'with_pretrain_chat': with_pretrain_chat,
        'answer_keys': answer_keys,
        'span_window_tokens': SPAN_WINDOW_TOKENS,
        'questions': questions_out,
    }


def _md_inline(text: str, max_len: int = 60) -> str:
    s = text.replace('\n', ' ').replace('|', '\\|')
    if len(s) > max_len:
        s = s[: max_len - 3] + '...'
    return s.replace('`', "'")


def _render_markdown(report: dict[str, Any]) -> str:
    tag = report['round_tag']
    keys = report['answer_keys']
    diag = report.get('with_pretrain_chat')
    lines: list[str] = [
        f'# Pretrain vs SFT — {tag}' + (' (diagnostic)' if diag else ''),
        '',
        f'Generated **{report["run_at"]}** on `{report["device"]}`.',
        '',
        'Same factual questions: **pretrain** completes the `prefix`; **SFT** answers in ChatML chat.',
    ]
    if diag:
        lines.append('Diagnostic arm: **pretrain_chat** runs the pretrain checkpoint through the ChatML chat path.')
    lines.append(f'Answers per question: `{"`, `".join(keys)}`.')
    lines.append(f'`span_hit@{report["span_window_tokens"]}`: gold span in first {report["span_window_tokens"]} generated tokens.')
    lines.append('Full text: see the JSON log.')
    lines.append('')

    # Group questions by category (preserve first-seen order).
    by_cat: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    cat_order: list[str] = []
    for qid, row in report['questions'].items():
        cat = row.get('category', '') or 'uncategorized'
        if cat not in cat_order:
            cat_order.append(cat)
        by_cat[cat].append((qid, row))

    header = '| id | ' + ' | '.join(keys) + ' |'
    divider = '| -- | ' + ' | '.join(['---'] * len(keys)) + ' |'

    for cat in cat_order:
        lines.append(f'## {cat}')
        lines.append('')
        lines.append(header)
        lines.append(divider)
        for qid, row in by_cat[cat]:
            ans = row['answers']
            cells = [_md_inline(ans.get(k, '')) for k in keys]
            lines.append(f'| {qid} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    # span_hit@N accuracy per answer key.
    lines.append(f'## span_hit@{report["span_window_tokens"]} accuracy')
    lines.append('')
    lines.append('| key | hits | scored | rate |')
    lines.append('| --- | ---- | ------ | ---- |')
    for key in keys:
        hits = scored = 0
        for row in report['questions'].values():
            val = row.get('span_hit', {}).get(key)
            if val is None:
                continue
            scored += 1
            hits += 1 if val else 0
        rate = f'{hits / scored:.2f}' if scored else '—'
        lines.append(f'| {key} | {hits} | {scored} | {rate} |')
    lines.append('')

    # Heuristic tally (secondary).
    tally: Counter[str] = Counter()
    for row in report['questions'].values():
        for key in keys:
            h = row.get('heuristic', {}).get(key, '')
            if h:
                tally[f'{key}:{h}'] += 1
    lines.append('## Heuristic tally (secondary)')
    lines.append('')
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
        '  --prompts eval_prompts_compare.yaml \\'
        + ('' if not diag else '\n  --with-pretrain-chat \\'),
        f'  --round-tag {tag}',
        '```',
        '',
    ])
    return '\n'.join(lines).rstrip() + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description='Pretrain vs SFT comparison eval')
    p.add_argument('--pretrain', default='qllm_v11_e3k3_pretrain.pt')
    p.add_argument('--sft', default='qllm_v11_e3k3_chat.pt')
    p.add_argument('--prompts', default='eval_prompts_compare.yaml')
    p.add_argument('--out-md', default='')
    p.add_argument('--out-json', default='')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--round-tag', default='')
    p.add_argument(
        '--with-pretrain-chat',
        action='store_true',
        help='Diagnostic: also run the pretrain checkpoint through the ChatML chat path',
    )
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
        with_pretrain_chat=args.with_pretrain_chat,
    )
    if args.round_tag:
        report['round_tag'] = args.round_tag
    tag = report['round_tag']

    # Diagnostic runs never overwrite the shipped SAMPLES doc.
    suffix = '-compare-diagnostic' if args.with_pretrain_chat else '-compare'
    json_suffix = '_compare_diagnostic' if args.with_pretrain_chat else '_compare'
    out_md = Path(args.out_md) if args.out_md else Path(f'SAMPLES_{tag}{suffix}.md')
    out_json = Path(args.out_json) if args.out_json else Path(f'../logs/v11/{tag}{json_suffix}.json')

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {out_json}')

    md = _render_markdown(report)
    out_md.write_text(md, encoding='utf-8')
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()

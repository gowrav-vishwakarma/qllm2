#!/usr/bin/env python3
"""Batch chat eval for shipped HF checkpoints (Round 1+)."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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

PROFILE_DEFAULTS = {
    'recommended': {
        'description': 'Recommended inference (--no-think)',
        'add_no_think_hint': True,
        'strip_thinking': True,
        'temperatures': [0.0, 0.7],
        'max_new_tokens': 256,
    },
    'raw': {
        'description': 'Raw model output (training-default system)',
        'add_no_think_hint': False,
        'strip_thinking': False,
        'temperatures': [0.0, 0.7],
        'max_new_tokens': 512,
    },
}


def _load_suite(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    profiles = {**PROFILE_DEFAULTS, **(data.get('profiles') or {})}
    for name, cfg in profiles.items():
        merged = {**PROFILE_DEFAULTS.get(name, {}), **cfg}
        profiles[name] = merged
    data['profiles'] = profiles
    return data


def _system_for_profile(base_system: str, profile_cfg: dict[str, Any]) -> str:
    if profile_cfg.get('add_no_think_hint'):
        return f'{base_system}\n\n{NO_THINK_HINT}'.strip()
    return base_system


def _heuristic_note(prompt_id: str, prompt: str, reply: str, raw: str) -> str:
    blob = f'{reply} {raw}'.lower()
    if prompt_id in ('france_lower', 'france_title', 'usa_capital'):
        return 'ok' if 'paris' in blob or 'washington' in blob else 'miss'
    if prompt_id == 'math_2plus2':
        stripped = reply.strip()
        if re.match(r'^4\b', stripped) or re.match(r'^The answer is 4\b', stripped, re.I):
            return 'ok'
        if re.match(r'^2\s*\+\s*2\s*=\s*4', stripped):
            return 'ok'
        return 'fail'
    if prompt_id == 'math_multiply':
        return 'ok' if '391' in reply.replace(',', '') else 'fail'
    if prompt_id == 'largest_planet':
        return 'ok' if 'jupiter' in blob else 'miss'
    if THINK_START in raw:
        return 'think_block'
    if len(reply.split()) > 80:
        return 'ramble'
    return ''


def _run_eval(
    *,
    checkpoint: Path,
    suite: dict[str, Any],
    profile_names: list[str],
    seed: int,
) -> dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(str(checkpoint), device=device)
    tokenizer = get_chat_tokenizer(model.config.vocab_size)
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    base_system = DEFAULT_SYSTEM
    prompts = suite['prompts']
    profiles = suite['profiles']
    results: list[dict[str, Any]] = []

    for profile_name in profile_names:
        cfg = profiles[profile_name]
        system = _system_for_profile(base_system, cfg)
        max_new = int(cfg['max_new_tokens'])
        max_prompt_tokens = model.config.max_seq_len - max_new

        for temp in cfg['temperatures']:
            temp_f = float(temp)
            if temp_f > 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            for item in prompts:
                messages = [{'role': 'user', 'content': item['prompt'].strip()}]
                reply, stopped, n_new, raw = generate_reply(
                    model,
                    tokenizer,
                    messages,
                    device=device,
                    im_end_id=im_end_id,
                    max_new_tokens=max_new,
                    temperature=temp_f,
                    max_prompt_tokens=max_prompt_tokens,
                    default_system=system,
                    strip_thinking_blocks=bool(cfg.get('strip_thinking')),
                )
                results.append(
                    {
                        'id': item['id'],
                        'category': item['category'],
                        'prompt': item['prompt'],
                        'notes': item.get('notes', ''),
                        'profile': profile_name,
                        'temperature': temp_f,
                        'system': system,
                        'max_new_tokens': max_new,
                        'reply': reply,
                        'raw_reply': raw,
                        'has_thinking': THINK_START in raw,
                        'new_tokens': n_new,
                        'stopped_on_im_end': stopped,
                        'heuristic': _heuristic_note(item['id'], item['prompt'], reply, raw),
                    }
                )
                print(
                    f"[{profile_name} T={temp_f}] {item['id']}: "
                    f"{reply[:80]!r}{'...' if len(reply) > 80 else ''}"
                )

    return {
        'round_tag': suite.get('round_tag', 'unknown'),
        'checkpoint': str(checkpoint),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'device': str(device),
        'seed': seed,
        'base_system': base_system,
        'profiles_run': profile_names,
        'n_generations': len(results),
        'results': results,
    }


def _md_fence(text: str, info: str = 'text') -> str:
    """Wrap text in a fenced markdown block (handles nested ``` in model output)."""
    n = 3
    while ('`' * n) in text:
        n += 1
    fence = '`' * n
    return f'{fence}{info}\n{text}\n{fence}'


def _md_inline(text: str, max_len: int | None = None) -> str:
    """Escape backticks for inline markdown code."""
    s = text.replace('\n', ' ')
    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 3] + '...'
    return s.replace('`', "'")


def _render_markdown(report: dict[str, Any], suite: dict[str, Any]) -> str:
    lines: list[str] = []
    tag = report['round_tag']
    lines.append(f'# Sample Q&A — {tag}')
    lines.append('')
    lines.append(
        f'Generated **{report["run_at"]}** from `{Path(report["checkpoint"]).name}` '
        f'({report["n_generations"]} generations on {report["device"]}).'
    )
    lines.append('')
    lines.append('## What to ask at 2B tokens')
    lines.append('')
    lines.append('| Category | Training source | Expectation |')
    lines.append('| -------- | --------------- | ----------- |')
    lines.append('| web_factual | DCLM-Edu + FineWeb-Edu (~92% pretrain) | Hit-or-miss; may ramble |')
    lines.append('| edu_science | Web blend | Often verbose |')
    lines.append('| instruction | smoltalk2 SFT (hard filter) | ChatML + short answers |')
    lines.append('| toy_code | smoltalk2 SFT | Sometimes a stub |')
    lines.append('| math_stress | Not targeted | Usually wrong |')
    lines.append('| reasoning_format | smoltalk2 Mid + 15% think SFT | May emit thinking blocks |')
    lines.append('| prompt_sensitivity | GPT-2 BPE | Case changes output |')
    lines.append('| chat_social | smoltalk2 SFT | Usually coherent |')
    lines.append('')
    lines.append('## Run settings')
    lines.append('')
    lines.append(f'- Base system: `{report["base_system"]}`')
    lines.append(f'- Profiles: `{", ".join(report["profiles_run"])}`')
    lines.append(f'- Seed (sampling): `{report["seed"]}` when temperature > 0')
    lines.append('')
    for name in report['profiles_run']:
        cfg = suite['profiles'][name]
        lines.append(
            f'- **{name}**: {cfg.get("description", "")} — '
            f'temps {cfg["temperatures"]}, max_new_tokens={cfg["max_new_tokens"]}, '
            f'strip_thinking={cfg.get("strip_thinking")}'
        )
    lines.append('')
    lines.append('Reproduce:')
    lines.append('')
    lines.append('```bash')
    lines.append('cd hf_release')
    lines.append('uv run python eval_chat.py \\')
    lines.append(f'  --checkpoint {Path(report["checkpoint"]).name} \\')
    lines.append('  --prompts eval_prompts_round1.yaml \\')
    lines.append(f'  --out-md SAMPLES_{tag}.md \\')
    lines.append('  --out-json ../logs/v11/round1_chat_eval.json')
    lines.append('```')
    lines.append('')
    lines.append('## Summary (recommended profile, temperature 0.0)')
    lines.append('')
    lines.append('| id | category | heuristic | stopped | reply (truncated) |')
    lines.append('| -- | -------- | --------- | ------- | ----------------- |')
    for row in report['results']:
        if row['profile'] != 'recommended' or row['temperature'] != 0.0:
            continue
        trunc = _md_inline(row['reply'], max_len=72)
        trunc = trunc.replace('|', '\\|')
        lines.append(
            f'| {row["id"]} | {row["category"]} | {row["heuristic"] or "—"} | '
            f'{row["stopped_on_im_end"]} | {trunc} |'
        )
    lines.append('')
    lines.append('## Full transcripts')
    lines.append('')

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report['results']:
        by_cat[row['category']].append(row)

    for category in sorted(by_cat.keys()):
        lines.append(f'### {category}')
        lines.append('')
        for row in by_cat[category]:
            lines.append(
                f'#### `{row["id"]}` — profile `{row["profile"]}`, T={row["temperature"]}'
            )
            lines.append('')
            lines.append(f'**User:** {row["prompt"]}')
            lines.append('')
            lines.append('**Assistant:**')
            lines.append('')
            lines.append(_md_fence(row['reply']))
            if row['has_thinking'] and row['reply'] != row['raw_reply']:
                lines.append('')
                raw_preview = row['raw_reply'] if len(row['raw_reply']) <= 400 else row['raw_reply'][:400] + '...'
                lines.append(f'*(raw before strip)*')
                lines.append('')
                lines.append(_md_fence(raw_preview, info='text raw-before-strip'))
            lines.append('')
            meta = (
                f'new_tokens={row["new_tokens"]}, stopped_on_im_end={row["stopped_on_im_end"]}, '
                f'has_thinking={row["has_thinking"]}'
            )
            if row['heuristic']:
                meta += f', heuristic={row["heuristic"]}'
            lines.append(f'*{meta}*')
            lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description='Batch chat eval (HF release bundle)')
    p.add_argument('--checkpoint', default='qllm_v11_e3k3_chat.pt')
    p.add_argument('--prompts', default='eval_prompts_round1.yaml')
    p.add_argument('--out-md', default='SAMPLES_round-2b-gate.md')
    p.add_argument('--out-json', default='../logs/v11/round1_chat_eval.json')
    p.add_argument('--profiles', default='recommended,raw')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    suite_path = Path(args.prompts)
    if not suite_path.is_file():
        raise SystemExit(f'Prompts file not found: {suite_path}')

    suite = _load_suite(suite_path)
    profile_names = [x.strip() for x in args.profiles.split(',') if x.strip()]
    for name in profile_names:
        if name not in suite['profiles']:
            raise SystemExit(f'Unknown profile: {name}')

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f'Checkpoint not found: {ckpt}')

    report = _run_eval(
        checkpoint=ckpt,
        suite=suite,
        profile_names=profile_names,
        seed=args.seed,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {out_json}')

    md = _render_markdown(report, suite)
    out_md = Path(args.out_md)
    out_md.write_text(md, encoding='utf-8')
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()

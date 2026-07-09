#!/usr/bin/env python3
"""Profile smoltalk2 SFT *targets* — what the model actually imitates.

Motivation (2026-07-09 regression review): masking is correct and the base knows
the facts, so the SFT regression is a distribution/style problem in the assistant
targets ("The question asks..." prose, language switches, reasoning openings). This
script samples the assistant responses and profiles them so the split allowlist and
the think cap can be chosen from evidence instead of guesses.

For each sampled conversation it looks at the FIRST assistant turn (the direct answer
to the first user question) and reports:
  * first-word / first-3-word prefix histograms (the "mode collapse" signal)
  * share starting with generic openers (The / In / Let's / The question / ...)
  * assistant length distribution (words)
  * think vs no_think share
  * turn-count distribution
  * suspected non-English share (cheap heuristic; uses langdetect if installed)
  * per-split kept counts

Modes:
  default : stream through the SAME filter training uses (allowlist + caps + think cap)
  --raw   : bypass all filtering (see the pre-fix distribution round-6b trained on)

Usage:
    uv run python scripts/profile_sft_targets.py --limit 1500
    uv run python scripts/profile_sft_targets.py --raw --think-fraction 1.0 --limit 1500
    uv run python scripts/profile_sft_targets.py --limit 2000 --out-json logs/v11/sft_profile.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v7.data import (
    _cap_keep,
    _messages_have_think,
    _should_filter_smoltalk,
    _smoltalk2_row_iter,
    _smoltalk2_split_cap,
    _smoltalk2_split_names,
    _smoltalk2_sft_splits,
    _think_keep,
)

GENERIC_OPENERS = (
    'the question', 'the answer', 'in this', 'in the context', "let's", 'lets',
    'to answer', 'this concept', 'this is', 'sure', 'here is', 'here are',
    'well,', 'i think', 'i cannot', "i'm sorry", 'as an ai',
)

# Cheap non-English stopword markers (only used if langdetect is unavailable).
NON_EN_MARKERS = {
    'de': (' der ', ' die ', ' und ', ' ist ', ' ich ', ' hier ', ' sie ', ' einige '),
    'fr': (' le ', ' la ', ' les ', ' est ', ' une ', ' vous ', ' avec '),
    'es': (' el ', ' la ', ' los ', ' una ', ' con ', ' para ', ' porque '),
    'it': (' il ', ' che ', ' una ', ' sono ', ' della '),
}

try:  # optional, better language ID if present
    from langdetect import detect as _lang_detect  # type: ignore
except Exception:  # noqa: BLE001
    _lang_detect = None


def _first_assistant(messages: list[dict]) -> str:
    for m in messages:
        if (m.get('role') or '').lower() == 'assistant':
            return (m.get('content') or '').strip()
    return ''


def _detect_non_english(text: str) -> bool:
    sample = text[:600]
    if _lang_detect is not None and len(sample) > 20:
        try:
            return _lang_detect(sample) != 'en'
        except Exception:  # noqa: BLE001
            pass
    padded = f' {sample.lower()} '
    for markers in NON_EN_MARKERS.values():
        if sum(1 for mk in markers if mk in padded) >= 2:
            return True
    letters = [c for c in sample if c.isalpha()]
    if letters:
        non_ascii = sum(1 for c in letters if ord(c) > 127)
        if non_ascii / len(letters) > 0.15:
            return True
    return False


def _first_words(text: str, n: int) -> str:
    words = re.findall(r"[\w'|<>/]+", text)
    return ' '.join(words[:n]).lower()


def _iter_examples(*, raw: bool, think_fraction: float, skip_rows: int,
                   per_split_limit: int, hard_filter: bool, seq_len: int):
    """Balanced sample: up to ``per_split_limit`` kept rows from each split.

    Splits are concatenated in sorted order upstream, so a global prefix limit would
    only ever see the first split — iterate split-by-split for representative coverage.
    """
    splits = _smoltalk2_split_names('SFT') if raw else _smoltalk2_sft_splits()
    # Bail out of a split after this many raw rows even if we haven't hit the kept
    # target — otherwise a split that is fully filtered (e.g. all-<think> under
    # think_fraction=0) would stream end-to-end and hang.
    read_cap = max(500, per_split_limit * 40)
    for split in splits:
        kept = read = 0
        for _, msgs in _smoltalk2_row_iter(
            'SFT', skip_rows=skip_rows, splits=[split], emit_source=True,
        ):
            read += 1
            keep = True
            if not raw:
                if not _cap_keep(msgs, _smoltalk2_split_cap(split)):
                    keep = False
                elif not _think_keep(msgs, think_fraction):
                    keep = False
            if keep and hard_filter and _should_filter_smoltalk(
                msgs, split, 'hard', max_chars=seq_len * 6
            ):
                keep = False
            if keep:
                yield split, msgs
                kept += 1
            if per_split_limit and kept >= per_split_limit:
                break
            if read >= read_cap:
                break


def profile(*, raw: bool, think_fraction: float, limit: int, skip_rows: int,
            hard_filter: bool, seq_len: int) -> dict[str, Any]:
    splits = _smoltalk2_split_names('SFT') if raw else _smoltalk2_sft_splits()
    per_split_limit = max(1, limit // max(1, len(splits)))
    print(f'  sampling up to {per_split_limit} rows from each of {len(splits)} split(s)'
          f'{" + hard filter" if hard_filter else ""}')
    first_word: Counter[str] = Counter()
    first_three: Counter[str] = Counter()
    per_split: Counter[str] = Counter()
    turn_counts: Counter[int] = Counter()
    lengths: list[int] = []
    generic = think = non_en = empty = 0
    n = 0

    for split, msgs in _iter_examples(
        raw=raw, think_fraction=think_fraction, skip_rows=skip_rows,
        per_split_limit=per_split_limit, hard_filter=hard_filter, seq_len=seq_len,
    ):
        ans = _first_assistant(msgs)
        per_split[split] += 1
        turn_counts[len(msgs)] += 1
        if _messages_have_think(msgs):
            think += 1
        if not ans:
            empty += 1
        else:
            words = ans.split()
            lengths.append(len(words))
            first_word[_first_words(ans, 1)] += 1
            first_three[_first_words(ans, 3)] += 1
            low = ans.lower()
            if any(low.startswith(op) for op in GENERIC_OPENERS):
                generic += 1
            if _detect_non_english(ans):
                non_en += 1
        n += 1
        if n and n % 250 == 0:
            print(f'  ...{n} sampled')
        if limit and n >= limit:
            break

    scored = len(lengths)
    lengths_sorted = sorted(lengths)
    p90 = lengths_sorted[int(0.9 * (scored - 1))] if scored else 0
    return {
        'raw': raw,
        'think_fraction': think_fraction,
        'hard_filter': hard_filter,
        'sampled': n,
        'first_turn_scored': scored,
        'empty_first_assistant': empty,
        'think_share': round(think / n, 3) if n else 0.0,
        'non_english_share': round(non_en / scored, 3) if scored else 0.0,
        'generic_opener_share': round(generic / scored, 3) if scored else 0.0,
        'answer_words_mean': round(sum(lengths) / scored, 1) if scored else 0.0,
        'answer_words_median': median(lengths) if scored else 0,
        'answer_words_p90': p90,
        'top_first_word': first_word.most_common(20),
        'top_first_three': first_three.most_common(20),
        'turn_counts': dict(sorted(turn_counts.items())),
        'per_split': dict(per_split.most_common()),
    }


def _print_report(rep: dict[str, Any]) -> None:
    mode = 'RAW (unfiltered)' if rep['raw'] else f"FILTERED (think_fraction={rep['think_fraction']})"
    print(f"\n=== SFT target profile — {mode} ===")
    print(f"sampled={rep['sampled']}  first-turn scored={rep['first_turn_scored']}  "
          f"empty={rep['empty_first_assistant']}")
    print(f"think_share            : {rep['think_share']:.1%}")
    print(f"non_english_share      : {rep['non_english_share']:.1%}")
    print(f"generic_opener_share   : {rep['generic_opener_share']:.1%}")
    print(f"answer_words mean/med/p90: {rep['answer_words_mean']} / "
          f"{rep['answer_words_median']} / {rep['answer_words_p90']}")
    print('\ntop first word:')
    for word, cnt in rep['top_first_word']:
        print(f"  {cnt:5d}  {word!r}")
    print('\ntop first 3 words:')
    for phrase, cnt in rep['top_first_three'][:15]:
        print(f"  {cnt:5d}  {phrase!r}")
    print('\nturn counts (len(messages) -> n):')
    for turns, cnt in rep['turn_counts'].items():
        print(f"  {turns:3d} turns: {cnt}")
    print('\nper split kept:')
    for split, cnt in rep['per_split'].items():
        print(f"  {cnt:6d}  {split}")


def main() -> None:
    p = argparse.ArgumentParser(description='Profile smoltalk2 SFT assistant targets')
    p.add_argument('--limit', type=int, default=1500, help='Conversations to sample')
    p.add_argument('--think-fraction', type=float, default=0.15)
    p.add_argument('--skip-rows', type=int, default=0)
    p.add_argument('--raw', action='store_true',
                   help='Bypass split allowlist / caps / think cap (pre-fix distribution)')
    p.add_argument('--hard-filter', action='store_true',
                   help="Also apply the SFT 'hard' filter (length cap etc.) — matches training")
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--out-json', default='')
    args = p.parse_args()

    rep = profile(
        raw=args.raw,
        think_fraction=args.think_fraction,
        limit=args.limit,
        skip_rows=args.skip_rows,
        hard_filter=args.hard_filter,
        seq_len=args.seq_len,
    )
    _print_report(rep)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Ask SCI-1 about a long text, with and without the package (torch + transformers only).

    python run_package.py --context notes.txt --question "Which city did Marta move to?"            # both answers
    python run_package.py --context notes.txt --question "..." --mode package --strict
    python run_package.py --demo                       # a 32k-token needle: the model alone vs the package
    python run_package.py --demo --demo_tokens 8192 --device cpu

BARE = the model alone: the whole context streamed into its fixed state, then the question.
PACKAGE = the same state plus a text library of the context: the library fetches the one sentence the question is
about, and the model reads that sentence plus the question on its state (package_lite.py says how).
Both use greedy decoding and stop at the end of the first sentence. The memory held is printed: the model's state is
the same size at any length; the library grows with the context (2 bytes per token of text, plus its word index).

Question form: --form plain (default) wraps the question as "\\n\\nQuestion: ...\\nAnswer:"; --form chat as a ChatML
user turn (for SCI-1 Chat); --form raw uses your text exactly (e.g. a cloze "... The code for X is").
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modeling_pam2 import load_model, load_tokenizer  # noqa: E402
from package_lite import Package  # noqa: E402

FORMS = {'plain': '\n\nQuestion: {q}\nAnswer:', 'chat': '\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n', 'raw': '{q}'}
# RULER-style haystack (Hsieh et al. 2024's repeated noise) and a needle in the style of its niah tasks
NOISE = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. '
ADJ = ['lively', 'quiet', 'brisk', 'gentle', 'hollow', 'silver', 'rusty', 'calm', 'eager', 'faint']
NOUN = ['otter', 'orchid', 'harbor', 'lantern', 'meadow', 'pebble', 'falcon', 'thimble', 'compass', 'kettle']


def pick_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def demo_doc(tok, n_tokens: int, seed: int):
    """a haystack of about n_tokens with one needle at a random depth; returns (context, question, gold)."""
    rng = random.Random(seed)
    key, val = f'{rng.choice(ADJ)}-{rng.choice(NOUN)}', str(rng.randint(1_000_000, 9_999_999))
    per = len(tok.encode(NOISE))
    reps = max(2, n_tokens // per)
    at = rng.randrange(1, reps)
    needle = f'One of the special magic numbers for {key} is: {val}. '
    ctx = NOISE * at + needle + NOISE * (reps - at)
    q = (f'\n\nWhat is the special magic number for {key} mentioned in the provided text? '
         f'The special magic number for {key} mentioned in the provided text is')
    return ctx, q, val, at / reps


def mb(n: int) -> str:
    return f'{n / 1e6:.2f} MB'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=None, help='the release .pt (default: the one .pt file next to this script)')
    ap.add_argument('--context', help='a text file: the session (any length)')
    ap.add_argument('--question', help='the question (wrapped by --form)')
    ap.add_argument('--form', default='plain', choices=list(FORMS))
    ap.add_argument('--mode', default='both', choices=['bare', 'package', 'both'])
    ap.add_argument('--strict', action='store_true', help='package: abstain when the text organs miss (no F2)')
    ap.add_argument('--max_new', type=int, default=40)
    ap.add_argument('--demo', action='store_true', help='a needle in a haystack instead of --context / --question')
    ap.add_argument('--demo_tokens', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    if not a.demo and not (a.context and a.question):
        ap.error('give --context and --question, or --demo')

    ck = a.checkpoint
    if ck is None:
        pts = sorted(HERE.glob('*.pt'))
        if len(pts) != 1:
            ap.error(f'found {len(pts)} .pt files next to run_package.py; pass --checkpoint')
        ck = str(pts[0])
    dev = pick_device(a.device)
    tok = load_tokenizer()
    tok.model_max_length = 10 ** 9          # the state is O(1): no 1,024-token limit (silences the GPT-2 length warning)
    t0 = time.time()
    model = load_model(ck, device=dev)
    pkg = Package(model, tok, max_new=a.max_new)
    print(f'loaded {Path(ck).name} on {dev} in {time.time() - t0:.1f} s')

    if a.demo:
        ctx, q, gold, depth = demo_doc(tok, a.demo_tokens, a.seed)
        print(f'demo: a {a.demo_tokens}-token haystack, needle at depth {depth:.2f}, gold {gold}')
    else:
        ctx = Path(a.context).read_text()
        q, gold = FORMS[a.form].format(q=a.question), None

    t0 = time.time()
    s = pkg.read(ctx)
    n = len(s['ids'])
    print(f'read {n:,} tokens in {time.time() - t0:.1f} s')

    if a.mode in ('bare', 'both'):
        t0 = time.time()
        ans = pkg.bare(ctx, q)
        print(f'\nBARE    ({time.time() - t0:.1f} s): {ans.strip()!r}' + (f'   right: {gold in ans}' if gold else ''))
        print(f'  memory held: model state {mb(s["state_bytes"])} (the same at any length)')
    if a.mode in ('package', 'both'):
        t0 = time.time()
        ans, info = pkg.answer(ctx, q, mode='strict' if a.strict else 'lenient')
        print(f'\nPACKAGE ({time.time() - t0:.1f} s): {ans.strip()!r}' + (f'   right: {gold in ans}' if gold else ''))
        print(f'  fetched by {info["path"]}: {(info["row_text"] or "(abstained)").strip()[:160]!r}')
        b = info['bytes_held']
        print(f'  memory held: model state {mb(b["mind_state"])} + library text {mb(b["library_text"])} '
              f'+ word index ~{mb(b["word_token_indexes_est"])} (+ address {mb(b["address_int8"])} int8, used only when '
              f'the text organs miss)')


if __name__ == '__main__':
    main()

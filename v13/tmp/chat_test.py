#!/usr/bin/env python3
"""Chat-style generation probe for a V13 checkpoint.

Loads a trained ckpt, runs chat-like + fact-recall + math prompts through
model.generate() (O(1)/token recurrent decode), prints raw responses, and
scores math/fact correctness. Used to judge chat-style quality, repetition
(v6 Bug-8 failure mode), fact recall, and basic math, per the user's post-500M
request.

Usage: .venv/bin/python -m v13.tmp.chat_test <checkpoint.pt> [--max_new N] [--out out.json]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO = '/home/gowrav/Development/qllm2'
sys.path.insert(0, REPO)

from v13.eval_checkpoints import load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# (label, prompt, expected substring for fact/math scoring or None)
PROMPTS = [
    ("chat_explain", "Explain in one sentence why the sky is blue.", None),
    ("chat_bullets", "Tell me about the Roman Empire in three short bullet points.", None),
    ("fact_capital", "What is the capital of France? Answer in one line.", "paris"),
    ("fact_year", "In what year did World War II end? One line.", "1945"),
    ("fact_author", "Who wrote the play Hamlet? One line.", "shakespeare"),
    ("math_add", "What is 37 + 25? Give only the number.", "62"),
    ("math_mult", "What is 12 * 8? Give only the number.", "96"),
    ("math_chain", "I have 3 apples, buy 2 more, then give away 1. How many do I have? One line.", "4"),
    ("code_prime", "Write a Python function is_prime(n). Code only.", "def is_prime"),
]


def _first_number(text: str):
    m = re.search(r'-?\d+', text)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--max_new', type=int, default=80)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--rep_penalty', type=float, default=1.2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"loading {args.checkpoint} on {device}")
    model, cfg = load_model(args.checkpoint, device)
    model.eval()
    tok = AutoTokenizer.from_pretrained('gpt2')

    results = []
    n_correct = 0
    n_scored = 0
    for label, prompt, expected in PROMPTS:
        ids = tok(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.max_new,
                temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, repetition_penalty=args.rep_penalty,
            )
        gen_ids = out[0, ids.shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        entry = {'label': label, 'prompt': prompt, 'response': text}
        if expected is not None:
            n_scored += 1
            if label.startswith('math_'):
                ok = _first_number(text) == expected
            else:
                ok = expected.lower() in text.lower()
            entry['expected'] = expected
            entry['correct'] = bool(ok)
            n_correct += int(ok)
        results.append(entry)
        mark = ''
        if expected is not None:
            mark = '  CORRECT' if entry.get('correct') else f"  (expected ~{expected})"
        print(f"\n{'='*70}\n[{label}]{mark}")
        print(f"  P> {prompt}")
        print(f"  A> {text}")
    print(f"\n{'='*70}\nfact/math score: {n_correct}/{n_scored}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()

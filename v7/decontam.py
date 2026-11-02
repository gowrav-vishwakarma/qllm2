"""SCI-1 decontamination (Lead / Library, 2026-11-02): a word-level 13-gram filter of every training source against the test sets
we report or might report. Mandatory for the benchmark arena.

Normalisation: lowercase, keep [a-z0-9]+ words. Each word -> crc32; a 13-gram hash = sum(w_i * P^(12-i)) mod 2^64 (numpy uint64 wrap).
The index is a sorted unique uint64 array saved as .npy; its sha256 goes in the run header so an audit can rebuild it.
Item text: question + answer (or each choice) for short sets; facts + question only for long-context sets (never a haystack).
Build:   python -m v7.decontam build  [--out .cache/decontam/v1.npy]
Filter:  DecontamFilter(path).hit(text) -> bool  (a doc with ANY hit is dropped by the blend; counts logged per source)
"""
from __future__ import annotations

import hashlib, json, re, sys, zlib
from pathlib import Path

import numpy as np

N = 13
P = np.uint64(1_000_003)
_W = re.compile(r'[a-z0-9]+')
POW = np.array([pow(1_000_003, N - 1 - i, 2 ** 64) for i in range(N)], dtype=np.uint64)


def ngram_hashes(text: str) -> np.ndarray:
    words = _W.findall(text.lower())
    if len(words) < N:
        return np.zeros(0, dtype=np.uint64)
    wh = np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))
    win = np.lib.stride_tricks.sliding_window_view(wh, N)
    with np.errstate(over='ignore'):
        return (win * POW).sum(axis=1, dtype=np.uint64)


def _items():
    """Yield (set_name, text) for every protected test item."""
    from datasets import load_dataset

    def ld(*a, **k):
        try:
            return load_dataset(*a, **k)
        except Exception as e:  # noqa: BLE001 - report and continue; the header lists the sets actually covered
            print(f'  [decontam] could not load {a} {k}: {type(e).__name__}: {e}', flush=True)
            return []
    for sp in ('validation', 'test'):
        for d in ld('Rowan/hellaswag', split=sp):
            for e in d['endings']:
                yield 'hellaswag', f"{d['activity_label']}: {d['ctx']} {e}"
    for cfg in ('ARC-Easy', 'ARC-Challenge'):
        for sp in ('validation', 'test'):
            for d in ld('allenai/ai2_arc', cfg, split=sp):
                for c in d['choices']['text']:
                    yield 'arc', f"{d['question']} {c}"
    for sp in ('validation', 'test'):
        for d in ld('baber/piqa', split=sp):
            yield 'piqa', f"{d['goal']} {d['sol1']}"; yield 'piqa', f"{d['goal']} {d['sol2']}"
    for sp in ('validation', 'test'):
        for d in ld('allenai/winogrande', 'winogrande_xl', split=sp):
            for o in (d['option1'], d['option2']):
                yield 'winogrande', d['sentence'].replace('_', o)
    for sp in ('validation', 'test'):
        for d in ld('allenai/openbookqa', 'main', split=sp):
            for c in d['choices']['text']:
                yield 'openbookqa', f"{d['question_stem']} {c}"
    for d in ld('EleutherAI/lambada_openai', 'default', split='test'):
        yield 'lambada', d['text']
    for d in ld('openai/gsm8k', 'main', split='test'):
        yield 'gsm8k_test', f"{d['question']} {d['answer']}"
    for sp in ('test', 'validation'):
        for d in ld('cais/mmlu', 'all', split=sp):
            for c in d['choices']:
                yield 'mmlu', f"{d['question']} {c}"
    for d in ld('Muennighoff/babi', split='test'):          # BABILong's base: bAbI facts + question + answer, never a haystack
        yield 'babi', f"{d.get('passage', '')} {d.get('question', '')} {d.get('answer', '')}"
    root = Path(__file__).resolve().parents[1]
    lib = root / 'v14' / 'decontam_items.json'                # Library: A26 / bench / styles / v16 prompts / story texts
    if lib.exists():
        for s in json.load(open(lib)):
            yield 'library_items', s


def build(out: str = '.cache/decontam/v1.npy') -> dict:
    hs, counts, grams = [], {}, {}
    for name, text in _items():
        h = ngram_hashes(text)
        counts[name] = counts.get(name, 0) + 1
        grams[name] = grams.get(name, 0) + len(h)
        if len(h):
            hs.append(h)
    idx = np.unique(np.concatenate(hs)) if hs else np.zeros(0, dtype=np.uint64)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, idx)
    sha = hashlib.sha256(open(out, 'rb').read()).hexdigest()
    meta = {'path': out, 'sha256': sha, 'n_grams': int(idx.size), 'items': counts, 'grams_per_set': grams, 'n': N}
    json.dump(meta, open(out.replace('.npy', '.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1), flush=True)
    return meta


class DecontamFilter:
    def __init__(self, path: str):
        self.idx = np.load(path)
        self.sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()

    def hit(self, text: str) -> bool:
        h = ngram_hashes(text)
        if not h.size or not self.idx.size:
            return False
        pos = np.searchsorted(self.idx, h)
        pos[pos >= self.idx.size] = self.idx.size - 1
        return bool((self.idx[pos] == h).any())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        build(sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '--out' else '.cache/decontam/v1.npy')

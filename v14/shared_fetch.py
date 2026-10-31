"""SL1: shared-library fetch by TEXT (v14/EXPERIMENTS_V14.md "SL1", pre-registered 23:05 IST). Retrieval only, CPU.

The shared library holds separate texts, so it is fetched by text (EUREKA 2026-10-31: the mind's state is a
stream address, not an embedder). Methods, each returning a ranked list of entries for a question:
  organ4      exact match of the question's last 4 tokens (the current organ), latest match
  organ_long  the LONGEST n-gram (n = 8..2) of the question's tail found in the library, latest match
  bm25        BM25 over normalised words (lower-case; a leading "the" and a plural "s" stripped)
  sl1         organ_long when it matches with n >= 3, else bm25
  minilm      all-MiniLM-L6-v2 (local cache), mean-pooled, cosine (named baseline)
Pack: the pack_check builder (definitions inside held-out prose paragraphs + prose distractors), questions
" Query: <key> refers to" with the exact key or a paraphrase (Title / UPPER / the K / plural).

    .venv/bin/python -m v14.shared_fetch --defs 200 --distractors 200 --seed 20000 --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.spread import filler_tokens  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402
from v14.pack_check import HOLDOUT, natural_phrase, prose  # noqa: E402

STOP = {'query', 'refers', 'to', 'the', 'a', 'an', 'of', 'definition', 'is', 'and', 'in'}


def norm_words(text):
    out = []
    for w in re.findall(r"[A-Za-z0-9']+", text.lower()):
        if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.post = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for w in tf:
                self.post[w].append(i)

    def rank(self, q, k=5):
        sc = defaultdict(float)
        for w in set(q):
            if w not in self.idf:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]


class Organ:
    """n-gram -> latest entry, for n = 2..8, over GPT-2 tokens of the library."""

    def __init__(self, entry_ids, nmax=8):
        self.nmax = nmax
        self.maps = {n: {} for n in range(2, nmax + 1)}
        for e, ids in enumerate(entry_ids):
            for n in range(2, nmax + 1):
                m = self.maps[n]
                for i in range(len(ids) - n + 1):
                    m[tuple(ids[i:i + n])] = e
        self.bytes = sum(len(m) for m in self.maps.values()) * 12      # hash (8) + entry (4)

    def exact(self, q, n):
        return self.maps[n].get(tuple(q[-n:])) if len(q) >= n else None

    def longest(self, q):
        for n in range(self.nmax, 1, -1):
            e = self.exact(q, n)
            if e is not None:
                return e, n
        return None, 0


def minilm_embed(texts, batch=64):
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = 'sentence-transformers/all-MiniLM-L6-v2'
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    m = AutoModel.from_pretrained(name, local_files_only=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            x = tok(texts[i:i + batch], padding=True, truncation=True, max_length=256, return_tensors='pt')
            h = m(**x).last_hidden_state
            mask = x['attention_mask'].unsqueeze(-1).float()
            e = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--defs', type=int, default=200)
    ap.add_argument('--distractors', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    rng = random.Random(a.seed)
    filler = filler_tokens(HOLDOUT)
    entries, facts, seen = [], [], set()
    for j in range(a.defs):
        kind = 'synthetic' if j % 2 == 0 else 'natural'
        while True:
            k = _recall_key(rng) if kind == 'synthetic' else natural_phrase(rng, tok, filler)
            if k not in seen:
                seen.add(k)
                break
        v = rng.choice(_RECALL_VALUE_NOUNS)
        text = prose(rng, tok, filler, 100, 0.0, 0.5) + f" Definition: {k} refers to {v}. " + prose(rng, tok, filler, 100, 0.0, 0.5)
        facts.append(dict(entry=len(entries), key=k, val=v, kind=kind))
        entries.append(text)
    for j in range(a.distractors):
        entries.append(prose(rng, tok, filler, 200, 0.0, 0.5))
    order = list(range(len(entries)))
    rng.shuffle(order)
    remap = {old: new for new, old in enumerate(order)}
    entries = [entries[i] for i in order]
    for f in facts:
        f['entry'] = remap[f['entry']]
    t0 = time.time()
    entry_ids = [tok.encode(' ' + e, add_special_tokens=False) for e in entries]
    organ = Organ(entry_ids)
    bm = BM25([norm_words(e) for e in entries])
    t_index = time.time() - t0
    t0 = time.time()
    E = minilm_embed(entries)
    t_embed = time.time() - t0
    jobs = []
    for f in facts:
        jobs.append((f, 'exact', f['key']))
        w = f['key'].split()
        forms = [f['key'].title(), f['key'].upper(), 'the ' + f['key']]
        if len(w) == 2:
            forms.append(w[0] + ' ' + (w[1][:-1] if w[1].endswith('s') else w[1] + 's'))
        jobs.append((f, 'para', rng.choice([x for x in forms if x != f['key']])))
    methods = ('organ4', 'organ_long', 'bm25', 'sl1', 'minilm')
    res = defaultdict(lambda: {m: [0, 0] for m in methods})
    n_cell = Counter()
    qtexts = [f" Query: {qk} refers to" for _, _, qk in jobs]
    Q = minilm_embed(qtexts)
    for (f, form, qk), qt, qe in zip(jobs, qtexts, Q):
        cell = f"{f['kind']}_{form}"
        n_cell[cell] += 1
        qids = tok.encode(qt, add_special_tokens=False)
        e4 = organ.exact(qids, 4)
        el, nl = organ.longest(qids)
        br = bm.rank([w for w in norm_words(qt) if w not in STOP])
        sl = [el] if (el is not None and nl >= 3) else br
        mr = list(np.argsort(-(E @ qe))[:5])
        ranked = {'organ4': [e4] if e4 is not None else [], 'organ_long': [el] if el is not None else [],
                  'bm25': br, 'sl1': sl, 'minilm': mr}
        for m in methods:
            r = ranked[m]
            res[cell][m][0] += int(bool(r) and r[0] == f['entry'])
            res[cell][m][1] += int(f['entry'] in r[:5])
    out = {cell: {m: dict(top1=round(v[0] / n_cell[cell], 3), top5=round(v[1] / n_cell[cell], 3)) for m, v in d.items()}
           for cell, d in res.items()}
    n_tok = sum(len(x) for x in entry_ids)
    report = dict(seed=a.seed, entries=len(entries), tokens=n_tok, results=out, n=dict(n_cell),
                  index_mb_per_1M_tokens=dict(organ_2to8=round(organ.bytes / n_tok, 2),
                                              minilm=round(E.size * 4 / n_tok, 2)),
                  index_seconds=round(t_index, 1), embed_seconds=round(t_embed, 1))
    json.dump(report, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    print({k: report[k] for k in ('entries', 'tokens', 'index_mb_per_1M_tokens', 'index_seconds', 'embed_seconds')})


if __name__ == '__main__':
    main()

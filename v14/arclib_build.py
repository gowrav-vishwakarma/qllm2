"""ARC-LIB / BENCH-LIB library build (CPU): a decontaminated passage library + a BM25 index, IDENTICAL for every body.

Sources (one library for all four benchmarks):
  arc       the AI2 ARC Corpus (14.6M science sentences, ARC-V1-Feb2018), one passage per sentence.
  cosmo     Mind's cosmo stream (v7.data._open_source_iter, holdout excluded), docs cut into passages of whole sentences
  fineweb   up to 60 words: everyday / physical text for HellaSwag and PIQA.
Kept: 6-80 words, deduplicated (lower-case text). DROPPED when EITHER 13-gram filter hits: Mind's index (v7/decontam.py,
.cache/decontam/v1.npy: HellaSwag / ARC / PIQA val + test and the rest) or the A27 audit's independent index (own hash).
Positive control first: 300 planted test strings must be flagged by both, else the build stops.
Writes <out>/passages.txt (one per line), <out>/source.npy (source id per passage), <out>/bm25.npz (term-major CSC BM25
weights, HashingVectorizer 2**22, english stop words, k1 1.2, b 0.75) and <out>/meta.json (counts, drops per source / set).
    python v14/arclib_build.py --out /media/gowrav/d_drive/qllm2/arc_lib/lib_v1 --everyday cosmo:100000,fineweb:100000
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')
WORDS = re.compile(r"[A-Za-z0-9']+")
G = {}


def passages(doc: str, max_words: int = 60):
    cur, n = [], 0
    for s in SENT.split(' '.join(doc.split())):
        w = len(WORDS.findall(s))
        if cur and n + w > max_words:
            yield ' '.join(cur); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        yield ' '.join(cur)


def _check(chunk):
    """per passage: (mind hit, audit hit, audit set id or -1)."""
    from v7.decontam import ngram_hashes
    from v14.decontam_audit import grams, lookup
    mind, aud = G['mind'], G['aud']; out = []
    for t in chunk:
        h = ngram_hashes(t); m = False
        if h.size:
            pos = np.minimum(np.searchsorted(mind, h), mind.size - 1); m = bool((mind[pos] == h).any())
        g, _ = grams(t); hit, sets, _ = lookup(aud, g)
        out.append((m, bool(hit.size), int(sets[0]) if hit.size else -1))
    return out


def check_all(texts, procs):
    chunks = [texts[i:i + 20000] for i in range(0, len(texts), 20000)]
    with mp.get_context('fork').Pool(procs) as pool:
        return [r for c in pool.imap(_check, chunks) for r in c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arc_corpus', default='/media/gowrav/d_drive/qllm2/arc_lib/ARC_Corpus.txt')
    ap.add_argument('--everyday', default='cosmo:100000,fineweb:100000'); ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--arc_limit', type=int, default=None, help='smoke: first N corpus lines'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    import subprocess
    meta = dict(args=vars(a), hash=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                                  cwd=ROOT).stdout.strip(), sources={})
    from v14.decontam_audit import build_index
    G['mind'] = np.load(ROOT / '.cache/decontam/v1.npy'); G['aud'] = build_index()
    inv = {v: k for k, v in G['aud']['sets'].items()}
    meta['decontam'] = dict(mind_index=str(ROOT / '.cache/decontam/v1.npy'), mind_grams=int(G['mind'].size),
                            audit_grams=int(G['aud']['H'].size), audit_sets=dict(G['aud']['n_items']))
    # positive control: planted long test strings must be flagged by both filters
    import random
    rng = random.Random(0)
    long_items = [t for t in G['aud']['strings'] if len(WORDS.findall(t)) >= 20]
    ctrl = ['The river ran past the old mill. ' + t for t in rng.sample(long_items, 300)]
    r = check_all(ctrl, a.procs)
    meta['control'] = dict(planted=300, mind=sum(x[0] for x in r), audit=sum(x[1] for x in r))
    print('[lib] positive control', meta['control'], flush=True)
    assert meta['control']['mind'] >= 295 and meta['control']['audit'] >= 295, 'a decontam filter missed planted strings'
    texts, src, seen = [], [], set()

    def add(name, sid, it):
        n_in = n_len = n_dup = 0; keep = []
        for t in it:
            n_in += 1; w = len(WORDS.findall(t))
            if w < 6 or w > 80:
                n_len += 1; continue
            k = hash(t.lower())
            if k in seen:
                n_dup += 1; continue
            seen.add(k); keep.append(t)
        r = check_all(keep, a.procs)
        drop_m = sum(x[0] for x in r); drop_a = sum(x[1] for x in r)
        by_set = {}
        for x in r:
            if x[2] >= 0:
                by_set[inv[x[2]]] = by_set.get(inv[x[2]], 0) + 1
        kept = [t for t, x in zip(keep, r) if not (x[0] or x[1])]
        texts.extend(kept); src.extend([sid] * len(kept))
        meta['sources'][name] = dict(read=n_in, length_drop=n_len, dup_drop=n_dup, decontam_drop_mind=drop_m,
                                     decontam_drop_audit=drop_a, decontam_drop_by_set_audit=by_set, kept=len(kept))
        print(f'[lib] {name}', meta['sources'][name], f'({time.time() - t0:.0f} s)', flush=True)

    def arc_lines():
        with open(a.arc_corpus, encoding='utf-8', errors='ignore') as f:
            for i, l in enumerate(f):
                if a.arc_limit and i >= a.arc_limit:
                    break
                yield ' '.join(l.split())
    add('arc', 0, arc_lines())
    import v7.data as D
    for sid, spec in enumerate(a.everyday.split(','), start=1):
        name, n = spec.split(':'); n = int(n); cnt = {}
        it = D._open_source_iter(name, D.SOURCE_REGISTRY[name], edu_score_min=3, fineweb_name='sample-10BT', exclude_holdout=True,
                                 holdout_pct=5, skip=0, doc_counters=cnt, mix_seed=43)

        def docs(it=it, n=n):
            for i, d in enumerate(it):
                if i >= n:
                    break
                yield from passages(d)
        add(name, sid, docs())
    with open(out / 'passages.txt', 'w') as f:
        for t in texts:
            f.write(t.replace('\n', ' ') + '\n')
    np.save(out / 'source.npy', np.asarray(src, dtype=np.int8))
    meta['n_passages'] = len(texts); meta['source_ids'] = ['arc'] + [s.split(':')[0] for s in a.everyday.split(',')]
    print(f'[lib] {len(texts):,} passages; BM25 ... ({time.time() - t0:.0f} s)', flush=True)
    from v14.arclib_index import build_bm25
    meta['bm25'] = build_bm25(texts, out / 'bm25.npz')
    meta['sec'] = round(time.time() - t0)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print('[lib] DONE', json.dumps({k: meta[k] for k in ('n_passages', 'control', 'sec')}), flush=True)


if __name__ == '__main__':
    main()

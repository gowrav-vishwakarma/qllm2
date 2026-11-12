"""A27: independent 13-gram decontamination audit of the SCI-1 training sources (pre-registered in EXPERIMENTS_V14 "A27").

Independent of Mind's filter (v7/decontam.py): own word split, own hash (crc32 words + a different polynomial), own
set-labelled index. Each source is sampled twice through the SAME iterator the blend uses (v7.data._open_source_iter, SCI-1's
exact edu score / holdout exclusion / seed):
    RAW      = before Mind's filter: what the filter has to remove (contamination of the data itself);
    FILTERED = after Mind's filter (v7.data._decontam_wrap with DECONTAM_INDEX set): must be 0, else the filter has a hole.
Report: per source x test set, % of docs with >= 1 shared 13-gram, examples, the most frequent hits (boilerplate), and the
test strings hit at all (a clean subset of a benchmark = its items not in that list).

    DECONTAM_INDEX=.cache/decontam/v1.npy python v14/decontam_audit.py --docs 20000 --out logs/v14_a27_decontam_<hash>_<stamp>.json
Sample = the first N docs of each source's stream (the prefix SCI-1 reads first); CPU only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

N = 13
WORD = re.compile(r"[a-z0-9]+")
P2 = 998_244_353
POW = np.array([pow(P2, N - 1 - i, 2 ** 64) for i in range(N)], dtype=np.uint64)       # not Mind's polynomial
SALT = np.uint64(0x9E3779B97F4A7C15)
SOURCES_SCI1 = ('dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,think,pysci,b2s,recall4k,proto4k')


def grams(text: str):
    """(uint64 hashes of every 13-gram, the word list)."""
    words = WORD.findall(text.lower())
    if len(words) < N:
        return np.zeros(0, dtype=np.uint64), words
    wh = np.fromiter(((zlib.crc32(w.encode()) ^ 0xA5A5A5A5) for w in words), dtype=np.uint64, count=len(words))
    win = np.lib.stride_tricks.sliding_window_view(wh, N)
    with np.errstate(over='ignore'):
        return (win * POW).sum(axis=1, dtype=np.uint64) ^ SALT, words


def test_items():
    """(set name, text) for every protected item: Mind's item list (same definitions) + BABILong qa1-10 facts/question/target
    + IFEval prompts (chat ramp, 2026-11-12)."""
    from v7.decontam import _items
    yield from _items()
    try:
        from datasets import load_dataset
        ds = load_dataset('RMT-team/babilong', '0k')
        for qa in sorted(ds.keys()):
            for r in ds[qa]:
                yield 'babilong', f"{r['input']} {r['question']} {r['target']}"
    except Exception as e:                                                  # noqa: BLE001
        print(f'  [audit] babilong not loaded: {type(e).__name__}: {e}', flush=True)
    try:                                                                    # chat ramp (R5, 2026-11-12): IFEval prompts
        from datasets import load_dataset
        for r in load_dataset('google/IFEval', split='train'):
            yield 'ifeval', r['prompt']
    except Exception as e:                                                  # noqa: BLE001
        print(f'  [audit] ifeval not loaded: {type(e).__name__}: {e}', flush=True)


def build_index():
    sets, names, strings, H, S, I = {}, [], [], [], [], []
    for name, text in test_items():
        sid = sets.setdefault(name, len(sets)); names.append(name)
        i = len(strings); strings.append(text)
        g, _ = grams(text)
        if g.size:
            H.append(g); S.append(np.full(g.size, sid, dtype=np.int16)); I.append(np.full(g.size, i, dtype=np.int64))
    H, S, I = np.concatenate(H), np.concatenate(S), np.concatenate(I)
    o = np.argsort(H, kind='stable')
    return dict(H=H[o], S=S[o], I=I[o], sets=sets, strings=strings, n_items=collections.Counter(names))


def lookup(idx, g):
    """positions in g that hit, with the (set id, string id) of each hit."""
    if g.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.int64)
    pos = np.searchsorted(idx['H'], g)
    pos[pos >= idx['H'].size] = idx['H'].size - 1
    hit = np.nonzero(idx['H'][pos] == g)[0]
    return hit, idx['S'][pos[hit]], idx['I'][pos[hit]]


def hf_iter(name):
    """hf:<repo>:<config or ->:<split>: stream rows; a chat row (messages / conversations) becomes its joined turn texts,
    else the first string field among text / prompt / content / instruction + response / output."""
    from datasets import load_dataset
    _, repo, cfg, split = name.split(':', 3)
    ds = load_dataset(repo, None if cfg in ('', '-') else cfg, split=split, streaming=True)
    for r in ds:
        msgs = r.get('messages') or r.get('conversations')
        if isinstance(msgs, list):
            yield '\n'.join(str(m.get('content') or m.get('value') or '') for m in msgs if isinstance(m, dict))
            continue
        parts = [r[k] for k in ('text', 'prompt', 'instruction', 'input', 'content', 'response', 'output', 'completion')
                 if isinstance(r.get(k), str) and r[k]]
        if parts:
            yield '\n'.join(parts)


def audit_source(name, it, n_docs, idx, tag):
    inv = {v: k for k, v in idx['sets'].items()}
    st = dict(source=name, view=tag, docs=0, words=0, docs_hit=collections.Counter(), docs_hit_any=0,
              examples=collections.defaultdict(list), gram_freq=collections.Counter(), strings_hit=set(), toks=[])
    from v7.data import get_chat_tokenizer
    tok = get_chat_tokenizer()
    t0 = time.time()
    for text in it:
        g, words = grams(text)
        st['docs'] += 1; st['words'] += len(words)
        if len(st['toks']) < 3000:                                         # doc length in tokens (median / p90 for ramp sizing)
            st['toks'].append(len(tok.encode(text, add_special_tokens=False)))
        h, sids, strs = lookup(idx, g)
        if h.size:
            st['docs_hit_any'] += 1
            for sid in set(sids.tolist()):
                st['docs_hit'][inv[sid]] += 1
                if len(st['examples'][inv[sid]]) < 3:
                    j = int(h[np.nonzero(sids == sid)[0][0]])
                    st['examples'][inv[sid]].append(dict(doc=st['docs'] - 1, ngram=' '.join(words[j:j + N]), test=idx['strings'][int(strs[np.nonzero(sids == sid)[0][0]])][:120]))
            for hh in g[h].tolist():
                st['gram_freq'][hh] += 1
            st['strings_hit'].update(strs.tolist())
        if st['docs'] >= n_docs:
            break
        if st['docs'] % 2000 == 0:
            print(f'  [{name}/{tag}] {st["docs"]:,} docs, {st["docs_hit_any"]} hit, {time.time() - t0:.0f}s', flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default=SOURCES_SCI1); ap.add_argument('--docs', type=int, default=20000)
    ap.add_argument('--out', required=True); ap.add_argument('--views', default='raw,filtered')
    ap.add_argument('--mix_seed', type=int, default=43)
    ap.add_argument('--fineweb_name', default='sample-10BT', help='fineweb-edu config (sample-100BT replaces the dry local slice)')
    a = ap.parse_args()
    import v7.data as D
    t0 = time.time()
    idx = build_index()
    print(f'[audit] index: {idx["H"].size:,} 13-grams from {len(idx["strings"]):,} test strings, sets {dict(idx["n_items"])} '
          f'({time.time() - t0:.0f}s)', flush=True)
    res = dict(args=vars(a), n_items=dict(idx['n_items']), n_grams=int(idx['H'].size), decontam_index=os.environ.get('DECONTAM_INDEX'),
               sources={}, hit_strings={})
    inv = {v: k for k, v in idx['sets'].items()}
    hit_all = collections.defaultdict(set)
    # POSITIVE CONTROL: 300 docs = filler prose with one planted test string (>= 13 words) each, from every set. The audit must flag
    # all of them, and so must Mind's filter (a hole in either shows here, not as a silent 0.0 %).
    import random
    rng = random.Random(0)
    long_items = [(i, t) for i, t in enumerate(idx['strings']) if len(WORD.findall(t.lower())) >= N]
    planted = rng.sample(long_items, min(300, len(long_items)))
    filler = 'The river ran past the old mill and the town grew around it over many years. '
    ctrl_docs = [filler * 5 + t + ' ' + filler * 5 for _, t in planted]
    st = audit_source('control', iter(ctrl_docs), len(ctrl_docs), idx, 'planted')
    mine = st['docs_hit_any']
    theirs = None
    if os.environ.get('DECONTAM_INDEX'):
        from v7.decontam import DecontamFilter
        f = DecontamFilter(os.environ['DECONTAM_INDEX'])
        theirs = sum(bool(f.hit(d)) for d in ctrl_docs)
    res['control'] = dict(planted=len(ctrl_docs), flagged_by_audit=mine, flagged_by_mind_filter=theirs,
                          by_set=dict(collections.Counter(next(k for k, v in idx['sets'].items() if v == int(idx['S'][np.nonzero(idx['I'] == i)[0][0]])) for i, _ in planted)))
    print(f'[audit] POSITIVE CONTROL: {mine}/{len(ctrl_docs)} planted docs flagged by the audit, {theirs} by Mind\'s filter', flush=True)
    for name in a.sources.split(','):
        for view in a.views.split(','):
            cnt = {}
            if name.startswith('hf:'):                                     # a candidate not yet in the registry: hf:repo:config:split
                it = hf_iter(name)
            else:
                spec = D.SOURCE_REGISTRY[name]
                it = D._open_source_iter(name, spec, edu_score_min=3, fineweb_name=a.fineweb_name, exclude_holdout=True,
                                         holdout_pct=5, skip=0, doc_counters=cnt, mix_seed=a.mix_seed)
            if view == 'filtered':
                if not os.environ.get('DECONTAM_INDEX'):
                    continue
                it = D._decontam_wrap(name, it, cnt)
            st = audit_source(name, it, a.docs, idx, view)
            top = st['gram_freq'].most_common(5)
            res['sources'].setdefault(name, {})[view] = dict(
                docs=st['docs'], words=st['words'], docs_hit_any=st['docs_hit_any'],
                tokens_median=int(np.median(st['toks'])) if st['toks'] else None,
                tokens_p90=int(np.percentile(st['toks'], 90)) if st['toks'] else None,
                pct_any=round(100 * st['docs_hit_any'] / max(st['docs'], 1), 4),
                pct_by_set={k: round(100 * v / max(st['docs'], 1), 4) for k, v in sorted(st['docs_hit'].items())},
                examples={k: v for k, v in st['examples'].items()}, top_gram_hits=top, dropped_by_filter=cnt.get(f'{name}#decon', 0))
            for i in st['strings_hit']:
                hit_all[view].add(i)
            print(f'[audit] {name} {view}: {st["docs"]:,} docs, {st["docs_hit_any"]} with a hit '
                  f'({res["sources"][name][view]["pct_any"]}%) {dict(st["docs_hit"])}', flush=True)
            json.dump(res, open(a.out, 'w'), indent=1, default=str)
    for view, ids in hit_all.items():
        by = collections.defaultdict(list)
        for i in sorted(ids):
            by[inv[int(idx['S'][np.nonzero(idx['I'] == i)[0][0]])]].append(idx['strings'][i])
        res['hit_strings'][view] = {k: dict(n_hit=len(v), of=res['n_items'].get(k), strings=v[:200]) for k, v in by.items()}
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
    print('A27 DONE', flush=True)


if __name__ == '__main__':
    main()

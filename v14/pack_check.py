"""Cross-document check of the pack install tool (v14/pack.py): the library and the session are DIFFERENT texts.

A pack of definition entries (each "Definition: K refers to V." inside a paragraph of held-out prose; half
synthetic keys, half real two-word phrases) plus prose-only distractor entries is installed with the mind.
Each session is other held-out prose followed by " Query: K refers to". Reported per key type:
  key   : top-1 / top-5 entry by the learned key (session key vs library keys), and the answer after the
          mind re-reads the top-1 entry's text in front of the query (20 value candidates, chance 0.05)
  organ : exact 4-gram match of the query's last 4 tokens -> entry, and its answer (the token after it)
plus the install meta (rows per 16 tokens, MB and seconds per 1M tokens).

    python -m v14.pack_check --defs 200 --distractors 200 --session 1024 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from memory_probes.spread import filler_tokens  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402
from v14 import pack as P  # noqa: E402

HOLDOUT = str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')


def natural_phrase(rng, tok, filler):
    for _ in range(1000):
        i = rng.randrange(0, len(filler) - 2)
        txt = tok.decode(filler[i:i + 2].tolist())
        w = txt.strip().split()
        if len(w) == 2 and all(x.isalpha() and len(x) >= 4 for x in w) and txt.startswith(' '):
            return txt.strip()
    return _recall_key(rng)


def prose(rng, tok, filler, n, lo, hi):
    s = rng.randrange(int(lo * len(filler)), int(hi * len(filler)) - n)
    return tok.decode(filler[s:s + n].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--defs', type=int, default=200)
    ap.add_argument('--distractors', type=int, default=200)
    ap.add_argument('--session', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=18000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    mind, head = P.Mind(), P.KeyHead()
    tok = mind.tok
    filler = filler_tokens(HOLDOUT)
    # library text from the first half of the held-out prose, sessions from the second half
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
        entries.append(dict(id=f'def{j}', text=text, source='pack_check', time='2026-10-31', trust='test'))
    for j in range(a.distractors):
        entries.append(dict(id=f'prose{j}', text=prose(rng, tok, filler, 200, 0.0, 0.5), source='pack_check',
                            time='2026-10-31', trust='test'))
    order = list(range(len(entries)))
    rng.shuffle(order)
    remap = {old: new for new, old in enumerate(order)}
    entries = [entries[i] for i in order]
    for f in facts:
        f['entry'] = remap[f['entry']]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.write_pack(out / 'pack.jsonl', dict(pack_id='pack_check', title='cross-document check', license='test',
                                          created='2026-10-31'), entries)
    meta = P.install(out / 'pack.jsonl', out / 'library', mind=mind, head=head)
    lib = P.Library(out / 'library')
    cand = [tok.encode(' ' + v, add_special_tokens=False)[0] for v in _RECALL_VALUE_NOUNS]
    res = {k: dict(n=0, key1=0, key5=0, key_ans=0, organ_hit=0, organ_ans=0) for k in ('synthetic', 'natural')}
    for f in facts:
        r = res[f['kind']]
        r['n'] += 1
        sess = tok.encode(prose(rng, tok, filler, a.session, 0.5, 1.0), add_special_tokens=False)
        q = tok.encode(f" Query: {f['key']} refers to", add_special_tokens=False)
        kid = tok.encode(f" {f['key']}", add_special_tokens=False)
        ids = sess + q
        qpos = len(sess) + next(i for i in range(len(q)) if q[i:i + len(kid)] == kid) + len(kid) - 1
        top = lib.by_key(P.query_key(mind, head, ids, qpos), 5)
        r['key1'] += int(top[0][0] == f['entry'])
        r['key5'] += int(any(t[0] == f['entry'] for t in top))
        e0, e1 = lib.offs[top[0][0]].tolist()
        reread = lib.T[e0:e1].tolist() + q
        state, _ = mind.read(reread)
        with torch.no_grad():
            m = mind.m
            x = torch.tensor([reread], device=mind.dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = m(x)[0][0, -1].float()
        r['key_ans'] += int(cand[int(lg[cand].argmax())] == cand[_RECALL_VALUE_NOUNS.index(f['val'])])
        org = lib.by_organ(ids)
        if org is not None:
            r['organ_hit'] += int(org[0] == f['entry'])
            r['organ_ans'] += int(int(lib.T[org[1]]) == cand[_RECALL_VALUE_NOUNS.index(f['val'])])
    for k, r in res.items():
        n = max(1, r['n'])
        res[k] = {kk: (round(vv / n, 3) if kk != 'n' else vv) for kk, vv in r.items()}
    report = dict(library=meta, results=res, session_tokens=a.session, seed=a.seed)
    json.dump(report, open(out / 'report.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

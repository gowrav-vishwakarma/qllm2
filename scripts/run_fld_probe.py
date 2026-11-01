"""cu2 probe (mind side, free generation only; MLB-1 A7): one field from a multi-field row + stop, and verify.

Everything here is held out of the cu2 sources (v7/data.py fld / vfy): values from the crc32 held-out
split, the question formats _FLD_QS_PROBE, natural keys = two-word phrases from held-out prose (as
`v16/bench.py --keys natural`), synthetic keys = the bench's 2-syllable `_recall_key`; the row format is
either a trained one or the held-out _FLD_ROW_PROBE.

Cells (one JSON line per item):
  fld      rows (1 or 3, the target among them) + a probe question about one field -> " <value>."
  missing  the asked field is not in the key's row -> " unknown."
  vfy      ONE row right before the question: the key's own row (-> value), another key's row or a
           prose sentence (-> " unknown.")
  fld_doc  the row read once at t0, L tokens of held-out filler, then the question (-> " <value>.")
Each cell runs with the held-out question formats ('qset': 'out', transfer) and the trained ones ('in').
Scoring: greedy decode (whole vocabulary) up to 16 tokens; the text up to the first "." or newline.
  exact  = it equals the expected answer and the "." follows (the stop);
  runon  = it starts with the expected value but runs on without the stop;
  wrong  = anything else.
  contains (lenient, reported) = the expected value appears in that text (e.g. "room: south wing 70.").
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'v16'))
import v7.data as D  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from run_skillpack_probe import Body, filler_block  # noqa: E402
from memory_probes.spread import filler_tokens, _SEP  # noqa: E402
from bench import natural_key  # noqa: E402

PROSE = ('The museum is closed on Mondays during the winter.', 'Most of the rain fell before noon.',
         'She took the early train to the coast.', 'The committee will meet again next month.',
         'Two of the windows were left open overnight.', 'The recipe calls for a pinch of salt.',
         'He spent the afternoon repairing the fence.', 'The river rose after the storm.')


def decode(body, tok, ids, cap=16):
    out = []
    for _ in range(cap):
        t = int(body.last_logits(ids + out).argmax()); out.append(t)
        txt = tok.decode(out)
        if '.' in txt or '\n' in txt:
            break
    return tok.decode(out)


def judge(gen, want):
    """want is ' <value>.' or ' unknown.'"""
    head = gen.split('\n')[0]
    val = want[:-1]
    if head.startswith(val + '.') or head == want:
        return 'exact'
    if head.startswith(val):
        return 'runon'
    return 'wrong'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--docs', type=int, default=100)
    p.add_argument('--seed', type=int, default=31000)
    p.add_argument('--doc_L', type=int, default=2048)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    model, tok, _ = _load_v13_sempty(a.checkpoint, 'baseline_real_pm', dev); model.eval(); body = Body(model, dev)
    filler = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    sep = enc(_SEP)
    out = open(a.out, 'a')

    def key(rng, nat):
        return natural_key(rng, tok, filler, 0, 0) if nat else D._recall_key(rng)

    qset_now = ['out']

    def write(cell, rng_tag, nat, extra, ids, want, t0):
        gen = decode(body, tok, ids)
        out.write(json.dumps({'arch': body.arch, 'ckpt': str(a.checkpoint), 'cell': cell, 'keys': 'natural' if nat else 'synthetic', 'qset': qset_now[0],
                              **extra, 'want': want, 'gen': gen, 'judge': judge(gen, want),
                              'contains': int(want[1:-1] in gen.split('\n')[0]), 'sec': round(time.time() - t0, 2)}) + '\n')

    for d in range(a.docs):
        for nat, qset in ((True, 'out'), (False, 'out'), (True, 'in'), (False, 'in')):
            # qset 'out' = the held-out question formats (transfer); 'in' = the trained ones (skill, still held-out
            # values and keys). The 'out' streams keep their earlier seeds, so those rows are unchanged.
            rng = random.Random(a.seed * 1000 + d * 2 + nat + (0 if qset == 'out' else 7_000_000))
            q = rng.choice(D._FLD_QS_PROBE if qset == 'out' else D._FLD_QS); qset_now[0] = qset
            for heldfmt in (False, True):
                for nrows in (1, 3):
                    t0 = time.time()
                    keys = []
                    while len(keys) < nrows:
                        k = key(rng, nat)
                        if k not in keys:
                            keys.append(k)
                    fmt = D._FLD_ROW_PROBE if heldfmt else rng.choice(D._FLD_ROWS)
                    rows = [D._fld_row(rng, k, rng.randint(2, 4), held=True, fmt=fmt) for k in keys]
                    i = rng.randrange(nrows); f, v = rng.choice(rows[i][1])
                    ids = enc(''.join(r for r, _ in rows) + q.format(f=f, k=keys[i]))
                    write('fld', None, nat, {'doc': d, 'heldfmt': heldfmt, 'nrows': nrows, 'field': f}, ids, f' {v}.', t0)
            # missing field (trained row format, 1 row)
            t0 = time.time(); k = key(rng, nat); row, fv = D._fld_row(rng, k, 2, held=True)
            f = rng.choice([x for x in D._FLD_FIELDS if x not in dict(fv)])
            write('missing', None, nat, {'doc': d}, enc(row + q.format(f=f, k=k)), ' unknown.', t0)
            # verify: one row right before the question
            k = key(rng, nat)
            for kind in ('right', 'wrong_key', 'prose'):
                t0 = time.time()
                if kind == 'right':
                    row, fv = D._fld_row(rng, k, rng.randint(1, 3), held=True); f, v = rng.choice(fv); want = f' {v}.'
                elif kind == 'wrong_key':
                    other = key(rng, nat)
                    while other == k:
                        other = key(rng, nat)
                    row, _ = D._fld_row(rng, other, rng.randint(1, 3), held=True); f = rng.choice(D._FLD_FIELDS); want = ' unknown.'
                else:
                    row = ' ' + rng.choice(PROSE); f = rng.choice(D._FLD_FIELDS); want = ' unknown.'
                write('vfy', None, nat, {'doc': d, 'kind': kind}, enc(row + q.format(f=f, k=k)), want, t0)
            # fld_doc: one read at t0, L filler, question (natural keys only)
            if nat and a.doc_L and qset == 'out':
                t0 = time.time(); k = key(rng, nat); row, fv = D._fld_row(rng, k, 3, held=True); f, v = rng.choice(fv)
                ids = enc(row) + sep + filler_block(np.asarray(filler), a.doc_L, a.seed * 1000 + d, a.doc_L) + sep + enc(q.format(f=f, k=k).lstrip())
                write('fld_doc', None, nat, {'doc': d, 'L': a.doc_L}, ids, f' {v}.', t0)
        out.flush()
        if (d + 1) % 25 == 0:
            print(f'  {d + 1}/{a.docs} docs', flush=True)


if __name__ == '__main__':
    main()

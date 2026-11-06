"""DIL3 (PRE-REGISTERED 2026-11-06 15:32 IST, EXPERIMENTS_V14 "DIL3"): slow-head clock + retention dilation for held rules read
far back (Mind's clk3, hook v13_sempty/tmp/clk3_slowheads.py at 7cb3ef0). Release checkpoint, 8k, 100 docs per cell.
  R x K  (pkg_suite's cell: pack, filler, record row, "K ->"; first token = target). Rule twin = session without the pack.
  told   (A26 UNIT, train wording, seed 26000 * 1000 + 8192; the protocol twin read of pkg_suite's told cell).
Arms (same docs):
  a  today's read (R x K: package apply, echo s 8 from the pack end to the end; told: no dilation)
  b  echo-only dilation s 4 over the GAP only (rule / pack end + 64 -> the fetched row's start; row and question undilated)
  c  b + clk3: the K = 6 slowest notebook heads (measured on THIS body), retention r^(1/4) AND the clock g / 4 on the gap
    python v14/dil3.py --ckpt <weights> --docs 100 --out logs/v14_dil3_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16'), str(ROOT / 'v13_sempty' / 'tmp')):
    if p not in sys.path:
        sys.path.insert(0, p)
from clk3_slowheads import ST, patch, select_slow  # noqa: E402
from memory_probes.spread import _SEP, filler_tokens, split_units  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v7.data import _recall_key  # noqa: E402
from v14.a26_protocol import EXAM, FILL, make_doc, score  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402
from v16.package import Package  # noqa: E402

S4 = 4.0


def find(hay, needle, start=0):
    for i in range(start, len(hay) - len(needle) + 1):
        if hay[i:i + len(needle)] == needle:
            return i
    return None


def state(pkg, ids, gap, arm, slow):
    """the state after ids under arm a / b / c; gap = (start, end) absolute (b, c only)."""
    if arm == 'a' or gap is None or gap[1] <= gap[0]:
        return pkg._state(ids)
    if arm == 'c':
        m = torch.zeros(1, len(ids)); m[0, gap[0]:gap[1]] = 1.0
        ST.update(heads=slow, s=S4, couple=True, mask=m)
    try:
        return pkg._state(ids, (gap[0], gap[1], S4, 'echo'))
    finally:
        ST.update(heads=None, mask=None)


@torch.no_grad()
def rxk(pkg, out, base, docs, filler, slow, L=8192):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    sep, nl = enc(_SEP), enc('\n')
    for task in ('affine_map_new', 'shift_wide_new'):
        for sd in range(8800, 8800 + docs // 2):
            ex = b2_example(tok, task, 8, sd); header, units = split_units(ex)
            pack = enc(header) + sum((enc(u) for u in units[:-1]), [])
            lead, v, tail = re.match(r'^(\s*)(\S+)(\s*->.*)$', units[-1], re.S).groups()
            K = _recall_key(random.Random(sd * 31 + 7)); field = 'x' if task.startswith('affine') else 'letter'
            row = enc(f' Record: {K} | {field}: {v}.') + nl
            fill = filler_block(filler, L, sd, L)
            S = pack + sep + fill + sep + row; T = sep + fill + sep + row
            q = enc(lead + v + tail)                                          # the planner's value substitution (fetched 1.00)
            rec = dict(base, cell='rxk', sub=task, doc=sd, L=L)
            _, info = pkg.apply(S, f'{lead}{K}{tail}', {'pack': pack, 'query': lead + '{v}' + tail}, dilate=8.0)
            out.write(json.dumps(dict(rec, arm='a', right=int(info.get('first_token') == ex.target_token_id))) + '\n')
            rs, rt = len(S) - len(row), len(T) - len(row)
            for arm in ('b', 'c'):
                stS, offS = state(pkg, S, (len(pack) + 64, rs), arm, slow)
                stT, offT = state(pkg, T, (64, rt), arm, slow)
                la, _, _ = pkg.mind.feed(q, stS, offS); lb, _, _ = pkg.mind.feed(q, stT, offT)
                t = int((la + 4.0 * (la - lb)).argmax())
                out.write(json.dumps(dict(rec, arm=arm, right=int(t == ex.target_token_id))) + '\n')
            out.flush()


@torch.no_grad()
def told(pkg, out, base, docs, filler, slow, L=8192):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    rng = random.Random(26000 * 1000 + L + 0)
    for di in range(docs):
        doc = make_doc(rng, tok, filler, 'unit', L, 'train')
        qi = enc(EXAM.format(q=doc['question']))
        _, info = pkg.answer(doc['held'], EXAM.format(q=doc['question']))
        row = doc['held'][info['row'][0]:info['row'][1]] if info['row'] else []
        rh = info['row'][0] if info['row'] else len(doc['held'])
        rn = find(doc['noprot'], row) if row else None
        rn = len(doc['noprot']) if rn is None else rn
        rec = dict(base, cell='told', doc=di, L=L, unit=doc['unit'], target=doc['target'])
        for arm in ('a', 'b', 'c'):
            sh, oh = state(pkg, doc['held'], (1024, rh), arm, slow)
            sn, on = state(pkg, doc['noprot'], (1024, rn), arm, slow)
            ans = read(pkg, sh, oh, row + qi, sn, on, row + qi)
            out.write(json.dumps(dict(rec, arm=arm, answer=ans, **score('unit', ans, doc))) + '\n')
        out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--docs', type=int, default=100); ap.add_argument('--cells', default='rxk,told')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='DIL3', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device); model = pkg.mind.m
    meta = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    base = dict(step=meta.get('global_step'), tokens=meta.get('global_tokens'), hash=h)
    layers = [b.pam for b in model.blocks if getattr(b, 'pam', None) is not None]; patch(layers)
    filler = filler_tokens(str(FILL)); t0 = time.time()
    heads, slow = select_slow(model, layers, filler)
    sl = [d for d in sorted(heads, key=lambda d: -d['realized_half_life'])[:6]]
    print('slow heads (layer, head, half-life, warp g):', [(d['layer'], d['head'], d['realized_half_life'], d['warp_g']) for d in sl],
          f'({time.time() - t0:.0f} s)', flush=True)
    out = open(a.out, 'w'); out.write(json.dumps(dict(base, cell='meta', slow=sl)) + '\n')
    for cell in a.cells.split(','):
        t0 = time.time()
        (rxk if cell == 'rxk' else told)(pkg, out, base, a.docs, filler, slow)
        rows = [json.loads(l) for l in open(a.out) if json.loads(l)['cell'] == cell]
        acc = {arm: sum(r['right'] for r in rows if r['arm'] == arm) / max(1, sum(r['arm'] == arm for r in rows)) for arm in 'abc'}
        print(f'{cell}: ' + ' '.join(f'{k} {v:.3f}' for k, v in acc.items()) + f' ({time.time() - t0:.0f} s)', flush=True)
    print('DIL3 DONE', flush=True)


if __name__ == '__main__':
    main()

"""A26 (PRE-REGISTERED 2026-11-02 10:33 IST, v14/EXPERIMENTS_V14.md "A26"): a stated PROTOCOL applied to a FETCHED result.

A research notebook: ONE protocol sentence near the start (UNIT: "all masses here are in mg"; ROUND: "report values to
one decimal place"; CONVERT: "report masses in mg" with rows in g), then held-out prose with 8 result rows; the asked
row sits ~512 tokens before the question, the protocol d = 2k / 8k / 32k before it. Natural protocol wordings (5 per
family) and natural question styles (5), exam form.

Bodies and arms (the SAME docs and the SAME fetched row for every body):
  PAM mind via v16/package.py:  held (the row only; the protocol must come from the state), refresh (the protocol
      re-inserted every 2,048 tokens: the v0 refresh rule, TEXT), noprot (control), alone (the mind on its session
      state, no row), trained (the protocol as 8 "Report rule." demos; the planner takes the value out of the fetched
      row and writes "\\n{v} ->"; A23's apply_value).
  window bodies (T++, Mamba-2) via bench's Reader: asis (T++: the last 2,048 tokens; Mamba-2: the whole session),
      lib_held / lib_refresh / lib_noprot (the last 2,048 tokens of that session + the package's row + question,
      contrast vs without the row: A18's arm).
Scoring: free generation, decimal-aware stops, <= 16 tokens; right = value AND convention (see score()).

    python v14/a26_protocol.py --pam b2p2=<ckpt> --win t15tpp=<ckpt>,t15mamba=<ckpt> --lengths 8192,32768,2048 \\
        --docs 20 --device cuda --out logs/v14_a26_protocol_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from v16.bench import Reader  # noqa: E402
from v16.package import Package  # noqa: E402

FILL = ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'
QUANT = {'mass': ('masses', ['mg', 'g', 'kg', 'ng', 'μg']),
         'volume': ('volumes', ['mL', 'L', 'μL']),
         'concentration': ('concentrations', ['mM', 'μM', 'nM']),
         'length': ('lengths', ['nm', 'mm', 'cm'])}
MATERIALS = ['lysate B', 'serum A', 'buffer 3', 'pellet 2', 'supernatant C', 'extract D', 'culture 5', 'filtrate A',
             'eluate 4', 'homogenate B', 'plasma C', 'digest 7', 'fraction 9', 'stock solution E', 'control F', 'blank G']
PROTO = {
    'unit': ["Note: all {qp} in this notebook are recorded in {u}.", "Unless stated otherwise, {qp} are given in {u}.",
             "Protocol: every {q} below is reported in {u}.", "All {q} values in this record are in {u}.",
             "Units: {qp} throughout are in {u}."],
    'round': ["Protocol: report every measured value rounded to one decimal place.",
              "Note: round all reported values to one decimal place.",
              "All results should be reported to one decimal place.",
              "Reporting rule: give each value with exactly one decimal.",
              "Unless stated otherwise, values are reported to the nearest 0.1."],
    'convert': ["Protocol: report all masses in mg.", "Note: masses must be reported in milligrams (mg).",
                "All masses in the summary are to be given in mg.", "Reporting rule: convert every mass to mg.",
                "Unless stated otherwise, report masses in mg."]}
QUEST = ["What was the {q} of sample {s}?", "Report the {q} of sample {s}.", "What {q} was recorded for sample {s}?",
         "Give the measured {q} for {s}.", "{s}: what was its {q}?"]
ROWS = [" Sample {s} ({m}): {q} {v}.", " For sample {s} ({m}) the measured {q} was {v}.", " {s} ({m}), {q}: {v}."]
EXAM = "\n\nQuestion: {q}\nAnswer:"
LEAD, REFRESH, ASKED_BACK, WINDOW, MAX_NEW, ALPHA = 256, 2048, 512, 2048, 16, 4.0
LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'


def _tenths(rng, lo=5, hi=999):
    while True:                                    # a value with one decimal whose last digit is not 0
        x = rng.randint(lo, hi)
        if x % 10:
            return x


def item(rng, cell):
    """one result: (row value text, planner value, target value text, target unit / None, raw text for ROUND)."""
    if cell == 'unit':
        t = _tenths(rng); v = f'{t / 10:.1f}'
        return v, v, v
    if cell == 'round':
        t = _tenths(rng, 10, 999)
        while True:                                # raw thousandths: rounds half-up to t tenths; half round UP
            delta = rng.choice([d for d in range(-45, 45) if d != 0])
            raw = t * 100 + delta
            if raw % 10:
                break
        r = f'{raw / 1000:.3f}'
        return r, r, f'{t / 10:.1f}'
    t = _tenths(rng)                               # convert: mg with one decimal = g with four
    return f'{t / 10000:.4f}', f'{t / 10000:.4f}', f'{t / 10:.1f}'


def make_doc(rng, tok, fill, cell, d, wording='a26'):
    """wording 'a26' = the registered (held-out) wordings; 'train' = G1 of the addendum: UNIT on the TRAINED quantities
    (mass / volume) with v14/protocol_docs.py's own protocol wordings, question styles and row templates."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    if wording == 'train':
        from v14 import protocol_docs as P
        assert cell == 'unit', 'G1 (trained wording) is defined for the UNIT cell only'
        q = rng.choice(list(P.UNIT_Q)); qp, Qp, units = P.UNIT_Q[q]; u = rng.choice(units)
        proto = rng.choice(P.UNIT_W).format(q=q, qp=qp, Qp=Qp, u=u)
        quest_t, rows_t, mat_list = P.QUESTIONS, P.ROW_T, P.MATERIALS
    else:
        q = 'mass' if cell == 'convert' else rng.choice(list(QUANT))
        qp, units = QUANT[q]
        u = 'mg' if cell == 'convert' else rng.choice(units)
        proto = rng.choice(PROTO[cell]).format(q=q, qp=qp, u=u)
        quest_t, rows_t, mat_list = QUEST, ROWS, MATERIALS
    ids = set()
    while len(ids) < 8:
        ids.add(rng.choice(LETTERS) + str(rng.randint(1, 29)))
    samples = sorted(ids); rng.shuffle(samples)          # sorted first: set order depends on PYTHONHASHSEED
    mats = rng.sample(mat_list, 8)
    rows, asked = [], None
    for i, s in enumerate(samples):
        rv, pv, tv = item(rng, cell)
        shown = rv if cell == 'unit' else f'{rv} {"g" if cell == "convert" else u}'
        rows.append(enc(rng.choice(rows_t).format(s=s, m=mats[i], q=q, v=shown)))
        if i == 0:
            asked = dict(sample=s, row_value=rv, planner_value=pv, target=tv, unit=u)
    L = d + LEAD
    start = rng.randrange(0, len(fill) - L - 1)
    body = fill[start:start + L].tolist()
    dot = enc('.')[0]

    def snap(p_):                                  # insert at a sentence start of the filler (after its next '.')
        for j in range(p_, min(p_ + 64, L - 1)):
            if body[j] == dot:
                return j + 1
        return p_
    lead = snap(LEAD)
    pos = [snap(L - ASKED_BACK)] + sorted(snap(rng.randrange(LEAD + 64, L - ASKED_BACK - 128)) for _ in range(7))
    order = sorted(range(8), key=lambda i: pos[i])
    demo_rng = random.Random(rng.random())
    demos = []
    for _ in range(8):
        rv, pv, tv = item(demo_rng, cell)
        y = f'{tv} {u}' if cell in ('unit', 'convert') else tv
        demos.append(f'\n{pv} -> {y}')
    pack = 'Report rule.' + ''.join(demos)

    def build(head, refresh=False):
        out, cur = [], 0
        events = sorted([(lead, 'proto')] * (head is not None) + [(pos[i], i) for i in order],
                        key=lambda e: e[0])
        if refresh and head is not None:
            # copies anchored from the QUESTION backwards (d5da73d anchored them from the start, which left the newest
            # copy ~2.0-2.2k tokens back: outside a 2k window); the newest now sits just before the asked row
            newest = L - ASKED_BACK - 64
            events = sorted(events + [(snap(p), 'proto') for p in range(newest, lead + 64, -REFRESH)], key=lambda e: e[0])
        for p_, what in events:
            out += body[cur:p_]; cur = p_
            out += head if what == 'proto' else rows[what]
        return out + body[cur:]
    sent = enc(' ' + proto)
    question = rng.choice(quest_t).format(q=q, s=asked['sample'])
    q_cap = question[0].upper() + question[1:]
    return dict(held=build(sent), refresh=build(sent, refresh=True), noprot=build(None),
                trained=build(enc('\n') + enc(pack) + enc('\n')), pack=pack, proto=proto, question=q_cap, q=q, wording=wording,
                rows_text=[tok.decode(r).strip() for r in rows], **asked)


def num_re(v):
    return r'(?<![\d.])' + re.escape(v) + r'(?!\d|\.\d)'


def unit_re(u):
    return r'(?<![A-Za-zμ])' + re.escape(u) + r'(?![A-Za-z])'


def score(cell, ans, doc):
    if cell == 'round':
        rounded = re.search(num_re(doc['target']), ans) is not None
        raw = re.search(num_re(doc['row_value']), ans) is not None
        return dict(value_ok=int(rounded or raw), conv_ok=int(rounded and not raw), right=int(rounded and not raw))
    val = re.search(num_re(doc['target']), ans) is not None
    conv = re.search(unit_re(doc['unit']), ans) is not None
    return dict(value_ok=int(val), conv_ok=int(conv), right=int(val and conv))


def planner_value(row_text, q):
    m = re.search(re.escape(q) + r'[^0-9]*?(\d+(?:\.\d+)?)', row_text or '')
    return m.group(1) if m else None


def win_greedy(R, pkg, a, b=None, alpha=ALPHA):
    """greedy (contrast with b) over a window body's last logits, the package's decimal-aware stops."""
    out, held = [], False
    for _ in range(MAX_NEW):
        la = R.last_logits(a + out).float()
        lg = la if b is None else la + alpha * (la - R.last_logits(b + out).float())
        t = int(lg.argmax())
        if held:
            if not (t < len(pkg._digit_start) and pkg._digit_start[t]):
                break
            held = False
        out.append(t)
        if t in pkg.stops:
            if t == pkg.dot and len(out) >= 2 and out[-2] < len(pkg._digit_end) and pkg._digit_end[out[-2]]:
                held = True
            else:
                break
    return out


def meta(ckpt):
    pl = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
    keep = {k: pl[k] for k in ('step', 'tokens_seen', 'total_tokens', 'tokens', 'epoch', 'best_val_loss') if k in pl}
    return dict(keep, arch=(pl.get('config') or {}).get('arch', 'pam'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pam', default='', help='name=ckpt[,name=ckpt]')
    ap.add_argument('--win', default='', help='name=ckpt[,name=ckpt] (transformer / mamba2)')
    ap.add_argument('--cells', default='unit,round,convert'); ap.add_argument('--lengths', default='8192,32768,2048')
    ap.add_argument('--docs', type=int, default=20); ap.add_argument('--seed', type=int, default=26000)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    ap.add_argument('--guard_mb', type=int, default=2000)
    ap.add_argument('--wording', default='a26', choices=['a26', 'train'], help="'train' = addendum G1 (UNIT only)")
    ap.add_argument('--arms', default='all', help="comma list of arms to run (e.g. refresh,lib_refresh); 'all' = every arm")
    a = ap.parse_args()
    head = dict(unit='A26', hash=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                                cwd=ROOT).stdout.strip(), started=time.strftime('%Y-%m-%d %H:%M'), args=vars(a))
    pams = dict(x.split('=', 1) for x in a.pam.split(',') if x)
    wins = dict(x.split('=', 1) for x in a.win.split(',') if x)
    pk = {n: Package(c, device=a.device) for n, c in pams.items()}
    ref = next(iter(pk.values()))                        # the package that fetches the row every body reads
    tok = ref.tok
    wr = {}
    for n, c in wins.items():
        m, _, _ = _load_v13_sempty(Path(c), 'baseline_real_pm', torch.device(a.device)); m.eval()
        wr[n] = Reader(m, meta(c)['arch'])
    head['bodies'] = {n: dict(ckpt=c, params=sum(p.numel() for p in (pk[n].mind.m if n in pk else wr[n].m).parameters()),
                              **meta(c)) for n, c in {**pams, **wins}.items()}
    print('HEADER ' + json.dumps(head, default=str), flush=True)
    fill = filler_tokens(str(FILL))
    outf = open(a.out, 'a')
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    for d in [int(x) for x in a.lengths.split(',')]:
        for cell in a.cells.split(','):
            rng = random.Random(a.seed * 1000 + d + ['unit', 'round', 'convert'].index(cell))
            for di in range(a.docs):
                if a.device.startswith('cuda'):
                    free = torch.cuda.mem_get_info()[0] / 2 ** 20
                    if free < a.guard_mb:
                        print(f'GUARD: {free:.0f} MB free < {a.guard_mb}; stopping', flush=True)
                        return
                doc = make_doc(rng, tok, fill, cell, d, a.wording)
                qi = enc(EXAM.format(q=doc['question']))
                base = dict(cell=cell, d=d, doc=di, proto=doc['proto'], question=doc['question'], target=doc['target'],
                            unit=doc['unit'], row_value=doc['row_value'], quantity=doc['q'], wording=doc['wording'])
                t0 = time.time()
                _, finfo = ref.answer(doc['held'], EXAM.format(q=doc['question']))
                row = finfo['row']
                rows_tok = {k: (doc[k][row[0]:row[1]] if row else []) for k in ('held',)}
                want = (lambda arm: a.arms == 'all' or arm in a.arms.split(','))  # noqa: E731
                for n, pkg in pk.items():
                    for arm in [x for x in ('held', 'refresh', 'noprot') if want(x)]:
                        ans, info = pkg.answer(doc[arm], EXAM.format(q=doc['question']))
                        outf.write(json.dumps(dict(base, body=n, arm=arm, answer=ans, path=info['path'],
                                                   row=(info.get('row_text') or '')[-90:], **score(cell, ans, doc))) + '\n')
                    if not want('alone') and not want('trained'):
                        continue
                    s = pkg.read(doc['held']); st = copy.deepcopy(s['state'])
                    la, sa, oa = pkg.mind.feed(qi, st, s['off'])
                    alone = tok.decode(pkg._gen(la, sa, oa, la, copy.deepcopy(sa), oa, 0.0, MAX_NEW))
                    outf.write(json.dumps(dict(base, body=n, arm='alone', answer=alone, **score(cell, alone, doc))) + '\n')
                    fr = pkg.fetch_rows(doc['trained'], EXAM.format(q=doc['question']), k=1)
                    pv = planner_value(fr[0][3] if fr else '', doc['q'])
                    if pv is None:
                        ans, ok = '', dict(value_ok=0, conv_ok=0, right=0)
                    else:
                        ans, _ = pkg.apply_value(doc['trained'], {'pack': doc['pack'], 'query': '\n{v} ->'}, pv, max_new=MAX_NEW)
                        ok = score(cell, ans, doc)
                    outf.write(json.dumps(dict(base, body=n, arm='trained', answer=ans, planner_value=pv, **ok)) + '\n')
                for n, R in wr.items():
                    held = doc['held']
                    ctx = held if (R.arch == 'mamba2' or len(held) <= WINDOW) else held[-WINDOW:]
                    if want('asis'):
                        ans = tok.decode(win_greedy(R, ref, ctx + qi))
                        outf.write(json.dumps(dict(base, body=n, arm='asis', answer=ans, **score(cell, ans, doc))) + '\n')
                    for arm in [x for x in ('held', 'refresh', 'noprot') if want('lib_' + x)]:
                        win = doc[arm][-WINDOW:]
                        r_ = rows_tok['held']
                        ans = tok.decode(win_greedy(R, ref, win + r_ + qi, win + qi)) if r_ else ''
                        outf.write(json.dumps(dict(base, body=n, arm='lib_' + arm, answer=ans, **score(cell, ans, doc))) + '\n')
                outf.flush()
                print(d, cell, di, f'{time.time() - t0:.1f} s', flush=True)
    print('A26 DONE', flush=True)


if __name__ == '__main__':
    main()

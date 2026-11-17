"""SCOPE probe (Argon via Lead 2026-11-17 12:19 IST; PRE-REGISTERED in EXPERIMENTS_V14 "SCOPE probe"): the standing 4-tuple
(B, C, scope_fu, prec_u1) on the plain twin, at 2k + 8k. One FAR-TOLD doc skeleton (A28's make_doc, trained wording), no new prose.
Teacher-forced like TOLD-MARGIN: one pass per stream; logits at the step after the gold NUMBER (prefix = fetched row + question +
" " + value). Units are scored by their first token. twin = held + 4 (held - noprot).
  P presence   no protocol (the noprot stream).  P_fu = any unit of either quantity beats '.'; B = mean l(gold) - l('.').
  C strength   own protocol: TOLD-MARGIN on the same docs (same rng, so C reproduces told_margin.py): M_id, M_gate, C.
  S scope      protocol for the OTHER quantity, with that quantity's real units. HELD-OUT direction only (asked volume, protocol
               about mass; never in training, protocol_docs.SCOPE_HELD_ASKED). Right = bare value: S_fu = any unit beats '.'
               on the twin. Diagnostic (no bar): S_bound_fu = only the asked quantity's units are counted.
  R precedence own protocol u1; the asked row states value + u2 (same quantity, u2 != u1). Right = u2: R_u1 = u1 is the top token
               of {all units, '.'} on the twin; R_u2 likewise.
    python v14/scope_probe.py --ckpt <weights> --lengths 2048,8192 --docs 100 --out logs/v14_scope_<tag>_<hash>_<stamp>.jsonl
--fullctx (wio scope test, Lead 11-17): ONE read path for PAM / nochrono / GDN. Two plain full-context passes, held doc and noprot
doc, each + the GOLD asked row (oracle, no package fetch) + question + number; last-position logits via run_skillpack_probe.Body.
GDN runs in the qllm_gdn venv. This C is NOT comparable to the default (fetched-row, state-fork) path.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import filler_tokens  # noqa: E402
from v14 import protocol_docs as P  # noqa: E402
from v14.a26_protocol import EXAM, FILL, make_doc  # noqa: E402
from v14.a32b import find  # noqa: E402
from v14.cgate import mismatch  # noqa: E402
from v16.package import Package  # noqa: E402

ALPHA = 4.0


@torch.no_grad()
def logits_at(pkg, doc_held, doc_noprot, question, enc, target):
    """held / noprot / twin next-token logits after the gold number (the row is fetched from the held doc)."""
    q = EXAM.format(q=question); qi = enc(q)
    _, info = pkg.answer(doc_held, q)
    row = doc_held[info['row'][0]:info['row'][1]] if info['row'] else []
    pre = row + qi + enc(' ' + target)
    sh, sn = pkg.read(doc_held), pkg.read(doc_noprot)
    lh, _, _ = pkg.mind.feed(pre, copy.deepcopy(sh['state']), sh['off'])
    ln, _, _ = pkg.mind.feed(pre, copy.deepcopy(sn['state']), sn['off'])
    lh, ln = lh.float().flatten(), ln.float().flatten()
    return lh, ln, lh + ALPHA * (lh - ln)


class FullCtx:
    """--fullctx reader: plain full-context passes (any v13_sempty arch incl. gdn), gold asked row."""

    def __init__(self, ckpt, device):
        from run_memory_behavioral import _load_v13_sempty
        from run_skillpack_probe import Body
        m, self.tok, _ = _load_v13_sempty(Path(ckpt), 'baseline_real_pm', torch.device(device))
        m.eval(); self.body = Body(m, torch.device(device)); self.arch = self.body.arch

    def logits(self, doc, question, enc, target):
        pre = enc(' ' + doc['rows_text'][0]) + enc(EXAM.format(q=question)) + enc(' ' + target)
        lh = self.body.last_logits(doc['held'] + pre); ln = self.body.last_logits(doc['noprot'] + pre)
        return lh, ln, lh + ALPHA * (lh - ln)


def swap_row_unit(doc, rng, enc, tok):
    """R cell: the asked row (rows_text[0]) states value + u2 (same quantity, u2 != the protocol's u1), in held AND noprot."""
    units = P.UNIT_Q[doc['q']][2]; u2 = rng.choice([u for u in units if u != doc['unit']])
    old_t = doc['rows_text'][0]; v = doc['row_value']
    assert v in old_t
    new_t = old_t.replace(v, f'{v} {u2}', 1)
    out = dict(doc, u2=u2, rows_text=[new_t] + doc['rows_text'][1:])
    for k in ('held', 'noprot'):
        seq = doc[k]
        for pre in ('', ' '):                        # rows are encoded standalone, with or without a leading space
            cand = enc(pre + old_t); p0 = find(seq, cand)
            if p0 is not None:
                out[k] = seq[:p0] + enc(pre + new_t) + seq[p0 + len(cand):]
                break
        else:
            raise AssertionError('asked row not found')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--lengths', default='2048,8192')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    ap.add_argument('--fullctx', action='store_true')
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='SCOPE probe', hash=h, args=vars(a))), flush=True)
    if a.fullctx:
        fc = FullCtx(a.ckpt, a.device); tok = fc.tok; print('fullctx arch', fc.arch, flush=True)
        read3 = lambda d, enc_: fc.logits(d, d['question'], enc_, d['target'])  # noqa: E731
    else:
        pkg = Package(a.ckpt, device=a.device); tok = pkg.tok
        read3 = lambda d, enc_: logits_at(pkg, d['held'], d['noprot'], d['question'], enc_, d['target'])  # noqa: E731
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    first = {q: {u: enc(' ' + u)[0] for u in units} for q, (_, _, units) in P.UNIT_Q.items()}
    all_u = sorted({t for m in first.values() for t in m.values()})
    dot = enc('.')[0]
    fill = filler_tokens(str(FILL)); out = open(a.out, 'w'); summ = {}
    for L in [int(x) for x in a.lengths.split(',')]:
        acc = {k: [] for k in ('B', 'P_fu', 'C', 'C_mid_twin', 'S_fu', 'S_fu_held', 'S_bound_fu', 'R_u1', 'R_u2')}
        # P + C: TOLD-MARGIN's docs (same rng); S: own stream, held-out direction only; R: own stream
        rng_pc = random.Random(26000 * 1000 + L + 0)
        rng_s, mrng = random.Random(26000 * 1000 + L + 11), random.Random(26000 * 37 + L)
        rng_r, urng = random.Random(26000 * 1000 + L + 13), random.Random(26000 * 41 + L)
        for di in range(a.docs):
            rec = dict(L=L, doc=di, hash=h)
            doc = make_doc(rng_pc, tok, fill, 'unit', L, 'train'); units = first[doc['q']]
            gold = units[doc['unit']]; others = [t for u, t in units.items() if u != doc['unit']]
            lh, ln, lt = read3(doc, enc)
            hg, ng = float(lh[gold] - lh[dot]), float(ln[gold] - ln[dot])
            rec |= dict(quantity=doc['q'], unit=doc['unit'], B=ng, P_fu=int(float(ln[all_u].max() - ln[dot]) > 0), C=hg - ng,
                        held_m_gate=hg, noprot_m_gate=ng, twin_m_id=float(lt[gold] - lt[others].max()),
                        twin_m_gate=float(lt[gold] - lt[dot]))
            while True:                                      # S: asked = the held-out quantity
                ds = make_doc(rng_s, tok, fill, 'unit', L, 'train')
                if ds['q'] == P.SCOPE_HELD_ASKED:
                    break
            ds = mismatch(ds, mrng, enc)
            sh, _, st = read3(ds, enc)
            ask_u = list(first[ds['q']].values())
            rec |= dict(S_proto=ds['proto'], S_fu=int(float(st[all_u].max() - st[dot]) > 0),
                        S_fu_held=int(float(sh[all_u].max() - sh[dot]) > 0),
                        S_bound_fu=int(float(st[ask_u].max() - st[dot]) > 0))
            dr = swap_row_unit(make_doc(rng_r, tok, fill, 'unit', L, 'train'), urng, enc, tok)
            _, _, rt = read3(dr, enc)
            cand = all_u + [dot]; top = cand[int(rt[cand].argmax())]
            u1t, u2t = first[dr['q']][dr['unit']], first[dr['q']][dr['u2']]
            rec |= dict(R_u1_unit=dr['unit'], R_u2_unit=dr['u2'], R_u1=int(top == u1t), R_u2=int(top == u2t))
            rec['C_mid_twin'] = int(rec['twin_m_id'] > 0)
            for k in acc:
                acc[k].append(rec[k])
            out.write(json.dumps(rec) + '\n'); out.flush()
        n = a.docs; m = {k: sum(v) / n for k, v in acc.items()}
        summ[L] = m
        print(f"L {L}: 4-tuple B {m['B']:+.2f} C {m['C']:+.3f} scope_fu {m['S_fu']:.2f} prec_u1 {m['R_u1']:.2f} | P_fu {m['P_fu']:.2f} "
              f"twin unit_acc {m['C_mid_twin']:.2f} | S_fu_held {m['S_fu_held']:.2f} S_bound_fu {m['S_bound_fu']:.2f} | R_u2 {m['R_u2']:.2f}",
              flush=True)
    print('SUMMARY ' + json.dumps(summ), flush=True)
    print('SCOPE DONE', flush=True)


if __name__ == '__main__':
    main()

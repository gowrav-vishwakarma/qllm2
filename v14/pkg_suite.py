"""Per-milestone PACKAGE suite for SCI-1 (Lead, user decision 2026-11-05 13:40 IST; pre-registered in EXPERIMENTS_V14 "PKG-SUITE").

One row per milestone: BARE (the model alone on the session + question, plain greedy, no fetch, no contrast) vs PACKAGE (the
package path for that cell) on the same docs:
  rxk      rule held only in state x fact fetched at question time (Mind's R x K docs, A23): package = fetch the record, write
           "<v> ->", skill contrast (8k: echo dilation s 8); bare = "K ->" asked with the record in view. Free = first token.
  rewrite  a told rule (A26 UNIT prose protocol) compiled next to the question (A29); bare = state + question.
  told     a told rule held in state (A28 / A31): package = fetched row + question, rule twin (= the state fork); bare = alone.
  recall   needle among 8 (kv), latest-wins (kvupd), two-hop (hop2) through the text organ (v16/bench.py docs): package =
           v16/package.py answer (text fetch -> warm + contrast; hop2 = a TEXT planner chain of two answers); bare = state
           only. 20-way (first answer token over bench's 20 value nouns) and free (value in the sentence). MB held per doc.
    python v14/pkg_suite.py --ckpt <weights> --tag c4a --cells rxk,rewrite,told,recall --out logs/v14_pkg_suite_<tag>_<hash>_<stamp>.jsonl
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
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.spread import _SEP, filler_tokens, split_units  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v7.data import _RECALL_VALUE_NOUNS, _recall_key  # noqa: E402
from v14.a26_protocol import EXAM, FILL, MAX_NEW, make_doc, planner_value, score  # noqa: E402
from v14.a28_twin import read  # noqa: E402
from v14.a29_rewrite import parse_protocol  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402
from v16.bench import build as bench_build  # noqa: E402
from v16.package import Package  # noqa: E402


def sent_ok(text, val):
    return int(re.search(r'\b' + re.escape(str(val)) + r'\b', text or '', re.I) is not None)


@torch.no_grad()
def bare(pkg, session, q_ids):
    """the model alone: session state + question, plain greedy (decimal-aware stops). Returns (text, first-step logits)."""
    s = pkg.read(session); st = copy.deepcopy(s['state'])
    la, sa, oa = pkg.mind.feed(q_ids, st, s['off'])
    first = la.clone()
    return pkg.tok.decode(pkg._gen(la, sa, oa, la, copy.deepcopy(sa), oa, 0.0, MAX_NEW)), first


@torch.no_grad()
def pkg_first(pkg, session, row, q_ids):
    """the package read's first-step logits (warm + no-row contrast, as Package._decode)."""
    s = pkg.read(session)
    la, _, _ = pkg.mind.feed(row + q_ids, copy.deepcopy(s['state']), s['off'])
    lb, _, _ = pkg.mind.feed(q_ids, copy.deepcopy(s['state']), s['off'])
    return la + pkg.alpha * (la - lb)


def cell_rxk(pkg, out, base, lengths, docs, seed, filler):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    sep, nl = enc(_SEP), enc('\n')
    for task in ('affine_map_new', 'shift_wide_new'):
        for sd in range(seed, seed + docs // 2):
            ex = b2_example(tok, task, 8, sd); header, units = split_units(ex)
            pack = enc(header) + sum((enc(u) for u in units[:-1]), [])
            lead, v, tail = re.match(r'^(\s*)(\S+)(\s*->.*)$', units[-1], re.S).groups()
            K = _recall_key(random.Random(sd * 31 + 7)); field = 'x' if task.startswith('affine') else 'letter'
            row = enc(f' Record: {K} | {field}: {v}.') + nl
            for L in lengths:
                sess = pack + sep + filler_block(filler, L, sd, L) + sep + row
                ans, info = pkg.apply(sess, f'{lead}{K}{tail}', {'pack': pack, 'query': lead + '{v}' + tail},
                                      dilate=8.0 if L >= 4096 else None)
                btxt, bl = bare(pkg, sess, enc(f'{lead}{K}{tail}'))
                rec = dict(base, cell='rxk', sub=task, doc=sd, L=L)
                out.write(json.dumps(dict(rec, arm='package', right=int(info.get('first_token') == ex.target_token_id), answer=ans,
                                          value_fetched=int(info.get('value') == v))) + '\n')
                out.write(json.dumps(dict(rec, arm='bare', right=int(int(bl.argmax()) == ex.target_token_id), answer=btxt)) + '\n')
            out.flush()


def cell_rewrite(pkg, out, base, lengths, docs, seed, filler, wording):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    for L in lengths:
        rng = random.Random(seed * 1000 + L + 0)
        for di in range(docs):
            doc = make_doc(rng, tok, filler, 'unit', L, wording)
            q = EXAM.format(q=doc['question'])
            _, info = pkg.answer(doc['held'], q); row_text = info.get('row_text') or ''
            unit = parse_protocol(tok.decode(doc['held']), doc['q']); v = planner_value(row_text, doc['q'])
            if unit is None or v is None:
                ans = ''
            else:
                drng = random.Random(di * 7919 + L)
                pack = 'Report rule.' + ''.join(f'\n{x / 10:.1f} -> {x / 10:.1f} {unit}' for x in drng.sample(range(11, 999), 3))
                ans, _ = pkg.apply_value(doc['held'] + enc('\n\n') + enc(pack), {'pack': pack, 'query': '\n{v} ->'}, v, max_new=MAX_NEW)
            btxt, _ = bare(pkg, doc['held'], enc(q))
            rec = dict(base, cell='rewrite', wording=wording, doc=di, L=L, unit=doc['unit'], target=doc['target'])
            out.write(json.dumps(dict(rec, arm='package', answer=ans, parser_ok=int(unit == doc['unit']), **score('unit', ans, doc))) + '\n')
            out.write(json.dumps(dict(rec, arm='bare', answer=btxt, **score('unit', btxt, doc))) + '\n')
            out.flush()


def cell_told(pkg, out, base, lengths, docs, seed, filler):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    for L in lengths:
        rng = random.Random(seed * 1000 + L + 0)
        for di in range(docs):
            doc = make_doc(rng, tok, filler, 'unit', L, 'train')
            q = EXAM.format(q=doc['question']); qi = enc(q)
            _, info = pkg.answer(doc['held'], q)
            row = doc['held'][info['row'][0]:info['row'][1]] if info['row'] else []
            sh = pkg.read(doc['held']); st_h, off_h = copy.deepcopy(sh['state']), sh['off']
            sn = pkg.read(doc['noprot']); st_n, off_n = copy.deepcopy(sn['state']), sn['off']
            twin = read(pkg, st_h, off_h, row + qi, st_n, off_n, row + qi)
            btxt, _ = bare(pkg, doc['held'], qi)
            rec = dict(base, cell='told', wording='train', doc=di, L=L, unit=doc['unit'], target=doc['target'])
            out.write(json.dumps(dict(rec, arm='package', answer=twin, **score('unit', twin, doc))) + '\n')
            out.write(json.dumps(dict(rec, arm='bare', answer=btxt, **score('unit', btxt, doc))) + '\n')
            out.flush()


def cell_recall(pkg, out, base, lengths, docs_by_len, seed, filler):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    cand = [enc(f' {v}')[0] for v in _RECALL_VALUE_NOUNS]
    for L in lengths:
        for task in ('kv', 'kvupd', 'hop2'):
            rng = random.Random(seed * 1000 + L + ('kv', 'kvupd', 'hop2').index(task))
            for di in range(docs_by_len[L]):
                ids, ans_tok, qlen, meta = bench_build(rng, tok, filler, L, task, 8)
                sess, q = ids[:-qlen], ids[-qlen:]
                t0 = time.time()
                btxt, bl = bare(pkg, sess, q)
                qtext = tok.decode(q)
                if task == 'hop2':                                              # TEXT planner chain: K1 -> K2 -> V
                    a1, i1 = pkg.answer(sess, qtext)
                    k2 = (a1 or '').strip().rstrip('.').strip()
                    q2 = f' Query: {k2} maps to'
                    ptxt, info = pkg.answer(sess, q2)
                    row = sess[info['row'][0]:info['row'][1]] if info['row'] else []
                    pl = pkg_first(pkg, sess, row, enc(q2)) if row else None
                else:
                    ptxt, info = pkg.answer(sess, qtext)
                    row = sess[info['row'][0]:info['row'][1]] if info['row'] else []
                    pl = pkg_first(pkg, sess, row, q) if row else None
                rec = dict(base, cell='recall', task=task, doc=di, L=L, mb=info.get('bytes_held'), sec=round(time.time() - t0, 1))
                out.write(json.dumps(dict(rec, arm='package', answer=ptxt, free=sent_ok(ptxt, meta['val']), path=info.get('path'),
                                          w20=int(pl is not None and cand[int(pl[cand].argmax())] == ans_tok))) + '\n')
                out.write(json.dumps(dict(rec, arm='bare', answer=btxt, free=sent_ok(btxt, meta['val']),
                                          w20=int(cand[int(bl[cand].argmax())] == ans_tok))) + '\n')
                out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--cells', default='rxk,rewrite,told,recall'); ap.add_argument('--docs', type=int, default=100)
    ap.add_argument('--recall_docs', default='32768:20'); ap.add_argument('--rewrite_wording', default='a26')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='PKG-SUITE', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device)
    meta = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    base = dict(tag=a.tag, step=meta.get('global_step'), tokens=meta.get('global_tokens'), hash=h)
    filler = filler_tokens(str(FILL)); out = open(a.out, 'a')
    T = {}
    for cell in a.cells.split(','):
        t0 = time.time()
        if cell == 'rxk':
            cell_rxk(pkg, out, base, [2048, 8192], a.docs, 8800, filler)
        elif cell == 'rewrite':
            cell_rewrite(pkg, out, base, [2048, 8192], a.docs, 26000, filler, a.rewrite_wording)
        elif cell == 'told':
            cell_told(pkg, out, base, [2048, 8192], a.docs, 26000, filler)
        elif cell == 'recall':
            dl = {int(k): int(v) for k, v in (x.split(':') for x in a.recall_docs.split(','))}
            cell_recall(pkg, out, base, sorted(dl), dl, 27000, filler)
        T[cell] = round(time.time() - t0)
        print(f'{cell} done in {T[cell]} s', flush=True)
    print('PKG-SUITE DONE ' + json.dumps(T), flush=True)


if __name__ == '__main__':
    main()

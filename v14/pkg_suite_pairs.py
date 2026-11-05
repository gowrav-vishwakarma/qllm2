"""PKG-SUITE per-item PAIRS for the review page (Lead, 2026-11-05 15:00): one row per item, "model alone" vs "model + library".

Rebuilds each item's question / gold / context deterministically from pkg_suite.py's seeds (CPU, tokenizer only) and joins the
bare and package answers from a suite jsonl. Fields per row:
  tag, step, tokens, cell (rxk / rewrite / told / recall), task, L, doc,
  context (the short text the item is about: the rule / protocol, the record row; not the filler),
  question, gold, bare_answer, bare_ok, package_answer, package_ok, package_path (recall), mb (recall)
    python v14/pkg_suite_pairs.py logs/v14_pkg_suite_<tag>_<hash>_<stamp>.jsonl   -> writes <same>_pairs.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)


def main(path):
    from memory_probes.spread import filler_tokens, split_units
    from v7.data import get_chat_tokenizer, _recall_key
    from v14.a26_protocol import EXAM, FILL, make_doc
    from v14.rules_ood import build_example as b2_example
    from v16.bench import build as bench_build
    tok = get_chat_tokenizer(); filler = filler_tokens(str(FILL))
    rows = [json.loads(l) for l in open(path)]
    by = {}
    for r in rows:
        k = (r['cell'], r.get('sub') or r.get('task') or '', r['L'], r['doc'])
        by.setdefault(k, {})[r['arm']] = r
    info = {}
    Ls = sorted({r['L'] for r in rows if r['cell'] in ('rxk', 'rewrite', 'told')})
    docs = {c: max([r['doc'] for r in rows if r['cell'] == c] or [-1]) + 1 for c in ('rewrite', 'told', 'recall')}
    # rxk: seeds 8800.. per task, the same question at every L
    for task in ('affine_map_new', 'shift_wide_new'):
        for sd in sorted({r['doc'] for r in rows if r['cell'] == 'rxk' and r.get('sub') == task}):
            ex = b2_example(tok, task, 8, sd); header, units = split_units(ex)
            lead, v, tail = re.match(r'^(\s*)(\S+)(\s*->.*)$', units[-1], re.S).groups()
            K = _recall_key(random.Random(sd * 31 + 7)); field = 'x' if task.startswith('affine') else 'letter'
            ctx = header + ''.join(units[:-1]) + f'\n[... filler ...]\n Record: {K} | {field}: {v}.'
            for L in Ls:
                info[('rxk', task, L, sd)] = dict(context=ctx, question=f'{lead}{K}{tail}', gold=ex.target_text.strip())
    for cell, wording in (('rewrite', next((r.get('wording') for r in rows if r['cell'] == 'rewrite'), 'a26')), ('told', 'train')):
        for L in Ls:
            rng = random.Random(26000 * 1000 + L + 0)
            for di in range(docs[cell]):
                d = make_doc(rng, tok, filler, 'unit', L, wording)
                info[(cell, '', L, di)] = dict(context=f"{d['proto']}\n[... filler ...]\n{d['rows_text'][0]}",
                                               question=EXAM.format(q=d['question']).strip(), gold=f"{d['target']} {d['unit']}")
    for L in sorted({r['L'] for r in rows if r['cell'] == 'recall'}):
        for ti, task in enumerate(('kv', 'kvupd', 'hop2')):
            rng = random.Random(27000 * 1000 + L + ti)
            for di in range(docs['recall']):
                ids, ans, qlen, meta = bench_build(rng, tok, filler, L, task, 8)
                q = tok.decode(ids[-qlen:]).strip()
                if task == 'hop2':
                    ctx = f"Record: {meta['key']} maps to {meta['hop_key']}. ... Record: {meta['hop_key']} maps to {meta['val']}. (+ a distractor chain)"
                elif task == 'kvupd':
                    ctx = f"Definition: {meta['key']} refers to <old value>. ... later: Definition: {meta['key']} refers to {meta['val']}. (+ 7 other definitions)"
                else:
                    ctx = f"Definition: {meta['key']} refers to {meta['val']}. (+ 7 other definitions, {L // 1024}k tokens of prose)"
                info[('recall', task, L, di)] = dict(context=ctx, question=q, gold=meta['val'])
    out = open(str(path).replace('.jsonl', '_pairs.jsonl'), 'w'); n = 0
    for k, arms in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        b, p = arms.get('bare', {}), arms.get('package', {})
        ok = lambda r: r.get('right', r.get('free'))  # noqa: E731
        any_r = p or b
        row = dict(tag=any_r.get('tag'), step=any_r.get('step'), tokens=any_r.get('tokens'), cell=k[0], task=k[1], L=k[2], doc=k[3],
                   **info.get(k, {}), bare_answer=b.get('answer'), bare_ok=ok(b), package_answer=p.get('answer'), package_ok=ok(p))
        if k[0] == 'recall':
            row.update(package_path=p.get('path'), mb=p.get('mb'), bare_w20=b.get('w20'), package_w20=p.get('w20'))
        out.write(json.dumps(row, ensure_ascii=False) + '\n'); n += 1
    print(f'{n} pairs -> {out.name}')


if __name__ == '__main__':
    main(sys.argv[1])

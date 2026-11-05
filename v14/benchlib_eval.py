"""BENCH-LIB (PRE-REGISTERED 2026-11-05 14:40 IST, EXPERIMENTS_V14): the library arm on ARC-e / ARC-c / HellaSwag / PIQA, per item.

Step 1 (CPU, once per library; every body reads the SAME passages):
    python v14/benchlib_eval.py retrieve --lib <lib_dir>            -> <lib_dir>/retrieved_<task>.jsonl
  query = the question / context text + all choices; BM25 top-3 (v14/arclib_index.py).
Step 2 (GPU, per body):
    python v14/benchlib_eval.py score --lib <lib_dir> --ckpt <weights> --tag c3b --out logs/v14_benchlib_<tag>_<hash>_<stamp>.jsonl
                                      [--suite_log <pkg_suite jsonl>]   (merges the rows into that milestone's pairs file)
  Arms per item, all on mc_eval's own scorer (lm-eval 0.4.9.1 prompts, " " + choice continuations):
    bare     = the prompt as is (no library);
    library  = "<p1>\\n<p2>\\n<p3>\\n\\n" + the prompt;
    oracle   = "The answer is <gold choice>.\\n\\n" + the prompt (ARC only; an upper bound on using a passage).
  ok = acc_norm (argmax log-likelihood / characters; lm-eval's headline for these four); acc kept too.
Pairs rows: cell arc_e / arc_c / hellaswag / piqa, context = the 3 passages, question = the prompt, gold = the gold choice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

TASKS = {'arc_easy': 'arc_e', 'arc_challenge': 'arc_c', 'hellaswag': 'hellaswag', 'piqa': 'piqa'}


def query_text(prompt, choices):
    q = prompt.replace('Question:', ' ').replace('\nAnswer:', ' ')
    return ' '.join((q + ' ' + ' '.join(choices)).split())


def retrieve(a):
    from mc_eval import load_task
    from v14.arclib_index import BM25
    t0 = time.time(); bm = BM25(a.lib); print(f'[retrieve] index loaded ({time.time() - t0:.0f} s)', flush=True)
    for task in a.tasks.split(','):
        out = Path(a.lib) / f'retrieved_{task}.jsonl'
        n = 0
        with open(out, 'w') as f:
            for i, (prompt, choices, gold) in enumerate(load_task(task)):
                if a.limit and i >= a.limit:
                    break
                n += 1
                top = bm.top(query_text(prompt, choices), 3)
                f.write(json.dumps(dict(i=i, passages=[t for _, _, t in top], ids=[j for j, _, _ in top],
                                        source=[int(bm.src[j]) for j, _, _ in top], score=[round(s, 2) for _, s, _ in top])) + '\n')
        print(f'[retrieve] {task}: {n} items -> {out} ({time.time() - t0:.0f} s)', flush=True)


def score(a):
    import torch
    from mc_eval import Scorer, load_task
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='BENCH-LIB', hash=h, args=vars(a))), flush=True)
    meta = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    base = dict(tag=a.tag, step=meta.get('global_step'), tokens=meta.get('global_tokens'), hash=h)
    sc = Scorer(a.ckpt, device=a.device, batch_size=a.batch_size)
    lib_meta = json.load(open(Path(a.lib) / 'meta.json'))
    out = open(a.out, 'w')
    for task in a.tasks.split(','):
        t0 = time.time()
        ret = [json.loads(l) for l in open(Path(a.lib) / f'retrieved_{task}.jsonl')]
        items = list(load_task(task))[:len(ret)]
        arms = ['bare', 'library'] + (['oracle'] if task.startswith('arc') else [])
        pref = {'bare': lambda r, ch, g: '', 'library': lambda r, ch, g: ''.join(p + '\n' for p in r['passages']) + '\n',
                'oracle': lambda r, ch, g: f'The answer is {ch[g]}.\n\n'}
        res = {}
        for arm in arms:
            reqs, spans = [], []
            for (prompt, choices, gold), r in zip(items, ret):
                ctx = pref[arm](r, choices, gold) + prompt
                spans.append((len(reqs), len(choices))); reqs += [(ctx, ' ' + c) for c in choices]
            ll = sc.loglik(reqs)
            res[arm] = []
            for (s, n), (_, choices, gold) in zip(spans, items):
                v = ll[s:s + n]; vn = [x / len(c) for x, c in zip(v, choices)]
                res[arm].append((max(range(n), key=lambda k: vn[k]), max(range(n), key=lambda k: v[k])))
            print(f'[score] {task} {arm}: acc_norm {sum(p == g for (p, _), (_, _, g) in zip(res[arm], items)) / len(items):.4f} '
                  f'({time.time() - t0:.0f} s)', flush=True)
        for i, ((prompt, choices, gold), r) in enumerate(zip(items, ret)):
            row = dict(base, cell=TASKS[task], task=task, L=None, doc=i, context='\n'.join(r['passages']), question=prompt,
                       gold=choices[gold], choices=choices, library=dict(source=[lib_meta['source_ids'][s] for s in r['source']],
                                                                         bm25=r['score']))
            for arm, key in (('bare', 'bare'), ('library', 'package'), ('oracle', 'oracle')):
                if arm in res:
                    pn, pa = res[arm][i]
                    row.update({f'{key}_answer': choices[pn], f'{key}_ok': int(pn == gold), f'{key}_acc': int(pa == gold)})
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
        out.flush()
    out.close()
    if a.suite_log:                                         # the milestone's pairs file (same merge as pkg_pairs_more.py)
        from v14.pkg_pairs_more import merge
        more = a.suite_log.replace('.jsonl', '_pairsx.jsonl'); cells = [TASKS[t] for t in a.tasks.split(',')]
        rest = [l for l in open(more) if json.loads(l)['cell'] not in cells] if Path(more).exists() else []
        open(more, 'w').writelines(rest + open(a.out).readlines())
        k, n = merge(a.suite_log.replace('.jsonl', '_pairs.jsonl'), more, cells)
        print(f'merged: {k} kept + {n} new rows -> pairs', flush=True)
    print('BENCH-LIB DONE', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['retrieve', 'score']); ap.add_argument('--lib', required=True)
    ap.add_argument('--tasks', default='arc_easy,arc_challenge,hellaswag,piqa'); ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--ckpt'); ap.add_argument('--tag'); ap.add_argument('--out'); ap.add_argument('--suite_log', default=None)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--batch_size', type=int, default=32)
    a = ap.parse_args()
    retrieve(a) if a.step == 'retrieve' else score(a)


if __name__ == '__main__':
    main()

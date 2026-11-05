"""Three more per-item "model alone" vs "model + library" cells for the review page (Lead, user 2026-11-05 14:35 IST), in
pkg_suite_pairs.py's row format, on the EXACT items the page's probes score (v16/humaneval/run.sh):
  s8      run_learning_probe --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000
          --spread 0,2048,4096 (2,400 items). Package: the session (examples spread through filler, query cut off) is read into
          the state, the library re-shows the header + 8 examples right before the query (placed by the harness, as MLB-1's
          skill pack v1; fetching a pack by key is R x K's path), read = the package's (v0.3 warm; was warm + contrast to 2026-11-05).
  needle  run_memory_behavioral --context-lengths 128..8192 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40
          (seed 1000; 2,520 items). Package: v16/package.py answer's text fetch of the record row, read = the package's (v0.3 warm);
          when the library fetches nothing the package answers as the model alone (path None, labelled).
  skill   run_skillpack_probe --variants skillpack --lengths 0,2048,4096,8192 --docs 100 (seed 8000; 800 items). The probe's
          "re-read" arm is pack + query in a FRESH context (no session), so it is not the package's read; the package arm here
          is the s8 read: session state (pack + filler), pack re-shown before the query, the package's warm read.
Bare = the model alone on the session + query (the same last-position logits as the probe's whole-prompt forward).
Both arms: pick among the probe's candidates (ok) and free (argmax over the whole vocab == the answer).
Writes <suite_log>_pairsx.jsonl and merges its cells into <suite_log>_pairs.jsonl (pkg_suite_pairs.py merges it back too).
    python v14/pkg_pairs_more.py --ckpt <weights> --suite_log logs/v14_pkg_suite_<tag>_<hash>_<stamp>.jsonl [--cells s8,needle,skill]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')):
    if p not in sys.path:
        sys.path.insert(0, p)
from memory_probes.behavioral import build_suite as needle_suite  # noqa: E402
from memory_probes.spread import _SEP, filler_tokens, split_units, spread_suite  # noqa: E402
from run_skillpack_probe import filler_block  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402
from v16.package import Package  # noqa: E402

PROBE_FILLER = str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
S8_TASKS, SKILL_TASKS = ('shift_wide_new', 'affine_map_new'), ('shift_wide_new', 'affine_map_new')
SEEDS = dict(s8=range(7000, 7400), needle=range(1000, 1040), skill=range(8000, 8100))          # the page's probe items
HELDOUT = dict(s8=range(7400, 7500), needle=range(2000, 2020), skill=range(8100, 8150))       # never on the page: gate calibration


@torch.no_grad()
def arms(pkg, sess, q, shown):
    """(bare, package, warm) logits at the query's last position. bare = state(sess) + q; package = the package's read
    (v0.3: warm, state(sess) + shown + q, no contrast; the retired no-row contrast w + 4 (w - b) is recomputable from the
    saved cand_bare / cand_warm); shown = [] -> the package answers as the model alone."""
    s = pkg.read(sess)
    lb, _, _ = pkg.mind.feed(q, copy.deepcopy(s['state']), s['off'])
    if not shown:
        return lb, lb, lb
    la, _, _ = pkg.mind.feed(shown + q, copy.deepcopy(s['state']), s['off'])
    return lb, la, la


def judge(tok, logits, cand, target):
    c = logits[cand].float(); pick = cand[int(c.argmax())]
    return tok.decode([pick]).strip(), int(pick == target), int(int(logits.argmax()) == target)


def row(base, cell, task, L, doc, context, question, gold, tok, lb, lp, cand, target, la=None, **extra):
    """la = the warm read (shown + query, no contrast); its candidate logits and the bare ones are saved so other reads
    (warm alone, gated by agreement or by the bare read's confidence) are computed offline: contrast = w + 4 (w - b)."""
    ba, bok, bfree = judge(tok, lb, cand, target); pa, pok, pfree = judge(tok, lp, cand, target)
    r4 = lambda x: [round(float(v), 4) for v in x[cand].float().tolist()]  # noqa: E731
    return dict(base, cell=cell, task=task, L=L, doc=doc, context=context, question=question, gold=gold,
                candidates=[tok.decode([c]).strip() for c in cand], bare_answer=ba, bare_ok=bok, bare_free=bfree,
                package_answer=pa, package_ok=pok, package_free=pfree, gold_index=cand.index(target),
                cand_bare=r4(lb), cand_warm=r4(la if la is not None else lb), **extra)


def cell_s8(pkg, base, out, limit=None):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    exs = [b2_example(tok, t, 8, sd) for t in S8_TASKS for sd in SEEDS['s8']]
    for sx in spread_suite(exs, tok, (0, 2048, 4096), filler_tokens(PROBE_FILLER))[:limit]:
        task, L = sx.task.split('@'); orig = next(e for e in exs if e.task == task and e.seed == sx.seed)
        header, units = split_units(orig); q = enc(units[-1]); shown = enc(header) + sum((enc(u) for u in units[:-1]), [])
        sess = sx.prompt_ids[:-len(q)]
        assert sx.prompt_ids[-len(q):] == q
        lb, lp, la = arms(pkg, sess, q, shown)
        out.write(json.dumps(row(base, 's8', task, int(L), sx.seed, header + ''.join(units[:-1]) +
                                 ('' if L == '0' else f'\n[the 8 examples were spread through {L} tokens of text]'),
                                 units[-1].strip(), orig.target_text.strip(), tok, lb, lp, list(sx.candidate_token_ids),
                                 sx.target_token_id, la=la, package_path='examples re-shown before the question')) + '\n')


def cell_needle(pkg, base, out, limit=None):
    tok = pkg.tok
    exs = needle_suite(tok, context_lengths=(128, 256, 512, 1024, 2048, 4096, 8192), positions=(0, 0.5, 1),
                       association_counts=(1, 4, 8), seeds=tuple(SEEDS['needle']), candidate_count=8)
    for ex in exs[:limit]:
        key = ex.prompt_text.rsplit('Memory query:', 1)[1].split()[0]
        q = tok.encode(f'\nMemory query: {key} means', add_special_tokens=False)
        assert ex.prompt_ids[-len(q):] == q
        sess = ex.prompt_ids[:-len(q)]
        _, info = pkg.answer(sess, tok.decode(q))
        shown = sess[info['row'][0]:info['row'][1]] if info.get('row') else []
        lb, lp, la = arms(pkg, sess, q, shown)
        recs = '\n'.join(l for l in ex.prompt_text.split('\n') if l.startswith('Memory record'))
        out.write(json.dumps(row(base, 'needle', f'n{ex.associations}_pos{ex.target_position:g}', ex.context_tokens, ex.seed,
                                 recs + f'\n[the records sit at position {ex.target_position:g} of {ex.context_tokens} tokens]',
                                 f'Memory query: {key} means', ex.target_text, tok, lb, lp, list(ex.candidate_token_ids),
                                 ex.target_token_id, la=la, package_path=info.get('path'),
                                 package_row=tok.decode(shown).strip() if shown else None)) + '\n')


def cell_skill(pkg, base, out, limit=None):
    tok = pkg.tok; enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    filler = filler_tokens(PROBE_FILLER); sep = enc(_SEP)
    for task in SKILL_TASKS:
        for sd in (SEEDS['skill'] if limit is None else SEEDS['skill'][:max(1, limit // 8)]):
            ex = b2_example(tok, task, 8, sd); header, units = split_units(ex)
            pack = enc(header) + sum((enc(u) for u in units[:-1]), []); q = enc(units[-1])
            for L in (0, 2048, 4096, 8192):
                fill = sep + filler_block(filler, L, sd, L) + sep if L else []
                lb, lp, la = arms(pkg, pack + fill, q, pack)
                out.write(json.dumps(row(base, 'skill', task, L, sd, header + ''.join(units[:-1]) +
                                         (f'\n[then {L} tokens of unrelated text]' if L else ''), units[-1].strip(),
                                         ex.target_text.strip(), tok, lb, lp, list(ex.candidate_token_ids), ex.target_token_id, la=la,
                                         package_path='pack re-shown before the question (session state kept)')) + '\n')


def merge(pairs_path, more_path, cells):
    """drop these cells' rows from the pairs file and append the new ones (either script may run first)."""
    keep = [l for l in open(pairs_path) if json.loads(l)['cell'] not in cells] if Path(pairs_path).exists() else []
    new = [l for l in open(more_path) if json.loads(l)['cell'] in cells]
    with open(pairs_path, 'w') as f:
        f.writelines(keep + new)
    return len(keep), len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--suite_log', required=True)
    ap.add_argument('--cells', default='s8,needle,skill'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=None, help='smoke: first N items per cell')
    ap.add_argument('--heldout', action='store_true', help='HELDOUT seeds; rows go to --out only (never merged into the pairs)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import subprocess
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='PKG pairs more (s8 / needle / skill)', hash=h, args=vars(a))), flush=True)
    pkg = Package(a.ckpt, device=a.device)
    meta = torch.load(a.ckpt, map_location='cpu', weights_only=False, mmap=True)
    suite0 = json.loads(open(a.suite_log).readline())
    base = dict(tag=suite0.get('tag'), step=meta.get('global_step'), tokens=meta.get('global_tokens'), hash=h)
    more = a.suite_log.replace('.jsonl', '_pairsx.jsonl'); cells = a.cells.split(',')
    if a.heldout:                                        # calibration items: their own file, never the page's pairs
        assert a.out, '--heldout needs --out'
        SEEDS.update(HELDOUT); more = a.out; base['tag'] = f"{base['tag']}_heldout"
    if Path(more).exists():                              # a rerun replaces these cells
        rest = [l for l in open(more) if json.loads(l)['cell'] not in cells]
        open(more, 'w').writelines(rest)
    with open(more, 'a') as out:
        for cell in cells:
            t0 = time.time()
            {'s8': cell_s8, 'needle': cell_needle, 'skill': cell_skill}[cell](pkg, base, out, a.limit); out.flush()
            n = [json.loads(l) for l in open(more) if json.loads(l)['cell'] == cell]
            print(f"{cell}: {len(n)} items, bare {sum(r['bare_ok'] for r in n) / len(n):.3f}, package "
                  f"{sum(r['package_ok'] for r in n) / len(n):.3f} ({time.time() - t0:.0f} s)", flush=True)
    if not a.heldout:
        k, n = merge(a.suite_log.replace('.jsonl', '_pairs.jsonl'), more, cells)
        print(f'merged: {k} kept + {n} new rows -> pairs', flush=True)


if __name__ == '__main__':
    main()

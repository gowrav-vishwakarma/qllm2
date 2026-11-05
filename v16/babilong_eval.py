"""BABILong entry (user, Nov 1: participate in known benchmarks; cap at 128k). Dataset RMT-team/babilong (HF): configs by
length ('0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k'), splits qa1..qa10 (100 rows each), fields input /
question / target. Two columns per body: AS IS (the body reads the whole input, greedy answer) and, for PAM minds,
the PACKAGE (v16/package.py: stream the input into the state, fetch the supporting facts by text / surprise keys, warm +
contrast read). Scoring = BABILong's: the target string appears in the generated answer (case-insensitive). No tuning
on the benchmark; the "as is / fine-tuned" column of the leaderboard entries is stated in the write-up.
Usage: python v16/babilong_eval.py --ckpt <path> --arch pam|transformer|mamba2 --lengths 0k,4k,32k --tasks qa1,qa2,qa3,qa4,qa5
       --docs 100 --device cuda [--package] [--mode lenient|strict]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'v16'))
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from bench import Reader  # noqa: E402

NL = 198
PROMPT = "\n\nQuestion: {q}\nAnswer:"          # the plain exam form; the chat form is a second pass if wanted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arch', default='pam', choices=['pam', 'transformer', 'mamba2'])
    ap.add_argument('--lengths', default='0k,1k,2k,4k,8k,16k,32k'); ap.add_argument('--tasks', default='qa1,qa2,qa3,qa4,qa5')
    ap.add_argument('--docs', type=int, default=100); ap.add_argument('--device', default='cuda'); ap.add_argument('--max_new', type=int, default=16)
    ap.add_argument('--package', action='store_true'); ap.add_argument('--mode', default='lenient', choices=['lenient', 'strict']); ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--skip_asis', action='store_true', help='package column only (no full-input forward of the body as is)')
    ap.add_argument('--chain', action='store_true', help='also the PLANNER-CHAIN column pkg_chain (Library: the planner composes '
                    'object -> holder -> sub-question; labelled TEXT in PACKAGE.md 3c; the mind-composes pkg column stays the headline)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    model, tok, cfg = _load_v13_sempty(Path(a.ckpt), 'baseline_real_pm', torch.device(a.device)); model.eval()
    R = Reader(model, a.arch)
    pkg = None
    if a.package:
        from v16.package import Package
        pkg = Package(a.ckpt, device=a.device)
    name = Path(a.ckpt).parent.name; ts = time.strftime('%Y%m%d_%H%M')
    out = Path(a.out) if a.out else ROOT / 'logs' / f'v16_babilong_{name}_{ts}.jsonl'
    f = open(out, 'a')
    summary = {}
    for L in a.lengths.split(','):
        ds = load_dataset('RMT-team/babilong', L)
        for task in a.tasks.split(','):
            rows = ds[task]
            n = min(a.docs, len(rows)); hit_asis = hit_pkg = 0; t0 = time.time()
            for i in range(n):
                ex = rows[i]; text = ex['input']; q = ex['question'].strip(); tgt = ex['target'].strip()
                rec = dict(arch=a.arch, ckpt=a.ckpt, L=L, task=task, doc=i, target=tgt, question=q, n_tokens=None)
                try:
                    if not a.skip_asis:
                        ids = tok.encode(text + PROMPT.format(q=q), add_special_tokens=False); rec['n_tokens'] = len(ids)
                        gen = tok.decode(R.greedy(ids, max_new=a.max_new, stops=(NL,))).strip()
                        rec['asis'] = gen; rec['asis_ok'] = int(tgt.lower() in gen.lower()); hit_asis += rec['asis_ok']
                    if pkg is not None:                       # Library 80e678c: top-k rows (name / latest / F2 + hops by name) read together
                        rows_k = pkg.fetch_rows(text, q, k=a.k)
                        ptxt, pinfo = pkg.answer_rows(text, PROMPT.format(q=q), k=a.k, mode=a.mode) if rows_k else ('', {'abstain': 1})
                        ptxt = (ptxt or '').strip(); rec['pkg'] = ptxt; rec['pkg_ok'] = int(tgt.lower() in ptxt.lower()); hit_pkg += rec['pkg_ok']
                        rec['pkg_rows'] = [(r[2], r[3][:100]) for r in rows_k]; rec['pkg_abstain'] = pinfo.get('abstain')
                        s_ = pkg._session                         # MB held (BENCH-PKG): mind state + the library's token text
                        rec['mb'] = dict(mind_state=s_['state_bytes'], library_text=2 * len(s_['ids'])) if s_ else None
                        if a.chain:                               # Library: planner chain (TEXT), a second package column
                            ctxt, cinfo = pkg.answer_chain(text, q, PROMPT, k=a.k, mode=a.mode)
                            ctxt = (ctxt or '').strip(); rec['pkg_chain'] = ctxt; rec['pkg_chain_ok'] = int(tgt.lower() in ctxt.lower())
                            rec['pkg_chain_plan'] = cinfo.get('chain')
                except Exception as e:
                    rec['error'] = f'{type(e).__name__}: {str(e)[:160]}'
                f.write(json.dumps(rec) + '\n'); f.flush()
            summary[(L, task)] = (hit_asis / n, hit_pkg / n if pkg else None)
            print(f"{L:5s} {task:4s} n={n} as_is={hit_asis / n:.2f}" + (f" package={hit_pkg / n:.2f}" if pkg else '') + f" ({(time.time() - t0) / n:.1f} s/doc)", flush=True)
    print('out', out)


if __name__ == '__main__':
    main()

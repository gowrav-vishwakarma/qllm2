"""A22 (registered 2026-11-01 12:39 IST): the PACKAGE on the user's story test (Lead's v16/story_eval.py), outputs shown.

Same story prompts (build_prompt, seed 1) and the 11 questions in chat and plain forms. Per question: fetch = text steps
(token organ, word organ 1-2, step 4) then F2 (the mind's 2 most surprising question tokens, block-8 max-sim over the
story's sentences); read = b2p2 warm + no-row contrast, up to 40 tokens, stops as story_eval; STRICT twin = text steps
only (abstain on a miss); b2p2 ALONE = warm, no row, plain greedy. Report: logs/v14_story_package_<hash>_<stamp>.md.

    python v14/a22_story.py --lengths 2048,4096,8192 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v16')]
from memory_probes.spread import filler_tokens  # noqa: E402
from story_eval import IM_END, IM_START, NL, QUESTIONS, build_prompt  # noqa: E402
from v16.bench import Reader  # noqa: E402
from v14.a6_reread import CKPT, Mind  # noqa: E402
from v14.a13_manykey import capture, maxsim_scores, sentence_rows  # noqa: E402
from v14.package_fetch import text_fetch  # noqa: E402

ALPHA = 4.0


@torch.no_grad()
def decode(mind, prefix, st, off, stops, max_new, twin=None):
    """greedy from a copy of the doc state; with twin (the no-row prefix), the contrast l + a (l - l_twin)."""
    la, sa, oa = mind.feed(prefix, copy.deepcopy(st), off)
    if twin is not None:
        lb, sb, ob = mind.feed(twin, copy.deepcopy(st), off)
    out = []
    for _ in range(max_new):
        lg = la if twin is None else la + ALPHA * (la - lb)
        t = int(lg.argmax()); out.append(t)
        if t in stops:
            break
        la, sa, oa = mind.feed([t], sa, oa)
        if twin is not None:
            lb, sb, ob = mind.feed([t], sb, ob)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', default='2048,4096,8192'); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max_new', type=int, default=40)
    ap.add_argument('--forms', default='chat,plain')
    a = ap.parse_args()
    dev = torch.device(a.device)
    mind = Mind(CKPT, dev); tok = mind.tok; R = Reader(mind.m, 'pam')
    fill = filler_tokens(str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    im_end = tok.convert_tokens_to_ids(IM_END) if IM_END in tok.get_vocab() else None
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rep = Path('/home/gowrav/Development/qllm2/logs') / f"v14_story_package_{h}_{time.strftime('%Y%m%d_%H%M')}.md"
    out = [f"# Story comprehension: the PACKAGE (text fetch + F2, warm + contrast read) vs b2p2 ALONE, seed {a.seed}\n",
           "b2p2 (102.9M, state 1.15 MB). Fetch path: text0 = token organ, text1-2 = word organ, text4 = rare-in-English "
           "word, F2 = the mind's surprise-keyed address. Strict = text fetch only (abstain on a miss). Keyword scorer as "
           "v16/story_eval.py.\n"]
    for L in [int(x) for x in a.lengths.split(',')]:
        text = build_prompt(tok, fill, L, a.seed)
        pf = ROOT / 'v16' / 'prompts' / f'story_L{L}_s{a.seed}.txt'
        same = pf.exists() and pf.read_text() == text
        base = tok.encode(text, add_special_tokens=False); end = len(base)
        _, st, off = mind.feed(base)
        rows = sentence_rows(base, end)
        out.append(f"\n## L = {L} (prompt {end} tokens; identical to {pf.name}: {same})\n\n"
                   "| question | form | expected | path | fetched row | package (warm + contrast) | strict | b2p2 alone |\n"
                   "|---|---|---|---|---|---|---|---|")
        hits = {k: 0 for k in ('package', 'strict', 'alone')}
        for qname, q, exp in QUESTIONS:
            for form in a.forms.split(','):
                if form == 'chat':
                    qt = f"\n\n{IM_START}user\n{q}{IM_END}\n{IM_START}assistant\n"; stops = tuple(t for t in (NL, im_end) if t is not None)
                else:
                    qt = f"\n\nQuestion: {q}\nAnswer:"; stops = (NL,)
                qi = tok.encode(qt, add_special_tokens=False); ids = base + qi
                row, info = text_fetch(ids, end, step4=True, bigram=False)
                path = f"text{info['step']}" if row is not None else None
                if row is None:
                    got, pre = capture(R, ids, (8,))
                    surp = R.surprisal(ids, pre.cpu()); qs = surp[end:].clone(); qs[:2] = -1.0
                    sal = [end + int(i) for i in qs.topk(2).indices]
                    v = got[8]; V = torch.nn.functional.normalize(v - v[:end].mean(0), dim=-1)
                    row = rows[int(maxsim_scores(V[:end], V[sal], rows).argmax())]; path = 'F2'
                span = base[row[0]:row[1]]
                pkg = tok.decode(decode(mind, span + qi, st, off, stops, a.max_new, twin=qi))
                alone = tok.decode(decode(mind, qi, st, off, stops, a.max_new))
                strict = pkg if path != 'F2' else '[abstain: no text match]'
                res = {'package': pkg, 'strict': strict, 'alone': alone}
                for k, txt in res.items():
                    hits[k] += int(exp.lower() in txt.lower())
                cell = lambda t: ('OK ' if exp.lower() in t.lower() else '-- ') + '`' + t.replace('\n', ' / ').replace('|', '\\|').strip()[:90] + '`'
                out.append(f"| {qname} | {form} | {exp} | {path} | `{tok.decode(span).replace(chr(10), ' ').replace('|', '/').strip()[-80:]}` | "
                           f"{cell(pkg)} | {cell(strict) if path != 'F2' else 'abstain'} | {cell(alone)} |")
                print(f"L={L} {qname:13s} {form:5s} {path:5s} pkg {'OK' if exp.lower() in pkg.lower() else '--'} {pkg[:60]!r} | alone {alone[:40]!r}", flush=True)
        n = len(QUESTIONS) * len(a.forms.split(','))
        out.append(f"\nkeyword hits (of {n}): package {hits['package']}, strict {hits['strict']}, b2p2 alone {hits['alone']}")
        print(f"L={L} hits package {hits['package']} strict {hits['strict']} alone {hits['alone']} of {n}", flush=True)
        rep.write_text('\n'.join(out) + '\n')
    print('report', rep)


if __name__ == '__main__':
    main()

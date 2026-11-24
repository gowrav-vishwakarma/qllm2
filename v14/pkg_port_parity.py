"""PKG-PORT parity: the standalone HF-card package (hf_release_*/package_lite.py on modeling_pam2.py) vs the repo's
v16/package.py, on the same checkpoint (Lead 2026-11-24, part (3) of the user's GO). Same docs as the measured cells:
  RULER-style niah_single / niah_multikey at 4k and 8k (v14/ruler_pkg.build, seed 28000 = the BENCH-PKG docs), 3 each;
  PKG-SUITE recall kv / kvupd at 8k (v16/bench.build, seed 27000 stream), 2 each;
  NATURAL RECALL prose at 2k, indirect + synonym questions (v14/natural_recall.make_doc), 2 each.
Per item, both sides: BARE answer (state + question, greedy, 24 tokens), PACKAGE answer + fetch path + row, and the F2 row
(forced, so the address capture is checked on every item), and the state bytes. Pass = every field equal on every item.
    python v14/pkg_port_parity.py --ckpt <repo ckpt> --release_dir hf_release_sci1 --release_ckpt hf_release_sci1/sci1_386m_49b.pt \
        --out logs/v14_pkg_port_parity_<tag>_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
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
from v14.a26_protocol import FILL  # noqa: E402
from v14.natural_recall import make_doc as nat_doc  # noqa: E402
from v14.ruler_pkg import build as ruler_build  # noqa: E402
from v16.bench import build as bench_build  # noqa: E402
from v16.package import Package as RepoPackage  # noqa: E402


def items(tok, filler):
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    out = []
    for L in (4096, 8192):
        for ti, task in enumerate(('niah_single', 'niah_multikey')):
            rng = random.Random(28000 * 1000 + L + ti)
            for di in range(3):
                sess, q, gold = ruler_build(rng, tok, filler, L, task)
                out.append(dict(cell=f'ruler_{task}', L=L, doc=di, sess=sess, q=q, gold=gold['value']))
    for ti, task in enumerate(('kv', 'kvupd')):
        rng = random.Random(27000 * 1000 + 8192 + ti)
        for di in range(2):
            ids, _, qlen, meta = bench_build(rng, tok, filler, 8192, task, 8)
            out.append(dict(cell=f'recall_{task}', L=8192, doc=di, sess=ids[:-qlen], q=tok.decode(ids[-qlen:]), gold=str(meta['val'])))
    for si, style in ((2, 'indirect'), (3, 'synonym')):
        rng = random.Random(19000 * 1000 + 2048 * 10 + si)
        for di in range(2):
            d = nat_doc(rng, enc, filler, 2048, style)
            qlen = len(enc('\n\n' + d['question']))
            out.append(dict(cell=f'natrec_{style}', L=2048, doc=di, sess=d['ids'][:-qlen], q='\n\n' + d['question'], gold=d['gold']))
    return out


@torch.no_grad()
def repo_side(pkg, it, max_new):
    s = pkg.read(it['sess']); q = pkg.tok.encode(it['q'], add_special_tokens=False)
    la, sa, oa = pkg.mind.feed(q, copy.deepcopy(s['state']), s['off'])
    bare = pkg.tok.decode(pkg._gen(la, sa, oa, None, None, None, 0.0, max_new))
    text, info = pkg.answer(it['sess'], it['q'])
    f2row, sal = pkg._f2(q)
    return dict(bare=bare, package=text, path=info['path'], row=list(info['row']) if info['row'] else None,
                f2_row=list(f2row), f2_salient=sal, state_bytes=s['state_bytes'])


@torch.no_grad()
def port_side(pkg, it, max_new):
    s = pkg.read(it['sess'])
    bare = pkg.bare(it['sess'], it['q'], max_new=max_new)
    text, info = pkg.answer(it['sess'], it['q'])
    f2row, sal = pkg._f2(pkg.encode(it['q']))
    return dict(bare=bare, package=text, path=info['path'], row=list(info['row']) if info['row'] else None,
                f2_row=list(f2row), f2_salient=sal, state_bytes=s['state_bytes'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--release_dir', required=True); ap.add_argument('--release_ckpt', required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max_new', type=int, default=24); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('HEADER ' + json.dumps(dict(unit='PKG-PORT parity', hash=h, args=vars(a))), flush=True)
    repo = RepoPackage(a.ckpt, device=a.device)
    sys.path.insert(0, str(Path(a.release_dir).resolve()))
    from modeling_pam2 import load_model, load_tokenizer
    from package_lite import Package as PortPackage
    tok2 = load_tokenizer()
    port = PortPackage(load_model(a.release_ckpt, device=a.device), tok2)
    assert [repo.tok.encode(x) for x in ('Hello world.', '<|im_end|> 3.14')] == [tok2.encode(x) for x in ('Hello world.', '<|im_end|> 3.14')]
    its = items(repo.tok, filler_tokens(str(FILL)))
    out = open(a.out, 'w'); n_ok = 0; keys = ('bare', 'package', 'path', 'row', 'f2_row', 'f2_salient', 'state_bytes')
    for it in its:
        t0 = time.time()
        r = repo_side(repo, it, a.max_new); p = port_side(port, it, a.max_new)
        diff = [k for k in keys if r[k] != p[k]]
        n_ok += int(not diff)
        rec = dict(cell=it['cell'], L=it['L'], doc=it['doc'], gold=it['gold'], equal=int(not diff), diff=diff, repo=r, port=p,
                   gold_in_bare=int(it['gold'].lower() in r['bare'].lower()), gold_in_package=int(it['gold'].lower() in r['package'].lower()),
                   sec=round(time.time() - t0, 1), hash=h)
        out.write(json.dumps(rec) + '\n'); out.flush()
        print(f"{it['cell']:22s} L {it['L']:5d} doc {it['doc']} {'EQUAL' if not diff else 'DIFF ' + str(diff)} path {r['path']} "
              f"bare {r['bare'].strip()[:40]!r} package {r['package'].strip()[:40]!r} gold {it['gold']!r}", flush=True)
    print(f'PARITY {n_ok} / {len(its)} items equal on every field ({", ".join(keys)})', flush=True)
    print('PORT-PARITY DONE', flush=True)


if __name__ == '__main__':
    main()

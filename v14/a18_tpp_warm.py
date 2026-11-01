"""A18 (pre-registered 2026-11-01 09:49 IST): "T++ + library" warm rows, a comparison for Lead's text-library arm.

Every body reads the SAME text-only library row (package_fetch.text_fetch, strict). T++ (tfc1b): COLD (row + question),
WINDOW (last 2,048 session tokens + row + question, contrast twin = window + question), FULL (whole session + row +
question with its twin; first --full_docs docs per cell), WINDOW without contrast. PAM b2p2: warm + no-row contrast.
T++ has no KV cache in this class, so decoding re-runs the context (bench Reader.last_logits: bf16, flash attention);
the rows state the DEPLOYED cost: bytes held (KV / state) and tokens run per answer.

    python v14/a18_tpp_warm.py --out logs/v14_a18_tpp_warm_<hash>_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.spread import filler_tokens  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from v16.bench import Reader, build  # noqa: E402
from v14.a6_reread import CKPT, FILL, Mind, STOPS, sent_ok  # noqa: E402
from v14.package_fetch import text_fetch  # noqa: E402

TPP = '/home/gowrav/Development/qllm2/checkpoints_v13_sempty/tfc1b_main_b788731/best_model.pt'
CELLS = 'kv/synthetic,kv/natural,para/synthetic,para/natural,tool/synthetic,tool/natural,natq/natural'
WINDOW, ALPHA, MAX_NEW = 2048, 4.0, 16


def tpp_greedy(R, a, b=None, alpha=ALPHA):
    """greedy over T++ last logits; with b, contrast l_a + alpha (l_a - l_b), both streams fed the output."""
    out = []
    for _ in range(MAX_NEW):
        la = R.last_logits(a + out).float()
        lg = la if b is None else la + alpha * (la - R.last_logits(b + out).float())
        t = int(lg.argmax()); out.append(t)
        if t in STOPS:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=18000); ap.add_argument('--L', type=int, default=32768)
    ap.add_argument('--docs', type=int, default=40); ap.add_argument('--full_docs', type=int, default=20)
    ap.add_argument('--needles', type=int, default=8); ap.add_argument('--cells', default=CELLS)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    tm, tok, tcfg = _load_v13_sempty(Path(TPP), 'baseline_real_pm', dev); tm.eval()
    R = Reader(tm, 'transformer')
    kv_per_tok = 2 * tcfg.n_layer * tcfg.d_model * 2                       # K and V, fp16
    pam = Mind(CKPT, dev)
    fill = filler_tokens(FILL)
    filler = fill if a.L + 2 < len(fill) else np.concatenate([fill] * (a.L // len(fill) + 2))
    outf = open(a.out, 'a')
    for cell in a.cells.split(','):
        task, keys = cell.split('/')
        rng = random.Random(a.seed * 1000 + a.L + zlib.crc32(task.encode()) % 997)   # = bench.py main()
        for d in range(a.docs):
            t0 = time.time()
            ids, ans, qlen, meta = build(rng, tok, filler, a.L, task, a.needles, keys=keys)
            end = len(ids) - qlen; q = ids[end:]; val = str(meta['val'])
            row, info = text_fetch(ids, end)
            rec = dict(task=task, keys=keys, seed=a.seed, doc=d, L=a.L, abstain=info['abstain'], step=info['step'],
                       kv_bytes_per_token=kv_per_tok)
            if row is not None:
                span = ids[row[0]:row[1]]
                win = ids[max(0, end - WINDOW):end]
                arms = {'tpp_cold': (span + q, None, 0),
                        'tpp_win': (win + span + q, win + q, len(win)),
                        'tpp_win_plain': (win + span + q, None, len(win))}
                if d < a.full_docs:
                    arms['tpp_full'] = (ids[:end] + span + q, ids[:end] + q, end)
                for name, (sa, sb, ctx) in arms.items():
                    g = tpp_greedy(R, sa, sb)
                    txt = tok.decode(g)
                    rec[name] = sent_ok(txt, val); rec[name + '_text'] = txt
                    rec[name + '_kv_bytes'] = kv_per_tok * (ctx + len(span) + len(q))
                    rec[name + '_answer_tokens'] = (len(span) + len(q) + len(g)) * (2 if sb is not None else 1)
                _, st, off = pam.feed(ids[:end])
                g = pam.greedy_contrast(span + q, copy.deepcopy(st), off, q, copy.deepcopy(st), off, ALPHA)
                txt = tok.decode(g)
                rec['pam_warmc'] = sent_ok(txt, val); rec['pam_warmc_text'] = txt
                rec['pam_state_bytes'] = 1154832; rec['pam_answer_tokens'] = 2 * (len(span) + len(q) + len(g))
            rec['sec'] = round(time.time() - t0, 1)
            outf.write(json.dumps(rec) + '\n'); outf.flush()
            print(cell, d, rec['abstain'], {k: rec.get(k) for k in ('tpp_cold', 'tpp_win', 'tpp_win_plain', 'tpp_full', 'pam_warmc')},
                  rec['sec'], flush=True)


if __name__ == '__main__':
    main()

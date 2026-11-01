"""rxk (EXPERIMENTS_SEMPY "rxk"): a RULE held in state x a FACT fetched at the query. Free primary, 20-way reported.

Streams per doc (pack = B2 header + 8 demos, held-out values; fill = sep + L filler tokens + sep; row = " Record: K | x: v."):
  pack + fill + row + NL + q         the mind (rule in state only, fact in the fetched row)
  fill + row + NL + q                the no-pack stream (contrast twin) = the FLOOR when scored alone
  fill + pack + row + NL + q         the CEILING: the pack re-inserted at the query (RAG + few-shot, no state needed)
q is the trained query line "<v> ->" (explicit: the package substituted the value) or "K ->" (implicit: the mind must take v
from the row). Decodes on the mind stream: plain; contrast l_p + a (l_p - l_n) with a = 4; adaptive a = 4 JSD / ln 2;
echo dilation s = 8 over the fill (both streams) + contrast a = 4, for L >= 4096.
"""
from __future__ import annotations

import argparse, json, math, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from memory_probes.rules import score_candidate_logits  # noqa: E402
from memory_probes.spread import _SEP, filler_tokens, split_units  # noqa: E402
from run_memory_behavioral import _load_v13_sempty  # noqa: E402
from run_skillpack_probe import Body, filler_block  # noqa: E402
from v14.rules_ood import build_example as b2_example  # noqa: E402
from v7.data import _recall_key  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tasks', default='affine_map_new,shift_wide_new')
    p.add_argument('--lengths', default='0,2048,4096,8192')
    p.add_argument('--docs', type=int, default=50)
    p.add_argument('--seed', type=int, default=8800)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mem_frac', type=float, default=0.0)
    p.add_argument('--filler', default=str(ROOT / '.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = torch.device(a.device)
    if a.mem_frac and dev.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    model, tok, _ = _load_v13_sempty(a.checkpoint, 'baseline_real_pm', dev); model.eval(); body = Body(model, dev)
    filler = filler_tokens(a.filler); enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    sep, nl = enc(_SEP), enc('\n')
    out = open(a.out, 'a')

    def logits(ids, dil=None):
        body.m._dilate = dil
        try:
            return body.last_logits(ids).float()
        finally:
            body.m._dilate = None

    def judge(ex, lg):
        return int(int(lg.argmax()) == ex.target_token_id), int(score_candidate_logits(ex, lg[ex.candidate_token_ids].cpu().tolist())['correct'])

    for task in a.tasks.split(','):
        for seed in range(a.seed, a.seed + a.docs):
            ex = b2_example(tok, task, 8, seed)
            header, units = split_units(ex)
            pack = enc(header) + sum((enc(u) for u in units[:-1]), [])
            qtext = units[-1]
            m = re.match(r'^(\s*)(\S+)(\s*->.*)$', qtext, re.S)
            lead, v, tail = m.group(1), m.group(2), m.group(3)
            K = _recall_key(random.Random(seed * 31 + 7))
            field = 'x' if task.startswith('affine') else 'letter'
            row = enc(f' Record: {K} | {field}: {v}.') + nl
            q_exp, q_imp = enc(qtext), enc(f'{lead}{K}{tail}')
            for L in [int(x) for x in a.lengths.split(',')]:
                t0 = time.time()
                fill = sep + filler_block(filler, L, seed, L) + sep if L else []
                rec = {'arch': body.arch, 'ckpt': str(a.checkpoint), 'sub': task, 'doc': seed, 'L': L, 'key': K, 'v': v}
                for name, q in (('exp', q_exp), ('imp', q_imp)):
                    lp = logits(pack + fill + row + q); ln = logits(fill + row + q)
                    rec[f'{name}_plain'] = judge(ex, lp)
                    rec[f'{name}_floor'] = judge(ex, ln)
                    rec[f'{name}_ceiling'] = judge(ex, logits(fill + pack + row + q))
                    rec[f'{name}_c4'] = judge(ex, lp + 4.0 * (lp - ln))
                    P, Q = F.softmax(lp, -1), F.softmax(ln, -1); M = 0.5 * (P + Q)
                    jsd = 0.5 * float((P * (P.clamp_min(1e-12) / M.clamp_min(1e-12)).log()).sum() + (Q * (Q.clamp_min(1e-12) / M.clamp_min(1e-12)).log()).sum())
                    aa = 4.0 * jsd / math.log(2); rec[f'{name}_ada'] = judge(ex, lp + aa * (lp - ln)); rec[f'{name}_ada_a'] = round(aa, 3)
                    if L >= 4096:
                        lpd = logits(pack + fill + row + q, (len(pack), len(pack) + len(fill), 8.0, 'echo'))
                        lnd = logits(fill + row + q, (0, len(fill), 8.0, 'echo'))
                        rec[f'{name}_dil8_c4'] = judge(ex, lpd + 4.0 * (lpd - lnd))
                rec['sec'] = round(time.time() - t0, 2)
                out.write(json.dumps(rec) + '\n'); out.flush()
        print(f'  {task} done', flush=True)


if __name__ == '__main__':
    main()

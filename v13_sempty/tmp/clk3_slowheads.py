"""clk3 (EXPERIMENTS_SEMPY "clk3"): eval-time dilation of ONLY the slow notebook heads (selected by MEASURED realized retention,
not by bias), plus echo, on the trained SCI-1 M9 (c3b). clk1 dilated every notebook head and clk2 whole memories (echo vs
notebook), never a head subset.

1. Realized retention per (notebook layer, head) on held-out text: mean over tokens of ln r_t (r_t = the per-token retention the
   layer actually applies, input-dependent) -> realized half-life h = ln 0.5 / mean ln r. Bias-only half-lives printed beside it.
2. Slow set = the K heads with the longest realized half-life (K = 6 of 48).
3. s8 spread probe (b2 suite, shift_wide_new + affine_map_new, 8 shots, seed 7000, --trials, spread 0 / 2048 / 4096 / 8192) with
   retention r -> r^(1/s) on FILLER tokens only (the knockout mask), for the arms below; echo via the model's own span hook.
4. Prose guard: held-out NLL (the probe's filler cache, n_seq x seq_len) with the same dilation on positions >= 256.
Arms: base | slow_s4 | slow_s8 | all_s4 (every notebook head, clk1-like) | echo_s4 (clk2-like) | slow_echo_s4.
    python v13_sempty/tmp/clk3_slowheads.py --checkpoint <weights> --out <json> [--trials 200] [--K 6]
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'v13_sempty' / 'tmp')]
from memory_probes.rules import aggregate                                  # noqa: E402
from memory_probes.behavioral import score_candidate_logits                # noqa: E402
from memory_probes.spread import filler_tokens                             # noqa: E402
from run_memory_behavioral import _load_v13_sempty, _last_logits           # noqa: E402
from sempyt.tensor import named                                            # noqa: E402
from s8_knockouts import spread_with_mask                                  # noqa: E402

ST = {'mask': None, 'heads': None, 's': 1.0, 'logr': None}


def patch(layers):
    for li, L in enumerate(layers):
        orig = L._decay
        def dec(tokens, _o=orig, _li=li, _L=L):
            ret = _o(tokens); batch, time = tokens.layout[0], tokens.layout[1]
            r = ret.raw(batch, _L.heads, time)                                    # [B, H, T]
            if ST['logr'] is not None:
                ST['logr'][_li].append(torch.log(r.float().clamp_min(1e-12)).mean(dim=(0, 2)).cpu())
            hs = ST['heads']
            if hs is not None and ST['s'] != 1.0 and ST['mask'] is not None:
                sel = torch.zeros(r.shape[1], dtype=torch.bool, device=r.device)
                for (l, h) in hs:
                    if l == _li:
                        sel[h] = True
                if sel.any():
                    m = (ST['mask'].to(r.device)[:, None, :] > 0) & sel[None, :, None]
                    r = torch.where(m, r.float().clamp_min(1e-12) ** (1.0 / ST['s']), r.float()).to(r.dtype)
            return named(r, (batch, _L.heads, time))
        L._decay = dec


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--trials', type=int, default=200); p.add_argument('--K', type=int, default=6); p.add_argument('--seed', type=int, default=7000)
    p.add_argument('--lengths', default='0,2048,4096,8192'); p.add_argument('--n_seq', type=int, default=24); p.add_argument('--seq_len', type=int, default=2048)
    p.add_argument('--filler', default='.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
    a = p.parse_args(); dev = torch.device('cuda'); t0 = time.time()
    model, tok, _ = _load_v13_sempty(Path(a.checkpoint), 'baseline_real_pm', dev); model.eval()
    layers = [b.pam for b in model.blocks if getattr(b, 'pam', None) is not None]; patch(layers)
    H = layers[0].heads.size; filler = filler_tokens(a.filler)
    # 1. realized retention on held-out text
    ST['logr'] = [[] for _ in layers]
    rng = np.random.default_rng(0)
    seqs = [filler[s:s + a.seq_len] for s in rng.integers(0, len(filler) - a.seq_len, size=a.n_seq)]
    with torch.no_grad():
        for sq in seqs:
            model._hidden_to_lm(torch.tensor([sq.tolist()], device=dev))
    heads = []
    for li, L in enumerate(layers):
        lr = torch.stack(ST['logr'][li]).mean(0)
        bias = (L.dt_bias.detach().float() + L.dt_proj.bias.detach().float()).cpu() if getattr(L.dt_proj, 'bias', None) is not None else L.dt_bias.detach().float().cpu()
        for h in range(H):
            real_h = math.log(0.5) / float(lr[h]); bias_h = math.log(0.5) / -float(torch.nn.functional.softplus(bias[h]))
            heads.append(dict(layer=li, head=h, realized_half_life=round(real_h, 1), bias_half_life=round(bias_h, 1)))
    ST['logr'] = None
    slow = sorted(heads, key=lambda d: -d['realized_half_life'])[:a.K]
    slow_set = [(d['layer'], d['head']) for d in slow]; all_set = [(d['layer'], d['head']) for d in heads]
    print('slow heads (realized):', [(d['layer'], d['head'], d['realized_half_life'], d['bias_half_life']) for d in slow], flush=True)
    print('realized half-life median %.1f, max %.1f' % (np.median([d['realized_half_life'] for d in heads]), slow[0]['realized_half_life']), flush=True)
    arms = {'base': (None, 1.0, False), 'slow_s4': (slow_set, 4.0, False), 'slow_s8': (slow_set, 8.0, False),
            'all_s4': (all_set, 4.0, False), 'echo_s4': (None, 4.0, True), 'slow_echo_s4': (slow_set, 4.0, True)}
    # 2. s8 probe items with the filler mask
    from v14.rules_ood import build_example
    tasks, lengths = ('shift_wide_new', 'affine_map_new'), [int(x) for x in a.lengths.split(',')]
    exs = [build_example(tok, t, 8, sd) for t in tasks for sd in range(a.seed, a.seed + a.trials)]
    items = [r for ex in exs for L in lengths for r in [spread_with_mask(ex, tok, L, filler)] if r is not None]
    res = {}
    for arm, (hs, s, echo) in arms.items():
        ST['heads'], ST['s'] = hs, s; rows = []; t1 = time.time()
        with torch.no_grad():
            for ex, mask in items:
                ST['mask'] = torch.tensor([mask], dtype=torch.float32)
                f = [i for i, v in enumerate(mask) if v]
                model._dilate = (f[0], f[-1] + 1, s, 'echo') if (echo and f) else None
                logits = _last_logits('v13_sempty', model, torch.tensor([ex.prompt_ids], device=dev))[0]
                rows.append({**ex.to_dict(), **score_candidate_logits(ex, logits[ex.candidate_token_ids].float().cpu().tolist())})
            # prose guard: dilation on positions >= 256
            nll = []
            for sq in seqs:
                ids = torch.tensor([sq.tolist()], device=dev)
                m = [0] * 256 + [1] * (len(sq) - 256); ST['mask'] = torch.tensor([m], dtype=torch.float32)
                model._dilate = (256, len(sq), s, 'echo') if echo else None
                lg, _, _ = model(ids)
                nll.append(float(torch.nn.functional.cross_entropy(lg[0, :-1].float(), ids[0, 1:])))
        model._dilate = None
        per = {g['task']: g['accuracy'] for g in aggregate(rows)}
        cell = lambda L: float(np.mean([per[f'{t}@{L}'] for t in tasks if f'{t}@{L}' in per])) if any(f'{t}@{L}' in per for t in tasks) else float('nan')
        res[arm] = dict(per_task=per, s8_D=float(np.nanmean([cell(2048), cell(4096)])),
                        D_far=float(np.nanmean([cell(4096), cell(8192)])), nll=float(np.mean(nll)), sec=round(time.time() - t1, 1))
        print(f"{arm:13s} s8 D {res[arm]['s8_D']:.3f}  far(4k+8k) {res[arm]['D_far']:.3f}  @0/2k/4k/8k "
              + '/'.join(f'{cell(L):.3f}' for L in lengths) + f"  NLL {res[arm]['nll']:.4f}", flush=True)
    json.dump(dict(checkpoint=a.checkpoint, heads=heads, slow=slow, arms=res), open(a.out, 'w'), indent=1)
    print(f'clk3 DONE ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()

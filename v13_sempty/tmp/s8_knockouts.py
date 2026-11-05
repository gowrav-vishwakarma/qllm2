"""s8 knockouts (EXPERIMENTS_SEMPY "kn"; Lead GO 2026-11-05): which mechanism does the TRAINED PAM use to carry an induced rule across
filler? The s8 spread probe exactly as the ctl readout (b2 suite, shift_wide_new + affine_map_new, 8 shots, 400 trials, seed 7000,
spread 0 / 2048 / 4096, the holdout filler cache), with one mechanism switched off. This measures what the trained model USES, not
what training needs (a model trained without a part may compensate; the ablation ladder answers that).

Arms (filler = the inserted filler tokens between the examples; the examples, separators and the query are untouched):
  on      : nothing switched off (must reproduce the ctl readout, s8 D 0.523)
  erase0  : notebook delta erase gate b_e = 0 on filler tokens (pure additive writes there: no erase-at-key)
  globfade: on filler tokens the key-selective erase is replaced by a global fade of the same mean strength: b_e = 0 and
            retention g <- g (1 - b_e / K)  (a unit key spans 1 of K directions: equal erased energy on an isotropic notebook)
  clock1  : Chrono warp g = 1 on filler tokens (the clock advances like plain RoPE through the filler)
  noecho  : the E1 echo read zeroed everywhere (whole-model switch)
  noread  : the M2 shared readers zeroed everywhere (whole-model switch)
    python v13_sempty/tmp/s8_knockouts.py --checkpoint <ckpt> --out <json> [--arms on,erase0,...]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from memory_probes.rules import aggregate                                  # noqa: E402
from memory_probes.behavioral import score_candidate_logits                # noqa: E402
from memory_probes.spread import filler_tokens, split_units, _SEP          # noqa: E402
from run_memory_behavioral import _load_v13_sempty, _last_logits           # noqa: E402
from sempyt.tensor import named                                            # noqa: E402
from dataclasses import replace                                            # noqa: E402

MASK = {'m': None}        # [1, T] float, 1 on filler tokens (set per example)


def spread_with_mask(example, tok, length, filler):
    """memory_probes.spread.spread_example, also returning the filler mask (same rng, same ids)."""
    header, units = split_units(example)
    head_ids = tok.encode(header, add_special_tokens=False)
    unit_ids = [tok.encode(u, add_special_tokens=False) for u in units]
    base = len(head_ids) + sum(len(u) for u in unit_ids); gaps = len(unit_ids) - 1
    sep = tok.encode(_SEP, add_special_tokens=False)
    if length == 0:
        chunks = [([], []) for _ in range(gaps)]
    else:
        per_gap = (length - base) // gaps - 2 * len(sep)
        if per_gap < 16:
            return None
        rng = np.random.default_rng([example.seed, length, sum(map(ord, example.task))])
        starts = rng.integers(0, len(filler) - per_gap, size=gaps)
        chunks = [(sep + filler[s:s + per_gap].tolist() + sep, [0] * len(sep) + [1] * per_gap + [0] * len(sep)) for s in starts]
    ids = list(head_ids) + list(unit_ids[0]); mask = [0] * len(ids)
    for (c, cm), unit in zip(chunks, unit_ids[1:]):
        ids += c + unit; mask += cm + [0] * len(unit)
    return replace(example, example_id=f'{example.example_id}_L{length}', task=f'{example.task}@{length}', prompt_ids=ids), mask


def install(model, arm):
    """Patch the trained model's RealPAMLayers / echo / readers for one arm; returns an undo function."""
    undo = []
    layers = [b.pam for b in model.blocks if getattr(b, 'pam', None) is not None]
    for L in layers:
        if arm in ('erase0', 'globfade'):
            orig_g, orig_d = L._delta_gates, L._decay
            K = L.head_dim
            def gates(tokens, _o=orig_g):
                bw, be = _o(tokens)
                m = MASK['m'].to(be.dtype)[:, None, :]                        # [B,1,T]
                return bw, be * (1 - m)
            L._delta_gates = gates
            if arm == 'globfade':
                def decay(tokens, _od=orig_d, _og=orig_g, _K=K):
                    ret = _od(tokens)                                          # NamedTensor (batch, heads, time)
                    batch, time = tokens.layout[0], tokens.layout[1]
                    _, be = _og(tokens)
                    m = MASK['m'].to(be.dtype)[:, None, :]
                    r = ret.raw(batch, L.heads, time) * (1 - m * be / _K)
                    return named(r, (batch, L.heads, time))
                L._decay = decay
            undo.append(lambda L=L, g=orig_g, d=orig_d: (setattr(L, '_delta_gates', g), setattr(L, '_decay', d)))
        if arm == 'clock1' and getattr(L, 'chrono', False):
            wp = L.warp_proj; orig_f = wp.forward
            def fwd(x, _o=orig_f):
                out = _o(x)
                batch, time = x.layout[0], x.layout[1]
                r = out.raw(batch, time, out.layout[-1]) * (1 - MASK['m'].to(out.data.dtype))[..., None]
                return named(r, (batch, time, out.layout[-1]))
            wp.forward = fwd
            undo.append(lambda wp=wp, f=orig_f: setattr(wp, 'forward', f))
    saved = {}
    if arm == 'noecho' and getattr(model, 'echo', False):
        saved['echo'] = model.echo_out.weight.detach().clone(); model.echo_out.weight.data.zero_()
        undo.append(lambda: model.echo_out.weight.data.copy_(saved['echo']))
    if arm == 'noread':
        for i, b in enumerate(model.blocks):
            if getattr(b, 'reader', None) is not None:
                saved[i] = b.reader.o_proj.weight.detach().clone(); b.reader.o_proj.weight.data.zero_()
                undo.append(lambda b=b, i=i: b.reader.o_proj.weight.data.copy_(saved[i]))
    return lambda: [u() for u in reversed(undo)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--out', required=True)
    p.add_argument('--arms', default='on,erase0,globfade,clock1,noecho,noread'); p.add_argument('--trials', type=int, default=400)
    p.add_argument('--seed', type=int, default=7000); p.add_argument('--lengths', default='0,2048,4096')
    p.add_argument('--filler', default='.cache/v7_tokens/pretrain_holdout_v1_v50261_t500000_e3_p5.pt')
    a = p.parse_args(); dev = torch.device('cuda')
    model, tok, _ = _load_v13_sempty(Path(a.checkpoint), 'baseline_real_pm', dev); model.eval()
    from v14.rules_ood import build_example
    tasks, lengths = ('shift_wide_new', 'affine_map_new'), [int(x) for x in a.lengths.split(',')]
    filler = filler_tokens(a.filler)
    exs = [build_example(tok, t, 8, sd) for t in tasks for sd in range(a.seed, a.seed + a.trials)]
    items = [r for ex in exs for L in lengths for r in [spread_with_mask(ex, tok, L, filler)] if r is not None]
    res = {}
    for arm in a.arms.split(','):
        t0 = time.time(); undo = install(model, arm); rows = []
        with torch.no_grad():
            for ex, mask in items:
                ids = torch.tensor([ex.prompt_ids], device=dev)
                MASK['m'] = torch.tensor([mask], device=dev, dtype=torch.float32)
                logits = _last_logits('v13_sempty', model, ids)[0]
                rows.append({**ex.to_dict(), **score_candidate_logits(ex, logits[ex.candidate_token_ids].float().cpu().tolist())})
        undo()
        agg = aggregate(rows); per = {g['task']: g['accuracy'] for g in agg}
        s8 = float(np.mean([per[f'{t}@{L}'] for t in tasks for L in (2048, 4096)]))
        res[arm] = {'per_task': per, 's8_D': s8, 'sec': round(time.time() - t0, 1)}
        print(f'{arm:9s} s8 D {s8:.3f}  ' + '  '.join(f'{k}={v:.3f}' for k, v in per.items()), flush=True)
    json.dump({'checkpoint': a.checkpoint, 'arms': res}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

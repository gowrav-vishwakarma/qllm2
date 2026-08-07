"""Does the fact module BIND key->value, or does it shortcut?

The fact stage reached answer-masked val PPL ~1.9-2.6 while scoring at chance on
the held-out behavioral probe. Both numbers can be true at once because the
training documents put the answer *in the context*: a model can score well by
emitting "some value token that appeared in this document" without ever writing
a key->value association into memory.

This scores the trained fact model on its OWN validation distribution under
progressively tighter candidate restrictions, which separates the two failure
modes:

  * ``ctx_restricted`` accuracy ~= 1/len(in-context values)
        -> SHORTCUT. The model emits a plausible context value but cannot say
           which key it belongs to. Binding was never learned, in-distribution
           or otherwise; the data/loss design is at fault.
  * ``ctx_restricted`` accuracy ~= 1.0 but the behavioral probe stays at chance
        -> TRANSFER FAILURE. Binding works on the training distribution but did
           not generalize; the readout is pattern-specific, not a generic
           retrieval operation. ``--grid`` then attributes which shift breaks it.

CLI:
    .venv/bin/python -m v12.diagnose_fact_shortcut \
        --checkpoint checkpoints_v12_curriculum/fact_retrieval/best_model.pt \
        --n_val 512 --seq_len 1024
    .venv/bin/python -m v12.diagnose_fact_shortcut --grid --trials 200 \
        --checkpoint ... --output grid.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch

from v12.eval_recall import load_v12


@torch.inference_mode()
def _all_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token logits at every position via the tied complex head."""
    from v12.complex_ops import imag_part, real_part

    lm = model._hidden_to_lm(input_ids)[0]          # [B,T,dim,2]
    return (
        real_part(lm) @ model.embed.embed_real.weight.T
        + imag_part(lm) @ model.embed.embed_imag.weight.T
    )


def diagnose(model, tokenizer, cfg, *, device, n_val=512, seq_len=1024,
             batch_size=4, seed=0, template_split='train', value_split='train',
             key_split='train', value_pool_limit=None):
    from v12.fact_data import FactRecallDataset

    ds = FactRecallDataset(n_val, seq_len, tokenizer, seed=seed + 12345,
                           template_split=template_split,
                           value_split=value_split, key_split=key_split,
                           value_pool_limit=value_pool_limit)
    pool_ids = torch.tensor([tid for _, tid in ds.value_pool], dtype=torch.long)
    pool_set = set(int(t) for t in pool_ids)
    pool_lut = torch.full((cfg.vocab_size,), -1, dtype=torch.long)
    pool_lut[pool_ids] = torch.arange(len(pool_ids))

    stats = {k: 0 for k in (
        'n', 'top1_correct', 'top1_is_value', 'top1_in_context',
        'pool_correct', 'ctx_correct',
    )}
    ctx_sizes, ranks = [], []

    for start in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        input_ids = torch.stack([it['input_ids'] for it in items]).to(device)
        labels = torch.stack([it['labels'] for it in items]).to(device)
        masks = torch.stack([it['loss_mask'] for it in items])

        logits = _all_logits(model, input_ids).float().cpu()

        for b in range(len(items)):
            positions = torch.nonzero(masks[b], as_tuple=False).flatten().tolist()
            prefix_all = input_ids[b].cpu().tolist()
            for pos in positions:
                gold = int(labels[b, pos].item())
                if gold not in pool_set:
                    continue  # multi-token value fragment; not scoreable
                row = logits[b, pos]

                # Values visible in the context when this token is predicted.
                seen = sorted({t for t in prefix_all[:pos + 1] if t in pool_set}
                              | {gold})
                seen_t = torch.tensor(seen, dtype=torch.long)

                top1 = int(row.argmax().item())
                pool_top = int(pool_ids[row[pool_ids].argmax()].item())
                ctx_top = int(seen_t[row[seen_t].argmax()].item())

                # Rank of gold among the in-context values (1 = best).
                order = seen_t[row[seen_t].argsort(descending=True)].tolist()

                stats['n'] += 1
                stats['top1_correct'] += int(top1 == gold)
                stats['top1_is_value'] += int(top1 in pool_set)
                stats['top1_in_context'] += int(top1 in set(seen))
                stats['pool_correct'] += int(pool_top == gold)
                stats['ctx_correct'] += int(ctx_top == gold)
                ctx_sizes.append(len(seen))
                ranks.append(order.index(gold) + 1)

        if (start // batch_size) % 20 == 0:
            print(f'  scored {stats["n"]} value positions', flush=True)

    n = max(stats['n'], 1)
    ctx_sizes = np.asarray(ctx_sizes, dtype=float)
    return {
        'n_value_positions': stats['n'],
        'mean_in_context_values': float(ctx_sizes.mean()) if len(ctx_sizes) else 0.0,
        'top1_vocab_correct': stats['top1_correct'] / n,
        'top1_is_some_value': stats['top1_is_value'] / n,
        'top1_is_context_value': stats['top1_in_context'] / n,
        'pool_restricted_correct': stats['pool_correct'] / n,
        'pool_restricted_chance': 1.0 / len(pool_ids),
        'ctx_restricted_correct': stats['ctx_correct'] / n,
        'ctx_restricted_chance': float(np.mean(1.0 / ctx_sizes)) if len(ctx_sizes) else 0.0,
        'ctx_mean_reciprocal_rank': float(np.mean(1.0 / np.asarray(ranks))) if ranks else 0.0,
        'value_pool_size': len(pool_ids),
    }


# ---------------------------------------------------------------------------
# Transfer grid: separate PROMPT-TEMPLATE shift from KEY/VALUE-VOCAB shift.
#
# The 2026-07 fact module scored 0.925 in-distribution and chance on the
# behavioral probe -- but the probe differs from the training loader on BOTH
# axes at once, so that number could not attribute the failure. This crosses
# them independently:
#
#             vocab=train                vocab=heldout
#  tmpl=train   in-distribution baseline   pure vocabulary transfer
#  tmpl=heldout pure template transfer     BOTH shifted -- the go/no-go cell
#
# Every cell runs the REAL training generator with splits swapped, so the only
# thing that varies is the axis under test. In particular the filler is drawn
# from the same bank in all four cells: an eval-only filler would shift with the
# template and confound the very attribution this grid exists to make.
# ---------------------------------------------------------------------------

_GRID_AXES = (('train', 'train'), ('train', 'heldout'),
              ('heldout', 'train'), ('heldout', 'heldout'))


def run_grid_cell(model, tokenizer, cfg, *, device, template_split, vocab_split,
                  trials=200, seq_len=1024, batch_size=4, seed=7,
                  value_pool_limit=None):
    """One grid cell. ``vocab_split`` switches BOTH the value pool and the key
    surface form, matching what the behavioral probe changes."""
    r = diagnose(model, tokenizer, cfg, device=device, n_val=trials,
                 seq_len=seq_len, batch_size=batch_size, seed=seed,
                 template_split=template_split, value_split=vocab_split,
                 key_split=vocab_split, value_pool_limit=value_pool_limit)
    r['template'] = template_split
    r['vocab'] = vocab_split
    r['trials'] = trials
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--n_val', type=int, default=512)
    p.add_argument('--seq_len', type=int, default=1024)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--grid', action='store_true',
                   help='Run the template x vocab transfer grid instead.')
    p.add_argument('--trials', type=int, default=200,
                   help='Examples per grid cell (with --grid).')
    p.add_argument('--value_pool', type=int, default=0,
                   help='Cap the value vocabulary to N words; must match the '
                        '--fact_value_pool the checkpoint was trained with.')
    args = p.parse_args()
    pool_limit = args.value_pool or None

    if args.grid:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, tokenizer, cfg = load_v12(args.checkpoint, device)
        print(f'Loaded V12 checkpoint: {args.checkpoint}')
        print(f'  params={sum(q.numel() for q in model.parameters()):,} device={device}')
        cells = []
        for template_split, vocab_split in _GRID_AXES:
            c = run_grid_cell(model, tokenizer, cfg, device=device,
                              template_split=template_split, vocab_split=vocab_split,
                              trials=args.trials, seq_len=args.seq_len,
                              batch_size=args.batch_size, seed=args.seed + 7,
                              value_pool_limit=pool_limit)
            cells.append(c)
            print(f"  [tmpl={template_split:7} / vocab={vocab_split:7}] "
                  f"top1={c['top1_vocab_correct']:.3f} "
                  f"ctx={c['ctx_restricted_correct']:.3f} "
                  f"(chance {c['ctx_restricted_chance']:.3f})", flush=True)

        def cell(t, v):
            return next(c for c in cells if c['template'] == t and c['vocab'] == v)

        print('\n=== transfer grid (ctx-restricted accuracy) ===')
        print(f"{'':16}{'vocab=train':>14}{'vocab=heldout':>16}")
        for t in ('train', 'heldout'):
            print(f"  tmpl={t:<11}{cell(t, 'train')['ctx_restricted_correct']:>14.3f}"
                  f"{cell(t, 'heldout')['ctx_restricted_correct']:>16.3f}")
        print(f"  {'chance':<13}{cell('train', 'train')['ctx_restricted_chance']:>14.3f}"
              f"{cell('train', 'heldout')['ctx_restricted_chance']:>16.3f}")

        # The in-distribution cell is the CONTROL. A model that never learned to
        # bind has nothing to transfer, so a chance score in the shifted cells
        # says nothing about generalization -- read the control first.
        base = cell('train', 'train')
        both = cell('heldout', 'heldout')
        base_lift = base['ctx_restricted_correct'] - base['ctx_restricted_chance']
        lift = both['ctx_restricted_correct'] - both['ctx_restricted_chance']
        print(f"\n  control (in-distribution): {base['ctx_restricted_correct']:.3f} "
              f"vs chance {base['ctx_restricted_chance']:.3f}  (lift {base_lift:+.3f})")
        print(f"  GATE (both axes shifted)  : {both['ctx_restricted_correct']:.3f} "
              f"vs chance {both['ctx_restricted_chance']:.3f}  (lift {lift:+.3f})")
        if base['ctx_restricted_correct'] < 0.5:
            print('  INCONCLUSIVE: the model does not bind even in-distribution, so')
            print('                the shifted cells measure nothing. Undertrained or')
            print('                the task is too hard at this budget -- fix that first.')
        elif both['ctx_restricted_correct'] > 0.5:
            print('  PASS: binding is a general operation, not a memorized pattern.')
        elif lift < 0.05:
            print('  FAIL: binding works in-distribution but is a surface pattern;')
            print('        diversity on these axes did not induce transfer.')
        else:
            print('  PARTIAL: some transfer, well short of in-distribution.')

        if args.output:
            with open(args.output, 'w') as fh:
                json.dump({'schema_version': 'v12-fact-transfer-grid/v2',
                           'created_at': datetime.now(timezone.utc).isoformat(),
                           'checkpoint': args.checkpoint, 'cells': cells}, fh, indent=2)
            print(f'\nResults saved to {args.output}')
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, cfg = load_v12(args.checkpoint, device)
    print(f'Loaded V12 checkpoint: {args.checkpoint}')
    print(f'  params={sum(q.numel() for q in model.parameters()):,} device={device}')

    r = diagnose(model, tokenizer, cfg, device=device, n_val=args.n_val,
                 seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed,
                 value_pool_limit=pool_limit)

    print('\n=== fact shortcut diagnosis ===')
    print(f"  value positions scored     : {r['n_value_positions']}")
    print(f"  value pool size            : {r['value_pool_size']}")
    print(f"  mean in-context values     : {r['mean_in_context_values']:.2f}")
    print()
    print(f"  top-1 over full vocab      : {r['top1_vocab_correct']:.3f}  (exact answer)")
    print(f"  top-1 is SOME pool value   : {r['top1_is_some_value']:.3f}  (learned 'emit a value')")
    print(f"  top-1 is a CONTEXT value   : {r['top1_is_context_value']:.3f}  (learned 'emit a context value')")
    print()
    print(f"  restricted to value pool   : {r['pool_restricted_correct']:.3f}  "
          f"(chance {r['pool_restricted_chance']:.5f})")
    print(f"  restricted to context values: {r['ctx_restricted_correct']:.3f}  "
          f"(chance {r['ctx_restricted_chance']:.3f})   <-- BINDING TEST")
    print(f"  MRR among context values   : {r['ctx_mean_reciprocal_rank']:.3f}")

    lift = r['ctx_restricted_correct'] - r['ctx_restricted_chance']
    print()
    if lift < 0.05:
        print('  VERDICT: SHORTCUT. Binding is at chance even in-distribution;')
        print('           answer-masked PPL came from "emit a plausible context value".')
    elif r['ctx_restricted_correct'] > 0.8:
        print('  VERDICT: binding works on this distribution. Run --grid to see')
        print('           whether it survives a template or vocabulary shift.')
    else:
        print(f'  VERDICT: PARTIAL binding (+{lift:.3f} over chance).')

    if args.output:
        payload = {'schema_version': 'v12-fact-shortcut/v1',
                   'created_at': datetime.now(timezone.utc).isoformat(),
                   'checkpoint': args.checkpoint, **r}
        with open(args.output, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print(f'\nResults saved to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

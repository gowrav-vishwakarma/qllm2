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
        -> TRANSFER FAILURE. Binding works for the 50 training values but did
           not generalize to the disjoint eval vocabulary; the readout is
           value-specific, not a generic retrieval operation.

CLI:
    .venv/bin/python -m v12.diagnose_fact_shortcut \
        --checkpoint checkpoints_v12_curriculum/fact_retrieval/best_model.pt \
        --n_val 512 --seq_len 1024
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
             batch_size=4, seed=0):
    from v12.fact_data import FactRecallDataset

    ds = FactRecallDataset(n_val, seq_len, tokenizer, seed=seed + 12345)
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
# The training loader and the behavioral probe differ on BOTH axes at once, so a
# chance score on the probe is uninterpretable. This builds the 2x2 so the two
# shifts can be attributed independently:
#
#            vocab=train                vocab=heldout
#  tmpl=fact   in-distribution baseline   pure vocabulary transfer
#  tmpl=probe  pure template transfer     == what eval_recall measures
# ---------------------------------------------------------------------------

_CONSONANTS = 'bdfgklmnprstvz'
_VOWELS = 'aeiou'


def _make_keys(rng, n, vocab):
    from memory_probes.behavioral import KEYS as PROBE_KEYS
    if vocab == 'heldout':
        return rng.sample(list(PROBE_KEYS), n)
    out, seen = [], set()
    while len(out) < n:
        k = ''.join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
                    + rng.choice(_CONSONANTS) for _ in range(2))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _make_value_pool(tokenizer, vocab):
    if vocab == 'heldout':
        from memory_probes.behavioral import single_token_values
        return single_token_values(tokenizer)
    from v12.fact_data import _build_value_pool
    return _build_value_pool(tokenizer)


def _fillers(tokenizer, template):
    if template == 'probe':
        from memory_probes.behavioral import FILLER
        return tokenizer.encode(FILLER, add_special_tokens=False)
    from v12.fact_data import _FILLER_BANK
    return tokenizer.encode(' '.join(_FILLER_BANK), add_special_tokens=False)


def _render(template, idx, key, value=None):
    """Record line (value given) or query stem (value None)."""
    if template == 'probe':
        if value is None:
            return f'\nMemory query: {key} means'
        return f'Memory record {idx + 1}: {key} means {value}.\n'
    if value is None:
        return f'Query: {key} means'
    return f'Record: {key} means {value}. '


def build_grid_example(rng, tokenizer, *, template, vocab, value_pool,
                       seq_len, max_facts=8, max_distractors=4):
    """One store-then-query document under a chosen template + vocabulary."""
    n_facts = rng.randint(2, max_facts)
    keys = _make_keys(rng, n_facts, vocab)
    vals = [value_pool[i][0] for i in rng.sample(range(len(value_pool)), n_facts)]

    target = rng.randrange(n_facts)
    order = [i for i in range(n_facts) if i != target] + [target]

    items = [(keys[i], vals[i]) for i in order]
    # Hard negatives: fresh keys reusing existing values (kills frequency cues).
    for _ in range(rng.randint(0, max_distractors)):
        dk = _make_keys(rng, 1, 'train')[0]
        items.insert(rng.randrange(len(items)), (dk, vals[rng.randrange(n_facts)]))

    rec_ids = []
    for i, (k, v) in enumerate(items):
        rec_ids += tokenizer.encode(_render(template, i, k, v), add_special_tokens=False)
    query_ids = tokenizer.encode(_render(template, 0, keys[target]),
                                 add_special_tokens=False)

    budget = seq_len - len(query_ids)
    if len(rec_ids) >= budget:
        rec_ids = rec_ids[len(rec_ids) - budget:]
        filler = []
    else:
        unit = _fillers(tokenizer, template)
        need = budget - len(rec_ids)
        filler = (unit * ((need + len(unit) - 1) // len(unit)))[:need]

    prompt = rec_ids + filler + query_ids
    gold = tokenizer.encode(f' {vals[target]}', add_special_tokens=False)[0]
    in_ctx = sorted({tid for _, tid in value_pool
                     if tid in set(prompt)} | {gold})
    return prompt, gold, in_ctx


@torch.inference_mode()
def run_grid_cell(model, tokenizer, cfg, *, device, template, vocab,
                  trials=200, seq_len=1024, seed=7):
    import random as _random

    from v12.eval_recall import _last_logits

    rng = _random.Random(seed)
    value_pool = _make_value_pool(tokenizer, vocab)
    correct = ctx_correct = 0
    chances, ranks = [], []

    for _ in range(trials):
        prompt, gold, in_ctx = build_grid_example(
            rng, tokenizer, template=template, vocab=vocab,
            value_pool=value_pool, seq_len=seq_len)
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        row = _last_logits(model, ids)[0].float().cpu()

        ctx_t = torch.tensor(in_ctx, dtype=torch.long)
        correct += int(int(row.argmax().item()) == gold)
        ctx_correct += int(int(ctx_t[row[ctx_t].argmax()].item()) == gold)
        order = ctx_t[row[ctx_t].argsort(descending=True)].tolist()
        ranks.append(order.index(gold) + 1)
        chances.append(1.0 / len(in_ctx))

    return {
        'template': template, 'vocab': vocab, 'trials': trials,
        'top1_vocab_correct': correct / trials,
        'ctx_restricted_correct': ctx_correct / trials,
        'ctx_restricted_chance': float(np.mean(chances)),
        'ctx_mean_reciprocal_rank': float(np.mean([1.0 / r for r in ranks])),
    }


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
    args = p.parse_args()

    if args.grid:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, tokenizer, cfg = load_v12(args.checkpoint, device)
        print(f'Loaded V12 checkpoint: {args.checkpoint}')
        print(f'  params={sum(q.numel() for q in model.parameters()):,} device={device}')
        cells = []
        for template in ('fact', 'probe'):
            for vocab in ('train', 'heldout'):
                c = run_grid_cell(model, tokenizer, cfg, device=device,
                                  template=template, vocab=vocab,
                                  trials=args.trials, seq_len=args.seq_len,
                                  seed=args.seed + 7)
                cells.append(c)
                print(f"  [{template:5} / {vocab:7}] top1={c['top1_vocab_correct']:.3f} "
                      f"ctx={c['ctx_restricted_correct']:.3f} "
                      f"(chance {c['ctx_restricted_chance']:.3f})", flush=True)
        print('\n=== transfer grid (ctx-restricted accuracy) ===')
        print(f"{'':12} {'vocab=train':>14} {'vocab=heldout':>16}")
        for template in ('fact', 'probe'):
            row = [c for c in cells if c['template'] == template]
            a = next(c for c in row if c['vocab'] == 'train')
            b = next(c for c in row if c['vocab'] == 'heldout')
            print(f"  tmpl={template:<7}{a['ctx_restricted_correct']:>14.3f}"
                  f"{b['ctx_restricted_correct']:>16.3f}")
        print(f"  chance      {cells[0]['ctx_restricted_chance']:>14.3f}"
              f"{cells[1]['ctx_restricted_chance']:>16.3f}")
        if args.output:
            with open(args.output, 'w') as fh:
                json.dump({'schema_version': 'v12-fact-transfer-grid/v1',
                           'created_at': datetime.now(timezone.utc).isoformat(),
                           'checkpoint': args.checkpoint, 'cells': cells}, fh, indent=2)
            print(f'\nResults saved to {args.output}')
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, cfg = load_v12(args.checkpoint, device)
    print(f'Loaded V12 checkpoint: {args.checkpoint}')
    print(f'  params={sum(q.numel() for q in model.parameters()):,} device={device}')

    r = diagnose(model, tokenizer, cfg, device=device, n_val=args.n_val,
                 seq_len=args.seq_len, batch_size=args.batch_size, seed=args.seed)

    print('\n=== fact shortcut diagnosis ===')
    print(f"  value positions scored     : {r['n_value_positions']}")
    print(f"  value pool size            : {r['value_pool_size']}")
    print(f"  mean in-context values     : {r['mean_in_context_values']:.2f}")
    print()
    print(f"  top-1 over full vocab      : {r['top1_vocab_correct']:.3f}  (exact answer)")
    print(f"  top-1 is SOME pool value   : {r['top1_is_some_value']:.3f}  (learned 'emit a value')")
    print(f"  top-1 is a CONTEXT value   : {r['top1_is_context_value']:.3f}  (learned 'emit a context value')")
    print()
    print(f"  restricted to 50-value pool: {r['pool_restricted_correct']:.3f}  "
          f"(chance {r['pool_restricted_chance']:.3f})")
    print(f"  restricted to context values: {r['ctx_restricted_correct']:.3f}  "
          f"(chance {r['ctx_restricted_chance']:.3f})   <-- BINDING TEST")
    print(f"  MRR among context values   : {r['ctx_mean_reciprocal_rank']:.3f}")

    lift = r['ctx_restricted_correct'] - r['ctx_restricted_chance']
    print()
    if lift < 0.05:
        print('  VERDICT: SHORTCUT. Binding is at chance even in-distribution;')
        print('           answer-masked PPL came from "emit a plausible context value".')
    elif r['ctx_restricted_correct'] > 0.8:
        print('  VERDICT: TRANSFER FAILURE. Binding works on trained values;')
        print('           it does not generalize to the held-out eval vocabulary.')
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

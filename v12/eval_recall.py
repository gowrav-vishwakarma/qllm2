"""V12 behavioral recall evaluator (single_assoc@2048 and the full grid).

The V11 runner (``scripts/run_memory_behavioral.py``) instantiates ``V11LM`` and
cannot score a V12 checkpoint (grammar base, fact module, or packed stack). This
evaluator loads a ``V12LM`` from the standard ``{config, model_state_dict}``
checkpoint schema and reuses ``memory_probes/behavioral.py`` example generation +
contrastive ``score_candidate_logits`` so V12 numbers are directly comparable to
the V11 / Mamba / transformer baselines.

The behavioral vocab (KEYS/VALUES) is HELD OUT from the fact-training loader
(``v12/fact_data.py`` uses a disjoint value pool), so this is a genuine
generalization test of key->value recall, not train-on-test.

CLI:
    .venv/bin/python -m v12.eval_recall --checkpoint packed_v12/model.pt \
        --context-lengths 128,512,1024,2048 --association-counts 1,4,8 --trials 60
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from memory_probes.behavioral import build_suite, score_candidate_logits


def load_v12(checkpoint: str, device: torch.device):
    """Load a V12LM from a {config, model_state_dict} checkpoint (eval mode)."""
    from v12.model import V12Config, V12LM
    from v7.data import get_chat_tokenizer

    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    raw = payload.get('config') or {}
    cfg = V12Config(**{k: v for k, v in raw.items()
                       if k in V12Config.__dataclass_fields__})
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    model = V12LM(cfg)
    model.load_state_dict(payload['model_state_dict'])
    model.to(device).eval()

    tokenizer = get_chat_tokenizer()
    if len(tokenizer) != cfg.vocab_size:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return model, tokenizer, cfg


@torch.inference_mode()
def _last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token logits at the final position via the tied complex head."""
    from v12.complex_ops import imag_part, real_part

    lm = model._hidden_to_lm(input_ids)[0]          # [B,T,dim,2]
    last = lm[:, -1]                                # [B,dim,2]
    return (
        real_part(last) @ model.embed.embed_real.weight.T
        + imag_part(last) @ model.embed.embed_imag.weight.T
    )


def _aggregate(rows: Sequence[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get('skipped'):
            continue
        key = (row['context_tokens'], row['target_position'], row['associations'])
        groups[key].append(row)
    out = []
    for key, values in sorted(groups.items()):
        out.append({
            'context_tokens': key[0],
            'target_position': key[1],
            'associations': key[2],
            'n': len(values),
            'accuracy': float(np.mean([r['correct'] for r in values])),
            'mean_reciprocal_rank': float(np.mean([1.0 / r['target_rank'] for r in values])),
            'mean_target_margin': float(np.mean([r['target_margin'] for r in values])),
        })
    return out


def run_recall_eval(
    model,
    tokenizer,
    cfg,
    *,
    device: torch.device,
    context_lengths: Sequence[int] = (128, 512, 1024, 2048),
    positions: Sequence[float] = (0.0, 0.5, 1.0),
    association_counts: Sequence[int] = (1, 4, 8),
    trials: int = 60,
    seed: int = 1000,
    candidate_count: int = 8,
) -> dict:
    """Score the behavioral suite; returns rows + aggregates (+ single_assoc headline)."""
    seeds = tuple(range(seed, seed + trials))
    max_context = getattr(cfg, 'max_seq_len', None)
    examples = build_suite(
        tokenizer,
        context_lengths=list(context_lengths),
        positions=list(positions),
        association_counts=list(association_counts),
        seeds=seeds,
        candidate_count=candidate_count,
    )
    rows = []
    for index, example in enumerate(examples, start=1):
        base = example.to_dict()
        if max_context is not None and example.context_tokens > int(max_context):
            rows.append({**base, 'skipped': True,
                         'reason': f'context exceeds model limit {max_context}'})
            continue
        ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=device)
        logits = _last_logits(model, ids)[0]
        candidate_logits = logits[example.candidate_token_ids].float().cpu().tolist()
        rows.append({**base, **score_candidate_logits(example, candidate_logits)})
        if index % 50 == 0 or index == len(examples):
            print(f'  scored {index}/{len(examples)} examples', flush=True)

    aggregates = _aggregate(rows)
    # single_assoc = 1 association; headline @ the longest context requested.
    headline_ctx = max(context_lengths)
    single = [a for a in aggregates
              if a['associations'] == 1 and a['context_tokens'] == headline_ctx]
    single_assoc = float(np.mean([a['accuracy'] for a in single])) if single else None
    return {
        'aggregates': aggregates,
        'rows': rows,
        'single_assoc_context': headline_ctx,
        'single_assoc_accuracy': single_assoc,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='V12 behavioral recall evaluation')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--context-lengths', default='128,512,1024,2048')
    p.add_argument('--positions', default='0,0.5,1')
    p.add_argument('--association-counts', default='1,4,8')
    p.add_argument('--trials', type=int, default=60)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--candidate-count', type=int, default=8)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--keep-rows', action='store_true',
                   help='Include per-example rows in --output (default is aggregates only)')
    return p


def _ints(v: str):
    return tuple(int(x) for x in v.split(',') if x.strip())


def _floats(v: str):
    return tuple(float(x) for x in v.split(',') if x.strip())


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, cfg = load_v12(args.checkpoint, device)
    print(f"Loaded V12 checkpoint: {args.checkpoint}")
    print(f"  params={sum(p.numel() for p in model.parameters()):,} device={device}")

    result = run_recall_eval(
        model, tokenizer, cfg, device=device,
        context_lengths=_ints(args.context_lengths),
        positions=_floats(args.positions),
        association_counts=_ints(args.association_counts),
        trials=args.trials, seed=args.seed, candidate_count=args.candidate_count,
    )
    print("\n=== V12 recall aggregates ===")
    for a in result['aggregates']:
        print(f"  ctx={a['context_tokens']:>5} pos={a['target_position']:<3} "
              f"n={a['associations']}: acc={a['accuracy']:.3f} "
              f"mrr={a['mean_reciprocal_rank']:.3f} margin={a['mean_target_margin']:+.3f}")
    sa = result['single_assoc_accuracy']
    print(f"\nsingle_assoc@{result['single_assoc_context']}: "
          f"{sa:.3f}" if sa is not None else "single_assoc: n/a")

    if args.output:
        n_rows = len(result.get('rows') or [])
        out = {
            'schema_version': 'memory-probes-behavioral/v1',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'model_type': 'v12',
            'model_identity': str(args.checkpoint),
            'device': str(device),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            **{k: v for k, v in result.items() if k != 'rows' or args.keep_rows},
        }
        if not args.keep_rows:
            out['rows_omitted'] = True
            out['n_rows'] = n_rows
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
        print(f"Results saved to {args.output}"
              + ("" if args.keep_rows else f" (aggregates, {n_rows} rows omitted)"))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run tokenizer-matched behavioral memory scoring on one trained LM.

Examples:
  uv run python scripts/run_memory_behavioral.py \
    --model-type v11 --checkpoint /path/to/best_model.pt \
    --preset v11_e3_k3_chat --output logs/memory_probes/publication/gpu/v11_behavior.json

  uv run python scripts/run_memory_behavioral.py \
    --model-type transformer --checkpoint /path/to/best_model.pt \
    --output logs/memory_probes/publication/gpu/transformer_behavior.json

  uv run python scripts/run_memory_behavioral.py \
    --model-type hf --model-id state-spaces/mamba-130m-hf \
    --output logs/memory_probes/publication/gpu/mamba_behavior.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_probes.behavioral import build_suite, score_candidate_logits


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(',') if item.strip())


def _load_v11(checkpoint: Path, preset: str, device: torch.device):
    from v11.model import V11LM, get_config
    from v7.data import get_chat_tokenizer

    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = get_config(preset)
    for key, value in (payload.get('config') or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.dropout = 0.0
    cfg.gradient_checkpointing = False
    model = V11LM(cfg)
    model.load_state_dict(payload['model_state_dict'])
    model.to(device).eval()

    tokenizer = get_chat_tokenizer()
    if len(tokenizer) != cfg.vocab_size:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return model, tokenizer, cfg


def _load_transformer(checkpoint: Path, device: torch.device):
    from transformers import AutoTokenizer
    from v6.transformer_baseline import TransformerConfig, TransformerLM

    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config_data = payload.get('config')
    if not config_data:
        raise ValueError('Transformer checkpoint has no saved config')
    cfg = TransformerConfig(**config_data)
    cfg.dropout = 0.0
    model = TransformerLM(cfg)
    model.load_state_dict(payload['model_state_dict'])
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained('gpt2'), cfg


def _load_hf(model_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    return model, tokenizer, model.config


@torch.inference_mode()
def _last_logits(model_type: str, model, input_ids: torch.Tensor) -> torch.Tensor:
    if model_type == 'v11':
        from v11.complex_ops import imag_part, real_part

        hidden = model._hidden_to_lm(input_ids)[0]
        last = hidden[:, -1]
        return (
            real_part(last) @ model.embed.embed_real.weight.T
            + imag_part(last) @ model.embed.embed_imag.weight.T
        )

    if model_type == 'transformer':
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        hidden = model.drop(model.token_embed(input_ids) + model.pos_embed(positions))
        for block in model.blocks:
            hidden = block(hidden)
        return model.lm_head(model.ln_f(hidden[:, -1]))

    base = model.base_model
    output = base(input_ids=input_ids, return_dict=True)
    hidden = output.last_hidden_state[:, -1]
    return model.get_output_embeddings()(hidden)


def _config_dict(config) -> dict[str, Any]:
    if hasattr(config, 'to_dict'):
        raw = config.to_dict()
    else:
        raw = vars(config)
    keep = (
        'model_type', 'vocab_size', 'max_seq_len', 'max_position_embeddings',
        'dim', 'hidden_size', 'n_layers', 'num_hidden_layers', 'n_heads',
        'num_attention_heads', 'n_states', 'state_size',
    )
    return {key: raw[key] for key in keep if key in raw}


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get('skipped'):
            continue
        key = (row['context_tokens'], row['target_position'], row['associations'])
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        output.append({
            'context_tokens': key[0],
            'target_position': key[1],
            'associations': key[2],
            'n': len(values),
            'accuracy': float(np.mean([row['correct'] for row in values])),
            'mean_reciprocal_rank': float(np.mean([
                1.0 / row['target_rank'] for row in values
            ])),
            'mean_target_margin': float(np.mean([
                row['target_margin'] for row in values
            ])),
        })
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Behavioral memory contrastive scoring')
    parser.add_argument('--model-type', choices=('v11', 'transformer', 'hf'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-id', default='state-spaces/mamba-130m-hf')
    parser.add_argument('--preset', default='v11_e3_k3_chat')
    parser.add_argument('--context-lengths', default='128,512,1024,2048')
    parser.add_argument('--positions', default='0,0.5,1')
    parser.add_argument('--association-counts', default='1,4,8')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--candidate-count', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.model_type in ('v11', 'transformer') and args.checkpoint is None:
        raise SystemExit(f'--checkpoint is required for {args.model_type}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == 'v11':
        model, tokenizer, config = _load_v11(args.checkpoint, args.preset, device)
        identity = str(args.checkpoint)
    elif args.model_type == 'transformer':
        model, tokenizer, config = _load_transformer(args.checkpoint, device)
        identity = str(args.checkpoint)
    else:
        model, tokenizer, config = _load_hf(args.model_id, device)
        identity = args.model_id

    lengths = _parse_ints(args.context_lengths)
    positions = _parse_floats(args.positions)
    associations = _parse_ints(args.association_counts)
    seeds = tuple(range(args.seed, args.seed + args.trials))
    max_context = getattr(config, 'max_seq_len', None)
    if max_context is None and args.model_type == 'transformer':
        max_context = getattr(config, 'max_position_embeddings', None)

    examples = build_suite(
        tokenizer,
        context_lengths=lengths,
        positions=positions,
        association_counts=associations,
        seeds=seeds,
        candidate_count=args.candidate_count,
    )
    rows = []
    for index, example in enumerate(examples, start=1):
        base = example.to_dict()
        if max_context is not None and example.context_tokens > int(max_context):
            rows.append({
                **base,
                'skipped': True,
                'reason': f'context exceeds model limit {max_context}',
            })
            continue
        ids = torch.tensor([example.prompt_ids], dtype=torch.long, device=device)
        logits = _last_logits(args.model_type, model, ids)[0]
        candidate_logits = logits[example.candidate_token_ids].float().cpu().tolist()
        rows.append({**base, **score_candidate_logits(example, candidate_logits)})
        if index % 20 == 0 or index == len(examples):
            print(f'  scored {index}/{len(examples)} examples', flush=True)

    result = {
        'schema_version': 'memory-probes-behavioral/v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model_type': args.model_type,
        'model_identity': identity,
        'preset': args.preset if args.model_type == 'v11' else None,
        'device': str(device),
        'platform': platform.platform(),
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'config': _config_dict(config),
        'protocol': {
            'metric': 'contrastive next-token accuracy over single-token values',
            'context_lengths': list(lengths),
            'positions': list(positions),
            'association_counts': list(associations),
            'seeds': list(seeds),
            'candidate_count': args.candidate_count,
        },
        'rows': rows,
        'aggregates': _aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Results saved to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

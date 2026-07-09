#!/usr/bin/env python3
"""Export a weights-only checkpoint + config.json for a Hugging Face release round.

Writes the HF bundle (weights + config) and, optionally, records the round in the
server manifest (releases/server_manifest.json) with a sha256 so the RTX4090 pull
script can fetch only new/changed rounds.

Usage:
    uv run python scripts/export_hf_release.py \
        --src checkpoints_v11_sft_chat_smoltalk_v2/best_model.pt \
        --round round-2b-gate --tag round-2b-gate \
        --pretrain_tokens_total 2000000000 --round_tokens 2000000000 \
        --record-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = 'gowravvishwakarma/qllm-pam-v11-e3k3-chat'


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description='Export HF release weights + metadata')
    p.add_argument('--src', default='checkpoints_v11_sft_chat_smoltalk_v2/best_model.pt')
    p.add_argument('--out-dir', default='hf_release')
    p.add_argument('--round', default='', help='Round id, e.g. round-2b-gate')
    p.add_argument('--tag', default='', help='HF revision tag (defaults to --round)')
    p.add_argument('--pretrain_tokens_total', type=int, default=0)
    p.add_argument('--round_tokens', type=int, default=0)
    p.add_argument('--sft_dataset', default='smoltalk2 (SFT config, think-capped)')
    p.add_argument('--pretrain_corpus',
                   default='DCLM-Edu + FineWeb-Edu + smoltalk2-Mid (reasoning/chat blend)')
    p.add_argument(
        '--out-name',
        default='qllm_v11_e3k3_chat.pt',
        help='Weights filename in hf_release (e.g. qllm_v11_e3k3_pretrain.pt)',
    )
    p.add_argument(
        '--config-name',
        default='config.json',
        help='Config JSON filename in hf_release (e.g. config_pretrain.json)',
    )
    p.add_argument(
        '--checkpoint-type',
        choices=('sft', 'pretrain'),
        default='',
        help='Metadata label; inferred from --out-name when omitted',
    )
    p.add_argument('--record-manifest', action='store_true',
                   help='Append this round to releases/server_manifest.json')
    p.add_argument('--manifest', default='releases/server_manifest.json')
    args = p.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.round
    ckpt_type = args.checkpoint_type or (
        'pretrain' if 'pretrain' in args.out_name else 'sft'
    )
    sft_dataset = args.sft_dataset
    if ckpt_type == 'pretrain' and sft_dataset == 'smoltalk2 (SFT config, think-capped)':
        sft_dataset = '(none — pretrain base)'

    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    config = dict(ckpt['config'])
    n_params = sum(t.numel() for t in ckpt['model_state_dict'].values())
    per_source_docs = dict(ckpt.get('per_source_docs') or {})
    per_source_tokens = dict(ckpt.get('per_source_tokens') or {})

    metadata = {
        'architecture': 'qllm-pam-v11-e3k3',
        'preset': 'v11_e3_k3_chat',
        'model_type': 'qllm_pam',
        'params': n_params,
        'vocab_size': config['vocab_size'],
        'round': args.round,
        'hf_tag': tag,
        'checkpoint_type': ckpt_type,
        'training': {
            'pretrain_corpus': args.pretrain_corpus,
            'pretrain_tokens_total': args.pretrain_tokens_total or ckpt.get('global_tokens', 0),
            'round_tokens': args.round_tokens,
            'sft_dataset': sft_dataset,
            'chat_template': 'ChatML (<|im_start|>/<|im_end|>) + <think>/</think>',
            'gate': 'phase-aware GSP (content-aware)',
            'val_ppl': float(ckpt.get('best_val_ppl', 0)),
            'val_loss': float(ckpt.get('best_val_loss', 0)),
            'per_source_docs': per_source_docs,
            'per_source_tokens': per_source_tokens,
        },
    }

    export = {
        'model_state_dict': ckpt['model_state_dict'],
        'config': config,
        'metadata': metadata,
    }
    weights_path = out_dir / args.out_name
    torch.save(export, weights_path)

    config_json = {**config, **metadata}
    config_path = out_dir / args.config_name
    config_path.write_text(json.dumps(config_json, indent=2) + '\n')

    size_bytes = weights_path.stat().st_size
    print(f'Exported {weights_path} ({size_bytes / 1e6:.1f} MB, {n_params:,} params)')
    print(f'Wrote {config_path}')

    if args.record_manifest:
        if not args.round:
            raise SystemExit('--record-manifest requires --round')
        sha = _sha256(weights_path)
        man_path = REPO_ROOT / args.manifest
        man_path.parent.mkdir(parents=True, exist_ok=True)
        if man_path.exists():
            manifest = json.loads(man_path.read_text())
        else:
            manifest = {'repo_id': DEFAULT_REPO, 'latest': None, 'rounds': {}}
        manifest.setdefault('rounds', {})
        manifest['rounds'][args.round] = {
            'hf_tag': tag,
            'sft_ckpt': str(src) if ckpt_type == 'sft' else manifest['rounds'].get(args.round, {}).get('sft_ckpt'),
            'pretrain_ckpt': str(src) if ckpt_type == 'pretrain' else manifest['rounds'].get(args.round, {}).get('pretrain_ckpt'),
            'hf_export_bundle': str(weights_path) if ckpt_type == 'sft' else manifest['rounds'].get(args.round, {}).get('hf_export_bundle'),
            'hf_pretrain_bundle': str(weights_path) if ckpt_type == 'pretrain' else manifest['rounds'].get(args.round, {}).get('hf_pretrain_bundle'),
            'sha256': sha if ckpt_type == 'sft' else manifest['rounds'].get(args.round, {}).get('sha256'),
            'pretrain_sha256': sha if ckpt_type == 'pretrain' else manifest['rounds'].get(args.round, {}).get('pretrain_sha256'),
            'size_bytes': size_bytes if ckpt_type == 'sft' else manifest['rounds'].get(args.round, {}).get('size_bytes'),
            'pretrain_size_bytes': size_bytes if ckpt_type == 'pretrain' else manifest['rounds'].get(args.round, {}).get('pretrain_size_bytes'),
            'pretrain_tokens_total': metadata['training']['pretrain_tokens_total'],
            'val_ppl': metadata['training']['val_ppl'],
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        manifest['latest'] = args.round
        man_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Recorded {args.round} in {man_path} (sha256 {sha[:12]}..., latest={args.round})')


if __name__ == '__main__':
    main()

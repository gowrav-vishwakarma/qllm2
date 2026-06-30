#!/usr/bin/env python3
"""Export weights-only checkpoint + config.json for Hugging Face release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description='Export HF release weights')
    p.add_argument(
        '--src',
        default='checkpoints_v11_sft_chat_smoltalk/best_model.pt',
    )
    p.add_argument('--out-dir', default='hf_release')
    args = p.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    config = dict(ckpt['config'])
    n_params = sum(t.numel() for t in ckpt['model_state_dict'].values())

    metadata = {
        'architecture': 'qllm-pam-v11-e3k3',
        'preset': 'v11_e3_k3_chat',
        'model_type': 'qllm_pam',
        'params': n_params,
        'vocab_size': config['vocab_size'],
        'training': {
            'pretrain_corpus': 'DCLM-Edu + FineWeb-Edu',
            'pretrain_tokens': 10_000_000_000,
            'sft_dataset': 'SmolTalk2 (hard filter)',
            'sft_epochs': 1,
            'chat_template': 'ChatML',
            'val_ppl': float(ckpt.get('best_val_ppl', 0)),
            'val_loss': float(ckpt.get('best_val_loss', 0)),
        },
    }

    export = {
        'model_state_dict': ckpt['model_state_dict'],
        'config': config,
        'metadata': metadata,
    }
    weights_path = out_dir / 'qllm_v11_e3k3_chat.pt'
    torch.save(export, weights_path)

    config_json = {**config, **metadata}
    config_path = out_dir / 'config.json'
    config_path.write_text(json.dumps(config_json, indent=2) + '\n')

    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f'Exported {weights_path} ({size_mb:.1f} MB, {n_params:,} params)')
    print(f'Wrote {config_path}')


if __name__ == '__main__':
    main()

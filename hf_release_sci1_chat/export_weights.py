#!/usr/bin/env python3
"""Export a SCI-1 training checkpoint into a release file + config.json.

The full training save (~4.6 GB) also holds the optimizer state, the data
cursors and the run args. The release file keeps only what inference needs:

    {'config': <architecture dict>, 'model_state_dict': <weights>,
     'global_step': int, 'global_tokens': int, 'source': str}

    python export_weights.py \
        --checkpoint /media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/sci1_long3r_91a87ed/final_step_749500.pt \
        --out sci1_chat_386m.pt

Only ``torch`` is needed (the repo is not imported). The weights are checked
by loading them into ``modeling_pam2.PAM2LM`` with ``strict=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modeling_pam2 import PAM2Config, PAM2LM, map_repo_state_dict  # noqa: E402

DEFAULT_SRC = ('/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/'
               'sci1_long3r_91a87ed/final_step_749500.pt')


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default=DEFAULT_SRC)
    ap.add_argument('--out', default=str(HERE / 'sci1_chat_386m.pt'))
    ap.add_argument('--config_json', default=str(HERE / 'config.json'))
    ap.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'],
                    help='storage dtype of the weights (float32 = the training master weights, exact)')
    args = ap.parse_args()

    src = Path(args.checkpoint)
    print(f'reading {src} ({src.stat().st_size / 1e9:.2f} GB) ...', flush=True)
    ck = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
    print('  top-level keys:', sorted(ck.keys()))
    raw_cfg = dict(ck['config'])
    cfg = PAM2Config.from_dict(raw_cfg)          # asserts the supported configuration
    sd = map_repo_state_dict(ck['model_state_dict'])
    dtype = getattr(torch, args.dtype)
    sd = {k: v.detach().to(dtype).contiguous().clone() for k, v in sd.items()}

    model = PAM2LM(cfg)
    model.load_state_dict(sd, strict=True)
    n_params = sum(v.numel() for v in sd.values())
    print(f'  {len(sd)} tensors, {n_params:,} params, loads into PAM2LM with strict=True')

    out = {
        'config': raw_cfg,                     # the repo PAMConfig dict, unchanged (loads in both codebases)
        'model_state_dict': sd,
        'global_step': int(ck.get('global_step', -1)),
        'global_tokens': int(ck.get('global_tokens', -1)),
        'source': f'qllm2 v13_sempty run sci1_long3r_91a87ed, {src.name}',
    }
    dst = Path(args.out)
    torch.save(out, dst)
    digest = sha256(dst)
    print(f'wrote {dst} ({dst.stat().st_size / 1e9:.2f} GB), sha256 {digest}')

    meta = {
        'model_type': 'pam2',
        'architecture': 'PAM2LM (qllm2 v13_sempty, real arm: sparse delta notebooks + Chrono clock + '
                        'shared reads + echo memory)',
        'name': 'SCI-1',
        'preset': (ck.get('args') or {}).get('preset', 'baseline_real_pm'),
        **cfg.to_dict(),
        'n_params': n_params,
        'tokenizer': {
            'base': 'gpt2',
            'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<think>', '</think>'],
            'ids': {'<|endoftext|>': 50256, '<|im_start|>': 50257, '<|im_end|>': 50258,
                    '<think>': 50259, '</think>': 50260},
            'pad_token': '<|endoftext|>',
            'chat_template': 'ChatML: <|im_start|>{role}\\n{content}<|im_end|>\\n ... <|im_start|>assistant\\n',
            'stop_token': '<|im_end|>',
        },
        'state_per_sequence_bytes_fp32': (len(cfg.pam_layers) + 1) * cfg.n_heads * cfg.head_dim ** 2 * 4
                                          + len(cfg.pam_layers) * cfg.n_heads * 4 + cfg.dim * 4,
        'training': {
            'global_step': out['global_step'],
            'global_tokens': out['global_tokens'],
            'train_seq_len': (ck.get('args') or {}).get('seq_len'),
            'best_val_loss': ck.get('best_val_loss'),
            'pretrain_sources': (ck.get('args') or {}).get('pretrain_sources'),
            'pretrain_weights': (ck.get('args') or {}).get('pretrain_weights'),
            'source_checkpoint': src.name,
        },
        'weights_file': dst.name,
        'weights_dtype': args.dtype,
        'weights_sha256': digest,
    }
    with open(args.config_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'wrote {args.config_json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Upload hf_release/ to Hugging Face (after verify.sh passes)."""

from __future__ import annotations

import argparse
import os

# Must be set before huggingface_hub import (avoids XET 403 on some tokens).
os.environ['HF_HUB_DISABLE_XET'] = '1'

from pathlib import Path  # noqa: E402

from huggingface_hub import HfApi  # noqa: E402


DEFAULT_REPO = 'gowravvishwakarma/qllm-pam-v11-e3k3-chat'

UPLOAD_FILES = (
    'README.md',
    'config.json',
    'modeling_qllm.py',
    'run_chat.py',
    'requirements.txt',
    'PUSH_TO_HF.md',
    'qllm_v11_e3k3_chat.pt',
)


def main() -> None:
    p = argparse.ArgumentParser(description='Upload QLLM HF release bundle')
    p.add_argument('--repo-id', default=DEFAULT_REPO)
    p.add_argument(
        '--folder',
        default='hf_release',
        help='Path to release folder (relative to repo root)',
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    folder = root / args.folder
    if not folder.is_dir():
        raise SystemExit(f'Missing folder: {folder}')

    for name in UPLOAD_FILES:
        if not (folder / name).exists():
            raise SystemExit(f'Missing required file: {folder / name}')

    print('Reminder: run cd hf_release && bash verify.sh before uploading.')

    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', exist_ok=True)
    print(f'Uploading {folder} -> {args.repo_id} ...')

    for name in UPLOAD_FILES:
        path = folder / name
        print(f'  {name} ({path.stat().st_size / (1024 * 1024):.1f} MB)')
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type='model',
            commit_message=f'Add {name}' if name != 'README.md' else 'Add QLLM PAM V11 model card and release bundle',
        )

    print(f'Done: https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()

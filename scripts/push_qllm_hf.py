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
    'LICENSE',
    'config.json',
    'modeling_qllm.py',
    'run_chat.py',
    'eval_chat.py',
    'eval_prompts_round1.yaml',
    'SAMPLES_round-2b-gate.md',
    'requirements.txt',
    'PUSH_TO_HF.md',
    'verify.sh',
    'verify_legacy.sh',
    'qllm_v11_e3k3_chat.pt',
)

# Shared code/docs for HF `main` — never overwrite legacy weights or config.json.
MAIN_CODE_FILES = (
    'README.md',
    'modeling_qllm.py',
    'run_chat.py',
    'eval_chat.py',
    'eval_prompts_round1.yaml',
    'requirements.txt',
    'PUSH_TO_HF.md',
    'verify.sh',
    'verify_legacy.sh',
)


def main() -> None:
    p = argparse.ArgumentParser(description='Upload QLLM HF release bundle')
    p.add_argument('--repo-id', default=DEFAULT_REPO)
    p.add_argument(
        '--folder',
        default='hf_release',
        help='Path to release folder (relative to repo root)',
    )
    p.add_argument(
        '--only',
        nargs='+',
        metavar='FILE',
        help='Upload only these files (e.g. --only README.md)',
    )
    p.add_argument(
        '--revision',
        default=None,
        help='HF revision/tag to publish this round under (e.g. round-2b-gate). '
             'Creates the branch/tag if missing; main is left untouched unless omitted.',
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    folder = root / args.folder
    if not folder.is_dir():
        raise SystemExit(f'Missing folder: {folder}')

    files = list(args.only) if args.only else list(UPLOAD_FILES)
    for name in files:
        path = folder / name
        if not path.exists():
            raise SystemExit(f'Missing required file: {path}')

    if not args.only:
        print('Reminder: run cd hf_release && bash verify.sh before uploading.')

    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', exist_ok=True)

    revision = args.revision
    if revision:
        # Publish onto a dedicated revision branch so `main` is untouched and each
        # round is pullable via `--revision <tag>`.
        try:
            api.create_branch(
                args.repo_id, repo_type='model', branch=revision, exist_ok=True,
            )
            print(f'Revision branch ready: {revision}')
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f'Could not create revision {revision}: {exc}')

    dest = f'{args.repo_id}@{revision}' if revision else args.repo_id
    print(f'Uploading {folder} -> {dest} ...')

    for name in files:
        path = folder / name
        print(f'  {name} ({path.stat().st_size / (1024 * 1024):.1f} MB)')
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo_id,
            repo_type='model',
            revision=revision,
            commit_message=(
                'Update model card (MIT license)'
                if name == 'README.md'
                else f'Add {name}'
            ),
        )

    if revision:
        print(f'Done: https://huggingface.co/{args.repo_id}/tree/{revision}')
        print(f'Pull: huggingface-cli download {args.repo_id} --revision {revision}')
    else:
        print(f'Done: https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()

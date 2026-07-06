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
    'qllm_v11_e3k3_chat.pt',
)

# Shared code/docs for HF `main` — code-only refresh via push_hf_main_code_only.sh.
MAIN_CODE_FILES = (
    'README.md',
    'modeling_qllm.py',
    'run_chat.py',
    'eval_chat.py',
    'eval_prompts_round1.yaml',
    'SAMPLES_round-2b-gate.md',
    'requirements.txt',
    'PUSH_TO_HF.md',
    'verify.sh',
)


def resolve_upload_files(revision: str | None) -> list[str]:
    """Build upload list; SAMPLES file follows the round revision tag."""
    samples = (
        f'SAMPLES_{revision}.md'
        if revision and revision not in ('main',)
        else 'SAMPLES_round-2b-gate.md'
    )
    return [
        samples if name == 'SAMPLES_round-2b-gate.md' else name
        for name in UPLOAD_FILES
    ]


def archive_main_as(api: HfApi, repo_id: str, tag: str) -> None:
    """Snapshot current HF main into an immutable tag before overwriting main."""
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            revision='main',
            repo_type='model',
            exist_ok=False,
        )
        print(f'Archived current main as tag: {tag}')
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f'Could not archive main as {tag}: {exc}')


def upload_files(
    api: HfApi,
    repo_id: str,
    folder: Path,
    files: tuple[str, ...] | list[str],
    revision: str | None,
) -> None:
    """Upload files to repo_id@revision (revision=None -> main)."""
    label = revision or 'main'
    dest = f'{repo_id}@{label}'
    print(f'Uploading {folder} -> {dest} ...')

    for name in files:
        path = folder / name
        if not path.exists():
            raise SystemExit(f'Missing required file: {path}')
        print(f'  {name} ({path.stat().st_size / (1024 * 1024):.1f} MB)')
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type='model',
            revision=revision,
            commit_message=(
                'Update model card (MIT license)'
                if name == 'README.md'
                else f'Add {name}'
            ),
        )

    if revision:
        print(f'Done: https://huggingface.co/{repo_id}/tree/{revision}')
        print(f'Pull: huggingface-cli download {repo_id} --revision {revision}')
    else:
        print(f'Done: https://huggingface.co/{repo_id}')


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
             'Use "main" to publish directly to main. Creates the branch if missing.',
    )
    p.add_argument(
        '--archive-main-as',
        metavar='TAG',
        default=None,
        help='Before overwriting main, snapshot current main as an immutable tag '
             '(e.g. v1-old-deprecated-10B-sft). Fails if the tag already exists.',
    )
    p.add_argument(
        '--also-main',
        action='store_true',
        help='After uploading to --revision <round tag>, also upload the full bundle '
             'to main so the default branch tracks the latest round.',
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    folder = root / args.folder
    if not folder.is_dir():
        raise SystemExit(f'Missing folder: {folder}')

    files = list(args.only) if args.only else resolve_upload_files(args.revision)
    for name in files:
        path = folder / name
        if not path.exists():
            raise SystemExit(f'Missing required file: {path}')

    if not args.only:
        print('Reminder: run cd hf_release && bash verify.sh before uploading.')

    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', exist_ok=True)

    revision = args.revision
    upload_to_main = revision in (None, 'main')
    also_main = args.also_main and revision not in (None, 'main')

    if args.archive_main_as and not (upload_to_main or also_main):
        raise SystemExit('--archive-main-as requires uploading to main (--revision main) '
                         'or --also-main')

    if args.archive_main_as:
        archive_main_as(api, args.repo_id, args.archive_main_as)

    if revision and revision != 'main':
        try:
            api.create_branch(
                args.repo_id, repo_type='model', branch=revision, exist_ok=True,
            )
            print(f'Revision branch ready: {revision}')
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f'Could not create revision {revision}: {exc}')
        upload_files(api, args.repo_id, folder, files, revision=revision)

    if upload_to_main:
        upload_files(api, args.repo_id, folder, files, revision=None)

    if also_main:
        main_files = resolve_upload_files(revision)
        for name in main_files:
            path = folder / name
            if not path.exists():
                raise SystemExit(f'Missing required file for main: {path}')
        upload_files(api, args.repo_id, folder, main_files, revision=None)


if __name__ == '__main__':
    main()

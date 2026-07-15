#!/usr/bin/env bash
# Offline FineWeb-Edu shard downloader (no HF auth; bypasses Xet CDN 403).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/fineweb-edu/sample-10BT logs/v11/recall_ab_sweep

TARGET_SHARDS="${TARGET_SHARDS:-8}"   # 000..007 (~3B filtered tok after edu filter)
LOG="logs/v11/recall_ab_sweep/fineweb_download.log"

export HF_HUB_DISABLE_XET=1

echo "[fineweb-dl] target=$TARGET_SHARDS shards -> data/fineweb-edu/sample-10BT" | tee -a "$LOG"

uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import hf_hub_download

target = int(os.environ.get('TARGET_SHARDS', '8'))
out_dir = 'data/fineweb-edu'
os.makedirs(f'{out_dir}/sample-10BT', exist_ok=True)

for i in range(target):
    name = f'{i:03d}_00000.parquet'
    flat = f'{out_dir}/sample-10BT/{name}'
    nested = f'{out_dir}/sample/10BT/{name}'
    if os.path.isfile(flat):
        print(f'skip {name} (exists, {os.path.getsize(flat)} bytes)', flush=True)
        continue
    if os.path.isfile(nested):
        os.makedirs(os.path.dirname(flat), exist_ok=True)
        if os.path.realpath(nested) != os.path.realpath(flat):
            os.link(nested, flat)
        print(f'linked {name} from nested ({os.path.getsize(flat)} bytes)', flush=True)
        continue
    print(f'downloading {name} ...', flush=True)
    p = hf_hub_download(
        repo_id='HuggingFaceFW/fineweb-edu', repo_type='dataset',
        filename=f'sample/10BT/{name}',
        local_dir=out_dir,
    )
    os.makedirs(os.path.dirname(flat), exist_ok=True)
    src = p if os.path.isfile(p) else nested
    if os.path.isfile(src) and not os.path.isfile(flat):
        if os.path.realpath(src) != os.path.realpath(flat):
            try:
                os.link(src, flat)
            except OSError:
                import shutil
                shutil.copy2(src, flat)
    print(f'ok {name} ({os.path.getsize(flat) if os.path.isfile(flat) else 0} bytes)', flush=True)
print('done', flush=True)
PY

echo "[fineweb-dl] complete" | tee -a "$LOG"
ls -lh data/fineweb-edu/sample-10BT/ | tee -a "$LOG"

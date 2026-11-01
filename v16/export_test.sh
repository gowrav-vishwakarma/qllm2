#!/bin/bash
# v16 export test (Lead): copy the MANIFEST files into a fresh folder and run one bench cell on CPU there (the sempyt .pth and the checkpoint path are external).
set -e
SRC=/home/gowrav/Development/qllm2; DST=${1:-/tmp/claude-1000/-home-gowrav-Development-qllm2/683d2316-c093-4e15-b19c-044555bd5293/scratchpad/v16_export}
rm -rf "$DST"; mkdir -p "$DST"
grep -v '^#' "$SRC/v16/MANIFEST" | while read f; do [ -z "$f" ] && continue; mkdir -p "$DST/$(dirname "$f")"; cp "$SRC/$f" "$DST/$f"; done
cd "$DST"; export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/local/bin:$PATH
PACKAGE_STRICT=1 PACKAGE_NORM=1 THREADS=4 "$SRC/.venv/bin/python" v16/bench.py --device cpu --ckpt "$SRC/checkpoints_v13_sempty/c1ctlb_main_7ae6f1f/best_model.pt" --arch pam --task kv --needles 8 --keys synthetic --lengths 2048 --docs 2 --seed 3 --text_package --skip_state --out "$DST/export_smoke.jsonl" > "$DST/export_smoke.log" 2>&1
python3 -c "import json; rs=[json.loads(l) for l in open('$DST/export_smoke.jsonl')]; assert len(rs)==2 and not any('error' in r for r in rs), rs; print('EXPORT-OK', [r['TP_20way'] for r in rs])"

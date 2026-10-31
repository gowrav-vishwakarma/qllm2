#!/bin/bash
# lg8r on the 4090 (EXPERIMENTS_SEMPY "lg8r" pre-registration, 2026-11-01): waits for Lead's
# package table, fine-tunes b2p2 at 8k windows for 50M tokens with the document-boundary reset
# and answer-in-window weighting, then the fixed readouts (reach, half-lives, guards).
# Run from a clean worktree at the launch hash:  tmux new-session -d -s lg8r "bash v13_sempty/tmp/lg8r_chain.sh"
# Resumable: finished arm skipped (.done), half-done arm resumes from latest.pt, finished probes skipped.
cd "$(dirname "$0")/../.." || exit 1
export PATH=/usr/local/cuda/bin:$PATH
H=${SCREEN_HASH:-$(git rev-parse --short HEAD)}; C=checkpoints_v13_sempty; PY=.venv/bin/python
B2P2=/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt
A=/media/gowrav/d_drive/qllm2/attr; X="symbolic\|Warning\|warn"
echo "lg8r chain: code $H in $PWD, $(date -u +%FT%TZ)"
PKG=/home/gowrav/Development/qllm2/logs/v16_pkg_table_driver_gpu_20261031_2332.log
until grep -q PKG-TABLE-DONE "$PKG" 2>/dev/null || ! pgrep -f tmp_pkg_table_gpu.sh >/dev/null; do sleep 60; done
need=21000   # budget: lg32 fwd+bwd 28.3 GB at 32k x 1 -> ~14 GB at 8k x 2, x 1.4 = ~20 GB (+ margin); alone on the card
until [ "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F", " '{print $1-$2}')" -ge $need ]; do sleep 60; done
echo "=== lg8r train start $(date -u +%FT%TZ)"
SCREEN=lg8r SCREEN_HASH=$H ARMS="main:ECHO=1,SHARE_READS=1" BASE="LR=3e-4 WARMUP=100 TARGET_TOKENS=50003968 VAL_EVERY=1526 SAVE_EVERY=200 KEEP_EVERY=0 KEEP_LAST=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128 SEED=45 SOURCES=dclm,fineweb,smoltalk2_mid,recall,b2gap WEIGHTS=0.43,0.40,0.10,0.04,0.03 INIT_FROM=$B2P2 SEQ=8192 BATCH=2 B2GAP_MAX=7500 RESET_AT_EOS=1 ANSWER_IN_WINDOW=1 MAX_RETRIES=5" \
  bash v13_sempty/tmp_scratch_screen.sh
B=$C/lg8r_main_$H/best_model.pt
[ -f $B ] || { echo "lg8r: no best_model.pt, stop"; exit 1; }
TS=$(date -u +%Y%m%d_%H%M); mkdir -p logs/diag logs/learning_probes
SP=logs/diag/v13_sempty_skillpack_lg8r_skillpack_${H}.jsonl
[ -s $SP ] || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack --lengths 0,2048,4096,8192,32768 --docs 100 --mem_frac 0.4 --out $SP 2>&1 | grep -v "$X"
HL=logs/diag/v13_sempty_halflives_lg8r_affine_L4096_${H}.txt
[ -s $HL ] || $PY v13_sempty/tmp/gates_diag.py $B 4096 12 affine_map_new 2>&1 | grep -v "$X" > $HL
LP=logs/learning_probes/v13_sempty_lg8r_main_${H}_b2_t400.json
[ -f $LP ] || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --trials 400 2>&1 | grep -E "per task|saved"
export ATTR_ROOT=$PWD
[ -f $A/loss_b2p2_ref.npz ] || $PY v13_sempty/tmp/attr3b.py score $A/data.npz $A/loss_b2p2_ref.npz b2p2=$B2P2 2>&1 | grep -v "$X"
[ -f $A/loss_lg8r_${H}.npz ] || $PY v13_sempty/tmp/attr3b.py score $A/data.npz $A/loss_lg8r_${H}.npz lg8r=$B 2>&1 | grep -v "$X"
$PY v13_sempty/tmp/attr3b.py report $A/data.npz lg8r $A/loss_lg8r_${H}.npz $A/loss_b2p2_ref.npz 2>&1 | grep -v "$X" | tee logs/diag/v13_sempty_attr3b_lg8r_vs_b2p2_${H}.txt
echo "=== readouts"
$PY - "$SP" <<'PYEOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]
for L in sorted({d['L'] for d in r}):
    x = [d for d in r if d['L'] == L]; per = {s: sum(d['A_state'] for d in x if d['sub'] == s) / max(1, sum(d['sub'] == s for d in x)) for s in ('affine_map_new', 'shift_wide_new')}
    print(f"reach L={L}: {sum(d['A_state'] for d in x) / len(x):.3f}  (affine {per['affine_map_new']:.2f}, shift {per['shift_wide_new']:.2f}, n {len(x)})")
PYEOF
$PY v13_sempty/tmp/hl_summary.py logs/diag/v13_sempty_halflives_b2p2_affine_L4096_82cd5a5.txt $HL
echo "=== lg8r DONE $(date -u +%FT%TZ)"

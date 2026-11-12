#!/bin/bash
# SCI-1 milestone CHECK (EXPERIMENTS_SEMPY "SCI-1"): one full checkpoint -> s8 D, skill pack 2k / 4k + contrast, far needles (t40),
# the known-suite subset (scripts/mc_eval.py: HellaSwag / ARC-e / PIQA, = lm-eval), and 3 shown generations. Holdout, the
# per-source split and WikiText come from the training log's val lines at the same step.
#   bash v13_sempty/tmp/sci1_check.sh <ckpt.pt> <name>        (runs beside training; probes cap their VRAM)
cd "$(dirname "$0")/../.." || exit 1
B=$1; n=$2; PY=.venv/bin/python; X="symbolic\|Warning\|warn"; mkdir -p logs/diag logs/learning_probes logs/memory_probes logs/sci1
echo "=== CHECK $n ($B) $(date -u +%FT%TZ)"
LP=logs/learning_probes/v13_sempty_${n}_spread_s7000_t400.json
# ok FILE [LINES]: complete output (2026-11-12: an 11:00 crash left a 1-line clk2 file that [ -s ] took as done): a .jsonl needs
# >= LINES lines, a .json must parse
ok() { [ -s "$1" ] || return 1; case "$1" in *.jsonl) [ "$(wc -l < "$1")" -ge "${2:-1}" ] || { rm -f "$1"; return 1; };; *.json) python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1" 2>/dev/null;; *) true;; esac; }
ok $LP || $PY scripts/run_learning_probe.py --model-type v13_sempty --checkpoint $B --output $LP --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|free \(A7|saved"
S=logs/diag/v13_sempty_skillpack_${n}.jsonl
ok $S 1600 || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants skillpack,contrast --alphas 0,4 --lengths 0,2048,4096,8192 --docs 100 --mem_frac ${MEM_FRAC:-0.12} --out $S 2>&1 | grep -v "$X" | tail -1
D=logs/diag/v13_sempty_clk2_echo_${n}.jsonl
ok $D 100 || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 8192 --docs 50 --seed 8700 --mem_frac ${MEM_FRAC:-0.12} --out $D 2>&1 | grep -v "$X" | tail -1
T=logs/memory_probes/v13_sempty_${n}_t40_behavioral.json
ok $T || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $B --preset baseline_real_pm --context-lengths 128,256,512,1024,2048,4096,8192 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 8192 --output $T 2>&1 | grep -v scored | tail -1
M=logs/sci1/v13_sempty_mc_${n}.json
ok $M || $PY scripts/mc_eval.py --ckpt $B --out $M 2>&1 | grep -E '^(hellaswag|arc|piqa)'
# think_eval back for cont4 (the `worked` keep rule: '<think>' leak in OFF answers < 5 %; OFF accuracy = GSM8K test, 200 items)
K=logs/sci1/v13_sempty_think_${n}.json
# dropped 2026-11-12 (not in the .checked gate; ~8 min): [ -f $K ] || $PY scripts/think_eval.py --ckpt $B --n 200 --out $K 2>&1 | grep -E "^think_eval"
# Reach curve past the trained SEQ 4096 (Lead + Mind step back 11-04):
D2=logs/diag/v13_sempty_clk2_echo_long_${n}.jsonl
ok $D2 100 || $PY scripts/run_skillpack_probe.py --checkpoint $B --variants dilate --dilations 1,8 --dilate_where echo --lengths 16384,32768 --docs 25 --seed 8700 --mem_frac ${MEM_FRAC:-0.12} --out $D2 2>&1 | grep -v "$X" | tail -1
T2=logs/memory_probes/v13_sempty_${n}_t40_long_behavioral.json
ok $T2 || $PY scripts/run_memory_behavioral.py --model-type v13_sempty --checkpoint $B --preset baseline_real_pm --context-lengths 16384 --positions 0,0.5,1 --association-counts 1,4,8 --trials 40 --max-context 16384 --output $T2 2>&1 | grep -v scored | tail -1
$PY scripts/sci_gen.py --ckpt $B 2>&1 | grep -v "$X" | tee logs/sci1/v13_sempty_gens_${n}.txt   # the user reads these without the log
echo "=== CHECK $n DONE $(date -u +%FT%TZ)"

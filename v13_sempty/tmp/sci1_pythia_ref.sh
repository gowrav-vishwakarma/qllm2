#!/bin/bash
# SCI-1 reference block (Lead + user GO 16:25 IST 2026-11-03): Pythia-410M (the Pile, 2.10M tokens/step) at revisions bracketing
# our milestones, scored with the SCI-1 CHECK's own harness: mc_eval (HellaSwag / ARC-e / PIQA, acc + acc_norm, fp32 = lm-eval)
# and s8 (b2 spread probe, seed 7000, 400 trials, 0 / 2k / 4k, plus 8k: Pythia is trained at 2k). One job; RAM guard (< 12 GB
# free -> stop); yields to the M4 CHECK (exits if sci1_check.sh is running). HF cache = d_drive (~/.cache/huggingface symlink).
#   bash v13_sempty/tmp/sci1_pythia_ref.sh          (in tmux)
cd "$(dirname "$0")/../.." || exit 1
H=$(git rev-parse --short HEAD); PY=.venv/bin/python; O=logs/sci1ref; mkdir -p $O; export PATH=/usr/local/cuda/bin:$PATH
guard() {   # run "$@" with the RAM guard and the M4 yield
  ( setsid "$@" ) & local P=$!
  while kill -0 $P 2>/dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt 12 ] || pgrep -f "tmp/sci1_check.sh" >/dev/null; then
      echo "STOP ($a GB free or the M4 CHECK started) $(date -u +%FT%TZ)"; kill -TERM -- -$P 2>/dev/null; pkill -TERM -P $P; exit 3; fi
    sleep 10; done; wait $P
}
for rev in step1000 step2000 step5000 step143000; do
  pgrep -f "tmp/sci1_check.sh" >/dev/null && { echo "yield to the M4 CHECK"; exit 3; }
  echo "=== pythia-410m $rev $(date -u +%FT%TZ)"
  M=$O/v13_sempty_pythia410m_${rev}_mc_${H}.json
  [ -f $M ] || guard $PY scripts/mc_eval.py --hf EleutherAI/pythia-410m --revision $rev --fp32 --batch_size 16 --out $M
  S=$O/v13_sempty_pythia410m_${rev}_spread_s7000_t400_${H}.json
  [ -f $S ] || guard $PY scripts/run_learning_probe.py --model-type hf --model-id EleutherAI/pythia-410m --revision $rev --output $S \
      --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096,8192 2>&1 | grep -E "per task|free \(A7|saved"
done
echo "=== reference DONE $(date -u +%FT%TZ)"

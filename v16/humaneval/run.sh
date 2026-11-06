#!/bin/bash
# Human-eval pages for an SCI-1 checkpoint: every test item (prompt, model answer, correct answer), grouped, phone-friendly.
#   bash v16/humaneval/run.sh <ckpt.pt>      # dump all tests on the 4090 (~15 min), then build the site
#   bash v16/humaneval/run.sh serve [step]   # serve the newest (or that step's) site over ngrok; prints the URL
#   bash v16/humaneval/run.sh stop           # close the server and the tunnel
# Outputs (not in git): /media/gowrav/d_drive/qllm2/humaneval/step_<N>/. Runs at the SCI-1 code (worktree W).
# One job on the 4090, a RAM guard (stops below 12 GB free; the box hung 2026-11-02). Model must be the local copy.
set -e
H=$(cd "$(dirname "$0")" && pwd); W=${W:-/home/gowrav/Development/qllm_sci1check}; PY=$W/.venv/bin/python
BASE=/media/gowrav/d_drive/qllm2/humaneval; PORT=${PORT:-8765}
if [ "$1" = stop ]; then tmux kill-session -t he_ngrok 2>/dev/null || true; tmux kill-session -t he_web 2>/dev/null || true; echo stopped; exit 0; fi
if [ "$1" = serve ]; then
  O=$BASE/${2:+step_$2}; [ -n "$2" ] && [ -d "$BASE/$2" ] && O=$BASE/$2; [ -n "$2" ] || O=$(ls -dt $BASE/step_* | head -1)
  tmux kill-session -t he_web 2>/dev/null || true; tmux kill-session -t he_ngrok 2>/dev/null || true
  tmux new-session -d -s he_web "python3 $H/server.py $PORT $O/site"
  tmux new-session -d -s he_ngrok "ngrok http $PORT --log stdout > $O/ngrok.log"
  for i in $(seq 20); do U=$(curl -s localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | cut -d'"' -f4); [ -n "$U" ] && break; sleep 1; done
  echo "serving $O -> $U"; exit 0
fi
CK=$1; [ -s "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
STEP=${STEP:-$(basename "$CK" | grep -o '[0-9]\{6\}')}; O=$BASE/${NAME:-step_$STEP}; mkdir -p $O   # NAME / STEP / THINK override for a non-milestone file (e.g. the decay)
cd $W; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
guard() { while kill -0 $1 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 12 ]; then echo "RAM GUARD: ${a} GB free, stopping" | tee -a $O/run.log; kill -TERM $1; fi; sleep 10; done; }
step() { "$@" >> $O/run.log 2>&1 & p=$!; guard $p; wait $p; }
echo "=== humaneval $CK $(date -u +%FT%TZ) code $(git rev-parse --short HEAD)" | tee $O/run.log
[ -s $O/m4_full.json ] || step $PY $H/dump.py $W $CK $O/m4_full.json 0
[ -s $O/rec_s8.jsonl ] || step $PY $H/rec.py $W run_learning_probe $O/rec_s8.jsonl --model-type v13_sempty --checkpoint $CK --output $O/s8.json \
  --suite b2 --tasks shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096
[ -s $O/rec_skill.jsonl ] || step $PY $H/rec.py $W run_skillpack_probe $O/rec_skill.jsonl --checkpoint $CK --variants skillpack \
  --lengths 0,2048,4096,8192 --docs 100 --mem_frac 0.5 --out $O/skill.jsonl
[ -s $O/rec_needle.jsonl ] || step $PY $H/rec.py $W run_memory_behavioral $O/rec_needle.jsonl --model-type v13_sempty --checkpoint $CK \
  --preset baseline_real_pm --context-lengths 128,256,512,1024,2048,4096,8192 --positions 0,0.5,1 --association-counts 1,4,8 \
  --trials 40 --max-context 8192 --output $O/needle.json
$PY $H/build.py $O $STEP $THINK | tee -a $O/run.log
echo "=== done $(date -u +%FT%TZ); serve with: bash $H/run.sh serve $STEP" | tee -a $O/run.log

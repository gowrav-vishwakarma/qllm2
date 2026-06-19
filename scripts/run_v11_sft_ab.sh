#!/usr/bin/env bash
# Phase C SFT A/B: WikiText base vs DCLM pretrain base (sequential, one GPU).
#
# Usage:
#   tmux new-session -d -s v11_sft_ab './scripts/run_v11_sft_ab.sh'
#   tmux attach -t v11_sft_ab

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

WIKI_CKPT="${WIKI_CKPT:-checkpoints_v11_e3_k3/best_model.pt}"
DCLM_CKPT="${DCLM_CKPT:-checkpoints_v11_e3_k3_dclm/best_model.pt}"
SMOKE_LOG="${SMOKE_LOG:-logs/v11/sft_ab_smoke_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p logs/v11

echo "============================================================"
echo "  V11 SFT A/B — $(date -Iseconds)"
echo "  A: wiki base -> checkpoints_v11_sft_wiki"
echo "  B: dclm base -> checkpoints_v11_sft_dclm"
echo "============================================================"

echo "[pre-SFT smoke] writing $SMOKE_LOG"
if [[ "${SKIP_PRE_SMOKE:-0}" != "1" ]]; then
  {
    echo "=== Pre-SFT smoke $(date -Iseconds) ==="
    ./scripts/smoke_chat_v11.py --checkpoint "$WIKI_CKPT" --label wiki_base
    echo
    ./scripts/smoke_chat_v11.py --checkpoint "$DCLM_CKPT" --label dclm_base
  } | tee "$SMOKE_LOG"
else
  echo "  (skipped — SKIP_PRE_SMOKE=1)"
fi

echo
echo "[SFT A] WikiText base..."
CKPT_DIR=checkpoints_v11_sft_wiki ./scripts/run_v11_sft_smoltalk.sh "$WIKI_CKPT"

echo
echo "[SFT B] DCLM pretrain base..."
CKPT_DIR=checkpoints_v11_sft_dclm ./scripts/run_v11_sft_smoltalk.sh "$DCLM_CKPT"

POST_SMOKE="${SMOKE_LOG%.log}_post.log"
echo
echo "[post-SFT smoke] writing $POST_SMOKE"
{
  echo "=== Post-SFT smoke $(date -Iseconds) ==="
  ./scripts/smoke_chat_v11.py --checkpoint checkpoints_v11_sft_wiki/best_model.pt --label sft_wiki
  echo
  ./scripts/smoke_chat_v11.py --checkpoint checkpoints_v11_sft_dclm/best_model.pt --label sft_dclm
} | tee "$POST_SMOKE"

echo
echo "============================================================"
echo "  SFT A/B complete — $(date -Iseconds)"
echo "  Pre-smoke:  $SMOKE_LOG"
echo "  Post-smoke: $POST_SMOKE"
echo "============================================================"

#!/usr/bin/env bash
# Verify hf_release runs with the legacy ~10B checkpoint (v1-old-deprecated-10B-sft: vocab 50259, mag gate).
set -euo pipefail

cd "$(dirname "$0")"

LEGACY_CKPT="${LEGACY_CKPT:-}"
LEGACY_DIR="${LEGACY_DIR:-/tmp/qllm-legacy-v1}"
LEGACY_REVISION="${LEGACY_REVISION:-v1-old-deprecated-10B-sft}"
REPO_ID="${HF_REPO_ID:-gowravvishwakarma/qllm-pam-v11-e3k3-chat}"

if [[ -z "$LEGACY_CKPT" ]]; then
  LEGACY_CKPT="${LEGACY_DIR}/qllm_v11_e3k3_chat.pt"
fi

if [[ ! -f "$LEGACY_CKPT" ]]; then
  echo "=== Downloading legacy weights from HF $LEGACY_REVISION ==="
  mkdir -p "$LEGACY_DIR"
  if command -v hf >/dev/null 2>&1; then
    hf download "$REPO_ID" qllm_v11_e3k3_chat.pt \
      --revision "$LEGACY_REVISION" --local-dir "$LEGACY_DIR"
  else
    huggingface-cli download "$REPO_ID" qllm_v11_e3k3_chat.pt \
      --revision "$LEGACY_REVISION" --local-dir "$LEGACY_DIR"
  fi
  LEGACY_CKPT="${LEGACY_DIR}/qllm_v11_e3k3_chat.pt"
fi

if [[ ! -f "$LEGACY_CKPT" ]]; then
  echo "FAIL: legacy checkpoint not found at $LEGACY_CKPT"
  echo "Set LEGACY_CKPT=/path/to/legacy/qllm_v11_e3k3_chat.pt or ensure huggingface-cli is logged in."
  exit 1
fi

PROMPT='What is the capital of France?'
echo "=== QLLM HF legacy verification ($LEGACY_REVISION / vocab 50259) ==="
echo "Checkpoint: $LEGACY_CKPT"
echo "Prompt: $PROMPT"
echo

OUT=$(uv run python run_chat.py \
  --checkpoint "$LEGACY_CKPT" \
  --prompt "$PROMPT" \
  --temperature 0.0 \
  --max_new_tokens 32 2>&1) || {
  echo "$OUT"
  echo "FAIL: run_chat.py exited with error on legacy checkpoint"
  exit 1
}

echo "$OUT"

if echo "$OUT" | grep -qi 'paris'; then
  echo "PASS: reply mentions Paris"
else
  echo "FAIL: reply does not mention Paris"
  exit 1
fi

if echo "$OUT" | grep -q 'stopped_on_im_end=True'; then
  echo "PASS: stopped on im_end"
else
  echo "FAIL: did not stop on im_end"
  exit 1
fi

echo
echo "=== Legacy checks passed ==="

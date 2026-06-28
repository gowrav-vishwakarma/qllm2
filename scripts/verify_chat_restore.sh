#!/usr/bin/env bash
# Post-SFT chat verification (same probes as the Tulu-3 regression test).
#
# Usage:
#   ./scripts/verify_chat_restore.sh [checkpoint] [temperature]
#
# Default checkpoint: checkpoints_v11_sft_chat_smoltalk/best_model.pt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

CKPT="${1:-checkpoints_v11_sft_chat_smoltalk/best_model.pt}"
TEMP="${2:-0.3}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  echo "Run on RTX first: tmux new-session -d -s v11_sft_smoltalk './scripts/run_v11_sft_smoltalk_chat.sh'"
  exit 1
fi

echo "=== smoke (fixed prompts) ==="
uv run python scripts/smoke_chat_v11.py \
  --checkpoint "$CKPT" \
  --preset v11_e3_k3_chat \
  --temperature "$TEMP"

echo ""
echo "=== regression probes ==="
uv run python - "$CKPT" "$TEMP" <<'PY'
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path('.').resolve()))
from v7.data import IM_END, format_chat_prompt, get_chat_tokenizer
from v11.model import V11LM, get_config

ckpt_path = sys.argv[1]
temp = float(sys.argv[2])
probes = [
    "what is 2+2?",
    "why are leaves green?",
    "write a python code to add two numbers",
    "my name is gowrav",
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = get_chat_tokenizer()
im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
cfg = get_config('v11_e3_k3_chat')
cfg.vocab_size = len(tokenizer)
model = V11LM(cfg)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()
m = model._orig_mod if hasattr(model, '_orig_mod') else model

for user in probes:
    prompt = format_chat_prompt(user)
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = m.generate(
            ids,
            max_new_tokens=256,
            temperature=temp,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=im_end_id,
        )
    reply = tokenizer.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
    reply = reply.split(IM_END, 1)[0].strip()
    print(f"\nUser> {user}")
    print(f"Assistant> {reply}")
PY

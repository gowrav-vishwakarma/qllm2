#!/usr/bin/env bash
# Legacy entrypoint — use pull_v11_training_ckpt.sh (v2 pretrain dir).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/pull_v11_training_ckpt.sh" "$@"

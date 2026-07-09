#!/usr/bin/env bash
# Smart, incremental pull of shipped model rounds from the training server (GCP)
# to a local machine (e.g. RTX4090) for Hugging Face publishing.
#
# Uses the server catalog (releases/server_manifest.json) + a per-machine local
# state (~/.qllm/v11_pull_state.json) to fetch ONLY rounds that are missing or
# whose sha256 changed. Running on a second machine won't re-download rounds that
# machine already has.
#
# Usage:
#   ./scripts/pull_v11_release.sh                 # pull latest round if new
#   ./scripts/pull_v11_release.sh --list          # show server vs local, no download
#   ./scripts/pull_v11_release.sh --round round-2b-gate
#   ./scripts/pull_v11_release.sh --dry-run
#   ./scripts/pull_v11_release.sh --all           # all rounds (not just latest)
#
# Env overrides:
#   REMOTE=ubuntu@HOST  REMOTE_DIR=/home/ubuntu/Development/qllm-private
#   SSH_KEY=~/.ssh/gowrav-personal  LOCAL_RELEASES_DIR=./releases  PULL_STATE=~/.qllm/v11_pull_state.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

REMOTE="${REMOTE:-ubuntu@34.131.232.173}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/Development/qllm-private}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/gowrav-personal}"
LOCAL_RELEASES_DIR="${LOCAL_RELEASES_DIR:-$(pwd)/releases}"
PULL_STATE="${PULL_STATE:-$HOME/.qllm/v11_pull_state.json}"
MACHINE_ID="${MACHINE_ID:-$(hostname)}"
PY="${PY:-python3}"

MODE="pull"        # pull | list
ROUND=""
DRY_RUN=0
ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) MODE="list" ;;
    --round) ROUND="$2"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    --all) ALL=1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

SSH_OPTS=(-o BatchMode=yes)
[[ -f "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")

tmp_manifest="$(mktemp)"
trap 'rm -f "$tmp_manifest"' EXIT

echo "[pull] fetching server manifest from ${REMOTE}:${REMOTE_DIR}/releases/server_manifest.json"
scp "${SSH_OPTS[@]}" "${REMOTE}:${REMOTE_DIR}/releases/server_manifest.json" "$tmp_manifest"

mkdir -p "$(dirname "$PULL_STATE")"
[[ -f "$PULL_STATE" ]] || echo "{\"machine_id\": \"${MACHINE_ID}\", \"rounds\": {}}" > "$PULL_STATE"

# Resolve which rounds to consider.
rounds_to_check() {
  if [[ -n "$ROUND" ]]; then
    echo "$ROUND"
  elif [[ "$ALL" == "1" ]]; then
    "$PY" -c "import json,sys; print('\n'.join(json.load(open('$tmp_manifest')).get('rounds',{}).keys()))"
  else
    "$PY" -c "import json; m=json.load(open('$tmp_manifest')); print(m.get('latest') or '')"
  fi
}

# Field getters (server + local).
server_field() { "$PY" -c "import json;m=json.load(open('$tmp_manifest'));print(m.get('rounds',{}).get('$1',{}).get('$2',''))"; }
local_sha()    { "$PY" -c "import json;m=json.load(open('$PULL_STATE'));print(m.get('rounds',{}).get('$1',{}).get('sha256',''))"; }

update_local_state() {  # round sha local_path
  "$PY" - "$PULL_STATE" "$1" "$2" "$3" "$MACHINE_ID" <<'PY'
import json, os, sys, tempfile
state_path, rnd, sha, local_path, machine = sys.argv[1:6]
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
state.setdefault('machine_id', machine)
state.setdefault('rounds', {})
state['rounds'][rnd] = {'sha256': sha, 'local_bundle': local_path,
                        'pulled_at': __import__('datetime').datetime.now(
                            __import__('datetime').timezone.utc).isoformat()}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(state_path) or '.')
with os.fdopen(fd, 'w') as f:
    json.dump(state, f, indent=2)
os.replace(tmp, state_path)
print(f"  [state] updated {rnd} -> {sha[:12]}...")
PY
}

any=0
while IFS= read -r rnd; do
  [[ -z "$rnd" ]] && continue
  any=1
  ssha="$(server_field "$rnd" sha256)"
  sbundle="$(server_field "$rnd" hf_export_bundle)"
  ssize="$(server_field "$rnd" size_bytes)"
  lsha="$(local_sha "$rnd")"
  if [[ -z "$ssha" ]]; then
    echo "[pull] round '$rnd' not found in server manifest" >&2
    continue
  fi
  status="NEW"
  [[ "$lsha" == "$ssha" ]] && status="up-to-date"
  [[ -n "$lsha" && "$lsha" != "$ssha" ]] && status="CHANGED"
  printf '  %-16s %-11s server=%s local=%s (%s bytes)\n' "$rnd" "$status" "${ssha:0:12}" "${lsha:0:12}" "$ssize"

  [[ "$MODE" == "list" ]] && continue
  [[ "$status" == "up-to-date" ]] && continue
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "    [dry-run] would pull ${REMOTE}:${REMOTE_DIR}/${sbundle}"
    continue
  fi

  dest_dir="${LOCAL_RELEASES_DIR}/${rnd}"
  mkdir -p "$dest_dir"
  dest="${dest_dir}/qllm_v11_e3k3_chat.pt"
  echo "    pulling ${sbundle} -> ${dest}"
  scp "${SSH_OPTS[@]}" "${REMOTE}:${REMOTE_DIR}/${sbundle}" "${dest}.partial"
  # Verify sha256 before committing.
  got="$("$PY" -c "import hashlib;print(hashlib.sha256(open('${dest}.partial','rb').read()).hexdigest())")"
  if [[ "$got" != "$ssha" ]]; then
    echo "    ERROR: sha256 mismatch (got ${got:0:12}, want ${ssha:0:12}); keeping .partial" >&2
    exit 1
  fi
  mv -f "${dest}.partial" "$dest"
  update_local_state "$rnd" "$ssha" "$dest"
  echo "    done: $(ls -lh "$dest" | awk '{print $5, $9}')"
  # DR backup: pretrain checkpoint that produced this SFT round.
  if [[ "$DRY_RUN" != "1" ]]; then
    ROUND="$rnd" ./scripts/pull_v11_training_ckpt.sh
  else
    echo "    [dry-run] would also pull pretrain DR checkpoint"
  fi
done < <(rounds_to_check)

[[ "$any" == "0" ]] && echo "[pull] nothing to do (no rounds in manifest)"
echo "[pull] local state: $PULL_STATE"

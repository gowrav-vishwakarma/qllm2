#!/usr/bin/env bash
# Registry helpers: list modules, resolve a target, publish a checkpoint.
#
# Usage:
#   v12/scripts/registry.sh list
#   v12/scripts/registry.sh resolve reasoning
#   v12/scripts/registry.sh stack "grammar@1,fact_retrieval@>=1"
#   v12/scripts/registry.sh publish <ckpt> <module_id> <version> <role> \
#         [group_id] [requires]
#     e.g. publish ckpts/fact/slim.pt fact_retrieval 1.0 group fact_retrieval \
#              "grammar@>=1.0:prelayer"
#
# Env: PY, REGISTRY, AUTHOR.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
REGISTRY="${REGISTRY:-v12_registry}"
AUTHOR="${AUTHOR:-local}"

cmd="${1:-help}"; shift || true
case "$cmd" in
  list)
    "$PY" -m v12.registry --registry "$REGISTRY" list ;;
  resolve)
    "$PY" -m v12.registry --registry "$REGISTRY" resolve "$@" ;;
  stack)
    "$PY" -m v12.registry --registry "$REGISTRY" stack "$@" ;;
  publish)
    ckpt="$1"; mid="$2"; ver="$3"; role="$4"; group="${5:-}"; requires="${6:-}"
    args=(-m v12.publish --checkpoint "$ckpt" --module_id "$mid" --version "$ver" \
          --role "$role" --provenance "$AUTHOR" --registry "$REGISTRY" --overwrite)
    [ -n "$group" ] && args+=(--group_id "$group")
    [ -n "$requires" ] && args+=(--requires "$requires")
    echo "+ $PY ${args[*]}"
    "$PY" "${args[@]}" ;;
  *)
    echo "Usage: $0 {list|resolve <target>|stack \"id@spec,...\"|publish <ckpt> <id> <ver> <role> [group] [requires]}"
    exit 1 ;;
esac

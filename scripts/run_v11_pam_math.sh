#!/usr/bin/env bash
# Deprecated wrapper — use ./scripts/run_memory_probes.sh
exec "$(dirname "$0")/run_memory_probes.sh" "$@"

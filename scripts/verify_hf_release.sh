#!/usr/bin/env bash
# Wrapper: verify hf_release from repo root.
set -euo pipefail
cd "$(dirname "$0")/../hf_release"
exec bash verify.sh

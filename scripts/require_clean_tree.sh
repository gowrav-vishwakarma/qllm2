# Source from a run launcher (from the repo root) BEFORE it names its log:
#
#   source scripts/require_clean_tree.sh "${PIN_HASH:-}"
#
# Logs and checkpoints are named after the commit hash, so a run must start
# from committed code (AGENTS.md: an uncommitted edit once cost hours of bad
# v13 runs). This refuses to launch when:
#   1. tracked files have uncommitted changes (untracked files are ignored), or
#   2. a PIN hash is given (resuming a run named after an older commit) and the
#      code differs between that commit and HEAD. Docs (*.md) and logs/ may
#      differ: a docs-only pull mid-run is fine.
# Overrides, for smoke tests only (a loud warning goes into the log):
#   ALLOW_DIRTY=1        skip check 1
#   ALLOW_CODE_DRIFT=1   skip check 2

_rct_pin="${1:-}"
_rct_dirty=$(git status --porcelain --untracked-files=no)
if [ -n "$_rct_dirty" ]; then
  if [ "${ALLOW_DIRTY:-0}" = "1" ]; then
    echo "!!! WARNING: launching on UNCOMMITTED code (ALLOW_DIRTY=1) -- the hash" \
         "in this run's name does NOT describe the code that ran:"
    echo "$_rct_dirty"
  else
    echo "REFUSING to launch: tracked files have uncommitted changes."
    echo "Commit (and push/pull) first, or set ALLOW_DIRTY=1 for a smoke test:"
    echo "$_rct_dirty"
    exit 3
  fi
fi
if [ -n "$_rct_pin" ] && [ "$_rct_pin" != "$(git rev-parse --short HEAD)" ]; then
  _rct_drift=$(git diff --name-only "$_rct_pin" HEAD -- . ':(exclude)*.md' ':(exclude)logs')
  if [ -n "$_rct_drift" ]; then
    if [ "${ALLOW_CODE_DRIFT:-0}" = "1" ]; then
      echo "!!! WARNING: code changed since pinned $_rct_pin (ALLOW_CODE_DRIFT=1):"
      echo "$_rct_drift"
    else
      echo "REFUSING to launch: code changed since the pinned commit $_rct_pin" \
           "(HEAD $(git rev-parse --short HEAD)); resume on the pinned code" \
           "(git checkout $_rct_pin) or start a new run. Changed:"
      echo "$_rct_drift"
      exit 3
    fi
  fi
fi
unset _rct_pin _rct_dirty _rct_drift

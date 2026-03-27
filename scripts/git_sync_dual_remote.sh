#!/usr/bin/env bash
# Sync local branch with a private remote (default: origin) and a public remote
# (default: public_repo) without dropping merged PRs from public.
#
# Strategy: rebase onto private (keeps linear history with your primary remote),
# then MERGE from public (a merge commit descends from both remotes, so push to
# both succeeds as a fast-forward). This avoids the rebase-both problem where
# rewriting SHAs makes one push non-fast-forward.
#
# Usage:
#   ./scripts/git_sync_dual_remote.sh
#   ./scripts/git_sync_dual_remote.sh --dry-run
#
# Environment (optional):
#   GIT_SYNC_PRIVATE_REMOTE   default: origin
#   GIT_SYNC_PUBLIC_REMOTE    default: public_repo
#   GIT_SYNC_BRANCH           default: current branch (must not be detached)
#   GIT_SYNC_ALLOW_DIRTY=1    stash uncommitted changes, sync commits, then stash pop
#
# If a pull stops on conflicts: resolve, `git add`, and continue (rebase --continue or
# commit for merge), then either push both remotes manually or re-run this script.

set -euo pipefail

PRIVATE_REMOTE="${GIT_SYNC_PRIVATE_REMOTE:-origin}"
PUBLIC_REMOTE="${GIT_SYNC_PUBLIC_REMOTE:-public_repo}"
ALLOW_DIRTY="${GIT_SYNC_ALLOW_DIRTY:-0}"
DRY_RUN=0
BRANCH_OVERRIDE=""

usage() {
  sed -n '1,20p' "$0" | tail -n +2
}

die() {
  echo "git_sync_dual_remote: $*" >&2
  exit 1
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    echo
  else
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    -n | --dry-run)
      DRY_RUN=1
      shift
      ;;
    --private-remote)
      PRIVATE_REMOTE="${2:?}"
      shift 2
      ;;
    --public-remote)
      PUBLIC_REMOTE="${2:?}"
      shift 2
      ;;
    --branch)
      BRANCH_OVERRIDE="${2:?}"
      shift 2
      ;;
    *)
      die "unknown option: $1 (try --help)"
      ;;
  esac
done

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  die "not inside a git repository"
fi

BRANCH="${BRANCH_OVERRIDE:-${GIT_SYNC_BRANCH:-}}"
if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git branch --show-current 2>/dev/null || true)"
fi
[[ -n "$BRANCH" ]] || die "detached HEAD or empty branch; checkout a branch or set GIT_SYNC_BRANCH"

current="$(git branch --show-current 2>/dev/null || true)"
[[ "$current" == "$BRANCH" ]] || die "checked out branch is '$current', not '$BRANCH' (checkout first or pass --branch after switching)"

STASHED=0
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  if [[ "$ALLOW_DIRTY" != "1" ]]; then
    die "working tree is dirty; commit/stash or set GIT_SYNC_ALLOW_DIRTY=1"
  elif [[ "$DRY_RUN" -eq 0 ]]; then
    echo "git_sync_dual_remote: stashing local changes (GIT_SYNC_ALLOW_DIRTY=1)..."
    git stash push -u -m "git_sync_dual_remote auto-stash $(date +%Y%m%d-%H%M%S)"
    STASHED=1
  fi
fi

sync_stash_exit_note() {
  if [[ "$STASHED" -eq 1 ]]; then
    echo "git_sync_dual_remote: sync did not finish; uncommitted work is in git stash (git stash list)" >&2
  fi
}
if [[ "$STASHED" -eq 1 ]]; then
  trap sync_stash_exit_note EXIT
fi

git_remote_exists() {
  git remote get-url "$1" >/dev/null 2>&1
}

git_remote_exists "$PRIVATE_REMOTE" || die "remote '$PRIVATE_REMOTE' is not configured"
git_remote_exists "$PUBLIC_REMOTE" || die "remote '$PUBLIC_REMOTE' is not configured"

echo "Syncing branch '$BRANCH': private=$PRIVATE_REMOTE public=$PUBLIC_REMOTE"

run git fetch "$PRIVATE_REMOTE" "$BRANCH"
run git fetch "$PUBLIC_REMOTE" "$BRANCH"

# Step 1: rebase onto private — keeps linear history with your primary remote.
run git pull "$PRIVATE_REMOTE" "$BRANCH" --rebase

# Step 2: MERGE from public (not rebase). A merge commit has both remotes as
# ancestors, so pushing to both is a fast-forward. If already up-to-date this
# is a no-op (no empty merge commit).
run git pull "$PUBLIC_REMOTE" "$BRANCH" --no-rebase

# Step 3: push to both. Private sees the merge commit as a descendant of its
# old HEAD; public sees it as a descendant of its old HEAD too.
run git push "$PRIVATE_REMOTE" "$BRANCH"
run git push "$PUBLIC_REMOTE" "$BRANCH"

if [[ "$STASHED" -eq 1 ]]; then
  trap - EXIT
  if [[ "$DRY_RUN" -eq 0 ]]; then
    echo "git_sync_dual_remote: restoring stashed local changes..."
    git stash pop || die "stash pop failed; resolve conflicts then: git stash pop"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run finished; no changes were made."
else
  echo "Done. Both remotes should match this branch."
fi

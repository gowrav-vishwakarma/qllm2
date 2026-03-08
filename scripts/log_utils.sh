#!/usr/bin/env bash
# Shared logging utilities for reproducible, traceable experiment runs.
#
# Provides:
#   make_log_dir <version> <run_name>  -> deterministic log path
#   write_run_info <log_dir> <desc> <args>  -> RUN_INFO.txt with full provenance
#
# Log path format:
#   logs/<version>/<run_name>_<YYYYMMDD_HHMMSS>_<commit>[_dirty]
#
# Example:
#   source ./scripts/log_utils.sh
#   LOG_DIR=$(make_log_dir "v6" "small_matched")
#   write_run_info "$LOG_DIR" "V6 small-matched on RTX 4090" "--size small-matched --epochs 10"

set -e

_git_short() {
    git rev-parse --short HEAD 2>/dev/null || echo "nogit"
}

_git_full() {
    git rev-parse HEAD 2>/dev/null || echo "unknown"
}

_git_branch() {
    git branch --show-current 2>/dev/null || echo "detached"
}

_git_is_dirty() {
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        echo "yes"
    else
        echo "no"
    fi
}

# make_log_dir <version> <run_name>
#   Outputs a path like: logs/v6/small_matched_20260308_163000_abc1234
#   Appends _dirty if working tree has uncommitted changes.
make_log_dir() {
    local version="$1"
    local run_name="$2"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local commit
    commit=$(_git_short)
    local suffix=""
    if [[ "$(_git_is_dirty)" == "yes" ]]; then
        suffix="_dirty"
    fi
    echo "logs/${version}/${run_name}_${ts}_${commit}${suffix}"
}

# make_group_prefix <version> <group_name>
#   Like make_log_dir but for a group of related runs (e.g. ablation).
#   Returns a directory prefix; individual runs go in subdirectories.
#   Example: logs/v6/ablation_tiny_20260308_163000_abc1234/
make_group_prefix() {
    local version="$1"
    local group_name="$2"
    make_log_dir "$version" "$group_name"
}

# write_run_info <log_dir> <description> <args>
#   Creates RUN_INFO.txt with full provenance metadata.
write_run_info() {
    local log_dir="$1"
    local description="$2"
    local args="$3"
    mkdir -p "$log_dir"

    local python_ver="unknown"
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        python_ver=$(eval "$PYTHON_BIN --version" 2>/dev/null || echo "unknown")
    fi

    cat > "${log_dir}/RUN_INFO.txt" <<RUNEOF
================================================================================
RUN INFO
================================================================================
Description : ${description}
Date (UTC)  : $(date -u +"%Y-%m-%d %H:%M:%S")
Date (local): $(date +"%Y-%m-%d %H:%M:%S %Z")
Hostname    : $(hostname)
Platform    : $(uname -srm)

Git commit  : $(_git_full)
Git short   : $(_git_short)
Git branch  : $(_git_branch)
Git dirty   : $(_git_is_dirty)

Python      : ${python_ver}
Arguments   : ${args}
================================================================================
RUNEOF

    if [[ "$(_git_is_dirty)" == "yes" ]]; then
        {
            echo ""
            echo "UNCOMMITTED CHANGES AT RUN TIME:"
            echo "--------------------------------"
            git diff --stat 2>/dev/null || echo "(could not get diff stat)"
            echo ""
            git diff 2>/dev/null | head -200 || true
            if [[ $(git diff 2>/dev/null | wc -l) -gt 200 ]]; then
                echo "... (diff truncated at 200 lines)"
            fi
        } >> "${log_dir}/RUN_INFO.txt"
    fi

    echo "[log-utils] Run info written to ${log_dir}/RUN_INFO.txt"
}

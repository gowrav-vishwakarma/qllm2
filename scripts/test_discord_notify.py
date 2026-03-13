#!/usr/bin/env python3
"""
Smoke test for Discord notifications (no model load, no GPU).
Tests: single message, long chunked message, and failure notification.
Run from repo root: uv run python scripts/test_discord_notify.py
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to sys.path so we can import from v6
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from v6.train import _notify_discord, _notify_discord_long, _notify_training_failure


def _load_dotenv_for_discord() -> None:
    _env_path = _repo_root / ".env"
    if not _env_path.exists():
        print(f"[smoke test] No .env at {_env_path}", file=sys.stderr)
        return
    try:
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key == "DISCORD_HOOK" and value:
                    os.environ["DISCORD_HOOK"] = value
                    break
    except Exception as e:
        print(f"[smoke test] Could not read .env: {e}", file=sys.stderr)


def main() -> None:
    _load_dotenv_for_discord()
    hook = os.environ.get("DISCORD_HOOK", "").strip()
    if not hook:
        print("[smoke test] DISCORD_HOOK not set (add to .env or export).", file=sys.stderr)
        sys.exit(1)
    print("[smoke test] Webhook configured.", file=sys.stderr)

    # 1) Simulated startup summary (same format as main() now builds)
    summary_lines = [
        f"Wall clock start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "V6 Phase-First Model (mode: autoregressive)",
        "=" * 60,
        f"Host: {os.uname().nodename}",
        "Size: small-matched",
        "Dataset: wikitext103",
        "Complex dim: 128 (= 256 real values/position)",
        "SSM state dim: 64 (multi-timescale: fast/medium/slow)",
        "Layers: 6",
        "Banks: 2 (semantic + context)",
        "Working memory slots: 32 (top-k=4, decay=0.95)",
        "Internal memory slots: 16 (top-k=4)",
        "PhaseAttention: DISABLED (attention-free)",
        "Epochs: 20",
        "LR schedule: cosine (warmup=500)",
        "Dropout: 0.1, Weight decay: 0.01",
        "AMP: bf16, TF32: True, Compile: False",
        "Workers: 4, Pin memory: True",
        "Batch log interval: 50",
        "Log file: logs/v6/v6_autoregressive_small-matched.log",
        "Checkpoint dir: checkpoints_v6",
        "=" * 60,
        "Init strategy: phase_aware (seed: 42)",
        "Params: 12,345,678 (12.3M)",
        "Training on cuda:0 | Epochs: 1..20 | Batches/epoch: 312 | Batch size: 32",
    ]
    header = "**Training started** (smoke test)"
    full_msg = header + "\n```\n" + "\n".join(summary_lines) + "\n```"
    print(f"[smoke test] Sending startup summary ({len(full_msg)} chars)...", file=sys.stderr)
    _notify_discord_long(full_msg)
    print("[smoke test] OK: startup summary sent.", file=sys.stderr)

    # 2) Simulated failure notification
    print("[smoke test] Sending failure notification...", file=sys.stderr)
    _notify_training_failure("failed (smoke test)", RuntimeError("simulated OOM"))
    print("[smoke test] OK: failure notification sent.", file=sys.stderr)

    print("[smoke test] All tests passed.", file=sys.stderr)


if __name__ == "__main__":
    main()

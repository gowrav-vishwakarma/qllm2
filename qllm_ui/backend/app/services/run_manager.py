"""Manages training run subprocesses."""
from __future__ import annotations

import asyncio
import subprocess
import signal
import uuid
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class RunInfo:
    id: str
    process: Optional[subprocess.Popen] = None
    status: str = "pending"
    output_dir: str = ""
    started_at: float = 0
    log_lines: list[str] = field(default_factory=list)


class RunManager:
    def __init__(self):
        self.runs: dict[str, RunInfo] = {}

    def start_run(self, output_dir: str) -> str:
        run_id = str(uuid.uuid4())[:8]
        run_info = RunInfo(id=run_id, output_dir=output_dir, started_at=time.time())

        train_py = Path(output_dir) / "train.py"
        if not train_py.exists():
            raise FileNotFoundError(f"train.py not found in {output_dir}")

        proc = subprocess.Popen(
            ["python", str(train_py)],
            cwd=output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        run_info.process = proc
        run_info.status = "running"
        self.runs[run_id] = run_info

        return run_id

    def stop_run(self, run_id: str) -> bool:
        run_info = self.runs.get(run_id)
        if not run_info or not run_info.process:
            return False

        try:
            run_info.process.send_signal(signal.SIGINT)
            run_info.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            run_info.process.kill()
        run_info.status = "stopped"
        return True

    def get_status(self, run_id: str) -> dict:
        run_info = self.runs.get(run_id)
        if not run_info:
            return {"status": "not_found"}

        if run_info.process and run_info.status == "running":
            retcode = run_info.process.poll()
            if retcode is not None:
                run_info.status = "completed" if retcode == 0 else "failed"

        return {
            "id": run_info.id,
            "status": run_info.status,
            "output_dir": run_info.output_dir,
            "started_at": run_info.started_at,
        }

    def read_output(self, run_id: str) -> Optional[str]:
        """Read one line from the process stdout (non-blocking)."""
        run_info = self.runs.get(run_id)
        if not run_info or not run_info.process or not run_info.process.stdout:
            return None
        try:
            line = run_info.process.stdout.readline()
            if line:
                run_info.log_lines.append(line.rstrip())
                return line.rstrip()
        except Exception:
            pass
        return None

    def list_runs(self) -> list[dict]:
        result = []
        for run_id in self.runs:
            result.append(self.get_status(run_id))
        return result


run_manager = RunManager()

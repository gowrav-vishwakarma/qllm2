#!/usr/bin/env python3
"""
Small streaming log grapher for V5/V6 training logs.

Runs an aiohttp web app on port 8086 by default and serves a lightweight UI
that tails a selected log file over SSE and draws live charts in the browser.
Supports both V5 format (batch keyword, samples/s) and V6 format (no batch,
tok/s avg, ETA, GPU).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "logs"
INDEX_HTML = Path(__file__).with_name("index.html")
STATIC_DIR = Path(__file__).with_name("vendor")
GPU_HISTORY_LIMIT = 600

TS_PREFIX_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<body>.*)$")
TRAIN_HEADER_RE = re.compile(r"Epochs:\s+\d+\.\.\d+,\s+Batches/epoch:\s+(?P<total>\d+)")

BATCH_RE_NEW = re.compile(
    r"\[(?P<epoch>\d+)\]\s+batch\s+(?P<batch>\d+)/(?P<total>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+ppl=(?P<ppl>[-+0-9.eE]+)\s+"
    r"div=(?P<div>[-+0-9.eE]+)\s+wdiv=(?P<wdiv>[-+0-9.eE]+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+\|\s+inst\s+(?P<inst_tok>[-+0-9.eE]+)\s+tok/s\s+"
    r"avg\s+(?P<avg_tok>[-+0-9.eE]+)\s+tok/s\s+\|\s+"
    r"(?P<progress>[-+0-9.eE]+)%\s+eta\s+(?P<eta>[0-9:]+)"
    r"(?:\s+\|\s+mem\s+(?P<mem_alloc>[-+0-9.eE]+)/(?P<mem_reserved>[-+0-9.eE]+)G\s+"
    r"peak\s+(?P<mem_peak>[-+0-9.eE]+)G)?"
)

BATCH_RE_OLD = re.compile(
    r"\[(?P<epoch>\d+)\]\s+batch\s+(?P<batch>\d+)/(?P<total>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+ppl=(?P<ppl>[-+0-9.eE]+)\s+"
    r"div=(?P<div>[-+0-9.eE]+)"
    r"(?:\s+wdiv=(?P<wdiv>[-+0-9.eE]+))?\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+\|\s+"
    r"(?:(?P<samples_per_sec>[-+0-9.eE]+)\s+samples/s\s+\|\s+)?"
    r"(?P<tok>[-+0-9.eE]+)\s+tok/s"
)

# V6 batch: "  [1] 50/19277 (0%) loss=... ppl=... div=... wdiv=... lr=... | 29377 tok/s (avg 10332) ETA 190m32s | GPU 1.6/17.5GB"
BATCH_RE_V6 = re.compile(
    r"\[\s*(?P<epoch>\d+)\]\s+(?P<batch>\d+)/(?P<total>\d+)\s+\((?P<progress>\d+)%\)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+ppl=(?P<ppl>[-+0-9.eE]+)\s+div=(?P<div>[-+0-9.eE]+)\s+wdiv=(?P<wdiv>[-+0-9.eE]+)\s+lr=(?P<lr>[-+0-9.eE]+)\s+\|\s+"
    r"(?P<inst_tok>[-+0-9.eE]+)\s+tok/s\s+\(avg\s+(?P<avg_tok>[-+0-9.eE]+)\)\s+ETA\s+[\d]+m[\d]+s"
    r"(?:\s+\|\s+GPU\s+(?P<mem_alloc>[-+0-9.eE]+)/(?P<mem_reserved>[-+0-9.eE]+)GB)?"
)

# V6 epoch: "Epoch 1/10 | Train Loss: ... PPL: ... div=... wdiv=... | N tok/s | Time: ...s | Val Loss: ... PPL: ... *best*"
EPOCH_RE_V6 = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+\|\s+"
    r"Train Loss:\s+(?P<train_loss>[-+0-9.eE]+)\s+PPL:\s+(?P<train_ppl>[-+0-9.eE]+)"
    r"(?:\s+div=[-+0-9.eE]+\s+wdiv=[-+0-9.eE]+\s+\|\s+[-+0-9.eE]+\s+tok/s\s+\|)?"
    r"\s+(?:\|\s+)?Time:\s+(?P<time_s>[-+0-9.eE]+)s\s+\|\s+Val Loss:\s+(?P<val_loss>[-+0-9.eE]+)\s+PPL:\s+(?P<val_ppl>[-+0-9.eE]+)\s*(?:\*best\*)?"
)

EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+\|\s+"
    r"Train Loss:\s+(?P<train_loss>[-+0-9.eE]+)\s+PPL:\s+(?P<train_ppl>[-+0-9.eE]+)"
    r"(?:\s+\|\s+Div:\s+(?P<div>[-+0-9.eE]+)\s+\(w\s+(?P<wdiv>[-+0-9.eE]+)\))?"
    r"(?:\s+\|\s+Tok/s:\s+(?P<avg_tok>[-+0-9.eE]+))?"
    r"\s+\|\s+Time:\s+(?P<time_s>[-+0-9.eE]+)s"
    r"(?:\s+\|\s+Val Loss:\s+(?P<val_loss>[-+0-9.eE]+)\s+PPL:\s+(?P<val_ppl>[-+0-9.eE]+))?"
)


@dataclass
class ParserState:
    batches_per_epoch: Optional[int] = None
    epoch_batches: Dict[int, int] = field(default_factory=dict)


@dataclass
class GPUMonitor:
    command: Optional[List[str]]
    command_label: Optional[str]
    gpu_index: int = 0
    sample_interval_s: float = 1.0
    history_limit: int = GPU_HISTORY_LIMIT
    error: Optional[str] = None
    samples: List[Dict[str, Any]] = field(default_factory=list)
    last_sample: Optional[Dict[str, Any]] = None
    last_sample_monotonic: float = 0.0
    seq: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def detect(cls, gpu_index: int = 0) -> "GPUMonitor":
        nvidia_dmon = shutil.which("nvidia-dmon")
        if nvidia_dmon:
            return cls(
                command=[nvidia_dmon, "-s", "pucvmet", "-c", "1"],
                command_label="nvidia-dmon",
                gpu_index=gpu_index,
            )

        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            return cls(
                command=[nvidia_smi, "dmon", "-s", "pucvmet", "-c", "1"],
                command_label="nvidia-smi dmon",
                gpu_index=gpu_index,
            )

        return cls(
            command=None,
            command_label=None,
            gpu_index=gpu_index,
            error="nvidia-dmon / nvidia-smi not found",
        )

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.command is not None,
            "command": self.command_label,
            "gpu_index": self.gpu_index,
            "error": self.error,
        }

    def history(self) -> List[Dict[str, Any]]:
        return list(self.samples)

    async def sample(self) -> Optional[Dict[str, Any]]:
        if self.command is None:
            return None

        now = time.monotonic()
        if self.last_sample is not None and (now - self.last_sample_monotonic) < self.sample_interval_s:
            return self.last_sample

        async with self.lock:
            now = time.monotonic()
            if self.last_sample is not None and (now - self.last_sample_monotonic) < self.sample_interval_s:
                return self.last_sample

            sample = await self._run_once()
            self.last_sample_monotonic = now
            if sample is None:
                return self.last_sample

            self.seq += 1
            sample["seq"] = self.seq
            self.last_sample = sample
            self.samples.append(sample)
            if len(self.samples) > self.history_limit:
                del self.samples[:-self.history_limit]
            self.error = None
            return sample

    async def _run_once(self) -> Optional[Dict[str, Any]]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        except (FileNotFoundError, asyncio.TimeoutError) as exc:
            self.error = str(exc)
            return None
        except Exception as exc:
            self.error = str(exc)
            return None

        if proc.returncode != 0:
            self.error = stderr.decode("utf-8", errors="replace").strip() or f"exit {proc.returncode}"
            return None

        parsed = self._parse_output(stdout.decode("utf-8", errors="replace"))
        if parsed is None:
            self.error = "no parsable dmon row"
        return parsed

    def _parse_output(self, text: str) -> Optional[Dict[str, Any]]:
        rows = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line)

        target = None
        fallback = None
        for line in rows:
            parts = line.split()
            if not parts:
                continue
            if fallback is None:
                fallback = parts
            if parts[0] == str(self.gpu_index):
                target = parts
                break

        parts = target or fallback
        if parts is None or len(parts) < 15:
            return None

        fb_mb = parse_dmon_float(parts, 14)
        return {
            "kind": "gpu",
            "ts_ms": int(time.time() * 1000),
            "gpu_index": int(parse_dmon_float(parts, 0) or self.gpu_index),
            "pwr_w": parse_dmon_float(parts, 1),
            "temp_c": parse_dmon_float(parts, 2),
            "sm_util": parse_dmon_float(parts, 4),
            "mem_util": parse_dmon_float(parts, 5),
            "fb_gb": (fb_mb / 1024.0) if fb_mb is not None else None,
        }


def parse_dmon_float(parts: List[str], index: int) -> Optional[float]:
    if index >= len(parts):
        return None
    token = parts[index]
    if token in {"-", "N/A"}:
        return None
    return to_float(token)


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_line(line: str, state: ParserState) -> Optional[Dict[str, Any]]:
    line = line.rstrip("\n")
    ts = None
    body = line
    m = TS_PREFIX_RE.match(line)
    if m:
        ts = m.group("ts")
        body = m.group("body")

    m = TRAIN_HEADER_RE.search(body)
    if m:
        state.batches_per_epoch = int(m.group("total"))
        return {
            "kind": "meta",
            "ts": ts,
            "batches_per_epoch": state.batches_per_epoch,
        }

    m = BATCH_RE_NEW.search(body)
    if m:
        epoch = int(m.group("epoch"))
        batch = int(m.group("batch"))
        total = int(m.group("total"))
        state.epoch_batches[epoch] = total
        state.batches_per_epoch = total
        return {
            "kind": "batch",
            "ts": ts,
            "epoch": epoch,
            "batch": batch,
            "total_batches": total,
            "step": (epoch - 1) * total + batch + 1,
            "loss": float(m.group("loss")),
            "ppl": float(m.group("ppl")),
            "div": float(m.group("div")),
            "wdiv": float(m.group("wdiv")),
            "lr": float(m.group("lr")),
            "inst_tok_s": float(m.group("inst_tok")),
            "avg_tok_s": float(m.group("avg_tok")),
            "progress": float(m.group("progress")),
            "eta": m.group("eta"),
            "mem_alloc_gb": to_float(m.group("mem_alloc")),
            "mem_reserved_gb": to_float(m.group("mem_reserved")),
            "mem_peak_gb": to_float(m.group("mem_peak")),
        }

    m = BATCH_RE_OLD.search(body)
    if m:
        epoch = int(m.group("epoch"))
        batch = int(m.group("batch"))
        total = int(m.group("total"))
        tok = float(m.group("tok"))
        state.epoch_batches[epoch] = total
        state.batches_per_epoch = total
        return {
            "kind": "batch",
            "ts": ts,
            "epoch": epoch,
            "batch": batch,
            "total_batches": total,
            "step": (epoch - 1) * total + batch + 1,
            "loss": float(m.group("loss")),
            "ppl": float(m.group("ppl")),
            "div": float(m.group("div")),
            "wdiv": to_float(m.group("wdiv")),
            "lr": float(m.group("lr")),
            "inst_tok_s": tok,
            "avg_tok_s": tok,
            "progress": 100.0 * (batch + 1) / total,
            "eta": None,
            "mem_alloc_gb": None,
            "mem_reserved_gb": None,
            "mem_peak_gb": None,
        }

    m = BATCH_RE_V6.search(body)
    if m:
        epoch = int(m.group("epoch"))
        batch = int(m.group("batch"))
        total = int(m.group("total"))
        state.epoch_batches[epoch] = total
        state.batches_per_epoch = total
        return {
            "kind": "batch",
            "ts": ts,
            "epoch": epoch,
            "batch": batch,
            "total_batches": total,
            "step": (epoch - 1) * total + batch + 1,
            "loss": float(m.group("loss")),
            "ppl": float(m.group("ppl")),
            "div": float(m.group("div")),
            "wdiv": float(m.group("wdiv")),
            "lr": float(m.group("lr")),
            "inst_tok_s": float(m.group("inst_tok")),
            "avg_tok_s": float(m.group("avg_tok")),
            "progress": float(m.group("progress")),
            "eta": None,
            "mem_alloc_gb": to_float(m.group("mem_alloc")),
            "mem_reserved_gb": to_float(m.group("mem_reserved")),
            "mem_peak_gb": None,
        }

    m = EPOCH_RE.search(body)
    if m:
        epoch = int(m.group("epoch"))
        total_batches = state.epoch_batches.get(epoch) or state.batches_per_epoch
        step = epoch * total_batches if total_batches is not None else None
        return {
            "kind": "epoch",
            "ts": ts,
            "epoch": epoch,
            "step": step,
            "train_loss": float(m.group("train_loss")),
            "train_ppl": float(m.group("train_ppl")),
            "div": to_float(m.group("div")),
            "wdiv": to_float(m.group("wdiv")),
            "avg_tok_s": to_float(m.group("avg_tok")),
            "time_s": float(m.group("time_s")),
            "val_loss": to_float(m.group("val_loss")),
            "val_ppl": to_float(m.group("val_ppl")),
        }

    m = EPOCH_RE_V6.search(body)
    if m:
        epoch = int(m.group("epoch"))
        total_batches = state.epoch_batches.get(epoch) or state.batches_per_epoch
        step = epoch * total_batches if total_batches is not None else None
        return {
            "kind": "epoch",
            "ts": ts,
            "epoch": epoch,
            "step": step,
            "train_loss": float(m.group("train_loss")),
            "train_ppl": float(m.group("train_ppl")),
            "div": None,
            "wdiv": None,
            "avg_tok_s": None,
            "time_s": float(m.group("time_s")),
            "val_loss": to_float(m.group("val_loss")),
            "val_ppl": to_float(m.group("val_ppl")),
        }

    return None


def list_logs(log_dir: Path) -> List[Dict[str, Any]]:
    files = []
    for path in sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        files.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return files


def resolve_log_file(log_dir: Path, name: Optional[str]) -> Path:
    logs = list_logs(log_dir)
    if not logs:
        raise web.HTTPNotFound(text=f"No .log files found in {log_dir}")
    if name is None:
        return log_dir / logs[0]["name"]
    candidate = (log_dir / Path(name).name).resolve()
    if candidate.parent != log_dir.resolve() or not candidate.exists():
        raise web.HTTPNotFound(text=f"Log file not found: {name}")
    return candidate


def parse_log_file(path: Path) -> Dict[str, Any]:
    state = ParserState()
    batches: List[Dict[str, Any]] = []
    epochs: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_line(line, state)
            if parsed is None:
                continue
            if parsed["kind"] == "batch":
                batches.append(parsed)
            elif parsed["kind"] == "epoch":
                epochs.append(parsed)

    return {
        "file": path.name,
        "batches_per_epoch": state.batches_per_epoch,
        "batches": batches,
        "epochs": epochs,
    }


async def index(_: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML.read_text(encoding="utf-8"), content_type="text/html")


async def api_logs(request: web.Request) -> web.Response:
    log_dir: Path = request.app["log_dir"]
    return web.json_response({"logs": list_logs(log_dir)})


async def api_history(request: web.Request) -> web.Response:
    log_dir: Path = request.app["log_dir"]
    path = resolve_log_file(log_dir, request.query.get("file"))
    gpu_monitor: GPUMonitor = request.app["gpu_monitor"]
    await gpu_monitor.sample()
    payload = parse_log_file(path)
    payload["gpu_samples"] = gpu_monitor.history()
    payload["gpu_monitor"] = gpu_monitor.status()
    return web.json_response(payload)


async def sse_events(request: web.Request) -> web.StreamResponse:
    log_dir: Path = request.app["log_dir"]
    gpu_monitor: GPUMonitor = request.app["gpu_monitor"]
    path = resolve_log_file(log_dir, request.query.get("file"))

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    state = ParserState()
    offset = path.stat().st_size if path.exists() else 0
    last_gpu_seq = 0

    async def send(event: str, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"))
        await response.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))

    await send("status", {"state": "connected", "file": path.name})

    try:
        while True:
            if request.transport is None or request.transport.is_closing():
                break

            if not path.exists():
                await asyncio.sleep(1.0)
                continue

            size = path.stat().st_size
            if size < offset:
                offset = 0
                state = ParserState()
                await send("reset", {"reason": "truncated"})

            if size > offset:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(offset)
                    for line in f:
                        parsed = parse_line(line, state)
                        if parsed is not None:
                            await send("log", parsed)
                    offset = f.tell()

            gpu_sample = await gpu_monitor.sample()
            if gpu_sample is not None and gpu_sample.get("seq", 0) > last_gpu_seq:
                last_gpu_seq = gpu_sample["seq"]
                await send("gpu", gpu_sample)

            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await response.write_eof()
        except Exception:
            pass

    return response


def build_app(log_dir: Path, gpu_index: int = 0) -> web.Application:
    app = web.Application()
    app["log_dir"] = log_dir
    app["gpu_monitor"] = GPUMonitor.detect(gpu_index=gpu_index)
    app.router.add_get("/", index)
    app.router.add_static("/static/", STATIC_DIR)
    app.router.add_get("/api/logs", api_logs)
    app.router.add_get("/api/history", api_history)
    app.router.add_get("/events", sse_events)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming log grapher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()

    app = build_app(args.log_dir.resolve(), gpu_index=args.gpu_index)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

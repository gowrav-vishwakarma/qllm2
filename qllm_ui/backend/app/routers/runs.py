"""Training run management endpoints."""

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from app.services.run_manager import run_manager

router = APIRouter()


class StartRunRequest(BaseModel):
    output_dir: str


@router.post("/start")
async def start_run(req: StartRunRequest):
    try:
        run_id = run_manager.start_run(req.output_dir)
        return {"runId": run_id}
    except Exception as e:
        return {"error": str(e)}


@router.post("/{run_id}/stop")
async def stop_run(run_id: str):
    success = run_manager.stop_run(run_id)
    return {"status": "stopped" if success else "not_found"}


@router.get("/{run_id}/status")
async def get_status(run_id: str):
    return run_manager.get_status(run_id)


@router.get("/list")
async def list_runs():
    return run_manager.list_runs()


@router.websocket("/ws/runs/{run_id}/logs")
async def ws_logs(websocket: WebSocket, run_id: str):
    await websocket.accept()
    try:
        while True:
            status = run_manager.get_status(run_id)
            if status["status"] in ("completed", "failed", "stopped", "not_found"):
                remaining = True
                while remaining:
                    line = run_manager.read_output(run_id)
                    if line:
                        await websocket.send_text(line)
                    else:
                        remaining = False
                await websocket.send_text(f"[STATUS] Run {status['status']}")
                break

            line = run_manager.read_output(run_id)
            if line:
                await websocket.send_text(line)
            else:
                await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

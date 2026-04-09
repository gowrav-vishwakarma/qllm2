"""Save/load project JSON files."""

import json
from pathlib import Path
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter()

PROJECTS_DIR = Path("./projects")
PROJECTS_DIR.mkdir(exist_ok=True)


@router.post("/save")
async def save_project(data: dict):
    name = data.get("project", {}).get("name", "untitled")
    filename = f"{name}.json"
    filepath = PROJECTS_DIR / filename
    filepath.write_text(json.dumps(data, indent=2))
    return {"path": str(filepath)}


@router.get("/load")
async def load_project(path: str = Query(...)):
    filepath = Path(path)
    if not filepath.exists():
        filepath = PROJECTS_DIR / path
    if not filepath.exists():
        return {"error": "File not found"}
    data = json.loads(filepath.read_text())
    return data


@router.get("/list")
async def list_projects():
    files = sorted(PROJECTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in files]

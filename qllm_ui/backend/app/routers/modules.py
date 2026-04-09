"""Module registry and port inference API."""

from fastapi import APIRouter
from pydantic import BaseModel
from app.services.port_parser import infer_ports

router = APIRouter()


class InferPortsRequest(BaseModel):
    code: str


@router.post("/infer-ports")
async def infer_ports_endpoint(req: InferPortsRequest):
    try:
        result = infer_ports(req.code)
        return result
    except Exception as e:
        return {"error": str(e), "inputs": [], "outputs": [], "constructorParams": []}


@router.get("/list")
async def list_modules():
    return {"message": "Use frontend built-in registry; this endpoint is for future server-side modules"}

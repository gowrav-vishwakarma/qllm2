"""Code generation endpoint."""

from fastapi import APIRouter
from app.services.code_generator import generate_code

router = APIRouter()


@router.post("/generate")
async def generate(data: dict):
    try:
        result = generate_code(data)
        return result
    except Exception as e:
        return {"error": str(e), "outputDir": "", "files": []}

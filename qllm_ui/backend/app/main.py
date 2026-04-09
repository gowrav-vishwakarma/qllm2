from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import projects, modules, codegen, runs

app = FastAPI(title="QLLM Architecture Builder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(modules.router, prefix="/api/modules", tags=["modules"])
app.include_router(codegen.router, prefix="/api/codegen", tags=["codegen"])
app.include_router(runs.router, prefix="/api/runs", tags=["runs"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}

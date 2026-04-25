"""FastAPI entrypoint — run from backend/ with: uvicorn app.main:app --reload --port 8000"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, models, predict

app = FastAPI(
    title="Gearbox Fault Diagnosis API",
    version="0.1.0",
    description="MVP API for gearbox fault inference.",
)

# Dev-only CORS (frontend Vite dev server). Adjust for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(predict.router, prefix="/api", tags=["predict"])

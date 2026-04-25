from fastapi import APIRouter
import torch

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        cuda=torch.cuda.is_available(),
        torch_version=torch.__version__,
    )

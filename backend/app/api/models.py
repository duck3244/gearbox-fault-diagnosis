from fastapi import APIRouter, Depends, HTTPException

from app.schemas import ModelMeta
from app.services.predictor_pool import PredictorPool, get_predictor_pool

router = APIRouter()


@router.get("/models", response_model=list[str])
async def list_models(pool: PredictorPool = Depends(get_predictor_pool)) -> list[str]:
    return pool.list_models()


@router.get("/models/{model_id}/meta", response_model=ModelMeta)
async def get_model_meta(
    model_id: str,
    pool: PredictorPool = Depends(get_predictor_pool),
) -> ModelMeta:
    try:
        cached = await pool.get(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"model_id not found: {model_id}")

    p = cached.predictor
    return ModelMeta(
        model_id=model_id,
        model_type=p.model_type,
        input_size=p.input_size,
        num_classes=p.num_classes,
        label_names=list(p.label_names),
        val_acc=p.val_acc,
        epoch=p.epoch,
    )

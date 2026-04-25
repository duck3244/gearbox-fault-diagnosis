"""Pydantic v2 schemas for the Gearbox Fault Diagnosis API."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


# Pydantic v2에서 'model_*' 접두 필드명은 protected namespace 경고가 있음.
# 여기서는 명확성을 위해 model_id / model_type을 계속 쓰되, 경고만 끈다.
_CFG = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    model_config = _CFG

    status: Literal["ok"] = "ok"
    cuda: bool
    torch_version: str


class ModelMeta(BaseModel):
    model_config = _CFG

    model_id: str
    model_type: Literal["CNN", "ResNet", "MLP"]
    input_size: int
    num_classes: int
    label_names: list[str]
    val_acc: Optional[float] = None
    epoch: Optional[int] = None


class PredictionItem(BaseModel):
    model_config = _CFG

    sample_index: int
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model_config = _CFG

    model_id: str
    model_type: str
    input_size: int
    n_samples: int
    predictions: list[PredictionItem]
    elapsed_ms: float


class HistoryEntry(BaseModel):
    model_config = _CFG

    timestamp: str
    model_id: str
    n_samples: int
    top_class: str
    elapsed_ms: float

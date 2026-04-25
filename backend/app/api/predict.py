"""POST /predict + GET /history.

Defensive behaviors:
- A3: raw signal-only parsing. No column-name auto-detection or dropping.
- A4: 20 MB upload cap, chunked read (never materialize beyond the limit).
- A6: explicit input-dimension check against scaler.mean_.shape[0] → 400.
- A5: per-predictor asyncio.Lock serializes inference, run_in_threadpool offloads
  the blocking torch call off the event loop.
"""

from __future__ import annotations

import io
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas import HistoryEntry, PredictResponse, PredictionItem
from app.services.predictor_pool import PredictorPool, get_predictor_pool

router = APIRouter()

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_SUFFIXES = {".csv", ".npy"}
HISTORY_MAXLEN = 100
HISTORY: "deque[HistoryEntry]" = deque(maxlen=HISTORY_MAXLEN)


async def _read_upload_capped(file: UploadFile) -> bytes:
    chunk_size = 1 << 20  # 1 MB
    buf = io.BytesIO()
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )
        buf.write(chunk)
    return buf.getvalue()


def _suffix(filename: Optional[str]) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _parse_signal(filename: Optional[str], data: bytes, expected_dim: int) -> np.ndarray:
    suffix = _suffix(filename)
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix or '<none>'!r} (allowed: .csv, .npy)",
        )

    if suffix == ".npy":
        try:
            # allow_pickle=False → avoid code-execution via crafted .npy
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to parse .npy: {e}")
    else:
        try:
            df = pd.read_csv(io.BytesIO(data), header=None)
            arr = df.to_numpy()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"failed to parse .csv: {e}")

    if not np.issubdtype(arr.dtype, np.number):
        raise HTTPException(status_code=400, detail=f"non-numeric data in upload (dtype={arr.dtype})")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"expected 1D or 2D array, got ndim={arr.ndim}",
        )
    if arr.shape[1] != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"input dimension mismatch: got {arr.shape[1]}, "
                f"model expects {expected_dim}"
            ),
        )
    return arr.astype(np.float32, copy=False)


def _majority_class(items: list[PredictionItem]) -> str:
    counts: dict[str, int] = {}
    for it in items:
        counts[it.predicted_class] = counts.get(it.predicted_class, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


@router.post("/predict", response_model=PredictResponse)
async def predict(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    pool: PredictorPool = Depends(get_predictor_pool),
) -> PredictResponse:
    try:
        cached = await pool.get(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"model_id not found: {model_id}")

    predictor = cached.predictor
    data = await _read_upload_capped(file)
    signals = _parse_signal(file.filename, data, expected_dim=predictor.input_size)

    start = time.perf_counter()
    async with cached.lock:
        results = await run_in_threadpool(predictor.predict_batch, signals)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    items = [
        PredictionItem(
            sample_index=i,
            predicted_class=r["predicted_class"],
            predicted_index=r["predicted_index"],
            confidence=r["confidence"],
            probabilities=r["probabilities"],
        )
        for i, r in enumerate(results)
    ]

    HISTORY.appendleft(
        HistoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            model_id=model_id,
            n_samples=len(items),
            top_class=_majority_class(items) if items else "",
            elapsed_ms=elapsed_ms,
        )
    )

    return PredictResponse(
        model_id=model_id,
        model_type=predictor.model_type,
        input_size=predictor.input_size,
        n_samples=len(items),
        predictions=items,
        elapsed_ms=elapsed_ms,
    )


@router.get("/history", response_model=list[HistoryEntry])
async def history(limit: int = 20) -> list[HistoryEntry]:
    limit = max(1, min(limit, HISTORY_MAXLEN))
    return list(HISTORY)[:limit]

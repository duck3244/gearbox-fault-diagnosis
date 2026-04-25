"""Predictor cache with LRU eviction and per-model inference lock.

Responsibilities:
- Validate model_id against a strict format (A1 path-traversal defense).
- Resolve <results>/<model_id>/best_model.pth and ensure it stays within results/.
- Cache GearboxPredictor instances (expensive torch.load + GPU transfer).
- Serialize inference on a single predictor via asyncio.Lock (A5 CUDA OOM / races).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Re-use the existing CLI-oriented GearboxPredictor. Import path works because
# uvicorn is launched from backend/, putting it on sys.path.
from inference import GearboxPredictor  # type: ignore  # noqa: E402

MODEL_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")
DEFAULT_MAX_CACHE = 4


@dataclass
class CachedPredictor:
    predictor: GearboxPredictor
    lock: asyncio.Lock


class PredictorPool:
    def __init__(self, results_dir: Path, max_size: int = DEFAULT_MAX_CACHE):
        self.results_dir = results_dir.resolve()
        self.max_size = max_size
        self._cache: dict[str, CachedPredictor] = {}
        self._order: list[str] = []
        self._registry_lock = asyncio.Lock()

    # ---- public ------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return discovered model_ids sorted newest-first. No filesystem state is cached."""
        if not self.results_dir.exists():
            return []
        ids: list[str] = []
        for d in sorted(self.results_dir.iterdir(), reverse=True):
            if (
                d.is_dir()
                and MODEL_ID_PATTERN.match(d.name)
                and (d / "best_model.pth").is_file()
            ):
                ids.append(d.name)
        return ids

    async def get(self, model_id: str) -> CachedPredictor:
        """Return cached predictor (loading on miss). Raises ValueError / FileNotFoundError."""
        path = self._resolve(model_id)

        async with self._registry_lock:
            cached = self._cache.get(model_id)
            if cached is not None:
                self._touch(model_id)
                return cached

            # Loading torch is blocking; do it inside the registry lock to avoid
            # duplicate concurrent loads of the same checkpoint.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = GearboxPredictor(str(path), device=device)
            entry = CachedPredictor(predictor=predictor, lock=asyncio.Lock())
            self._cache[model_id] = entry
            self._order.append(model_id)
            self._evict()
            return entry

    # ---- internal ----------------------------------------------------------

    def _resolve(self, model_id: str) -> Path:
        if not MODEL_ID_PATTERN.match(model_id):
            raise ValueError(f"invalid model_id format: {model_id!r}")
        candidate = (self.results_dir / model_id / "best_model.pth").resolve()
        try:
            candidate.relative_to(self.results_dir)
        except ValueError as e:
            raise ValueError(f"model path escapes results root: {model_id!r}") from e
        if not candidate.is_file():
            raise FileNotFoundError(f"model not found: {model_id}")
        return candidate

    def _touch(self, model_id: str) -> None:
        self._order.remove(model_id)
        self._order.append(model_id)

    def _evict(self) -> None:
        while len(self._order) > self.max_size:
            victim = self._order.pop(0)
            self._cache.pop(victim, None)


# Single app-wide pool. Using lazy init so tests can override easily.
_POOL: Optional[PredictorPool] = None


def get_predictor_pool() -> PredictorPool:
    global _POOL
    if _POOL is None:
        # backend/ == parents[2] of this file
        backend_root = Path(__file__).resolve().parents[2]
        _POOL = PredictorPool(results_dir=backend_root / "results")
    return _POOL

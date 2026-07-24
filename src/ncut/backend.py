from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

_gpu_warmed = False


class Backend(Enum):
    CPU = "cpu"
    GPU = "gpu"

    @classmethod
    def from_string(cls, value: Any) -> "Backend":
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower() if value is not None else ""
        if normalized == "cpu" or normalized == "":
            return cls.CPU
        if normalized == "gpu":
            return cls.GPU

        raise ValueError(
            f"Unknown backend: {value!r}. Expected 'cpu' or 'gpu'."
        )


def is_gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:
        return False


def get_array_module(backend: Backend) -> Any:
    backend = Backend.from_string(backend)

    if backend == Backend.GPU:
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "CuPy is not installed. Install a CUDA-enabled CuPy build to use the GPU backend."
            ) from exc

        if not cp.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Ensure the NVIDIA driver and CUDA runtime are installed."
            )

        return cp

    return np


def warm_gpu_backend() -> bool:
    global _gpu_warmed

    if _gpu_warmed:
        return True

    try:
        import cupy as cp

        if not cp.cuda.is_available():
            return False

        warmup = cp.asarray([0.0], dtype=cp.float64)
        _ = cp.exp(warmup)
        cp.cuda.Stream.null.synchronize()
        _gpu_warmed = True
        return True
    except Exception:
        return False


def to_numpy(array: Any) -> Any:
    try:
        import cupy as cp
    except Exception:
        cp = None

    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)

    return array

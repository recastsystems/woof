"""The CUDA forecast requirement, shared by CLI and direct model callers.

CPU preprocessing and NumPy setup states are not a CPU forecast dycore.
This boundary prevents their reaching CUDA allocation/launch failures after
preparation, or a downstream caller mistaking a setup state for a CPU solver.
"""

from __future__ import annotations

import importlib
import os

from woof.local_gpu import no_local_gpu


FORECAST_GPU_REQUIREMENT = (
    "Forecasts require an NVIDIA GPU and a driver supporting CUDA 12.x "
    "with pip install 'recast-woof[gpu-cu12]' (CuPy >=14.0 and its CUDA toolkit); "
    "on a CUDA-13-only system use 'recast-woof[gpu-cu13]' instead. "
    "There is no CPU forecast step. VRAM depends on the domain and physics; "
    "use woof check to size the run. "
    "Without a GPU you can fetch, prepare (woof prep --preprocess-backend "
    "cpu), run woof doctor, render existing output, and extract soundings.")


class ForecastGPURequired(RuntimeError):
    """A forecast cannot start; ``detected`` holds the observed runtime gap.

    Direct callers may catch this exception and display its message. CLI
    commands display the same message and return 2 without a traceback.
    """

    def __init__(self, detected: str):
        self.detected = detected
        super().__init__(
            f"ForecastGPURequired: {FORECAST_GPU_REQUIREMENT} Detected: {detected}.")


# Successful checks only: no device probe on the stepping hot path, and a
# failed check can recover after the caller repairs its environment.
_working_runtime = None


def forecast_gpu_gap() -> str | None:
    """Return a concrete gap, or None when CUDA can see an NVIDIA device.

    Does not compile kernels or allocate arrays. Doctor retains its deeper
    isolated compiler/cuBLAS checks. The explicit local-device opt-out is
    honoured before any CUDA contact.
    """
    global _working_runtime
    if no_local_gpu():
        return "local GPU access disabled by GPUWM_NO_LOCAL_GPU"
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is not None and mask.strip() in ('', '-1'):
        return f"CUDA_VISIBLE_DEVICES={mask!r} hides all NVIDIA devices"
    if _working_runtime is not None:
        return None
    try:
        cp = importlib.import_module('cupy')
    except Exception as error:
        return f"CuPy unavailable ({type(error).__name__}: {error})"
    from woof.capabilities import GPU_RUNTIME_MINIMUM, installed_cupy_version

    version = installed_cupy_version()
    if version is not None and version < GPU_RUNTIME_MINIMUM:
        return f"CuPy {version[0]}.{version[1]} is older than the required 14.0"
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as error:
        return f"CUDA driver/device unavailable ({type(error).__name__}: {error})"
    if count < 1:
        return "CuPy loaded, but CUDA reports 0 NVIDIA devices"
    _working_runtime = cp
    return None


def require_forecast_gpu() -> None:
    """Raise :class:`ForecastGPURequired` before forecast allocation or step."""
    gap = forecast_gpu_gap()
    if gap is not None:
        raise ForecastGPURequired(gap) from None


__all__ = ['ForecastGPURequired', 'require_forecast_gpu']

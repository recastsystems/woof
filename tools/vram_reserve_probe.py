"""Measure, per card and driver model, every grid-independent term the
VRAM fit gate charges.

Three terms, three separate measurements, one process each so nothing
contaminates the next:

* ``context``    -- device bytes a bare CUDA context holds before any
                    woof allocation.  :data:`woof.core.preflight.
                    CUDA_CONTEXT_BYTES` claims 432 MiB for every card on
                    every platform from one 2026-07-26 WDDM/5090 reading.
* ``law``        -- the launch-time local-memory backing store, measured
                    against a SYNTHETIC kernel at several known frame
                    sizes.  Validates the reservation law BOTH
                    DIRECTIONS: a frame inside the default stack must
                    step ~zero, and a frame above it must step exactly
                    ``(frame - stack) x SMs x threads/SM``.
* ``frames``     -- the widest per-thread local frame the real shipped
                    kernel modules compile to, read off the driver.

Run one mode per process::

    python tools/vram_reserve_probe.py context
    python tools/vram_reserve_probe.py law
    python tools/vram_reserve_probe.py frames

Each prints one JSON object on the last line.  ``all`` runs the three in
fresh subprocesses and merges them, which is the only correct way to read
``context`` (a CUDA context cannot be torn down inside a process) and the
only correct way to read the default stack limit (the runtime RAISES
``cudaLimitStackSize`` for the life of a process the first time a fatter
kernel loads).
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIB = 1024 ** 2


def _smi_used_bytes() -> int | None:
    """Device-wide used bytes from NVML, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0]) * MIB
    except Exception:
        return None


def _host() -> dict:
    return {
        "platform": sys.platform,
        "node": platform.node(),
        "python": sys.version.split()[0],
    }


def _device(cp) -> dict:
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return {
        "name": name.decode() if isinstance(name, bytes) else str(name),
        "multiprocessor_count": int(props["multiProcessorCount"]),
        "max_threads_per_multiprocessor": int(
            props["maxThreadsPerMultiProcessor"]),
        "total_bytes": int(cp.cuda.runtime.memGetInfo()[1]),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cupy": cp.__version__,
    }


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

def mode_context() -> dict:
    """Device cost of a bare CUDA context, measured two ways.

    ``memGetInfo`` deltas are exact on every platform this ships to; the
    NVML device-wide figure additionally sees other processes, which is
    what a WDDM desktop spends and what an idle Linux card does not have.
    """

    smi_before = _smi_used_bytes()
    import cupy as cp

    # Nothing has touched the device yet in this process.  Force context
    # creation with the smallest possible allocation, then free it: the
    # context stays, the byte does not.
    free_after_boot = None
    total = None
    buf = cp.cuda.alloc(1)
    free_after_boot, total = cp.cuda.runtime.memGetInfo()
    del buf
    cp.get_default_memory_pool().free_all_blocks()
    free_settled, _ = cp.cuda.runtime.memGetInfo()
    smi_after = _smi_used_bytes()
    stack_limit = int(cp.cuda.runtime.deviceGetLimit(0))

    return {
        "mode": "context",
        "host": _host(),
        "device": _device(cp),
        "default_stack_limit_bytes": stack_limit,
        # total - free, with the process's own single byte released
        "context_bytes_memgetinfo": int(total - free_settled),
        "context_bytes_at_first_alloc": int(total - free_after_boot),
        "smi_used_before_bytes": smi_before,
        "smi_used_after_bytes": smi_after,
        "context_bytes_nvml": (None if (smi_before is None or smi_after is None)
                               else int(smi_after - smi_before)),
    }


# ---------------------------------------------------------------------------
# law
# ---------------------------------------------------------------------------

_SYNTH = r"""
extern "C" __global__ void frame_kernel(float* out, int n, int stride) {
    float buf[NFLOATS];
    #pragma unroll 1
    for (int i = 0; i < NFLOATS; ++i) buf[i] = out[0] + (float)i;
    float s = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < NFLOATS; ++i) s += buf[(i * stride) % NFLOATS];
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = s + (float)n;
}
"""


def mode_law() -> dict:
    """Measure the local-memory backing store at several frame sizes.

    The instrument is validated in both directions in one sweep: the
    ``nfloats=1`` row compiles to a frame at or under the default stack
    and MUST step ~zero, and the fat rows must step the law's product.
    A step that appears for every frame size would be the instrument
    measuring something else; a step that never appears would be an
    instrument that cannot see the term at all.
    """

    import cupy as cp

    stack_limit = int(cp.cuda.runtime.deviceGetLimit(0))
    props = cp.cuda.runtime.getDeviceProperties(0)
    capacity = (int(props["multiProcessorCount"])
                * int(props["maxThreadsPerMultiProcessor"]))

    out = cp.zeros(1024, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    rows = []
    # 1 float  -> frame under the default stack (negative control)
    # the rest -> frames spanning the range the shipped kernel set uses
    #             (176 B .. 24,064 B) and beyond it
    for nfloats in (1, 64, 256, 1024, 2816, 6016, 8192):
        src = _SYNTH.replace("NFLOATS", str(nfloats))
        module = cp.RawModule(code=src, options=("-std=c++17",))
        fn = module.get_function("frame_kernel")
        frame = int(fn.attributes["local_size_bytes"])
        free_before, total = cp.cuda.runtime.memGetInfo()
        smi_before = _smi_used_bytes()
        fn((64,), (256,), (out, 1, 7))
        cp.cuda.runtime.deviceSynchronize()
        free_after, _ = cp.cuda.runtime.memGetInfo()
        smi_after = _smi_used_bytes()
        predicted = max(0, frame - stack_limit) * capacity
        rows.append({
            "nfloats": nfloats,
            "frame_bytes": frame,
            "measured_step_bytes": int(free_before - free_after),
            "predicted_step_bytes": int(predicted),
            "smi_step_bytes": (None if (smi_before is None or smi_after is None)
                               else int(smi_after - smi_before)),
        })

    return {
        "mode": "law",
        "host": _host(),
        "device": _device(cp),
        "default_stack_limit_bytes": stack_limit,
        "resident_thread_capacity": capacity,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

_SYMBOL = re.compile(
    r'extern\s+"C"\s+__global__\s+'
    r'(?:__launch_bounds__\s*\([^)]*\)\s*)?'
    r'void\s+([A-Za-z_][A-Za-z0-9_]*)')


def mode_frames() -> dict:
    """Widest per-thread local frame per shipped kernel module.

    The frame is a property of the COMPILER and the source, not of the
    card, so this is expected to agree across cards; it is measured on
    each anyway, because "expected to" is what a calibration is for.
    """

    import cupy as cp  # noqa: F401  -- load_module needs it

    from woof.core.kernels import load_module, kernel_local_size_bytes
    from woof.core.preflight import (PRODUCTION_LOADED_KERNEL_UNITS,
                                      load_production_kernel_unit)

    kdir = ROOT / "woof" / "core" / "kernels"
    frames: dict[str, int] = {}
    uncompilable: list[str] = []
    for path in sorted(kdir.glob("*.cu")):
        try:
            if path.stem in PRODUCTION_LOADED_KERNEL_UNITS:
                # Compiled only through its own loader; measure that image.
                module, text = load_production_kernel_unit(path.stem)
            else:
                module, text = load_module(path.stem), path.read_text()
        except Exception:  # noqa: BLE001
            uncompilable.append(path.stem)
            continue
        widest = 0
        for name in sorted(set(_SYMBOL.findall(text))):
            try:
                local_bytes = kernel_local_size_bytes(module.get_function(name))
            except Exception:  # noqa: BLE001
                continue
            widest = max(widest, local_bytes)
        frames[path.stem] = widest
    return {
        "mode": "frames",
        "host": _host(),
        "device": _device(cp),
        "frames": frames,
        "uncompilable": sorted(uncompilable),
    }


MODES = {"context": mode_context, "law": mode_law, "frames": mode_frames}


def mode_all() -> dict:
    merged = {"mode": "all", "host": _host(), "parts": {}}
    for name in ("context", "law", "frames"):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), name],
            capture_output=True, text=True, timeout=3600,
            cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                except Exception:  # noqa: BLE001
                    payload = None
                break
        merged["parts"][name] = payload if payload is not None else {
            "error": proc.stderr[-4000:], "returncode": proc.returncode}
    return merged


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "all"
    fn = MODES.get(mode)
    payload = mode_all() if mode == "all" else (fn() if fn else None)
    if payload is None:
        print(f"unknown mode {mode!r}; one of {sorted(MODES)} or 'all'",
              file=sys.stderr)
        return 2
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

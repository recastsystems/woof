"""Thin metadata and CUDA-driver seam to CU99's Rust scheduler.

Only device addresses and scalar launch metadata cross this ABI. The
Rust loop retains the bounded workspace policy without a Python callback
or a host readback between tiles. Field arithmetic stays in kf_old.cu.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

ABI_VERSION = 2
BRIDGE_ENV = "WOOF_CU99_LAUNCH_BRIDGE"
_LIBRARY = None
_DRIVER = None


class Arguments(ctypes.Structure):
    _fields_ = [("pointers", ctypes.c_uint64 * 28),
                ("scalars", ctypes.c_float * 3),
                ("options", ctypes.c_int32 * 10)]


class StepPlan(ctypes.Structure):
    _fields_ = [("clock_dt", ctypes.c_double), ("stepcu", ctypes.c_int64),
                ("column_stepcu", ctypes.c_int64), ("due", ctypes.c_int32),
                ("advance", ctypes.c_int32)]


def plan_step(*, elapsed, epoch=0.0, dt, clock_dt=0.0, cudt_minutes,
              due_override=None):
    """Marshal clock metadata; Rust owns every CU99 cadence decision.

    Python only chooses which CUDA calls to submit from the returned plan.
    No device value, completion or field arithmetic enters this seam.
    """
    lib = load()
    result = StepPlan()
    status = lib.gpuwm_cu99_plan_step(float(elapsed), float(epoch), float(dt),
        float(clock_dt or 0.0), float(cudt_minutes),
        -1 if due_override is None else int(bool(due_override)), ctypes.byref(result))
    if status:
        raise ValueError(f"CU99 Rust scheduler rejected clock metadata ({status})")
    return result


def library_name():
    if os.name == "nt":
        return "rw_cu99_launch.dll"
    if os.uname().sysname == "Darwin":
        return "librw_cu99_launch.dylib"
    return "librw_cu99_launch.so"


def resolve():
    from woof.bridges import default_bridge_dir, packaged_bridge_dir
    filename = library_name()
    override = os.environ.get(BRIDGE_ENV)
    root = Path(__file__).resolve().parents[2]
    candidates = ([Path(override)] if override else []) + [
        root / "tools/rustwx/target/release" / filename,
        root / "tools/rustwx/target/debug" / filename,
        root / "libexec/bridges" / filename,
        packaged_bridge_dir() / filename, default_bridge_dir() / filename]
    for path in candidates:
        if path.is_file():
            return path.resolve()
        if override:
            raise FileNotFoundError(f"{BRIDGE_ENV} names a missing CU99 Rust bridge: {path}")
    raise FileNotFoundError("CU99 Rust launch bridge is missing; build it with "
        "cargo build --manifest-path tools/rustwx/Cargo.toml --release --locked --offline -p rw-cu99-launch")


def load():
    global _LIBRARY
    if _LIBRARY is None:
        lib = ctypes.CDLL(str(resolve()))
        lib.gpuwm_cu99_launch_abi_version.restype = ctypes.c_uint32
        if lib.gpuwm_cu99_launch_abi_version() != ABI_VERSION:
            raise RuntimeError("CU99 Rust launch ABI mismatch; rebuild rw-cu99-launch")
        lib.gpuwm_cu99_launch_tiles.restype = ctypes.c_int32
        lib.gpuwm_cu99_launch_tiles.argtypes = (ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.POINTER(Arguments), ctypes.c_int32)
        lib.gpuwm_cu99_plan_step.restype = ctypes.c_int32
        lib.gpuwm_cu99_plan_step.argtypes = (ctypes.c_double,) * 5 + (
            ctypes.c_int32, ctypes.POINTER(StepPlan))
        _LIBRARY = lib
    return _LIBRARY


def _cuda_driver():
    global _DRIVER
    if _DRIVER is None:
        # Resolve CUDA only for submission. The native scalar scheduler
        # also runs on CPU build/test hosts without an installed driver.
        # A global symbol preserves an explicitly preloaded recorder.
        driver = ctypes.CDLL(None) if os.name != "nt" else ctypes.WinDLL("nvcuda.dll")
        if not hasattr(driver, "cuLaunchKernel"):
            driver = ctypes.CDLL("libcuda.so.1")
        _DRIVER = driver
    return _DRIVER


def launch_tiles(kernel, arrays, scalars, options, *, tile, stream):
    lib = load()
    if len(arrays) != 28 or len(scalars) != 3 or len(options) != 10:
        raise ValueError("CU99 launch argument layout changed; update both sides of the Rust ABI")
    function = getattr(kernel, "kernel", kernel)
    args = Arguments((ctypes.c_uint64 * 28)(*(int(array.data.ptr) for array in arrays)),
        (ctypes.c_float * 3)(*scalars), (ctypes.c_int32 * 10)(*options))
    result = lib.gpuwm_cu99_launch_tiles(
        ctypes.cast(_cuda_driver().cuLaunchKernel, ctypes.c_void_p), int(function.ptr),
        int(stream.ptr), ctypes.byref(args), int(tile))
    if result > 0:
        from cupy.cuda.driver import CUDADriverError
        raise CUDADriverError(result)
    if result < 0:
        raise ValueError(f"CU99 Rust launch rejected metadata ({result}) before submission")

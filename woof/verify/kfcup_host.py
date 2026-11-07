"""The KF-CuP CPU reference: ``kfcup.cu`` compiled as C++ and called by ctypes.

``kfcup_host.cpp`` includes the exact string the kernel loader hands NVRTC
(``woof.core.kernels.module_source("kfcup")``), so this reference and the
CUDA kernel are one source, compiled twice.  The library is built on demand
with the host C++ compiler at ``-O2 -ffp-contract=off`` (IEEE binary32 per
operation, no contraction: the device's strict arm) into a cache directory
keyed by the source digest; ``source=`` builds a variant from an edited
string, which is how the planted-mutation tests prove the grader can fail.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIBS: dict[str, ctypes.CDLL] = {}


def _cache_dir() -> Path:
    root = os.environ.get("WOOF_KFCUP_HOST_CACHE")
    path = Path(root) if root else Path(tempfile.gettempdir()) / "gpuwm-kfcup-host"
    path.mkdir(parents=True, exist_ok=True)
    return path


def host_library(source: str | None = None) -> ctypes.CDLL:
    """Build (once per source digest) and load the CPU reference library."""
    if source is None:
        from woof.core.kernels import module_source
        source = module_source("kfcup")
    harness = (_HERE / "kfcup_host.cpp").read_text(encoding="utf-8")
    digest = hashlib.sha256((source + "\0" + harness).encode("utf-8")).hexdigest()[:20]
    if digest in _LIBS:
        return _LIBS[digest]
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if cxx is None:
        raise RuntimeError(
            "the KF-CuP CPU reference needs a host C++ compiler (g++); none is on PATH")
    work = _cache_dir() / digest
    lib = work / "libkfcup_host.so"
    if not lib.exists():
        work.mkdir(parents=True, exist_ok=True)
        (work / "kfcup_module_source.cuh").write_text(source, encoding="utf-8")
        (work / "kfcup_host.cpp").write_text(harness, encoding="utf-8")
        tmp = work / f"libkfcup_host.{os.getpid()}.so"
        subprocess.run(
            [cxx, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
             "-fPIC", "-shared", "-w", "-I", str(work), str(work / "kfcup_host.cpp"),
             "-o", str(tmp)], check=True)
        os.replace(tmp, lib)
    handle = ctypes.CDLL(str(lib))
    f = handle.kfcup_host_run
    P = ctypes.c_void_p
    F = ctypes.c_float
    I = ctypes.c_int
    f.argtypes = [P, P, P, P, P, P, P, P, P, P, F, F, F, F, F, I, I, F, F, F, F, I, I, I]
    f.restype = I
    _LIBS[digest] = handle
    return handle


def _f(x) -> float:
    # A binary32 word as a Python float: exact, and ctypes rounds it back to
    # the same binary32 word.
    return float(np.float32(x))


def _ptr(a: np.ndarray):
    if not a.flags.c_contiguous:
        raise ValueError("host buffers must be C-contiguous")
    return a.ctypes.data_as(ctypes.c_void_p)


def run_columns(buffers: dict, *, dt, dx, ntst, numbins, thbinsize, rbinsize,
                mindeepfreq, minshallowfreq, flags, table=None, source=None) -> int:
    """Run ``kfcup_column`` over every column on the CPU, in place.

    ``buffers`` holds ``lev_in``, ``sca_in``, ``kpbl``, ``held``, ``iheld`` and
    ``lev_out`` in the kernel's packed layout (woof.verify.kfcup_oracle).
    Returns the kernel's status word.
    """
    if table is None:
        from woof.core.kf import load_kf_table
        table = load_kf_table()
    lib = host_library(source)
    lev_in = np.ascontiguousarray(buffers["lev_in"], np.float32)
    nlev, nz, ncol = lev_in.shape
    keep = [lev_in]
    args = []
    for name, dtype in (("sca_in", np.float32), ("kpbl", np.int32), ("held", np.float32),
                        ("iheld", np.int32), ("lev_out", np.float32)):
        a = buffers[name]
        if a.dtype != dtype or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be a C-contiguous {np.dtype(dtype)} array")
        args.append(a)
    tab = [np.ascontiguousarray(x, np.float32) for x in
           (table.temperature.T, table.qsat.T, table.thetae_base, table.log_ratio)]
    keep += tab
    status = lib.kfcup_host_run(
        _ptr(lev_in), *(_ptr(a) for a in args), *(_ptr(t) for t in tab),
        _f(table.pressure_top), _f(table.pressure_reciprocal), _f(table.thetae_reciprocal),
        _f(dt), _f(dx), int(ntst), int(numbins), _f(thbinsize), _f(rbinsize),
        _f(mindeepfreq), _f(minshallowfreq), int(flags), int(nz), int(ncol))
    del keep
    return int(status)


__all__ = ["host_library", "run_columns", "run_history", "run_radiation"]


def run_history(avg, w, *, stepcu, source=None):
    """Grade the actual compiled W0AVG update, including held columns."""
    if avg.dtype != np.float32 or w.dtype != np.float32:
        raise ValueError('history buffers must be float32')
    nz, ncol = avg.shape
    if w.shape != (nz+1,ncol) or stepcu < 1:
        raise ValueError('history needs staggered W and positive STEPCU')
    fn = host_library(source).kfcup_host_history
    fn.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    fn.restype = None
    fn(_ptr(avg),_ptr(w),int(stepcu),nz,ncol)


def run_radiation(values, *, icloud, forced, source=None):
    """Run the same device leaf over captured radiation cells on the CPU."""
    out={name:np.ascontiguousarray(values[key],np.float32).copy()
         for name,key in [('cldfra','cldfra1'),('qc','qc_in'),('cldfra_cup','cldfra_cup_in')]}
    keys=['flag','shall','cubot','cutop','k']
    arrays=[np.ascontiguousarray(values[key],np.int32 if key in ('flag','k') else np.float32)
            for key in keys]
    n=out['qc'].size
    if any(a.size!=n for a in arrays):
        raise ValueError('radiation inputs must have equal sizes')
    fn=host_library(source).kfcup_host_merge
    fn.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*3
    fn.restype=None
    fn(*(_ptr(out[k]) for k in ('cldfra','qc','cldfra_cup')),
       *(_ptr(a) for a in arrays),int(icloud),int(forced),n)
    return out

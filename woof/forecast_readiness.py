"""Forecast readiness with a measured, compiler/header-bound probe cache.

Doctor's public diagnostic stays cold. Forecasts still execute and read back both
kernels on every call, but may reuse a successful cold compile only while the
compiler, header trees, runtime, device architecture and probe bytes match.
Unknown discovery layouts fall back to the ordinary cold diagnostic.
"""
from __future__ import annotations

from woof.backend import compilation_target as _compilation_target

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

from woof.compiler_metadata import _atomic, _digest


def _files(root):
    """Bind every filename and byte, including extensionless CCCL headers."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"header root is not a directory: {root}")
    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Unknown directory-link layouts need the real cold probe, not an
        # incomplete hash. File symlinks are followed and their bytes bound.
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("linked header directory requires a cold probe")
        for name in sorted(files):
            path = Path(directory) / name
            rows.append((str(path.relative_to(root)),
                         hashlib.sha256(path.read_bytes()).hexdigest()))
    if not rows:
        raise ValueError(f"empty header tree: {root}")
    return {"root": str(root), "files": sorted(rows)}


def binding(cp, source):
    """Conservatively support the discovered CuPy14 CUDA header layout."""
    from cupy._core import core
    from cupy.cuda import compiler
    from woof.kernel_aot import compiler_identity

    if str(cp.__version__).split(".")[0] != "14" or cp.cuda.runtime.is_hip:
        raise ValueError("unbound CuPy header discovery layout")
    try:
        # cuda-pathfinder is a requirement of the CuPy 14 wheels (cupy-cuda12x
        # and cupy-cuda13x alike); an older CuPy has no such module.
        from cuda.pathfinder import find_nvidia_header_directory
    except ImportError as error:
        raise ValueError(f"unbound CuPy header discovery layout: {error}") from error
    # These options can add compiler search locations outside the known trees.
    for name in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "NVRTC_OPTIONS",
                 "CUPY_NVRTC_FLAGS", "CUB_PATH", "CUPY_CUDA_COMPILE_EXTRA_OPTIONS"):
        if os.environ.get(name):
            raise ValueError(f"{name} requires an ordinary cold probe")
    if not (hasattr(compiler, "_set_kernel_cache_backend") and
            hasattr(compiler, "_DiskKernelCacheBackend")):
        raise ValueError("unbound CuPy disk-cache selection API")
    roots = [Path(core._get_header_dir_path()).resolve(strict=True),
             Path(find_nvidia_header_directory("cudart")).resolve(strict=True),
             Path(find_nvidia_header_directory("nvrtc")).resolve(strict=True)]
    cccl = list(core._get_cccl_include_options())
    for option in cccl:
        if not option.startswith("-I"):
            raise ValueError("unbound CCCL compiler option")
        extra = Path(option[2:]).resolve(strict=True)
        if not any(extra.is_relative_to(root) for root in roots):
            roots.append(extra)
    # CUDA_PATH can change the header selection in older CUDA discovery paths.
    if os.environ.get("CUDA_PATH"):
        roots.append((Path(os.environ["CUDA_PATH"]) / "include").resolve(strict=True))
    return {
        "schema": "gpuwm.forecast-readiness/v2",
        "compiler": compiler_identity(),
        "compiler_python": hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest(),
        "cache_backend": hashlib.sha256(Path(sys.modules[
            compiler._DiskKernelCacheBackend.__module__].__file__).read_bytes()).hexdigest(),
        "core_binary": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "headers": [_files(root) for root in sorted(set(roots))],
        "cccl": cccl,
        "cuda_path": os.environ.get("CUDA_PATH", ""),
        "driver": cp.cuda.runtime.driverGetVersion(),
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
        "sm": str(_compilation_target(xp=cp)),
        "probe": hashlib.sha256(source.encode()).hexdigest(),
        "environment": {key: os.environ.get(key, "") for key in (
            "CUPY_ACCELERATORS", "CUPY_CACHE_IN_MEMORY", "CUPY_COMPILE_WITH_PTX",
            "CUPY_CUDA_COMPILE_WITH_DEBUG", "CUPY_NVRTC_USE_PCH",
            "CUPY_DISABLE_JITIFY_CACHE", "CUPY_TF32", "CUDA_MODULE_LOADING")},
    }


def _inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _load(root, key):
    try:
        row = json.loads((root / key / "ready.json").read_text())
        directory = root / key / row["generation"]
        if (row["key"] != key or _digest(row["binding"]) != key
                or Path(row["generation"]).name != row["generation"]
                or not row["generation"].startswith("cold-")
                or not any(name.startswith("cupy/") for name in row["files"])
                or _inventory(directory) != row["files"]):
            raise ValueError("readiness artifact digest mismatch")
        return directory
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError, OSError) as error:
        warnings.warn(f"bound readiness cache rejected; cold probe: {error}", RuntimeWarning)
        return None


def _execute(source, directory, identity=None):
    previous = os.environ.get("CUPY_CACHE_DIR")
    os.environ["CUPY_CACHE_DIR"] = str(directory / "cupy")
    from cupy.cuda import compiler
    old_backend = getattr(compiler, "_kernel_cache_backend", None)
    try:
        if hasattr(compiler, "_set_kernel_cache_backend"):
            # CuPy14 selects its backend at import time, before binding. An
            # environment change alone would silently reuse an unrelated cache.
            compiler._set_kernel_cache_backend(
                compiler._DiskKernelCacheBackend(str(directory / "cupy")))
        elif identity is None:
            # Unsupported cache API: a fresh interpreter selects the cold
            # directory before CuPy imports, exactly as ordinary doctor does.
            child = subprocess.run([sys.executable, "-c", source],
                                   capture_output=True, text=True, check=True)
            return json.loads(child.stdout)
        else:
            raise ValueError("unbound CuPy disk-cache selection API")
        if identity is not None:
            from woof.kernel_warm_pack import install_metadata
            install_metadata(compiler, directory, identity)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            try:
                exec(compile(source, "<gpuwm-readiness-probe>", "exec"), {})
            except SystemExit as error:
                if error.code not in (0, None):
                    raise
        return json.loads(output.getvalue())
    finally:
        if old_backend is not None:
            compiler._set_kernel_cache_backend(old_backend)
        if previous is None:
            os.environ.pop("CUPY_CACHE_DIR", None)
        else:
            os.environ["CUPY_CACHE_DIR"] = previous


def run(source, *, root=None):
    """Called only inside the dedicated readiness child process."""
    try:
        import cupy as cp
        before = binding(cp, source)
    except Exception as error:
        warnings.warn(f"readiness binding unavailable; cold probe: {error}", RuntimeWarning)
        with tempfile.TemporaryDirectory(prefix="gpuwm-readiness-cold-") as temp:
            result = _execute(source, Path(temp))
        result["readiness_cache"] = "cold-unbound"
        result["readiness_fallback"] = str(error)
        return result
    key = _digest(before)
    root = Path(root or os.environ.get("WOOF_KERNEL_PLAN_DIR") or
                Path.home() / ".cache" / "woof" / "kernel-plans") / "readiness"
    directory = _load(root, key)
    reused = directory is not None
    if directory is None:
        try:
            (root / key).mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="cold-", dir=root / key))
        except OSError as error:
            warnings.warn(f"readiness cache unavailable; cold probe: {error}", RuntimeWarning)
            with tempfile.TemporaryDirectory(prefix="gpuwm-readiness-cold-") as temp:
                result = _execute(source, Path(temp))
            result["readiness_cache"] = "cold-unbound"
            result["readiness_fallback"] = str(error)
            return result
    published = False
    try:
        result = _execute(source, directory, before["compiler"])
        # Never publish a successful proof of a tree that changed under it.
        if binding(cp, source) != before:
            raise RuntimeError("compiler/header binding changed during readiness probe")
        success = result.get("self_contained") == result.get("toolkit_headers") == "ok"
        if success and not reused:
            try:
                _atomic(root / key / "ready.json", json.dumps({
                    "key": key, "generation": directory.name,
                    "binding": before, "files": _inventory(directory),
                }, sort_keys=True).encode())
                published = True
            except OSError as error:
                warnings.warn(f"readiness proof not retained: {error}", RuntimeWarning)
        result["readiness_cache"] = "bound-warm" if reused else "bound-cold"
        result["readiness_key"] = key
        return result
    except (OSError, ValueError, RuntimeError) as error:
        # Execution failures are normally captured by the probe itself. A
        # binding/cache failure must not certify readiness from stale images.
        return {"probe": f"bound readiness failed: {error}"}
    finally:
        if not reused and not published:
            shutil.rmtree(directory)


def main():
    from woof.doctor import _NVRTC_HEADER_PROBE
    print(json.dumps(run(_NVRTC_HEADER_PROBE)))

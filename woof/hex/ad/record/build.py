# SPDX-License-Identifier: Apache-2.0
"""Build the recorder libraries from the C sources shipped in woof/hex/ad/record/native.

Three LD_PRELOAD libraries, built once per (source, compiler, CUPTI) and kept under cache_root()/native/<key>:

* recorder.so         the CUPTI driver-API recorder (driver_trace_cupti_held.c): every launch, copy, allocation and
                      module load of the ordinary forward, boundary payloads and retained uploads;
* cublas-observer.so  the cuBLAS SGEAM argument observer (checks only, the call is forwarded);
* library-observer.so the CuPy CUB/Thrust host-call observer (checks only).

Needs a C/C++ compiler (CC/CXX, else gcc/g++) and the zlib and OpenSSL libcrypto development libraries. A missing
one is a DEFECT naming it, never a silent skip.

The boundary digest kernel (native/sha256.cu, binder_sha256) ships prebuilt for sm_89 and sm_120; any other
architecture compiles the same source with NVRTC from the installed toolkit into the cache.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import env

SOURCES = ("driver_trace_cupti_held.c", "driver_trace_cupti_fallback.h", "upload_capture_thread.h", "d2h_capture.h",
           "upload_capture.h", "observe_cublas_geam.c", "observe_library_calls.cc")


def _tool(var: str, names: tuple[str, ...]) -> str:
    named = os.environ.get(var)
    if named:
        return named
    for name in names:
        hit = shutil.which(name)
        if hit:
            return hit
    env.defect(f"no {' or '.join(names)} on PATH and {var} is not set: the recorder libraries are built from source")


def _version(cc: str) -> str:
    try:
        return subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def build(force: bool = False, quiet: bool = False) -> dict:
    """Paths of the three libraries (built if needed) and the build receipt."""
    cc, cxx = _tool("CC", ("gcc", "cc")), _tool("CXX", ("g++", "c++"))
    libcupti, cupti_include = env.cupti()
    root = env.cuda_root()
    key = hashlib.sha256()
    for name in SOURCES:
        key.update(name.encode() + b"\0" + (env.NATIVE / name).read_bytes())
    key.update(_version(cc).encode() + _version(cxx).encode() + env.sha256_file(libcupti).encode())
    out = env.cache_root() / "native" / key.hexdigest()[:16]
    libs = {"recorder": out / "recorder.so", "cublas_observer": out / "cublas-observer.so",
            "library_observer": out / "library-observer.so"}
    if not force and all(p.is_file() for p in libs.values()) and (out / "build.json").is_file():
        return {**{k: str(v) for k, v in libs.items()}, "receipt": json.loads((out / "build.json").read_text())}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".build-", dir=out.parent))
    native = env.NATIVE
    commands = [
        [cc, "-O2", "-shared", "-fPIC", "-Wall", "-Wno-unused-function", "-Wno-deprecated-declarations",
         f"-I{cupti_include}", f"-I{root / 'include'}", str(native / "driver_trace_cupti_held.c"),
         "-o", str(tmp / "recorder.so"), str(libcupti), f"-Wl,-rpath,{libcupti.parent}",
         "-ldl", "-lpthread", "-lz", "-lcrypto"],
        [cc, "-O2", "-shared", "-fPIC", str(native / "observe_cublas_geam.c"), "-o", str(tmp / "cublas-observer.so"),
         "-ldl"],
        [cxx, "-O2", "-shared", "-fPIC", "-std=c++17", str(native / "observe_library_calls.cc"),
         "-o", str(tmp / "library-observer.so"), "-ldl"],
    ]
    for command in commands:
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            first = next((l for l in done.stderr.splitlines() if "error" in l), done.stderr.strip()[-300:])
            env.defect(f"building {Path(command[command.index('-o') + 1]).name} failed: {first}")
    receipt = {"schema": "hexad.record-build/v1", "key": key.hexdigest(), "cc": _version(cc), "cxx": _version(cxx),
               "cupti": libcupti.name, "cupti_sha256": env.sha256_file(libcupti),
               "libraries": {p.name: env.sha256_file(p) for p in sorted(tmp.glob("*.so"))}}
    (tmp / "build.json").write_text(json.dumps(receipt, indent=1))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    if not quiet:
        print(json.dumps(receipt))
    return {**{k: str(v) for k, v in libs.items()}, "receipt": receipt}


def digest_image(card: str) -> Path:
    """The binder_sha256 cubin for the card's architecture."""
    sm = env.compute_capability(card)
    # Shipped gzipped: a raw CUDA ELF in the wheel stops auditwheel (it parses every ELF it finds as a host binary).
    shipped = env.DATA / "digest" / f"sha256-{sm}.cubin.gz"
    if shipped.is_file():
        packed = shipped.read_bytes()
        target = env.cache_root() / "digest" / f"sha256-{sm}-{hashlib.sha256(packed).hexdigest()[:16]}.cubin"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(gzip.decompress(packed))
            os.replace(tmp, target)
        return target
    source = (env.NATIVE / "sha256.cu").read_bytes()
    target = env.cache_root() / "digest" / f"sha256-{sm}-{hashlib.sha256(source).hexdigest()[:16]}.cubin"
    if target.is_file():
        return target
    try:
        from cupy.cuda import nvrtc
    except ImportError as error:
        env.defect(f"no shipped digest cubin for {sm} and NVRTC is unavailable: {error}")
    program = nvrtc.createProgram(source.decode(), "sha256.cu", [], [])
    try:
        nvrtc.compileProgram(program, [f"-arch={sm}", "-std=c++17"])
    except nvrtc.NVRTCError:
        env.defect(f"NVRTC could not build the digest kernel for {sm}: {nvrtc.getProgramLog(program)[-300:]}")
    cubin = nvrtc.getCUBIN(program)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(cubin)
    os.replace(tmp, target)
    return target

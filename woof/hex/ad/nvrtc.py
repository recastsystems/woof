# SPDX-License-Identifier: Apache-2.0
"""Build one derivative image for the card present with NVRTC (the CUDA runtime compiler the install already carries).

A row's derivative images were built for the cards the row was proven on. On another card class the door rebuilds
them from their sources with the same options and only the architecture changed, so the image is the same code for
this card (no PTX JIT, no CPU fallback):

    python -m woof.hex.ad.nvrtc --src split2.main.cu --arch sm_89 --out split2.sm_89.cubin \\
        [--nvrtc libnvrtc.so.13] [--opt=--maxrregcount=128 ...] [--strict] [--include DIR] [--define NAME=V]

``--strict`` is the option set every hex derivative image uses (--std=c++17 --fmad=false --ftz=false
--prec-div=true --prec-sqrt=true --device-as-default-execution-space). The receipt (stdout, one JSON line) names the
NVRTC version, options, seconds and the source and cubin sha256.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

STRICT = ["--std=c++17", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true",
          "--device-as-default-execution-space"]


class NvrtcError(RuntimeError):
    pass


def find_nvrtc(explicit: str | None = None) -> str:
    """--nvrtc, else $AD_NVRTC_LIB, else the nvidia-cuda-nvrtc wheel of this install (nvidia/cu13/lib)."""
    if explicit:
        return explicit
    if os.environ.get("AD_NVRTC_LIB"):
        return os.environ["AD_NVRTC_LIB"]
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    for loc in (spec.submodule_search_locations if spec else []) or []:
        found = sorted(glob.glob(os.path.join(loc, "*", "lib", "libnvrtc.so.*[0-9]")))
        found = [f for f in found if "builtins" not in f]
        if found:
            return found[-1]
    raise NvrtcError("no libnvrtc found: pass --nvrtc or set AD_NVRTC_LIB (pip install nvidia-cuda-nvrtc)")


def compile_cubin(source: str, name: str, options: list[str], lib: str) -> tuple[bytes, str, tuple[int, int]]:
    nv = ctypes.CDLL(lib)
    major, minor = ctypes.c_int(), ctypes.c_int()
    nv.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    prog = ctypes.c_void_p()
    rc = nv.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None)
    if rc != 0:
        raise NvrtcError(f"nvrtcCreateProgram failed ({rc})")
    opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
    rc = nv.nvrtcCompileProgram(prog, len(options), opts)
    size = ctypes.c_size_t()
    nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nv.nvrtcGetProgramLog(prog, log)
    text = log.value.decode(errors="replace")
    if rc != 0:
        raise NvrtcError(f"NVRTC {major.value}.{minor.value} failed ({rc}):\n" + "\n".join(text.splitlines()[-40:]))
    nv.nvrtcGetCUBINSize(prog, ctypes.byref(size))
    if size.value == 0:
        raise NvrtcError("NVRTC produced no CUBIN (an arch of the form compute_XX gives PTX only; use sm_XX)")
    buf = ctypes.create_string_buffer(size.value)
    nv.nvrtcGetCUBIN(prog, buf)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return buf.raw, text, (major.value, minor.value)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m woof.hex.ad.nvrtc", description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--arch", required=True, help="sm_XY of the card")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nvrtc", default=None)
    p.add_argument("--strict", action="store_true", help="the hex derivative-image option set")
    p.add_argument("--opt", action="append", default=[], help="one more NVRTC option (repeatable; use --opt=VALUE)")
    p.add_argument("--include", action="append", default=[], type=Path)
    p.add_argument("--define", action="append", default=[])
    a = p.parse_args(argv)
    options = (list(STRICT) if a.strict else []) + [f"-arch={a.arch}"] + list(a.opt)
    options += [f"-I{d.resolve()}" for d in a.include] + [f"-D{x}" for x in a.define]
    source = a.src.read_text(encoding="utf-8", errors="replace")
    lib = find_nvrtc(a.nvrtc)
    t0 = time.monotonic()
    try:
        image, log, version = compile_cubin(source, a.src.name, options, lib)
    except NvrtcError as error:
        print(f"DEFECT: woof.hex.ad.nvrtc: {a.src}: {error}", file=sys.stderr)
        return 3
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(image)
    print(json.dumps({"nvrtc": list(version), "lib": lib, "options": options, "arch": a.arch,
                      "seconds": round(time.monotonic() - t0, 1), "source": str(a.src),
                      "source_sha256": hashlib.sha256(a.src.read_bytes()).hexdigest(), "out": str(a.out),
                      "cubin_sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

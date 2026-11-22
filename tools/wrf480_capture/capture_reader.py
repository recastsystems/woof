"""Read wrf480_capture dumps (module_gpuwm_capture.F record format) into numpy.

    from tools.wrf480_capture.capture_reader import read_capture, capture_index
    args = read_capture(path)          # {name: Arg(value, lbound)}
    idx = capture_index(directory)     # {(subroutine, step, call): {"entry": p, "exit": p}}

A record is 'GPCR', int32 name length, name, int32 type (1 real32, 2 real64,
3 int32, 4 logical stored as int32 0/1, 5 int64), int32 rank, int32 lbound[rank],
int32 shape[rank], then the data in Fortran (column-major) order, in the byte order
the build wrote (WRF GNU builds: big-endian via -fconvert); values come back native.  Arrays come back
as Fortran-ordered numpy arrays indexed from 0; ``lbound`` gives each axis's WRF
lower bound (ims, kms, jms ...), so WRF index i on axis a is ``i - lbound[a]``.

    python3 tools/wrf480_capture/capture_reader.py DIR     # prints an inventory
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {1: "f4", 2: "f8", 3: "i4",
           4: "i4", 5: "i8"}
_NAME = re.compile(r"(?P<sub>.+)-step(?P<step>\d{5})-call(?P<call>\d{3})-(?P<phase>entry|exit)\.bin$")


@dataclass(frozen=True)
class Arg:
    value: np.ndarray
    lbound: tuple[int, ...]
    logical: bool = False


def _byte_order(raw) -> str:
    """WRF's GNU configure compiles with -fconvert=big-endian, which also applies
    to these stream writes; a build without it writes little-endian.  The first
    record's name length (a short identifier) decides."""
    if len(raw) < 8:
        return ">"
    be = int(np.frombuffer(raw, ">i4", 1, 4)[0])
    return ">" if 0 < be < 256 else "<"


def read_capture(path) -> dict[str, Arg]:
    raw = Path(path).read_bytes()
    bo = _byte_order(raw)
    i4 = np.dtype(bo + "i4")
    out, pos = {}, 0
    while pos < len(raw):
        if raw[pos:pos + 4] != b"GPCR":
            raise ValueError(f"{path}: bad record magic at byte {pos}")
        pos += 4
        nlen = int(np.frombuffer(raw, i4, 1, pos)[0])
        pos += 4
        name = raw[pos:pos + nlen].decode()
        pos += nlen
        typ, rank = (int(x) for x in np.frombuffer(raw, i4, 2, pos))
        pos += 8
        lb = tuple(int(x) for x in np.frombuffer(raw, i4, rank, pos))
        pos += 4 * rank
        shape = tuple(int(x) for x in np.frombuffer(raw, i4, rank, pos))
        pos += 4 * rank
        dt = np.dtype(bo + _DTYPES[int(typ)])
        count = int(np.prod(shape)) if rank else 1
        data = np.frombuffer(raw, dt, count, pos).astype(dt.newbyteorder("="))
        pos += count * dt.itemsize
        value = data.reshape(shape, order="F") if rank else data[0]
        if name in out:
            raise ValueError(f"{path}: duplicate record {name}")
        out[name] = Arg(value, lb, int(typ) == 4)
    return out


def capture_index(directory) -> dict[tuple[str, int, int], dict[str, Path]]:
    idx: dict[tuple[str, int, int], dict[str, Path]] = {}
    for p in sorted(Path(directory).glob("*.bin")):
        m = _NAME.match(p.name)
        if m:
            key = (m["sub"], int(m["step"]), int(m["call"]))
            idx.setdefault(key, {})[m["phase"]] = p
    return idx


def main(argv):
    idx = capture_index(argv[1])
    for (sub, step, call), phases in sorted(idx.items()):
        entry = read_capture(phases["entry"]) if "entry" in phases else {}
        words = sum(np.size(a.value) for a in entry.values())
        print(f"{sub:22s} step {step:4d} call {call:2d} phases {sorted(phases)} "
              f"args {len(entry)} entry words {words}")


if __name__ == "__main__":
    main(sys.argv)

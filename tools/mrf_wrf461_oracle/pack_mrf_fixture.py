"""Turn ``mrf-oracle-out.bin`` into the packaged CSV fixture.

Every float is written as its 8-hex-digit IEEE-754 binary32 word, so the
fixture carries signed zeros, subnormals and NaN payloads exactly; KPBL is a
decimal integer.  Two files:

``mrf-levels.csv``  one row per (case, step, k): the step's 3-D inputs and
                    the six tendencies WRF's MRF wrapper returned
                    (RTHBLTEN already divided by PI3D, module_bl_mrf.F:233-237).
``mrf-surface.csv`` one row per (case, step): the 2-D inputs and the INOUT /
                    OUT words after the call (UST, HOL, PBL, KPBL2D, ZOL,
                    REGIME, WSPD).

The packer refuses a run in which WRF wrote memory level kme (k = nz+1) of
any tendency, or wrote RQIBLTEN with FLAG_QI false: both would mean the
driver's memory layout is not WRF's and the fixture would pin the wrong
words.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

SENTINEL = np.float32(12345.0)
MAGIC = 0x4D524644
IN3 = ("u", "v", "t", "qv", "qc", "qi", "p", "pi", "dz8w", "z")
OUT3 = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten")
IN2 = ("xland", "psfc", "znt", "ust_in", "psim", "psih", "hfx", "qfx", "tsk",
       "gz1oz0", "wspd_in", "br", "zol_in", "regime_in", "hol_in", "pbl_in")
OUT2 = ("ust", "hol", "pbl", "kpbl", "zol", "regime", "wspd")


def _w(x) -> str:
    return f"{int(np.asarray(x, np.float32).view(np.uint32)):08x}"


def pack(src: Path, dst: Path) -> None:
    raw = src.read_bytes()
    magic, ncase, nz, nstep = struct.unpack_from("<4i", raw, 0)
    if magic != MAGIC:
        raise SystemExit("not an MRF oracle output")
    off = 16
    kme = nz + 1
    lv = [",".join(("case", "step", "k") + IN3 + OUT3)]
    sf = [",".join(("case", "step", "dt", "flag_qi") + IN2 + OUT2)]
    for _ in range(nstep * ncase):
        s, c = struct.unpack_from("<2i", raw, off)
        off += 8
        dt = np.frombuffer(raw, "<f4", 1, off)[0]
        off += 4
        flag = struct.unpack_from("<i", raw, off)[0]
        off += 4
        in2 = np.frombuffer(raw, "<f4", 16, off)
        off += 64
        in3 = np.frombuffer(raw, "<f4", 10 * nz, off).reshape(10, nz)
        off += 40 * nz
        out3 = np.frombuffer(raw, "<f4", 6 * kme, off).reshape(6, kme)
        off += 24 * kme
        ust, hol, pbl = np.frombuffer(raw, "<f4", 3, off)
        off += 12
        kpbl = struct.unpack_from("<i", raw, off)[0]
        off += 4
        zol, regime, wspd = np.frombuffer(raw, "<f4", 3, off)
        off += 12
        if not np.all(out3[:, nz].view(np.uint32)
                      == SENTINEL.view(np.uint32)):
            raise SystemExit(f"case {c} step {s}: WRF wrote level kme")
        if not flag and not np.all(out3[5].view(np.uint32)
                                   == SENTINEL.view(np.uint32)):
            raise SystemExit(f"case {c} step {s}: RQIBLTEN written with "
                             "FLAG_QI false")
        for k in range(nz):
            lv.append(",".join([str(c - 1), str(s - 1), str(k)]
                               + [_w(in3[i, k]) for i in range(10)]
                               + [_w(out3[i, k]) for i in range(6)]))
        sf.append(",".join([str(c - 1), str(s - 1), _w(dt), str(flag)]
                           + [_w(x) for x in in2]
                           + [_w(ust), _w(hol), _w(pbl), str(kpbl),
                              _w(zol), _w(regime), _w(wspd)]))
    if off != len(raw):
        raise SystemExit(f"trailing bytes: read {off} of {len(raw)}")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "mrf-levels.csv").write_text("\n".join(lv) + "\n",
                                        encoding="ascii", newline="\n")
    (dst / "mrf-surface.csv").write_text("\n".join(sf) + "\n",
                                         encoding="ascii", newline="\n")
    print(f"packed {ncase} cases x {nstep} steps x {nz} levels")


if __name__ == "__main__":
    pack(Path(sys.argv[1]), Path(sys.argv[2]))

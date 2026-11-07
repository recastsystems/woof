"""Gather the old-KF oracle's per-case outputs into one fixture.

usage: collect.py CASES.bin OUTCOMES.txt OUTDIR FIXTURE.npz

The fixture groups cases by level count, because one GPU launch has one
nz.  For every group ``g`` (``nz40``, ``nz49``, ...) it holds the case
inputs exactly as the oracle read them, the driver knobs, the arm-A
outputs, and ``g/fatal``: 0 for a column that completed, otherwise the
code of the WRF fatal it reached (1 surface qg, 2 mass budget, 3 QVERR,
9 any other message), with its outputs left at zero.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

INPUTS = ("u", "v", "w", "t", "th", "qv", "dz8w", "p", "rho", "pi",
          "w0avg")
INTS = ("stepcu", "adapt", "warm_rain", "f_qi", "f_qs", "f_qr",
        "kf_edrates")
FLOATS = ("dt", "dx", "cudt", "nca")
OUT3 = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
        "rqscuten", "w0avg_out", "udr_kf", "ddr_kf", "uer_kf", "der_kf")
OUT2 = ("raincv", "pratec", "nca_out", "timec_kf")

FATAL_CODES = (
    ("qg = 0 at the surface", 1),
    ("MASS DOES NOT BALANCE", 2),
    ("KAIN-FRITSCH", 2),
    ("QVERR", 3),
)


def read_cases(path: Path):
    data = path.read_bytes()
    magic, version, ncase = struct.unpack_from("<3i", data, 0)
    assert magic == 0x4B464F4C and version == 1
    off = 12
    cases = []
    for _ in range(ncase):
        ints = struct.unpack_from("<8i", data, off)
        off += 32
        floats = struct.unpack_from("<4f", data, off)
        off += 16
        nz = ints[0]
        case = {"nz": nz}
        case.update(zip(INTS, ints[1:]))
        case.update(zip(FLOATS, floats))
        for name in INPUTS:
            n = nz + 1 if name == "w" else nz
            case[name] = np.frombuffer(data, "<f4", n, off).copy()
            off += 4 * n
        cases.append(case)
    assert off == len(data)
    return cases


def fatal_code(message: str) -> int:
    for needle, code in FATAL_CODES:
        if needle in message:
            return code
    return 9


def main() -> None:
    cases_path, outcomes_path, outdir, fixture = map(Path, sys.argv[1:5])
    cases = read_cases(cases_path)
    outcomes = {}
    for line in outcomes_path.read_text().splitlines():
        parts = line.split(maxsplit=3)
        if parts[0] != "A":
            continue
        outcomes[int(parts[1])] = (int(parts[2]),
                                   parts[3] if len(parts) > 3 else "")
    groups: dict[int, list[int]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(case["nz"], []).append(index)
    arrays = {}
    for nz, members in sorted(groups.items()):
        g = f"nz{nz}"
        arrays[f"{g}/case_index"] = np.asarray(members, dtype=np.int32)
        for name in INPUTS:
            arrays[f"{g}/{name}"] = np.stack([cases[i][name]
                                              for i in members])
        for name in INTS:
            arrays[f"{g}/{name}"] = np.asarray(
                [cases[i][name] for i in members], dtype=np.int32)
        for name in FLOATS:
            arrays[f"{g}/{name}"] = np.asarray(
                [cases[i][name] for i in members], dtype=np.float32)
        out3 = {name: np.zeros((len(members), nz), np.float32)
                for name in OUT3}
        out2 = {name: np.zeros(len(members), np.float32) for name in OUT2}
        act = np.zeros(len(members), np.int32)
        fatal = np.zeros(len(members), np.int32)
        for row, index in enumerate(members):
            code, message = outcomes[index]
            if code != 0:
                if code != 7:
                    raise SystemExit(
                        f"case {index} exited {code} without a WRF fatal")
                fatal[row] = fatal_code(message)
                continue
            raw = (outdir / f"A-{index}.bin").read_bytes()
            status, onz, flag = struct.unpack_from("<3i", raw, 0)
            assert status == 0 and onz == nz
            act[row] = flag
            values = np.frombuffer(raw, "<f4", 4, 12)
            for k, name in enumerate(OUT2):
                out2[name][row] = values[k]
            off = 28
            for name in OUT3:
                out3[name][row] = np.frombuffer(raw, "<f4", nz, off)
                off += 4 * nz
            assert off == len(raw)
        arrays.update({f"{g}/{k}": v for k, v in out3.items()})
        arrays.update({f"{g}/{k}": v for k, v in out2.items()})
        arrays[f"{g}/cu_act_flag"] = act
        arrays[f"{g}/fatal"] = fatal
    np.savez_compressed(fixture, **arrays)
    total = sum(len(m) for m in groups.values())
    print(f"{total} cases in {len(groups)} level groups -> {fixture}")


if __name__ == "__main__":
    main()

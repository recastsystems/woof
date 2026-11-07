"""Read the slab column oracle and grade the CUDA column against it.

The oracle is ``tools/slab_wrf461_oracle``: the byte-unmodified WRF v4.6.1
``phys/module_sf_slab.F`` at gfortran -O0, driven through its own ``SLAB``
entry for every case of ``generate_cases.py`` over ``nstep`` consecutive
surface calls.  Its packaged copy is ``tests/data/oracles/slab``; the
build regenerates both files and ``check_slab.sh`` requires them to be
byte-identical to the packaged ones.

:func:`run_cuda_columns` replays every case through
:func:`woof.core.slab.launch_slab` -- the production launcher, on the
production kernel source -- and :func:`compare` reports, field by field,
how many words differ and the largest ULP distance.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests/data/oracles/slab"
OUTPUT_FIELDS = ("hfx", "qfx", "lh", "capg", "tsk", "qsfc", "chklowq",
                 "tslb1", "tslb2", "tslb3", "tslb4", "tslb5", "flhc", "flqc")
CONSTANT_NAMES = ("svp1", "svp2", "svp3", "svpt0", "ep2", "karman", "eomeg",
                  "stbolt", "xlv", "rcp", "p1000mb")
COLUMN_FIELDS = ("t1", "qv1", "p1", "flhc", "flqc", "psfc", "xland", "tmn",
                 "hfx", "qfx", "lh", "tsk", "qsfc", "chklowq", "gsw", "glw",
                 "capg", "thc", "snowc", "emiss", "mavail")


@dataclass(frozen=True)
class SlabOracle:
    cases: list            # per case: ifsnow, radiation, deltsm, label
    columns: np.ndarray    # (ncase, 21) float32 initial column
    tslb: np.ndarray       # (ncase, 5) float32 initial soil temperature
    zs: np.ndarray
    dzs: np.ndarray
    nstep: int
    constants: dict        # WRF's module_model_constants as the oracle used
    output: np.ndarray     # (ncase, nstep, 14) float32


def load(directory: Path = ORACLE_DIR) -> SlabOracle:
    directory = Path(directory)
    raw_in = (directory / "slab-input.bin").read_bytes()
    raw_out = (directory / "slab-output.bin").read_bytes()
    labels = json.loads((directory / "slab-cases.json").read_text(
        encoding="utf-8"))["cases"]
    return _load_bytes(raw_in, raw_out, labels)


def load_native_capture(input_path: Path, output_path: Path) -> SlabOracle:
    """Read one SLAB1D call recorded by capture_native.py in a real WRF run."""
    return _load_bytes(Path(input_path).read_bytes(),
                       Path(output_path).read_bytes())


def _load_bytes(raw_in, raw_out, labels=None) -> SlabOracle:
    # GFORTRAN_CONVERT_UNIT can override even OPEN(CONVERT=...). Detect
    # the stream order from the magic, without changing any REAL word.
    order = ">" if struct.unpack_from(">i", raw_in, 0)[0] == 5150 else "<"
    magic, ncase = struct.unpack_from(order + "ii", raw_in, 0)
    if magic != 5150:
        raise ValueError("slab-input.bin has the wrong magic")
    offset = 8
    cases, columns, tslb = [], [], []
    zs = dzs = None
    nstep = None
    if labels is None:
        labels = [{"label": f"native-column-{index}"} for index in range(ncase)]
    for index in range(ncase):
        ifsnow, radiation, steps = struct.unpack_from(order + "iii", raw_in, offset)
        offset += 12
        values = np.frombuffer(raw_in, order + "f4", 1 + 21 + 15, offset)
        offset += 4 * (1 + 21 + 15)
        nstep = steps if nstep is None else nstep
        if steps != nstep:
            raise ValueError("every case must run the same number of steps")
        cases.append({"ifsnow": ifsnow, "radiation": radiation,
                      "deltsm": np.float32(values[0]),
                      "label": labels[index]["label"]})
        columns.append(values[1:22])
        tslb.append(values[22:27])
        zs, dzs = values[27:32].copy(), values[32:37].copy()
    magic, ncase_out = struct.unpack_from(order + "ii", raw_out, 0)
    if magic != 5152 or ncase_out != ncase:
        raise ValueError("slab-output.bin does not answer slab-input.bin")
    constants = dict(zip(CONSTANT_NAMES,
                         np.frombuffer(raw_out, order + "f4", 11, 8)))
    output = np.frombuffer(raw_out, order + "f4", offset=52).reshape(
        ncase, nstep, len(OUTPUT_FIELDS))
    return SlabOracle(cases, np.array(columns, np.float32),
                      np.array(tslb, np.float32), zs, dzs, nstep, constants,
                      output.copy())


def run_cuda_columns(oracle: SlabOracle, *, constants=None) -> np.ndarray:
    """Every case through the production launcher; ``(ncase, nstep, 14)``.

    Cases sharing ``(deltsm, ifsnow, radiation)`` run in one launch, side by
    side -- land, water and ice columns in the same grid -- as a forecast
    runs them.
    """
    import cupy as cp
    from woof.core.slab import launch_slab

    out = np.empty_like(oracle.output)
    groups: dict[tuple, list[int]] = {}
    for index, case in enumerate(oracle.cases):
        key = (float(case["deltsm"]), case["ifsnow"], case["radiation"])
        groups.setdefault(key, []).append(index)
    zs = cp.asarray(oracle.zs)
    dzs = cp.asarray(oracle.dzs)
    for (deltsm, ifsnow, radiation), members in groups.items():
        cols = oracle.columns[members]
        f = {name: cp.ascontiguousarray(cp.asarray(cols[:, k]).reshape(1, -1))
             for k, name in enumerate(COLUMN_FIELDS)}
        f["tslb"] = cp.ascontiguousarray(
            cp.asarray(oracle.tslb[members].T).reshape(5, 1, -1))
        for step in range(oracle.nstep):
            launch_slab(f, t1=f["t1"], qv1=f["qv1"], p1=f["p1"],
                        dt=np.float32(deltsm), ifsnow=ifsnow,
                        radiation=bool(radiation), zs=zs, dzs=dzs,
                        constants=constants)
            words = [f[name].ravel() for name in OUTPUT_FIELDS[:7]]
            words += [f["tslb"][layer].ravel() for layer in range(5)]
            words += [f[name].ravel() for name in OUTPUT_FIELDS[12:]]
            out[members, step, :] = cp.asnumpy(cp.stack(words, axis=1))
    return out


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in float32 representable steps (signed-magnitude ordered)."""
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Per field: differing words (bit patterns) and the max ULP distance."""
    report = {}
    for k, name in enumerate(OUTPUT_FIELDS):
        a = actual[..., k]
        e = expected[..., k]
        differ = a.view(np.uint32) != e.view(np.uint32)
        report[name] = {"words": int(a.size),
                        "differing": int(np.count_nonzero(differ)),
                        "max_ulp": int(ulp_distance(a, e).max())}
    return report


__all__ = ["CONSTANT_NAMES", "ORACLE_DIR", "OUTPUT_FIELDS", "SlabOracle",
           "compare", "load", "load_native_capture", "run_cuda_columns", "ulp_distance"]

#!/usr/bin/env python3
"""Measure woof Morrison against the committed unmodified-WRF fixture.

This is intentionally a measurement command, not the standing assertion.
It prints every field's observed FP32 total-order ULP maximum and a machine-
readable JSON summary. The pytest gate (tests/test_morrison_wrf461_parity.py)
imports :func:`measure_oracle`: WRF's arithmetic alone (``--row37-off``,
the unit built with MORR_ROW37_OFF) is bitwise, and the production build
includes row-37 and nonnegative sedimentation corrections. The row-37
contract can also be measured in isolation from the transport correction.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof.core.fp32_ulp import (  # noqa: E402
    bitwise_identical,
    fp32_ulp_distance,
)

DEFAULT_FIXTURE_DIR = REPO_ROOT / "tests" / "data" / "oracles" / "morrison" / "oracle"

LEVEL_INPUTS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg",
    "ni", "ns", "nr", "ng", "rho", "pii", "pressure", "dz",
    "qrcuten", "qscuten", "qicuten",
)
LEVEL_OUTPUTS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg",
    "ni", "ns", "nr", "ng",
)
SURFACE_FIELDS = (
    "rainnc", "rainncv", "snownc", "snowncv",
    "graupelnc", "graupelncv", "sr",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _f32(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([row[name] for row in rows], dtype=np.float32)


def _record(
    aggregate: dict[str, dict[str, object]],
    field: str,
    got: np.ndarray,
    want: np.ndarray,
    *,
    case: int,
    mode: int,
    surface: bool = False,
) -> None:
    got = np.ascontiguousarray(got, dtype=np.float32)
    want = np.ascontiguousarray(want, dtype=np.float32)
    distance = fp32_ulp_distance(got, want)
    bits_got = got.view(np.uint32)
    bits_want = want.view(np.uint32)
    mismatch = bits_got != bits_want
    entry = aggregate.setdefault(
        field,
        {
            "max_ulp": 0,
            "mismatch_count": 0,
            "value_count": 0,
            "bitwise": True,
            "worst": None,
            "nonfinite_count": 0,
            "minimum": float("inf"),
            "mismatch_cases": [],
        },
    )
    if mismatch.any() and [case, mode] not in entry["mismatch_cases"]:
        entry["mismatch_cases"].append([case, mode])
    local_max = int(distance.max()) if distance.size else 0
    entry["nonfinite_count"] = int(entry["nonfinite_count"]) + int(
        np.count_nonzero(~np.isfinite(got)))
    if got.size:
        entry["minimum"] = min(float(entry["minimum"]), float(got.min()))
    entry["mismatch_count"] = int(entry["mismatch_count"]) + int(mismatch.sum())
    entry["value_count"] = int(entry["value_count"]) + int(distance.size)
    entry["bitwise"] = bool(entry["bitwise"]) and bitwise_identical(got, want)
    if local_max > int(entry["max_ulp"]) or (
            local_max == int(entry["max_ulp"])
            and entry["worst"] is None and mismatch.any()):
        flat = int(np.argmax(distance))
        entry["max_ulp"] = local_max
        entry["worst"] = {
            "case": case,
            "mode": mode,
            "k": None if surface else flat + 1,
            "got": float(got.ravel()[flat]),
            "want": float(want.ravel()[flat]),
            "got_bits": f"0x{int(bits_got.ravel()[flat]):08x}",
            "want_bits": f"0x{int(bits_want.ravel()[flat]):08x}",
        }


def measure_oracle(
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    *,
    row37: bool = True,
    nonnegative_sedimentation: bool = True,
) -> dict[str, object]:
    """Run every fixture column and return field/overall parity metrics.

    ``row37=False`` compiles the unit with ``MORR_ROW37_OFF``: WRF's
    arithmetic alone, without the owner-ruled row-37 corrections (the
    nucleation vapour cap and the wide non-finite freezing rates).
    """
    import cupy as cp

    # Disabling just the new transport correction isolates the older row-37
    # contract; production and the raw WRF oracle are still measured separately.
    if row37 and nonnegative_sedimentation:
        return _measure(fixture_dir, row37=True)
    from woof.core import kernels

    original = kernels.load_module

    def row37_off_load_module(name: str):
        if name != "morrison":
            return original(name)
        source = kernels.module_source("morrison")
        if not nonnegative_sedimentation:
            correction = "            dmass = FMAXK(dmass, 0.0f);"
            assert source.count(correction) == 1
            source = source.replace(correction, "")
        module = cp.RawModule(
            code=("" if row37 else "#define MORR_ROW37_OFF 1\n") + source,
            options=kernels.module_options("morrison"))
        module.compile()
        return module

    # The breakage the restore prevents: the swap used to outlive the
    # call, so every later Morrison launch in the same process ran WRF's
    # arithmetic without the row-37 corrections; on W1 the cold-rain and
    # cold-cloud finite-state tests that ran after the parity test in one
    # pytest process went non-finite (18 failures) while each passed alone.
    kernels.load_module = row37_off_load_module
    kernels.get_kernel.cache_clear()
    try:
        return _measure(fixture_dir, row37=row37)
    finally:
        kernels.load_module = original
        kernels.get_kernel.cache_clear()


def _measure(fixture_dir: Path, *, row37: bool) -> dict[str, object]:
    import cupy as cp


    from woof.core.morrison import WORK_T3D, launch_morrison
    from woof.core.refl import launch_refl10cm_morrison

    fixture_dir = Path(fixture_dir)
    level_rows = _read_rows(fixture_dir / "morrison-levels.csv")
    surface_rows = _read_rows(fixture_dir / "morrison-surface.csv")
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in level_rows:
        grouped[(int(row["case"]), int(row["mode"]))].append(row)
    surfaces = {
        (int(row["case"]), int(row["mode"])): row for row in surface_rows
    }
    if set(grouped) != set(surfaces):
        raise ValueError("level/surface fixture keys differ")

    aggregate: dict[str, dict[str, object]] = {}
    for (case, mode), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["k"]))
        nz = int(rows[0]["nz"])
        if len(rows) != nz or [int(row["k"]) for row in rows] != list(
                range(1, nz + 1)):
            raise ValueError(f"case={case} mode={mode}: non-contiguous levels")
        dt = np.float32(rows[0]["dt"])

        host = {
            name: _f32(rows, f"{name}_in").reshape(nz, 1, 1)
            for name in LEVEL_INPUTS
        }
        device = {
            name: cp.asarray(host[name])
            for name in (
                "theta", "qv", "qc", "qr", "qi", "qs", "qg",
                "ni", "ns", "nr", "ng", "rho", "pii", "pressure", "dz",
            )
        }
        # WRF's production INUM=1 wrapper neither accepts nor returns NC3D:
        # it constructs fixed 250 cm-3 cloud number internally.  woof's
        # wider state API carries nc, but its process stage performs that
        # same overwrite before the first read, so its incoming value is
        # deliberately an inert zero here and is not compared as an output.
        device["nc"] = cp.zeros((nz, 1, 1), dtype=cp.float32)
        cu = {
            name: cp.asarray(host[name])
            for name in ("qrcuten", "qscuten", "qicuten")
        }

        surface = surfaces[(case, mode)]
        precip = {
            name: cp.asarray(
                np.asarray([[surface[f"{name}_in"]]], dtype=np.float32))
            for name in SURFACE_FIELDS
        }
        effective = {
            name: cp.empty((nz, 1, 1), dtype=cp.float32)
            for name in ("effc", "effr", "effi", "effs")
        }

        work = launch_morrison(
            device["theta"], device["qv"], device["qc"], device["qr"],
            device["qi"], device["qs"], device["qg"], device["nc"],
            device["nr"], device["ni"], device["ns"], device["ng"],
            device["rho"], device["pii"], device["pressure"], device["dz"],
            precip["rainnc"], precip["rainncv"],
            precip["snownc"], precip["snowncv"],
            precip["graupelnc"], precip["graupelncv"], precip["sr"],
            float(dt), **effective, **cu, morr_rimed_ice=mode,
        )

        # WRF's refl10cm_hm reads the scheme's own T1D (the WORK stack's
        # temperature plane) and the unchanged prepared pressure.
        t1d = cp.ascontiguousarray(work[WORK_T3D * nz:(WORK_T3D + 1) * nz])
        refl = cp.empty_like(device["theta"])
        launch_refl10cm_morrison(
            device["qv"], device["qr"], device["nr"],
            device["qs"], device["ns"], device["qg"], device["ng"],
            t1d, device["pressure"], refl, morr_rimed_ice=mode,
        )
        cp.cuda.get_current_stream().synchronize()

        for name in LEVEL_OUTPUTS:
            got = cp.asnumpy(device[name]).reshape(nz)
            want = _f32(rows, f"{name}_out")
            _record(aggregate, name, got, want, case=case, mode=mode)
        _record(
            aggregate,
            "refl_10cm",
            cp.asnumpy(refl).reshape(nz),
            _f32(rows, "refl_10cm_out"),
            case=case,
            mode=mode,
        )
        for name in SURFACE_FIELDS:
            _record(
                aggregate,
                name,
                cp.asnumpy(precip[name]).reshape(1),
                np.asarray([surface[f"{name}_out"]], dtype=np.float32),
                case=case,
                mode=mode,
                surface=True,
            )

    worst_field = max(
        aggregate,
        key=lambda name: (int(aggregate[name]["max_ulp"]), name),
    )
    return {
        "fixture_dir": str(fixture_dir),
        "row37": row37,
        "column_count": len(grouped),
        "level_count": len(level_rows),
        "fields": aggregate,
        "overall": {
            "max_ulp": int(aggregate[worst_field]["max_ulp"]),
            "worst_field": worst_field,
            "mismatch_count": sum(
                int(entry["mismatch_count"]) for entry in aggregate.values()
            ),
            "value_count": sum(
                int(entry["value_count"]) for entry in aggregate.values()
            ),
            "bitwise": all(
                bool(entry["bitwise"]) for entry in aggregate.values()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture_dir", nargs="?", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--row37-off", action="store_true",
                        help="WRF's arithmetic alone (MORR_ROW37_OFF)")
    args = parser.parse_args()
    report = measure_oracle(
        args.fixture_dir,
        row37=not args.row37_off,
    )
    if not args.json:
        print("field,max_ulp,mismatches,values")
        for name, entry in sorted(report["fields"].items()):
            print(
                f"{name},{entry['max_ulp']},{entry['mismatch_count']},"
                f"{entry['value_count']}"
            )
        print(
            "overall,"
            f"{report['overall']['max_ulp']},"
            f"{report['overall']['mismatch_count']},"
            f"{report['overall']['value_count']}"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

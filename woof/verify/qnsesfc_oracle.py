"""Grade the QNSE surface layer against its WRF v4.6.1 oracle.

The fixture (tests/data/oracles/qnsesfc/qnsesfc-wrf461.npz) is written by
tools/qnsesfc_wrf461_oracle/build.sh: the byte-unmodified
phys/module_sf_qnsesfc.F compiled by gfortran with WRF's own optimised phys
flags, driven column by column by run_qnsesfc.F90.  It carries
QNSESFCINIT's PSIM/PSIH tables and, for each set of columns, the INOUT words
every QNSESFC call started from (``pre``) and the post-call words of every
INOUT and output field (``out``).

Two comparisons:

* :func:`compare_tables` -- the tables WOOF builds on the host
  (:func:`woof.core.myjsfc_tables.build_qnse_psi_tables`) against the
  tables QNSESFCINIT built, word for word.  CPU only.
* :func:`measure` -- the CUDA kernel against QNSESFC, every INOUT and output
  field, bitwise, in two modes: ``replay`` hands the kernel WRF's own
  pre-call state at every call (each call graded alone), ``free_run``
  lets the kernel carry its own state from call to call (what a forecast
  does), with the land columns' HFX/QFX overwritten before every call by
  the same stand-in land-surface fluxes the Fortran driver applied.

ULP distance is measured on the float32 total order, so +0 and -0 are 0
apart and two NaNs are 0 apart. Mismatch counts compare raw uint32 words,
including zero signs and NaN payloads; a word the kernel flushed to zero where
WRF kept a subnormal is counted separately (``flushed_subnormal``) because
default arithmetic may compile with flush-to-zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

F = np.float32
FIXTURE = (Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles"
           / "qnsesfc" / "qnsesfc-wrf461.npz")
TABLE_NAMES = ("psim1", "psih1", "psim2", "psih2")
SCALAR_NAMES = ("dzeta1", "dzeta2", "ztmin1", "ztmax1", "ztmin2", "ztmax2",
                "fh01", "fh02")
#: launcher state name -> the case-file key of its initial value.
_STATE_KEYS = {"ust": "ustar", "znt": "znt", "thz0": "thz0", "qz0": "qz0",
               "uz0": "uz0", "vz0": "vz0", "qsfc": "qsfc", "akhs": "akhs",
               "akms": "akms"}


def load(path: Path | str = FIXTURE) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def set_names(fixture: dict) -> list[str]:
    return sorted({key.split("/")[0] for key in fixture
                   if key.endswith("/meta")})


def ulp_distance(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    got = np.ascontiguousarray(got, dtype=F)
    want = np.ascontiguousarray(want, dtype=F)
    dist = np.abs(monotone_fp32_key(got) - monotone_fp32_key(want))
    both_nan = np.isnan(got) & np.isnan(want)
    return np.where(both_nan, 0, dist)


def _stats(got: np.ndarray, want: np.ndarray) -> dict:
    dist = ulp_distance(got, want)
    tiny = np.finfo(F).tiny
    flushed = int(np.sum((got == 0.0) & (want != 0.0) & (np.abs(want) < tiny)))
    with np.errstate(invalid="ignore"):
        absdiff = np.abs(got.astype(np.float64) - want.astype(np.float64))
        scale = np.abs(want.astype(np.float64))
        rel = np.where(scale > 0.0, absdiff / np.where(scale > 0, scale, 1.0),
                       absdiff)
    absdiff = np.where(np.isnan(absdiff), 0.0, absdiff)
    rel = np.where(np.isnan(rel), 0.0, rel)
    return {"max_ulp": int(dist.max(initial=0)),
            "mismatch": int(np.count_nonzero(
                np.ascontiguousarray(got, dtype=F).view(np.uint32) !=
                np.ascontiguousarray(want, dtype=F).view(np.uint32))),
            "count": int(dist.size),
            "max_abs": float(absdiff.max(initial=0.0)),
            "max_rel": float(rel.max(initial=0.0)),
            "flushed_subnormal": flushed}


def compare_tables(fixture: dict) -> dict:
    """WOOF's QNSESFCINIT tables against the oracle's, word for word."""
    from woof.core.myjsfc_tables import build_qnse_psi_tables
    tables = build_qnse_psi_tables()
    report = {}
    for name in TABLE_NAMES:
        report[name] = _stats(np.asarray(tables[name], F),
                              fixture[f"tables/{name}"])
    want = fixture["tables/scalars"]
    got = np.asarray([tables[name] for name in SCALAR_NAMES], F)
    report["scalars"] = _stats(got, want)
    return report


def _case(fixture: dict, set_name: str) -> dict:
    return {key.split("/", 2)[2]: value for key, value in fixture.items()
            if key.startswith(f"{set_name}/in/")}


def _device_inputs(cp, case: dict, surface_inputs) -> tuple[dict, dict]:
    def dev(array):
        return cp.ascontiguousarray(cp.asarray(array, dtype=F))
    nz, ncol = case["dz"].shape
    columns = {"dz": dev(case["dz"].reshape(nz, 1, ncol)),
               "tke": dev(case["q2"].reshape(nz, 1, ncol))}
    surface = {
        "u1": case["u"][0], "v1": case["v"][0], "t1": case["t"][0],
        "th1": case["th"][0], "qv1": case["qv"][0], "qc1": case["qc"][0],
        "p1": case["pmid"][0], "psfc": case["pint"][0], "tsk": case["tsk"],
        "xland": case["xland"], "mavail": case["mavail"],
        "z0base": case["z0base"], "ht": case["ht"],
    }
    surface = {name: dev(np.asarray(value).reshape(1, ncol))
               for name, value in surface.items() if name in surface_inputs}
    return columns, surface


def measure(fixture: dict, *, sf_sfclay_physics: int = 4) -> dict:
    """Run the CUDA kernel over every set and call; grade every field.

    ``sf_sfclay_physics`` picks the launcher row; 4 (QNSE) is the port
    under test.  Passing 2 runs the Eta (MYJ) kernel on QNSE's columns,
    the measurement that shows the grade can tell the two schemes apart.
    """
    import cupy as cp
    from woof.core import myjsfc as launcher

    row = launcher.ETA_FAMILY_SURFACE_LAYERS[sf_sfclay_physics]
    inout = launcher.MYJ_SFCLAY_INOUT
    names = inout + launcher.MYJ_SFCLAY_OUTPUTS
    fields = [str(f) for f in fixture["fields"]]
    pre_fields = [str(f) for f in fixture["pre_fields"]]
    if tuple(fields) != tuple(names):
        raise ValueError(f"fixture fields {fields} are not the launcher's "
                         f"{names}")
    report = {"replay": {}, "free_run": {}, "per_set": {}}
    collected = {mode: {name: ([], []) for name in names}
                 for mode in ("replay", "free_run")}
    for set_name in set_names(fixture):
        ncol, nz, nsteps, it0 = (int(v) for v in fixture[f"{set_name}/meta"])
        case = _case(fixture, set_name)
        pre_all = fixture[f"{set_name}/pre"]           # (nsteps, NPRE, ncol)
        want_all = fixture[f"{set_name}/out"]          # (nsteps, NF, ncol)
        land = case["xland"] < F(1.5)
        columns, surface = _device_inputs(cp, case, row.surface_inputs)
        per_set = {}
        for mode in ("replay", "free_run"):
            carried = {name: case[_STATE_KEYS[name]].astype(F).copy()
                       for name in inout}
            carried_flux = {"hfx": np.zeros(ncol, F), "qfx": np.zeros(ncol, F)}
            for step in range(nsteps):
                if mode == "replay":
                    pre = pre_all[step]
                    carried = {name: pre[pre_fields.index(name)].astype(F)
                               for name in inout}
                    flux = {name: pre[pre_fields.index(name)].astype(F)
                            for name in ("hfx", "qfx")}
                else:
                    flux = {name: np.where(
                        land, case[f"{name}_lsm"][step], carried_flux[name]
                    ).astype(F) for name in ("hfx", "qfx")}
                state = {name: cp.ascontiguousarray(
                    cp.asarray(carried[name].reshape(1, ncol), F))
                    for name in inout}
                outputs = {name: cp.zeros((1, ncol), dtype=F)
                           for name in launcher.MYJ_SFCLAY_OUTPUTS}
                for name in ("hfx", "qfx"):
                    outputs[name][...] = cp.asarray(flux[name].reshape(1, ncol))
                launcher.launch_myj_sfclay(
                    columns, surface, state, outputs, itimestep=it0 + step,
                    sf_sfclay_physics=sf_sfclay_physics)
                cp.cuda.Device().synchronize()
                got_all = {**state, **outputs}
                for name in names:
                    got = cp.asnumpy(got_all[name]).reshape(ncol)
                    want = want_all[step, fields.index(name)]
                    collected[mode][name][0].append(got)
                    collected[mode][name][1].append(want)
                    stats = _stats(got, want)
                    if stats["mismatch"]:
                        per_set.setdefault(f"{mode}/step{step}", {})[name] = stats
                carried = {name: cp.asnumpy(state[name]).reshape(ncol)
                           for name in inout}
                carried_flux = {name: cp.asnumpy(outputs[name]).reshape(ncol)
                                for name in ("hfx", "qfx")}
        report["per_set"][set_name] = per_set
    for mode in ("replay", "free_run"):
        for name in names:
            got = np.concatenate(collected[mode][name][0])
            want = np.concatenate(collected[mode][name][1])
            report[mode][name] = _stats(got, want)
    return report


def failing_fields(report: dict) -> dict:
    """``{mode: {field: stats}}`` for every field with any bit mismatch."""
    return {mode: {name: stats for name, stats in report[mode].items()
                   if stats["mismatch"]}
            for mode in ("replay", "free_run")}


def format_report(tables: dict, report: dict | None, header: str) -> str:
    lines = [header, "== QNSESFCINIT tables (host build vs WRF)"]
    for name, stats in tables.items():
        lines.append(f"  {name:9s} max_ulp {stats['max_ulp']:>12d}  "
                     f"bit_mismatch {stats['mismatch']:>6d}/{stats['count']:<6d}"
                     f"  max_abs {stats['max_abs']:.3e}")
    if report is None:
        return "\n".join(lines) + "\n"
    for mode in ("replay", "free_run"):
        lines.append(f"== {mode}")
        for name, stats in report[mode].items():
            lines.append(
                f"  {name:8s} max_ulp {stats['max_ulp']:>12d}  bit_mismatch "
                f"{stats['mismatch']:>5d}/{stats['count']:<5d}  max_abs "
                f"{stats['max_abs']:.3e}  max_rel {stats['max_rel']:.3e}  "
                f"flushed_subnormal {stats['flushed_subnormal']}")
    lines.append("== per set, calls with any bit mismatch")
    for set_name, per_set in report["per_set"].items():
        lines.append(f"  {set_name}: " + (", ".join(
            f"{key}[{','.join(sorted(v))}]" for key, v in per_set.items())
            or "none"))
    return "\n".join(lines) + "\n"


__all__ = ["FIXTURE", "load", "set_names", "ulp_distance", "compare_tables",
           "measure", "failing_fields", "format_report"]

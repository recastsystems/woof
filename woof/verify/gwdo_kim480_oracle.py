"""Measure the WRF v4.8.0 gwd_opt = 1 kernel against WRF v4.8.0's own words.

The reference words come from ``tools/gwdo_kim_wrf480_oracle``: build.sh
links ``run_gwdo_kim480.F90`` against the object files of the native WRF
v4.8.0 strict referee build itself (``WRF-4.8.0-serial-pristine`` on test host B:
GNU Fortran 13.3.0, glibc 2.39, ``-O2 -fno-fast-math -ffp-contract=off
-fno-tree-vectorize``), so SUBROUTINE gwdo and bl_gwdo_run run as the
referee's machine code.  Input sets, each one stream file
``gwdo-kim480-inputs-S.bin`` (layout in run_gwdo_kim480.F90) and its
``.npz``:

* ``L50``, ``L90``: the 149 regime columns of the WRF v4.6.1 oracle
  (tests/data/oracles/gwdo_kim) with ELVMAX added per column;
* ``L60top3``: the same regimes on a 60-level grid topped at 3 hPa, which
  reaches the mesospheric limiter (p <= 7.5 hPa);
* ``real-*``: real columns, the physics-driver words of a WOOF-ArWen HRRR-recipe
  history frame with the HRRR static's orographic statistics and ELVMAX
  from the 30 arc-second terrain (make_inputs.py ``real``).

:func:`run_gpu` runs the production launcher
(:func:`woof.core.terrain_drag.launch_gwdo`, ``gwdo_version="wrf_480"``)
over every case and :func:`compare` counts the words that are not
bit-identical per output field.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from woof.verify import gwdo_kim_oracle as go

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "gwdo_kim480")
LEVEL_INPUTS = go.LEVEL_INPUTS
COLUMN_INPUTS = ("var", "oc1", "sina", "cosa", "elvmax")
LEVEL_FIELDS = go.LEVEL_FIELDS
SURFACE_FIELDS = go.SURFACE_FIELDS
OUTPUT_FIELDS = go.OUTPUT_FIELDS


def write_inputs_bin(path, f) -> None:
    """The 4.8.0 driver's stream file from the input arrays."""
    nz, ncol = f["u"].shape
    ncase = len(f["dx"])
    with open(path, "wb") as fh:
        np.asarray([ncol, nz, ncase], "<i4").tofile(fh)
        np.asarray(f["dx"], "<f4").tofile(fh)
        np.asarray(f["dt"], "<f4").tofile(fh)
        np.asarray(f["dx_factor"], "<f4").tofile(fh)
        np.asarray(f["if_nonhyd"], "<i4").tofile(fh)
        for name in LEVEL_INPUTS:
            np.asarray(f[name], "<f4").tofile(fh)
        np.asarray(f["p_interface"], "<f4").tofile(fh)
        for name in COLUMN_INPUTS:
            np.asarray(f[name], "<f4").tofile(fh)
        np.asarray(f["oa"], "<f4").tofile(fh)
        np.asarray(f["ol"], "<f4").tofile(fh)


def write_inputs_bin471(path, f) -> None:
    """The same columns in the WRF v4.6.1/v4.7.1 driver's layout
    (tools/gwdo_kim_wrf461_oracle/run_gwdo_kim.F90): no ELVMAX, no
    dx_factor or if_nonhyd, one case per distinct (dx, dt)."""
    nz, ncol = f["u"].shape
    dx, dt = distinct_dx_dt(f)
    with open(path, "wb") as fh:
        np.asarray([ncol, nz, len(dx)], "<i4").tofile(fh)
        np.asarray(dx, "<f4").tofile(fh)
        np.asarray(dt, "<f4").tofile(fh)
        for name in LEVEL_INPUTS:
            np.asarray(f[name], "<f4").tofile(fh)
        np.asarray(f["p_interface"], "<f4").tofile(fh)
        for name in ("var", "oc1", "sina", "cosa"):
            np.asarray(f[name], "<f4").tofile(fh)
        np.asarray(f["oa"], "<f4").tofile(fh)
        np.asarray(f["ol"], "<f4").tofile(fh)


def distinct_dx_dt(f):
    seen = []
    for a, b in zip(np.asarray(f["dx"], np.float32),
                    np.asarray(f["dt"], np.float32)):
        if (a, b) not in seen:
            seen.append((a, b))
    return (np.asarray([s[0] for s in seen], np.float32),
            np.asarray([s[1] for s in seen], np.float32))


def read_inputs_bin(path) -> dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    ncol, nz, ncase = (int(x) for x in np.frombuffer(raw[:12], "<i4"))
    off = 12

    def take(count, shape=None, dtype="<f4"):
        nonlocal off
        a = np.frombuffer(raw, dtype, count, off).copy()
        off += 4 * count
        return a if shape is None else a.reshape(shape)

    out = {"dx": take(ncase), "dt": take(ncase), "dx_factor": take(ncase),
           "if_nonhyd": take(ncase, dtype="<i4")}
    for name in LEVEL_INPUTS:
        out[name] = take(nz * ncol, (nz, ncol))
    out["p_interface"] = take((nz + 1) * ncol, (nz + 1, ncol))
    for name in COLUMN_INPUTS:
        out[name] = take(ncol)
    out["oa"] = take(4 * ncol, (4, ncol))
    out["ol"] = take(4 * ncol, (4, ncol))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return out


read_outputs_bin = go.read_outputs_bin


def load_fixture(name: str, directory=FIXTURE_DIR):
    directory = Path(directory)
    with np.load(directory / f"gwdo-kim480-inputs-{name}.npz") as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(directory / f"gwdo-kim480-wrf480-{name}.npz") as f:
        wrf = {k: f[k] for k in f.files}
    return inputs, wrf


def run_gpu(inputs, *, gwdo_version: str = "wrf_480",
            cases=None) -> dict[str, np.ndarray]:
    """``launch_gwdo`` once per case, laid out as the fixture.

    ``gwdo_version="wrf_471"`` runs the frozen WRF v4.6.1/v4.7.1 kernel on
    the distinct (dx, dt) cases (ELVMAX and the two keys unread), for the
    two-sided replay against the WRF v4.7.1 referee's objects.
    """
    import cupy as cp
    from woof.core.terrain_drag import launch_gwdo

    nz, ncol = inputs["u"].shape

    def lev(a):
        return cp.asarray(np.ascontiguousarray(a, np.float32)
                          .reshape(a.shape[0], 1, ncol))

    def row(a):
        return np.ascontiguousarray(a, np.float32).reshape(1, ncol)

    atm = {"u": lev(inputs["u"]), "v": lev(inputs["v"]),
           "temperature": lev(inputs["t"]), "qv": lev(inputs["qv"]),
           "pressure": lev(inputs["p"]), "exner": lev(inputs["pi"]),
           "p_interface": lev(inputs["p_interface"]), "z": lev(inputs["z"]),
           "dz": lev(inputs["dz"])}
    oa = [row(inputs["oa"][m]) for m in range(4)]
    ol = [row(inputs["ol"][m]) for m in range(4)]
    if cases is None:
        if gwdo_version == "wrf_480":
            cases = list(zip(inputs["dx"], inputs["dt"], inputs["dx_factor"],
                             inputs["if_nonhyd"]))
        else:
            dx, dt = distinct_dx_dt(inputs)
            cases = [(a, b, 2.0, 1) for a, b in zip(dx, dt)]
    out = {k: [] for k in OUTPUT_FIELDS}
    for dx, dt, dxf, nh in cases:
        du = lev(inputs["rublten0"])
        dv = lev(inputs["rvblten0"])
        diag = {}
        launch_gwdo(atm, du, dv, var=row(inputs["var"]),
                    con=row(inputs["oc1"]), oa=oa, ol=ol,
                    sina=row(inputs["sina"]), cosa=row(inputs["cosa"]),
                    dx=float(dx), dt=float(dt), diagnostics=diag,
                    gwdo_version=gwdo_version,
                    omax=row(inputs["elvmax"]), dx_factor=float(dxf),
                    if_nonhyd=bool(nh))
        out["rublten"].append(cp.asnumpy(du)[:, 0, :])
        out["rvblten"].append(cp.asnumpy(dv)[:, 0, :])
        out["dtaux3d"].append(cp.asnumpy(diag["DTAUX3D"])[:, 0, :])
        out["dtauy3d"].append(cp.asnumpy(diag["DTAUY3D"])[:, 0, :])
        out["dusfcg"].append(cp.asnumpy(diag["DUSFCG"])[0])
        out["dvsfcg"].append(cp.asnumpy(diag["DVSFCG"])[0])
    return {k: np.ascontiguousarray(np.stack(v)) for k, v in out.items()}


compare = go.compare
activity = go.activity


def blocking_columns(inputs, wrf) -> list[int]:
    """Per case, the columns whose drag reaches the lowest level with a
    stress that falls linearly through several levels: a flow-blocking
    column (bl_gwdo.F90:575-590), counted so a fixture can show it has
    them.  A column counts when WRF moved its lowest two levels' RUBLTEN."""
    moved = (wrf["rublten"].view(np.uint32)
             != inputs["rublten0"][None].view(np.uint32))
    return [int((moved[c, 0] & moved[c, 1]).sum())
            for c in range(moved.shape[0])]

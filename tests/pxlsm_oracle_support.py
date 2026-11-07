"""Read the Pleim-Xiu oracle fixtures and run the port on them.

The fixtures (woof/data/pxlsm/oracle/) are what
tools/pxlsm_wrf461_oracle/build.sh wrote: one case file per call geometry
(gen_cases.py) and the as-built WRF v4.6.1 pxlsm's state after each of its
NSTEPS calls (run_pxlsm.F90).  The field order is
tools/pxlsm_wrf461_oracle/schema.py's, imported from there so the reader
and the writer cannot disagree.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from woof.core.fp32_ulp import monotone_fp32_key

REPO = Path(__file__).resolve().parents[1]
ORACLE_DIR = REPO / "woof" / "data" / "pxlsm" / "oracle"
TOOL_DIR = REPO / "tools" / "pxlsm_wrf461_oracle"


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(
        f"pxlsm_oracle_{name}", TOOL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


schema = _load_tool("schema")

#: Oracle 2-D outputs as the port names them (the driver's TA2/QA2 are the
#: grid's T2/Q2, module_surface_driver.F:3632).
PORT_NAME = {"ta2": "t2", "qa2": "q2"}

#: The fields every step of the output file carries, in order.
OUTPUT_FIELDS = (tuple(schema.INOUT_2D) + tuple(schema.OUT_2D) + ("isltyp",)
                 + tuple(schema.SOIL_3D))


@dataclass
class OracleCase:
    name: str
    nx: int
    ny: int
    nlcat: int
    nscat: int
    nsteps: int
    smois_init: int
    soil_nudge: int
    modis_veg: int
    anal_interval: int
    itimestep0: int
    dt: float
    dtbl: float
    curr_secs0: float
    dzs: tuple[float, float]
    a2: dict
    atm: dict
    soil: dict
    isltyp: np.ndarray
    landusef: np.ndarray
    soilcbot: np.ndarray
    expected: list


def cases_index() -> list[dict]:
    return json.loads((ORACLE_DIR / "cases-index.json").read_text())


def read_case(name: str, *, directory: Path = ORACLE_DIR) -> OracleCase:
    raw = (directory / f"case-{name}.bin").read_bytes()
    hi = np.frombuffer(raw, dtype="<i4", count=16)
    hf = np.frombuffer(raw, dtype="<f4", count=8, offset=64)
    nx, ny, nlcat, nscat, nsteps = (int(v) for v in hi[:5])
    off = 96
    plane = nx * ny

    def take(count, dtype="<f4"):
        nonlocal off
        a = np.frombuffer(raw, dtype=dtype, count=count, offset=off).copy()
        off += 4 * count
        return a

    a2 = {}
    for field in schema.IN_2D + schema.INOUT_2D + schema.OUT_2D:
        a2[field] = take(plane).reshape(ny, nx)
    isltyp = take(plane, "<i4").reshape(ny, nx)
    atm = {field: take(2 * plane).reshape(ny, 2, nx) for field in schema.ATM_3D}
    soil = {field: take(2 * plane).reshape(ny, 2, nx) for field in schema.SOIL_3D}
    landusef = take(nlcat * plane).reshape(ny, nlcat, nx)
    take(nscat * plane)                         # SOILCTOP: pxlsm never reads it
    soilcbot = take(nscat * plane).reshape(ny, nscat, nx)
    if off != len(raw):
        raise ValueError(f"{name}: {len(raw) - off} trailing bytes")

    out = np.fromfile(directory / f"out-{name}.bin", dtype="<u4")
    per_step = (len(schema.INOUT_2D) + len(schema.OUT_2D) + 1) * plane \
        + 2 * 2 * plane
    if out.size != nsteps * per_step:
        raise ValueError(f"{name}: output holds {out.size} words, "
                         f"expected {nsteps * per_step}")
    expected = []
    for step in range(nsteps):
        words = out[step * per_step:(step + 1) * per_step]
        record, o = {}, 0
        for field in schema.INOUT_2D + schema.OUT_2D + ("isltyp",):
            record[field] = words[o:o + plane].reshape(ny, nx)
            o += plane
        for field in schema.SOIL_3D:
            record[field] = words[o:o + 2 * plane].reshape(ny, 2, nx)
            o += 2 * plane
        expected.append(record)
    return OracleCase(
        name=name, nx=nx, ny=ny, nlcat=nlcat, nscat=nscat, nsteps=nsteps,
        smois_init=int(hi[5]), soil_nudge=int(hi[6]), modis_veg=int(hi[7]),
        anal_interval=int(hi[8]), itimestep0=int(hi[9]), dt=float(hf[0]),
        dtbl=float(hf[1]), curr_secs0=float(hf[2]),
        dzs=(float(hf[3]), float(hf[4])), a2=a2, atm=atm, soil=soil,
        isltyp=isltyp, landusef=landusef, soilcbot=soilcbot,
        expected=expected)


def device_state(case: OracleCase):
    """The port's field dict for ``case``'s initial state, on the device."""
    import cupy as cp
    from woof.core.pxlsm import (PXLSM_CARRIED, PXLSM_INPUTS,
                                  PXLSM_PUBLISHED)
    fields = {}
    for field in schema.IN_2D + schema.INOUT_2D + schema.OUT_2D:
        fields[PORT_NAME.get(field, field)] = cp.asarray(
            np.ascontiguousarray(case.a2[field]))
    fields["isltyp"] = cp.asarray(np.ascontiguousarray(case.isltyp))
    for field in schema.SOIL_3D:
        fields[field] = cp.asarray(np.ascontiguousarray(
            case.soil[field].transpose(1, 0, 2)))
    for name in PXLSM_INPUTS + PXLSM_CARRIED + PXLSM_PUBLISHED:
        assert name in fields, name
    atm = {
        "dz8w1": case.atm["dz8w"][:, 0, :], "dz8w2": case.atm["dz8w"][:, 1, :],
        "qv3d1": case.atm["qv3d"][:, 0, :], "qv3d2": case.atm["qv3d"][:, 1, :],
        "t3d1": case.atm["t3d"][:, 0, :], "th3d1": case.atm["th3d"][:, 0, :],
        "rho1": case.atm["rho"][:, 0, :],
    }
    atm = {k: cp.asarray(np.ascontiguousarray(v)) for k, v in atm.items()}
    landusef = cp.asarray(np.ascontiguousarray(case.landusef.transpose(1, 0, 2)))
    soilcbot = cp.asarray(np.ascontiguousarray(case.soilcbot.transpose(1, 0, 2)))
    return fields, atm, landusef, soilcbot


def port_words(fields) -> dict:
    """The port's state as the oracle's per-step record, in uint32 words."""
    import cupy as cp
    record = {}
    for field in schema.INOUT_2D + schema.OUT_2D:
        record[field] = cp.asnumpy(fields[PORT_NAME.get(field, field)]).view(np.uint32)
    record["isltyp"] = cp.asnumpy(fields["isltyp"]).view(np.uint32)
    for field in schema.SOIL_3D:
        record[field] = np.ascontiguousarray(
            cp.asnumpy(fields[field]).transpose(1, 0, 2)).view(np.uint32)
    return record


def run_port(case: OracleCase, *, table_override=None):
    """Run the port over ``case`` and yield its record after every call.

    CURR_SECS advances by DT in binary32, as run_pxlsm.F90 advances it.
    ``table_override`` (nlcat -> (9, nlcat) array) replaces the packaged
    land-use table, for the planted-bit test.
    """
    from woof.core import pxlsm
    fields, atm, landusef, soilcbot = device_state(case)
    curr = np.float32(case.curr_secs0)
    original = pxlsm._device_table
    if table_override is not None:
        import cupy as cp
        table = cp.asarray(table_override)
        pxlsm._device_table = lambda nlcat: table
    try:
        for step in range(case.nsteps):
            pxlsm.launch_pxlsm(
                fields, atm, landusef=landusef, soilcbot=soilcbot,
                dt=case.dt, dtbl=case.dtbl,
                itimestep=case.itimestep0 + step, curr_secs=float(curr),
                smois_init=case.smois_init, soil_nudge=case.soil_nudge,
                modis_veg=case.modis_veg, anal_interval=case.anal_interval,
                domain_shape=(case.ny, case.nx), dzs=case.dzs)
            yield port_words(fields)
            curr = np.float32(curr + np.float32(case.dt))
    finally:
        pxlsm._device_table = original


def as_float(words) -> np.ndarray:
    return np.asarray(words, dtype=np.uint32).view(np.float32)


def ulp_distance(a_words, b_words) -> np.ndarray:
    """Units in the last place between two binary32 word arrays."""
    a = np.ascontiguousarray(a_words, dtype=np.uint32).view(np.float32)
    b = np.ascontiguousarray(b_words, dtype=np.uint32).view(np.float32)
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))

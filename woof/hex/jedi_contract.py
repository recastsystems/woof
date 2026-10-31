"""The JEDI file contract for RRFS v2 (stage H6, tier T1).

RRFS v2's analysis is NOAA's MPAS-JEDI (RDASApp), unchanged, on CPU.  It and
the model exchange ONE file, ``mpasout.nc``, in native MPAS form:

* model -> JEDI: the 27 fields of the workflow's ``background`` stream
  (rrfs-workflow 85ea55c0 ``fix/stream_list/hrrrv5/stream_list.atmosphere.background``);
* JEDI -> model: 12 of them written back in place through the ``analysis``
  stream (``stream_list.atmosphere.analysis``);
* the next forecast restarts with ``config_do_DAcycling = true``, which in
  upstream MPAS-A v8.4.1 (``mpas_atm_core.F:522-530``) re-runs
  ``atm_init_coupled_diagnostics`` on the restored uncoupled fields.

This module is that contract and nothing else.  It adds NO dycore arithmetic:
the DA-cycling recoupling is hex's existing transcription of
``atm_init_coupled_diagnostics`` (``woof.hex.driver.load_mpas_initial_state``,
the routine that follows native cold-start recoupling numerics), fed the
analysis's uncoupled fields exactly as upstream feeds them on a DA-cycling
restart.

What the file carries and where each field comes from is recorded per
variable (``hex_provenance``) and in the sidecar receipt, because three of the
27 are not produced by hex's current physics suite (see FIELD_ROLES).  The
soil fields are a declared interim: hex's land model carries four Noah-MP
layers and RRFS v2's namelist declares nine RUC levels, so the four layers are
mapped onto the nine depths by :func:`interim_soil_levels` until hex carries
nine soil levels of its own (stage H4a).  The mapping is labelled on the file,
in the receipt and in this module, and never presented as RUC state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


# -- NetCDF goes through the engine's Rust bridges --------------------------
# The 2.5.0 Python boundary puts every NetCDF read and write on Rust.  The
# engine ships both sides: rw_netcdf behind ``woof.netcdf_bridge`` (read)
# and the netcdf-writer cdylib behind ``woof.io.nc_writer_bridge`` (CDF-1,
# CDF-2 and CDF-5 with a record dimension).  There is no netCDF4-python
# fallback: a missing bridge is refused by name, because a silent fallback
# would be the Python data path this replaced.
RUST_NETCDF_WRITER = "woof.io.nc_writer_bridge (netcdf-writer, CDF-5)"
RUST_NETCDF_READER = "woof.netcdf_bridge (rw_netcdf)"


def _rust_cdf5_schema() -> Any:
    from woof.io import nc_writer_bridge

    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        raise ContractRefusal(
            f"the JEDI contract writes CDF-5 through {RUST_NETCDF_WRITER}, "
            f"which is unavailable here: {reason}"
        )
    return nc_writer_bridge.ClassicSchema("cdf5")


def _rust_open(path: Any) -> Any:
    from woof.netcdf_bridge import open_dataset

    return open_dataset(Path(path))


# -- the public stream lists (rrfs-workflow 85ea55c0, hrrrv5) ---------------
BACKGROUND_FIELDS: tuple[str, ...] = (
    "cldfrac", "lai", "pressure_base", "pressure_p", "re_cloud", "re_ice",
    "re_snow", "rho", "scalars", "skintemp", "smois", "snowc", "snowh",
    "surface_pressure", "theta", "tslb", "u", "u10", "uReconstructZonal",
    "uReconstructMeridional", "v10", "vegfra", "w", "xice", "refl10cm",
    "t2m", "q2",
)
ANALYSIS_FIELDS: tuple[str, ...] = (
    "pressure_p", "rho", "scalars", "surface_pressure", "theta", "u",
    "uReconstructZonal", "uReconstructMeridional", "refl10cm", "w", "t2m",
    "q2",
)
# The analysed fields a DA-cycling restart actually consumes: upstream
# atm_init_coupled_diagnostics reads rho, theta, qv, u and w and RECOMPUTES
# pressure_p, theta_m, rho_zz, ru, rw, rho_p, rtheta_p and exner from them.
# surface_pressure, the reconstructed winds, t2m, q2 and refl10cm are
# diagnostics the model recomputes before it reads them again.
RECOUPLED_INPUTS: tuple[str, ...] = ("rho", "theta", "scalars", "u", "w")

# Six WSM6-family species.  JEDI maps these names (jedivar_header.yaml).
DEFAULT_SCALAR_NAMES: tuple[str, ...] = ("qv", "qc", "qr", "qi", "qs", "qg")
# The rest of RRFS v2's ``scalars`` array (TEMPO, hail- and aerosol-aware, as
# the hrrrv5 namelist selects).  MPAS-JEDI writes its analysis back through
# the ``analysis`` stream INTO the existing mpasout.nc, and RDASApp's MPAS
# cannot add a variable to an existing file: with any of these absent the
# write fails ("MPAS IO Error: Undefined field", MPAS_stream_mgr_write
# ierr=-3; MEASURED 2026-10-05).  Hex's WSM6 carries none of them,
# so they are written as MPAS's allocation default 0 and named as absent.
RRFS_EXTRA_SCALARS: tuple[str, ...] = ("nc", "ni", "nr", "ng", "nifa", "nwfa", "volg")

# RUC LSM's nine levels (metres below the surface).  RRFS v2 runs RUC with
# config_nsoillevels = 9 (exrrfs_fcst.sh:53, jedivar_header.yaml geometry).
RUC_NINE_LEVEL_DEPTHS_M: tuple[float, ...] = (
    0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00,
)
# Noah-MP's four layers, thicknesses 0.1/0.3/0.6/1.0 m, as layer centres.
NOAH_FOUR_LAYER_CENTRES_M: tuple[float, ...] = (0.05, 0.25, 0.70, 1.50)

NATIVE_SOIL_LABEL = "native soil column of the selected land surface, {levels} levels, no mapping"


def soil_mapping_label(provenance: Mapping[str, str]) -> str:
    """What the file's soil fields are, read off their provenance lines."""

    line = str(provenance.get("smois", ""))
    if INTERIM_SOIL_LABEL in line:
        return INTERIM_SOIL_LABEL
    return line.removeprefix("physics: ").removesuffix(" (moisture)")


INTERIM_SOIL_LABEL = (
    "INTERIM soil mapping, not RUC state: hex's Noah-MP four layers "
    "(centres 0.05/0.25/0.70/1.50 m) linearly interpolated in depth onto "
    "RUC's nine levels (0/0.01/0.04/0.10/0.30/0.60/1.00/1.60/3.00 m), held "
    "constant above the first and below the last centre.  Replaced when hex "
    "carries nine soil levels of its own (RRFS port plan stage H4a)."
)

SCHEMA = "gpuwm-hex.jedi-contract/v1"
MPAS_STRLEN = 64

# The workflow's jedivar streams read the 181-field ``da_state`` stream from
# the same mpasout.nc at geometry construction (input_interval initial_only),
# and RDASApp's MPAS refuses the file outright when a dimension any da_state
# field is declared on is absent ("the 'nLags' dimension is not defined",
# MEASURED 2026-10-05).  These are the namelist-defined dimensions
# those fields use, at RDASApp MPAS (sorc/mpas 70bdd40) Registry defaults:
# input_soil_temperature_lag = 140, noznlev = 59.  The fields themselves are
# tier T3 (the full 181-field stream); T1 declares the dimensions only.
DA_STATE_DIMENSIONS: dict[str, int] = {"nLags": 140, "nOznLevels": 59}

# Where each of the 27 comes from.  "dycore": hex's MPAS-A v8.4.1 state;
# "physics": hex's own physics suite state or diagnostic; "init": carried from
# the initial-condition file because hex's suite does not prognose it;
# "absent": hex's suite does not compute it and the file carries MPAS's
# allocation default (0) -- named, never silent.
FIELD_ROLES: dict[str, str] = {
    "pressure_base": "dycore", "pressure_p": "dycore", "rho": "dycore",
    "scalars": "dycore", "surface_pressure": "dycore", "theta": "dycore",
    "u": "dycore", "uReconstructZonal": "dycore",
    "uReconstructMeridional": "dycore", "w": "dycore",
    "skintemp": "physics", "smois": "physics (interim 4->9 mapping)",
    "tslb": "physics (interim 4->9 mapping)", "u10": "physics",
    "v10": "physics", "t2m": "physics", "q2": "physics",
    "refl10cm": "physics",
    "cldfrac": "physics if exported, else absent",
    "re_cloud": "physics if exported, else absent",
    "re_ice": "physics if exported, else absent",
    "re_snow": "physics if exported, else absent",
    "snowc": "physics if exported, else init",
    "snowh": "physics if exported, else init",
    "xice": "physics if exported, else init",
    "vegfra": "physics if exported, else init",
    "lai": "physics if exported, else init lai12m at the valid date",
}

_UNITS = {
    "cldfrac": ("unitless", "horizontal cloud fraction"),
    "lai": ("m^2 m^{-2}", "leaf area index"),
    "pressure_base": ("Pa", "Base state pressure"),
    "pressure_p": ("Pa", "Perturbation pressure"),
    "re_cloud": ("m", "effective radius of cloud water droplets"),
    "re_ice": ("m", "effective radius of cloud ice crystals"),
    "re_snow": ("m", "effective radius of snow crystals"),
    "rho": ("kg m^{-3}", "Dry air density"),
    "skintemp": ("K", "Skin temperature"),
    "smois": ("m^3 m^{-3}", "soil moisture"),
    "snowc": ("unitless", "flag indicating snow coverage (1 for snow cover)"),
    "snowh": ("m", "physical snow depth"),
    "surface_pressure": ("Pa", "Diagnosed surface pressure"),
    "theta": ("K", "Potential temperature"),
    "tslb": ("K", "soil temperature"),
    "u": ("m s^{-1}", "Horizontal normal velocity at edges"),
    "u10": ("m s^{-1}", "10-meter zonal wind"),
    "uReconstructZonal": ("m s^{-1}", "Zonal component of reconstructed horizontal velocity at cell centers"),
    "uReconstructMeridional": ("m s^{-1}", "Meridional component of reconstructed horizontal velocity at cell centers"),
    "v10": ("m s^{-1}", "10-meter meridional wind"),
    "vegfra": ("unitless", "vegetation fraction"),
    "w": ("m s^{-1}", "Vertical velocity"),
    "xice": ("unitless", "fractional area coverage of sea-ice"),
    "refl10cm": ("dBZ", "10 cm radar reflectivity"),
    "t2m": ("K", "2-meter temperature"),
    "q2": ("kg kg^{-1}", "2-meter specific humidity"),
}
_SCALAR_LONG = {
    "qv": "Water vapor mixing ratio", "qc": "Cloud water mixing ratio",
    "qr": "Rain water mixing ratio", "qi": "Ice mixing ratio",
    "qs": "Snow mixing ratio", "qg": "Graupel mixing ratio",
    "nc": "Cloud water number concentration", "ni": "Cloud ice number concentration",
    "nr": "Rain number concentration", "ng": "Graupel number concentration",
    "nifa": "Ice-friendly aerosol number concentration",
    "nwfa": "Water-friendly aerosol number concentration",
    "volg": "Graupel particle volume",
}
_SCALAR_UNITS = {"nc": "nb kg^{-1}", "ni": "nb kg^{-1}", "nr": "nb kg^{-1}", "ng": "nb kg^{-1}",
                 "nifa": "nb kg^{-1}", "nwfa": "nb kg^{-1}", "volg": "m^3 kg^{-1}"}


class ContractRefusal(ValueError):
    """A file that would break the contract is refused by name."""


def interim_soil_levels(values: np.ndarray) -> np.ndarray:
    """Map ``[4, nCells]`` Noah-MP layers onto RUC's nine depths (INTERIM).

    Linear in depth between layer centres, constant outside them.  Exact for
    a column that is constant in depth; never invents a value outside the
    column's own range.
    """

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] != len(NOAH_FOUR_LAYER_CENTRES_M):
        raise ContractRefusal(
            f"interim soil mapping takes [4, nCells] Noah-MP layers, got {array.shape}"
        )
    source = np.asarray(NOAH_FOUR_LAYER_CENTRES_M, dtype=np.float64)
    target = np.asarray(RUC_NINE_LEVEL_DEPTHS_M, dtype=np.float64)
    upper = np.clip(np.searchsorted(source, target, side="right"), 1, len(source) - 1)
    lower = upper - 1
    weight = (target - source[lower]) / (source[upper] - source[lower])
    weight = np.clip(weight, 0.0, 1.0)
    data = array.astype(np.float64)
    mapped = data[lower, :] * (1.0 - weight)[:, None] + data[upper, :] * weight[:, None]
    return mapped.astype(array.dtype, copy=False)


def mpas_xtime(valid: datetime) -> str:
    return valid.strftime("%Y-%m-%d_%H:%M:%S")


def monthly_to_date(monthly: np.ndarray, valid: datetime) -> np.ndarray:
    """``[12, nCells]`` mid-month values to one date, MPAS's linear form.

    The form of ``monthly_interp_to_date`` (mpas_atmphys_date_time.F): values
    sit on the 15th of each month and a date between two mid-months takes the
    day-weighted mean of the two.
    """

    table = np.asarray(monthly, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] != 12:
        raise ContractRefusal(f"monthly table must be [12, nCells], got {table.shape}")
    import calendar

    year, month, day = valid.year, valid.month, valid.day
    if day >= 15:
        first = month - 1
        second = month % 12
        days = calendar.monthrange(year, month)[1]
        span = float(days)
        offset = float(day - 15)
    else:
        first = (month - 2) % 12
        second = month - 1
        prev_month = 12 if month == 1 else month - 1
        prev_year = year - 1 if month == 1 else year
        span = float(calendar.monthrange(prev_year, prev_month)[1])
        offset = float(day + span - 15)
    fraction = offset / span
    return ((1.0 - fraction) * table[first] + fraction * table[second]).astype(np.float32)


@dataclass(frozen=True)
class BackgroundRecord:
    path: str
    bytes: int
    sha256: str
    receipt_path: str


def _orient(name: str, array: np.ndarray, n_cells: int, n_levels: int) -> np.ndarray:
    """Normalize an exported physics array to ``[nCells]`` or ``[nLevels, nCells]``.

    The physics seam exports WRF-shaped arrays ([nz, 1, ncol], [1, 1, ncol],
    [ncol] ...), possibly one column wider on a limited-area mesh (native's
    garbage element).  Singleton axes are dropped and the garbage column is
    stripped; any other shape is refused by name.
    """

    data = np.asarray(array)
    data = np.squeeze(data)
    if data.ndim >= 1 and data.shape[-1] == n_cells + 1:
        data = data[..., :n_cells]
    if data.ndim == 1 and data.shape[0] == n_cells:
        return np.ascontiguousarray(data, dtype=np.float32)
    if data.ndim == 2:
        if data.shape == (n_levels, n_cells):
            return np.ascontiguousarray(data, dtype=np.float32)
        if data.shape == (n_cells, n_levels):
            return np.ascontiguousarray(data.T, dtype=np.float32)
        if data.shape[1] == n_cells + 1 and data.shape[0] == n_levels:
            return np.ascontiguousarray(data[:, :n_cells], dtype=np.float32)
    raise ContractRefusal(
        f"exported physics field {name!r} has shape {np.asarray(array).shape}; "
        f"expected [nCells]={n_cells} or [nLevels, nCells]=({n_levels}, {n_cells})"
    )


def assemble_background_fields(
    *,
    snapshot_arrays: Mapping[str, np.ndarray],
    pressure_perturbation: np.ndarray,
    pressure_base: np.ndarray,
    seam_fields: Mapping[str, np.ndarray],
    init_fallback: Mapping[str, np.ndarray],
    scalar_names: Sequence[str],
    n_cells: int,
    n_edges: int,
    n_levels: int,
    soil_levels: int,
    valid: datetime,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The 27 fields in hex's logical layout, plus one provenance line each.

    ``snapshot_arrays`` is a captured history frame (garbage column already
    stripped); ``seam_fields`` is the physics seam's exported state keyed by
    leaf name; ``init_fallback`` carries the init's snowc/snowh/xice/vegfra
    and its monthly lai12m.  Every field is checked for shape and finiteness.
    """

    fields: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}

    def take(name: str, value: np.ndarray, source: str, shape: tuple[int, ...]) -> None:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        if array.shape != shape:
            raise ContractRefusal(f"{name}: shape {array.shape} != expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ContractRefusal(f"{name}: non-finite values in the background")
        fields[name] = array
        provenance[name] = source

    cell3 = (n_levels, n_cells)
    snap = snapshot_arrays
    for name in scalar_names:
        take(name, snap[name], "dycore: hex prognostic scalar", cell3)
    take("theta", snap["theta"], "dycore: theta_m / (1 + rv/rd qv), hex prep", cell3)
    take("rho", snap["rho"], "dycore: dry density rho_zz * zz, hex prep", cell3)
    take("surface_pressure", snap["surface_pressure"], "dycore: hex prep diagnosed surface pressure", (n_cells,))
    # The history capture strips a limited-area run's garbage CELL; the edge
    # arrays still carry native's garbage EDGE, stripped here.
    take("u", _strip(snap["normal_u"], n_edges), "dycore: hex normal velocity at edges", (n_levels, n_edges))
    take("w", snap["w"], "dycore: hex vertical velocity at interfaces", (n_levels + 1, n_cells))
    take("uReconstructZonal", snap["u_zonal"], "dycore: hex MPAS reconstruction", cell3)
    take("uReconstructMeridional", snap["v_meridional"], "dycore: hex MPAS reconstruction", cell3)
    take("pressure_p", _strip(pressure_perturbation, n_cells, 1), "dycore: hex saved pressure perturbation", cell3)
    take("pressure_base", _strip(pressure_base, n_cells, 1), "dycore: hex reference pressure_base", cell3)
    take("skintemp", snap["tsk"], "physics: hex land/sea surface skin temperature", (n_cells,))
    take("t2m", snap["t2"], "physics: hex surface-layer diagnostic", (n_cells,))
    take("q2", snap["q2"], "physics: hex surface-layer diagnostic", (n_cells,))
    take("u10", snap["u10"], "physics: hex surface-layer diagnostic", (n_cells,))
    take("v10", snap["v10"], "physics: hex surface-layer diagnostic", (n_cells,))
    take("refl10cm", snap["refl10cm"], "physics: the selected microphysics row's reflectivity at this frame", cell3)

    if "smois" not in snap or "tslb" not in snap:
        # A land-surface-off forecast publishes no soil column (land surface
        # off, golden 0.4.0); the MPAS-JEDI state carries smois/tslb and a
        # made-up column would hand the analysis soil no scheme ever set.
        raise ContractRefusal(
            "the frame carries no soil column (land surface off): the MPAS-JEDI "
            "state needs smois and tslb, so it cannot be written from this run")
    soil4_m = np.asarray(snap["smois"], dtype=np.float32)
    soil4_t = np.asarray(snap["tslb"], dtype=np.float32)
    if soil_levels == soil4_m.shape[0]:
        # The selected land surface's own column, no mapping: four Noah-MP
        # layers on the frozen row, RUC's nine on a suite row (stage H4a).
        native = NATIVE_SOIL_LABEL.format(levels=soil_levels)
        take("smois", soil4_m, "physics: " + native + " (moisture)", (soil_levels, n_cells))
        take("tslb", soil4_t, "physics: " + native + " (temperature)", (soil_levels, n_cells))
    elif soil_levels == len(RUC_NINE_LEVEL_DEPTHS_M):
        take("smois", interim_soil_levels(soil4_m), "physics: " + INTERIM_SOIL_LABEL, (soil_levels, n_cells))
        take("tslb", interim_soil_levels(soil4_t), "physics: " + INTERIM_SOIL_LABEL, (soil_levels, n_cells))
    else:
        raise ContractRefusal(
            f"soil levels {soil_levels}: hex carries {soil4_m.shape[0]} and the "
            f"only declared mapping is onto RUC's {len(RUC_NINE_LEVEL_DEPTHS_M)}"
        )

    candidates = {
        "snowc": ("snowc",), "snowh": ("snowh",), "xice": ("xice",),
        "vegfra": ("vegfra",), "lai": ("lai",),
        "cldfrac": ("cldfra", "cldfrac"), "re_cloud": ("re_cloud", "effc"),
        "re_ice": ("re_ice", "effi"), "re_snow": ("re_snow", "effs"),
    }
    three_d = {"cldfrac", "re_cloud", "re_ice", "re_snow"}
    for name, keys in candidates.items():
        shape = cell3 if name in three_d else (n_cells,)
        hit = next((key for key in keys if key in seam_fields), None)
        if hit is not None:
            value = _orient(hit, seam_fields[hit], n_cells, n_levels)
            source = f"physics: hex physics field {hit!r}"
            if name.startswith("re_"):
                positive = value[value > 0]
                if positive.size and float(np.median(positive)) > 1.0e-3:
                    # Microns, not metres: MPAS's re_* are metres.
                    value = (value.astype(np.float64) * 1.0e-6).astype(np.float32)
                    source += " (microns converted to metres)"
            take(name, value, source, shape)
            continue
        if name == "lai" and "lai12m" in init_fallback:
            take(
                name,
                monthly_to_date(init_fallback["lai12m"], valid),
                "init: lai12m interpolated to the valid date (hex's suite carries no lai)",
                shape,
            )
        elif name in init_fallback:
            take(name, init_fallback[name], "init: carried from the initial condition (hex's suite carries no such state)", shape)
        else:
            take(
                name,
                np.zeros(shape, dtype=np.float32),
                "absent: hex's current suite does not compute it; MPAS allocation default 0 (unused by conventional observation operators)",
                shape,
            )
    for name in RRFS_EXTRA_SCALARS:
        if name not in fields:
            take(
                name,
                np.zeros(cell3, dtype=np.float32),
                "absent: a TEMPO scalar hex's WSM6 does not carry; MPAS allocation default 0, "
                "present so the analysis can be written back into the file",
                cell3,
            )
    missing = [
        name for name in BACKGROUND_FIELDS
        if name != "scalars" and name not in fields
    ]
    if missing:
        raise ContractRefusal(f"background lacks {missing}")
    return fields, provenance


def _strip(array: Any, n_cells: int, axis_from_end: int = 1) -> np.ndarray:
    data = np.asarray(array)
    if data.shape[-1] == n_cells + 1:
        data = data[..., :n_cells]
    return np.ascontiguousarray(data)


def _dims(name: str, shape: tuple[int, ...], n_cells: int, n_edges: int, n_levels: int, n_soil: int) -> tuple[str, ...]:
    if shape == (n_cells,):
        return ("Time", "nCells")
    if shape == (n_levels, n_cells):
        return ("Time", "nCells", "nVertLevels")
    if shape == (n_levels + 1, n_cells):
        return ("Time", "nCells", "nVertLevelsP1")
    if shape == (n_levels, n_edges):
        return ("Time", "nEdges", "nVertLevels")
    if shape == (n_soil, n_cells):
        return ("Time", "nCells", "nSoilLevels")
    raise ContractRefusal(f"{name}: no MPAS dimension set for shape {shape}")


# OU1 (ours, fixed outright): what proves an analysis is THIS run's state on THIS run's elements.
# The background carries the init's element identity; MPAS-JEDI writes its analysis back INTO that
# file (the analysis stream touches only the state), so the identity survives the round trip, and
# the restart compares it with the init's before any field is consumed.
IDENTITY_FIELDS: dict[str, str] = {
    "indexToCellID": "nCells", "latCell": "nCells", "lonCell": "nCells",
    "indexToEdgeID": "nEdges", "latEdge": "nEdges", "lonEdge": "nEdges",
}


def read_identity(path: Path, *, time_index: int = 0) -> dict[str, np.ndarray]:
    """The element identity of an MPAS file, in its stored types, through the Rust reader."""

    dataset = _rust_open(path)
    out: dict[str, np.ndarray] = {}
    for name, dim in IDENTITY_FIELDS.items():
        if name not in dataset.variables:
            raise ContractRefusal(
                f"{path}: no {name}, so its element order cannot be proven; an analysis on another "
                "element order would be recoupled onto this run's cells and edges without an error"
            )
        variable = dataset.variables[name]
        variable.set_auto_mask(False)
        dims = tuple(variable.dimensions)
        has_time = dims[:1] == ("Time",)
        if (dims[1:] if has_time else dims) != (dim,):
            raise ContractRefusal(f"{path}: {name} has dimensions {dims}, expected ({dim},)")
        stored = np.dtype(variable.dtype)
        # The Rust reader hands numbers back as f64; the stored type is exact for them.
        values = np.ascontiguousarray(np.asarray(variable[time_index] if has_time else variable[:]), dtype=stored)
        if not np.all(np.isfinite(values)):
            raise ContractRefusal(f"{path}: {name} carries non-finite values")
        if name.startswith("indexTo") and (stored.kind not in "iu" or np.any(values <= 0)):
            raise ContractRefusal(f"{path}: {name} must hold positive integer IDs")
        out[name] = values
    return out


def require_mpas_global_order(identity: Mapping[str, np.ndarray], what: str) -> None:
    """Refuse an element order MPAS IO cannot read back onto the same elements.

    MPAS-JEDI reads and writes every decomposed field at file offset ID-1 (MPAS-A v8.4.2
    src/framework/add_field_indices.inc: the IO indices are indexTo{Cell,Edge}ID(1:n*Solve)).  A file
    whose element k does not carry ID k+1 is read by MPAS-JEDI with its fields on other elements, so
    every observation would be compared with another column.
    """

    for name in ("indexToCellID", "indexToEdgeID"):
        ids = np.asarray(identity[name])
        if not np.array_equal(ids, np.arange(1, ids.size + 1)):
            raise ContractRefusal(
                f"{what}: {name} is not 1..{ids.size} in file order.  MPAS-JEDI places element ID g at "
                "file offset g-1 (MPAS-A add_field_indices.inc), so it would read this file's fields on "
                "other cells and edges; run the DA cycle on the mesh in MPAS global order"
            )


def read_xtimes(path: Path) -> list[str]:
    """Every record's xtime through the Rust reader, canonical YYYY-MM-DD_hh:mm:ss."""

    dataset = _rust_open(path)
    if "xtime" not in dataset.variables:
        raise ContractRefusal(f"{path}: no xtime, so no record can be matched to the restart's valid time")
    variable = dataset.variables["xtime"]
    if tuple(variable.dimensions[:1]) != ("Time",) or len(variable.dimensions) != 2:
        raise ContractRefusal(f"{path}: xtime must be xtime(Time, StrLen), got {tuple(variable.dimensions)}")
    try:
        rows = np.asarray(variable[:])
    except Exception as error:  # an engine whose rw_netcdf predates character reads
        raise ContractRefusal(
            f"{path}: the Rust NetCDF reader cannot read xtime ({error}); the restart's valid time "
            "cannot be checked without it"
        ) from error
    times: list[str] = []
    for row in rows:
        text = row.tobytes().decode("ascii", "replace").rstrip("\x00 ")
        try:
            times.append(mpas_xtime(datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")))
        except ValueError as error:
            raise ContractRefusal(f"{path}: xtime {text!r} is not YYYY-MM-DD_hh:mm:ss") from error
    return times


def analysis_record_for_restart(
    analysis_path: Path,
    *,
    init_path: Path,
    expected_valid_time: str,
    n_cells: int,
    n_edges: int,
) -> int:
    """OU1: the record of ``analysis_path`` a DA-cycling restart consumes, after proving it.

    Exactly one record must carry the restart's valid time (MPAS appends a record when it writes
    another time into an existing file, so record 0 can be another cycle's state), and the file's
    element identity must equal the init's, element for element.  Refused before anything is written.
    """

    try:
        expected = mpas_xtime(datetime.strptime(str(expected_valid_time), "%Y-%m-%d_%H:%M:%S"))
    except ValueError as error:
        raise ContractRefusal(f"restart valid time {expected_valid_time!r} is not YYYY-MM-DD_hh:mm:ss") from error
    times = read_xtimes(analysis_path)
    matching = [index for index, value in enumerate(times) if value == expected]
    if len(matching) != 1:
        raise ContractRefusal(
            f"{analysis_path}: expected exactly one record at the restart's valid time {expected}, found "
            f"{len(matching)} (records: {times}); a different time would restart this run from another "
            "cycle's state"
        )
    analysis = _rust_open(analysis_path)
    for dim, size in (("nCells", n_cells), ("nEdges", n_edges)):
        if dim not in analysis.dimensions or len(analysis.dimensions[dim]) != int(size):
            raise ContractRefusal(f"{analysis_path}: {dim} is not this run's {size}")
    mine = read_identity(analysis_path, time_index=matching[0])
    theirs = read_identity(init_path)
    for name in IDENTITY_FIELDS:
        a, b = mine[name], theirs[name]
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            raise ContractRefusal(
                f"{analysis_path}: {name} differs from the init's ({init_path}); the analysis is on another "
                "mesh or another element order and would be recoupled onto the wrong cells and edges"
            )
    return matching[0]


def write_background(
    path: Path,
    *,
    fields: Mapping[str, np.ndarray],
    provenance: Mapping[str, str],
    valid: datetime,
    n_cells: int,
    n_edges: int,
    n_levels: int,
    soil_levels: int,
    scalar_names: Sequence[str],
    global_attributes: Mapping[str, str] | None = None,
    n_vertices: int | None = None,
    da_state_dimensions: Mapping[str, int] | None = None,
    identity_path: Path | None = None,
) -> BackgroundRecord:
    """Write ``mpasout.nc`` in native MPAS form: CDF-5, MPAS names and dims.

    Written to a temporary name in the same folder and renamed into place,
    then a ``<file>.receipt.json`` sidecar records the per-field SHA-256 and
    provenance, so a half-written file is never visible under its real name.

    ``identity_path`` (the run's init; the driver always passes it) adds the
    init's element identity (cell/edge IDs and coordinates, stored types) so the
    analysis written back into this file can be proven to be on this run's
    elements at restart (OU1).  A file without it is refused by the restart.
    """

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    identity: dict[str, np.ndarray] = {}
    if identity_path is not None:
        identity = read_identity(Path(identity_path))
        for name, dim in IDENTITY_FIELDS.items():
            size = n_cells if dim == "nCells" else n_edges
            if identity[name].shape != (size,):
                raise ContractRefusal(
                    f"{identity_path}: {name} has {identity[name].shape[0]} values, this run has {size} {dim}"
                )
        require_mpas_global_order(identity, f"{identity_path}")
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(handle)
    os.unlink(temporary)
    field_hashes: dict[str, str] = {}
    try:
        schema = _rust_cdf5_schema()
        time = schema.def_dim("Time", 0, unlimited=True)
        strlen = schema.def_dim("StrLen", MPAS_STRLEN)
        dims_by_name = {
            "Time": time,
            "StrLen": strlen,
            "nCells": schema.def_dim("nCells", n_cells),
            "nEdges": schema.def_dim("nEdges", n_edges),
            "nVertLevels": schema.def_dim("nVertLevels", n_levels),
            "nVertLevelsP1": schema.def_dim("nVertLevelsP1", n_levels + 1),
            "nSoilLevels": schema.def_dim("nSoilLevels", soil_levels),
        }
        if n_vertices is not None:
            dims_by_name["nVertices"] = schema.def_dim("nVertices", int(n_vertices))
        for name, size in dict(DA_STATE_DIMENSIONS if da_state_dimensions is None else da_state_dimensions).items():
            dims_by_name[name] = schema.def_dim(name, int(size))
        # The caller's attributes REPLACE a default of the same name, so every name is written once.  THE
        # BREAKAGE THIS PREVENTS: the MPAS-physics route names its own ``source``, the writer refused the
        # repeated name ("duplicate global attribute name 'source'") and that route could never write a JEDI
        # background.  The ArWen route passes none: unchanged.
        attributes = {
            "model_name": "mpas",
            "core_name": "atmosphere",
            "source": "woof hex: MPAS-A v8.4.2 CUDA dycore, hex's own physics",
            "hex_contract_schema": SCHEMA,
            "hex_stream": "background (rrfs-workflow 85ea55c0 hrrrv5 stream list, 27 fields)",
            "hex_soil_mapping": soil_mapping_label(provenance),
            "config_start_time": mpas_xtime(valid),
        }
        attributes.update({key: str(value) for key, value in (global_attributes or {}).items()})
        for key, value in attributes.items():
            schema.put_global_attr(key, value)
        xtime = schema.def_var("xtime", "S1", (time, strlen))
        fixed: list[tuple[int, np.ndarray]] = []
        for name, values in identity.items():
            varid = schema.def_var(name, values.dtype, (dims_by_name[IDENTITY_FIELDS[name]],))
            schema.put_var_attr(varid, "hex_provenance", "identity: copied from the run's init (OU1)")
            fixed.append((varid, values))
            field_hashes[name] = hashlib.sha256(values.tobytes()).hexdigest()
        ordered = [name for name in BACKGROUND_FIELDS if name != "scalars"]
        species = list(scalar_names) + [
            name for name in RRFS_EXTRA_SCALARS if name in fields and name not in scalar_names
        ]
        ordered[ordered.index("rho") + 1:ordered.index("rho") + 1] = species
        records: list[tuple[int, np.ndarray]] = []
        for name in ordered:
            # Adding +0.0 turns a negative zero into +0.0 and leaves every
            # other float32 bit for bit.  JEDI's write-back of an
            # unchanged field (background + zero increment) yields +0.0,
            # so a -0.0 here (hex's closed limited-area boundary edges
            # carry u = -0.0) would fail the no-observation round trip
            # bitwise while equal in value.  MEASURED 2026-10-05:
            # 89,624 u values on 2,758 edges, all -0.0 -> +0.0.
            array = np.asarray(fields[name], dtype=np.float32) + np.float32(0.0)
            dims = _dims(name, array.shape, n_cells, n_edges, n_levels, soil_levels)
            varid = schema.def_var(name, "f4", tuple(dims_by_name[dim] for dim in dims))
            units, long_name = _UNITS.get(
                name, (_SCALAR_UNITS.get(name, "kg kg^{-1}"), _SCALAR_LONG.get(name, name))
            )
            schema.put_var_attr(varid, "units", units)
            schema.put_var_attr(varid, "long_name", long_name)
            schema.put_var_attr(varid, "hex_provenance", provenance[name])
            # One record in MPAS order: nCells or nEdges leading, levels fastest.
            payload = array if array.ndim == 1 else array.T
            records.append((varid, np.ascontiguousarray(payload)))
            field_hashes[name] = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        with schema.create(temporary) as writer:
            for varid, values in fixed:
                writer.write_var(varid, values)
            writer.write_record(0, xtime, mpas_xtime(valid).ljust(MPAS_STRLEN).encode("ascii"))
            for varid, payload in records:
                writer.write_record(0, varid, payload)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    digest = _sha256_file(path)
    receipt = {
        "schema": SCHEMA,
        "file": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "format": "CDF-5 (NETCDF3_64BIT_DATA)",
        "writer": RUST_NETCDF_WRITER,
        "valid_time": mpas_xtime(valid),
        "dimensions": {
            "nCells": n_cells, "nEdges": n_edges, "nVertLevels": n_levels,
            "nVertLevelsP1": n_levels + 1, "nSoilLevels": soil_levels,
        },
        "stream": "background",
        "field_count_stream_list": len(BACKGROUND_FIELDS),
        "scalars": list(scalar_names),
        "fields": {
            name: {"sha256": field_hashes[name], "provenance": provenance[name]}
            for name in field_hashes if name in provenance
        },
        "identity": {
            "source": None if identity_path is None else str(identity_path),
            "fields": {name: field_hashes[name] for name in identity},
        },
        "soil_mapping": soil_mapping_label(provenance),
    }
    receipt_path = path.with_name(path.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return BackgroundRecord(str(path), path.stat().st_size, digest, str(receipt_path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_contract_fields(
    path: Path,
    names: Sequence[str],
    *,
    scalar_names: Sequence[str] = DEFAULT_SCALAR_NAMES,
    time_index: int = 0,
) -> dict[str, np.ndarray]:
    """Read named contract fields back in hex's logical layout.

    ``scalars`` expands to the species names.  A field the file lacks is
    refused by name: an analysis missing a field it promised would otherwise
    restart from the background's value without anyone knowing.
    """

    wanted: list[str] = []
    for name in names:
        wanted.extend(scalar_names if name == "scalars" else [name])
    out: dict[str, np.ndarray] = {}
    dataset = _rust_open(path)
    missing = [name for name in wanted if name not in dataset.variables]
    if missing:
        raise ContractRefusal(f"{path}: lacks contract fields {missing}")
    for name in wanted:
        variable = dataset.variables[name]
        variable.set_auto_mask(False)
        data = np.asarray(variable[time_index] if variable.dimensions[0] == "Time" else variable[:])
        if data.ndim == 2:
            data = data.T
        out[name] = np.ascontiguousarray(data, dtype=np.float32)
    return out


def compare_fields(
    before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Per-field identity and the size of any difference, in ULP and value."""

    report: dict[str, dict[str, Any]] = {}
    for name in sorted(before):
        a = np.asarray(before[name], dtype=np.float32)
        b = np.asarray(after[name], dtype=np.float32)
        if a.shape != b.shape:
            report[name] = {"shape_before": list(a.shape), "shape_after": list(b.shape), "identical": False}
            continue
        ai = a.view(np.int32).astype(np.int64)
        bi = b.view(np.int32).astype(np.int64)
        ai = np.where(ai < 0, -(ai & 0x7FFFFFFF), ai)
        bi = np.where(bi < 0, -(bi & 0x7FFFFFFF), bi)
        ulp = np.abs(ai - bi)
        diff = (b.astype(np.float64) - a.astype(np.float64))
        report[name] = {
            "identical": bool(np.array_equal(a.view(np.int32), b.view(np.int32))),
            "changed_values": int(np.count_nonzero(ulp)),
            "values": int(a.size),
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "max_abs": float(np.abs(diff).max()) if diff.size else 0.0,
            "rms": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        }
    return report


def write_uncoupled_state(
    path: Path,
    analysis: Mapping[str, np.ndarray],
    *,
    scalar_names: Sequence[str],
    n_cells: int,
    n_edges: int,
    n_levels: int,
) -> None:
    """The DA-cycling restart's inputs as a native-named file for the loader.

    Carries ONLY rho, theta, the species, u and w: the uncoupled fields
    upstream's atm_init_coupled_diagnostics reads on a DA-cycling restart.
    pressure_p is deliberately absent, so the loader recomputes it from the
    recoupled exner exactly as upstream does (F: atm_init_coupled_diagnostics,
    final loop) instead of consuming the analysis's own pressure_p.
    """

    schema = _rust_cdf5_schema()
    dims_by_name = {
        "Time": schema.def_dim("Time", 0, unlimited=True),
        "nCells": schema.def_dim("nCells", n_cells),
        "nEdges": schema.def_dim("nEdges", n_edges),
        "nVertLevels": schema.def_dim("nVertLevels", n_levels),
        "nVertLevelsP1": schema.def_dim("nVertLevelsP1", n_levels + 1),
    }
    records: list[tuple[int, np.ndarray]] = []
    for name in ("rho", "theta", *scalar_names, "u", "w"):
        array = np.asarray(analysis[name], dtype=np.float32)
        dims = _dims(name, array.shape, n_cells, n_edges, n_levels, -1)
        varid = schema.def_var(name, "f4", tuple(dims_by_name[dim] for dim in dims))
        records.append((varid, np.ascontiguousarray(array.T)))
    with schema.create(Path(path)) as writer:
        for varid, payload in records:
            writer.write_record(0, varid, payload)


def recouple_da_cycling(
    analysis_path: Path,
    *,
    init_path: Path,
    mesh: Any,
    vertical: Any,
    terrain_metrics: Any,
    expected_reference: Any,
    expected_valid_time: str,
    scalar_names: Sequence[str],
    n_cells: int,
    n_edges: int,
    n_levels: int,
    allow_regional_sentinels: bool,
    time_seconds: float,
    work_dir: Path,
    reference_formed: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    """Upstream ``config_do_DAcycling``: recompute the coupled state.

    ``reference_formed`` forms the base state as atm_init_coupled_diagnostics does on a DA-cycling start
    (rtheta_base, exner_base and pressure_base from rho_base, theta_base and the model's zz), the base state a run
    whose own reference was formed that way holds (the MPAS suite route's loaders, cold_start); False reads the
    init's stored words, as the WOOF route's run reference does.  THE BREAKAGE THIS FIXES (fix/hex-lam-da-v842):
    every MPAS-suite --da-analysis on an init whose stored pressure_base is the init program's (the RRFS hrrrv5 ne3km
    init) was refused, "base state pressure_base differs from the run's own reference", though both name one state.

    Returns ``(PrognosticState, DrySavedDiagnostics, receipt)`` built by
    hex's own transcription of ``atm_init_coupled_diagnostics`` from the
    analysis's rho, theta, species, u and w.  The base state is re-read from
    the init, as upstream reads it from the restart, and must equal the run's
    own reference bit for bit or the restart is refused.

    OU1: the analysis must hold exactly one record at ``expected_valid_time``
    (the checkpoint's valid time) and the init's element identity; that record
    is the one read (:func:`analysis_record_for_restart`).
    """

    from woof.hex.driver import load_mpas_initial_state

    record = analysis_record_for_restart(
        analysis_path, init_path=init_path, expected_valid_time=expected_valid_time,
        n_cells=n_cells, n_edges=n_edges,
    )
    analysis = read_contract_fields(
        analysis_path, RECOUPLED_INPUTS, scalar_names=scalar_names, time_index=record,
    )
    for name, array in analysis.items():
        if not np.all(np.isfinite(array)):
            raise ContractRefusal(f"analysis field {name} carries non-finite values")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    temporary = work_dir / "da-cycling-uncoupled.nc"
    if temporary.exists():
        temporary.unlink()
    write_uncoupled_state(
        temporary, analysis, scalar_names=scalar_names,
        n_cells=n_cells, n_edges=n_edges, n_levels=n_levels,
    )
    try:
        state, reference, saved = load_mpas_initial_state(
            temporary,
            mesh,
            vertical,
            scalar_names=tuple(scalar_names),
            reference_path=init_path,
            terrain_metrics=terrain_metrics,
            allow_regional_sentinels=allow_regional_sentinels,
            return_saved_diagnostics=True,
            # The uncoupled file holds no rho_zz/theta_m/ru/rw, so cold_start changes only the base-state reads.
            cold_start=bool(reference_formed),
        )
    finally:
        temporary.unlink(missing_ok=True)
    for name in ("rho_base", "rho_theta_base", "pressure_base", "exner_base"):
        mine = np.asarray(getattr(reference, name))
        theirs = np.asarray(getattr(expected_reference, name))
        if mine.shape != theirs.shape or not np.array_equal(mine.view(np.uint32), np.asarray(theirs, dtype=mine.dtype).view(np.uint32)):
            raise ContractRefusal(
                f"DA-cycling base state {name} differs from the run's own reference; "
                "restarting on a different base state would move the perturbation "
                "equations under the analysis"
            )
    state.time_seconds = float(time_seconds)
    receipt = {
        "upstream": "MPAS-A v8.4.1 mpas_atm_core.F:522-530 config_do_DAcycling -> atm_init_coupled_diagnostics",
        "hex_routine": "woof.hex.driver.load_mpas_initial_state (public rho/theta/qv/u/w branch)",
        "analysis": str(analysis_path),
        "analysis_sha256": _sha256_file(Path(analysis_path)),
        "analysis_record": int(record),
        "analysis_valid_time": str(expected_valid_time),
        "analysis_identity_equals_init": True,
        "consumed": list(RECOUPLED_INPUTS),
        "recomputed": ["theta_m", "rho_zz", "ru", "rw", "rho_p", "rtheta_p", "exner", "pressure_p"],
        "diagnostics_recomputed_by_model": ["surface_pressure", "uReconstructZonal", "uReconstructMeridional", "t2m", "q2", "refl10cm"],
        "reference_identical_to_run": True,
        "time_seconds": float(time_seconds),
    }
    return state, saved, receipt

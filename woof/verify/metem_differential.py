"""The met_em/wrfinput differential oracle: WOOF-ArWen's own real-data ingest
scored, field by field, against the ``real.exe`` that read the same met_em.

WHY THIS EXISTS.  Until the wrfinput door landed, WOOF-ArWen had exactly one way
to turn weather into an initial state (``rw_wps``: GRIB straight to internal
state), so :mod:`woof.ingest.real` -- 3400 lines of WRF-real transcription --
had nothing independent to be checked against.  A matched
``met_em`` + ``wrfinput_d01`` pair closes that: ``real.exe`` produced the
wrfinput FROM the met_em, and :func:`woof.ingest.real.initialize_real` claims
to do the same job from the same input.  This module runs WOOF-ArWen on the met_em
and diffs the result against the wrfinput, per field.

This initializer comparison reads the actual ``ZNW`` from wrfinput and
passes it through ``make_vertical_coord(eta_levels=...)``. Every 3-D
difference is therefore measured at identical eta, independent of how the
producing namelist selected its grid. The automatic generator has a separate
untouched-Fortran oracle in ``tests/test_wrf_eta.py``. The optional
:func:`eta_ladder_comparison` remains a comparison of uniform/tanh alternatives
with the supplied grid; those alternatives are not the met_em door's default.

The comparison is CPU-only: it reads, ingests and diffs.  No forecast is run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.grid import make_vertical_coord, finalize_vertical_coord
from woof.ingest.horiz import HorizontalSnapshot
from woof.ingest.real import initialize_real


#: metgrid stacks the surface analysis as level ONE of every 3-D field and
#: the isobaric ladder above it (module_initialize_real.F:1047 selects the
#: column order by comparing level 2 against level num_metgrid_levels, which
#: is only meaningful if level 1 is not part of the ladder).  ArWen's
#: HorizontalSnapshot carries the ladder and the surface separately, so the
#: split happens here and is asserted, not assumed.
SURFACE_LEVEL_INDEX = 0


@dataclass(frozen=True)
class MetEmCase:
    """One met_em frame split into the inventory initialize_real consumes."""

    valid_time: datetime
    levels_hpa: np.ndarray
    fields: dict
    terrain: np.ndarray            # HGT_M: the target WRF terrain
    source_orography: np.ndarray   # SOILHGT: the terrain PSFC belongs to
    lane: str                      # "specific-humidity" or "relative-humidity"
    nx: int
    ny: int
    # LANDMASK: the target grid's land/water, which real.exe reads as
    # XLAND to size cold-start cloud droplets (make_DropletNumber).
    landmask: np.ndarray | None = None


def _var(ds, name, dtype=np.float32):
    return np.asarray(ds.variables[name][0], dtype=dtype)


def read_metem(path) -> MetEmCase:
    """Split one met_em frame into WOOF-ArWen's real-data forcing inventory.

    Selects the lane real.exe would select: FLAG_SH=1 means metgrid supplied
    SPECHUMD, and real.exe converts it to qv and then re-diagnoses RH from it
    (module_initialize_real.F:1138-1167) rather than using a supplied RH.
    """
    import netCDF4

    ds = netCDF4.Dataset(str(path))
    try:
        ny = len(ds.dimensions["south_north"])
        nx = len(ds.dimensions["west_east"])
        pres = _var(ds, "PRES", np.float64)
        psfc = _var(ds, "PSFC", np.float64)
        # The surface-level convention is CHECKED, not assumed: metgrid
        # writes PSFC into level one of PRES and SOILHGT into level one of
        # GHT.  A file that violates it is not one this split can read.
        if not np.array_equal(pres[SURFACE_LEVEL_INDEX], psfc):
            raise ValueError(
                f"{path}: PRES level {SURFACE_LEVEL_INDEX} is not PSFC, so "
                "this file does not use metgrid's surface-in-level-one "
                "convention and cannot be split into a ladder plus a "
                "surface analysis")
        soilhgt = _var(ds, "SOILHGT", np.float64)
        ght = _var(ds, "GHT", np.float64)
        if not np.array_equal(ght[SURFACE_LEVEL_INDEX], soilhgt):
            raise ValueError(
                f"{path}: GHT level {SURFACE_LEVEL_INDEX} is not SOILHGT")
        flag_sh = int(ds.getncattr("FLAG_SH")) if "FLAG_SH" in ds.ncattrs() else 0
        sl = slice(SURFACE_LEVEL_INDEX + 1, None)
        fields = {
            "TT": _var(ds, "TT")[sl],
            "GHT": _var(ds, "GHT")[sl],
            "UU": _var(ds, "UU")[sl],
            "VV": _var(ds, "VV")[sl],
            "PSFC": _var(ds, "PSFC"),
            "T2": _var(ds, "TT")[SURFACE_LEVEL_INDEX],
            "U10": _var(ds, "UU")[SURFACE_LEVEL_INDEX],
            "V10": _var(ds, "VV")[SURFACE_LEVEL_INDEX],
        }
        if flag_sh == 1:
            lane = "specific-humidity"
            fields["PRES"] = _var(ds, "PRES")[sl]
            fields["SPFH"] = _var(ds, "SPECHUMD")[sl]
            fields["Q2"] = _var(ds, "SPECHUMD")[SURFACE_LEVEL_INDEX]
            for name in ("QC", "QR", "QI", "QS", "QG"):
                if name in ds.variables:
                    fields[name] = _var(ds, name)[sl]
        elif np.any(np.ptp(pres[sl].reshape(pres[sl].shape[0], -1), axis=1) > 1.0):
            # RH on levels that are not isobaric (a model-level source).
            # Reading it on the flat ladder below would flatten its 3-D
            # pressure.  real.exe forms qv from RH, TT and the 3-D p_gc
            # with rh_to_mxrat1 (module_initialize_real.F:1113-1142); that
            # qv is carried here as specific humidity on the PRES lane.
            # Transcribed from WRF, not imported from woof.ingest, so the
            # oracle stays independent of the reader it scores.
            lane = "relative-humidity-3d-pressure"
            tt = _var(ds, "TT", np.float64)
            rh = np.clip(_var(ds, "RH", np.float64), 0.0, 100.0)
            es = 0.01 * rh * 6.112 * np.exp(17.67 * (tt - 273.15) / (tt - 29.65))
            qv = np.where(es < pres / 100.0,
                          np.maximum(0.622 * es / (pres / 100.0 - es), 1.0e-6),
                          1.0e-6)
            qv = np.where((pres < 10000.0) & (qv > 1.0e-5), 3.0e-6, qv)
            specific = (qv / (1.0 + qv)).astype(np.float32)
            fields["PRES"] = _var(ds, "PRES")[sl]
            fields["SPFH"] = specific[sl]
            fields["Q2"] = specific[SURFACE_LEVEL_INDEX]
            for name in ("QC", "QR", "QI", "QS", "QG"):
                if name in ds.variables:
                    fields[name] = _var(ds, name)[sl]
        else:
            lane = "relative-humidity"
            fields["RH"] = _var(ds, "RH")[sl]
            fields["RH2"] = _var(ds, "RH")[SURFACE_LEVEL_INDEX]
        times = ds.variables["Times"][0]
        stamp = b"".join(np.asarray(times).astype("S1").ravel()).decode()
        # metgrid with nocolons = .true. writes YYYY-MM-DD_HH_MM_SS; real.exe
        # reads both.  Parsed here, not through the door's reader, so the
        # oracle stays independent of the code it scores.
        stamp = stamp.strip()
        valid = datetime.strptime(
            stamp, "%Y-%m-%d_%H_%M_%S" if stamp[13:14] == "_" else "%Y-%m-%d_%H:%M:%S")
        return MetEmCase(
            valid_time=valid,
            levels_hpa=(pres[sl, 0, 0] / 100.0).astype(np.float64),
            fields=fields,
            terrain=_var(ds, "HGT_M", np.float64),
            source_orography=soilhgt,
            lane=lane, nx=nx, ny=ny,
            landmask=(_var(ds, "LANDMASK") if "LANDMASK" in ds.variables
                      else None))
    finally:
        ds.close()


def read_wrfinput_grid(path):
    """real.exe's own vertical grid and the scalars the base state was keyed on."""
    import netCDF4

    ds = netCDF4.Dataset(str(path))
    try:
        get = lambda n: np.asarray(ds.variables[n][0], dtype=np.float64)
        scalar = lambda n: float(np.asarray(ds.variables[n][:]).ravel()[0])
        attr = lambda n: ds.getncattr(n)
        return {
            "znw": get("ZNW"), "znu": get("ZNU"),
            "p_top": scalar("P_TOP"), "t00": scalar("T00"),
            "tlp": scalar("TLP"), "tiso": scalar("TISO"),
            "p_strat": scalar("P_STRAT"), "p00": scalar("P00"),
            "hybrid_opt": int(attr("HYBRID_OPT")),
            "etac": float(attr("ETAC")),
            "hypsometric_opt": int(attr("HYPSOMETRIC_OPT")),
            "mp_physics": int(attr("MP_PHYSICS")),
            "dx": float(attr("DX")), "dy": float(attr("DY")),
            "title": str(attr("TITLE")).strip(),
        }
    finally:
        ds.close()


def eta_ladder_comparison(znw_real, *, stretch_grid=None):
    """Compare uniform/tanh alternatives with the supplied real.exe grid.

    This diagnostic does not exercise WRF automatic generation. A large
    difference says those alternate grids cannot be compared level by level
    with this input, not that WOOF-ArWen cannot generate the requested WRF grid.
    """
    znw_real = np.asarray(znw_real, dtype=np.float64)
    nz = znw_real.size - 1
    uniform = make_vertical_coord(nz)
    best_stretch, best_err = None, np.inf
    for stretch in (np.arange(0.05, 4.001, 0.01) if stretch_grid is None
                    else np.asarray(stretch_grid, dtype=np.float64)):
        err = float(np.abs(make_vertical_coord(nz, stretch=float(stretch)).znw
                           - znw_real).max())
        if err < best_err:
            best_stretch, best_err = float(stretch), err
    thickness = -np.diff(znw_real)
    return {
        "nz": nz,
        "uniform_max_abs_deta": float(np.abs(uniform.znw - znw_real).max()),
        "best_tanh_stretch": best_stretch,
        "best_tanh_max_abs_deta": best_err,
        "real_thickness_ratio": float(thickness.max() / thickness.min()),
        "uniform_thickness_ratio": 1.0,
    }


def field_stats(name, arwen, wrf, unit=""):
    """Max abs, RMS, max relative and max-abs-over-range for one field."""
    a = np.asarray(arwen, dtype=np.float64)
    b = np.asarray(wrf, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"{name}: shapes differ {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    idx = np.unravel_index(int(np.argmax(d)), d.shape)
    span = float(b.max() - b.min())
    floor = max(1.0e-30, 0.01 * float(np.abs(b).max()))
    mask = np.abs(b) > floor
    return {
        "field": name, "unit": unit, "shape": tuple(int(s) for s in a.shape),
        "max_abs": float(d.max()),
        "rms": float(np.sqrt(((a - b) ** 2).mean())),
        "max_rel": (float((d[mask] / np.abs(b[mask])).max())
                    if mask.any() else float("nan")),
        "ref_range": span,
        "max_abs_over_range": float(d.max() / span) if span > 0 else float("nan"),
        "argmax": tuple(int(v) for v in idx),
        "arwen_at_argmax": float(a[idx]), "wrf_at_argmax": float(b[idx]),
    }


def run_differential(metem_path, wrfinput_path, *, mp_physics=None,
                     base_temp=290.0, preprocess_backend="cpu",
                     column_workers=4, preprocess_workers=4):
    """Ingest the met_em with WOOF-ArWen and diff every comparable field.

    ``mp_physics=None`` takes the scheme from the wrfinput.  An explicit
    value SUBSTITUTES a scheme, which is legitimate only for the
    scheme-independent half of the comparison (mass, base state,
    thermodynamics, winds) and is recorded in the result so a reader
    cannot mistake a substituted run for a faithful one.
    """
    import netCDF4

    case = read_metem(metem_path)
    grid = read_wrfinput_grid(wrfinput_path)
    nz = grid["znw"].size - 1
    substituted = mp_physics is not None and int(mp_physics) != grid["mp_physics"]
    mp = int(grid["mp_physics"] if mp_physics is None else mp_physics)

    coord = make_vertical_coord(nz, eta_levels=grid["znw"],
                                hybrid_opt=grid["hybrid_opt"],
                                etac=grid["etac"])
    finalize_vertical_coord(coord, grid["p_top"])
    cfg = RunConfig(nx=case.nx, ny=case.ny, nz=nz, dx=grid["dx"], dy=grid["dy"],
                    ztop=20000.0, dt=12.0, run_seconds=300.0,
                    moist=True, mp_physics=mp, base_temp=base_temp,
                    hybrid_opt=grid["hybrid_opt"], etac=grid["etac"],
                    hypsometric_opt=grid["hypsometric_opt"], terrain_opt=1)
    snapshot = HorizontalSnapshot(valid_time=case.valid_time,
                                  levels_hpa=case.levels_hpa,
                                  fields=case.fields)
    result = initialize_real(
        snapshot, cfg, coord, case.terrain, landmask=case.landmask,
        source_orography=case.source_orography, p_top=grid["p_top"],
        sfcp_to_sfcp=True, use_sh_qv=False, column_workers=column_workers,
        # real.exe makes its FLAG_SH surface test only on a FLAG_SH file.
        flag_sh_surface_fallback=(
            False if case.lane == "relative-humidity-3d-pressure" else None),
        preprocess_backend=preprocess_backend,
        preprocess_workers=preprocess_workers, state_backend="preprocess")

    ds = netCDF4.Dataset(str(wrfinput_path))
    try:
        w = lambda n: np.asarray(ds.variables[n][0], dtype=np.float64)
        state, base = result.state, result.base
        host = lambda a: np.asarray(a, dtype=np.float64)
        theta = host(state.thb) + host(state.thp)
        rows = [
            field_stats("HGT (terrain)", case.terrain, w("HGT"), "m"),
            field_stats("PSFC (surface pressure)",
                        result.surface_pressure, w("PSFC"), "Pa"),
            field_stats("MUB (base dry mass)", base.mub, w("MUB"), "Pa"),
            field_stats("MU+MUB (dry column mass)",
                        result.dry_mass, w("MU") + w("MUB"), "Pa"),
            field_stats("MU (mu perturbation)", host(state.mup), w("MU"), "Pa"),
            field_stats("PB (base pressure)", base.pb, w("PB"), "Pa"),
            field_stats("ALB (base specific volume)", base.alb, w("ALB"), "m3/kg"),
            field_stats("T_INIT (base theta-300)",
                        host(base.thb) - 300.0, w("T_INIT"), "K"),
            field_stats("PHB (base geopotential)", base.phb, w("PHB"), "m2/s2"),
            field_stats("P+PB (total pressure)",
                        result.total_pressure, w("P") + w("PB"), "Pa"),
            field_stats("T (theta-300, dry)", theta - 300.0, w("T"), "K"),
            field_stats("THM (moist theta-300)",
                        theta * (1.0 + (c.RV / c.RD) * host(state.qv)) - 300.0,
                        w("THM"), "K"),
            field_stats("QVAPOR", host(state.qv), w("QVAPOR"), "kg/kg"),
            field_stats("U", host(state.u), w("U"), "m/s"),
            field_stats("V", host(state.v), w("V"), "m/s"),
            field_stats("PH (perturbation geopotential)",
                        host(state.php), w("PH"), "m2/s2"),
            field_stats("PH+PHB (total geopotential)",
                        host(state.php) + base.phb, w("PH") + w("PHB"), "m2/s2"),
        ]
        if not substituted:
            for attr, name in (("qc", "QCLOUD"), ("qr", "QRAIN"),
                               ("qi", "QICE"), ("qs", "QSNOW"),
                               ("qg", "QGRAUP")):
                value = getattr(state, attr, None)
                if value is not None and name in ds.variables:
                    rows.append(field_stats(name, host(value), w(name), "kg/kg"))
        return {
            "metem": str(metem_path), "wrfinput": str(wrfinput_path),
            "wrfinput_title": grid["title"], "lane": case.lane,
            "nx": case.nx, "ny": case.ny, "nz": nz,
            "nsource": int(case.levels_hpa.size),
            "mp_physics_file": grid["mp_physics"], "mp_physics_used": mp,
            "mp_physics_substituted": bool(substituted),
            "eta": eta_ladder_comparison(grid["znw"]),
            "rows": rows, "result": result, "coord": coord,
        }
    finally:
        ds.close()


def format_table(report) -> str:
    """One fixed-width row per field, most-divergent last."""
    lines = [f"{'field':28s} {'unit':7s} {'max_abs':>12s} {'rms':>12s} "
             f"{'max_rel':>10s} {'maxabs/range':>13s}  argmax"]
    for row in report["rows"]:
        lines.append(
            f"{row['field']:28s} {row['unit']:7s} {row['max_abs']:12.6g} "
            f"{row['rms']:12.6g} {row['max_rel']:10.3e} "
            f"{row['max_abs_over_range']:13.3e}  {row['argmax']}")
    return "\n".join(lines)

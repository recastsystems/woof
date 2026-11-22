"""Mass-preserving condensate balance for completed cold initial conditions.

The producer's vapor-only hydrostatic solve precedes hydrometeor completion.
Keep its dry density and all water, integrate the MPAS loaded vertical relation,
and invert the cold-reader EOS for theta. This is a column balance, not a
horizontal balance or a forecast adjustment. Arrays use (cells, levels).

Authority (BSD-3-Clause; see licenses/LICENSE-MPAS-BSD-3-Clause.txt):
MPAS-Model v8.4.2, f34984b2c89353cf31c235b08d3b4acd96b37bb3,
src/core_init_atmosphere/mpas_init_atm_cases.F:5316-5319 (hydrostatic stencil),
src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3238-3256,6467,6784-6786
(total moist mass and its vertical force). The stock case-7 init reads only qv;
we extend its stencil to the same total water the forecast uses. The fixed-mass,
top-anchored completion and EOS inversion here are our own implementation.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

G = 9.80616
RD = 287.0
RV = 461.6
CP = 1004.5
P0 = 100000.0
CONDENSATE = ("qc", "qr", "qi", "qs", "qg")


def cold_pressure(rho, theta, qv):
    """Physical dry density and dry theta to the cold-reader pressure (Pa)."""
    return P0 * (RD * rho * theta * (1.0 + RV / RD * qv) / P0) ** (CP / (CP - RD))


def balance_columns(*, rho, theta, qv, condensate, zz, rho_base, theta_base,
                    fzm, fzp, rdzu, rdzw0):
    """Return pressure, theta and surface pressure; never change mass inputs.

    Only columns containing condensate are solved. Clear columns' theta is
    bit-identical, and the file completion does not write those columns at all.
    Pressure at the top mass level is held at its original cold-reader value.
    Double precision is used even for a single precision output file.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in
              (rho, theta, qv, condensate, zz, rho_base, theta_base)]
    rho, theta, qv, condensate, zz, rho_base, theta_base = arrays
    if rho.ndim != 2 or rho.shape[1] < 2 or any(a.shape != rho.shape for a in arrays):
        raise ValueError("balance requires matching (cells, levels) arrays and at least two levels")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("nonfinite cold-start thermodynamic or water field")
    if any(np.any(a <= 0) for a in (rho, theta, zz, rho_base, theta_base)):
        raise ValueError("cold-start density, theta and metric must be positive")
    if np.any(condensate < 0):
        raise ValueError("cold-start condensate mixing ratios must be nonnegative")
    # Vapour is taken as the producer wrote it and as the forecast will read it: neither clipped nor written.
    # THE BREAKAGE THIS PREVENTS (GFS 2024-05-21 12Z, hrrrv5, 3 km storm box, 2026-11-13): the RRFS producer's
    # interpolation left 16 points of qv down to -6.9e-6 kg/kg, and a refusal of any negative qv stopped every
    # default GFS-started hrrrv5 init.  Only a vapour that breaks the moist equation of state is refused.
    if np.any(1.0 + RV / RD * qv <= 0) or np.any(1.0 + qv + condensate <= 0):
        raise ValueError("cold-start vapour too negative for the moist equation of state")
    fzm, fzp, rdzu = (np.asarray(a, dtype=np.float64) for a in (fzm, fzp, rdzu))
    if any(a.shape != (rho.shape[1],) or not np.all(np.isfinite(a)) for a in (fzm, fzp, rdzu)):
        raise ValueError("balance requires one finite stencil weight per level")
    if np.any(rdzu[1:] <= 0) or not np.isfinite(rdzw0) or rdzw0 <= 0:
        raise ValueError("balance requires positive inverse layer spacings")

    active = np.any(condensate > 0, axis=1)
    p = cold_pressure(rho, theta, qv)
    result_theta = theta.copy()
    r = rho / zz
    qt = qv + condensate
    if np.any(active):
        pb = cold_pressure(rho_base[active] * zz[active], theta_base[active], 0.0)
        # rb is already metric coupled. qt includes mass species, never numbers.
        b = -G * (r[active] - rho_base[active] + r[active] * qt[active])
        pp = p[active] - pb
        for k in range(rho.shape[1] - 1, 0, -1):
            pp[:, k - 1] = pp[:, k] - (fzm[k] * b[:, k] + fzp[k] * b[:, k - 1]) / rdzu[k]
        loaded = pp + pb
        if not np.all(np.isfinite(loaded)) or np.any(loaded <= 0):
            raise ValueError("loaded hydrostatic balance produced invalid pressure")
        p[active] = loaded
        result_theta[active] = (P0 / (RD * rho[active] * (1.0 + RV / RD * qv[active]))
                                * (loaded / P0) ** ((CP - RD) / CP))
    surface = p[:, 0] + 0.5 * G / rdzw0 * (1.25 * r[:, 0] * (1 + qt[:, 0])
                                         - 0.25 * r[:, 1] * (1 + qt[:, 1]))
    return p, result_theta, surface


def balance_choice(requested: str | None) -> tuple[bool, str]:
    """``--condensate-balance`` as chosen: (enabled, chosen_by).

    Default yes under fast numerics: a start that carries the first guess's clouds and precipitation gets their
    weight in its hydrostatic balance, whatever model the first guess came from.  Default no under strict numerics,
    the proof mode, where the init must be the producer's own (MPAS-A case 7, or NOAA's RRFS fork, both of which
    balance on vapour alone).  THE BREAKAGE THIS PREVENTS: a strict-numerics GFS start that carries condensate was
    moved by up to ~0.5 K theta away from NOAA's RRFS init, so a NOAA-match proof would fail on the start alone.
    An explicit yes or no always wins.
    """
    if requested is not None:
        return requested == "yes", "--condensate-balance"
    from . import numerics

    if numerics.is_strict():
        return False, "strict numerics (the producer's own vapour-only balance)"
    return True, "default"


def complete_init(path: str | Path, *, enabled: bool = True, block_size: int = 1024) -> dict:
    """Complete a newly produced init, after all mass species have been written.

    Both init producers call this before recording their final output digest.
    Disabled means no file open or write. No density, water, number, wind or base
    field is written. Process bounded blocks; no full-state copy is necessary.
    """
    receipt = {"enabled": enabled, "method": "fixed-mass-loaded-column-v1", "columns": 0,
               "max_delta_theta_K": 0.0, "max_delta_pressure_Pa": 0.0}
    if not enabled:
        return receipt
    if block_size < 1:
        raise ValueError("block_size must be positive")
    from netCDF4 import Dataset

    started = time.monotonic()
    with Dataset(path, "r+") as ds:
        ds.set_auto_maskandscale(False)
        species = [n for n in CONDENSATE if n in ds.variables]
        receipt["species"] = species
        if not species:
            receipt["seconds"] = time.monotonic() - started
            return receipt

        def index(var, cells):
            if any(dim not in ("Time", "nCells", "nVertLevels") for dim in var.dimensions):
                raise ValueError(f"unsupported init dimensions for {var.name}: {var.dimensions}")
            return tuple(0 if dim == "Time" else cells if dim == "nCells" else slice(None)
                         for dim in var.dimensions)

        def read(name, cells):
            var = ds[name]
            return np.asarray(var[index(var, cells)], dtype=np.float64)

        for lo in range(0, len(ds.dimensions["nCells"]), block_size):
            sl = slice(lo, lo + block_size)
            masses = [read(n, sl) for n in species]
            if any(not np.all(np.isfinite(a)) or np.any(a < 0) for a in masses):
                raise ValueError("invalid condensate in cold init")
            condensate = sum(masses)
            local = np.flatnonzero(np.any(condensate > 0, axis=1))
            if not len(local):
                continue
            cells = lo + local
            d = {n: read(n, cells) for n in
                 ("rho", "theta", "qv", "zz", "rho_base", "theta_base")}
            d.update(condensate=condensate[local],
                     **{n: np.asarray(ds[n][:], dtype=np.float64) for n in ("fzm", "fzp", "rdzu")},
                     rdzw0=float(ds["rdzw"][0]))
            p, theta, surface = balance_columns(**d)
            oldp = cold_pressure(d["rho"], d["theta"], d["qv"])
            receipt["columns"] += len(cells)
            receipt["max_delta_theta_K"] = max(receipt["max_delta_theta_K"], float(np.max(abs(theta - d["theta"]))))
            receipt["max_delta_pressure_Pa"] = max(receipt["max_delta_pressure_Pa"], float(np.max(abs(p - oldp))))
            values = {"theta": theta, "pressure": p, "surface_pressure": surface}
            if "pressure_p" in ds.variables:
                values["pressure_p"] = p - read("pressure_base", cells)
            for name, value in values.items():
                if name in ds.variables:
                    ds[name][index(ds[name], cells)] = value
    receipt["seconds"] = time.monotonic() - started
    return receipt

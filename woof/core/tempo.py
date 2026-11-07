"""TEMPO microphysics adapter (scheme name "tempo").

TEMPO (NCAR/TEMPO 9adf9ef, tempo_v3.0.3, the RRFS v2 pin; Apache-2.0) is
the successor of Thompson microphysics: aerosol-aware (prognostic nc, nwfa,
nifa) and hail-aware (prognostic graupel number ``ng`` and graupel volume
``volg``).  The column physics runs on the GPU through
:mod:`woof.core.tempo_runtime`; this adapter maps the WOOF state onto
TEMPO's ``tempo_run`` arguments the way MPAS/RRFS calls it:

* t1d = theta * pi in float64 (float32 full theta and Exner from pressure), p1d = full
  pressure, w1d = ``w`` at the bottom face of each mass level (the WRF
  Thompson convention WOOF's mp=28 uses), dz1d from geopotential.
* qb1d is TEMPO's graupel volume in L/kg; WOOF stores ``volg`` in m3/kg
  (MPAS's name and unit), so the adapter scales by 1000 on the way in and
  out.
* After the call: theta via ``moist_physics_finish`` (identical to every
  other scheme), RAINNCV = rain + ice + snow + graupel (mm), SNOWNCV = snow
  + ice, GRAUPELNCV = graupel, SR = frozen / (RAINNCV + 1e-12) as WRF's
  Thompson driver writes it; REFL_10CM from TEMPO's own reflectivity; the
  effective radii in microns for radiation; the lowest-level water-friendly
  aerosol emission ``nwfa += nwfa2d * dt`` exactly as TEMPO's
  ``tempo_aerosol_surface_emissions``.
"""
from __future__ import annotations

import os

import cupy as cp

from woof.core import constants as c
from woof.core.microphysics import (
    MicrophysicsDiagnostics,
    moist_physics_finish,
    save_pre_mp_theta,
)
from woof.core.state import DTYPE

#: State fields a TEMPO call needs (the scheme's prognostic set).
REQUIRED_STATE_FIELDS = ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ng",
                         "volg", "nwfa", "nifa")
REQUIRED_SURFACE_FIELDS = ("nwfa2d",)

_RUNTIME = {}
_FIRST_CALL = True


def _runtime(cfg, nz: int):
    """One device blob and compiled module per process, table set and nz.

    The column work area is batched to ``WOOF_TEMPO_WORK_BYTES`` (default
    1 GiB) so the call's transient footprint is bounded whatever the domain.
    """
    from woof.core.tempo_runtime import TempoRuntime

    work = int(os.environ.get("WOOF_TEMPO_WORK_BYTES", str(1 << 30)))
    blob_path = os.environ.get("WOOF_TEMPO_BLOB", "").strip()
    if blob_path:
        key = ("blob", blob_path, nz)
        if key not in _RUNTIME:
            _RUNTIME[key] = TempoRuntime(blob_path, nz_max=nz, max_work_bytes=work)
        return _RUNTIME[key]
    from woof.tempo_assets import ensure_tempo_tables

    root = ensure_tempo_tables()
    key = ("tables", str(root), nz)
    if key not in _RUNTIME:
        _RUNTIME[key] = TempoRuntime(table_dir=root, nz_max=nz, max_work_bytes=work)
    return _RUNTIME[key]


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False) -> MicrophysicsDiagnostics:
    """One TEMPO microphysics call over the whole mass grid."""
    from woof.core.tempo_runtime import TempoConfig

    missing = [n for n in REQUIRED_STATE_FIELDS + REQUIRED_SURFACE_FIELDS
               if getattr(state, n, None) is None]
    if missing:
        raise ValueError("TEMPO state lacks " + ", ".join(missing))
    nz, ny, nx = state.p.shape
    rt = _runtime(cfg, nz)
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    dz = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    w_mass = state.scratch((nz, ny, nx), "mp_tempo_w")
    th[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
               else thb + state.thp)
    rt.exner(state.p, pii)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]
    w_mass[...] = state.w[:-1]

    save_pre_mp_theta(state)
    fields = {"t": th, "p": state.p, "w": w_mass, "dz": dz, "qv": state.qv,
              "qc": state.qc, "qr": state.qr, "qi": state.qi, "qs": state.qs,
              "qg": state.qg, "ni": state.ni, "nr": state.nr, "nc": state.nc,
              "nwfa": state.nwfa, "nifa": state.nifa, "ng": state.ng, "qb": state.volg}
    ml = bool(cfg.tempo_ml_for_bl_nc)
    pbl = getattr(getattr(state, "physics", None), "fields", {})
    tcfg = TempoConfig(refl10cm=bool(refl_10cm_due), ml_for_bl_nc=ml,
                      surface_emission_before=bool(cfg.tempo_surface_emission_before))
    global _FIRST_CALL
    # Match TEMPO main's module SAVE flag across domains and nz-specific
    # runtime modules. Its first column seeds inside float64 column physics.
    rt.first_call = getattr(state, "tempo_first_call", _FIRST_CALL)
    out = rt.step(fields, float(dt), tcfg, exner=pii, surface_emission=state.nwfa2d,
                  qc_bl=pbl.get("qc_bl") if ml else None,
                  fraction_bl=pbl.get("cldfra_bl") if ml else None)
    _FIRST_CALL = False
    if hasattr(state, "tempo_first_call"):
        state.tempo_first_call = False

    sfc = out["surface"]
    surface_shape = (ny, nx)
    rainnc = state.scratch(surface_shape, "mp_rainnc")
    rainncv = state.scratch(surface_shape, "mp_rainncv")
    snownc = state.scratch(surface_shape, "mp_snownc")
    snowncv = state.scratch(surface_shape, "mp_snowncv")
    graupelnc = state.scratch(surface_shape, "mp_graupelnc")
    graupelncv = state.scratch(surface_shape, "mp_graupelncv")
    sr = state.scratch(surface_shape, "mp_sr")
    rain = sfc["rain_precip"]
    ice = sfc["ice_liquid_equiv_precip"]
    snow = sfc["snow_liquid_equiv_precip"]
    graupel = sfc["graupel_liquid_equiv_precip"]
    total = rain + ice + snow + graupel
    rainncv[...] = total
    snowncv[...] = snow + ice
    graupelncv[...] = graupel
    rainnc += rainncv
    snownc += snowncv
    graupelnc += graupelncv
    sr[...] = (snow + ice + graupel) / (total + 1.0e-12)

    col = out["column"]
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        refl[...] = col["refl10cm"]
        stash_refl_10cm(state, refl)
    # TEMPO effective_radius already bounds present species (2.51-50,
    # 2.51-125, 5.01-999 microns) and returns zero for absent species.
    # Convert its float64 result once to the radiation's float32 microns.
    state.effc[...] = col["re_cloud"] * 1.0e6
    state.effi[...] = col["re_ice"] * 1.0e6
    state.effs[...] = col["re_snow"] * 1.0e6
    hail = getattr(state, "tempo_max_hail_diameter", None)
    if hail is not None:
        hail[...] = sfc["max_hail_diameter_column"]

    moist_physics_finish(state, cfg, th, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)

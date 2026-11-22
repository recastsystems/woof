"""TEMPO microphysics adapter (scheme name "tempo").

TEMPO (NCAR/TEMPO; Apache-2.0; ``tempo_version`` selects the generation:
``v3.2.5``, the latest release and the default, or ``v3.0.3``, the RRFS v2
pin the bitwise claims are proven against) is the successor of Thompson
microphysics: aerosol-aware (prognostic nc, nwfa,
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


def adopt_checkpoint_generation(experiment, restart):
    """The experiment a resume runs: TEMPO keeps its checkpoint's generation.

    A checkpoint header echoes the TEMPO generation that integrated it, and
    a header without the key ran v3.0.3 (woof.io.restart
    ECHO_ABSENT_RUN_VALUES).  Named breakage: once v3.2.5 became the
    default, every TEMPO checkpoint written before it was refused on resume
    under its own unchanged configuration ("tempo_version differ(s) from
    the checkpoint") unless the operator found and stated
    tempo_version = "v3.0.3".  One forecast never splices two TEMPO
    generations, so the resume continues the checkpoint's, and says so
    when the configuration asks for another.  The keys are forecast-only
    (woof.ingest.prepared_cache excludes them from the prepared identity),
    so the prepared inputs stay bound.
    """
    if restart is None or not any(
            int(getattr(dc.run, "mp_physics", 0) or 0) == 900
            for dc in experiment.domains):
        return experiment
    from dataclasses import replace
    from pathlib import Path
    from woof.io.restart import read_restart_header
    try:
        stored = read_restart_header(Path(restart)).get("config") or {}
    except (OSError, ValueError, KeyError):
        # The restart guard reads the same file and names what is wrong.
        return experiment
    if int(stored.get("mp_physics", 0) or 0) != 900:
        return experiment
    kept = {"tempo_version": stored.get("tempo_version", "v3.0.3"),
            "tempo_hailhyperaware": bool(stored.get("tempo_hailhyperaware", True))}
    domains = []
    for dc in experiment.domains:
        run = dc.run
        if int(getattr(run, "mp_physics", 0) or 0) == 900 and any(
                getattr(run, key) != value for key, value in kept.items()):
            from woof.explain import warn as _warn
            _warn(f"TEMPO GENERATION KEPT FROM THE CHECKPOINT: {restart} was "
                  f"integrated with tempo_version={kept['tempo_version']!r} "
                  f"(hail-hyperaware {kept['tempo_hailhyperaware']}); this "
                  f"configuration asks for {run.tempo_version!r}.  The resumed "
                  "run keeps the checkpoint's generation; start a new forecast "
                  f"to use {run.tempo_version!r}.")
            run = replace(run, **kept)
        domains.append(dc if run is dc.run else replace(dc, run=run))
    return replace(experiment, domains=tuple(domains))


def tempo_version(cfg) -> str:
    """The configured TEMPO generation (RunConfig ``tempo_version``)."""
    from woof.core.tempo_runtime import TEMPO_DEFAULT_VERSION
    return getattr(cfg, "tempo_version", TEMPO_DEFAULT_VERSION)


def _runtime(cfg, nz: int):
    """One device blob and compiled module per process, table set and nz.

    The column work area is batched to ``WOOF_TEMPO_WORK_BYTES`` (default
    1 GiB) so the call's transient footprint is bounded whatever the domain.
    """
    from woof.core.tempo_runtime import TempoRuntime

    work = int(os.environ.get("WOOF_TEMPO_WORK_BYTES", str(1 << 30)))
    version = tempo_version(cfg)
    blob_path = os.environ.get("WOOF_TEMPO_BLOB", "").strip()
    if blob_path:
        key = ("blob", blob_path, nz, version)
        if key not in _RUNTIME:
            _RUNTIME[key] = TempoRuntime(blob_path, nz_max=nz, max_work_bytes=work,
                                         version=version)
        return _RUNTIME[key]
    from woof.tempo_assets import ensure_tempo_tables

    root = ensure_tempo_tables()
    key = ("tables", str(root), nz, version)
    if key not in _RUNTIME:
        _RUNTIME[key] = TempoRuntime(table_dir=root, nz_max=nz, max_work_bytes=work,
                                     version=version)
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
                      surface_emission_before=bool(cfg.tempo_surface_emission_before),
                      hailhyperaware=bool(getattr(cfg, "tempo_hailhyperaware", True)))
    global _FIRST_CALL
    if rt.version == "v3.0.3":
        # Match TEMPO main's module SAVE flag across domains and nz-specific
        # runtime modules. Its first column seeds inside float64 column physics.
        rt.first_call = getattr(state, "tempo_first_call", _FIRST_CALL)
    else:
        # v3.2.5 tempo_run: is_first_step = (itimestep == 1), every column
        # of the domain's first microphysics call (v3.1.1 #38).  itimestep is
        # the driver's serialized microphysics_updates + 1 (the WRF-door
        # convention, woof/core/microphysics.py), so a restart continues it.
        updates = getattr(getattr(state, "physics", None), "microphysics_updates", None)
        first = True if updates is None else int(updates) == 0
        rt.first_call = getattr(state, "tempo_first_call", first)
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
    from woof.core.hail import tempo_feeds_gsd_hail
    if tempo_feeds_gsd_hail(cfg):
        # The fork's gsd_diagnostics hail window: the scheme's own PSD
        # maximum diameter folded every microphysics call, lowest level and
        # column, in metres (TEMPO reports millimetres); gsd_diagnostics
        # resets the window and folds the run maxima.
        import cupy as cp
        for slot, key in (("hail_maxk1", "max_hail_diameter_sfc"),
                          ("hail_max2d", "max_hail_diameter_column")):
            window = state.existing_scratch(slot)
            if window is not None:
                cp.maximum(window, (sfc[key] * 1.0e-3).astype(window.dtype),
                           out=window)

    moist_physics_finish(state, cfg, th, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)

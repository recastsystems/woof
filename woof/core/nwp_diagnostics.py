"""WRF 4.7.1 NWP diagnostic driver, device-only at the shared step seam.

The common DomainState, history and checkpoint paths own all persistent planes.
Tracking consumers retain their independent completed-step UH windows.
"""
from __future__ import annotations
import numpy as np
from woof.core.nwp_windows import FIELDS

HISTORY = {name.lower(): (name, units, description) for name, units, description in (
    ("WSPD10MAX", "m s-1", "Maximum 10 m wind speed in history interval"),
    ("W_UP_MAX", "m s-1", "Maximum updraft velocity below 400 hPa"),
    ("W_DN_MAX", "m s-1", "Minimum downdraft velocity below 400 hPa"),
    ("W_MEAN", "m s-1", "History interval mean vertical velocity for eta 0.5 to 0.8"),
    ("GRPL_MAX", "kg m-2", "Maximum column integrated graupel"),
    ("REFD_MAX", "dBZ", "Maximum lowest-level derived radar reflectivity"),
    ("HAIL_MAXK1", "m", "Maximum PSD hail diameter at lowest level"),
    ("HAIL_MAX2D", "m", "Maximum PSD hail diameter in column"),
)}
PERSISTENT = tuple(name for name in HISTORY if not name.startswith("hail_"))
# Time-t work the end-of-step windows read.  The windows read phy_prep's
# interface pressure only as "p8w > 400 hPa", so the resident driver holds
# that comparison as one bit per interface level (nwp_below_400hpa, 32
# levels per FP32-sized word, word-major) and diag_print's surface interface
# plane (nwp_surface_pw), not the (nz+1)-level pressure; the base geopotential
# is read from the state, not copied; nwp_zero is the 2D stand-in for absent
# 10 m winds and reflectivity (absent qv/qg are kernel flags).  The native
# module's W_COLMEAN/NUMCOLPTS/GRPL_COLINT column outputs have no reader here
# and get no planes (the replay ABI still returns all twelve).  Seal4: the
# 3D copies (~240 MB at 550x550x49) put the 16 GB headline card over budget.
WORK = ("nwp_rho", "nwp_below_400hpa", "nwp_zero")


def surface_pw_held(cfg):
    """diag_print reads the surface interface pressure only without microphysics."""
    return bool(cfg.diag_print) and not cfg.mp_physics


def active(cfg):
    # GSD retains its separately pinned fork's package when selected.
    return cfg.nwp_diagnostics == 1 and not cfg.gsd_diagnostics


def below_400hpa_words(nz):
    """Words per column of the interface-level "p8w > 400 hPa" bit mask."""
    return (int(nz) + 1 + 31) // 32


def scratch_shapes(cfg):
    if not active(cfg):
        return {}
    plane = (cfg.ny, cfg.nx)
    shapes = {name: plane for name in (*HISTORY, "up_heli_max", *WORK)}
    shapes.update(nwp_rho=(cfg.nz, *plane),
                  nwp_below_400hpa=(below_400hpa_words(cfg.nz), *plane))
    if surface_pw_held(cfg):
        shapes["nwp_surface_pw"] = plane
    return shapes


def prepare(state, cfg):
    """Hold phy_prep's EOS density and the windows' pressure test at time t."""
    if not active(cfg) and not cfg.diag_print:
        return
    from woof.core.kernels import get_kernel
    state._native_nwp_prepared = True
    s = state.existing_scratch
    n = cfg.ny*cfg.nx
    phb3 = np.int32(state.phb.ndim == 3)
    if active(cfg):
        has_qv = state.qv is not None
        surface = s("nwp_surface_pw") if surface_pw_held(cfg) else None
        get_kernel("nwp_windows", "nwp_prepare_resident")(((n+127)//128,), (128,),
            (state.p, state.alt, state.qv if has_qv else state.alt, np.int32(has_qv),
             state.php, state.phb, phb3, state.fnm, state.fnp,
             s("nwp_rho"), s("nwp_below_400hpa"), s("nwp_rho") if surface is None else surface,
             np.int32(surface is not None), np.int32(cfg.nz), np.int32(n)))
        return
    get_kernel("nwp_windows", "nwp_prepare")(((n+127)//128,), (128,),
        (state.p, state.alt, state.qv if state.qv is not None else s("misc_zero3d"),
         state.php, state.phb, state.fnm, state.fnp,
         s("misc_rho"), s("misc_pressure_w"), s("misc_phb"),
         np.int32(cfg.nz), np.int32(n), phb3))


def sample_helicity(state, cfg):
    """Native cal_helicity runs in first_rk_step_part2, before RK transport."""
    if not active(cfg) and not cfg.afwa_severe_opt:
        return
    from woof.core.uh_diag import device_uh_step
    plane = (cfg.ny, cfg.nx)
    device_uh_step(state.u, state.v, state.w, state.php, state.phb,
        state.msfu, state.msfv, state.ht, state.dn, state.dnw, state.fnm, state.fnp,
        state.cf1, state.cf2, state.cf3, np.float32(1)/np.float32(cfg.dx),
        np.float32(1)/np.float32(cfg.dy), state.scratch(plane, "uh_diag_col"),
        state.scratch(plane, "uh_diag_use"), state.existing_scratch("up_heli_max"))


def update(state, cfg):
    """Native end-of-step reset, non-PSD windows, then scheme-specific PSD."""
    if not active(cfg):
        return
    from woof.core.kernels import get_kernel
    from woof.core.nwp_hail import update as update_psd
    s = state.existing_scratch
    zero = s("nwp_zero")
    qg = getattr(state, "qg", None)
    rho = s("nwp_rho")
    refl = s("refl_10cm")
    fields = getattr(state.physics, "fields", {})
    n = cfg.ny*cfg.nx
    minutes = cfg.output_interval_s / 60
    # The native fixed-step idump follows the domain's original history interval.
    epoch = float(getattr(state, "domain_start_offset", 0.) or 0.)
    step = getattr(state, "_nwp_itimestep", None)
    if step is None:  # Direct step API without an enclosing DomainClock.
        step = int(np.floor((state.elapsed_seconds-epoch) / cfg.dt + .5)) + 1
    # refl is read at its lowest level only, so the 2D zero stands in for it.
    get_kernel("nwp_windows", "nwp_window_resident")(((n+127)//128,), (128,),
        (state.w, s("nwp_below_400hpa"), state.znw, rho if qg is None else qg,
         np.int32(qg is not None), rho, state.php, state.phb, np.int32(state.phb.ndim == 3),
         fields.get("u10", zero), fields.get("v10", zero),
         zero if refl is None else refl, *(s(name.lower()) for name in FIELDS[:9]),
         np.int32(cfg.nz), np.int32(cfg.ny), np.int32(cfg.nx), np.float32(cfg.dt),
         np.float32(state.elapsed_seconds/60), np.int32(minutes), np.int32(step),
         np.int32(cfg.use_adaptive_time_step), np.int32(cfg.max_time_step)))
    mp = cfg.mp_physics
    kw = {}
    if mp in (8, 28, 38):
        kw.update(qr=state.qr, nr=state.nr)
        if mp == 38:
            kw.update(ng=state.ng, volume=state.qvolg)
    elif mp in (10, 40):
        kw.update(ng=state.ng, mpuse_hail=int(cfg.morr_rimed_ice))
    elif mp == 9:
        kw.update(qh=state.qh, ng=state.ng, nh=state.nh)
    elif mp == 18:
        # The admitted NSSL package is two-moment. Native NWP PSD is inactive
        # for it; still pass its actual number fields, preserving PRESENT.
        kw.update(qh=state.qh, ng=state.qng, nh=state.qnh)
    elif mp == 16:
        kw['mpuse_hail'] = cfg.wdm6_hail_opt
    elif mp == 97:
        kw.update(gsfcgce_hail=cfg.gsfcgce_hail, gsfcgce_2ice=cfg.gsfcgce_2ice)
    if qg is None and (mp in (2, 6, 8, 9, 10, 16, 18, 28, 38, 40)
                       or (mp == 97 and cfg.gsfcgce_2ice != 1)):
        # Every scheme the PSD leaf computes for carries graupel; without it
        # the stand-in below would be read as graupel.
        raise RuntimeError(f"native NWP PSD hail for mp_physics={mp} needs qg, "
                           "and this state carries none")
    # A scheme without graupel takes the leaf's no-PSD branch, which returns
    # before it reads any species, so density stands in for the absent qg.
    update_psd(rho if qg is None else qg, rho, s("hail_maxk1"), s("hail_max2d"),
               mp_physics=mp, **kw)
    # Preserve the existing extended run maxima without changing native windows.
    import cupy as cp
    for name in ('hail_maxk1', 'hail_max2d'):
        run = s(name+'_run')
        if run is not None:
            cp.maximum(run, s(name), out=run)


def registry_rows(registry):
    registry.setdefault('diagnostic_packages', {})['nwp_diagnostics'] = {
        '0': {'label': 'Off', 'fields': []},
        '1': {'label': 'WRF 4.7.1 NWP interval diagnostics',
              'fields': ['up_heli_max', *HISTORY],
              'source_pin': 'f52c197ed39d12e087d02c50f412d90d418f6186',
              'consumer': 'woof/core/nwp_diagnostics.py'}}
    registry.setdefault('diagnostic_state', {})['nwp'] = {
        slot: {'description': description, 'units': units, 'history': True, 'restart': True}
        for slot, (_, units, description) in HISTORY.items()}
    registry['parameters']['nwp_diagnostics'] = {
        'type': 'integer', 'default': 0, 'enum': [0, 1], 'per_domain': False,
        'consuming_read': 'woof/core/nwp_diagnostics.py'}

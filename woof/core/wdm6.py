"""WDM6 double-moment warm-rain microphysics (WRF ``mp_physics=16``).

WDM6 is WSM6's ice half carried unchanged beside a DOUBLE-MOMENT warm rain:
cloud droplet number ``nc``, rain number ``nr`` and a prognostic CCN
reservoir ``nn`` are transported and predicted, rain's PSD becomes a gamma
with shape ``mu = 1`` and a diagnosed intercept, and autoconversion,
accretion, self-collection, evaporation and freezing all move number as
well as mass.

The numerical authority is the byte-frozen WRF v4.6.1
``phys/module_mp_wdm6.F`` in the 1974 reference bundle; the CUDA kernel
(``woof/core/kernels/wdm6.cu``) is transcribed from it line by line and
carries the file:line citations.  The host coefficient block is
``woof/core/wdm6_constants.py``, which imports WSM6's ``_rgmma`` and
``rimed_ice_constants`` rather than copying them, because ``wdm6init``'s
``hail_opt`` arm (:2096-2108) sets exactly the five constants
``mp_wsm6_init`` sets.

WHAT THE ADAPTER OWES THE SCHEME THAT WSM6'S DOES NOT

* ``xland``.  WDM6 picks its autoconversion threshold per column from the
  land/sea mask -- ``qc0`` (maritime, ``xncr0 = 5e7``) where ``slmsk == 2``
  and ``qc1`` (continental, ``xncr1 = 5e8``) elsewhere
  (module_mp_wdm6.F:607-614; the driver's own comment at
  module_microphysics_driver.F:2495 spells the convention "1: land,
  2: water").  woof keeps XLAND on the physics driver's surface field
  dict, so :func:`apply` reads it from there and REFUSES rather than
  substituting a mask: guessing "all land" would silently move the
  continental threshold onto an ocean domain, which is a factor of ten in
  the autoconversion trigger.
* ``nn`` initialisation.  WRF fills the whole CCN array with the namelist
  ``ccn_conc`` on the first time step (:220-227), and ``ccn0`` reaches
  nothing else in the module -- ``wdm62D`` and ``wdm6init`` both declare it
  ``intent(in)`` and never read it.  woof performs that fill once at state
  allocation from ``cfg.wdm6_ccn_conc`` (``woof/core/state.py``), which is
  the NSSL ``qnn`` precedent and is exactly equivalent for a cold start.

EVIDENCE.  The column oracle tools/wdm6_wrf461_oracle grades the kernel
word for word against WRF v4.6.1's own module_mp_wdm6.F (with the three
documented defined behaviours of docs/wdm6_oracle_known_deltas.md) on 84
columns in seven configurations: every output word equal under the strict
build (tests/test_wdm6_wrf461_oracle.py).  No forecast-trajectory
comparison against WRF exists, so the registry keeps the option at
implemented-unverified.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from woof.config import RunConfig
from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE, DomainState
from woof.core.wsm_family import (check_surface, check_volume, finish_fields,
                                    precipitation_surface)
from woof.core.wdm6_constants import (WDM6_DEEP_KMAX, WDM6_NUMBER_SPECIES,
                                       WDM6_SHALLOW_KMAX,
                                       WDM6_VERTICAL_LEVEL_BOUNDS,
                                       wdm6_level_tier)

_COLUMN_TPB = 32
# The ladder itself lives in wdm6_constants (a CuPy-free leaf) so the memory
# preflight can price a WDM6 configuration on a host with no device; these
# names are the adapter's local spelling of it.
_SHALLOW_KMAX = WDM6_SHALLOW_KMAX
_KMAX = WDM6_DEEP_KMAX
VERTICAL_LEVEL_BOUNDS = WDM6_VERTICAL_LEVEL_BOUNDS
_kernel_capacity = wdm6_level_tier


def launch_wdm6(theta, qv, qc, qr, qi, qs, qg, nn, nc, nr,
                rho, pii, pressure, dz, xland,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0, count_status=None,
                conservative_surface: bool = False) -> None:
    """Launch one WDM6 call over contiguous FP32 ``(nz,ny,nx)`` fields.

    ``conservative_surface`` is correct-mode finding WB5
    (``RunConfig.wdm6_conservative_surface``): its own unit compiled with
    ``WDM6_PLM_SURFACE``.  False compiles exactly the unit this function
    compiled before the key existed (the matched WRF arithmetic).
    """
    if not isinstance(conservative_surface, bool):
        raise TypeError("conservative_surface must be a bool, got "
                        f"{conservative_surface!r}")
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WDM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    with np.errstate(over='ignore', under='ignore'):
        device_dt = DTYPE(dt)
    if not np.isfinite(device_dt) or device_dt <= 0:
        raise ValueError(f"WDM6 dt must remain finite and positive in float32, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg,
        "nn": nn, "nc": nc, "nr": nr,
        "rho": rho, "pii": pii, "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    check_volume(volume, shape)
    check_surface({
            "xland": xland,
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            "sr": sr}, (ny, nx))
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    if count_status is None:
        count_status = cp.empty((1,), dtype=cp.uint32)
    if count_status.shape != (1,) or count_status.dtype != np.dtype('uint32'):
        raise ValueError('WDM6 count status must be one uint32 word')
    count_status.fill(cp.uint32(0))
    if conservative_surface:
        defines = (() if capacity == _SHALLOW_KMAX
                   else (("WDM6_KMAX", capacity),))
        kernel = get_kernel_int_defines(
            "wdm6", "wdm6_column", defines + (("WDM6_PLM_SURFACE", 1),))
    else:
        kernel = (get_kernel("wdm6", "wdm6_column")
                  if capacity == _SHALLOW_KMAX else
                  get_kernel_int_defines(
                      "wdm6", "wdm6_column", (("WDM6_KMAX", capacity),)))
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, qg, nn, nc, nr,
         rho, pressure, pii, dz, xland,
         rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
         effc, effi, effs, device_dt, np.int32(hail_opt),
         np.int32(nz), np.int32(ny), np.int32(nx), count_status))
    from woof.core import health_ledger
    def describe(flags):
        if flags & 1:
            raise FloatingPointError('WDM6 substep geometry requires finite positive density, layer thickness and interval; the affected column was not updated')
        raise FloatingPointError('WDM6 substep count is not representable by its signed 32-bit counter for the supplied density/layer thickness and interval; correct those inputs or use a shorter interval. The affected column was not updated')
    flags = health_ledger.read_status(count_status, site='wdm6-counts', describe=describe)
    if flags:
        describe(flags)


def column_land_mask(state: DomainState, *, mp_physics: int = 16) -> cp.ndarray:
    """WRF XLAND for the microphysics call, or a refusal naming the gap.

    Separated from :func:`apply` so the requirement is testable on its own
    and so the refusal text lives beside the reason.  WDM6 is the first
    woof microphysics scheme whose PROCESS RATES read a surface field, and
    a fabricated mask is not a degraded answer -- it is the wrong
    autoconversion threshold everywhere it is wrong.
    """
    driver = getattr(state, "physics", None)
    fields = getattr(driver, "fields", None)
    xland = None if fields is None else fields.get("xland")
    if xland is None:
        scheme = 7 if mp_physics == 26 else 6
        source_lines = "584-590" if scheme == 7 else "607-614"
        raise ValueError(
            f"mp_physics={mp_physics} (WDM{scheme}) needs WRF's XLAND land/sea mask for its "
            f"per-column autoconversion threshold (module_mp_wdm{scheme}.F:{source_lines}) "
            "and the physics driver carries none; refusing to substitute an "
            "all-land or all-water mask. Construct the domain through "
            "woof.core.physics.initialize_physics, which builds XLAND from "
            "the landmask for every configuration.")
    return xland


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WDM6, and finish retained MP heating."""
    if int(cfg.mp_physics) in (14, 26):
        return _apply_family(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nn", "nc", "nr", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=16 state lacks WDM6 fields: "
                         + ", ".join(missing))
    xland = column_land_mask(state)
    from woof.core.wsm6 import _wsm6_prepare
    theta, rho, pii, dz = _wsm6_prepare(state, "wdm6")

    surface = precipitation_surface(state)

    from woof.core.microphysics import save_pre_mp_theta
    save_pre_mp_theta(state)
    launch_wdm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        state.nn, state.nc, state.nr,
        rho, pii, state.p, dz, cp.ascontiguousarray(xland),
        dt=dt, **surface, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wdm6_hail_opt,
        count_status=state.scratch((1,), 'wdm6_count_status', dtype=cp.uint32),
        conservative_surface=bool(
            getattr(cfg, "wdm6_conservative_surface", False)))
    return finish_fields(state, cfg, theta, pii, dt, surface,
                         refl_10cm_due=refl_10cm_due)


__all__ = ["WDM6_NUMBER_SPECIES", "apply", "column_land_mask", "launch_wdm6"]

from woof.core import constants as c
from woof.core.wdm6_constants import (wdm_scheme, wdm5_level_tier, WDM5_SHALLOW_KMAX)


def launch_wdm7(theta, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr,
                rho, pii, pressure, dz, xland,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, hailnc, hailncv, sr, dt: float, *,
                effc, effi, effs, count_status=None, positive_rain_transport=False) -> None:
    """Launch the separate stock WRF WDM7 unit, including its hail category."""
    _launch(theta, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr, rho, pii,
            pressure, dz, xland, rainnc, rainncv, snownc, snowncv,
            graupelnc, graupelncv, hailnc, hailncv, sr, dt, effc=effc,
            effi=effi, effs=effs, hail_opt=0, count_status=count_status,
            positive_rain_transport=positive_rain_transport)


def _launch(theta, qv, qc, qr, qi, qs, qg, qh, nn, nc, nr,
            rho, pii, pressure, dz, xland,
            rainnc, rainncv, snownc, snowncv,
            graupelnc, graupelncv, hailnc, hailncv, sr, dt: float, *,
            effc, effi, effs, hail_opt: int, count_status,
            positive_rain_transport=False) -> None:
    hail = qh is not None
    scheme = "WDM7" if hail else "WDM6"
    if hail and (hailnc is None or hailncv is None):
        raise ValueError("WDM7 needs its hail accumulators HAILNC/HAILNCV")
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"{scheme} fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    with np.errstate(over='ignore', under='ignore'):
        device_dt = DTYPE(dt)
    if not np.isfinite(device_dt) or device_dt <= 0:
        raise ValueError(f"{scheme} dt must remain finite and positive in float32, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg,
        "nn": nn, "nc": nc, "nr": nr,
        "rho": rho, "pii": pii, "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    if hail:
        volume["qh"] = qh
    for name, value in volume.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in {
            "xland": xland,
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            **({"hailnc": hailnc, "hailncv": hailncv} if hail else {}),
            "sr": sr}.items():
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    if count_status is None:
        count_status = cp.empty((1,), dtype=cp.uint32)
    if count_status.shape != (1,) or count_status.dtype != np.dtype('uint32'):
        raise ValueError(f'{scheme} count status must be one uint32 word')
    count_status.fill(cp.uint32(0))
    if hail:
        # Independent WDM7 unit at the requested column tier.
        kernel = get_kernel_int_defines(
            "wdm7", "wdm7_column",
            (("WDM6_KMAX", capacity), ("WDM_HAIL", 1))
            + ((("WDM7_POSITIVE_RAIN_TRANSPORT", 1),) if positive_rain_transport else ()))
        kernel(
            (blocks,), (_COLUMN_TPB,),
            (theta, qv, qc, qi, qr, qs, qg, qh, nn, nc, nr,
             rho, pressure, pii, dz, xland,
             rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
             hailnc, hailncv, sr, effc, effi, effs, device_dt,
             np.int32(nz), np.int32(ny), np.int32(nx), count_status))
    else:
        kernel = (get_kernel("wdm6", "wdm6_column")
                  if capacity == _SHALLOW_KMAX else
                  get_kernel_int_defines(
                      "wdm6", "wdm6_column", (("WDM6_KMAX", capacity),)))
        kernel(
            (blocks,), (_COLUMN_TPB,),
            (theta, qv, qc, qi, qr, qs, qg, nn, nc, nr,
             rho, pressure, pii, dz, xland,
             rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
             effc, effi, effs, device_dt, np.int32(hail_opt),
             np.int32(nz), np.int32(ny), np.int32(nx), count_status))
    from woof.core import health_ledger
    def describe(flags):
        if flags & 1:
            raise FloatingPointError(f'{scheme} substep geometry requires finite positive density, layer thickness and interval; the affected column was not updated')
        raise FloatingPointError(f'{scheme} substep count is not representable by its signed 32-bit counter for the supplied density/layer thickness and interval; correct those inputs or use a shorter interval. The affected column was not updated')
    flags = health_ledger.read_status(count_status, site=f'{scheme.lower()}-counts', describe=describe)
    if flags:
        describe(flags)


def launch_wdm5(theta, qv, qc, qr, qi, qs, nn, nc, nr,
                rho, pii, pressure, dz,
                rainnc, rainncv, snownc, snowncv, sr, dt: float, *,
                effc, effi, effs, count_status=None) -> None:
    """Launch one WDM5 call over contiguous FP32 ``(nz,ny,nx)`` fields.

    WRF's ``wdm5`` (module_mp_wdm5.F:74-339) without its radar diagnostic,
    which :func:`woof.core.refl.launch_refl10cm_wdm5` runs: five mass
    classes (no graupel), the three number moments, no land/sea input.
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WDM5 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = wdm5_level_tier(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    with np.errstate(over='ignore', under='ignore'):
        device_dt = DTYPE(dt)
    if not np.isfinite(device_dt) or device_dt <= 0:
        raise ValueError(f"WDM5 dt must remain finite and positive in float32, got {dt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs,
        "nn": nn, "nc": nc, "nr": nr,
        "rho": rho, "pii": pii, "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    for name, value in volume.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in {"rainnc": rainnc, "rainncv": rainncv,
                        "snownc": snownc, "snowncv": snowncv,
                        "sr": sr}.items():
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    if count_status is None:
        count_status = cp.empty((1,), dtype=cp.uint32)
    if count_status.shape != (1,) or count_status.dtype != np.dtype('uint32'):
        raise ValueError('WDM5 count status must be one uint32 word')
    count_status.fill(cp.uint32(0))
    kernel = (get_kernel("wdm5", "wdm5_column")
              if capacity == WDM5_SHALLOW_KMAX else
              get_kernel_int_defines(
                  "wdm5", "wdm5_column", (("WDM5_KMAX", capacity),)))
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, nn, nc, nr,
         rho, pressure, pii, dz,
         rainnc, rainncv, snownc, snowncv, sr,
         effc, effi, effs, device_dt,
         np.int32(nz), np.int32(ny), np.int32(nx), count_status))
    from woof.core import health_ledger

    def describe(flags):
        if flags & 1:
            raise FloatingPointError(
                "WDM5 substep geometry requires finite positive density, "
                "layer thickness and interval; the affected column was not "
                "updated")
        raise FloatingPointError(
            "WDM5 substep count is not representable by its signed 32-bit "
            "counter for the supplied density/layer thickness and interval; "
            "correct those inputs or use a shorter interval. The affected "
            "column was not updated")
    flags = health_ledger.read_status(count_status, site="wdm5-counts",
                                      describe=describe)
    if flags:
        describe(flags)


def _apply_family(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run the configured WDM scheme, and finish
    retained MP heating.

    WDM5 (14), WDM6 (16) and WDM7 (26) share preparation and finish. Their
    mass inventory, land/sea mask and kernel unit come from the family row
    (:data:`woof.core.wdm6_constants.WDM_SCHEMES`).
    """
    scheme = wdm_scheme(cfg.mp_physics)
    if scheme is None:
        raise ValueError(f"mp_physics={cfg.mp_physics} is not a WDM scheme")
    nz, ny, nx = state.p.shape
    required = tuple(name for name in scheme.mass_species
                     if name not in ("qv", "qc", "qr")) \
        + scheme.number_species + ("effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError(f"mp_physics={scheme.mp_physics} state lacks "
                         f"{scheme.name} fields: " + ", ".join(missing))
    xland = column_land_mask(state, mp_physics=cfg.mp_physics) if scheme.land_sea_mask else None
    from woof.core.wsm6 import _wsm6_prepare
    theta, rho, pii, dz = _wsm6_prepare(state, "wdm6")

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = graupelncv = None
    if scheme.graupel:
        graupelnc = state.scratch(surface, "mp_graupelnc")
        graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    if scheme.hail:
        hailnc = state.scratch(surface, "mp_hailnc")
        hailncv = state.scratch(surface, "mp_hailncv")

    from woof.core.microphysics import save_pre_mp_theta
    save_pre_mp_theta(state)
    if scheme.hail:
        launch_wdm7(
            theta, state.qv, state.qc, state.qr, state.qi, state.qs,
            state.qg, state.qh, state.nn, state.nc, state.nr,
            rho, pii, state.p, dz, cp.ascontiguousarray(xland),
            rainnc, rainncv, snownc, snowncv,
            graupelnc, graupelncv, hailnc, hailncv, sr, dt,
            effc=state.effc, effi=state.effi, effs=state.effs,
            count_status=state.scratch((1,), 'wdm6_count_status', dtype=cp.uint32),
            positive_rain_transport=True)
    elif scheme.graupel:
        launch_wdm6(
            theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
            state.nn, state.nc, state.nr,
            rho, pii, state.p, dz, cp.ascontiguousarray(xland),
            rainnc, rainncv, snownc, snowncv,
            graupelnc, graupelncv, sr, dt, effc=state.effc,
            effi=state.effi, effs=state.effs, hail_opt=cfg.wdm6_hail_opt,
            count_status=state.scratch((1,), 'wdm6_count_status', dtype=cp.uint32))
    else:
        launch_wdm5(
            theta, state.qv, state.qc, state.qr, state.qi, state.qs,
            state.nn, state.nc, state.nr, rho, pii, state.p, dz,
            rainnc, rainncv, snownc, snowncv, sr, dt, effc=state.effc,
            effi=state.effi, effs=state.effs,
            count_status=state.scratch((1,), 'wdm5_count_status',
                                       dtype=cp.uint32))
    surface_fields = dict(rainnc=rainnc, rainncv=rainncv, sr=sr,
                          snownc=snownc, snowncv=snowncv)
    if scheme.graupel:
        surface_fields.update(graupelnc=graupelnc, graupelncv=graupelncv)
    if scheme.hail:
        surface_fields.update(hailnc=hailnc, hailncv=hailncv)
    return finish_fields(state, cfg, theta, pii, dt, surface_fields,
                         refl_10cm_due=refl_10cm_due)

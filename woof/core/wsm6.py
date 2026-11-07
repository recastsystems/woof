"""WSM6 single-moment six-class microphysics (WRF ``mp_physics=6``).

The CUDA kernel owns one complete column and updates qv/qc/qr/qi/qs/qg,
potential temperature, surface rain/snow/graupel, frozen fraction, and the
scheme-native radiation effective radii.  WRF v4.6.1
``phys/physics_mmm/mp_wsm6.F90`` and ``mp_wsm6_effectRad.F90`` are the
primary numerical authorities.

The kernel is a word-for-word port of WRF's own call chain (the
``module_mp_wsm6.F`` wrapper, ``mp_wsm6_run`` with its semi-Lagrangian
sedimentation, and ``mp_wsm6_effectRad_run``), graded bit for bit against
those routines compiled by gfortran over 65 columns and six step/hail
configurations (tools/wsm6_wrf461_column_oracle,
tests/test_wsm6_wrf461_column_exact.py).  The column scheme is the claim;
whole-model trajectories are separate evidence.

The module also carries WSM6R, WRF's revised WSM6 (``mp_physics=106``,
``phys/module_mp_wsm6r.F``), as the family's second row (``WSM6_FAMILY``):
same six species, its own column kernel ``kernels/wsm6r.cu`` graded word for
word against WRF v4.6.1 by tests/test_wsm6r_wrf461_column_exact.py, and no
adapter module of its own.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from woof.config import RunConfig
from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE, DomainState
from woof.core.wsm_family import (check_surface, check_volume, finish_fields,
                                    precipitation_surface, prepare_fields)

from woof.core.device_cache import cuda_cache
from woof.core.wsm6_constants import (WSM6_DEEP_KMAX, WSM6_SHALLOW_KMAX,
                                       WSM6_VERTICAL_LEVEL_BOUNDS,
                                       WSM6R_RADII_F32,
                                       WSM6R_VERTICAL_LEVEL_BOUNDS,
                                       wrf_init_words, wrf_radar_words,
                                       wsm6_level_tier, wsm6r_init_words)

_COLUMN_TPB = 32
# The ladder itself lives in wsm6_constants (a CuPy-free leaf) so the memory
# preflight can price a WSM6 configuration on a host with no device; these
# names are the adapter's local spelling of it.
_SHALLOW_KMAX = WSM6_SHALLOW_KMAX
_KMAX = WSM6_DEEP_KMAX
VERTICAL_LEVEL_BOUNDS = WSM6_VERTICAL_LEVEL_BOUNDS
_kernel_capacity = wsm6_level_tier
_INIT_WORDS: dict = {}


def _init_words_on_device(hail_opt: int):
    """``mp_wsm6_init``'s float32 words for ``hail_opt``, uploaded once per device."""
    key = ("init", cp.cuda.runtime.getDevice(), int(hail_opt))
    words = _INIT_WORDS.get(key)
    if words is None:
        words = cp.asarray(wrf_init_words(hail_opt))
        _INIT_WORDS[key] = words
    return words


def prepare_model_graph(cfg):
    """Upload immutable scheme coefficients before cold stream capture.

    This does not call a column routine or advance any state. The normal
    launcher consumes the same cache and the same words.
    """
    if cfg.mp_physics == 6:
        _init_words_on_device(cfg.wsm6_hail_opt)
    elif cfg.mp_physics == 106:
        _wsm6r_coefficients()


def wsm6_radar_words_on_device(hail_opt: int, *, initialization="column_oracle"):
    """mp_radar's radar_init words for ``refl10cm_wsm6``: (float32, float64).

    ``initialization`` selects the column oracle's words or the stock
    scalar build's (wsm6_constants.wrf_radar_words).
    """
    key = ("radar", cp.cuda.runtime.getDevice(), int(hail_opt), initialization)
    words = _INIT_WORDS.get(key)
    if words is None:
        f32, f64 = wrf_radar_words(hail_opt, initialization=initialization)
        words = (cp.asarray(f32), cp.asarray(f64))
        _INIT_WORDS[key] = words
    return words


def wsm6_kernel(func: str, nz: int, conservative_surface: bool = False):
    """``func`` from ``kernels/wsm6.cu`` compiled at the WSM6_KMAX tier for ``nz``.

    ``conservative_surface`` is correct-mode finding WB5
    (``RunConfig.wsm6_conservative_surface``): its own tier compiled with
    ``WSM6_PLM_SURFACE``.  False compiles exactly the unit this function
    compiled before the key existed.
    """
    capacity = _kernel_capacity(nz)
    if conservative_surface:
        defines = (() if capacity == _SHALLOW_KMAX
                   else (("WSM6_KMAX", capacity),))
        return get_kernel_int_defines(
            "wsm6", func, defines + (("WSM6_PLM_SURFACE", 1),))
    if capacity == _SHALLOW_KMAX:
        return get_kernel("wsm6", func)
    return get_kernel_int_defines("wsm6", func, (("WSM6_KMAX", capacity),))


def wsm6_workspace_rows(nz: int) -> int:
    """Float rows per column of ``wsm6_column``'s workspace.

    KEEP IN STEP with ``WSM6_WS_ROWS`` in kernels/wsm6.cu: fifteen column
    arrays and the sedimentation pair at ``nz``, eight interface arrays at
    ``nz + 1``.  Sized at the run's ``nz``, not the compile tier.
    """
    nz = int(nz)
    return 17 * nz + 8 * (nz + 1)


def wsm6_tile_columns(kernel, ncol: int) -> int:
    """Columns in flight per launch: the card's resident capacity for
    ``kernel`` (occupancy, not the theoretical thread limit), floored at
    one block and capped at ``ncol``.

    The workspace is charged per column in flight; the local frame it
    replaced was charged per thread the card could ever hold, so a tile
    wider than the resident set would hand that cost back.  Tile width is
    memory shape only: every column runs the same thread program.
    """
    dev = cp.cuda.Device()
    sms = int(dev.attributes["MultiProcessorCount"])
    per_sm = int(dev.attributes["MaxThreadsPerMultiProcessor"]) // _COLUMN_TPB
    try:
        from cupy_backends.cuda.api import driver

        from woof.backend import active_blocks_per_multiprocessor
        resident = active_blocks_per_multiprocessor(
            getattr(kernel, "kernel", kernel).ptr, _COLUMN_TPB, 0)
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the thread limit
    per_sm = max(1, per_sm)
    return int(max(_COLUMN_TPB, min(int(ncol), sms * per_sm * _COLUMN_TPB)))


def _tile_blocks(tile: int) -> int:
    return (int(tile) + _COLUMN_TPB - 1) // _COLUMN_TPB


def launch_wsm6(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, dz,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0,
                conservative_surface: bool = False) -> None:
    """Launch one WSM6 call over contiguous FP32 ``(nz,ny,nx)`` fields.

    ``conservative_surface``: correct-mode finding WB5 (see wsm6_kernel).
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WSM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg, "rho": rho, "pii": pii,
        "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    check_volume(volume, shape)
    check_surface({
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            "sr": sr}, (ny, nx))
    ncol = ny * nx
    if not isinstance(conservative_surface, bool):
        raise TypeError("conservative_surface must be a bool, got "
                        f"{conservative_surface!r}")
    kernel = wsm6_kernel("wsm6_column", nz, conservative_surface)
    # The column arrays live in a workspace for the columns in flight
    # (kernels/wsm6.cu, COLUMN WORKSPACE): tiles of `tile` columns, each
    # launched with the same one-thread-one-column mapping, the tile passed
    # as a column OFFSET because a column tile of (nz, ny, nx) fields is
    # not contiguous.  The workspace returns to the pool with the call.
    tile = wsm6_tile_columns(kernel, ncol)
    if tile < ncol and tile % _COLUMN_TPB:
        # The last block of a partial tile would run the next tile's first
        # columns as well, so they would be stepped twice (measured: 2.4 %
        # of surface words moved at a 1,000-column tile).
        raise ValueError(f"WSM6 tile {tile} must be a multiple of "
                         f"{_COLUMN_TPB} columns when it splits the mesh")
    ws = cp.empty(_tile_blocks(tile) * _COLUMN_TPB * wsm6_workspace_rows(nz),
                  dtype=DTYPE)
    coef = _init_words_on_device(hail_opt)
    try:
        for lo in range(0, ncol, tile):
            count = min(tile, ncol - lo)
            kernel(
                (_tile_blocks(count),), (_COLUMN_TPB,),
                (theta, qv, qc, qi, qr, qs, qg, rho, pressure, pii, dz,
                 rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
                 effc, effi, effs, coef, DTYPE(dt),
                 np.int32(nz), np.int32(ny), np.int32(nx),
                 ws, np.int32(lo)))
    finally:
        del ws


@cuda_cache(maxsize=None, ready=True)
def _wsm6r_coefficients():
    """wsm6rinit's frozen float32 words on the current device."""
    words = np.asarray(wsm6r_init_words(), dtype=np.uint32).view(np.float32)
    return cp.asarray(words)


def launch_wsm6r(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, dz,
                 rainnc, rainncv, dt: float, *, work) -> None:
    """Launch one WSM6R (mp_physics=106) call over FP32 ``(nz,ny,nx)`` fields.

    The argument list is what WRF's driver hands ``wsm6r``
    (module_microphysics_driver.F:2362-2372): theta, the six moist
    species, rho, pi_phy, p, dz8w, dt, RAINNC and RAINNCV -- WSM6R reports
    no snow, graupel, SR or effective radius.  ``work`` is a
    ``(3, nz, ny, nx)`` FP32 scratch for fallk's per-level fall speeds.
    theta is updated in place (it holds the temperature during the call).
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WSM6R fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    minimum, _ = WSM6R_VERTICAL_LEVEL_BOUNDS
    if nz < minimum:
        raise ValueError(f"WSM6R requires nz >= {minimum}, got {nz}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg, "rho": rho, "pii": pii,
        "pressure": pressure, "dz": dz,
    }
    check_volume(volume, shape)
    if (work.shape != (3,) + shape or work.dtype != DTYPE
            or not work.flags.c_contiguous):
        raise ValueError(f"work must be C-contiguous float32 {(3,) + shape}, "
                         f"got {work.dtype} {work.shape}")
    check_surface({"rainnc": rainnc, "rainncv": rainncv}, (ny, nx))
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel("wsm6r", "wsm6r_column")(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, qg, rho, pressure, pii, dz,
         rainnc, rainncv, work, _wsm6r_coefficients(), DTYPE(dt),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_wsm6r_effective_radii(theta, pii, qc, qi, qs, rho, effc, effi,
                                 effs) -> None:
    """WSM6's mp_wsm6_effectRad_run on a WSM6R state, radii in microns.

    WRF's wsm6r computes no radius (has_reqc = 0, so WRF's RRTMG makes its
    own, as woof's legacy RRTMG arm does).  RTE+RRTMGP reads scheme radii,
    and WSM6R's size distributions are WSM6's, so woof evaluates WSM6's own
    routine on the post-call state at t = th*pii (kernels/wsm6r.cu).
    """
    shape = theta.shape
    nz, ny, nx = shape
    for name, value in {"theta": theta, "pii": pii, "qc": qc, "qi": qi,
                        "qs": qs, "rho": rho, "effc": effc, "effi": effi,
                        "effs": effs}.items():
        if (value.shape != shape or value.dtype != DTYPE
                or not value.flags.c_contiguous):
            raise ValueError(f"{name} must be C-contiguous float32 {shape}, "
                             f"got {value.dtype} {value.shape}")
    words = np.asarray([bits for _, bits in WSM6R_RADII_F32],
                       dtype=np.uint32).view(np.float32)
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel("wsm6r", "wsm6r_effective_radii")(
        (blocks,), (_COLUMN_TPB,),
        (theta, pii, qc, qi, qs, rho, effc, effi, effs,
         DTYPE(words[0]), DTYPE(words[1]),
         np.int32(nz), np.int32(ny), np.int32(nx)))


#: The WSM6 family by WRF selector: what each member's column kernel
#: produces beyond the six masses and theta.  WSM6R (106) carries WSM6's
#: species set, so it is this module's second row, not an adapter of its
#: own; its WRF wrapper reports RAINNC/RAINNCV only.
WSM6_FAMILY = {
    6: {"label": "WSM6", "radii": True, "frozen_surface": True},
    106: {"label": "WSM6R", "radii": True, "frozen_surface": False},
}


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run a WSM6-family scheme, finish MP heating."""
    mp = int(cfg.mp_physics)
    if mp not in WSM6_FAMILY:
        raise ValueError(f"mp_physics={mp} is not a WSM6-family scheme "
                         f"({sorted(WSM6_FAMILY)})")
    if mp == 106:
        return _apply_wsm6r(state, cfg, dt, refl_10cm_due=refl_10cm_due)
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=6 state lacks WSM6 fields: "
                         + ", ".join(missing))
    theta, rho, pii, dz = _wsm6_prepare(state, "wsm6", rounded_exner=True)
    surface = precipitation_surface(state)

    from woof.core.microphysics import save_pre_mp_theta
    save_pre_mp_theta(state)
    launch_wsm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        rho, pii, state.p, dz, dt=dt, **surface, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wsm6_hail_opt,
        conservative_surface=bool(
            getattr(cfg, "wsm6_conservative_surface", False)))
    return finish_fields(state, cfg, theta, pii, dt, surface,
                         refl_10cm_due=refl_10cm_due)


def _wsm6_prepare(state: DomainState, prefix: str, *, rounded_exner=False):
    """Shared named allocator, retained at the scratch-inventory pinned site."""
    nz, ny, nx = state.p.shape
    theta = state.scratch((nz, ny, nx), f"{prefix}_theta")
    rho = state.scratch((nz, ny, nx), f"{prefix}_rho")
    pii = state.scratch((nz, ny, nx), f"{prefix}_pii")
    dz = state.scratch((nz, ny, nx), f"{prefix}_dz")
    z8w = state.scratch((nz + 1, ny, nx), f"{prefix}_z8w")
    prepare_fields(state, theta, rho, pii, dz, z8w, rounded_exner=rounded_exner)
    return theta, rho, pii, dz


def _apply_wsm6r(state: DomainState, cfg: RunConfig, dt: float, *,
                 refl_10cm_due: bool = False):
    """mp_physics=106: WRF's wsm6r call, then the shared finish.

    WRF's wsm6r reports RAINNC and RAINNCV only, and no effective radius;
    the radii RTE+RRTMGP reads are WSM6's routine on this state
    (:func:`launch_wsm6r_effective_radii`).  The surface contract's SR
    slot is therefore zero, and
    :func:`woof.core.physics.microphysics_scheme_sr_available` says 106
    has none, so the land surface takes WRF's no-SR temperature proxy.
    Snow and graupel do not leave the column under their own names: WSM6R
    lumps every species' fallout into RAINNC, as WRF does.

    REFL_10CM is not produced by WRF's wsm6r at all; when an output frame
    is due, woof diagnoses it with WSM6's refl10cm_wsm6 operator
    (woof.core.refl), whose particle-size assumptions -- n0r 8e6, n0s
    2e6*exp(0.12*(t0c-t)), n0g 4e6, graupel density 500 -- are WSM6R's
    own PARAMETERs (module_mp_wsm6r.F:6-25).
    """
    nz, ny, nx = state.p.shape
    missing = [name for name in ("qi", "qs", "qg", "effc", "effi", "effs")
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=106 state lacks WSM6R fields: "
                         + ", ".join(missing))
    theta, rho, pii, dz = _wsm6_prepare(state, "wsm6r")
    work = state.scratch((3, nz, ny, nx), "wsm6r_work")
    surface = precipitation_surface(state, ("rain",))
    surface["sr"].fill(0.0)

    from woof.core.microphysics import save_pre_mp_theta
    save_pre_mp_theta(state)
    launch_wsm6r(theta, state.qv, state.qc, state.qr, state.qi, state.qs,
                 state.qg, rho, pii, state.p, dz, surface["rainnc"],
                 surface["rainncv"], dt,
                 work=work)
    launch_wsm6r_effective_radii(theta, pii, state.qc, state.qi, state.qs,
                                 rho, state.effc, state.effi, state.effs)
    return finish_fields(state, cfg, theta, pii, dt, surface,
                         refl_10cm_due=refl_10cm_due)


__all__ = ["WSM6_FAMILY", "apply", "launch_wsm6", "launch_wsm6r",
           "launch_wsm6r_effective_radii", "wsm6_kernel",
           "wsm6_radar_words_on_device"]

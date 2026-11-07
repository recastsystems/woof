"""Eta Ferrier microphysics (WRF ``mp_physics=95``, package ``etampnew``).

Transcription authority: byte-unmodified WRF v4.6.1
``phys/module_mp_etanew.F``; the CUDA translation unit is
``woof/core/kernels/etanew.cu`` and every block there cites its Fortran
lines.  This module owns three things:

* the device module, compiled through the direct NVRTC route with
  ``--ftz=false`` (CuPy's RawModule appends ``-ftz=true`` after a
  caller's options; subnormal precipitation fluxes and rime terms must
  survive) and, under ``GPUWM_WRF_EXACT=1``, the strict options of
  :mod:`woof.wrf_exact` (no multiply-add contraction, IEEE division);
* :func:`etanew_init` -- WRF's ``ETANEWinit`` (GPVS saturation tables,
  ``MY_GROWTH_RATES``, the restart vector and the module constants),
  computed on the device with WOOF's own float32 math routines;
* :func:`launch_column` -- WRF's ``ETAMP_NEW`` / ``EGCP01DRV`` /
  ``EGCP01COLUMN``, one thread per column, in place.

The forecast adapter :func:`apply` reads the scheme's carried state from
the DomainState (``qt``, ``f_ice_phy``, ``f_rain_phy``, ``f_rimef_phy``
and the restart vectors) and is dispatched from
:func:`woof.core.microphysics._dispatch_scheme` on the registry row.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.core.etanew_tables import TABLE_WORDS, load_tables

MODULE_KEY = "woof.core.etanew:etanew"
#: Both modes retain WRF operation order and gradual underflow.
BASE_OPTIONS = ("-std=c++17", "--ftz=false", "--fmad=false",
                "--prec-div=true", "--prec-sqrt=true")
_SOURCES = ("glibc_flt32.cuh", "etanew.cu")
_TPB = 64

#: MP_RESTART_STATE length (MY_T2 + 8, module_mp_etanew.F:161-170).
RESTART_STATE_WORDS = 43
#: TBPVS_STATE / TBPVS0_STATE length (NX, :31).
SATURATION_TABLE_WORDS = 7501
#: Module constants ETANEWinit sets outside the restart vector.
CONSTANT_WORDS = 17

#: The restart vector's named slots after MY_GROWTH(1:35).
RESTART_STATE_NAMES = tuple(f"MY_GROWTH({i})" for i in range(1, 36)) + (
    "C1XPVS0", "C2XPVS0", "C1XPVS", "C2XPVS", "CIACW", "CIACR", "CRACW",
    "CRAUT")


def module_options() -> tuple[str, ...]:
    """The NVRTC options for this process's arithmetic mode.

    The strict tuple is passed HERE, not only by the program hook, because
    the direct route's persistent image cache keys on the options the
    caller passes: a default image must never be loaded for a strict run.
    """
    from woof.wrf_exact import ENABLED, effective_options
    return effective_options(BASE_OPTIONS) if ENABLED else BASE_OPTIONS


def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _module(device: int, options: tuple[str, ...]):
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx, _ = compiler.compile_using_nvrtc(
            source, options, None, "etanew.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    record_module(MODULE_KEY, source=source, options=options, module=None)
    return module


def _kernel(name: str):
    import cupy as cp

    return _module(cp.cuda.Device().id, module_options()).get_function(name)


def frame_module():
    """Load the production image for frame checks and parallel prewarm."""
    import cupy as cp

    return _module(cp.cuda.Device().id, module_options())


@dataclass
class EtanewSetup:
    """What ETANEWinit leaves behind, on the device.

    ``restart_state``/``tbpvs``/``tbpvs0`` are WRF's ``mp_restart_state``,
    ``tbpvs_state`` and ``tbpvs0_state`` (Registry.EM_COMMON:1915-1917):
    restart fields, carried across a restart instead of recomputed.
    ``constants`` and ``tables`` are rebuilt by every init, as WRF rebuilds
    its module variables whenever ETANEWinit runs.
    """

    restart_state: object
    tbpvs: object
    tbpvs0: object
    constants: object
    tables: object
    dt: float


def etanew_init(dt: float, *, gsmdt: float = 0.0) -> EtanewSetup:
    """WRF ``ETANEWinit`` (module_mp_etanew.F:2088-2438) on the device.

    ``gsmdt`` is WRF's ``MPDT`` (minutes); 0, the Registry default, makes
    DTPH the model step.
    """
    import cupy as cp

    tables = cp.asarray(load_tables().flat, dtype=cp.float32)
    if tables.size != TABLE_WORDS:
        raise AssertionError("ETAMPNEW table size")
    tbpvs = cp.empty(SATURATION_TABLE_WORDS, dtype=cp.float32)
    tbpvs0 = cp.empty(SATURATION_TABLE_WORDS, dtype=cp.float32)
    rs = cp.zeros(RESTART_STATE_WORDS, dtype=cp.float32)
    consts = cp.zeros(CONSTANT_WORDS, dtype=cp.float32)
    dt_slot = cp.zeros(1, dtype=cp.float32)
    blocks = (SATURATION_TABLE_WORDS + 127) // 128
    _kernel("etanew_gpvs")((blocks,), (128,), (tbpvs, tbpvs0))
    _kernel("etanew_constants")(
        (1,), (1,), (tables, tbpvs0, np.float32(dt), np.float32(gsmdt),
                     rs, consts, dt_slot, np.int32(0)))
    return EtanewSetup(restart_state=rs, tbpvs=tbpvs, tbpvs0=tbpvs0,
                       constants=consts, tables=tables, dt=float(dt))


def launch_column(setup: EtanewSetup, th, qv, qt, qc, qr, qs, f_ice, f_rain,
                  f_rimef, p, pii, rho, dz8w, rainnc, rainncv, sr,
                  dt: float, *, restart_state=None, tbpvs=None,
                  tbpvs0=None) -> None:
    """Run ``etanew_column`` on device arrays, in place.

    3-D arrays are (nz, ny, nx) or (nz, ncol) float32 C-contiguous with
    level 0 at the surface; 2-D surface arrays have the trailing shape.
    ``restart_state``/``tbpvs``/``tbpvs0`` default to the setup's own (a
    restarted run passes the restored ones).
    """
    import cupy as cp

    nz = int(th.shape[0])
    if nz < 1:
        raise ValueError("Eta Ferrier needs at least one vertical level")
    ncol = int(th.size // nz)
    arrays = (th, qv, qt, qc, qr, qs, f_ice, f_rain, f_rimef, p, pii, rho,
              dz8w)
    for a in arrays:
        if a.dtype != cp.float32 or not a.flags.c_contiguous \
                or a.size != nz * ncol:
            raise ValueError("etanew_column needs C-contiguous float32 "
                             "(nz, ...) arrays of one shape")
    for a in (rainnc, rainncv, sr):
        if a.dtype != cp.float32 or not a.flags.c_contiguous \
                or a.size != ncol:
            raise ValueError("etanew_column surface arrays must be "
                             "C-contiguous float32 with one value per column")
    rs = setup.restart_state if restart_state is None else restart_state
    tb = setup.tbpvs if tbpvs is None else tbpvs
    tb0 = setup.tbpvs0 if tbpvs0 is None else tbpvs0
    blocks = (ncol + _TPB - 1) // _TPB
    _kernel("etanew_column")(
        (blocks,), (_TPB,),
        (th, qv, qt, qc, qr, qs, f_ice, f_rain, f_rimef, p, pii, rho, dz8w,
         rainnc, rainncv, sr, rs, tb, tb0, setup.constants, setup.tables,
         np.float32(dt), np.int32(nz), np.int32(ncol)))


def refresh_restart_state(setup: EtanewSetup, dt: float, rs,
                          dt_slot) -> None:
    """Rebuild the restart vector ``rs`` on the device when ``dt`` is not
    the step it was built for (``dt_slot[0]``); one single-thread launch,
    no host read, a no-op at a fixed step.

    WRF computes this vector once during initialization.  Forecast calls
    retain it across changing model steps and restarts, as ETAMP_NEW does.
    """
    import cupy as cp

    scratch_k = cp.empty_like(setup.constants)
    _kernel("etanew_constants")(
        (1,), (1,), (setup.tables, setup.tbpvs0, np.float32(dt),
                     np.float32(0.0), rs, scratch_k, dt_slot, np.int32(1)))


# ---------------------------------------------------------------------------
# Forecast adapter
# ---------------------------------------------------------------------------

#: The fields an mp_physics=95 DomainState carries for the scheme, beyond
#: qv/qc/qr: WRF's package line ``moist:qv,qc,qr,qs; scalar:qt;
#: state:f_ice_phy,f_rain_phy,f_rimef_phy`` (Registry.EM_COMMON:3045) and
#: the three restart vectors (:1915-1917).
STATE_FIELDS = ("qs", "qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy",
                "etanew_restart_state", "etanew_tbpvs", "etanew_tbpvs0",
                "etanew_restart_dt")

#: Device setups (tables + module constants) by device, built lazily.
_SETUPS: dict = {}
#: States whose restart vector presence has been checked (see apply).
_CHECKED: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _device_setup(state, dt: float) -> EtanewSetup:
    import cupy as cp

    device = int(cp.cuda.Device().id)
    setup = _SETUPS.get(device)
    if setup is None:
        setup = etanew_init(dt)
        _SETUPS[device] = setup
    return setup


def initialize_state(state, cfg) -> dict:
    """WRF ``ETANEWinit`` for a domain, once, at construction.

    Fills the three restart vectors when the state does not already carry
    them (``etanew_restart_dt`` is zero on a fresh allocation and the step
    a restored checkpoint was built for otherwise).

    Cold starts retain WRF ARW's zero qt and reset fractions.  This does
    not invoke the NMM-only condensate synchronization arm.
    """
    import cupy as cp

    receipt: dict[str, object] = {}
    dt = float(cfg.dt)
    setup = _device_setup(state, dt)
    if float(cp.asnumpy(state.etanew_restart_dt)[0]) == 0.0:
        refresh_restart_state(setup, dt, state.etanew_restart_state,
                              state.etanew_restart_dt)
        state.etanew_tbpvs[...] = setup.tbpvs
        state.etanew_tbpvs0[...] = setup.tbpvs0
        receipt["etanewinit"] = {"dtph_s": dt, "restart": False}
    else:
        receipt["etanewinit"] = {
            "dtph_s": float(cp.asnumpy(state.etanew_restart_dt)[0]),
            "restart": True}
    return receipt


def apply(state, cfg, dt: float, *, refl_10cm_due: bool = False):
    """Prepare WRF's fields, run ETAMP_NEW, and finish retained heating.

    The prep is ``moist_physics_prep_em`` as the Kessler adapter forms it
    (full theta, dry density 1/alt, Exner from full pressure, layer depths
    from the full geopotential); ETAMP_NEW receives ``dt`` as both its
    model step and its physics step (module_microphysics_driver.F:2577-2594,
    EGCP01DRV(DT, ...) at module_mp_etanew.F:259).
    """
    import cupy as cp
    from woof.core import constants as c
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         prepare_microphysics_exner,
                                         save_pre_mp_theta)
    from woof.core.state import DTYPE

    missing = [name for name in STATE_FIELDS
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=95 state lacks Eta Ferrier fields: "
                         + ", ".join(missing))
    nz, ny, nx = state.p.shape
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    th = state.scratch((nz, ny, nx), "mp_th")
    rho = state.scratch((nz, ny, nx), "mp_rho")
    pii = state.scratch((nz, ny, nx), "mp_pii")
    dz8w = state.scratch((nz, ny, nx), "mp_dz8w")
    z8w = state.scratch((nz + 1, ny, nx), "mp_z8w")
    th[...] = (state.total_theta() if getattr(state, "use_theta_m", 0)
               else thb + state.thp)
    rho[...] = 1.0 / state.alt
    prepare_microphysics_exner(state.p, pii)
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz8w[...] = z8w[1:] - z8w[:nz]

    rainnc = state.scratch((ny, nx), "mp_rainnc")
    rainncv = state.scratch((ny, nx), "mp_rainncv")
    sr = state.scratch((ny, nx), "mp_sr")

    setup = _device_setup(state, dt)
    if state not in _CHECKED:
        # Once per state, before any capture: the saturation tables are
        # filled only by initialize_state (or a restore); zero tables would
        # put every saturation pressure at zero.
        if float(cp.asnumpy(state.etanew_restart_dt)[0]) == 0.0:
            raise ValueError(
                "mp_physics=95 state has no ETANEWinit restart vector: "
                "woof.core.etanew.initialize_state did not run for this "
                "domain (physics.initialize_physics -> "
                "microphysics_cold_start), so the saturation tables are zero")
        _CHECKED[state] = True


    save_pre_mp_theta(state)
    launch_column(setup, th, state.qv, state.qt, state.qc, state.qr,
                  state.qs, state.f_ice_phy, state.f_rain_phy,
                  state.f_rimef_phy, state.p, pii, rho, dz8w, rainnc,
                  rainncv, sr, dt,
                  restart_state=state.etanew_restart_state,
                  tbpvs=state.etanew_tbpvs, tbpvs0=state.etanew_tbpvs0)
    # WRF's ETAMP_NEW call has no radar diagnostic. History membership
    # excludes this package rather than borrowing another particle distribution.
    moist_physics_finish(state, cfg, th, dt)
    return MicrophysicsDiagnostics(rainnc=rainnc, rainncv=rainncv, sr=sr)


__all__ = [
    "BASE_OPTIONS", "CONSTANT_WORDS", "EtanewSetup", "MODULE_KEY",
    "RESTART_STATE_NAMES", "RESTART_STATE_WORDS", "SATURATION_TABLE_WORDS",
    "STATE_FIELDS", "apply", "etanew_init", "initialize_state",
    "launch_column", "module_options", "module_source",
    "refresh_restart_state",
]

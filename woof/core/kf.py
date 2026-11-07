"""Kain--Fritsch cumulus column schemes and CUDA launchers.

The numerical authority is WRF v4.6.1 itself: ``kernels/kf.cu`` is a
line-for-line port of ``KF_eta_PARA`` (phys/module_cu_kfeta.F), graded at
0 ULP on every output of ``KF_eta_CPS`` against the unmodified Fortran by
the column oracle in ``tools/kf_wrf461_oracle`` (fixture and GPU gate:
``tests/test_kf_wrf461_oracle.py``).  This module owns the packaged
``KF_LUTAB`` data and the device-facing adapter for WRF ``cu_physics=1``.
Array layout is ``(nz, ny, nx)`` with x fastest.

CU99 uses kernels/kf_old.cu, transcribed from module_cu_kf.F and graded
by tools/kf_old_wrf461_oracle. One adapter serves both KF selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from woof.config import P3_MP_PHYSICS as _P3_MP_PHYSICS
from woof.core.device_cache import cuda_cache
from woof.core.mp_column_schemes import (SIX_MASS_TABLE_SCHEMES,
                                         TABLE_SCHEMES)
from pathlib import Path

import numpy as np


#: Eight warps share a column-order buffer, while each workspace warp
#: remains interleaved over the kernel's 32 lanes.
_TPB = 256
_WS_LANES = 32
_VALIDATION_TPB = 256
#: ``KF_KMAX`` in ``kernels/kf.cu`` and the ceiling on nz.  It is a REFUSAL
#: ceiling and nothing else since the column workspace landed: no array in
#: that file is sized by it.
_KMAX = 128
VERTICAL_LEVEL_BOUNDS = (8, _KMAX)
DTYPE = np.float32


# ---------------------------------------------------------------------------
# The per-thread column workspace
# ---------------------------------------------------------------------------
# One thread owns one whole column, and CUDA prices a per-thread local frame
# at the card's RESIDENT-THREAD CAPACITY -- one per-context backing store of
# ``(frame - 1024) * SMs * maxThreadsPerSM``, taken at first launch and never
# returned.  MEASURED on test host A (RTX 5070 Ti, 70 SMs x 1,536, sm_120, NVRTC
# 13.0.48) for the pre-workspace kernel: a 9,216 B frame at nz = 49 took
# 840.0 MiB.  So every column array of the WRF port -- 66 of them -- lives in
# a global workspace this module allocates, sized to the threads ACTUALLY in
# flight, and the columns are launched in tiles of that size.  The kernel's
# local frame is 0 B (MEASURED on W1, RTX PRO 6000 Blackwell, sm_120, NVRTC
# 12.9, 2026-10-07).
#
# This must match kf.cu's KFWS_SLOTS; tests/test_kf_workspace.py re-derives
# it from the .cu source and fails if either side moves alone.
KFWS_SLOTS = 66

#: One eight-warp block keeps the previous 256-column cap per SM.
#: The occupancy query can reduce the cap without increasing workspace.
KF_TILE_BLOCKS_PER_SM = 1


def kf_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole launch blocks.  Each warp owns a separate region
    interleaved across its 32 lanes, so the column-order buffer can group
    several warps without changing the workspace's access stride.

    The per-slot extent is the RUNTIME ``nz``, not ``KF_KMAX``: every loop in
    ``kf_column`` runs to ``nz`` and the highest index any of them forms is
    ``nz - 1``.
    """
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * KFWS_SLOTS * int(nz) * _TPB


def kf_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = KF_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        # A RawKernel wrapper carries the function as ``.kernel``; the
        # direct-NVRTC module's Function is the function itself.
        handle = getattr(fn, "kernel", fn)
        from woof.backend import active_blocks_per_multiprocessor
        resident = active_blocks_per_multiprocessor(
            handle.ptr, _TPB, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    return int(min(int(ncol), sms * per_sm * _TPB))


def kernel_capacity(nz: int) -> int:
    """Validate ``nz`` against ``KF_KMAX`` and return it.

    Nothing in ``kernels/kf.cu`` is sized by ``KF_KMAX`` any more -- the
    column arrays live in a runtime-sized global workspace -- so the module
    is compiled ONCE, at the source's own ceiling, instead of once per
    distinct level count.  What this still owns is the contract the ceiling
    names: ``nz`` past it is refused rather than silently truncated, which is
    also the guard ``kf_column`` carries.
    """
    nz = int(nz)
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if nz < minimum or nz > maximum:
        raise ValueError(
            f"KF requires {minimum} <= nz <= {maximum}, got {nz}")
    return nz


class KFPhaseMode(IntEnum):
    """WRF KF output-category contract derived from ``F_QI/F_QS``.

    The integer values are part of the CUDA ABI.  They distinguish WRF's
    four feedback branches instead of reducing the phase contract to a
    Morrison/non-Morrison boolean: the first two branches also apply the
    latent-fusion temperature adjustment required when frozen condensate is
    returned through liquid prognostics.
    """

    WARM_RAIN = 0
    NO_SEPARATE_SNOW = 1
    SEPARATE_SNOW = 2
    SEPARATE_ICE_SNOW = 3


def kf_phase_mode_for_microphysics(mp_physics: int) -> KFPhaseMode:
    """Resolve the supported WRF microphysics-to-KF phase flags.

    ``cu_physics=1`` is WRF's ``kfetascheme``
    (``Registry/Registry.EM_COMMON:3190``), whose driver arm
    (``phys/module_cumulus_driver.F:1015``) hands ``KF_eta_CPS`` the pair
    ``F_QI=f_qi, F_QS=f_qs`` at :1043.  ``KF_eta_CPS`` selects its feedback
    branch from exactly that pair (``phys/module_cu_kfeta.F:2622-2632``):
    with ``F_QS`` true it feeds the hydrometeor tendencies back directly, and
    with ``F_QI`` also true it returns a separate ``DQIDT`` instead of
    folding the ice into snow.

    ``mp_physics=28`` is therefore SEPARATE_ICE_SNOW for the same reason
    ``mp_physics=8`` is: ``Registry/Registry.EM_COMMON:3036`` declares the
    ``thompsonaero`` package as ``moist:qv,qc,qr,qi,qs,qg``, so ``P_QI`` and
    ``P_QS`` are both allocated and ``F_QI``/``F_QS`` are both true.  Before
    28 was admitted here, ``initialize_physics`` could not even construct an
    mp=28 + KF domain -- ``_cumulus_optional_tendency_components`` raised.

    ``mp_physics=9`` (Milbrandt-Yau) resolves the same way and by the same
    single test: ``Registry/Registry.EM_COMMON:3025`` declares ``package
    milbrandt2mom mp_physics==9 - moist:qv,qc,qr,qi,qs,qg,qh``, so ``P_QI``
    and ``P_QS`` are both allocated, ``F_QI``/``F_QS`` are both true and
    ``KF_eta_CPS`` takes the separate-DQIDT branch.  The extra ``qh`` in
    that package changes nothing here: KF's feedback pair is F_QI/F_QS and
    hail is not one of them.
    """
    from woof.core.column_schemes import COLUMN_SCHEMES
    row = COLUMN_SCHEMES.get(int(mp_physics))
    if row is not None:
        # WRF KF's phase flags come from the active Registry moist package.
        return (KFPhaseMode.SEPARATE_ICE_SNOW if "qi" in row.species and "qs" in row.species
                else KFPhaseMode.SEPARATE_SNOW if "qs" in row.species
                else KFPhaseMode.NO_SEPARATE_SNOW)
    # WRF 4.7.1 Registry.EM_COMMON:3040: UDM declares both qi and qs.
    if int(mp_physics) in (2, 6, 7, 8, 9, 10, 13, 14, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 106) + SIX_MASS_TABLE_SCHEMES:
        return KFPhaseMode.SEPARATE_ICE_SNOW
    if int(mp_physics) == 1:
        return KFPhaseMode.WARM_RAIN
    if int(mp_physics) == 95:
        # WRF 4.7.1 Registry.EM_COMMON:3050: etampnew has QS and no QI.
        return KFPhaseMode.SEPARATE_SNOW
    # mp_physics=3 (WSM3) takes this branch for the same two reasons
    # P3 does: module_physics_init.F:4482 leaves warm_rain .false. for
    # WSM3SCHEME, and the wsm3scheme package (Registry.EM_COMMON:3017)
    # declares moist:qv,qc,qr, so F_QS is false.
    if int(mp_physics) in (0, 3, 5, 15, 55) + _P3_MP_PHYSICS:
        # WRF sets WARM_RAIN only for Kessler.  With no microphysics scheme,
        # F_QS is false and KF uses the melting-level !F_QS closure.
        #
        # mp_physics=5 (Ferrier-Aligo) is the same case as P3 below:
        # fer_mp_hires is moist:qv,qc,qr,qi (Registry.EM_COMMON:3019), so
        # F_QS is false, and FER_MP_HIRES's init case does not set
        # warm_rain (module_physics_init.F:4499-4505).
        #
        # mp_physics=50 and 51 (P3) reach the SAME branch, and WRF's own control
        # flow says so without any interpretation on woof's part.  The
        # feedback cascade is
        # ``IF (warm_rain) ... ELSEIF (.NOT. F_QS) ... ELSEIF (F_QS) ...``
        # (phys/module_cu_kfeta.F:2599/:2607/:2622).  P3 does not set
        # ``warm_rain``: ``mp_init`` initialises it .false. at
        # module_physics_init.F:4459 and only the Kessler cases assign it
        # .true. (:4477, :4480), while P3's own case at :4568 does not
        # touch it (nor does the mp=51 case at :4570).  And ``F_QS`` is
        # false because the ``p3_1category`` and ``p3_1category_nc``
        # packages declare ``moist:qv,qc,qr,qi`` with no qs at all
        # (Registry.EM_COMMON:3038-3039).  warm_rain false + F_QS false selects
        # the melting-level closure at :2607 -- the one this mode names.
        # P3_2CATEGORY (52) has the same flags: p3_2category adds qi2
        # and its moments, with no qs (Registry.EM_COMMON:3040).
        #
        # The tendency contract matches: that branch sets DQIDT = 0 and
        # DQSDT = 0 (:2618, :2620) and folds ice into DQCDT and snow into
        # DQRDT, so KF contributes ``rqr`` only and asks P3's state for no
        # snow or ice tendency array it does not have
        # (_cumulus_optional_tendency_components, physics.py).
        #
        # This is a DERIVED admission, not a measured one: it is exactly as
        # verified as the mp=0 row beside it, and no P3+KF forecast has been
        # compared against WRF.  What IS executed is
        # tests/test_p3_port.py::
        # test_p3_runs_under_the_default_cumulus_scheme, which builds an
        # mp=50 + cu_physics=1 domain through initialize_physics and steps
        # microphysics on it.
        return KFPhaseMode.NO_SEPARATE_SNOW
    raise ValueError(
        f"KF has no verified phase-output contract for mp_physics="
        f"{mp_physics}")


def kf_has_rain_for_microphysics(mp_physics: int) -> bool:
    """WRF's ``F_QR`` for a KF-admitted microphysics scheme.

    Every admitted scheme but ``mp_physics=0`` carries a rain species, so
    KF_eta_CPS returns DQRDT as RQRCUTEN.  With no microphysics there is no
    QR and the rain tendency is folded into RQCCUTEN
    (module_cu_kfeta.F:492-505).
    """
    kf_phase_mode_for_microphysics(mp_physics)       # same admission
    return int(mp_physics) != 0


def _model_clock_dt(cfg) -> float:
    """WRF's model-clock ``dt`` for clock-defined cumulus arithmetic.

    The the reference case compatibility integrator advances internal substeps
    (``cfg.clock_dt > cfg.dt``); KF's driver formulas are defined on the
    model clock, the same idiom as ``lateral_boundary_clock_dt`` and the
    clock-scaled sixth-order diffusion factor.
    """
    clock_dt = float(getattr(cfg, "clock_dt", 0.0) or 0.0)
    return clock_dt if clock_dt > 0.0 else float(cfg.dt)


@dataclass(frozen=True)
class KFTable:
    """Host representation of WRF's private ``/KFLUT/`` state."""

    temperature: np.ndarray
    qsat: np.ndarray
    thetae_base: np.ndarray
    log_ratio: np.ndarray
    pressure_top: float
    pressure_reciprocal: float
    thetae_reciprocal: float


@lru_cache(maxsize=1)
def load_kf_table() -> KFTable:
    """Load the shipped FP32 rendering of WRF ``KF_LUTAB``.

    The generating transcription and SHA-256 provenance live beside the
    table in ``woof/data/kf_lutab``.
    """
    path = (Path(__file__).parents[1] / "data" / "kf_lutab" /
            "kf_lutab.npz")
    from woof.tables import table_path
    path = table_path("kf", path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.ascontiguousarray(data[name], dtype=np.float32)
                  for name in ("temperature", "qsat", "thetae_base",
                               "log_ratio")}
        scalars = {name: float(data[name]) for name in
                   ("pressure_top", "pressure_reciprocal",
                    "thetae_reciprocal")}
    if (arrays["temperature"].shape != (250, 220)
            or arrays["qsat"].shape != (250, 220)
            or arrays["thetae_base"].shape != (220,)
            or arrays["log_ratio"].shape != (200,)):
        raise ValueError(f"invalid KF_LUTAB dimensions in {path}")
    return KFTable(**arrays, **scalars)


@cuda_cache(maxsize=None, ready=True)
def _device_table_on(_device: int):
    """The KF_LUTAB on ONE card.  The argument is the cache key and nothing
    else -- it is the CURRENT device, which is what ``cp.asarray`` uploads to.

    Shared by every slab on the card, so it is published with its upload
    event (``device_cache.cuda_cache(ready=True)``): a slab stepping on
    another stream waits for the upload instead of reading the table before
    its bytes land.  MEASURED without it (lane 282-multigpu, one card split
    into 2x2 slabs, HRRR West Texas Kain-Fritsch suite): one of two runs
    stopped at its first step on "cumulus rthcuten contains a non-finite
    value", the other was byte-identical to the unsplit run.

    Keyed on the device because the alternative was measured to be fatal: a
    process-wide ``lru_cache`` handed every card the pointers of whichever
    one uploaded first, and on a dual-4090 (consumer Ada, no P2P) the second
    card's KF kernel died with CUDA_ERROR_ILLEGAL_ADDRESS -- which destroys
    the context for the whole process, so every later run in it also failed,
    at unrelated allocations.  The table is 250x220x2 + 220 + 200 float32
    = 0.42 MiB, so a copy per card is not a cost worth avoiding.
    """
    import cupy as cp

    table = load_kf_table()
    # TTAB/QSTAB go up in FORTRAN order (the (250, 220) host arrays are
    # TTAB(IT, IP) row-major; their transposes' C order is the Fortran
    # column-major order), so the kernel's TTAB(IT, IP) is word
    # (IT-1) + (IP-1)*250 exactly as the Fortran addresses it.
    return tuple(cp.asarray(np.ascontiguousarray(value)) for value in
                 (table.temperature.T, table.qsat.T, table.thetae_base,
                  table.log_ratio))


def _device_table():
    import cupy as cp

    return _device_table_on(cp.cuda.runtime.getDevice())


#: The KF translation unit's compile options, on its own direct-NVRTC
#: route (the ruc_spp/sfire pattern).  ``--fmad=false``: the WRF 4.6.1
#: column oracle is graded word for word and gfortran contracts nothing.
#: ``--ftz=false``: CuPy's RawModule appends ``-ftz=true`` to every module
#: it builds, and KF carries condensate through the subnormal range -- the
#: updraft's QLIQ/QICE decay by EXP(-CONV) per level above the freezing and
#: fallout levels.  MEASURED on W1 (RTX PRO 6000, sm_120, NVRTC 12.9) with
#: the ported kernel on the loader's flushed route: 41,127 of 1,848,256
#: output words off the oracle over the 4,096-column sweep (max 4,935,177
#: ULP in RTHCUTEN, 1,042 of 1,270 deep columns), and 0 unflushed.
KF_MODULE_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")
KF_MODULE_KEY = "woof.core.kernels:kf"


@lru_cache(maxsize=None)
def _kf_module(device, trigger=1):
    """kernels/kf.cu for one device, compiled unflushed and uncontracted."""
    import cupy as cp
    from woof import nvrtc_ptx_cache as compiler
    from woof.certify.kernel_manifest import record_module
    from woof.core.kernels import module_source
    from woof.kernel_compile_notice import observe_module_compile

    source = module_source("kf")
    if trigger != 1:
        source = f"#define KF_TRIGGER {int(trigger)}\n" + source
    key = KF_MODULE_KEY if trigger == 1 else f"{KF_MODULE_KEY}:trigger{trigger}"
    with cp.cuda.Device(device), observe_module_compile(key):
        binary, _ = compiler.compile_using_nvrtc(
            source, KF_MODULE_OPTIONS, None, "kf.cu")
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    from woof.physics_params import note_compiled
    note_compiled("kf")
    record_module(key, source=source, options=KF_MODULE_OPTIONS,
                  module=None)
    return module


def _kf_function(name: str, trigger=1):
    """One entry of kernels/kf.cu on the current device."""
    import cupy as cp

    return _kf_module(cp.cuda.runtime.getDevice(), trigger).get_function(name)


def launch_kf(u, v, temperature, qv, pressure, rho, exner, dz, w0avg, *,
              dx: float, dt: float, cudt_minutes: float,
              phase_mode: KFPhaseMode = KFPhaseMode.SEPARATE_ICE_SNOW,
              f_qr: bool = True, cloud_diagnostics: bool = False,
              trigger: int = 1, rqvften=None, tpart=None):
    """Run WRF KF (``KF_eta_PARA``) for a batch of FP32 device columns.

    The inputs are KF_eta_CPS's own: ``temperature`` (T), ``qv``,
    ``pressure`` (p_hyd), ``rho`` (WRF's (1+qv)/alt), ``exner`` (PI),
    ``dz`` (DZ8W), ``u``/``v`` and the ALREADY ADVANCED running-mean
    vertical velocity ``w0avg`` (:func:`advance_w0avg`).  ``dt`` is WRF's
    model DT, ``cudt_minutes`` the namelist CUDT (the shallow NCA is
    ``CUDT*60.``), ``phase_mode`` the (warm_rain, F_QS, F_QI) feedback
    branch and ``f_qr`` WRF's F_QR (false folds the rain tendency into
    RQCCUTEN, module_cu_kfeta.F:492-505).

    Returns KF_eta_CPS's outputs: the six tendencies, ``raincv`` (RAINCV =
    DT*PRATEC, also under the legacy key ``rainc``), ``pratec``,
    ``nca_seconds`` (NCA where the column convected, 0 elsewhere: the
    driver owns the timer), ``cutop``/``cubot``/``shall`` as WRF's REAL
    words, and diagnostics (``cape_before`` = ABE, ``cape_after`` = ABEG,
    ``timec``, ``updraft_mass_flux``/``downdraft_mass_flux``, and the
    integer views ``triggered``/``shallow``/``cloud_base``/``cloud_top``,
    0-based, -1 when the column did not convect).  With
    ``cloud_diagnostics`` it also returns WRF's CLDFRA_DP_KF,
    CLDFRA_SH_KF, QC_KF and QI_KF as ``cldfra_dp``/``cldfra_sh``/
    ``qc_kf``/``qi_kf``.
    """
    import cupy as cp

    columns = {"u": u, "v": v, "temperature": temperature, "qv": qv,
               "pressure": pressure, "rho": rho, "exner": exner,
               "dz": dz, "w0avg": w0avg}
    shape = temperature.shape
    if len(shape) != 3:
        raise ValueError(f"temperature must have shape (nz,ny,nx), got {shape}")
    nz, ny, nx = shape
    kernel_capacity(nz)
    for name, array in columns.items():
        if (not isinstance(array, cp.ndarray) or array.shape != shape
                or array.dtype != DTYPE or not array.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {shape}")
    for name, value in (("dx", dx), ("dt", dt)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"KF {name} must be positive and finite, got {value}")
    if not np.isfinite(cudt_minutes) or cudt_minutes < 0.0:
        raise ValueError(
            f"KF cudt_minutes must be finite and >= 0, got {cudt_minutes}")
    if isinstance(phase_mode, (bool, np.bool_)):
        raise ValueError("KF phase_mode must be an explicit KFPhaseMode")
    try:
        phase_mode = KFPhaseMode(phase_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid KF phase_mode {phase_mode!r}") from exc

    # The column kernel writes every output on every path.
    out = {name: cp.empty(shape, dtype=DTYPE) for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rqrcuten", "rqscuten",
            "updraft_mass_flux", "downdraft_mass_flux")}
    for name in ("raincv", "pratec", "nca_seconds", "cutop", "cubot",
                 "shall", "cape_before", "cape_after", "timec"):
        out[name] = cp.empty((ny, nx), dtype=DTYPE)
    if cloud_diagnostics:
        for name in ("cldfra_dp", "cldfra_sh", "qc_kf", "qi_kf"):
            out[name] = cp.empty(shape, dtype=DTYPE)
        cloud = (out["cldfra_dp"], out["cldfra_sh"], out["qc_kf"],
                 out["qi_kf"])
    else:
        dummy = cp.empty((1,), dtype=DTYPE)
        cloud = (dummy, dummy, dummy, dummy)
    table = load_kf_table()
    device_table = _device_table()
    if type(trigger) is not int or trigger not in (1, 2, 3):
        raise ValueError("KF trigger must be integer 1, 2 or 3")
    if trigger == 2 and tpart is None:
        tpart = trigger2_perturbations(temperature, rqvften, dz)
    kernel = _kf_function("kf_column", trigger)
    ncol = ny * nx
    # The column arrays live in a global workspace sized to the threads in
    # flight; columns go in tiles of `tile`, each launched with the same
    # one-thread-one-column mapping shifted by `col0`.
    tile = kf_tile_columns(kernel, ncol)
    ws = cp.empty(kf_workspace_floats(nz, tile), dtype=DTYPE)
    for col0 in range(0, ncol, tile):
        span = min(tile, ncol - col0)
        blocks = (span + _TPB - 1) // _TPB
        kernel((blocks,), (_TPB,), (
            u, v, temperature, qv, pressure, rho, exner, dz, w0avg,
            *((tpart,) if trigger == 2 else ()), *device_table,
            out["rthcuten"], out["rqvcuten"], out["rqccuten"],
            out["rqicuten"], out["rqrcuten"], out["rqscuten"],
            out["raincv"], out["pratec"], out["nca_seconds"],
            out["cutop"], out["cubot"], out["shall"],
            out["cape_before"], out["cape_after"], out["timec"],
            out["updraft_mass_flux"], out["downdraft_mass_flux"],
            *cloud, ws,
            DTYPE(table.pressure_top), DTYPE(table.pressure_reciprocal),
            DTYPE(table.thetae_reciprocal), DTYPE(dx), DTYPE(dt),
            DTYPE(cudt_minutes), np.int32(phase_mode),
            np.int32(1 if f_qr else 0),
            np.int32(1 if cloud_diagnostics else 0),
            np.int32(nz), np.int32(ny), np.int32(nx), np.int32(col0)))
    del ws
    # Integer views for callers that index levels (0-based; -1 = none).
    convected = out["shall"] < DTYPE(1.5)
    out["triggered"] = convected.astype(cp.int32)
    out["shallow"] = (out["shall"] == DTYPE(1.0)).astype(cp.int32)
    out["cloud_base"] = cp.where(
        convected, out["cubot"].astype(cp.int32) - 1, -1).astype(cp.int32)
    out["cloud_top"] = cp.where(
        convected, out["cutop"].astype(cp.int32) - 1, -1).astype(cp.int32)
    out["rainc"] = out["raincv"]
    return out


def trigger2_perturbations(temperature, rqvften, dz, *, separate=False):
    """Native KF_eta_CPS moisture-advection stencil on a whole domain."""
    import cupy as cp
    if rqvften is None:
        raise ValueError("kfeta_trigger=2 needs the dycore RQVFTEN export")
    shape = temperature.shape
    for name, value in (("temperature", temperature), ("rqvften", rqvften), ("dz", dz)):
        if value.shape != shape or value.dtype != DTYPE or not value.flags.c_contiguous:
            raise ValueError(f"KF trigger 2 {name} must be contiguous float32 {shape}")
    nz, ny, nx = shape
    h = cp.empty(shape, dtype=DTYPE)
    v = cp.empty(shape, dtype=DTYPE) if separate else np.uintp(0)
    _kf_function("kf_trigger2_tpart", 2)(
        (((ny * nx) + 255) // 256,), (256,),
        (temperature, rqvften, dz, h, v, np.int32(nz), np.int32(ny), np.int32(nx)))
    return (h, v) if separate else h


def w0avg_factors(*, cudt_minutes: float, dt: float, stepcu: int,
                  adaptive: bool) -> tuple:
    """KF_eta_CPS's W0AVG weights (module_cu_kfeta.F:209-230), as binary32.

    Fixed step: ``W0AVGfctr = TST-1``, ``W0fctr = 1``, ``W0den = TST`` with
    ``TST = FLOAT(STEPCU*2)``.  Adaptive step:
    ``W0AVGfctr = 2*MAX(CUDT*60,DT) - DT``, ``W0fctr = DT``,
    ``W0den = 2*MAX(CUDT*60,DT)``.  Every operation is the Fortran's
    single-precision one, in its order.
    """
    f32 = np.float32
    if adaptive:
        window = max(f32(f32(cudt_minutes) * f32(60.0)), f32(dt))
        den = f32(f32(2.0) * f32(window))
        return f32(den - f32(dt)), f32(dt), den
    tst = f32(int(stepcu) * 2)
    return f32(tst - f32(1.0)), f32(1.0), tst


def advance_w0avg(w0avg, w, *, factors) -> None:
    """Advance ``w0avg`` in place from full-level ``w`` (nz+1, ny, nx).

    ``W0 = 0.5*(w(k)+w(k+1))`` and
    ``W0AVG = (W0AVG*W0AVGfctr + W0*W0fctr)/W0den``, in the no-FMA KF unit
    so the update rounds as the Fortran does.
    """
    nz = int(w0avg.shape[0])
    ncol = int(np.prod(w0avg.shape[1:]))
    if (tuple(w.shape) != (nz + 1,) + tuple(w0avg.shape[1:])
            or w.dtype != DTYPE or w0avg.dtype != DTYPE
            or not w.flags.c_contiguous or not w0avg.flags.c_contiguous):
        raise ValueError("advance_w0avg needs C-contiguous float32 w "
                         "(nz+1, ...) and w0avg (nz, ...)")
    fctr, w0fctr, den = factors
    total = nz * ncol
    threads = 256
    _kf_function("kf_w0avg_update")(
        ((total + threads - 1) // threads,), (threads,),
        (w, w0avg, DTYPE(fctr), DTYPE(w0fctr), DTYPE(den),
         np.int32(nz), np.int32(ncol)))


def validate_kf_outputs(
        values: tuple, active: int, status, *, describe=None) -> int:
    """Return finite-check flags for native KF feedback arrays.

    ``describe`` opts this site into :mod:`woof.core.health_ledger`; see
    :func:`woof.core.microphysics.validate_surface_diagnostics`.

    ``values`` follows the driver's first-invalid order: the six 3-D rates
    ``rthcuten``, ``rqvcuten``, ``rqccuten``, ``rqicuten``, ``rqrcuten``,
    ``rqscuten``, followed by the 2-D ``nca_seconds`` and ``pratec`` fields.
    Every pointer is valid even when its bit is inactive; the validation
    kernel does not load inactive fields.
    """
    import cupy as cp

    if len(values) != 8:
        raise ValueError(
            f"native KF validation requires 8 arrays, got {len(values)}")
    if (isinstance(active, (bool, np.bool_))
            or not isinstance(active, (int, np.integer))
            or int(active) < 0 or int(active) > 0xff):
        raise ValueError(
            f"KF validation active mask must fit eight bits, got {active!r}")
    if (not isinstance(status, cp.ndarray) or status.shape != (1,)
            or status.dtype != cp.uint32 or not status.flags.c_contiguous):
        raise ValueError(
            "KF validation status must be a C-contiguous uint32 device "
            "array with shape (1,)")

    shape_3d = values[0].shape
    shape_2d = values[6].shape
    if len(shape_3d) != 3 or len(shape_2d) != 2:
        raise ValueError("native KF outputs must retain 3-D/2-D shapes")
    if shape_2d != shape_3d[1:]:
        raise ValueError(
            "native KF surface outputs must match the 3-D horizontal shape")
    for index, value in enumerate(values[:6]):
        if (not isinstance(value, cp.ndarray) or value.shape != shape_3d
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            raise ValueError(
                f"native KF 3-D output {index} must be a C-contiguous "
                f"float32 CuPy array with shape {shape_3d}")
    for index, value in enumerate(values[6:], start=6):
        if (not isinstance(value, cp.ndarray) or value.shape != shape_2d
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            raise ValueError(
                f"native KF 2-D output {index} must be a C-contiguous "
                f"float32 CuPy array with shape {shape_2d}")

    status.fill(cp.uint32(0))
    count_3d = values[0].size
    count_2d = values[6].size
    blocks = (max(count_3d, count_2d) + _VALIDATION_TPB - 1) // _VALIDATION_TPB
    from woof.core.kernels import get_kernel

    get_kernel("kf_validation", "kf_validate_outputs")(
        (blocks,), (_VALIDATION_TPB,),
        values + (np.uint32(active), status,
                  np.int64(count_3d), np.int64(count_2d)))
    from woof.core import health_ledger

    return health_ledger.read_status(
        status, site="kain-fritsch", describe=describe)


# ---------------------------------------------------------------------------
# The Kain-Fritsch family: one adapter, one row per WRF scheme
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KFNcaContract:
    """WRF's per-scheme convective-timer rules, as data.

    Both schemes hold a triggered column's tendencies and rain rate for NCA
    seconds and count NCA down by DT on every model step; they differ in the
    two comparisons, and both differences are WRF's, not woof's:

    ``hold_inclusive``
        whether the scheme skips a column at exactly NCA = 0.5*DT.
        KF-eta skips while ``NCA .ge. 0.5*DT``
        (``module_cu_kfeta.F:410``); the old scheme while
        ``NCA .gt. 0.5*DT`` (``module_cu_kf.F:235``).
    ``expiry_steps``
        ``advance_ppt`` zeroes the held tendencies once
        ``NINT(NCA/DT) .le. expiry_steps``: 1 under KFETASCHEME
        (``module_physics_addtendc.F:2201``), 0 under KFSCHEME (:2155), so
        the old scheme applies its tendencies one step longer.
    """

    hold_inclusive: bool
    expiry_steps: int


#: Every WRF cumulus scheme this adapter runs, by ``cu_physics``.
KF_NCA_CONTRACTS = {
    1: KFNcaContract(hold_inclusive=True, expiry_steps=1),
    99: KFNcaContract(hold_inclusive=False, expiry_steps=0),
}
KF_CU_SCHEMES = tuple(KF_NCA_CONTRACTS)


def kf_nca_contract(cu_physics: int) -> KFNcaContract:
    """The timer rules for ``cu_physics``; KF-eta's for any other held
    result, which is the contract the driver applied before cu=99."""
    return KF_NCA_CONTRACTS.get(int(cu_physics), KF_NCA_CONTRACTS[1])


# ---------------------------------------------------------------------------
# The previous Kain-Fritsch scheme (cu_physics = 99, KFSCHEME)
# ---------------------------------------------------------------------------
#: ``KO_SLOTS`` in ``kernels/kf_old.cu``; tests/test_kf_old_wrf461_parity.py
#: re-derives it from the source and fails if either side moves alone.
KF_OLD_WS_SLOTS = 78


def kf_old_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` old-KF columns in flight.

    Every slot spans ``nz + 1`` levels because the subsidence omega array
    does (``OMG(KTS:KTE+1)``, ``module_cu_kf.F:421``).
    """
    blocks = (int(columns) + _TPB - 1) // _TPB
    return blocks * KF_OLD_WS_SLOTS * (int(nz) + 1) * _TPB


@dataclass(frozen=True)
class KFOldSpecies:
    """The four flags ``KFCPS`` reads to fold its hydrometeor tendencies.

    ``warm_rain`` comes from ``mp_init`` (Kessler only,
    ``module_physics_init.F:4459-4480``); ``f_qi``, ``f_qs`` and ``f_qr``
    are true when the scheme's Registry moist package allocates that
    species.  The old scheme branches on ``F_QI`` where KF-eta branches on
    ``F_QS`` (``module_cu_kf.F:1983-2023``), so KF-eta's
    :class:`KFPhaseMode` is not reused for it.
    """

    warm_rain: bool
    f_qi: bool
    f_qs: bool
    f_qr: bool


#: ``Registry/Registry.EM_COMMON`` moist packages for the microphysics
#: schemes woof runs (config.MP_PHYSICS_ACCEPTED): passiveqv (0) is
#: qv only; kesslerscheme (1) qv,qc,qr; the six-class schemes 6, 8, 9, 10,
#: 16, 18 and 28 carry qi and qs; p3_1category (50) qv,qc,qr,qi.
_KF_OLD_SPECIES = {
    0: KFOldSpecies(False, False, False, False),
    1: KFOldSpecies(True, False, False, True),
    2: KFOldSpecies(False, True, True, True),
    3: KFOldSpecies(False, False, False, True),
    5: KFOldSpecies(False, True, False, True),
    6: KFOldSpecies(False, True, True, True),
    8: KFOldSpecies(False, True, True, True),
    9: KFOldSpecies(False, True, True, True),
    10: KFOldSpecies(False, True, True, True),
    16: KFOldSpecies(False, True, True, True),
    18: KFOldSpecies(False, True, True, True),
    24: KFOldSpecies(False, True, True, True),
    27: KFOldSpecies(False, True, True, True),
    28: KFOldSpecies(False, True, True, True),
    29: KFOldSpecies(False, True, True, True),
    30: KFOldSpecies(False, True, True, True),
    50: KFOldSpecies(False, True, False, True),
    106: KFOldSpecies(False, True, True, True),
    # TEMPO's native moist package carries qc, qr, qi, qs and qg
    # (core/tempo.py:REQUIRED_STATE_FIELDS), on Thompson's phase contract.
    900: KFOldSpecies(False, True, True, True),
}


def kf_old_species_for_microphysics(mp_physics: int) -> KFOldSpecies:
    """The KFCPS species flags for ``mp_physics``.

    Refused for a scheme without a row: the flags decide which tendency
    arrays the scheme writes and which phase closure it applies, and a
    guessed set would feed frozen condensate to a state with nowhere to
    put it, or melt ice the microphysics carries.
    """
    from woof.core.column_schemes import COLUMN_SCHEMES

    row = COLUMN_SCHEMES.get(int(mp_physics))
    if row is not None:
        return KFOldSpecies(False, "qi" in row.species,
                            "qs" in row.species, "qr" in row.species)
    try:
        return _KF_OLD_SPECIES[int(mp_physics)]
    except KeyError:
        raise ValueError(
            f"cu_physics=99 has no species flags for mp_physics="
            f"{mp_physics}: module_cu_kf.F folds its tendencies by F_QI, "
            "F_QS, F_QR and warm_rain, and woof has no Registry moist "
            "package row for this scheme to read them from") from None


def kf_old_tendency_components(mp_physics: int) -> tuple[str, ...]:
    """The optional coupled categories cu=99 writes for ``mp_physics``."""
    species = kf_old_species_for_microphysics(mp_physics)
    return tuple(name for name, on in (("rqr", species.f_qr),
                                       ("rqi", species.f_qi),
                                       ("rqs", species.f_qs)) if on)


#: ``module_cu_kf.F``'s wrf_error_fatal sites, by the code kf_old.cu
#: writes into ``fatal``.
KF_OLD_FATAL_MESSAGES = {
    1: "module_cu_kf.F: problem with kf scheme:  qg = 0 at the surface "
       "(:1586)",
    2: "module_cu_kf.F: KAIN-FRITSCH -- MASS DOES NOT BALANCE IN KF SCHEME "
       "(:1615-1617, :1930)",
    3: "module_cu_kf.F: QVERR -- column water changed by more than 0.05 % "
       "(:1951)",
}


def launch_kf_old(u, v, w, temperature, qv, pressure, rho, dz, exner,
                  w0avg, nca, *, dx: float, dt: float, cudt_minutes: float,
                  adaptive: bool, stepcu: int, species: KFOldSpecies,
                  kf_edrates: bool = False):
    """Run ``KFCPS`` over a batch of FP32 device columns.

    ``w`` is the (nz+1, ny, nx) full-level vertical velocity; ``w0avg`` is
    updated IN PLACE by the scheme's own running mean (``module_cu_kf.F``
    :191-219), for every column including held ones, as KFCPS does.
    ``nca`` is the driver's timer, read only: a column with
    ``nca > 0.5*dt`` is skipped exactly as WRF skips it.  Returns the WRF
    outputs, ``nca_out`` (the new NCA where the column triggered, the old
    one elsewhere) and ``fatal`` (see :data:`KF_OLD_FATAL_MESSAGES`).
    """
    import cupy as cp

    shape = temperature.shape
    if len(shape) != 3:
        raise ValueError(f"temperature must have shape (nz,ny,nx), got {shape}")
    nz, ny, nx = shape
    kernel_capacity(nz)
    for name, array, want in (
            ("u", u, shape), ("v", v, shape), ("temperature", temperature, shape),
            ("qv", qv, shape), ("pressure", pressure, shape),
            ("rho", rho, shape), ("dz", dz, shape), ("exner", exner, shape),
            ("w0avg", w0avg, shape), ("w", w, (nz + 1, ny, nx)),
            ("nca", nca, (ny, nx))):
        if (not isinstance(array, cp.ndarray) or array.shape != want
                or array.dtype != DTYPE or not array.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {want}")
    for name, value in (("dx", dx), ("dt", dt)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"KF {name} must be positive and finite, got {value}")
    if int(stepcu) < 1:
        raise ValueError(f"KF stepcu must be >= 1, got {stepcu}")

    out = {name: cp.empty(shape, dtype=DTYPE) for name in
           ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
            "rqscuten")}
    out.update(raincv=cp.empty((ny, nx), dtype=DTYPE),
               pratec=cp.empty((ny, nx), dtype=DTYPE),
               nca_out=cp.empty((ny, nx), dtype=DTYPE),
               cu_act_flag=cp.empty((ny, nx), dtype=cp.int32),
               fatal=cp.empty((ny, nx), dtype=cp.int32))
    if kf_edrates:
        for name in ("udr_kf", "ddr_kf", "uer_kf", "der_kf"):
            out[name] = cp.empty(shape, dtype=DTYPE)
        out["timec_kf"] = cp.empty((ny, nx), dtype=DTYPE)
        edrates = (out["udr_kf"], out["ddr_kf"], out["uer_kf"],
                   out["der_kf"], out["timec_kf"])
    else:
        # The kernel touches these only when kf_edrates == 1.
        placeholder = cp.empty((1,), dtype=DTYPE)
        edrates = (placeholder,) * 5
    from woof.core.kernels import get_kernel

    kernel = get_kernel("kf_old", "kf_old_column")
    ncol = ny * nx
    tile = kf_tile_columns(kernel, ncol)
    ws = cp.empty(kf_old_workspace_floats(nz, tile), dtype=DTYPE)
    from woof.core.cu99_launch import launch_tiles
    launch_tiles(kernel, (
            u, v, w, temperature, qv, pressure, rho, dz, exner, w0avg, nca,
            out["rthcuten"], out["rqvcuten"], out["rqccuten"],
            out["rqrcuten"], out["rqicuten"], out["rqscuten"],
            out["raincv"], out["pratec"], out["nca_out"],
            out["cu_act_flag"], out["fatal"], *edrates, ws),
            (dt, dx, cudt_minutes),
            (int(bool(adaptive)), int(stepcu), int(species.warm_rain),
             int(species.f_qi), int(species.f_qs), int(species.f_qr),
             int(bool(kf_edrates)), nz, ny, nx),
            tile=tile, stream=cp.cuda.get_current_stream())
    del ws
    return out


def raise_kf_old_fatal(fatal) -> None:
    """Stop where WRF's wrf_error_fatal would, naming its message.

    One device reduction per due cumulus call, recorded in the active
    health ledger without a host read. The breakage it prevents:
    a column WRF would have aborted on (negative surface vapour after the
    moisture borrow, a mass-flux imbalance, a 0.05 % water error) carrying
    zero tendencies into a forecast that then reports success.
    """
    import cupy as cp

    from woof.core.health_ledger import read_status
    from woof.core.kernels import get_kernel

    status = cp.zeros(1, dtype=cp.uint32)
    kernel = get_kernel("kf_old", "kf_old_fatal_status")
    kernel(((fatal.size + _TPB - 1) // _TPB,), (_TPB,),
           (fatal, status, np.int32(fatal.size)))

    def describe(flags):
        messages = [message for code, message in KF_OLD_FATAL_MESSAGES.items()
                    if flags & (1 << (code - 1))]
        raise RuntimeError(
            f"cu_physics=99 (Kain-Fritsch, module_cu_kf.F) reached WRF's "
            f"wrf_error_fatal: {'; '.join(messages)}. WRF stops "
            "the model here, and so does woof")

    flags = read_status(status, site="kf-old-native-fatal", describe=describe)
    if flags:
        describe(flags)


class KainFritsch:
    """Stateful Kain-Fritsch callable for :class:`PhysicsDriver`.

    ``scheme`` is the WRF ``cu_physics`` value: 1 (KF-eta,
    ``kernels/kf.cu``) or 99 (the previous scheme, ``kernels/kf_old.cu``).
    One adapter serves both; what differs is data -- the kernel, the
    species fold and the NCA timer rules in :data:`KF_NCA_CONTRACTS`.

    The driver remains the sole cadence authority.  Each call returns rates
    to hold through the next STEPCU interval and the matching RAINC increment.
    """

    def __init__(self, scheme: int = 1, *, kf_edrates: bool = False):
        if int(scheme) not in KF_NCA_CONTRACTS:
            raise ValueError(
                f"KainFritsch runs cu_physics {KF_CU_SCHEMES}, not {scheme}")
        self.scheme = int(scheme)
        if kf_edrates and self.scheme != 99:
            raise ValueError("kf_edrates requires the cu_physics=99 output kernel")
        self.kf_edrates = bool(kf_edrates)
        for name in ("udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf"):
            setattr(self, name, None)
        self.w0avg = None
        self._history_state = None
        self._history_time = None
        self._driver_ref = None

    def bind_driver(self, driver) -> None:
        """The driver's NCA timer, for cu=99's per-column hold test.

        A weak reference: the driver owns this callable, so a strong one
        would be a cycle.  Only the old scheme reads it -- KF-eta's kernel
        computes every column and the driver masks held ones.
        """
        import weakref

        self._driver_ref = weakref.ref(driver)

    def ensure_trigger_history(self, state):
        """Materialise ``w0avg`` NOW rather than on the first due call.

        ``w0avg`` is a CARRIER (``restart.CUMULUS_CALLABLE_ARRAYS``), so it
        belongs to the set a tiled or decomposed run ships between buffers.
        Allocating it lazily means an inventory taken at construction does not
        contain it and an inventory taken after the first cumulus event does:
        the carrier set CHANGES IDENTITY MID-RUN.  A seam plan is built once
        from the inventory and cannot follow that, so
        ``tilestream.multigpu.MultiGPUDomain.refresh_arrays`` aborts the run
        the moment cumulus first fires -- at cudt=5 min and dt=15 s that is
        step 20, twenty steps after anything could still be called a
        start-up failure.  ``tilestream.driver.make_physics_tile_state``
        works around it by throwing a warm-up step away; the array is
        (nz, ny, nx) zeros and there is nothing to warm up.

        Idempotent, and it re-allocates on a state or shape change exactly as
        the lazy path did, so a driver moved to a new state still gets a
        fresh mean rather than the previous domain's memory.
        """
        import cupy as cp

        shape = (int(state.w.shape[0]) - 1,) + tuple(
            int(n) for n in state.w.shape[1:])
        if (self.w0avg is None or self._history_state is not state
                or tuple(self.w0avg.shape) != shape):
            self.w0avg = cp.zeros(shape, dtype=DTYPE)
            self._history_state = state
            self._history_time = None
            if self.kf_edrates:
                for name in ("udr_kf", "ddr_kf", "uer_kf", "der_kf"):
                    setattr(self, name, cp.zeros(shape, dtype=DTYPE))
                self.timec_kf = cp.zeros(shape[1:], dtype=DTYPE)
        return self.w0avg

    def update_trigger_history(self, *, state, cfg):
        """Advance WRF's running-mean trigger velocity for one due call.

        WRF's cumulus driver early-returns on non-due steps
        (``module_cumulus_driver.F:830-864``, ``RETURN`` at :863), so the
        W0AVG loop at the top of ``KF_eta_CPS``
        (``module_cu_kfeta.F:218-251``) executes exactly once per due
        event.  Fixed step: weight ``1/TST``, ``TST = 2*STEPCU`` from the
        model clock.  Adaptive step (``ADAPT_STEP_FLAG``): the DT-weighted
        form over a ``2*MAX(CUDT*60, DT)`` window.  The physics driver
        invokes this once per due call, immediately before the column
        scheme consumes the mean and before the per-column NCA skip test,
        so held columns still refresh their trigger memory.  The update
        runs in the no-FMA KF unit (:func:`advance_w0avg`), so it rounds
        as the Fortran does.
        """
        if self.scheme == 99:
            # KFCPS advances W0AVG itself, inside the scheme call, for every
            # column (module_cu_kf.F:191-219), and kf_old.cu transcribes that
            # loop with the scheme; advancing it here as well would count
            # the sample twice.
            self.ensure_trigger_history(state)
            self._history_time = float(state.elapsed_seconds)
            return
        self.ensure_trigger_history(state)
        clock_dt = _model_clock_dt(cfg)
        stepcu = (1 if cfg.cudt_minutes <= 0.0 else
                  max(int(np.floor(
                      cfg.cudt_minutes * 60.0 / clock_dt + 0.5)), 1))
        factors = w0avg_factors(
            cudt_minutes=float(cfg.cudt_minutes), dt=clock_dt,
            stepcu=stepcu,
            adaptive=bool(getattr(cfg, "use_adaptive_time_step", False)))
        advance_w0avg(self.w0avg, state.w, factors=factors)
        self._history_time = float(state.elapsed_seconds)

    def __call__(self, *, atmosphere, fields, state, cfg):
        # Direct callers still receive the first WRF history update.  A driver
        # call at the same elapsed time has already run the hook and must not
        # double-count that sample.
        if (self.w0avg is None or self._history_state is not state
                or self._history_time != float(state.elapsed_seconds)):
            self.update_trigger_history(state=state, cfg=cfg)
        if self.scheme == 99:
            return self._call_kfcps(atmosphere=atmosphere, state=state,
                                    cfg=cfg)
        # WRF hands the scheme its model DT (module_cumulus_driver.F:1028),
        # which is the outer clock when the case integrates internal
        # substeps, and the namelist CUDT in minutes (cudt_pass, :766-769):
        # the shallow-convection NCA is CUDT*60. (module_cu_kfeta.F:2573),
        # which is 0 at cudt = 0, not one model step.
        clock_dt = _model_clock_dt(cfg)
        # WRF's physics density (1+qv)/alt, as phy_prep hands it to the
        # cumulus driver (module_big_step_utilities_em.F:4856): KF's layer
        # masses are DP = RHO*G*DZ8W (module_cu_kfeta.F:779).
        phase_mode = kf_phase_mode_for_microphysics(cfg.mp_physics)
        result = launch_kf(
            atmosphere["u"], atmosphere["v"], atmosphere["temperature"],
            atmosphere["qv"], atmosphere["pressure"], atmosphere["rho"],
            atmosphere["exner"], atmosphere["dz"], self.w0avg,
            dx=cfg.dx, dt=clock_dt, cudt_minutes=float(cfg.cudt_minutes),
            phase_mode=phase_mode,
            trigger=cfg.kfeta_trigger, rqvften=getattr(state, "rqvften", None),
            f_qr=kf_has_rain_for_microphysics(cfg.mp_physics),
            cloud_diagnostics=cfg.cu_rad_feedback)
        from woof.core.physics import _NativeKFCumulusResult

        # PRATEC = PPTFLX*(1-FBFRC)/DXSQ in mm s-1 and NCA come straight
        # from the scheme (module_cu_kfeta.F:2570-2573, 2642); the driver
        # forms RAINCV = DT*PRATEC itself, as WRF does.
        separate_ice = phase_mode == KFPhaseMode.SEPARATE_ICE_SNOW
        separate_snow = phase_mode in (
            KFPhaseMode.SEPARATE_SNOW, KFPhaseMode.SEPARATE_ICE_SNOW)
        nca = result["nca_seconds"]
        pratec = result["pratec"]
        return _NativeKFCumulusResult(
            owner=self,
            rthcuten=result["rthcuten"], rqvcuten=result["rqvcuten"],
            rqccuten=result["rqccuten"],
            rqicuten=(result["rqicuten"] if separate_ice else None),
            rqrcuten=result["rqrcuten"],
            rqscuten=(result["rqscuten"] if separate_snow else None),
            rainc=result["rainc"], nca_seconds=nca, pratec=pratec,
            cloud_base=(result['cubot'] if 99 in (cfg.ra_lw_physics,cfg.ra_sw_physics) else None),
            cloud_top=(result['cutop'] if 99 in (cfg.ra_lw_physics,cfg.ra_sw_physics) else None),
            cloud_fields=({name:result[source] for name,source in
                (("qc_cu","qc_kf"),("qi_cu","qi_kf"),
                 ("cldfra_dp","cldfra_dp"),("cldfra_sh","cldfra_sh"))}
                if cfg.cu_rad_feedback else None))

    def _call_kfcps(self, *, atmosphere, state, cfg):
        """One due call of the previous scheme (cu_physics = 99).

        The inputs are the ones module_cumulus_driver.F:958-989 hands
        KFCPS: the phy_prep temperature, hydrostatic pressure, moist
        density (1+qv)/alt, layer depth and Exner function, the mass-point
        winds, the full-level W, and the model DT.  KFCPS returns WRF's
        own PRATEC and NCA; the driver applies the hold with this scheme's
        timer rules (:data:`KF_NCA_CONTRACTS`).
        """
        import cupy as cp

        from woof.core.cu99_launch import plan_step
        plan = plan_step(elapsed=state.elapsed_seconds, dt=cfg.dt,
            clock_dt=getattr(cfg, "clock_dt", 0.0),
            cudt_minutes=cfg.cudt_minutes)
        clock_dt, stepcu = plan.clock_dt, plan.column_stepcu
        driver = None if self._driver_ref is None else self._driver_ref()
        nca = getattr(driver, "cu_nca", None)
        ny, nx = atmosphere["temperature"].shape[1:]
        if nca is None or tuple(nca.shape) != (ny, nx):
            # A direct caller with no driver: every column is due, which is
            # kfinit's cold start (NCA = -100, module_cu_kf.F:2650).
            nca = cp.full((ny, nx), DTYPE(-100.0), dtype=DTYPE)
        species = kf_old_species_for_microphysics(cfg.mp_physics)
        result = launch_kf_old(
            atmosphere["u"], atmosphere["v"],
            cp.ascontiguousarray(state.w, dtype=DTYPE),
            atmosphere["temperature"], atmosphere["qv"],
            atmosphere["pressure"], atmosphere["rho"], atmosphere["dz"],
            atmosphere["exner"], self.w0avg, nca,
            dx=cfg.dx, dt=clock_dt, cudt_minutes=float(cfg.cudt_minutes),
            adaptive=bool(getattr(cfg, "use_adaptive_time_step", False)),
            stepcu=stepcu, species=species, kf_edrates=self.kf_edrates)
        raise_kf_old_fatal(result["fatal"])
        if self.kf_edrates:
            # KFCPS leaves diagnostics untouched on held columns (:235-255).
            eligible = nca <= DTYPE(0.5) * DTYPE(clock_dt)
            for name in ("udr_kf", "ddr_kf", "uer_kf", "der_kf", "timec_kf"):
                mask = eligible if name == "timec_kf" else eligible[None]
                cp.copyto(getattr(self, name), result[name], where=mask)
        from woof.core.physics import _NativeKFCumulusResult

        return _NativeKFCumulusResult(
            owner=self,
            rthcuten=result["rthcuten"], rqvcuten=result["rqvcuten"],
            rqccuten=result["rqccuten"],
            rqicuten=result["rqicuten"] if species.f_qi else None,
            rqrcuten=result["rqrcuten"] if species.f_qr else None,
            rqscuten=result["rqscuten"] if species.f_qs else None,
            rainc=result["raincv"], nca_seconds=result["nca_out"],
            pratec=result["pratec"])


__all__ = ["KFPhaseMode", "KFTable", "KainFritsch", "KFWS_SLOTS",
           "KF_TILE_BLOCKS_PER_SM", "advance_w0avg", "kernel_capacity",
           "kf_has_rain_for_microphysics",
           "kf_phase_mode_for_microphysics", "kf_tile_columns",
           "kf_workspace_floats", "launch_kf", "load_kf_table",
'validate_kf_outputs', 'w0avg_factors', 'KFNcaContract', 'KF_NCA_CONTRACTS', 'KF_CU_SCHEMES', 'kf_nca_contract', 'KF_OLD_WS_SLOTS', 'kf_old_workspace_floats', 'KFOldSpecies', 'kf_old_species_for_microphysics', 'kf_old_tendency_components', 'KF_OLD_FATAL_MESSAGES', 'launch_kf_old', 'raise_kf_old_fatal']


# KF-CuP uses the existing KF table cache. Packed fields follow the WRF
# KF_cup_CPS call, with outputs retained across STEPCU calls and NCA holds.
from woof.core.kfcup_contract import (
    KFCUP_LEVEL_INPUTS, KFCUP_SURFACE_INPUTS, KFCUP_HELD_SCALARS,
    KFCUP_HELD_INTS, KFCUP_LEVEL_OUTPUTS, KFCUP_WS_SLOTS,
    KFCUP_VERTICAL_LEVEL_BOUNDS, KFCUP_PARAMETER_DEFAULTS, KFCUP_STATE_ROWS,
)
KFCUP_METADATA = {
    'selector':10, 'callable':'woof.core.kf.KFCumulusPotential',
    'kernel_modules':('kfcup',), 'cadence':'held-model-clock',
    'state':KFCUP_STATE_ROWS, 'parameters':KFCUP_PARAMETER_DEFAULTS,
    'radiation_override':{
        'cloud_fraction':'cldfra_cup', 'liquid_floor':1.0e-3,
        'liquid_floor_weight':'cldfra_cup', 'ice_action':'preserve',
        'shallow_floor_parameter':'shallowcu_forced_ra', 'shallow_floor':0.36,
        'scheme_source':'phys/module_cu_kfcup.F',
        'radiation_source':'phys/module_radiation_driver.F:1510-1557'},
}



def kfcup_workspace_floats(nz: int, columns: int, numbins: int) -> int:
    """One warp-interleaved workspace, bounded by the launched tile size."""
    return ((int(columns)+31)//32)*32*(KFCUP_WS_SLOTS*(int(nz)+1)
                                      +int(numbins)**2+2*int(numbins))


def launch_kfcup_columns(buffers, *, dt, dx, ntst, numbins, thbinsize, rbinsize,
                         mindeepfreq, minshallowfreq, flags, workspace=None, status=None):
    """Run the complete CuP column on device, retaining all output words."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    lev=buffers['lev_in']
    if lev.ndim!=3 or lev.shape[0]!=len(KFCUP_LEVEL_INPUTS):
        raise ValueError('KF-CuP packed level inputs require (12,nz,ncol)')
    _,nz,ncol=lev.shape
    if not KFCUP_VERTICAL_LEVEL_BOUNDS[0]<=nz<=KFCUP_VERTICAL_LEVEL_BOUNDS[1]:
        raise ValueError(f'KF-CuP supports 8 through 128 levels, got {nz}')
    shapes={'lev_in':lev.shape,'sca_in':(len(KFCUP_SURFACE_INPUTS),ncol),
            'kpbl':(ncol,),'held':(len(KFCUP_HELD_SCALARS),ncol),
            'iheld':(len(KFCUP_HELD_INTS),ncol),
            'lev_out':(len(KFCUP_LEVEL_OUTPUTS),nz,ncol)}
    for key,shape in shapes.items():
        a=buffers[key]; dtype=np.int32 if key in ('kpbl','iheld') else np.float32
        if not isinstance(a,cp.ndarray) or a.dtype!=dtype or a.shape!=shape or not a.flags.c_contiguous:
            raise ValueError(f'KF-CuP {key} must be contiguous {dtype} with shape {shape}')
    if isinstance(numbins,(bool,np.bool_)) or int(numbins)!=numbins or not 1<=numbins<=46340:
        raise ValueError('KF-CuP numbins must be an integer in [1,46340] to keep the PDF square index within int32')
    for key,x in [('dt',dt),('dx',dx),('thbinsize',thbinsize),('rbinsize',rbinsize)]:
        if not np.isfinite(x) or x<=0:
            raise ValueError(f'KF-CuP {key} must be positive and finite')
    for key,x in [('mindeepfreq',mindeepfreq),('minshallowfreq',minshallowfreq)]:
        if not np.isfinite(x) or not 0<=x<=1:
            raise ValueError(f'KF-CuP {key} must be finite in [0,1]')
    if ntst<1 or flags<0 or flags>15:
        raise ValueError('KF-CuP needs positive STEPCU and a four-bit phase mask')
    table=load_kf_table()
    kernel=get_kernel('kfcup','kfcup_column')
    # At most one 32-thread block per SM is in this allocation. A larger
    # domain reuses the same workspace over disjoint column tiles.
    tile=min(ncol,32*int(cp.cuda.Device().attributes['MultiProcessorCount']))
    if tile==0:
        return
    needed=kfcup_workspace_floats(nz,tile,numbins)
    ws=cp.empty(needed,dtype=cp.float32) if workspace is None else workspace
    if ws.dtype != cp.float32 or ws.size < needed:
        raise ValueError("KF-CuP workspace is too small or not float32")
    status=cp.zeros(1,dtype=cp.int32) if status is None else status
    status.fill(0)
    args=tuple(buffers[k] for k in ('lev_in','sca_in','kpbl','held','iheld','lev_out'))
    args+=_device_table()+(DTYPE(table.pressure_top),DTYPE(table.pressure_reciprocal),
        DTYPE(table.thetae_reciprocal),DTYPE(dt),DTYPE(dx),np.int32(ntst),
        np.int32(numbins),DTYPE(thbinsize),DTYPE(rbinsize),DTYPE(mindeepfreq),
        DTYPE(minshallowfreq),np.int32(flags),ws,status,np.int32(nz),np.int32(ncol))
    for col0 in range(0,ncol,tile):
        kernel(((min(tile,ncol-col0)+31)//32,),(32,),args+(np.int32(col0),))
    from woof.core import health_ledger
    failure=health_ledger.read_status(status, site='kfcup', describe=_describe_kfcup_status)
    if failure:
        _describe_kfcup_status(failure)


def _describe_kfcup_status(failure):
    causes=[]
    if failure&1: causes.append('negative surface QG after compensating subsidence')
    if failure&2: causes.append('unbalanced mass-flux budget')
    if failure&4: causes.append('temperature lookup outside the 250 by 220 KF table')
    raise RuntimeError('KF-CuP refused output: '+', '.join(causes))


class KFCumulusPotential(KainFritsch):
    """WRF cu_physics=10, on the shared KF table and driver clock path."""
    held_state_rows = KFCUP_STATE_ROWS
    def __init__(self):
        super().__init__()
        self.cup_held=None
        self.cup_iheld=None
        self.cup_levels=None
        self._ws=None
        self._cup_status=None
        self._driver=None

    def ensure_trigger_history(self,state):
        import cupy as cp
        changed=self._history_state is not state
        avg=super().ensure_trigger_history(state)
        nz,ny,nx=avg.shape
        if changed or self.cup_levels is None or self.cup_levels.shape[1:]!=(nz,ny,nx):
            self.cup_held=cp.zeros((len(KFCUP_HELD_SCALARS),ny,nx),dtype=DTYPE)
            self.cup_iheld=cp.zeros((len(KFCUP_HELD_INTS),ny,nx),dtype=cp.int32)
            self.cup_levels=cp.zeros((len(KFCUP_LEVEL_OUTPUTS),nz,ny,nx),dtype=DTYPE)
            self.cup_held[0]=-100.0
            self.cup_held[4]=2.0
        return avg

    def restart_array_contract(self, state):
        nz,ny,nx=state.p.shape
        return {
            'cup_held': ((len(KFCUP_HELD_SCALARS),ny,nx), np.dtype('float32')),
            'cup_iheld': ((len(KFCUP_HELD_INTS),ny,nx), np.dtype('int32')),
            'cup_levels': ((len(KFCUP_LEVEL_OUTPUTS),nz,ny,nx), np.dtype('float32')),
        }

    def bind_driver(self,driver):
        self._driver=driver

    def update_trigger_history(self,*,state,cfg):
        from woof.core.kernels import get_kernel
        avg=self.ensure_trigger_history(state)
        stepcu=max(int(np.floor(cfg.cudt_minutes*60/_model_clock_dt(cfg)+0.5)),1)
        get_kernel('kfcup','kfcup_trigger_history')(((avg.size+255)//256,),(256,),
            (avg,state.w,DTYPE(2*stepcu),np.int32(avg.shape[1]*avg.shape[2]),np.int32(avg.shape[0])))
        self._history_time=float(state.elapsed_seconds)

    def __call__(self,*,atmosphere,fields,state,cfg):
        import cupy as cp
        from woof.core.kernels import get_kernel
        if self.w0avg is None or self._history_state is not state or self._history_time!=float(state.elapsed_seconds):
            self.update_trigger_history(state=state,cfg=cfg)
        nz,ny,nx=self.w0avg.shape
        dt=_model_clock_dt(cfg)
        zw=atmosphere['z_interface']
        z=atmosphere.get('z')
        if z is None:
            z=DTYPE(0.5)*(zw[:-1]+zw[1:])
        mapping=dict(u=atmosphere['u'],v=atmosphere['v'],th=atmosphere['theta'],
            t=atmosphere['temperature'],dz8w=atmosphere['dz'],p=atmosphere['pressure'],
            pi=atmosphere['exner'],rho=atmosphere['rho'],qv=atmosphere['qv'],z=z,
            z_at_w=zw[:-1],w0avg=self.w0avg)
        surface=dict(ht=fields.get('ht',zw[0]),hfx=fields['hfx'],qfx=fields['qfx'],
                     xland=fields['xland'],pblh=fields['pblh'])
        lev_in=cp.empty((12,nz,ny*nx),dtype=DTYPE)
        sca_in=cp.empty((5,ny*nx),dtype=DTYPE)
        packed_inputs=tuple(cp.ascontiguousarray(mapping[k],dtype=DTYPE)
                            for k in KFCUP_LEVEL_INPUTS)
        packed_surface=tuple(cp.ascontiguousarray(surface[k],dtype=DTYPE)
                             for k in KFCUP_SURFACE_INPUTS)
        get_kernel('kfcup','kfcup_pack')(((nz*ny*nx+255)//256,),(256,),
            (lev_in,sca_in,*packed_inputs,*packed_surface,np.int32(nz*ny*nx),np.int32(ny*nx)))
        b={'lev_in':lev_in,
           'sca_in':sca_in,
           'kpbl':cp.ascontiguousarray(fields['kpbl'].reshape(-1),dtype=cp.int32),
           'held':self.cup_held.reshape(len(KFCUP_HELD_SCALARS),-1),
           'iheld':self.cup_iheld.reshape(len(KFCUP_HELD_INTS),-1),
           'lev_out':self.cup_levels.reshape(len(KFCUP_LEVEL_OUTPUTS),nz,-1)}
        if self._driver is not None:
            self.cup_held[0]=self._driver.cu_nca
            self.cup_held[1]=self._driver.cu_raincv
            for i,key in enumerate(KFCUP_LEVEL_OUTPUTS[:6]):
                self.cup_levels[i]=self._driver.cu_rates[key]
        if 'cldfra_cup' in fields:
            self.cup_levels[KFCUP_LEVEL_OUTPUTS.index('cldfra_cup')]=fields['cldfra_cup']
        bins=getattr(cfg,'numbins',1)
        tile=min(ny*nx,32*int(cp.cuda.Device().attributes['MultiProcessorCount']))
        needed=kfcup_workspace_floats(nz,tile,bins)
        if self._ws is None or self._ws.size < needed:
            self._ws=cp.empty(needed,dtype=cp.float32)
        if self._cup_status is None:
            self._cup_status=cp.zeros(1,dtype=cp.int32)
        # Species availability is the allocated state, which Registry fixes.
        flags=(1 if cfg.mp_physics in (0,1) else 0)
        flags|=2 if getattr(state,'qr',None) is not None else 0
        flags|=4 if getattr(state,'qi',None) is not None else 0
        flags|=8 if getattr(state,'qs',None) is not None else 0
        launch_kfcup_columns(b,dt=dt,dx=cfg.dx,
            ntst=max(int(np.floor(cfg.cudt_minutes*60/dt+0.5)),1),
            flags=flags,workspace=self._ws,status=self._cup_status,**{k:getattr(cfg,k,v) for k,v in KFCUP_PARAMETER_DEFAULTS.items() if k!='shallowcu_forced_ra'})
        from woof.core.physics import _NativeKFCumulusResult
        outputs={key:self.cup_levels[i].reshape(nz,ny,nx) for i,key in enumerate(KFCUP_LEVEL_OUTPUTS)}
        # Publish only to held driver fields. Radiation never receives a
        # write into prognostic condensate from this scheme.
        for key in KFCUP_LEVEL_OUTPUTS[6:]:
            if key in fields: fields[key][...]=outputs[key]
        for i,key in enumerate(KFCUP_HELD_SCALARS[2:],start=2):
            if key in fields: fields[key][...]=self.cup_held[i].reshape(ny,nx)
        for i,key in enumerate(KFCUP_HELD_INTS):
            if key in fields: fields[key][...]=self.cup_iheld[i].reshape(ny,nx)
        raincv=self.cup_held[1].reshape(ny,nx)
        return _NativeKFCumulusResult(owner=self,rthcuten=outputs['rthcuten'],rqvcuten=outputs['rqvcuten'],
            rqccuten=outputs['rqccuten'],rqrcuten=outputs['rqrcuten'] if flags&2 else None,
            rqicuten=outputs['rqicuten'] if flags&4 else None,
            rqscuten=outputs['rqscuten'] if flags&8 else None,
            nca_seconds=self.cup_held[0].reshape(ny,nx),pratec=raincv/DTYPE(dt),rainc=raincv,raincv=raincv)


    def advance_cloud_clock(self, *, nca, dt, fields):
        """WRF KF-CuP cloud clear on the post-decrement hold clock."""
        from woof.core.kernels import get_kernel
        cf=fields['cldfra_cup']
        get_kernel('kfcup','kfcup_cloud_clock')(((nca.size+255)//256,),(256,),
            (nca,cf,DTYPE(dt),np.int32(nca.size),np.int32(cf.shape[0])))
        # Packed storage is the producer's restart-carried copy of the same
        # held cloud. Keep it current before the next eligible column call.
        self.cup_levels[KFCUP_LEVEL_OUTPUTS.index('cldfra_cup')]=cf

    def advance_rain_clock(self, *, rainc, pending_rainbl, surface_raincv,
                           nca, raincv, expiring, dt):
        """Accumulate the native per-step increment before the hold countdown."""
        from woof.core.kernels import get_kernel
        get_kernel('kfcup','kfcup_advance_clock')(((nca.size+255)//256,),(256,),
            (rainc,pending_rainbl,surface_raincv if surface_raincv is not None else rainc,
             nca,expiring,raincv,DTYPE(dt),np.int32(surface_raincv is not None),np.int32(nca.size)))



__all__ += ["KFCumulusPotential", "KFCUP_METADATA", "KFCUP_STATE_ROWS",
            "KFCUP_PARAMETER_DEFAULTS", "KFCUP_VERTICAL_LEVEL_BOUNDS",
            "KFCUP_WS_SLOTS", "kfcup_workspace_floats", "launch_kfcup_columns"]

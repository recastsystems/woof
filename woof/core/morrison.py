"""Morrison two-moment microphysics (WRF ``mp_physics=10``).

``kernels/morrison.cu`` is a literal transcription of WRF v4.6.1
``phys/module_mp_morr_two_moment.F`` (driver :563-925, MICRO :929-4062,
POLYSVP, the module GAMMA and refl10cm_hm), graded bitwise against WRF's
own Fortran by ``tools/morrison_wrf461_oracle``.  Three launches per call:

* ``mo_process``: one thread per cell, the first K loop (:1275-3334),
  float tendencies into the WORK stack and LTRUE per column;
* ``mo_sediment_64``/``_256``: five warps per 32 columns, one per
  hydrometeor category, WRF's split-step fallout (:3337-3667) and the
  driver's surface accumulation;
* ``mo_finalize``: one thread per cell, the final K loop (:3669-4054)
  and the driver's write-back.

The public launcher accepts contiguous FP32 ``(nz, ny, nx)`` fields and
updates potential temperature, six water categories and their moments in
place.  :func:`apply` prepares WRF's microphysics inputs from
:class:`~woof.core.state.DomainState`.

The owner-ruled divergence (PARITY-GAPS row 37: the deposition-nucleation
vapour cap and the wide re-evaluation of a non-finite freezing rate) is
compiled in by ``MORR_ROW37``; ``MORR_ROW37_OFF`` builds WRF's arithmetic
alone for the oracle gate. The forecast build also prevents negative
sedimentation mass after a source tendency cancels its category in FP32;
otherwise inherited fall speed exports negative precipitation. The raw WRF
oracle build retains the signed residue.
"""

from __future__ import annotations

import numpy as np

import cupy as cp

from woof.config import RunConfig
from woof.core import constants as c
from woof.core.kernels import get_kernel
from woof.core.morrison_constants import (NWORK, SCRATCH_INT_SURFACE_SLOTS,
                                           SCRATCH_VOLUME_SLOTS,
                                           SCRATCH_WORK_SLOT, WORK_T3D,
                                           rimed_ice_constants,
                                           work_chunk_columns,
                                           work_slot_shape)
from woof.core.state import DTYPE, DomainState

_CELL_TPB = 64
_COLUMN_TPB = 32
#: Sedimentation block: one warp per hydrometeor category over the same
#: _COLUMN_TPB columns (kernels/morrison.cu, mo_sediment_impl).
_SEDIMENT_TPB = 5 * _COLUMN_TPB
_SHALLOW_KMAX = 64
_KMAX = 256
VERTICAL_LEVEL_BOUNDS = (2, _KMAX)
# NWORK / WORK_T3D and the scratch-slot lists live in morrison_constants
# (CPU-only) for the VRAM registry; re-exported here.
__all__ = ["NWORK", "WORK_T3D", "SCRATCH_VOLUME_SLOTS",
           "SCRATCH_WORK_SLOT", "SCRATCH_INT_SURFACE_SLOTS",
           "work_chunk_columns", "work_slot_shape", "bound_snow_ratio",
           "LAUNCHERS", "launch_exner", "launch_morrison",
           "launch_refl10cm_wrf", "apply"]
#: Every function that launches a kernel of this unit.
LAUNCHERS = ("launch_morrison", "launch_refl10cm_wrf", "launch_exner")


def bound_snow_ratio(sr) -> None:
    """Cap independent FP32 frozen/total fallout sums at one.

    The frozen and total sums use different addition orders, so their
    quotient can round above one. The land surface needs a fraction.
    The former lower floor is retired: the forecast sedimentation kernel
    now prevents negative mass, rather than hiding negative fallout here.
    Nonfinite or negative values remain visible to the physics validator.
    The raw WRF oracle build retains its original signed arithmetic.
    """
    cp.minimum(sr, DTYPE(1.0), out=sr)


# theta = thb + thp and dz8w = z_w(k+1) - z_w(k), as the separate CuPy
# ufuncs rounded them (their FTZ mode included).  The Exner function is
# WRF's phy_prep (p/p1000mb)**rcp with WOOF's own powf: launch_exner.
_prepare_fields = cp.ElementwiseKernel(
    "raw float32 thb, raw float32 phb, raw float32 thp, raw float32 php, "
    "int32 ncol, int32 thb_full, int32 phb_full, float32 gravity",
    "float32 theta, float32 dz",
    r"""
    size_t k = i / ncol;
    size_t below = i;
    size_t above = i + ncol;
    theta = __fadd_rn(thb[thb_full ? i : k], thp[i]);
    float zb = __fdiv_rn(__fadd_rn(phb[phb_full ? below : k], php[below]), gravity);
    float za = __fdiv_rn(__fadd_rn(phb[phb_full ? above : k + 1], php[above]), gravity);
    dz = __fsub_rn(za, zb);
    """,
    # FTZ comes from CuPy's own -ftz=true; a second spelling is refused
    # by NVRTC 12 (see woof/core/mynn_pbl_gpu.py).
    "morrison_prepare_fields")


def _check(name, value, shape, dtype=DTYPE):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must be {np.dtype(dtype)}, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def launch_exner(pressure, pii) -> None:
    """``pii = (p / p1000mb)**rcp`` with WOOF's own powf (WRF phy_prep)."""
    _check("pressure", pressure, pressure.shape)
    _check("pii", pii, pressure.shape)
    n = pressure.size
    get_kernel("morrison", "mo_exner")(
        ((n + _CELL_TPB - 1) // _CELL_TPB,), (_CELL_TPB,),
        (pressure, pii, DTYPE(c.P0), DTYPE(c.RCP), np.int64(n)))


def launch_morrison(theta, qv, qc, qr, qi, qs, qg,
                    nc, nr, ni, ns, ng, rho, pii, pressure, dz,
                    rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, sr, dt: float,
                    *, effc=None, effr=None, effi=None, effs=None,
                    qrcuten=None, qscuten=None, qicuten=None,
                    morr_rimed_ice: int = 1,
                    columns_per_call=None, t1d_out=None,
                    _work_scratch=None, _ltrue_scratch=None):
    """Launch one WRF Morrison call over an FP32 column batch.

    Atmospheric and hydrometeor arrays are ``(nz, ny, nx)``; accumulated,
    per-call and frozen-fraction precipitation fields are ``(ny, nx)``.
    Every prognostic argument through ``ng`` and every precipitation field
    is updated in place.  ``rho`` receives the scheme's own air density
    (PRES/(R*T), :1325): WRF's wrapper takes a density argument but never
    reads it.  ``nc`` carries no input (INUM=1: the wrapper hands the
    scheme a placeholder zero, :812-815) and returns the scheme's
    prescribed NDCNST droplet number.  ``morr_rimed_ice`` follows WRF's
    Registry option: 1 = hail (default), 0 = graupel.  ``qrcuten``/
    ``qscuten``/``qicuten`` are the optional all-or-none cumulus mass rates
    that seed rain, snow and ice number (:1327-1343).

    ``columns_per_call`` splits the ``ny * nx`` columns into calls of at
    most that many columns (WRF's columns are independent, so the words
    do not depend on it); ``None`` is one call over every column.  The
    WORK stack holds NWORK (nz, columns) planes for one call's columns.
    ``t1d_out``, an (nz, ny, nx) array, receives the post-call
    temperature (WRF's T1D, WORK plane ``WORK_T3D``) of every column.

    Returns the WORK stack of the last call; for a one-call launch it is
    shaped ``(NWORK * nz, ny, nx)`` and its plane ``WORK_T3D`` holds T1D.
    """
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"Morrison fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if nz > maximum:
        raise ValueError(f"nz={nz} exceeds MORR_KMAX={maximum}")
    if nz < minimum:
        raise ValueError(f"Morrison requires nz >= {minimum}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    rimed_ice_constants(morr_rimed_ice)   # validates the selector
    for name, value in (
            ("theta", theta), ("qv", qv), ("qc", qc), ("qr", qr),
            ("qi", qi), ("qs", qs), ("qg", qg), ("nc", nc), ("nr", nr),
            ("ni", ni), ("ns", ns), ("ng", ng), ("rho", rho),
            ("pii", pii), ("pressure", pressure), ("dz", dz)):
        _check(name, value, shape)
    cu_values = (qrcuten, qscuten, qicuten)
    has_cu = any(value is not None for value in cu_values)
    if has_cu and not all(value is not None for value in cu_values):
        raise ValueError("qrcuten, qscuten, and qicuten must be supplied "
                         "together")
    if has_cu:
        for name, value in zip(("qrcuten", "qscuten", "qicuten"), cu_values):
            _check(name, value, shape)
        qrcu, qscu, qicu = cu_values
    else:
        # The kernel gates every read with has_cu.
        qrcu = qscu = qicu = rho
    for name, value in (
            ("rainnc", rainnc), ("rainncv", rainncv),
            ("snownc", snownc), ("snowncv", snowncv),
            ("graupelnc", graupelnc), ("graupelncv", graupelncv),
            ("sr", sr)):
        _check(name, value, (ny, nx))
    effective = {"effc": effc, "effr": effr, "effi": effi, "effs": effs}
    for name, value in effective.items():
        if value is None:
            effective[name] = cp.empty_like(theta)
        else:
            _check(name, value, shape)
    ncol = ny * nx
    chunk = ncol if columns_per_call is None else int(columns_per_call)
    if chunk < 1:
        raise ValueError(f"columns_per_call must be positive, got {chunk}")
    chunk = min(chunk, ncol)
    if _work_scratch is None:
        work = cp.empty((NWORK * nz * chunk,), dtype=DTYPE)
    else:
        work = _work_scratch
        if (work.dtype != DTYPE or not work.flags.c_contiguous
                or work.size < NWORK * nz * chunk):
            raise ValueError(
                f"_work_scratch must be contiguous float32 with at least "
                f"{NWORK * nz * chunk} values, got {work.dtype} {work.shape}")
    flat_work = work.reshape(-1)
    if t1d_out is not None:
        _check("t1d_out", t1d_out, shape)
    ltrue = (cp.empty((ny, nx), dtype=cp.int32)
             if _ltrue_scratch is None else _ltrue_scratch)
    _check("_ltrue_scratch", ltrue, (ny, nx), cp.int32)
    ltrue.fill(0)

    rimed = np.int32(morr_rimed_ice)
    process = get_kernel("morrison", "mo_process")
    sediment = get_kernel("morrison", "mo_sediment_64" if nz <= _SHALLOW_KMAX
                          else "mo_sediment_256")
    finalize = get_kernel("morrison", "mo_finalize")
    for col0 in range(0, ncol, chunk):
        ccols = min(chunk, ncol - col0)
        nlocal = nz * ccols
        cell_blocks = (nlocal + _CELL_TPB - 1) // _CELL_TPB
        column_blocks = (ccols + _COLUMN_TPB - 1) // _COLUMN_TPB
        span = (np.int64(nlocal), np.int32(ncol), np.int32(col0),
                np.int32(ccols))
        process(
            (cell_blocks,), (_CELL_TPB,),
            (theta, pii, pressure, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns,
             ng, qrcu, qscu, qicu, np.int32(has_cu), rho, flat_work, ltrue,
             DTYPE(dt), rimed) + span)
        sediment(
            (column_blocks,), (_SEDIMENT_TPB,),
            (pressure, dz, rho, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
             flat_work, ltrue, rainnc, rainncv, snownc, snowncv, graupelnc,
             graupelncv, sr, DTYPE(dt), rimed, np.int32(nz), np.int32(ncol),
             np.int32(col0), np.int32(ccols)))
        finalize(
            (cell_blocks,), (_CELL_TPB,),
            (theta, pii, pressure, rho, qv, qc, qr, qi, qs, qg, nc, nr, ni,
             ns, ng, flat_work, ltrue, effective["effc"], effective["effi"],
             effective["effs"], effective["effr"], DTYPE(dt), rimed) + span)
        if t1d_out is not None:
            plane = flat_work[WORK_T3D * nlocal:(WORK_T3D + 1) * nlocal]
            t1d_out.reshape(nz, ncol)[:, col0:col0 + ccols] = (
                plane.reshape(nz, ccols))
    if chunk == ncol:
        return flat_work[:NWORK * nz * ncol].reshape(NWORK * nz, ny, nx)
    return work


def launch_refl10cm_wrf(qv, qr, nr, qs, ns, qg, ng, t, p, refl, *,
                        morr_rimed_ice: int = 1) -> None:
    """WRF refl10cm_hm (:4502-4675) with the wrapper's -35 dBZ floor
    (:913-917), one thread per column.  ``t`` is the scheme's post-call
    temperature (T1D) and ``p`` the prepared pressure (P1D)."""
    shape = refl.shape
    if len(shape) != 3:
        raise ValueError(f"refl fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    for name, value in (("qv", qv), ("qr", qr), ("nr", nr), ("qs", qs),
                        ("ns", ns), ("qg", qg), ("ng", ng), ("t", t),
                        ("p", p), ("refl", refl)):
        _check(name, value, shape)
    rimed_ice_constants(morr_rimed_ice)
    ncol = ny * nx
    get_kernel("morrison", "mo_refl10cm")(
        ((ncol + _COLUMN_TPB - 1) // _COLUMN_TPB,), (_COLUMN_TPB,),
        (qv, qr, nr, qs, ns, qg, ng, t, p, refl, np.int32(morr_rimed_ice),
         np.int32(nz), np.int32(ncol)))


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields and apply Morrison to ``state`` in place.

    On a history step, reflectivity is evaluated after the scheme call from
    its post-call temperature (T1D, the WORK stack's MO_T3D plane) and
    moments plus the unchanged prepared pressure, matching
    module_mp_morr_two_moment.F:913-914.
    """
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=10 state lacks Morrison fields: "
                         + ", ".join(missing))

    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "morr_theta")
    rho = state.scratch((nz, ny, nx), "morr_rho")
    pii = state.scratch((nz, ny, nx), "morr_pii")
    dz = state.scratch((nz, ny, nx), "morr_dz")
    work = state.scratch(work_slot_shape(nz, ny * nx), SCRATCH_WORK_SLOT)
    ltrue = state.scratch((ny, nx), "morr_ltrue", dtype=np.int32)
    _prepare_fields(thb, phb, state.thp, state.php,
                    np.int32(ny * nx), np.int32(state.thb.ndim == 3),
                    np.int32(state.phb.ndim == 3), DTYPE(c.G), theta, dz)
    if getattr(state, "use_theta_m", 0):
        theta[...] = state.total_theta()
    launch_exner(state.p, pii)

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")
    # Lazy import (matches microphysics.apply's lazy Morrison dispatch):
    # the WRF moist_physics_prep_em/finish_em bracket around the scheme.
    from woof.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    cu_rates = (getattr(getattr(state, "physics", None), "cu_rates", None)
                if getattr(cfg, "cu_physics", 0) else None)
    t1d = (state.scratch((nz, ny, nx), "refl_t") if refl_10cm_due
           else None)
    launch_morrison(theta, state.qv, state.qc, state.qr,
                    state.qi, state.qs, state.qg,
                    state.nc, state.nr, state.ni, state.ns, state.ng,
                    rho, pii, state.p, dz,
                    rainnc, rainncv, snownc, snowncv,
                    graupelnc, graupelncv, sr, dt,
                    effc=state.effc, effr=state.effr,
                    effi=state.effi, effs=state.effs,
                    qrcuten=(None if cu_rates is None else
                              cu_rates["rqrcuten"]),
                    qscuten=(None if cu_rates is None else
                              cu_rates["rqscuten"]),
                    qicuten=(None if cu_rates is None else
                              cu_rates["rqicuten"]),
                    morr_rimed_ice=cfg.morr_rimed_ice,
                    columns_per_call=work_chunk_columns(ny * nx, nz),
                    t1d_out=t1d,
                    _work_scratch=work, _ltrue_scratch=ltrue)
    if refl_10cm_due:
        from woof.core.refl import stash_refl_10cm
        refl = state.scratch((nz, ny, nx), "refl_10cm")
        # WRF's T1D is the scheme's own post-call temperature, not
        # TH*PII recomposed; P1D is the pre-call prepared pressure
        # (F:730, :780, :913-914).  Each column call copied its WORK
        # T3D plane into refl_t.
        launch_refl10cm_wrf(state.qv, state.qr, state.nr, state.qs,
                            state.ns, state.qg, state.ng, t1d, state.p,
                            refl, morr_rimed_ice=cfg.morr_rimed_ice)
        stash_refl_10cm(state, refl)
    moist_physics_finish(state, cfg, theta, dt)
    bound_snow_ratio(sr)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)

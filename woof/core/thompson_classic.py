"""Classic Thompson (mp_physics=8) as WRF v4.6.1 computes it.

Launchers for ``woof/core/kernels/thompson_classic.cu``, the literal
transcription of ``module_mp_thompson.F``'s ``mp_gt_driver`` and
``mp_thompson`` for the classic scheme (no aerosol, no hail).  The unit
carries WRF's tendency structure: the source, condensation, evaporation and
sedimentation stages accumulate float tendencies and the state is written
once by the terminal apply, every expression in gfortran's order and
precision, glibc's libm, and no fused multiply-add.  Graded word for word
against WRF's own Fortran by ``tools/thompson_real_column_parity/
mp8_wrf_ulp.py`` and ``tests/test_thompson_classic_wrf461.py``.

Four launches per call (plus effective radius and, when due, reflectivity):

1. ``tc_entry``, one thread per cell: phy_prep's theta, Exner and layer
   depths, the driver's private graupel number, and per column WRF's
   no-microphysics exit and melting level;
2. ``tc_sources``, one thread per cell: :1798-3574, every process rate,
   the conservation limiters, the tendencies, the TAU+1 update, cloud
   condensation and rain evaporation;
3. ``tc_column``, one thread per column: fall speeds, the five
   sedimentation loops, the phase cleanup, the terminal apply, surface
   precipitation and theta;
4. ``tc_finish``: moist_physics_finish (theta increment, h_diabatic).
"""

from __future__ import annotations

import numpy as np

from woof.core import constants as c
from woof.core.kernels import get_kernel

DTYPE = np.float32
_TPB = 128
#: Plane counts of the classic unit's scratch (thompson_classic.cu
#: TC_NTEN, TC_NWORK, TC_NCOLPLANES): eleven tendencies, eighteen working
#: fields, nine fall-speed profiles and two flux profiles per column.
NTEN = 11
NWORK = 18
NCOLPLANES = 11
#: Every per-call scratch slot of the classic unit, ``(name, planes)``:
#: ``planes`` stacked copies of a level field (``nz + 1`` levels for the
#: column buffer).  Lifetime: each call writes every word it reads.
SCRATCH_VOLUME_SLOTS = (
    ("mp_th", 1), ("mp_pii", 1), ("mp_thompson_temperature", 1),
    ("mp_dz8w", 1), ("mp_thompson_saved_theta", 1),
    ("mp_thompson_graupel_number_shadow", 1),
    ("mp_thompson_classic_ten", NTEN), ("mp_thompson_classic_work", NWORK),
)
SCRATCH_INTERFACE_SLOTS = (("mp_thompson_classic_colbuf", NCOLPLANES),)
SCRATCH_INT_VOLUME_SLOTS = ("mp_thompson_classic_flags",)
SCRATCH_INT_SURFACE_SLOTS = ("mp_thompson_classic_micro",
                             "mp_thompson_classic_kmelt")


def _grid(n: int) -> tuple[int]:
    return ((int(n) + _TPB - 1) // _TPB,)


def _require(name, value, shape, dtype=DTYPE):
    if tuple(value.shape) != tuple(shape) or value.dtype != dtype:
        raise ValueError(f"{name} must be {np.dtype(dtype).name} {shape}, "
                         f"got {value.dtype} {tuple(value.shape)}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def classic_tables(table_owner):
    """The table arguments of ``tc_sources`` in its order."""
    a = table_owner.arrays
    names = ("tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2",
             "tcr_sacr1", "tms_sacr1", "tcr_sacr2", "tms_sacr2",
             "tnr_racs1", "tnr_racs2", "tnr_sacr1", "tnr_sacr2",
             "tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
             "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz",
             "tpi_qcfz", "tni_qcfz",
             "tps_iaus", "tni_iaus", "tpi_ide", "t_Efrw", "t_Efsw")
    return tuple(a[name] for name in names)


def apply_classic(state, cfg, dt: float, table_owner, *,
                  refl_10cm_due: bool = False, member_columns=None):
    """One classic Thompson call on ``state``; returns the surface fields.

    Reads and writes the DomainState surface ``_apply_thompson`` always
    used (thb/thp/phb/php/p/w and the eight classic species), keeps the
    microphysics theta in scratch ``mp_th`` and the Exner function in
    ``mp_pii``, and the private graupel number in
    ``mp_thompson_graupel_number_shadow`` for the reflectivity.

    ``member_columns``: the columns of one ensemble member when a packed
    all-member state (``members*ny`` rows) shares one ``(nz, ny, nx)``
    base-state slab (woof/ensemble/batch_physics.py); the base state is
    then read at the member's own column.  ``None``: every column.
    """
    nz, ny, nx = state.p.shape
    columns = ny * nx
    size = nz * columns
    shape = (nz, ny, nx)
    th = state.scratch(shape, "mp_th")
    pii = state.scratch(shape, "mp_pii")
    t1d = state.scratch(shape, "mp_thompson_temperature")
    dz = state.scratch(shape, "mp_dz8w")
    th_save = state.scratch(shape, "mp_thompson_saved_theta")
    ng1d = state.scratch(shape, "mp_thompson_graupel_number_shadow")
    # Plane-stacked (planes*levels, ny, nx) float32 slots: plane s, level k
    # is rows s*nz + k, the kernels' s*size + k*columns + column.
    ten = state.scratch((NTEN * nz, ny, nx), "mp_thompson_classic_ten")
    work = state.scratch((NWORK * nz, ny, nx), "mp_thompson_classic_work")
    colbuf = state.scratch((NCOLPLANES * (nz + 1), ny, nx),
                           "mp_thompson_classic_colbuf")
    flags = state.scratch(shape, "mp_thompson_classic_flags",
                          dtype=np.int32)
    column_micro = state.scratch((ny, nx), "mp_thompson_classic_micro",
                                 dtype=np.int32)
    column_kmelt = state.scratch((ny, nx), "mp_thompson_classic_kmelt",
                                 dtype=np.int32)
    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")

    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "p", "thp"):
        _require(name, getattr(state, name), shape)
    _require("php", state.php, (nz + 1, ny, nx))
    thb_full = int(state.thb.ndim == 3)
    phb_full = int(state.phb.ndim == 3)
    # The base state's column stride: a full (nz, Y, nx) base field holds
    # either every packed column or one member's (a shared slab).
    base_columns = {getattr(state, n).shape[1] * getattr(state, n).shape[2]
                    for n, full in (("thb", thb_full), ("phb", phb_full))
                    if full}
    if len(base_columns) > 1:
        raise ValueError("thb and phb are base-state slabs of different "
                         "column counts")
    stride = base_columns.pop() if base_columns else columns
    if member_columns is not None and stride not in (columns,
                                                     int(member_columns)):
        raise ValueError("the base-state slab is neither the packed state's "
                         "nor one member's")
    if columns % stride:
        raise ValueError("the base-state slab does not tile the packed "
                         "columns")
    # WRF copies w(i,k,j) into w1d(k): the lower full levels, contiguous
    # as a leading-axis slice of the contiguous interface field.
    w_lower = state.w[:-1]

    column_micro.fill(0)
    column_kmelt.fill(0)
    launch_classic_entry(
        state, th_save, pii, t1d, dz, ng1d, column_micro, column_kmelt,
        size=size, columns=columns, thb_full=thb_full, phb_full=phb_full,
        member_columns=stride)
    launch_classic_sources(
        state, t1d, ng1d, column_micro, column_kmelt, table_owner, ten,
        work, flags, dt, size=size, columns=columns)
    launch_classic_column(
        state, t1d, pii, th, ng1d, w_lower, dz, ten, work, flags,
        colbuf, (rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,
                 sr), dt, nz=nz, columns=columns)
    if refl_10cm_due:
        refl = state.scratch(shape, "refl_10cm")
        launch_refl10cm_classic(state.qv, state.qr, state.nr, state.qs,
                                state.qg, ng1d, t1d, state.p, refl)
        from woof.core.refl import stash_refl_10cm
        stash_refl_10cm(state, refl)
    launch_classic_effective_radius(t1d, state, size=size)
    launch_classic_vapour_fill(state)
    launch_classic_finish(th, state, th_save, cfg, dt, size=size)
    return rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr


def launch_classic_entry(state, th_save, pii, t1d, dz, ng1d, column_micro,
                         column_kmelt, *, size, columns, thb_full, phb_full,
                         member_columns):
    """``tc_entry``: phy_prep's theta, Exner and layer depths, the driver's
    private graupel number, and per column WRF's no-microphysics exit and
    melting level."""
    get_kernel("thompson_classic", "tc_entry")(
        _grid(size), (_TPB,),
        (state.thb, state.thp, state.phb, state.php, state.p,
         state.qv, state.qc, state.qi, state.qr, state.qs, state.qg,
         th_save, pii, t1d, dz, ng1d, column_micro, column_kmelt,
         DTYPE(c.P0), DTYPE(c.RCP), DTYPE(c.G),
         np.int32(size), np.int32(columns), np.int32(thb_full),
         np.int32(phb_full), np.int32(member_columns)))


def launch_classic_sources(state, t1d, ng1d, column_micro, column_kmelt,
                           table_owner, ten, work, flags, dt, *,
                           size, columns):
    """``tc_sources``: :1798-3574 per cell.  On return the state holds the
    entry rewrite (:1844-1942) and ``ten`` WRF's tendencies as the rain
    evaporation leaves them (its cp2): every working value is
    ``X1d + Xten*DT``."""
    get_kernel("thompson_classic", "tc_sources")(
        _grid(size), (_TPB,),
        (state.p, t1d, state.qv, state.qc, state.qi, state.qr, state.qs,
         state.qg, state.ni, state.nr, ng1d, column_micro, column_kmelt,
         *classic_tables(table_owner),
         ten, work, flags,
         DTYPE(dt), np.int32(size), np.int32(columns)))


def launch_classic_column(state, t1d, pii, th, ng1d, w_lower, dz, ten, work,
                          flags, colbuf, surface, dt, *, nz, columns):
    """``tc_column``: fall speeds, the five sedimentation loops, the phase
    cleanup, the terminal apply, surface precipitation and theta."""
    get_kernel("thompson_classic", "tc_column")(
        _grid(columns), (_TPB,),
        (t1d, pii, th, state.qv, state.qc, state.qi, state.qr, state.qs,
         state.qg, state.ni, state.nr, ng1d, w_lower, dz, ten, work,
         flags, colbuf, *surface,
         DTYPE(dt), np.int32(nz), np.int32(columns)))


def launch_classic_effective_radius(t1d, state, *, size):
    """``tc_effective_radius``: :1466-1478 and calc_effectRad, microns."""
    get_kernel("thompson_classic", "tc_effective_radius")(
        _grid(size), (_TPB,),
        (t1d, state.p, state.qv, state.qc, state.qi, state.ni, state.qs,
         state.effc, state.effi, state.effs, np.int32(size)))


def launch_classic_finish(th, state, th_save, cfg, dt, *, size):
    """``tc_finish``: moist_physics_finish (theta increment, h_diabatic)."""
    get_kernel("thompson_classic", "tc_finish")(
        _grid(size), (_TPB,),
        (th, state.thp, state.h_diabatic, th_save,
         DTYPE(DTYPE(cfg.mp_tend_lim) * DTYPE(dt)), DTYPE(dt),
         np.int32(cfg.no_mp_heating != 0), np.int32(size)))


#: Every launcher apply_classic calls, in module order (the mp=8 call-graph
#: receipt, tools/mp8_freeze_receipt.py, spies exactly these).
LAUNCHERS = ("launch_classic_entry", "launch_classic_sources",
             "launch_classic_column", "launch_refl10cm_classic",
             "launch_classic_effective_radius", "launch_classic_vapour_fill",
             "launch_classic_finish")


def launch_classic_vapour_fill(state) -> None:
    """mp_gt_driver's negative-vapour fill (:1436-1445), after the
    reflectivity and effective radii, which read the unfilled qv1d, and
    before moist_physics_finish (woof.core.microphysics.
    thompson_negative_vapour_fill)."""
    from woof.core.microphysics import thompson_negative_vapour_fill
    thompson_negative_vapour_fill(state.qv)


def launch_refl10cm_classic(qv, qr, nr, qs, qg, ng, t, p, refl) -> None:
    """WRF calc_refl10cm (:5710-6028) for the classic scheme, one thread
    per column, and the driver's MAX(-35, dBZ) (:1461)."""
    nz, ny, nx = p.shape
    columns = ny * nx
    get_kernel("thompson_classic", "tc_refl10cm")(
        _grid(columns), (_TPB,),
        (qv, qr, nr, qs, qg, ng, t, p, refl,
         np.int32(nz), np.int32(columns)))

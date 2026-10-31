"""Full-state model health validation for synchronized run boundaries.

The production path uses exactly one descriptor-driven CUDA kernel launch.
Every descriptor supplies a field pointer, bounds, optional auxiliary array,
and a stable status-class bit.  The kernel ORs every failing class and records
the lexicographically first ``(field, flat-index)`` pair.  Only that compact
record is copied to the host.  No value is clipped or repaired: a failed gate
is terminal for the worker.

The NumPy mirror below deliberately shares the descriptor/rule construction
with the CUDA path.  It is used by CPU corruption tests and by host-side LBC
or nest-table construction checks before those tables are uploaded.
"""

from __future__ import annotations

import dataclasses
import ctypes
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


# A four-domain NSSL-2 step currently reaches 528 descriptors after its lazy
# persistent microphysics and nest scratch is materialized (527 until the
# surface layer began publishing USTM as a surface field).  Keep a power-of-
# two ceiling with substantial headroom so scheduled physics can add state
# without invalidating a run after it has already advanced.
MAX_HEALTH_FIELDS = 1024

#: The ceiling of a FAST_SBM (mp_physics=30) domain.  The scheme transports
#: 142 spectral fields plus its bulk and number diagnostics, and a nested
#: child walks every one of them again as its nest boundary tables: a nested
#: 3 km + 1 km FAST_SBM tree measured 2,889 descriptors (RTX PRO 6000,
#: 2026-10-09) and stopped at its first health validation against 1024.
#: Taken ONLY where the state carries the bins, so every other domain keeps
#: its footprint.  48 B per descriptor: 192 KiB per FAST_SBM domain.
FSBM_HEALTH_FIELDS = 4096


def health_field_ceiling(state: Any) -> int:
    """The descriptor ceiling of one domain: :data:`FSBM_HEALTH_FIELDS`
    where the state carries FAST_SBM bins, else :data:`MAX_HEALTH_FIELDS`."""
    if getattr(state, "fast_sbm_bins", None) is not None:
        return FSBM_HEALTH_FIELDS
    return MAX_HEALTH_FIELDS


def health_field_ceiling_for_config(cfg: Any) -> int:
    """The same ceiling from a RunConfig (preflight pricing): FAST_SBM bins
    are allocated exactly when ``mp_physics == 30``."""
    if int(getattr(cfg, "mp_physics", 0) or 0) == 30:
        return FSBM_HEALTH_FIELDS
    return MAX_HEALTH_FIELDS


_INDEX_MASK = (1 << 48) - 1

# Stable status classes.  The individual failing field remains available in
# ValidationReport.first_bad_field; these bits make the device record compact
# while still reporting every affected physical/storage class in one pass.
STATUS_BITS = {
    "wind": 1 << 0,
    "geopotential": 1 << 1,
    "coupled_mass": 1 << 2,
    "pressure": 1 << 3,
    "specific_volume": 1 << 4,
    "theta": 1 << 5,
    "moisture": 1 << 6,
    "moment": 1 << 7,
    "effective_radius": 1 << 8,
    "held_tendency": 1 << 9,
    "lateral_boundary": 1 << 10,
    "nest_table": 1 << 11,
    "soil_temperature": 1 << 12,
    "soil_moisture": 1 << 13,
    "surface": 1 << 14,
    "generic": 1 << 15,
    "particle_volume": 1 << 16,
    # Chem species fields and process arrays (woof/core/chem_state.py).
    "chem": 1 << 17,
}

_LOWER = 1 << 0
_UPPER = 1 << 1
_STRICT_LOWER = 1 << 2
_ADD_AUX = 1 << 3
_AUX_LEVEL = 1 << 4
_INT32_STORAGE = 1 << 5

# Chunk sizing for the fused gate.  One block per descriptor left the whole
# check waiting on the single largest field while the rest of the device sat
# idle, so the launch also spreads each descriptor over ``ceil(size / chunk)``
# blocks.  The chunk is driven by the TOTAL element count, targeting roughly
# _CHUNK_TARGET_BLOCKS blocks over the whole inventory, so one huge descriptor
# cannot explode the count of blocks that exit immediately on the short ones.
# The measured time is flat over chunks of 8192-65536 elements, so the floor
# only has to keep tiny inventories from launching a block per 256 elements.
_CHUNK_TARGET_BLOCKS = 8192
_MIN_CHUNK_ELEMENTS = 4096

# These are immutable setup-time category maps, not floating model state.
# They cannot contain NaN/Inf, and their legal ranges depend on the selected
# Noah vegetation/soil tables (which validate them at initialization).  Keep
# them in collect_state_fields so the real inventory and dtype remain pinned,
# but deliberately exclude only these exact names from the per-step GPU gate.
# Every other non-float32 field fails descriptor construction unless it has an
# explicit range policy below.
GPU_INTEGER_EXCLUSIONS = frozenset({
    # WRF tile categories are integer indices, not physical bounds.
    "surface.mosaic_cat_index",
    "surface.ivgtyp",
    "surface.isltyp",
    # The urban type map (sf_urban_physics > 0): urban_var_init derives it
    # once from IVGTYP and the VEGPARM urban categories (0, or 1..11), no
    # step writes it, and its legal range depends on the URBPARM table the
    # run selected -- the same treatment as the category maps above.
    "surface.utype_urb2d",
    # SINT geometry index maps (T10/F16 device tables): built FP64-on-host,
    # range-validated against the REAL parent extents by nest_interp's
    # _check_table at registration, stored int32, and immutable thereafter
    # (the kernels consume them read-only; no step writes them).  Their
    # legal ranges depend on per-domain parent extents, which a fixed
    # FieldRule cannot express -- same treatment as the Noah category
    # fields above.  One entry per stagger class (mass/x-stag/y-stag).
    "nest.scratch.nest_sint_ci_m", "nest.scratch.nest_sint_ip_m",
    "nest.scratch.nest_sint_cj_m", "nest.scratch.nest_sint_jp_m",
    "nest.scratch.nest_sint_ci_x", "nest.scratch.nest_sint_ip_x",
    "nest.scratch.nest_sint_cj_x", "nest.scratch.nest_sint_jp_x",
    "nest.scratch.nest_sint_ci_y", "nest.scratch.nest_sint_ip_y",
    "nest.scratch.nest_sint_cj_y", "nest.scratch.nest_sint_jp_y",
})
_GPU_RANGE_CHECKED_INT32 = frozenset({
    # PWP atomically ORs solver failure bits; only zero is healthy.
    "surface._pwp_status",
    # G3 stores its four cloud levels and averaging clock as native integers.
    "surface.g3_isfc", "surface.g3_diag_clock",
    # Noah writes one of four energy-balance solution cases.
    "surface.ebal",
    # YSU writes a one-based model level; zero is the pre-first-call value.
    "surface.kpbl",
    # MYNN's EDMF plume top, also a one-based model level, but capped one
    # level lower than kpbl: module_bl_mynn.F:6392 ends DMP_mf with the
    # unconditional ``ktop=MIN(ktop,KTE-1)`` on every exit path, and the
    # only place the value grows is ``ktop=MAX(ktop,k)`` (:6362) inside
    # ``do k=kts+1,kte-1``.  Zero means "no plume" -- DMP_mf sets it at
    # :6033 before the ascent, mynn_bl_driver sets the whole row at :646
    # ahead of the initflag block, so it is also the pre-first-call value
    # and the value for bl_mynn_edmf=0.  collect_state_fields supplies the
    # state-dependent nz-1 cap.
    "surface.ktop_plume",
    # Noah-MP's top snow layer index: minus the number of ACTIVE snow layers,
    # so it is negative-or-zero and a naive ">= 0" bound would fire on the
    # first snowpack column.  The range is [-NSNOW, 0], NSNOW fixed at 3 by
    # module_sf_noahmpdrv.F:628.  Every writer in the pinned tree stays inside
    # it: SNOW_INIT (:2382-2402) assigns 0, -1, -2 or -3 and nothing else;
    # SNOWFALL sets -1 (module_sf_noahmplsm.F:6589); COMBINE only ever does
    # ISNOW = ISNOW + 1 (:6697, :6776) or ISNOW = 0 (:6727), both toward zero;
    # and DIVIDE closes with ISNOW = -MSNO (:6902) where MSNO reaches 3 at
    # most.  woof's port carries the identical statements
    # (noahmp_driver.py:260-279, noahmp_snow.py:234, :364, :387, :428, :519).
    # Not nz-dependent, so no state-dependent cap is needed.
    "surface.isnowxy",
    # Noah-MP's plant growth stage.  GROWING_GDD
    # (module_sf_noahmplsm.F:10783-10797) is the only writer and assigns 1..8
    # and nothing else; 0 is the Registry cold value and, under the admitted
    # opt_crop=0, the value forever, because CARBON_CROP never runs and
    # module_sf_noahmpdrv.F:1397 writes the unchanged INOUT back.  So [0, 8]
    # bounds both the crop and the no-crop case.
    "surface.pgsxy",
    # WRF's terrain shadow mask (topo_shading = 1, woof.core.topo_radiation).
    # toposhad_init and toposhad (module_radiation_driver.F:4505, :4661) zero
    # it and the four quadrant searches set 1 (:4694, :4712, :4730, :4748);
    # the ``= -1`` alternatives beside them are commented out in v4.7.1.  So
    # [0, 1], not nz-dependent.  Without a policy every topo_shading forecast
    # refused at initialization (seen on the first real run of the port).
    "surface.shadowmask",
    # WRF eclipse states: clear=0, partial=1, total=2, annular=3.
    "surface.ecmask",
})

#: ``NSNOW``, WRF's fixed snow-layer count (``module_sf_noahmpdrv.F:628``), and
#: therefore the magnitude of the most negative legal ``ISNOWXY``.  Spelled
#: here rather than imported so this module keeps no dependency on a physics
#: runner; ``tests/test_health_integer_policy.py`` asserts it equals
#: ``woof.core.noahmp_runtime.NSNOW`` so the two cannot drift apart.
NOAHMP_SNOW_LAYERS = 3

#: The largest plant growth stage ``GROWING_GDD`` assigns.
NOAHMP_MAX_GROWTH_STAGE = 8

#: The lid the theta ceiling below was calibrated at: WRF's default model
#: top, 100 hPa.
THETA_CEILING_REFERENCE_LID_PA = 10000.0

#: The ceiling that calibration produced, and the value this gate applied at
#: every lid until it was made lid-aware.
THETA_CEILING_AT_REFERENCE_LID_K = 600.0


def theta_ceiling_for_lid(p_top: float | None) -> float:
    """The total-theta ceiling for a run whose model top is ``p_top`` Pa.

    600 K is not a property of the atmosphere.  It is a property of the
    atmosphere UNDER A 100 hPa LID: potential temperature is a level's
    temperature carried down the dry adiabat to 1000 hPa, so the SAME
    physically ordinary stratospheric air reads ~600 K under a 100 hPa lid
    and ~950 K under a 20 hPa one.  Held flat, the ceiling therefore refused
    legitimate deep-top initial states -- a real 20 hPa-top ERA5 state tripped
    it -- while the thing it exists to catch, a thermodynamic runaway, reaches
    thousands of kelvin and is nowhere near either number.

    So the ceiling is carried, not widened.  The quantity kept fixed is the
    LID TEMPERATURE the gate admits: 600 K at 100 hPa is
    ``600 * (100/1000)**RCP`` = 310.6 K, which is what this gate has always
    admitted at the top of a column, and every lid now admits exactly that
    same temperature.  The result is byte-identical to the old ceiling at the
    reference lid and never tighter than it at a shallower one (a lid below
    the reference is clamped rather than allowed to close the gate on
    configurations that pass today).

    ``p_top=None`` is the accurate answer for a state whose base has not been
    loaded yet -- ``DomainState.p_top`` is None until :meth:`load_base` runs --
    and returns the calibrated ceiling rather than guessing a lid.
    """

    from woof.core.constants import RCP

    if p_top is None:
        return THETA_CEILING_AT_REFERENCE_LID_K
    lid = float(p_top)
    if (not math.isfinite(lid) or lid <= 0.0
            or lid >= THETA_CEILING_REFERENCE_LID_PA):
        return THETA_CEILING_AT_REFERENCE_LID_K
    return THETA_CEILING_AT_REFERENCE_LID_K * float(
        (THETA_CEILING_REFERENCE_LID_PA / lid) ** RCP)


def gpu_integer_policy(name: str, dtype: Any) -> int | None:
    """Device-gate storage policy for one collected field.

    Returns the descriptor storage flags for a field the gate checks, or
    ``None`` for a field carrying a documented exclusion.  Raises
    :class:`TypeError` for a dtype with no declared policy, which is the
    point of the function: an integer field added to any physics driver
    must be classified deliberately before a forecast can run, rather than
    silently admitted to a float32 gate.

    Module level rather than inline in :meth:`StateHealthValidator._refresh`
    so that the host-side census, the CPU tests and the device inventory all
    classify through this one implementation instead of three spellings of
    it.  A CPU test can therefore falsify the real gate without a device.
    """
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return 0
    # Four lattice coordinates, generated on the host at relocation and never
    # written by physics. nest_accumulation.advance validates the restored
    # coordinates against the archive extents before consuming them. They
    # are deliberately int64 for long tracks, not floating weather state.
    if (dtype == np.dtype(np.int64)
            and name == "nest.scratch.nest_accumulation/origin"):
        return None
    if dtype == np.dtype(np.int32):
        if name in GPU_INTEGER_EXCLUSIONS:
            return None
        if name in _GPU_RANGE_CHECKED_INT32:
            return _INT32_STORAGE
    raise TypeError(
        f"GPU health field {name!r} has unsupported dtype "
        f"{dtype}; add an explicit integer range policy or a "
        "documented exclusion before this inventory can run")


@dataclass(frozen=True)
class FieldRule:
    """Bounds and status class for one health field."""

    status_class: str
    lower: float | None = None
    upper: float | None = None
    strict_lower: bool = False
    #: Why THIS bound and not another one, appended to the refusal when the
    #: bound is derived from the run rather than fixed by a kernel.  A
    #: derived ceiling that reports only its number reads exactly like a
    #: silently loosened gate, which is the thing a reader must be able to
    #: rule out.  Never uploaded to the device: the kernel compares numbers.
    bound_note: str | None = None
    # Strict transport can copy a signed rounding residual to the outflow
    # rim before microphysics clamps the interior. Unknown geometry keeps
    # every floor, and finiteness and ceilings always cover the whole field.
    allow_lower_rim: bool = False
    rim_width: int = 0

    @property
    def status_bit(self) -> int:
        return STATUS_BITS[self.status_class]


@dataclass(frozen=True)
class HealthField:
    """One array plus optional auxiliary values used by a derived check.

    ``aux_mode='direct'`` checks ``values + auxiliary`` elementwise.
    ``aux_mode='level'`` broadcasts a one-dimensional auxiliary profile over
    each horizontal plane (the theta base profile case).
    """

    name: str
    values: Any
    rule: FieldRule
    auxiliary: Any | None = None
    aux_mode: str | None = None
    plane_size: int = 0


@dataclass(frozen=True)
class ValidationReport:
    """Compact result of one full-state validation pass."""

    ok: bool
    status_bits: int
    first_bad_field: str | None = None
    first_bad_index: tuple[int, ...] | None = None
    first_bad_flat_index: int | None = None
    first_bad_value: float | None = None
    reason: str | None = None
    phase: str | None = None

    @property
    def failing_classes(self) -> tuple[str, ...]:
        return tuple(name for name, bit in STATUS_BITS.items()
                     if self.status_bits & bit)


class HealthCheckError(FloatingPointError):
    """A synchronized model-state health gate failed."""

    def __init__(self, report: ValidationReport):
        self.report = report
        location = ("unknown" if report.first_bad_index is None
                    else str(report.first_bad_index))
        phase = "" if report.phase is None else f" during {report.phase}"
        super().__init__(
            f"full-state health gate failed{phase}: "
            f"{report.first_bad_field}{location}: {report.reason}; "
            f"classes={report.failing_classes}")


_FINITE = FieldRule("generic")


def _leaf(name: str) -> str:
    return name.lower().replace("[", ".").replace("]", "").split(".")[-1]


def rule_for_field(name: str, *, p_top: float | None = None) -> FieldRule:
    """Return the shared CPU/CUDA rule for a named model field.

    ``p_top`` is the run's configured model-top pressure in pascals, supplied
    by :func:`collect_state_fields` off the state's own base.  Only the total
    theta bound reads it, and only because that bound is a lid temperature
    rather than a fixed number; see :func:`theta_ceiling_for_lid`.
    """
    lower_name = name.lower()
    leaf = _leaf(name)
    from woof.core.fast_sbm_contract import BIN_SPECIES
    if leaf in BIN_SPECIES:
        return FieldRule("moment",0.0,1.e15) if leaf.startswith("ff8") else FieldRule("moisture",0.0,1.0)
    if lower_name.startswith("lbc."):
        return FieldRule("lateral_boundary")
    if lower_name.startswith("nest."):
        return FieldRule("nest_table")
    # ``rthften``/``rqvften`` are the dycore's exported advective forcing
    # pair (WRF RTHFTEN/RQVFTEN): rates in K s-1 and kg kg-1 s-1 with the
    # same standing as h_diabatic -- a held tendency has no physical
    # ceiling, and finiteness is the whole gate.
    if (lower_name.startswith("held.")
            or leaf in ("h_diabatic", "rthften", "rqvften")):
        return FieldRule("held_tendency")
    if leaf in ("u", "v"):
        return FieldRule("wind", -500.0, 500.0)
    if leaf == "w":
        return FieldRule("wind", -200.0, 200.0)
    if leaf == "thp":
        # State collection supplies total theta through the base-state
        # auxiliary descriptor; the perturbation itself is never bounded.
        # The ceiling is the run's LID carried down the dry adiabat rather
        # than a fixed number -- see theta_ceiling_for_lid, which reproduces
        # the calibrated 600 K exactly at the 100 hPa reference lid (the
        # value that admits the legal ~495 K WK82 20-km sounding with a
        # useful perturbation margin) and opens only as far as a deeper lid
        # physically requires.
        ceiling = theta_ceiling_for_lid(p_top)
        note = None
        if ceiling != THETA_CEILING_AT_REFERENCE_LID_K:
            note = (
                f"the ceiling is this run's {float(p_top):g} Pa model lid "
                f"carried down the dry adiabat: the same "
                f"{THETA_CEILING_AT_REFERENCE_LID_K:g} K admitted at the "
                f"{THETA_CEILING_REFERENCE_LID_PA:g} Pa reference lid, which "
                "is one admitted lid temperature and not a loosened gate; a "
                "value above it is runaway thermodynamics, not a deep model "
                "top")
        return FieldRule("theta", 100.0, ceiling, bound_note=note)
    if leaf == "php":
        return FieldRule("geopotential", -1.0e8, 1.0e8)
    if leaf == "mup":
        # State collection adds MUB2D, checking the coupled denominator.
        return FieldRule("coupled_mass", 0.0, 2.0e6, strict_lower=True)
    if leaf == "p":
        return FieldRule("pressure", 0.0, 2.0e6, strict_lower=True)
    if leaf == "al":
        # AL is the perturbation specific volume and may be negative.
        return FieldRule("specific_volume", -1.0e4, 1.0e4)
    if leaf == "alt":
        return FieldRule("specific_volume", 0.0, 1.0e4,
                         strict_lower=True)
    if leaf in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"):
        if WRF_EXACT:
            # Strict mode transcribes WRF's scalar updates, and WRF keeps
            # moisture non-negative only on its final positive-definite
            # stage: rk_update_scalar leaves qv = -3.19e-13 at the first step
            # of the combo-sweep round-3 pair, and the non-PD stages carry
            # and grow such values (qv = -2.4808048e-07 at step 11, the same
            # word in stock WRF 4.6.1 and in strict WOOF).  No fixed number
            # separates WRF's own state from broken transport, so strict
            # mode drops the sign bound and keeps the ceiling; a non-finite
            # value still fails the gate.
            return FieldRule(
                "moisture", None, 1.0,
                bound_note=("strict mode keeps WRF's own below-zero moisture "
                            "residuals, as WRF integrates with them"))
        return FieldRule("moisture", 0.0, 1.0)
    if leaf in ("nc", "nr", "ni", "ns", "ng", "qndrop", "qnr", "qni",
                "qns", "qng", "qnh", "qnn",
                # mp=28 aerosol number tracers, per kilogram.  WRF's own
                # terminal clamp holds them inside [11.1E6, 9999.E6] and
                # [5.0E3, 9999.E6] (module_mp_thompson.F:3977-3982), and
                # the deliberately unclamped surface emission at :1310-1327
                # can push the lowest level above the ceiling between the
                # clamp and the next call's entry pack.  The 1.0e15
                # "moment" ceiling is five decades of headroom over that
                # (1.0e15 / 9.999e9 = 1.0001e5), so this rule catches a
                # blow-up without pretending to enforce the scheme's own
                # bounds.
                #
                # THE BREAKAGE IT CAUGHT, twice, and the reading that keeps
                # it at this number.  A 2.7 km specified domain forced from
                # the monthly WIF climatology stopped at 3 h 36 min on
                # nwfa(k=0, j=ny-1, i=0), its lowest level at the
                # north-west boundary corner, reading 1.0114111e15; the
                # corner grew 74x in the first hour from 6.94e8, while qv
                # in that same cell moved 1.2 percent over that hour
                # (0.0044067 to 0.0044585) and 4.2 percent by the last
                # frame before the stop.  A 2.25 km specified parent
                # stopped at 3.9 h on nwfa(nz-1, ny-1, i), the lid layer of
                # its north boundary row, reading 1.0268e15 after growing
                # from 5.5e8.  Both grew on the specified ring, which the
                # final positive-definite scalar stage moved by its
                # vertical advection, a term WRF computes and never
                # applies (moist._exclude_specified_ring_advection).  With
                # that fixed the ring holds its table (5.46e8 at the lid,
                # 4 h),
                # five and a half decades under this ceiling, so the
                # ceiling never stood between a legitimate value and a
                # run: raising it would only have moved where the runaway
                # was noticed.
                "nwfa", "nifa"):
        if WRF_EXACT and leaf in ("nwfa", "nifa"):
            # Strict mode transcribes WRF's scalar transport, and WRF clamps
            # the two aerosol numbers only in a column Thompson works on: a
            # column it returns from at module_mp_thompson.F:2020 (no
            # condensate, subsaturated over ice) never reaches the clamp at
            # :3979-3982 and keeps its transport residual
            # (kernels/thompson_aerosol_state.cu, "THE NO-MICROPHYSICS
            # COLUMN").  Stock WRF 4.6.1 (pinned strict build, dmpar 16,
            # the two-domain UAE mp=28 WIF case, 2026-08-21 00Z) writes
            # QNIFA -3.70e-6, -3.29e-6 and -2.80e-6 on d01's top layer at
            # (j, i) = (87, 61), (58, 111) and (90, 50) at +1 h: interior
            # points, outside the nest and the 5-row boundary zone.  The
            # rim-only floor below stopped strict ArWen on that case at
            # nifa(43, 94, 47) = -7.2e-6, the same layer, edge distance 5.
            # As for moisture above, no fixed number separates WRF's own
            # residual from broken transport, so strict mode drops the sign
            # bound and keeps the ceiling; a non-finite value still fails.
            # The other numbers keep the rim-only floor below until a native
            # run shows WRF integrating an interior negative of theirs.
            return FieldRule(
                "moment", None, 1.0e15,
                bound_note=("strict mode keeps WRF's own below-zero aerosol "
                            "number residuals, as WRF integrates with them"))
        if WRF_EXACT:
            return FieldRule(
                "moment", 0.0, 1.0e15, allow_lower_rim=True,
                bound_note=("strict mode permits below-zero scalar residuals "
                            "on the specified rim only"))
        return FieldRule("moment", 0.0, 1.0e15)
    if leaf in ("qvolg", "qvolh"):
        if WRF_EXACT:
            return FieldRule(
                "particle_volume", 0.0, 1.0, allow_lower_rim=True,
                bound_note=("strict mode permits below-zero scalar residuals "
                            "on the specified rim only"))
        return FieldRule("particle_volume", 0.0, 1.0)
    if leaf.startswith("chem_"):
        # A chem species (ug kg-1 or ppmv of dry air).  Non-negative, as
        # the positive-definite transport and every process's epsilc floor
        # keep it, and 1.0e9 is four decades above the densest surface
        # smoke or dust ever measured (order 1e4 to 1e5 ug m-3), so the
        # ceiling catches a runaway, not a plume.
        return FieldRule("chem", 0.0, 1.0e9)
    if leaf.startswith("chemdiag_") or leaf.startswith("chemwork_"):
        # Process arrays and ledger totals: finiteness is the whole gate
        # (a ledger bucket or a flux accumulator has either sign).
        return FieldRule("chem")
    if leaf in ("effc", "effr", "effi", "effs"):
        # Morrison stores these diagnostics in microns.
        return FieldRule("effective_radius", 0.0, 1.0e6)
    if leaf in ("tsk", "tslb"):
        return FieldRule("soil_temperature", 150.0, 400.0)
    if leaf in ("smois", "sh2o"):
        # No lower bound.  real.exe's vertical interpolation writes small
        # negative deep-layer soil moisture near coasts (down to -0.032
        # m3 m-3 in RUC's nine layers on the gulf-coast combo case), WRF
        # integrates those words, and no land model here writes a negative
        # value (RUC's soilmoist bounds each layer to [qmin + 1.e-8, maxsmc],
        # module_sf_ruclsm.F:6033-6076).  The 0 floor therefore fired only on
        # input words WRF accepts: it named no breakage and refused 24 runs
        # of the 2026-10-07 combo sweep (round 2) once RUC's own check was
        # lifted.  Finiteness and the 1.0 ceiling stand.
        return FieldRule("soil_moisture", None, 1.0)
    if lower_name.startswith("surface."):
        if leaf == "_pwp_status":
            return FieldRule("surface", 0.0, 0.0,
                             bound_note="PWP failure bits: 1 Richardson mixing did not converge; 2 nonfinite ocean state")
        if leaf == "ebal":
            # Noah's energy-balance case (kernels/noah.cu): 0-3 are the land
            # SFLX branches, 4 marks an SFLX_GLACIAL (land-ice) column.
            # With the bound at 3, the first land-ice point of a real
            # forecast failed the post-d01-sync health gate (g16 ports-pbl).
            return FieldRule("surface", 0.0, 4.0)
        if leaf == "kpbl":
            # collect_state_fields supplies the state-dependent nz cap.
            return FieldRule("surface", 0.0)
        if leaf == "ktop_plume":
            # collect_state_fields supplies the state-dependent nz-1 cap.
            return FieldRule("surface", 0.0)
        if leaf == "isnowxy":
            # Noah-MP counts ACTIVE snow layers downward from the surface, so
            # the index is negative: 0 is snow-free and -NSNOW is a full pack.
            # Fixed by module_sf_noahmpdrv.F:628, not by the domain, so this
            # needs no state-dependent cap.  See _GPU_RANGE_CHECKED_INT32.
            return FieldRule("surface", float(-NOAHMP_SNOW_LAYERS), 0.0)
        if leaf == "pgsxy":
            return FieldRule("surface", 0.0,
                             float(NOAHMP_MAX_GROWTH_STAGE))
        if leaf == "ecmask":
            return FieldRule("surface", 0.0, 3.0)
        if leaf == "ecobsc":
            return FieldRule("surface", 0.0, 1.0)
        if leaf == "shadowmask":
            return FieldRule("surface", 0.0, 1.0)
        return FieldRule("surface")
    return _FINITE


def field_from_array(name: str, values: Any, *, auxiliary: Any | None = None,
                     aux_mode: str | None = None,
                     plane_size: int = 0,
                     p_top: float | None = None) -> HealthField:
    """Build one descriptor using :func:`rule_for_field`."""
    return HealthField(name, values, rule_for_field(name, p_top=p_top),
                       auxiliary, aux_mode, int(plane_size))


def _is_array(value: Any) -> bool:
    return (hasattr(value, "shape") and hasattr(value, "dtype")
            and hasattr(value, "size"))


def _walk_arrays(value: Any, prefix: str, *, seen: set[int] | None = None,
                 depth: int = 0) -> list[HealthField]:
    """Recursively collect arrays from one explicitly selected container."""
    if seen is None:
        seen = set()
    if value is None or depth > 8:
        return []
    if _is_array(value):
        return [field_from_array(prefix, value)]
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    result: list[HealthField] = []
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
    elif dataclasses.is_dataclass(value):
        items = [(field.name, getattr(value, field.name))
                 for field in dataclasses.fields(value)]
    elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)):
        items = list(enumerate(value))
    elif hasattr(value, "__dict__"):
        items = sorted(vars(value).items())
    else:
        return result
    for key, child in items:
        if isinstance(child, (str, bytes, int, float, bool, type(None))):
            continue
        result.extend(_walk_arrays(
            child, f"{prefix}.{key}", seen=seen, depth=depth + 1))
    return result


#: ``collect_state_fields(p_top=...)`` default: read the lid off the state.
_LID_FROM_STATE = object()


#: Soil columns no scheme touches when ``sf_surface_physics = 0``.
_INERT_WITHOUT_LAND_MODEL = frozenset({
    "surface.tslb", "surface.smois", "surface.sh2o", "surface.smcrel"})


def collect_state_fields(state: Any, *, backend: str = "cpu",
                         extra_tables: Mapping[str, Any] | None = None,
                         p_top: Any = _LID_FROM_STATE,
                         ) -> tuple[HealthField, ...]:
    """Collect the mutable synchronized state covered by the health gate.

    The explicit top-level list follows the restart manifest's cross-step
    state.  Physics-held arrays, surface/soil fields, resident LBC tables,
    and any nest-owned containers are added separately.  ``extra_tables``
    is the integration surface for multi-domain donor/receiver tables.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'")
    # The run's own model lid, read off the state rather than passed in, so
    # every road that gates a state -- the resident device gate, the CPU
    # mirror, the pinned-host store gate -- derives the same theta ceiling
    # from the same object.  ``DomainState.p_top`` is None until load_base
    # runs; theta_ceiling_for_lid answers with the calibrated ceiling then.
    # A road whose state object is a SLAB-HEIGHT TEMPLATE names its own lid
    # instead: the domain's base carries it, the template may not.
    lid = getattr(state, "p_top", None) if p_top is _LID_FROM_STATE else p_top
    p_top = None if lid is None else float(lid)
    result: list[HealthField] = []
    for name in (
            "u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
            "qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
            "ns", "ng", "nwfa", "nifa",
            "qh", "qndrop", "qnr", "qni", "qns", "qng",
            "qnh", "qnn", "qvolg", "qvolh", "effc", "effr", "effi",
            "effs", "h_diabatic",
            # The dycore's exported advective forcing pair.  Absent on a
            # state with no cumulus consumer, and the None skip below is
            # what keeps every other census unchanged.
            "rthften", "rqvften"):
        value = getattr(state, name, None)
        if value is None:
            continue
        if name == "mup" and getattr(state, "mub2d", None) is not None:
            result.append(field_from_array(
                name, value, auxiliary=state.mub2d, aux_mode="direct"))
        elif name == "thp" and getattr(state, "thb", None) is not None:
            thb = state.thb
            if getattr(thb, "ndim", 0) == 1:
                plane = int(np.prod(value.shape[1:], dtype=np.int64))
                result.append(field_from_array(
                    name, value, auxiliary=thb, aux_mode="level",
                    plane_size=plane, p_top=p_top))
            else:
                result.append(field_from_array(
                    name, value, auxiliary=thb, aux_mode="direct",
                    p_top=p_top))
        else:
            result.append(field_from_array(name, value, p_top=p_top))
    if getattr(state,"fast_sbm_bins",None) is not None:
        from woof.core.fast_sbm_contract import BIN_SPECIES
        for name in BIN_SPECIES:
            value=getattr(state,name)
            result.append(field_from_array(name,value,p_top=p_top))
    # Chem species fields and float32 process arrays, named by the chem
    # table; absent on a chem-off state, so every other census is unchanged.
    # The float64 ledger vectors are sums of these fields and carry no
    # state of their own, and the fused gate takes float32/int32 only.
    if getattr(state, "chem", None) is not None:
        from woof.state_serialization_contract import chem_state_attrs
        for name in chem_state_attrs(state):
            value = getattr(state, name, None)
            if (value is not None
                    and np.dtype(value.dtype) == np.dtype(np.float32)):
                result.append(field_from_array(name, value, p_top=p_top))

    driver = getattr(state, "physics", None)
    if driver is not None:
        for tendency_name in (
                "pbl_tendencies", "radiation_tendencies",
                "cumulus_tendencies"):
            tendency = getattr(driver, tendency_name, None)
            result.extend(_walk_arrays(
                tendency, f"held.{tendency_name.removesuffix('_tendencies')}"))
        for name in (
                "rthratenlw", "rthratensw", "cu_nca", "cu_pratec",
                "cu_raincv", "cu_rates", "_pending_rainbl"):
            value = getattr(driver, name, None)
            if value is not None:
                result.extend(_walk_arrays(value, f"held.{name}"))
        surface_fields = _walk_arrays(getattr(driver, "fields", None),
                                      "surface")
        nz = int(getattr(getattr(state, "p", None), "shape", (0,))[0])
        no_land_model = (getattr(driver, "scheme_dispatch", None) is not None
                         and driver.scheme_dispatch.get(
                             "sf_surface_physics", "") is None)
        for field in surface_fields:
            if no_land_model and field.name in _INERT_WITHOUT_LAND_MODEL:
                # sf_surface_physics = 0: no scheme reads or writes the soil
                # columns; they carry real.exe's words (all zero for TSLB)
                # into history as WRF's do.  A range rule on them refused
                # every no-land-model WRF-file run at the initial gate
                # (2026-10-07 combo sweep), so only finiteness is checked.
                field = dataclasses.replace(field, rule=_FINITE)
            if field.name == "surface.g3_isfc":
                field = dataclasses.replace(field, rule=FieldRule("surface", 0.0, float(nz)))
            elif field.name == "surface.g3_diag_clock":
                field = dataclasses.replace(field, rule=FieldRule("surface", 0.0, float(2**31-1)))
            elif field.name == "surface.kpbl":
                field = dataclasses.replace(
                    field, rule=FieldRule("surface", 0.0, float(nz)))
            elif field.name == "surface.ktop_plume":
                # MYNN's plume top sits one level BELOW kpbl's ceiling:
                # module_bl_mynn.F:6392 closes DMP_mf with
                # ``ktop=MIN(ktop,KTE-1)`` unconditionally, and woof's port
                # carries the same clamp (kernels/mynn_pbl.cu:3073,
                # mynn_pbl.py:2979).  The ``max(..., 0)`` matters only for a
                # state that carries no pressure array, where nz is 0 and the
                # sole legal value is the never-written 0.
                field = dataclasses.replace(
                    field, rule=FieldRule("surface", 0.0,
                                          float(max(nz - 1, 0))))
            result.append(field)
        result.extend(_walk_arrays(getattr(driver, "microphysics", None),
                                   "surface.microphysics"))

    # Restart-persistent scratch is mutable cross-step state too.  Keep this
    # explicit (matching restart.py's fail-loud philosophy) so transient RK
    # work buffers are not mistaken for synchronized state.
    persistent_scratch = {
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_hailnc", "mp_hailncv",
        "mp_sr", "mp_kessler_sr",
        "cu_rainc", "cu_nca", "cu_pratec", "cu_raincv",
        "cu_rthcuten", "cu_rqvcuten", "cu_rqccuten", "cu_rqicuten",
        "cu_rqrcuten", "cu_rqscuten",
    }
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        if slot in persistent_scratch:
            prefix = ("held.scratch" if slot.startswith("cu_")
                      else "surface.microphysics.scratch")
            result.append(field_from_array(f"{prefix}.{slot}", value))

    if backend == "gpu":
        # Current LBC upload deliberately packs every interval/field/side
        # value and tendency into one resident scratch allocation.  Validate
        # that allocation once instead of emitting ~100 overlapping views.
        scratch = getattr(state, "_scratch", {})
        packed_lbc = scratch.get("lbc_forcing_tables")
        if packed_lbc is not None:
            result.append(field_from_array("lbc.forcing_tables", packed_lbc))
            for slot, value in sorted(scratch.items()):
                if slot.startswith("lbc_weights_"):
                    result.append(field_from_array(f"lbc.{slot}", value))
        else:
            result.extend(_walk_arrays(
                getattr(state, "_lateral_boundary_device", None), "lbc"))
    else:
        result.extend(_walk_arrays(
            getattr(state, "lateral_boundaries", None), "lbc"))

    # Task-13/14 may attach nest tables under different owner-selected names.
    # Selecting only names containing ``nest`` avoids scanning transient RK
    # scratch while still making the health interface future-proof.
    for name, value in sorted(vars(state).items()):
        if "nest" in name.lower() and value is not None:
            result.extend(_walk_arrays(value, f"nest.{name}"))
    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        if slot.startswith("nest_"):
            result.append(field_from_array(f"nest.scratch.{slot}", value))
    if extra_tables:
        for name, value in sorted(extra_tables.items()):
            result.extend(_walk_arrays(value, f"nest.{name}"))

    ceiling = health_field_ceiling(state)
    if len(result) > ceiling:
        name = ("FSBM_HEALTH_FIELDS" if ceiling == FSBM_HEALTH_FIELDS
                else "MAX_HEALTH_FIELDS")
        raise ValueError(
            f"health field inventory has {len(result)} descriptors; "
            f"{name}={ceiling} must be raised explicitly")
    width = max(0, int(getattr(state, "_health_spec_zone", 0)))
    return tuple(dataclasses.replace(
        field, rule=dataclasses.replace(field.rule, rim_width=width))
        if field.rule.allow_lower_rim else field for field in result)


def _cpu_values(field: HealthField) -> np.ndarray:
    values = np.asarray(field.values)
    if field.auxiliary is None:
        return values
    auxiliary = np.asarray(field.auxiliary)
    if field.aux_mode == "direct":
        if auxiliary.shape != values.shape:
            raise ValueError(
                f"{field.name} auxiliary shape {auxiliary.shape} != "
                f"field shape {values.shape}")
        return values + auxiliary
    if field.aux_mode == "level":
        if auxiliary.ndim != 1 or values.shape[0] != auxiliary.size:
            raise ValueError(
                f"{field.name} level auxiliary shape {auxiliary.shape} is "
                f"incompatible with {values.shape}")
        return values + auxiliary.reshape((-1,) + (1,) * (values.ndim - 1))
    raise ValueError(f"unknown auxiliary mode {field.aux_mode!r}")


def _bad_mask(values: np.ndarray, rule: FieldRule) -> np.ndarray:
    bad = ~np.isfinite(values)
    if rule.lower is not None:
        low = values <= rule.lower if rule.strict_lower else values < rule.lower
        if rule.allow_lower_rim and rule.rim_width > 0 and values.ndim >= 2:
            width = rule.rim_width
            low[..., :width, :] = False
            low[..., -width:, :] = False
            low[..., :, :width] = False
            low[..., :, -width:] = False
        bad |= low
    if rule.upper is not None:
        bad |= values > rule.upper
    return bad


def _reason(value: float, rule: FieldRule, *, shape=(), flat=0) -> str:
    if not math.isfinite(value):
        return f"non-finite value {value!r}"
    if rule.lower is not None and (
            value <= rule.lower if rule.strict_lower else value < rule.lower):
        op = ">" if rule.strict_lower else ">="
        reason = f"value {value!r} violates lower bound {op} {rule.lower!r}"
        if rule.allow_lower_rim:
            if len(shape) >= 2:
                ny, nx = shape[-2:]
                j, i = (flat // nx) % ny, flat % nx
                distance = min(i, j, nx - 1 - i, ny - 1 - j)
                reason += f" at (i, j)=({i}, {j}), edge distance {distance}"
            reason += ("; interior negatives are not WRF behaviour and point "
                       "to a transport or microphysics defect; below-zero "
                       "residuals are permitted on the specified rim only")
        return reason
    if rule.upper is not None and value > rule.upper:
        reason = f"value {value!r} exceeds upper bound {rule.upper!r}"
        return reason if rule.bound_note is None else (
            f"{reason} ({rule.bound_note})")
    return "unknown invariant failure"


def _validate_fields_numpy(fields: Mapping[str, Any] | Sequence[HealthField],
                           *, phase: str | None = None) -> ValidationReport:
    """Compatibility scan for older bridges and uncommon host array formats."""
    if isinstance(fields, Mapping):
        descriptors = tuple(field_from_array(name, value)
                            for name, value in fields.items())
    else:
        descriptors = tuple(fields)
    status = 0
    first: tuple[int, int, np.ndarray, HealthField] | None = None
    for field_number, field in enumerate(descriptors):
        values = _cpu_values(field)
        bad = _bad_mask(values, field.rule)
        if bool(np.any(bad)):
            status |= field.rule.status_bit
            flat = int(np.flatnonzero(np.ravel(bad, order="C"))[0])
            if first is None:
                first = field_number, flat, values, field
    if first is None:
        return ValidationReport(True, status, phase=phase)
    _, flat, values, field = first
    value = float(np.ravel(values, order="C")[flat])
    return ValidationReport(
        False, status, field.name,
        tuple(int(i) for i in np.unravel_index(flat, values.shape)),
        flat, value, _reason(value, field.rule, shape=values.shape, flat=flat), phase)


class _HostHealthDescriptor(ctypes.Structure):
    _fields_ = [("values", ctypes.c_void_p), ("auxiliary", ctypes.c_void_p),
                ("size", ctypes.c_size_t), ("plane", ctypes.c_size_t),
                ("value_type", ctypes.c_uint32), ("auxiliary_type", ctypes.c_uint32),
                ("result_type", ctypes.c_uint32), ("auxiliary_mode", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("status_bit", ctypes.c_uint32),
                ("lower", ctypes.c_double), ("upper", ctypes.c_double)]


class _HostHealthResult(ctypes.Structure):
    _fields_ = [("status_bits", ctypes.c_uint32), ("first_field", ctypes.c_size_t),
                ("first_index", ctypes.c_size_t), ("first_value", ctypes.c_double)]


def _native_health_entry():
    from woof.core import portable_math
    try:
        library = portable_math._load()
        entry = getattr(library, "gpuwm_validate_health_fields_rim", None)
        supports_rim = entry is not None
        if entry is None:
            entry = getattr(library, "gpuwm_validate_health_fields", None)
    except (OSError, RuntimeError, FileNotFoundError):
        return None
    if entry is not None:
        entry.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                          ctypes.POINTER(_HostHealthResult)]
        entry.supports_rim = supports_rim
        if supports_rim:
            entry.argtypes += [ctypes.c_void_p]
        entry.restype = ctypes.c_int32
    return entry


def _rim_geometry(field: HealthField) -> tuple[int, int, int]:
    shape = field.values.shape
    if (not field.rule.allow_lower_rim or field.rule.rim_width <= 0
            or len(shape) < 2 or not all(shape[-2:])):
        return 0, 0, 0
    return int(shape[-2]), int(shape[-1]), field.rule.rim_width


def _native_health_descriptor(field):
    # Metadata only. Unusual arrays keep their original conversion and scan.
    host_types = (np.ndarray, np.generic, float, int)
    if not isinstance(field.values, host_types):
        return None
    values = np.asarray(field.values)
    types = {np.dtype(np.float32): 1, np.dtype(np.float64): 2}
    if (values.dtype not in types or not values.flags.c_contiguous
            or values.size >= 1 << 48 or field.rule.status_class not in STATUS_BITS
            or any(bound is not None and type(bound) not in (float, int)
                   for bound in (field.rule.lower, field.rule.upper))):
        return None
    auxiliary = None
    mode, plane, auxiliary_type = 0, 0, 0
    result_type = types[values.dtype]
    if field.auxiliary is not None:
        if not isinstance(field.auxiliary, host_types):
            return None
        auxiliary = np.asarray(field.auxiliary)
        if auxiliary.dtype not in types or not auxiliary.flags.c_contiguous:
            return None
        if field.aux_mode == "direct" and auxiliary.shape == values.shape:
            mode = 1
        elif (field.aux_mode == "level" and values.ndim > 0 and auxiliary.ndim == 1
              and values.shape[0] == auxiliary.size):
            mode, plane = 2, math.prod(values.shape[1:])
            if plane == 0:
                return None
        else:
            return None
        auxiliary_type = types[auxiliary.dtype]
        result_type = types.get(np.result_type(values, auxiliary))
        if result_type is None:
            return None
    flags = ((_LOWER if field.rule.lower is not None else 0)
             | (_UPPER if field.rule.upper is not None else 0)
             | (_STRICT_LOWER if field.rule.strict_lower else 0))
    try:
        lower = 0.0 if field.rule.lower is None else float(field.rule.lower)
        upper = 0.0 if field.rule.upper is None else float(field.rule.upper)
    except (ValueError, OverflowError):
        return None
    if result_type == 1 and any(math.isfinite(bound) and abs(bound) > float(np.finfo(np.float32).max)
                                for bound in (lower, upper)):
        # NumPy reports casts of finite out-of-range comparison scalars.
        # Keep that warning/exception behavior rather than silently using Inf.
        return None
    descriptor = _HostHealthDescriptor(values.ctypes.data,
        None if auxiliary is None else auxiliary.ctypes.data, values.size, plane,
        types[values.dtype], auxiliary_type, result_type, mode, flags,
        field.rule.status_bit, lower, upper)
    return descriptor, values, auxiliary


def validate_fields_cpu(fields: Mapping[str, Any] | Sequence[HealthField],
                        *, phase: str | None = None) -> ValidationReport:
    """Run the same host health rules through a bounded read-only Rust scan."""
    # NumPy's custom exception/callback/log policies are observable behavior
    # of the legacy auxiliary addition and comparisons. A native report must
    # not replace those exceptions or suppress a requested callback.
    if any(mode not in ("ignore", "warn") for mode in np.geterr().values()):
        return _validate_fields_numpy(fields, phase=phase)
    entry = _native_health_entry()
    if entry is None:
        return _validate_fields_numpy(fields, phase=phase)
    descriptors = (tuple(field_from_array(name, value) for name, value in fields.items())
                   if isinstance(fields, Mapping) else tuple(fields))
    if (not getattr(entry, "supports_rim", False)
            and any(_rim_geometry(field)[2] for field in descriptors)):
        # An older ABI cannot represent the allowance. The NumPy path
        # retains the floor and applies the same known domain geometry.
        return _validate_fields_numpy(descriptors, phase=phase)
    native, indices, keepalive, fallback = [], [], [], []
    for index, field in enumerate(descriptors):
        prepared = _native_health_descriptor(field)
        if prepared is None:
            # Preserve uncommon conversion/comparison exception ordering.
            # Integer, boolean and other real numeric arrays can coexist
            # with the native batch without sending dense floats back to it.
            if (not isinstance(field.values, (np.ndarray, np.generic, float, int))
                    or np.asarray(field.values).dtype.kind not in "biuf"):
                return _validate_fields_numpy(descriptors, phase=phase)
            fallback.append((index, field))
        else:
            descriptor, values, auxiliary = prepared
            native.append(descriptor)
            indices.append(index)
            keepalive.append((values, auxiliary))
    status, first = 0, None
    if native:
        from woof.core import portable_math
        jobs = (_HostHealthDescriptor * len(native))(*native)
        result = _HostHealthResult()
        args = [jobs, len(native), portable_math._workers(None), ctypes.byref(result)]
        if getattr(entry, "supports_rim", False):
            geometry = np.asarray([_rim_geometry(descriptors[i]) for i in indices],
                                  dtype=np.uintp)
            args.append(geometry.ctypes.data)
        code = int(entry(*args))
        if code:
            # An additive ABI cannot weaken the existing checked interface.
            return _validate_fields_numpy(descriptors, phase=phase)
        status = int(result.status_bits)
        if status:
            # Failure is rare and must retain NumPy's observable auxiliary
            # overflow/invalid warnings, exceptions, and exact diagnostic.
            return _validate_fields_numpy(descriptors, phase=phase)
        if result.first_field < len(native):
            index = indices[result.first_field]
            first = (index, int(result.first_index), float(result.first_value),
                     keepalive[result.first_field][0].shape)
    for index, field in fallback:
        values = _cpu_values(field)
        bad = _bad_mask(values, field.rule)
        if bool(np.any(bad)):
            status |= field.rule.status_bit
            flat = int(np.flatnonzero(np.ravel(bad, order="C"))[0])
            if first is None or index < first[0]:
                first = (index, flat, float(np.ravel(values, order="C")[flat]), values.shape)
    if first is None:
        return ValidationReport(True, status, phase=phase)
    index, flat, value, shape = first
    field = descriptors[index]
    return ValidationReport(False, status, field.name,
        tuple(int(i) for i in np.unravel_index(flat, shape)), flat, value,
        _reason(value, field.rule, shape=shape, flat=flat), phase)


def validate_state_cpu(state: Any, *, phase: str | None = None,
                       extra_tables: Mapping[str, Any] | None = None,
                       raise_on_error: bool = False) -> ValidationReport:
    """CPU mirror for a state-like object (used by corruption fixtures)."""
    report = validate_fields_cpu(
        collect_state_fields(state, backend="cpu", extra_tables=extra_tables),
        phase=phase)
    if raise_on_error and not report.ok:
        raise HealthCheckError(report)
    return report


def validate_store_fields(state: Any, carriers: Mapping[str, Any],
                          store: Mapping[str, Any], *,
                          domain_shape: tuple[int, int],
                          auxiliaries: Mapping[str, Any] | None = None,
                          phase: str | None = None,
                          extra_tables: Mapping[str, Any] | None = None,
                          p_top: Any = _LID_FROM_STATE,
                          ) -> tuple[ValidationReport, dict[str, Any]]:
    """THE FULL-STATE GATE, over a streamed domain's pinned HOST store.

    :class:`StateHealthValidator` needs a domain-shaped DEVICE state, which is
    precisely the object the store-direct road exists not to build -- and a
    verdict taken on the slab-height template instead would report one slab as
    the domain.  That is why the gate was left unarmed on that road.  It is not
    why it has to stay unarmed: a streamed domain IS domain-shaped, in pinned
    host memory, and :func:`validate_fields_cpu` is the production rule set
    already, sharing :func:`rule_for_field` and :func:`_bad_mask` with the
    kernel rather than restating them.  So the gate runs where the domain is.

    THE FIELD LIST IS NOT REBUILT HERE, and that is the whole design.  Two
    production walkers are run over the SAME slab-height template --
    :func:`collect_state_fields`, which says which arrays are gated and under
    which rule, and the run's own carrier inventory, which says what each array
    is called in the store -- and they are joined by ``id()``, because both
    return references to the same objects.  A field added to either side is
    therefore carried the day it is added; nothing here is a second list that
    can drift from the first.  ``backend="gpu"`` because the resident road's
    verdict is the one this must be comparable with, down to
    :func:`gpu_integer_policy`'s documented exclusions and its refusal of an
    undeclared integer dtype.

    ``domain_shape`` is ``(ny, nx)`` and is CHECKED against every substituted
    array rather than trusted, because a store accidentally holding slab-height
    arrays is exactly the silent pass this seam exists to prevent.

    ``auxiliaries`` supplies the DOMAIN-shaped form of the two descriptors that
    carry one -- ``thp``'s base-state theta and ``mup``'s base-state dry mass.
    A descriptor whose auxiliary has no domain-shaped substitute is refused
    rather than validated against the template's slab.

    Returns the report and a coverage record naming every gated field the store
    does not carry, so what the gate did NOT see is a number in a receipt
    instead of an absence a reader has to infer.
    """
    descriptors, coverage = _store_health_descriptors(
        state, carriers, store, domain_shape=domain_shape,
        auxiliaries=auxiliaries, extra_tables=extra_tables, p_top=p_top)
    started = time.perf_counter()
    report = validate_fields_cpu(descriptors, phase=phase)
    coverage['seconds'] = time.perf_counter() - started
    return report, coverage


def _store_health_descriptors(state, carriers, store, *, domain_shape,
                              auxiliaries=None, extra_tables=None,
                              p_top=_LID_FROM_STATE, device=False):
    """Bind the existing health rules to the canonical store's actual arrays."""
    auxiliaries = {} if auxiliaries is None else auxiliaries
    ny, nx = int(domain_shape[0]), int(domain_shape[1])
    key_by_id = {id(array): key for key, array in carriers.items()}
    descriptors: list[HealthField] = []
    covered: list[str] = []
    uncovered: list[str] = []
    excluded: list[str] = []
    seen: set[tuple] = set()
    for field in collect_state_fields(state, backend="gpu",
                                      extra_tables=extra_tables,
                                      p_top=p_top):
        if gpu_integer_policy(field.name, field.values.dtype) is None:
            excluded.append(field.name)
            continue
        key = key_by_id.get(id(field.values))
        values = None if key is None else store.get(key)
        if values is None:
            uncovered.append(field.name)
            continue
        values = values if device else np.asarray(values)
        if (values.ndim != int(getattr(field.values, "ndim", values.ndim))
                or np.dtype(values.dtype) != np.dtype(field.values.dtype)):
            raise ValueError(
                f"store carrier {key!r} is {values.ndim}-D {values.dtype} "
                f"against a gated field {field.name!r} of "
                f"{getattr(field.values, 'ndim', '?')}-D {field.values.dtype}")
        if values.ndim >= 2 and (
                int(values.shape[-2]) not in (ny, ny + 1)
                or int(values.shape[-1]) not in (nx, nx + 1)):
            raise ValueError(
                f"store carrier {key!r} is {tuple(values.shape)}, whose "
                f"horizontal extent is not this domain's {(ny, nx)}: the "
                "full-state gate must not report a verdict taken over one "
                "slab as a verdict over the domain")
        auxiliary, aux_mode, plane = field.auxiliary, field.aux_mode, 0
        if auxiliary is not None:
            if field.name not in auxiliaries:
                raise ValueError(
                    f"gated field {field.name!r} is checked against an "
                    "auxiliary array and no domain-shaped substitute for it "
                    "was supplied; validating a domain field against the "
                    "template's slab is the defect this gate exists to avoid")
            auxiliary = auxiliaries[field.name]
            if not device:
                auxiliary = np.asarray(auxiliary)
            # collect_state_fields' own rule, restated over the substituted
            # pair rather than inherited: a 1-D auxiliary broadcasts over each
            # horizontal plane, and the plane is the DOMAIN's, not the slab's.
            if auxiliary.ndim == 1:
                aux_mode = "level"
                plane = int(np.prod(values.shape[1:], dtype=np.int64))
            else:
                aux_mode = "direct"
        key_tuple = (key, values.shape, field.rule, id(auxiliary), aux_mode,
                     plane)
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        descriptors.append(HealthField(field.name, values, field.rule,
                                       auxiliary, aux_mode, plane))
        covered.append(field.name)
    coverage = {
        "fields_checked": len(descriptors),
        "elements_checked": int(sum(int(f.values.size) for f in descriptors)),
        "bytes_checked": int(sum(int(f.values.nbytes) for f in descriptors)),
        "seconds": None,
        "covered": tuple(covered),
        "not_in_store": tuple(uncovered),
        "excluded_integer_fields": tuple(excluded),
    }
    return tuple(descriptors), coverage


def cuda_source() -> str:
    """Complete NVRTC source for offline compilation (no device needed)."""
    from woof.core.kernels import _preamble
    path = Path(__file__).with_name("kernels") / "health.cu"
    return _preamble() + path.read_text(encoding="utf-8")


def _u64_words(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64).view(np.uint32)


def _word_floats(words: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(words, dtype=np.uint32).view(np.float32)


def _cuda_pointer(value: Any, name: str) -> int:
    interface = getattr(value, "__cuda_array_interface__", None)
    if interface is None:
        raise TypeError(f"GPU health field {name!r} is not a CUDA array")
    if not bool(value.flags.c_contiguous):
        raise TypeError(f"GPU health field {name!r} must be C-contiguous")
    return int(value.data.ptr)


class StateHealthValidator:
    """Reusable one-launch CUDA validator for one prepared DomainState."""

    def __init__(self, state: Any, *, extra_tables: Mapping[str, Any] | None = None,
                 field_provider=None):
        self.state = state
        self._field_provider = field_provider
        self.extra_tables = extra_tables
        # The restart classifier already treats integration_health_* scratch
        # as rebuilt.  Metadata is bit-packed into FP32 storage because the
        # state scratch API intentionally has one model dtype.  The
        # descriptor ceiling is the domain's own (health_field_ceiling).
        self._ceiling = health_field_ceiling(state)
        self._ptr = state.scratch((self._ceiling * 2,),
                                  "integration_health_field_ptr")
        self._aux = state.scratch((self._ceiling * 2,),
                                  "integration_health_aux_ptr")
        self._size = state.scratch((self._ceiling * 2,),
                                   "integration_health_field_size")
        self._bounds = state.scratch((self._ceiling, 2),
                                     "integration_health_bounds")
        self._flags = state.scratch((self._ceiling,),
                                    "integration_health_flags")
        self._planes = state.scratch((self._ceiling,),
                                     "integration_health_planes")
        self._status = state.scratch((self._ceiling * 2,),
                                     "integration_health_status_bits")
        self._rim = state.scratch((self._ceiling, 3),
                                  "integration_health_rim")
        self._result = state.scratch((4,), "integration_health_validation")
        self._chunk = _MIN_CHUNK_ELEMENTS
        self._chunk_count = 1
        self.fields: tuple[HealthField, ...] = ()
        self.excluded_integer_fields: tuple[HealthField, ...] = ()

    @property
    def host_scan_bytes(self) -> int:
        """Bytes scanned on the CPU; resident fields are scanned on the GPU."""
        return 0

    def _refresh(self) -> None:
        collected = (collect_state_fields(
            self.state, backend="gpu", extra_tables=self.extra_tables)
            if self._field_provider is None else self._field_provider())
        fields: list[HealthField] = []
        excluded: list[HealthField] = []
        storage_flags: list[int] = []
        pointers_by_field: list[int] = []
        seen: set[tuple] = set()
        for field in collected:
            flag = gpu_integer_policy(field.name, field.values.dtype)
            if flag is None:
                excluded.append(field)
                continue
            pointer = _cuda_pointer(field.values, field.name)
            # collect_state_fields reaches the cumulus-tendency and
            # microphysics-accumulator buffers twice, once as a driver
            # attribute and once as restart-persistent scratch.  A second
            # descriptor over the identical allocation, extent and rule reads
            # the same bytes to the same verdict, and atomicMin already keeps
            # the first registration, so scanning it again buys nothing.  The
            # auxiliary enters the key by object identity rather than by base
            # pointer: anything short of provably the same scan is kept.
            key = (pointer, int(field.values.size), flag, field.rule,
                   id(field.auxiliary), field.aux_mode, field.plane_size)
            if key in seen:
                continue
            seen.add(key)
            fields.append(field)
            storage_flags.append(flag)
            pointers_by_field.append(pointer)
        fields_tuple = tuple(fields)
        pointers = np.zeros(self._ceiling, dtype=np.uint64)
        auxiliaries = np.zeros(self._ceiling, dtype=np.uint64)
        sizes = np.zeros(self._ceiling, dtype=np.uint64)
        bounds = np.zeros((self._ceiling, 2), dtype=np.float32)
        flags = np.zeros(self._ceiling, dtype=np.uint32)
        planes = np.zeros(self._ceiling, dtype=np.uint32)
        bits = np.zeros(self._ceiling, dtype=np.uint64)
        rims = np.zeros((self._ceiling, 3), dtype=np.uint32)
        for index, field in enumerate(fields):
            pointers[index] = pointers_by_field[index]
            sizes[index] = int(field.values.size)
            rule = field.rule
            rims[index] = _rim_geometry(field)
            flags[index] |= storage_flags[index]
            if rule.lower is not None:
                flags[index] |= _LOWER
                bounds[index, 0] = np.float32(rule.lower)
            if rule.upper is not None:
                flags[index] |= _UPPER
                bounds[index, 1] = np.float32(rule.upper)
            if rule.strict_lower:
                flags[index] |= _STRICT_LOWER
            if field.auxiliary is not None:
                if np.dtype(field.auxiliary.dtype) != np.dtype(np.float32):
                    raise TypeError(
                        f"GPU health auxiliary for {field.name!r} must be "
                        "float32")
                auxiliaries[index] = _cuda_pointer(
                    field.auxiliary, field.name + " auxiliary")
                flags[index] |= _ADD_AUX
                if field.aux_mode == "level":
                    flags[index] |= _AUX_LEVEL
                    planes[index] = field.plane_size
                elif field.aux_mode != "direct":
                    raise ValueError(
                        f"unknown auxiliary mode {field.aux_mode!r}")
            bits[index] = rule.status_bit
        self._ptr.set(_word_floats(_u64_words(pointers)))
        self._aux.set(_word_floats(_u64_words(auxiliaries)))
        self._size.set(_word_floats(_u64_words(sizes)))
        self._bounds.set(bounds)
        self._flags.set(_word_floats(flags))
        self._planes.set(_word_floats(planes))
        self._status.set(_word_floats(_u64_words(bits)))
        self._rim.set(rims.view(np.float32))
        initial = np.asarray([0, (1 << 64) - 1], dtype=np.uint64)
        self._result.set(_word_floats(_u64_words(initial)))
        scanned = sizes[:len(fields)]
        self._chunk = max(_MIN_CHUNK_ELEMENTS,
                          int(scanned.sum()) // _CHUNK_TARGET_BLOCKS)
        largest = int(scanned.max(initial=0))
        self._chunk_count = max(1, (largest + self._chunk - 1) // self._chunk)
        self.fields = fields_tuple
        self.excluded_integer_fields = tuple(excluded)

    def validate(self, *, phase: str | None = None) -> ValidationReport:
        """Launch the fused gate once and return its compact host result."""
        import cupy as cp
        from woof.core.kernels import get_kernel

        self._refresh()
        nfield = len(self.fields)
        if not nfield:
            return ValidationReport(True, 0, phase=phase)
        kernel = get_kernel("health", "validate_full_state")
        kernel((self._chunk_count, nfield), (256,), (
            self._ptr, self._aux, self._size, self._bounds, self._flags,
            self._planes, self._status, self._rim, self._result, np.int32(nfield),
            np.uint64(self._chunk)))
        words = cp.asnumpy(self._result).view(np.uint64)
        status, packed = (int(words[0]), int(words[1]))
        if packed == (1 << 64) - 1:
            return ValidationReport(True, status, phase=phase)
        field_number = packed >> 48
        flat = packed & _INDEX_MASK
        if field_number >= nfield:
            raise RuntimeError(
                f"health kernel returned invalid field id {field_number}")
        field = self.fields[field_number]
        value = float(cp.asnumpy(field.values.reshape(-1)[flat]))
        if field.auxiliary is not None:
            if field.aux_mode == "direct":
                value += float(cp.asnumpy(
                    field.auxiliary.reshape(-1)[flat]))
            else:
                value += float(cp.asnumpy(
                    field.auxiliary.reshape(-1)[flat // field.plane_size]))
        return ValidationReport(
            False, status, field.name,
            tuple(int(i) for i in np.unravel_index(
                flat, tuple(field.values.shape))),
            flat, value, _reason(value, field.rule,
                                 shape=field.values.shape, flat=flat), phase)

    def require_healthy(self, *, phase: str | None = None
                        ) -> ValidationReport:
        report = self.validate(phase=phase)
        if not report.ok:
            raise HealthCheckError(report)
        # The compact result has synchronized the scan. These descriptors
        # are rebuilt from the current state on every validate(), so keeping
        # their array references now only pins replaced physics generations
        # across the next step. Keep failed scans intact for diagnostics.
        self.fields = ()
        self.excluded_integer_fields = ()
        return report

    def phase_observer(self, phase: str) -> ValidationReport:
        """Dycore-compatible debug observer: validate at a named phase."""
        return self.require_healthy(phase=phase)


def benchmark_validator(state: Any, *, repeats: int = 20,
                        step_wall_seconds: float | None = None,
                        extra_tables: Mapping[str, Any] | None = None
                        ) -> dict[str, float | int | None]:
    """Controller-only GPU pre-flight hook for the <=2% overhead gate.

    This function executes CUDA and is intentionally never called by CPU
    tests or lane workers.  The controller supplies the measured d01 step
    wall time to obtain ``overhead_percent``.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    import cupy as cp

    validator = StateHealthValidator(state, extra_tables=extra_tables)
    validator.require_healthy(phase="validator-benchmark-warmup")
    cp.cuda.runtime.deviceSynchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        validator.require_healthy(phase="validator-benchmark")
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    per_call = elapsed / repeats
    overhead = (None if step_wall_seconds is None else
                100.0 * per_call / float(step_wall_seconds))
    return {"repeats": repeats, "elapsed_seconds": elapsed,
            "seconds_per_validation": per_call,
            "step_wall_seconds": step_wall_seconds,
            "overhead_percent": overhead}


__all__ = [
    "FieldRule", "FSBM_HEALTH_FIELDS", "HealthCheckError", "HealthField",
    "MAX_HEALTH_FIELDS", "STATUS_BITS", "StateHealthValidator",
    "THETA_CEILING_AT_REFERENCE_LID_K", "THETA_CEILING_REFERENCE_LID_PA",
    "ValidationReport",
    "benchmark_validator", "collect_state_fields", "cuda_source",
    "field_from_array", "gpu_integer_policy", "health_field_ceiling",
    "health_field_ceiling_for_config", "rule_for_field",
    "theta_ceiling_for_lid",
    "validate_fields_cpu", "validate_state_cpu", "validate_store_fields",
]


def prepared_store_health_auxiliaries(bundle, cfg) -> dict[str, object]:
    """The DOMAIN-shaped form of the two gated fields' auxiliary arrays.

    ``collect_state_fields`` checks ``thp`` against the base-state theta and
    ``mup`` against the base-state dry mass, and on the store-direct road the
    descriptors come off the slab-height template, so both auxiliaries arrive
    one slab tall.  The domain's own are in ``bundle.base``, which
    :class:`woof.ingest.prepared_store.PreparedStore` publishes un-windowed
    for exactly this kind of reader.

    float32 and not float64, because :meth:`woof.core.state.DomainState
    .load_base` is what put these numbers on the card on the resident road and
    it casts them: comparing the same field against an FP64 auxiliary would be
    a second instrument, one rounding apart from the one being matched.
    """

    def host(value):
        if hasattr(value, '__cuda_array_interface__'):
            import cupy as cp
            value = cp.asnumpy(value)
        return np.asarray(value, dtype=np.float32)

    base = bundle.base
    out: dict[str, object] = {}
    thb = getattr(base, "thb", None)
    if thb is not None:
        out["thp"] = host(thb)
    mub = getattr(base, "mub", None)
    if mub is not None:
        # ``load_base``'s own branch: flat terrain carries a SCALAR dry mass
        # and fills the (ny, nx) plane with it, terrain carries the plane.
        out["mup"] = (np.full((int(cfg.ny), int(cfg.nx)), float(mub),
                              dtype=np.float32) if np.ndim(mub) == 0
                      else host(mub))
    return out



class StoreHealthValidator:
    """Existing whole-domain rules over a canonical host or device store."""

    def __init__(self, bundle, cfg):
        from woof.core.streaming import streamed_store_inventory

        self.bundle = bundle
        self.domain_shape = (int(cfg.ny), int(cfg.nx))
        self.carriers = streamed_store_inventory()(bundle.template, None)
        self.device = any(hasattr(value, '__cuda_array_interface__')
                          for value in bundle.store.values())
        self.auxiliaries = (
            {'thp': bundle.template.thb, 'mup': bundle.template.mub2d}
            if self.device else prepared_store_health_auxiliaries(bundle, cfg))
        self._device_validator = (StateHealthValidator(
            bundle.template, field_provider=self._device_fields)
            if self.device else None)

    @property
    def host_scan_bytes(self) -> int:
        """Bytes :meth:`validate` reads on the CPU: the whole host store,
        or none when the store is on the device and the kernel scans it."""
        if self.device:
            return 0
        return sum(int(getattr(value, 'nbytes', 0) or 0)
                   for value in self.bundle.store.values())

    def _device_fields(self):
        fields, self.coverage = _store_health_descriptors(
            self.bundle.template, self.carriers, self.bundle.store,
            domain_shape=self.domain_shape, auxiliaries=self.auxiliaries,
            p_top=getattr(self.bundle.base, 'p_top', None), device=True)
        return fields

    def validate(self, *, phase=None):
        if self._device_validator is not None:
            started = time.perf_counter()
            report = self._device_validator.validate(phase=phase)
            self.coverage['seconds'] = time.perf_counter() - started
            return report
        report, self.coverage = validate_store_fields(
            self.bundle.template, self.carriers, self.bundle.store,
            domain_shape=self.domain_shape, auxiliaries=self.auxiliaries,
            p_top=getattr(self.bundle.base, "p_top", None), phase=phase)
        return report

    def require_healthy(self, *, phase=None):
        report = self.validate(phase=phase)
        if not report.ok:
            raise HealthCheckError(report)
        return report


class _AttachedStoreBundle:
    """A streamed domain's store as the health rules read it: drained per read.

    ``StreamedDomain.store`` drains before it hands the mapping out (the
    deferred seam's scatter tail on one card; every slab of every rank on the
    ranked road, :attr:`tilestream.ranks.RankedRun.store`).  Breakage this
    prevents: the gate captured ``attached.store`` once, at construction, and
    scanned that mapping at every later phase with no drain.  2.8.7's
    ``StreamedDomain.impose_clock`` drained after every step (67e213588), which
    kept the mirror current by accident; once that per-step drain was removed
    (it made a 2-card run 0.78x of one card), the periodic and --health-debug
    gates of a 2-card run scanned the host mirror of the last drain while a
    NaN sat on the slabs.
    """

    def __init__(self, owner, template, base):
        self._owner = owner
        self.template = template
        self.base = base

    @property
    def store(self):
        return self._owner.store


def health_validator_for_domain(model, node):
    """Validate the domain's canonical state after any initialization route."""
    from types import SimpleNamespace

    attached = getattr(node.state, '_streamed_domain', None)
    if attached is not None:
        template = getattr(attached, 'template', None)
        if template is None:
            template = node.state
        # Classification comes from a real device template. Base arrays and
        # carriers belong to the current footprint, including after a move.
        geography = getattr(attached, '_geography', None) or {}
        base = SimpleNamespace(
            thb=geography.get('setup/thb', node.state.thb),
            mub=geography.get('setup/mub2d', node.state.mub2d),
            p_top=getattr(node.state, 'p_top', None))
        return StoreHealthValidator(
            _AttachedStoreBundle(attached, template, base), node.cfg.run)
    prepared = getattr(model, '_prepared_by_grid_id', {}).get(node.cfg.grid_id)
    bundle = getattr(prepared, 'streamed_store', None)
    if bundle is None:
        return StateHealthValidator(node.state)
    return StoreHealthValidator(bundle, node.cfg.run)

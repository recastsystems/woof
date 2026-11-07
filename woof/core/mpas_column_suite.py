"""Selectable MPAS column-physics suite (``run_mpas_column_suite``).

WHY A SECOND MODULE.  :mod:`woof.core.mpas_column_batch` is the pinned
2026-08-10 counterparty contract: its phase-1 chain is hard-wired to legacy
RRTMG, the revised MM5 surface layer, Noah-MP and YSU
(``mpas_column_batch.py`` RunConfig at its constructor), and the MPAS port
pins that file's bytes by SHA-256.  A dispatch arm inside it would move the
bytes every existing execution anchor is keyed to.  This module is the
same seam with those four slots turned into SELECTORS, and nothing else
changed: the class subclasses :class:`MpasColumnBatchPhysics`, reuses its
species rows, phase-2 microphysics, restart law and bookkeeping, and
overrides only what a selector touches -- the configuration the driver is
built from, the soil column it is handed, the drag statistics, and where
phase 1 reads its raw momentum rates.  The frozen module's bytes do not
move, and a configuration that selects the frozen four is still served by
the frozen module (:func:`run_mpas_column_suite` refuses to duplicate it).

THE SELECTORS (each a table row; adding a scheme is a row, not a branch):

* ``radiation_scheme``: ``"rrtmg_legacy"`` (ra_physics=4, legacy RRTMG).
* ``surface_layer_scheme``: ``"revised_mm5"`` (1) or ``"mynn"`` (5).
* ``land_surface_scheme``: ``"noahmp"`` (4, four soil layers) or ``"ruc"``
  (3, nine soil LEVELS at WRF's own RUC depths,
  :func:`woof.core.ruc.ruc_soil_geometry`).
* ``pbl_scheme``: ``"ysu"`` (1) or ``"mynn"`` (5, MYNN-EDMF at the
  configuration's own defaults).
* ``drag_scheme``: ``"off"`` or ``"gsl"`` (gwd_opt=3, WRF's GSL drag suite,
  :func:`woof.core.terrain_drag.launch_gwdo_gsl`): large-scale gravity-wave
  drag and blocking from the large-scale statistics, small-scale
  gravity-wave drag and form drag from the small-scale ones.  The drag rides
  the PBL slot exactly as WRF's PBL driver adds it, on the PBL cadence.

Pairing is WRF's: YSU needs the revised MM5 surface layer, and the MYNN
surface layer needs the MYNN PBL (both refused by name otherwise).

RAW MOMENTUM RATES.  The frozen seam reads raw du/dv from YSU's retained
output (``driver.last_ysu``), which no other PBL scheme fills.  Every PBL
scheme in the driver ends its due call in ``PhysicsDriver._couple_pbl_slot``,
which copies the slot's raw A-grid rates -- AFTER the gwd_opt drag has been
added -- into ``driver.pbl_raw_rates`` (the seam always runs a positive
``bldt``, so those buffers exist).  Phase 1 here reads du/dv from them, so
YSU, MYNN and the drag are one code path.  Those buffers persist across
non-due steps and are on the driver restart manifest, which is the hold the
frozen seam builds by hand.

GEOMETRY THE DRAG NEEDS.  The column state carries no eta coordinate, so
``znu`` is formed from the nominal interface heights through the
ICAO standard atmosphere (the 6.5 K/km troposphere to 11 km, isothermal
216.65 K above), normalized between the nominal surface and ``p_top``.  The drag reads it
only for ``kpblmax`` (the highest level with znu > 0.6, WRF
module_bl_gwdo_gsl.F:162-164).  MPAS winds are earth-relative, so the drag
sees ``sina = 0, cosa = 1`` exactly as native MPAS hands its own drag
(mpas_atmphys_driver_gwdo.F, gwdo_from_MPAS).

THE GENERIC ENTRY (``run_mpas_column_physics``, 2026-10-11).  The five
selector tables above admit what somebody typed into them.  The generic
entry at the foot of this module has no scheme table at all: its selection
is one mapping of RunConfig physics fields, the engine's own
``validate_run_config`` and ``validate_physics_capabilities`` judge it, the
transported species and scheme-owned state come from the engine's own
allocator (a host ``DomainState`` probe), the surface slots from
``physics_inventory.microphysics_scratch_slots``, the soil depth from
``config.soil_layer_count`` and the orographic statics from
``static.orographic.required_static_fields``.  Whatever
``PhysicsDriver.compute`` dispatches, it runs.  The slot-table entry above
is kept so a caller written against it keeps working.
"""

from __future__ import annotations

import dataclasses
import inspect
import numbers
from datetime import datetime
from types import MappingProxyType

import numpy as np

from woof.core.mpas_column_batch import (
    _GRAUPEL_SCHEMES,
    _MP_PHYSICS_BY_SCHEME,
    _PHASE1,
    _PHASE2,
    _SPECIES_BY_SCHEME,
    _STATE_RESTART_FIELDS_BY_SCHEME,
    _OUTPUT_BUFFERS,
    RESTART_SCHEMA,
    MpasColumnBatchPhysics,
    MpasColumnPhysicsTendencies,
    _ColumnBatchState,
    _interface_weights,
    _require_positive,
    resolve_column_batch_cadences,
)

__all__ = [
    "COLUMN_SEAM_SETTINGS",
    "ColumnPhysicsPlan",
    "MPAS_DEFAULT_PHYSICS",
    "MpasColumnPhysics",
    "PHYSICS_RESTART_SCHEMA",
    "column_physics_plan",
    "physics_selector_menu",
    "run_mpas_column_physics",
    "DRAG_SCHEMES",
    "MpasColumnSuiteTendencies",
    "FROZEN_SELECTION",
    "GENERATION_SELECTORS",
    "LAND_SURFACE_SCHEMES",
    "PBL_SCHEMES",
    "RADIATION_SCHEMES",
    "SURFACE_LAYER_SCHEMES",
    "MpasColumnSuitePhysics",
    "orographic_statistic_names",
    "resolve_generation",
    "resolve_suite_selection",
    "run_mpas_column_suite",
]

#: Selector rows.  Each value is the RunConfig assignment the row makes.
RADIATION_SCHEMES = MappingProxyType({
    "rrtmg_legacy": {"ra_physics": 4, "ra_rrtmg_variant": "legacy"},
})
SURFACE_LAYER_SCHEMES = MappingProxyType({
    "revised_mm5": {"sf_sfclay_physics": 1},
    "mynn": {"sf_sfclay_physics": 5},
})
LAND_SURFACE_SCHEMES = MappingProxyType({
    "noahmp": {"sf_surface_physics": 4, "num_soil_layers": 4},
    "ruc": {"sf_surface_physics": 3, "num_soil_layers": 9},
})
PBL_SCHEMES = MappingProxyType({
    "ysu": {"bl_pbl_physics": 1},
    "mynn": {"bl_pbl_physics": 5},
})
DRAG_SCHEMES = MappingProxyType({
    "off": {"gwd_opt": 0},
    "gsl": {"gwd_opt": 3},
})

#: The four slots the frozen ``mpas_column_batch`` hard-wires.  A selection
#: equal to this is the frozen seam's configuration and is served by the
#: frozen module, never re-served here under a second identity.
FROZEN_SELECTION = MappingProxyType({
    "radiation_scheme": "rrtmg_legacy",
    "surface_layer_scheme": "revised_mm5",
    "land_surface_scheme": "noahmp",
    "pbl_scheme": "ysu",
    "drag_scheme": "off",
})

_SELECTOR_TABLES = MappingProxyType({
    "radiation_scheme": RADIATION_SCHEMES,
    "surface_layer_scheme": SURFACE_LAYER_SCHEMES,
    "land_surface_scheme": LAND_SURFACE_SCHEMES,
    "pbl_scheme": PBL_SCHEMES,
    "drag_scheme": DRAG_SCHEMES,
})

#: Restart schema of THIS seam.  Distinct from the frozen one so a payload
#: of one can never restore into the other.
SUITE_RESTART_SCHEMA = "mpas-column-suite-v1"

#: Generation selectors the seam hands its RunConfig, each with the values
#: the engine's own ``validate_run_config`` admits and the slot it belongs
#: to.  A selector is passed only when its slot runs the scheme it selects
#: a generation OF: a ``bl_mynn_version`` on a YSU seam would be recorded
#: and never read, which is configuration the run does not have.  Adding a
#: selector is a row here; an empty mapping is the RunConfig defaults, so a
#: seam built with no generation is byte-for-byte the seam before this
#: table existed.
GENERATION_SELECTORS = MappingProxyType({
    # Shared TC exchange runs in PhysicsDriver for both horizontal engines.
    "tc_flux_curve": {
        "values": ("off", "gz-option0", "gz-option1", "gz-option2", "gz-psu",
                   "gz-k2d0", "powell2003", "donelan2004", "bell2012",
                   "emanuel2003-adapted"),
        "requires": ("surface_layer_scheme", ("revised_mm5", "mynn"))},
    "tc_spray": {
        "values": ("off", "bao2000", "barr2023"),
        "requires": ("surface_layer_scheme", ("revised_mm5", "mynn"))},
    "aer_opt": {
        "values": (0, 3),
        "requires": ("radiation_scheme", ("rrtmg_legacy",))},
    "tempo_ml_for_bl_nc": {
        "values": (False, True),
        "requires": ("microphysics_scheme", ("tempo",))},
    "tempo_surface_emission_before": {
        "values": (False, True),
        "requires": ("microphysics_scheme", ("tempo",))},
    "thompson_version": {
        "values": ("wrf_461", "wrf_39_noaa"),
        "requires": ("microphysics_scheme", ("thompson_aero",))},
    "thompson_fork_snow_fall": {
        "values": ("blend", "wrf_39_noaa"),
        "requires": ("microphysics_scheme", ("thompson_aero",))},
    "bl_mynn_version": {
        "values": ("wrf_461", "gsd_41"),
        "requires": ("pbl_scheme", ("mynn",))},
    "bl_mynn_mixlength": {
        "values": (1, 2),
        "requires": ("pbl_scheme", ("mynn",))},
    # MYNN number-species mixing by family (woof/config.py
    # MYNN_QN_MIX_KEYS).  The mixed rates leave phase 1 through
    # MpasColumnPhysicsTendencies.scalar_rates; the engine's validator
    # refuses them on a microphysics that does not carry the family.
    "bl_mynn_mixaerosols": {
        "values": (0, 1),
        "requires": ("pbl_scheme", ("mynn",))},
    "bl_mynn_mixnumcon": {
        "values": (0, 1),
        "requires": ("pbl_scheme", ("mynn",))},
    "mynn_sfclay_variant": {
        "values": ("wrf_461", "gsl_wrf39"),
        "requires": ("surface_layer_scheme", ("mynn",))},
    "rrtmg_cloud_optics_form": {
        "values": ("wrf_461", "noaa_wrf39"),
        "requires": ("radiation_scheme", ("rrtmg_legacy",))},
    "rrtmg_aer3_type_source": {
        "values": ("rural", "landmask"),
        "requires": ("radiation_scheme", ("rrtmg_legacy",))},
})


@dataclasses.dataclass(frozen=True)
class MpasColumnSuiteTendencies(MpasColumnPhysicsTendencies):
    """The frozen phase-one result plus rates for non-mass species.

    ``scalar_rates`` maps a transported species' own row name to its raw
    A-grid rate (``[level, column]`` float32, seam-owned, valid until the
    next phase 1): MYNN's ``nwfa``/``nifa`` under bl_mynn_mixaerosols=1 and
    ``nc``/``ni`` under bl_mynn_mixnumcon=1 (woof/config.py
    MYNN_QN_MIX_KEYS).  Empty when nothing is mixed.  A subclass rather
    than a field on the frozen contract, whose module bytes the MPAS port
    pins.
    """

    scalar_rates: object = dataclasses.field(
        default_factory=lambda: MappingProxyType({}))
    #: Raw vertical-velocity rate on the interfaces (``[level + 1,
    #: column]``, m s-2) when a phase-one scheme forces ``w`` (the SASE
    #: closure does); None otherwise.  Generic entry only.
    dw: object = None


def resolve_generation(generation, *, microphysics_scheme, selection):
    """Validate a generation mapping against the selected slots.

    Pure host logic.  Returns the RunConfig assignments (a plain dict,
    empty for ``None``).  The engine's ``validate_run_config`` still judges
    the combined configuration afterwards; this table only refuses what it
    would accept silently -- a selector for a scheme the seam does not run.
    """
    if generation is None:
        return {}
    chosen = dict(generation)
    slots = dict(selection, microphysics_scheme=microphysics_scheme)
    out = {}
    for key, value in chosen.items():
        row = GENERATION_SELECTORS.get(key)
        if row is None:
            raise ValueError(
                f"generation selector {key!r} has no row: the seam passes "
                f"only {sorted(GENERATION_SELECTORS)}")
        if value not in row["values"]:
            raise ValueError(
                f"{key} must be one of {list(row['values'])}, got {value!r}")
        slot, schemes = row["requires"]
        if slots.get(slot) not in schemes:
            raise ValueError(
                f"{key}={value!r} selects a generation of {list(schemes)} "
                f"but this seam's {slot} is {slots.get(slot)!r}; it would be "
                "recorded and never read")
        out[key] = value
    return out


def orographic_statistic_names(drag_scheme: str) -> tuple[str, ...]:
    """The engine's static field names the selected drag reads."""
    from woof.static.orographic import GWD_FIELDS
    gwd_opt = DRAG_SCHEMES[drag_scheme]["gwd_opt"]
    return tuple(GWD_FIELDS.get(gwd_opt, ()))


def resolve_suite_selection(*, radiation_scheme, surface_layer_scheme,
                            land_surface_scheme, pbl_scheme,
                            drag_scheme) -> dict:
    """Validate one selection and return its RunConfig assignments.

    Pure host logic, so the CPU contract tests cover every refusal.
    """
    chosen = {"radiation_scheme": radiation_scheme,
              "surface_layer_scheme": surface_layer_scheme,
              "land_surface_scheme": land_surface_scheme,
              "pbl_scheme": pbl_scheme,
              "drag_scheme": drag_scheme}
    assignments: dict = {}
    for key, value in chosen.items():
        table = _SELECTOR_TABLES[key]
        if not isinstance(value, str) or value not in table:
            raise ValueError(
                f"{key} must be one of {sorted(table)}, got {value!r}: a "
                "scheme without a row has no configuration to build")
        assignments.update(table[value])
    if pbl_scheme == "ysu" and surface_layer_scheme != "revised_mm5":
        raise ValueError(
            "pbl_scheme='ysu' needs surface_layer_scheme='revised_mm5': "
            "WRF's YSU reads the revised MM5 layer's exchange terms "
            "(module_pbl_driver.F, bl_pbl_physics=1 with sf_sfclay=1)")
    if surface_layer_scheme == "mynn" and pbl_scheme != "mynn":
        raise ValueError(
            "surface_layer_scheme='mynn' needs pbl_scheme='mynn': the MYNN "
            "surface layer publishes the closure terms only the MYNN PBL "
            "reads")
    if chosen == dict(FROZEN_SELECTION):
        raise ValueError(
            "this selection is the frozen seam's own four "
            "(woof.core.mpas_column_batch.run_mpas_column_batch); serving "
            "it here would give one configuration two identities and two "
            "byte pins.  Construct the frozen seam instead")
    return {"selection": chosen, "config": assignments}


def _nominal_znu(z_interface_nominal_m, p_top_pa: float) -> np.ndarray:
    """An eta-like mass-level coordinate from the nominal interfaces.

    Hydrostatic standard atmosphere over the nominal column (float64 setup
    arithmetic, float32 storage like every other static here), normalized
    so the nominal surface is 1 and ``p_top`` is 0.  Only ``kpblmax``
    reads it.
    """
    z = np.asarray(z_interface_nominal_m, dtype=np.float64)
    zmid = 0.5 * (z[:-1] + z[1:]) - z[0]
    p0 = 101325.0
    p11 = p0 * (216.65 / 288.15) ** 5.25588
    p = np.where(
        zmid <= 11000.0,
        p0 * np.power(1.0 - 2.25577e-5 * np.minimum(zmid, 11000.0), 5.25588),
        p11 * np.exp(-(zmid - 11000.0) / 6341.62))
    znu = (p - float(p_top_pa)) / (p0 - float(p_top_pa))
    return np.clip(znu, 0.0, 1.0).astype(np.float32)


def _nominal_znw(z_interface_nominal_m, p_top_pa: float) -> np.ndarray:
    """:func:`_nominal_znu`'s coordinate on the nominal interfaces.

    The same standard atmosphere and normalization, evaluated at the
    ``n_levels + 1`` interfaces, so the bottom is 1 and the column runs
    down toward 0 at ``p_top`` (WRF ZNW).
    """
    z = np.asarray(z_interface_nominal_m, dtype=np.float64)
    zi = z - z[0]
    p0 = 101325.0
    p11 = p0 * (216.65 / 288.15) ** 5.25588
    p = np.where(
        zi <= 11000.0,
        p0 * np.power(1.0 - 2.25577e-5 * np.minimum(zi, 11000.0), 5.25588),
        p11 * np.exp(-(zi - 11000.0) / 6341.62))
    znw = (p - float(p_top_pa)) / (p0 - float(p_top_pa))
    return np.clip(znw, 0.0, 1.0).astype(np.float32)


class _SuiteColumnState(_ColumnBatchState):
    """The frozen column state plus what the selectable slots read."""

    def __init__(self, cfg, fnm, fnp, p_top, terrain_height, xp, *,
                 species, znu):
        super().__init__(cfg, fnm, fnp, p_top, terrain_height, xp,
                         species=species)
        ny, nx = cfg.ny, cfg.nx
        # Earth-relative winds: native MPAS hands its drag sina=0, cosa=1.
        self.sina = xp.zeros((ny, nx), dtype=np.float32)
        self.cosa = xp.ones((ny, nx), dtype=np.float32)
        self.znu = xp.asarray(znu, dtype=np.float32)


class MpasColumnSuitePhysics(MpasColumnBatchPhysics):
    """The frozen two-phase seam with selectable phase-1 slots."""

    def __init__(self, *, n_levels, n_columns, dt,
                 radiation_seconds, surface_pbl_seconds,
                 radiation_scheme, surface_layer_scheme,
                 land_surface_scheme, pbl_scheme, drag_scheme,
                 start_time, latitude_deg, longitude_deg,
                 terrain_height_m, z_interface_nominal_m,
                 p_top_pa, dx_m,
                 cumulus_seconds=None, cumulus_scheme=None,
                 microphysics_scheme="wsm6",
                 gf_ishallow=None, dx_column_m=None,
                 cumulus_active_column=None,
                 landmask=1.0, xland=None, ivgtyp=10, isltyp=6,
                 vegfra=50.0, tsk=300.0, tmn=285.0, xice=0.0, snow=0.0,
                 snow_depth=0.0, soil_temperature=None,
                 soil_moisture=None, xice_threshold=None,
                 fractional_seaice=None, orographic_statistics=None,
                 wsm6_hail_opt=0, generation=None, cell_area_m2=None,
                 physics_mode=None, physics_correct_skip=()):
        # ---- pure validation FIRST (no device allocation before it) ----
        # Correct mode (woof/correct_mode.py): None is the engine default
        # (correct; exact under GPUWM_WRF_EXACT=1).  The resolution authors
        # every correction key, so exact builds the config it always did.
        from woof import correct_mode
        correct = correct_mode.resolve(
            physics_mode, physics_correct_skip,
            source="MpasColumnSuitePhysics physics_mode")
        if not isinstance(n_levels, numbers.Integral) or n_levels < 3:
            raise ValueError(
                f"n_levels must be an integer >= 3, got {n_levels!r}")
        if not isinstance(n_columns, numbers.Integral) or n_columns < 1:
            raise ValueError(
                f"n_columns must be a positive integer, got {n_columns!r}")
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime (UTC)")
        if microphysics_scheme not in _SPECIES_BY_SCHEME:
            raise ValueError(
                f"microphysics_scheme must be one of "
                f"{sorted(_SPECIES_BY_SCHEME)}, got {microphysics_scheme!r}")
        resolved = resolve_suite_selection(
            radiation_scheme=radiation_scheme,
            surface_layer_scheme=surface_layer_scheme,
            land_surface_scheme=land_surface_scheme,
            pbl_scheme=pbl_scheme, drag_scheme=drag_scheme)
        selection, assignments = resolved["selection"], resolved["config"]
        generation_assignments = resolve_generation(
            generation, microphysics_scheme=microphysics_scheme,
            selection=selection)
        cadences = resolve_column_batch_cadences(
            dt=dt, radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme)
        p_top = _require_positive("p_top_pa", p_top_pa)
        dx = _require_positive("dx_m", dx_m)
        if int(wsm6_hail_opt) not in (0, 1):
            raise ValueError(
                f"wsm6_hail_opt must be 0 or 1, got {wsm6_hail_opt!r}")
        if microphysics_scheme != "wsm6" and int(wsm6_hail_opt) != 0:
            raise ValueError(
                "wsm6_hail_opt is a WSM6 hail-mode knob; on a "
                f"microphysics_scheme={microphysics_scheme!r} seam it would "
                "be silently ignored.  Leave it 0")
        # Sea-ice classification is the land surface's own knob in WRF:
        # Noah-MP takes an explicit threshold, RUC resolves its threshold
        # from fractional_seaice (woof/core/ruc_runtime.py).  Each refuses
        # the other's knob instead of reading it and ignoring it.
        if land_surface_scheme == "noahmp":
            if fractional_seaice is not None:
                raise ValueError(
                    "fractional_seaice is RUC's sea-ice switch; Noah-MP "
                    "takes xice_threshold")
            threshold = 0.5 if xice_threshold is None else float(
                xice_threshold)
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "xice_threshold must be a finite fraction in [0, 1], "
                    f"got {xice_threshold!r}")
            seaice = {}
        else:
            if xice_threshold is not None:
                raise ValueError(
                    "xice_threshold is Noah-MP's classification threshold; "
                    "RUC resolves its own from fractional_seaice "
                    "(0: 0.5, 1: 0.02)")
            if fractional_seaice not in (0, 1):
                raise ValueError(
                    "land_surface_scheme='ruc' needs fractional_seaice 0 or "
                    f"1 named explicitly, got {fractional_seaice!r}")
            threshold = None
            seaice = {"fractional_seaice": int(fractional_seaice)}
        if gf_ishallow is None:
            gf_ishallow = 1 if cumulus_scheme == "gf" else 0
        if gf_ishallow not in (0, 1):
            raise ValueError(
                f"gf_ishallow must be 0 or 1, got {gf_ishallow!r}")
        if gf_ishallow and cumulus_scheme != "gf":
            raise ValueError("gf_ishallow=1 requires cumulus_scheme='gf'")
        nz, ncol = int(n_levels), int(n_columns)
        fnm, fnp = _interface_weights(z_interface_nominal_m)
        if fnm.shape[0] != nz:
            raise ValueError(
                f"z_interface_nominal_m has {fnm.shape[0] + 1} interfaces "
                f"but n_levels={nz} needs {nz + 1}")
        lat = np.asarray(latitude_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(longitude_deg, dtype=np.float32).reshape(-1)
        if lat.shape != (ncol,) or lon.shape != (ncol,):
            raise ValueError(
                "latitude_deg/longitude_deg must have one value per "
                f"column ({ncol}), got {lat.shape}/{lon.shape}")
        terrain = np.asarray(terrain_height_m, dtype=np.float32).reshape(-1)
        if terrain.shape != (ncol,):
            raise ValueError(
                f"terrain_height_m must have shape ({ncol},), got "
                f"{terrain.shape}")
        dx_column = None
        if dx_column_m is not None:
            dx_column = np.asarray(dx_column_m, dtype=np.float32).reshape(-1)
            if dx_column.shape != (ncol,):
                raise ValueError(
                    f"dx_column_m must have one value per column ({ncol},), "
                    f"got {dx_column.shape}")
            if not np.all(np.isfinite(dx_column)) or np.any(dx_column <= 0.0):
                raise ValueError("dx_column_m must be finite and positive")
        from woof.core.gf import cumulus_active_column as _cu_active
        active_column = _cu_active(cumulus_active_column, ncol, cumulus_scheme)
        # The soil column: the land surface's own geometry, never a default.
        n_soil = int(assignments["num_soil_layers"])
        soil = {}
        for name, value in (("soil_temperature", soil_temperature),
                            ("soil_moisture", soil_moisture)):
            if value is None:
                raise ValueError(
                    f"{name} is required: land_surface_scheme="
                    f"{land_surface_scheme!r} integrates a {n_soil}-level "
                    "soil column and a defaulted profile is substituted "
                    "state")
            array = (value if hasattr(value, "__cuda_array_interface__")
                     else np.asarray(value, dtype=np.float32))
            if tuple(array.shape) != (n_soil, ncol):
                raise ValueError(
                    f"{name} must have shape ({n_soil}, {ncol}) for "
                    f"land_surface_scheme={land_surface_scheme!r}, got "
                    f"{tuple(array.shape)}")
            soil[name] = array
        # Orographic statistics: exactly the drag's field set, both ways.
        need = orographic_statistic_names(drag_scheme)
        given = {} if orographic_statistics is None else dict(
            orographic_statistics)
        if need and orographic_statistics is None:
            raise ValueError(
                f"drag_scheme={drag_scheme!r} reads {list(need)}; without "
                "them the drag would run on zeros, which turns it off "
                "without saying so")
        missing = [name for name in need if name not in given]
        extra = sorted(name for name in given if name not in need)
        if missing or extra:
            raise ValueError(
                f"orographic_statistics for drag_scheme={drag_scheme!r} must "
                f"carry exactly {list(need)} (missing {missing}, extra "
                f"{extra})")
        oro = {}
        for name in need:
            value = np.asarray(given[name], dtype=np.float32).reshape(-1)
            if value.shape != (ncol,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"orographic statistic {name} must be finite with one "
                    f"value per column ({ncol},), got {value.shape}")
            oro[name] = value.reshape(1, ncol)
        top_nominal = float(np.asarray(
            z_interface_nominal_m, dtype=np.float64)[-1])
        znu = _nominal_znu(z_interface_nominal_m, p_top)

        from woof.config import RunConfig
        from woof.physics_compat import RRTMG_VARIANT_LEGACY
        mp_physics = _MP_PHYSICS_BY_SCHEME[microphysics_scheme]
        aerosol_source = ({"mp28_aerosol_source": "synthetic"}
                          if microphysics_scheme == "thompson_aero" else {})
        radiation = dict(assignments)
        if radiation.pop("ra_rrtmg_variant") != "legacy":
            raise AssertionError("only the legacy RRTMG row exists")
        cfg = RunConfig(
            nx=ncol, ny=1, nz=nz, dx=dx, dy=dx,
            ztop=top_nominal, dt=float(cadences["dt"]),
            run_seconds=0.0, moist=True, mp_physics=mp_physics,
            wsm6_hail_opt=int(wsm6_hail_opt),
            **aerosol_source,
            **seaice,
            **radiation,
            ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
            cu_physics=cadences["cu_physics"],
            ishallow=int(gf_ishallow),
            radt=cadences["radt_minutes"],
            radt_minutes=cadences["radt_minutes"],
            bldt=cadences["bldt_minutes"],
            cudt_minutes=cadences["cudt_minutes"],
            **correct.settings)
        if generation_assignments:
            import dataclasses
            from woof.config import validate_run_config
            cfg = dataclasses.replace(cfg, **generation_assignments)
            validate_run_config(cfg)
        from woof.config import soil_layer_count
        if soil_layer_count(cfg) != n_soil:
            raise ValueError(
                f"the engine resolves {soil_layer_count(cfg)} soil layers "
                f"for land_surface_scheme={land_surface_scheme!r}, the row "
                f"declares {n_soil}")
        self._cfg = cfg
        self._correct_mode = correct
        self._cadences = cadences
        self._dt = float(cadences["dt"])
        self._nz, self._ncol = nz, ncol
        self._start_time = start_time
        self._microphysics_scheme = microphysics_scheme
        self._has_graupel = microphysics_scheme in _GRAUPEL_SCHEMES
        self._species = _SPECIES_BY_SCHEME[microphysics_scheme]
        self._state_restart_fields = (
            _STATE_RESTART_FIELDS_BY_SCHEME[microphysics_scheme])
        self._selection = dict(selection)
        self._identity = {
            "schema": SUITE_RESTART_SCHEMA,
            "batch_schema": RESTART_SCHEMA,
            "n_levels": nz, "n_columns": ncol, "dt": self._dt,
            "stepra": cadences["stepra"], "stepbl": cadences["stepbl"],
            "stepcu": cadences["stepcu"],
            "cumulus_scheme": cumulus_scheme,
            "start_time": start_time.isoformat(),
            "p_top_pa": p_top, "dx_m": dx,
            "gf_ishallow": int(gf_ishallow),
            "gf_dx_column": ("per-column" if dx_column is not None
                             else "scalar"),
            # Present only with a mask, so an unmasked identity is unchanged.
            **({} if active_column is None else {
                "gf_active_column": (f"per-column: {int(active_column.sum())}"
                                     f" of {ncol} active")}),
            "wsm6_hail_opt": int(wsm6_hail_opt),
            "xice_threshold": threshold,
            "fractional_seaice": seaice.get("fractional_seaice"),
            "xland_source": ("native" if xland is not None
                             else "derived-from-landmask"),
            "microphysics": microphysics_scheme,
            "selection": dict(selection),
            "n_soil_levels": n_soil,
        }
        if correct.applied:
            # Present only in correct mode, as in the frozen batch.
            self._identity["physics_correct"] = list(correct.applied)
        area = None
        if cell_area_m2 is not None:
            area = np.asarray(cell_area_m2, dtype=np.float64).reshape(-1)
            if area.shape != (ncol,) or not np.all(np.isfinite(area)) or np.any(area <= 0):
                raise ValueError("cell_area_m2 must be finite and positive with one value per column")
        if cfg.tempo_surface_emission_before and microphysics_scheme == "tempo" and area is None:
            raise ValueError("native TEMPO surface emissions require supplied cell_area_m2; using nominal spacing would change the source rate on a variable mesh")
        if generation_assignments:
            # Only a seam that selected a generation carries the key, so a
            # restart payload of a generation-less seam is unchanged.
            self._identity["generation"] = dict(generation_assignments)

        # ---- device construction from here on --------------------------
        import cupy as cp
        self._cp = cp
        self._cell_area_m2 = None if area is None else cp.asarray(area)

        state = _SuiteColumnState(cfg, fnm, fnp, p_top, terrain, cp,
                                  species=self._species, znu=znu)
        self._state = state
        self._species_buffers = {
            name: getattr(state, name) for name in self._species}

        def surface(values):
            if np.ndim(values) == 0:
                return values
            if hasattr(values, "__cuda_array_interface__"):
                return values.reshape(1, ncol)
            return np.asarray(values).reshape(1, ncol)

        def soil_column(values):
            if hasattr(values, "__cuda_array_interface__"):
                return values.reshape(values.shape[0], 1, ncol)
            values = np.asarray(values)
            return values.reshape(values.shape[0], 1, ncol)

        self._momentum_cumulus = (
            _capturing_momentum_cumulus() if cumulus_scheme == "ntiedtke" else None)
        self._driver = _initialize_suite_driver(
            state, cfg,
            landmask=surface(landmask),
            xland=None if xland is None else surface(xland),
            xice_threshold=threshold,
            tsk=surface(tsk),
            soil_temperature=soil_column(soil["soil_temperature"]),
            soil_moisture=soil_column(soil["soil_moisture"]),
            ivgtyp=surface(ivgtyp), isltyp=surface(isltyp),
            vegfra=surface(vegfra), tmn=surface(tmn),
            xice=surface(xice), snow=surface(snow),
            snow_depth=surface(snow_depth),
            glw=None,
            radiation_start_time=start_time,
            radiation_latitude=lat.reshape(1, ncol),
            radiation_longitude=lon.reshape(1, ncol),
            terrain_drag_static=(dict(oro, HGT_M=terrain.reshape(1, ncol))
                                 if oro else None),
            cumulus=self._momentum_cumulus)
        self.seaice_optics_seed = _seed_ruc_fractional_seaice_optics(
            self._driver, cfg, lat, start_time)
        if not {"du", "dv"} <= set(self._driver.pbl_raw_rates):
            raise RuntimeError(
                "the driver holds no raw PBL momentum rates "
                f"({sorted(self._driver.pbl_raw_rates)}); the seam runs a "
                "positive bldt precisely so it does, and without them phase "
                "1 has no A-grid momentum to return")

        self._aerosol_init = None
        if microphysics_scheme == "thompson_aero":
            # Same placeholder reset as the frozen seam (its docstring,
            # FIRST CONTACT): thompson_init runs for real at first contact.
            for name in ("nwfa", "nifa", "nwfa2d"):
                getattr(state, name)[...] = cp.float32(0.0)

        shape3 = (nz, 1, ncol)
        self._gf_rthdynten = cp.zeros(shape3, dtype=cp.float32)
        self._gf_rqvdynten = cp.zeros(shape3, dtype=cp.float32)
        self._driver.gf_rthdynten = self._gf_rthdynten
        self._driver.gf_rqvdynten = self._gf_rqvdynten
        if dx_column is not None:
            self._driver.gf_dx_column = cp.asarray(
                dx_column.reshape(1, ncol))
        if active_column is not None:
            self._driver.gf_active_column = cp.asarray(
                active_column.reshape(1, ncol))
        # The buffer set the frozen run_phase1 (inherited) fills: u, v,
        # the interface fields and a caller exner are read in place for
        # the call, not mirrored (mpas_column_batch, census item 10).
        self._in = {name: cp.zeros(shape3, dtype=cp.float32)
                    for name in ("theta", "temperature", "pressure",
                                 "dz", "rho")}
        self._atmosphere = None
        state.prepared_physics_atmosphere = self._supplied_atmosphere
        self._out = {name: cp.zeros((nz, ncol), dtype=cp.float32)
                     for name in _OUTPUT_BUFFERS}
        # Seam-owned rate buffers for the number species MYNN mixes on
        # this configuration (empty unless a mixing key is on and the
        # microphysics carries the family).  Not on the restart manifest:
        # the engine validator admits these keys only with a PBL call on
        # every model step, so every phase 1 rewrites them before any
        # read, and a restored seam's first phase 1 refills them.
        from woof.config import mynn_mixed_number_species
        mixed = mynn_mixed_number_species(cfg)
        missing = [name for name in mixed if name not in self._species]
        if missing:
            raise ValueError(
                f"MYNN mixes {missing}, which microphysics_scheme="
                f"{microphysics_scheme!r} does not transport; its rates "
                "would have no species to land on")
        self._scalar_out = {name: cp.zeros((nz, ncol), dtype=cp.float32)
                            for name in mixed}
        self.last_scalar_rates = MappingProxyType({})
        self._step_index = 0
        self._pending_phase = _PHASE1
        state.elapsed_seconds = 0.0

    # ------------------------------------------------------------------
    @property
    def mixed_number_species(self) -> tuple[str, ...]:
        """Species whose MYNN mixing rates phase 1 returns (row names)."""
        return tuple(getattr(self, "_scalar_out", None) or ())

    @property
    def selection(self) -> dict:
        """The five selector values this seam was built with."""
        return dict(self._selection)

    @property
    def generation(self) -> dict:
        """The generation selectors this seam handed its RunConfig."""
        return dict(self._identity.get("generation", {}))

    @property
    def n_soil_levels(self) -> int:
        return int(self._identity["n_soil_levels"])

    @property
    def last_noahmp_census(self):
        census = getattr(self._driver, "last_noahmp_census", None)
        return None if census is None else dict(census)

    @property
    def last_ruc_census(self):
        census = getattr(self._driver, "last_ruc_census", None)
        return None if census is None else dict(census)

    def soil_state(self) -> dict:
        """Device copies of the soil column (``[level, column]``)."""
        f = self._driver.fields
        ncol = self._ncol
        return {name: f[name].reshape(f[name].shape[0], ncol).copy()
                for name in ("tslb", "smois") if name in f}

    def run_phase1(self, **kwargs) -> MpasColumnPhysicsTendencies:
        """The frozen phase 1, with du/dv from the PBL slot's raw rates.

        Everything but the momentum source is the frozen implementation:
        the parent returns its result built on the seam-owned output
        buffers, and the parent's ``last_ysu`` read finds None under any
        PBL scheme other than YSU.  Momentum is then taken from
        ``driver.pbl_raw_rates`` -- the raw A-grid slot rates after the
        drag -- for every scheme, YSU included, so the slot has one
        momentum path whichever scheme filled it.
        """
        result = super().run_phase1(**kwargs)
        raw = self._driver.pbl_raw_rates
        nz, ncol = self._nz, self._ncol
        self._out["du"][...] = raw["du"].reshape(nz, ncol)
        self._out["dv"][...] = raw["dv"].reshape(nz, ncol)
        if self._momentum_cumulus is not None:
            momentum = self._momentum_cumulus.last_momentum
            if momentum is None:
                raise RuntimeError("the selected momentum-producing cumulus returned no raw A-grid pair")
            self._cp.add(self._out["du"], momentum[0].reshape(nz, ncol), out=self._out["du"])
            self._cp.add(self._out["dv"], momentum[1].reshape(nz, ncol), out=self._out["dv"])
        scalar_out = getattr(self, "_scalar_out", None) or {}
        if scalar_out:
            # MYNN's mixed number-species rates.  The driver holds them as
            # PhysicsTendencies.extra_scalars (physics.py _run_mynn_pbl),
            # mass-coupled by chm = c1h*mu + c2h, which is exactly 1.0 on
            # this state (module docstring), so they ARE the raw rates.
            # A step without a PBL call would hand back the previous
            # step's rates as this step's; the validator forbids that
            # cadence and this refuses it if it ever arrives.
            if not result.surface_pbl_ran:
                raise RuntimeError(
                    "MYNN number-species mixing needs a PBL call on every "
                    "phase 1; this step had none, so its rates would be "
                    "the previous step's")
            extra = getattr(self._driver.pbl_tendencies,
                            "extra_scalars", None) or {}
            if set(extra) != set(scalar_out):
                raise RuntimeError(
                    f"MYNN returned mixing rates for {sorted(extra)}; this "
                    f"configuration mixes {sorted(scalar_out)}")
            for name, buffer in scalar_out.items():
                buffer[...] = extra[name].reshape(nz, ncol)
            result = MpasColumnSuiteTendencies(
                **{item.name: getattr(result, item.name)
                   for item in dataclasses.fields(result)},
                scalar_rates=MappingProxyType(dict(scalar_out)))
        # The hex backend's frozen phase one keeps only the masses of this
        # result; it reads the extra rates here (hexcore
        # cuda_suite_physics_v841 _with_scalar_rates).
        self.last_scalar_rates = getattr(
            result, "scalar_rates", MappingProxyType({}))
        return result


def _capturing_momentum_cumulus():
    """Retain the native New Tiedtke A-grid pair before ARW face coupling.

    MPAS adds this pair to the PBL rates before edge reconstruction. The
    callable owns no parameterization arithmetic and is called every step.
    """
    from woof.core.ntiedtke import NewTiedtke

    class Capturing(NewTiedtke):
        last_momentum = None

        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if result.rucuten is None or result.rvcuten is None:
                raise RuntimeError("New Tiedtke returned no momentum pair for the MPAS seam")
            self.last_momentum = (result.rucuten, result.rvcuten)
            return result

    return Capturing()


def _seed_ruc_fractional_seaice_optics(driver, cfg, latitude_deg,
                                       start_time):
    """MPAS ``landuse_init_forMPAS``'s sea-ice branch for a RUC column seam.

    THE BREAKAGE THIS PREVENTS: a column seam builds its driver without a
    land-use initialization, so ALBEDO/EMISS start at the allocator's flat
    0.20/0.95 on every column.  Under ``fractional_seaice = 1`` the RUC arm
    de-blends them each call as the ice part of an ice/open-water mixture
    (module_surface_driver.F:3461-3473, ``(ALBEDO - (1-XICE)*0.08)/XICE``),
    so a 2 % ice cell handed 0.20 became an ice albedo of 6.1 and an ice
    emissivity below zero, and step 0 died with "RUC CUDA snow preparation
    alb must be below 1" (hex 60-15 km zoom globe, GFS 2025-05-19 12Z,
    HRRR-shape and rrfs_v2_dev suites; worked around by
    config_frac_seaice=false).

    MPAS-A v8.4 (mpas_atmphys_landuse.F:265-301) and WRF's landuse_init
    (module_physics_init.F:1979-1990) both seed a sea-ice cell
    (XICE >= the RUC threshold, 0.02 here) from the table's ice row and
    blend it with open water: ALBBCK = ALBD(ISICE)/100, EMBCK =
    SFEM(ISICE), ALBEDO = XICE*ALBBCK + (1-XICE)*0.08, EMISS = XICE*EMBCK
    + (1-XICE)*0.98.  This is that branch, with MPAS's per-cell season
    (winter for day < 105 or >= 288, flipped south of the equator) because
    a column seam is one MPAS mesh, not one WRF domain.  Other cells are
    left as they are.  Returns a small receipt, or None when the seam is
    not RUC with fractional sea ice.
    """
    if (int(cfg.sf_surface_physics) != 3
            or int(getattr(cfg, "fractional_seaice", 0)) != 1):
        return None
    import cupy as cp
    from woof.core.landuse import load_landuse_table
    from woof.core.ruc import ruc_ice_category

    params = driver.ruc_params
    fields = driver.fields
    table = load_landuse_table(params.dataset_identifier)
    isice = ruc_ice_category(params.vegetation)
    shape = fields["xice"].shape
    lat = np.asarray(latitude_deg, dtype=np.float32).reshape(shape)
    julday = int(start_time.timetuple().tm_yday)
    base = 2 if (julday < 105 or julday >= 288) else 1
    season = np.where(lat < np.float32(0.0), 3 - base, base)
    if table.luseas == 1:
        season = np.ones(shape, dtype=np.int64)
    row = table.values[season - 1, isice - 1]
    # LANDUSE.TBL is read as default REAL, then ALBD/100. in REAL.
    albbck_ice = cp.asarray(
        row[..., 0].astype(np.float32) / np.float32(100.0))
    embck_ice = cp.asarray(row[..., 2].astype(np.float32))
    xice = fields["xice"]
    ice = xice >= cp.float32(params.xice_threshold)
    water = cp.float32(1.0) - xice
    albedo = xice * albbck_ice + water * cp.float32(0.08)
    emiss = xice * embck_ice + water * cp.float32(0.98)
    for name, value in (("albbck", albbck_ice), ("embck", embck_ice),
                        ("albedo", albedo), ("emiss", emiss)):
        fields[name][...] = cp.where(ice, value, fields[name]).astype(
            cp.float32)
    return {"policy": "mpas-landuse-init-seaice-v1",
            "ice_category": int(isice),
            "xice_threshold": float(params.xice_threshold),
            "seeded_cells": int(cp.count_nonzero(ice))}


def _initialize_suite_driver(state, cfg, **kwargs):
    """``initialize_physics`` for the selected slots.

    Noah-MP takes the explicit sea-ice threshold; RUC refuses one (it
    resolves its own), so the argument is passed only to Noah-MP.
    """
    from woof.core.physics import initialize_physics
    if int(cfg.sf_surface_physics) != 4:
        kwargs.pop("xice_threshold")
    return initialize_physics(state, cfg, **kwargs)


def run_mpas_column_suite(**kwargs) -> MpasColumnSuitePhysics:
    """Construct the selectable MPAS column-physics seam (keyword-only)."""
    return MpasColumnSuitePhysics(**kwargs)


# ======================================================================
# THE GENERIC ENTRY: any selection the engine's own registry accepts.
# ======================================================================

#: Restart schema of the generic seam (distinct from the frozen seam's and
#: the slot-table seam's, so a payload of one never restores into another).
PHYSICS_RESTART_SCHEMA = "mpas-column-physics-v1"

#: RunConfig fields the seam owns.  The column grid and the cadences are
#: constructor arguments kept as exact step counts; a second value arriving
#: through the selection would leave the driver's STEPRA/STEPBL/STEPCU and
#: the seam's own counters describing two different runs.
_SEAM_OWNED_FIELDS = ("nx", "ny", "nz", "dx", "dy", "ztop", "dt",
                      "run_seconds", "moist", "radt", "radt_minutes", "bldt",
                      "cudt_minutes")

#: Settings a column seam supplies when the caller states none, each with
#: the selection it belongs to and the reason.  A row is a fact about what
#: a column seam IS, not about a scheme's physics.
COLUMN_SEAM_SETTINGS = (
    {"field": "mp28_aerosol_source", "value": "synthetic",
     "when": {"mp_physics": 28},
     "why": "the seam has no ingest path and stages no WIF climatology; "
            "thompson_init runs at first contact on the caller's own "
            "aerosol arrays (woof/core/mpas_column_batch.py, FIRST "
            "CONTACT)"},
)

#: The frozen seam's selection in RunConfig spelling: what
#: ``mpas_column_batch`` hard-wires, for a caller that wants "today's
#: default" without restating it.  ``cu_physics``/``ishallow`` are the
#: caller's (the frozen seam takes them as arguments too).
MPAS_DEFAULT_PHYSICS = MappingProxyType({
    "mp_physics": 6, "ra_physics": 4, "ra_rrtmg_variant": "rrtmg_legacy",
    "sf_sfclay_physics": 1, "sf_surface_physics": 4, "bl_pbl_physics": 1,
})

#: WRF's earth rotation rate (share/module_model_constants.F EOMEG, s-1),
#: for the Coriolis pair a column caller's latitude implies.
_EOMEG = 7.2921e-5

#: Host probe grid.  Small, and with ny != nx so a (ny, nx) trailing shape
#: is unambiguous.
_PROBE_NY, _PROBE_NX, _PROBE_NZ = 2, 3, 40

#: The six mass rates of the fixed phase-one result, and the driver
#: tendency component each is read from (None: no phase-one scheme forces
#: it, so it is structurally zero).
_FIXED_MASS_RATES = (("dqv", "rqv"), ("dqc", "rqc"), ("dqr", "rqr"),
                     ("dqi", "rqi"), ("dqs", "rqs"), ("dqg", None))

#: Microphysics domain-construction steps that need the caller's bound
#: state and geometry, so they run at the first phase 2 rather than on the
#: seam's zero placeholder buffers.  Keyed by ``mp_physics``; a scheme
#: without a row has no such step.
# Both aerosol-aware Thompson members derive surface emissions from the
# first real layer depth; placeholder geometry would divide by zero.
_FIRST_CONTACT = {28: "_thompson_first_contact",
                  38: "_thompson_first_contact",
                  900: "_tempo_first_contact"}


def physics_selector_menu() -> dict:
    """The registry's own inventory of implemented options, per component.

    ``{component: {"selector_keys": [...], "options": {option_id:
    {"selectors": {...}, "label": ..., "maturity": ...}}}}``.  A caller that
    resolves its own spellings onto this needs no scheme list of its own.
    """
    from woof.physics_registry import component_options, registry_view
    menu = {}
    for component_id, component in registry_view()["components"].items():
        options = {}
        for option_id, option in component_options(component_id).items():
            selectors = option.get("selectors")
            if not selectors:
                continue
            options[option_id] = {
                "selectors": dict(selectors),
                "label": option.get("label"),
                "maturity": option.get("maturity")}
        menu[component_id] = {
            "selector_keys": list(component.get("selector_keys", [])),
            "options": options}
    return menu


def _selection_assignments(physics) -> dict:
    """Validate the selection mapping's KEYS and apply the seam settings."""
    from woof.config import RunConfig
    from woof.microphysics_schemes import resolve_mp_physics
    if physics is None or not hasattr(physics, "items"):
        raise TypeError(
            "physics must be a mapping of RunConfig physics fields, e.g. "
            f"{dict(MPAS_DEFAULT_PHYSICS)}")
    chosen = dict(physics)
    defaults = {item.name: item.default
                for item in dataclasses.fields(RunConfig)}
    owned = sorted(set(chosen) & set(_SEAM_OWNED_FIELDS))
    if owned:
        raise ValueError(
            f"physics names {owned}, which the seam owns: the column grid "
            "and the cadences are constructor arguments held as exact step "
            "counts, and a second value here would describe a different "
            "run than the one the counters keep")
    unknown = sorted(set(chosen) - set(defaults))
    if unknown:
        raise ValueError(
            f"physics names {unknown}, which are not RunConfig fields; a "
            "setting nothing reads would be recorded as configuration the "
            "run does not have")
    if "mp_physics" in chosen:
        chosen["mp_physics"] = resolve_mp_physics(chosen["mp_physics"])
    for row in COLUMN_SEAM_SETTINGS:
        if row["field"] in chosen:
            continue
        if all(chosen.get(key, defaults[key]) == value
               for key, value in row["when"].items()):
            chosen[row["field"]] = row["value"]
    # The soil geometry is the land surface's own.  RunConfig's default
    # count belongs to one scheme; where the caller states none, the seam
    # requests the geometry the engine has validated for the selected one.
    from woof.config import (LAND_SURFACE_SOIL_LAYERS,
                              validated_soil_layer_count)
    land = chosen.get("sf_surface_physics", defaults["sf_surface_physics"])
    if ("num_soil_layers" not in chosen and isinstance(land, int) and land
            and LAND_SURFACE_SOIL_LAYERS.get(land) is not None):
        chosen["num_soil_layers"] = validated_soil_layer_count(land)
    return chosen


def _selection_config(chosen, *, nz, ncol, ny, dx, ztop, cadences):
    """The RunConfig of one selection, judged by the engine itself."""
    from woof.config import RunConfig, validate_run_config
    from woof.physics_compat import validate_physics_capabilities
    cfg = RunConfig(
        nx=ncol, ny=ny, nz=nz, dx=dx, dy=dx, ztop=ztop,
        dt=float(cadences["dt"]), run_seconds=0.0, moist=True,
        radt=cadences["radt_minutes"],
        radt_minutes=cadences["radt_minutes"],
        bldt=cadences["bldt_minutes"],
        cudt_minutes=cadences["cudt_minutes"],
        **chosen)
    validate_run_config(cfg)
    validate_physics_capabilities(cfg)
    return cfg


def _resolve_cadences(*, dt, radiation_seconds, surface_pbl_seconds,
                      cumulus_seconds, cu_physics) -> dict:
    """Exact step multiples for the three cadences (pure host logic).

    ``cumulus_seconds=None`` is the model step in WRF's own spelling
    (``cudt_minutes = 0``); a value is a hold of that length.  Which
    schemes carry a hold is the engine validator's to say.
    """
    from woof.core.mpas_column_batch import _cadence_steps
    dt = _require_positive("dt", dt)
    stepra = _cadence_steps("radiation_seconds", radiation_seconds, dt)
    stepbl = _cadence_steps("surface_pbl_seconds", surface_pbl_seconds, dt)
    resolved = {"dt": dt, "stepra": stepra, "stepbl": stepbl,
                "radt_minutes": stepra * dt / 60.0,
                "bldt_minutes": stepbl * dt / 60.0,
                "stepcu": 0, "cudt_minutes": 0.0}
    if not cu_physics:
        if cumulus_seconds is not None:
            raise ValueError(
                "cumulus_seconds names a cadence for a cumulus scheme and "
                "cu_physics is 0")
        return resolved
    if cumulus_seconds is None:
        resolved["stepcu"] = 1
        return resolved
    stepcu = _cadence_steps("cumulus_seconds", cumulus_seconds, dt)
    resolved["stepcu"] = stepcu
    resolved["cudt_minutes"] = stepcu * dt / 60.0
    return resolved


def _mirror_shape(shape, ncol):
    """A probe array's shape on the column grid, or None if it has none."""
    if len(shape) < 2 or tuple(shape[-2:]) != (_PROBE_NY, _PROBE_NX):
        return None
    return tuple(shape[:-2]) + (1, ncol)


def _allocator_layout(cfg) -> dict:
    """What the engine's own allocator gives this selection, as a table.

    Two host ``DomainState`` probes on a tiny grid: this configuration and
    the all-defaults (dry, physics-off) one.  Everything the first carries
    and the second does not is state the selection asked for -- transported
    species, radii, cross-step carriers, emission fields -- with the
    allocator's own initial value.  RK time copies (``name0`` beside
    ``name``) are the dycore's and are left out; a view into a stacked
    array (a bin scheme's species) is recorded as that view.
    """
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.moist import WRF_MOIST_ARRAY_SPECIES, moist_species
    probe_cfg = dataclasses.replace(cfg, nx=_PROBE_NX, ny=_PROBE_NY)
    probe = DomainState(probe_cfg, array_module=np)
    bare = DomainState(
        RunConfig(nx=_PROBE_NX, ny=_PROBE_NY, nz=cfg.nz, dx=cfg.dx,
                  dy=cfg.dy, ztop=cfg.ztop, dt=cfg.dt, run_seconds=0.0),
        array_module=np)
    species = tuple(moist_species(probe))
    added = {name: value for name, value in vars(probe).items()
             if not name.startswith("_")
             and (name not in vars(bare)
                  or (getattr(bare, name) is None and value is not None))}
    arrays, scalars, vectors = {}, {}, {}
    for name, value in added.items():
        if isinstance(value, np.ndarray):
            if _mirror_shape(value.shape, 1) is None:
                # Grid-free scheme state (Eta Ferrier's restart vector,
                # its two saturation tables and their step stamp,
                # state.py ETANEWinit): one copy per domain, so the column
                # state carries it as the allocator made it.  Checked
                # before the RK-copy rule: RK copies are mass-grid fields,
                # and etanew_tbpvs0 is a table of its own, not a copy of
                # etanew_tbpvs.
                vectors[name] = value.copy()
                continue
            if name.endswith("0") and name[:-1] in added:
                continue
            arrays[name] = value
        elif value is None or isinstance(value, (bool, int, float, str)):
            scalars[name] = value
    fields = {}
    for name, value in arrays.items():
        owner = None
        for other, stack in arrays.items():
            if (other != name and stack.ndim == value.ndim + 1
                    and np.shares_memory(stack, value)):
                index = next(i for i in range(stack.shape[0])
                             if np.shares_memory(stack[i], value))
                owner = (other, index)
        first = value.reshape(-1)[0] if value.size else np.float32(0.0)
        if owner is None and not np.all(value == first):
            raise RuntimeError(
                f"the allocator initializes {name} to a non-uniform field; "
                "a column seam cannot rebuild that from a probe")
        fields[name] = {"shape": tuple(value.shape[:-2]),
                        "dtype": value.dtype, "fill": first, "view": owner}
    missing = [name for name in species if name not in fields]
    if missing:
        raise RuntimeError(
            f"the engine transports {missing} for this selection but its "
            "allocator probe carries no such field on the mass grid")
    return {
        "species": species,
        "mass_species": tuple(name for name in species
                              if name in WRF_MOIST_ARRAY_SPECIES),
        "fields": fields, "scalars": scalars, "vectors": vectors,
    }


@dataclasses.dataclass(frozen=True)
class ColumnPhysicsPlan:
    """What one selection needs from its caller (pure host, no device)."""

    #: every registry selector key and every stated key -> resolved value
    physics: object
    #: transported scalars in engine order, starting qv/qc/qr
    species: tuple
    #: the species that are mixing-ratio masses (water loading)
    mass_species: tuple
    #: the rest: numbers, volumes, aerosols, bins
    number_species: tuple
    #: scheme state the seam owns and restarts (not transported)
    state_fields: tuple
    #: the effective radii among ``state_fields`` (micron, [level, column])
    radii: tuple
    #: accumulated surface buckets the microphysics binds (beside RAINC)
    surface_accumulators: tuple
    #: per-call surface fields a phase-2 receipt carries
    surface_receipts: tuple
    #: soil levels the land surface integrates (0: no land surface)
    n_soil_levels: int
    #: per-column orographic statics the selection reads
    static_fields: tuple
    #: step counts and WRF-minute cadences, or None when not judged
    cadences: object
    #: restart identity of the selection (hash it for a pin)
    identity: object
    #: the judged RunConfig (probe geometry unless built by the seam)
    config: object = dataclasses.field(repr=False, compare=False)
    #: the allocator table the column state is built from
    layout: object = dataclasses.field(repr=False, compare=False)


def _jsonable(value):
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def _plan_from_config(cfg, chosen, cadences) -> ColumnPhysicsPlan:
    from woof.config import soil_layer_count
    from woof.core.physics_inventory import microphysics_scratch_slots
    from woof.physics_registry import registry_view
    from woof.static.orographic import required_static_fields
    layout = _allocator_layout(cfg)
    species = layout["species"]
    stacks = {row["view"][0] for row in layout["fields"].values()
              if row["view"] is not None}
    state_fields = tuple(sorted(
        name for name, row in layout["fields"].items()
        if name not in species and row["view"] is None))
    slots = tuple(component for component, _slot
                  in microphysics_scratch_slots(int(cfg.mp_physics)))
    resolved = {}
    for component in registry_view()["components"].values():
        for key in component.get("selector_keys", []):
            if hasattr(cfg, key):
                resolved[key] = _jsonable(getattr(cfg, key))
    for key in chosen:
        resolved[key] = _jsonable(getattr(cfg, key))
    resolved = dict(sorted(resolved.items()))
    n_soil = (int(soil_layer_count(cfg))
              if int(cfg.sf_surface_physics) else 0)
    statics = tuple(required_static_fields(
        int(getattr(cfg, "topo_wind", 0) or 0),
        int(getattr(cfg, "gwd_opt", 0) or 0)))
    identity = {
        "schema": PHYSICS_RESTART_SCHEMA, "batch_schema": RESTART_SCHEMA,
        "physics": dict(resolved),
        "species": list(species), "n_soil_levels": n_soil,
        "static_fields": list(statics),
    }
    from woof.config import radiation_scheme_ids
    if radiation_scheme_ids(cfg)[0] == 31:
        # This is a caller obligation at every RK stage, beside the column
        # thermal call. Keep it in the restart identity as well as the plan.
        identity["rk_momentum"] = "held-suarez-edge-v1"
    return ColumnPhysicsPlan(
        physics=MappingProxyType(resolved),
        species=species,
        mass_species=layout["mass_species"],
        number_species=tuple(name for name in species
                             if name not in layout["mass_species"]),
        state_fields=state_fields,
        radii=tuple(name for name in ("effc", "effi", "effs", "effr")
                    if name in state_fields and name not in stacks),
        surface_accumulators=tuple(
            name.upper() for name in slots if name.endswith("nc")),
        surface_receipts=tuple(
            name for name in slots if not name.endswith("nc")),
        n_soil_levels=n_soil, static_fields=statics,
        cadences=(None if cadences is None
                  else MappingProxyType(dict(cadences))),
        identity=MappingProxyType(identity),
        config=cfg, layout=layout)


def column_physics_plan(physics, *, n_levels=None, dt=None,
                        radiation_seconds=None, surface_pbl_seconds=None,
                        cumulus_seconds=None) -> ColumnPhysicsPlan:
    """Judge one selection and say what it needs (pure host, no device).

    With only ``physics`` the selection is judged on a probe grid and a
    probe cadence (every scheme each step), which decides everything but
    ``cadences`` (returned None): species, state, soil depth and statics
    do not depend on the grid or the step.  Pass ``n_levels``, ``dt`` and
    the cadences to have those judged too.
    """
    chosen = _selection_assignments(physics)
    timed = dt is not None
    if timed and (radiation_seconds is None or surface_pbl_seconds is None):
        raise ValueError(
            "dt was given without radiation_seconds and "
            "surface_pbl_seconds; pass all three or none")
    if not timed and any(value is not None for value in (
            radiation_seconds, surface_pbl_seconds, cumulus_seconds)):
        raise ValueError("a cadence was given without dt")
    step = float(dt) if timed else 60.0
    cadences = _resolve_cadences(
        dt=step,
        radiation_seconds=radiation_seconds if timed else step,
        surface_pbl_seconds=surface_pbl_seconds if timed else step,
        cumulus_seconds=cumulus_seconds if timed else None,
        cu_physics=int(chosen.get("cu_physics", 0) or 0))
    if not timed:
        # This probe discovers state and species without offering a timed
        # seam. Use the native every-step spelling: a positive bldt can
        # require held restart tendencies that this caller never requested.
        # Explicit timed plans still judge the seam's positive cadence.
        cadences["bldt_minutes"] = 0.0
    nz = _PROBE_NZ if n_levels is None else int(n_levels)
    cfg = _selection_config(chosen, nz=nz, ncol=_PROBE_NX, ny=_PROBE_NY,
                            dx=12000.0, ztop=20000.0, cadences=cadences)
    return _plan_from_config(cfg, chosen, cadences if timed else None)


class _GenericColumnState(_SuiteColumnState):
    """The column state with the engine allocator's own scheme fields."""

    def __init__(self, cfg, fnm, fnp, p_top, terrain_height, xp, *, znu,
                 znw, layout, latitude_deg):
        super().__init__(cfg, fnm, fnp, p_top, terrain_height, xp,
                         species=("qv", "qc", "qr"), znu=znu)
        ny, nx = cfg.ny, cfg.nx
        nz = cfg.nz
        # The interface partner of ``znu`` (an ARW state carries both; the
        # GFDL/Eta radiation tables read the pair, gfdleta.py GFDLTables).
        self.znw = xp.asarray(znw, dtype=np.float32)
        # Grid-free scheme state, one copy per domain as allocated.
        for name, value in layout["vectors"].items():
            setattr(self, name, xp.asarray(value.copy()))
        # The grid facts an ARW state carries and a scheme may read, in
        # the form they take on this state: identity mass coupling on the
        # interface levels as on the mass levels (c1f = 1, c2f = 0, beside
        # the base state's c1h = 1, c2h = 0), and the Coriolis pair of the
        # caller's own latitudes (WRF F = 2*EOMEG*sin(lat),
        # E = 2*EOMEG*cos(lat)).
        self.c1f = xp.ones(nz + 1, dtype=np.float32)
        self.c2f = xp.zeros(nz + 1, dtype=np.float32)
        latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
        self.f = xp.asarray(
            (2.0 * _EOMEG * np.sin(latitude)).astype(np.float32)
        ).reshape(ny, nx)
        self.e = xp.asarray(
            (2.0 * _EOMEG * np.cos(latitude)).astype(np.float32)
        ).reshape(ny, nx)
        fields = layout["fields"]
        # The frozen base state seeds a radius triple and scheme carriers
        # by scheme id; this state takes all of that from the allocator.
        for name in ("effc", "effi", "effs", "qs", "qg", "th_old", "qv_old",
                     "nwfa2d", "nifa2d"):
            if name not in fields and hasattr(self, name):
                setattr(self, name, None)
        for name, value in layout["scalars"].items():
            if not hasattr(self, name):
                setattr(self, name, value)
        for name, row in fields.items():
            if row["view"] is not None or name in ("qv", "qc", "qr"):
                continue
            setattr(self, name, xp.full(row["shape"] + (ny, nx), row["fill"],
                                        dtype=row["dtype"]))
        for name, row in fields.items():
            if row["view"] is not None:
                owner, index = row["view"]
                setattr(self, name, getattr(self, owner)[index])


def _capture_cumulus_momentum(driver, sink) -> None:
    """Keep the raw A-grid momentum pair of whichever cumulus scheme runs.

    ``PhysicsDriver`` couples a ``CumulusResult``'s ``rucuten/rvcuten``
    onto ARW's C-grid faces; an A-grid caller needs the pair before that.
    The driver's own callable is given a subclass of its own class whose
    call records the pair (None for a scheme that returns none), so the
    scheme's arithmetic, attributes and restart surface are untouched and
    no scheme is named here.
    """
    adapter = driver.cumulus_callable
    if adapter is None:
        return
    base = type(adapter)

    def __call__(self, **kwargs):
        result = base.__call__(self, **kwargs)
        pair = (getattr(result, "rucuten", None),
                getattr(result, "rvcuten", None))
        sink["pair"] = None if pair[0] is None else pair
        return result

    adapter.__class__ = type(base.__name__, (base,),
                             {"__call__": __call__, "__slots__": (),
                              "__module__": base.__module__})


class MpasColumnPhysics(MpasColumnBatchPhysics):
    """The two-phase column seam for ANY selection the engine accepts.

    ``physics`` is one mapping of RunConfig physics fields.  Nothing here
    lists schemes: the engine validates the combination, its allocator
    names the species and state, and ``PhysicsDriver.compute`` and
    ``microphysics.apply`` -- the same function objects the ARW step calls
    -- dispatch them.  See :func:`column_physics_plan` for what a selection
    needs before construction.
    """

    def __init__(self, *, physics, n_levels, n_columns, dt,
                 radiation_seconds, surface_pbl_seconds,
                 start_time, latitude_deg, longitude_deg,
                 terrain_height_m, z_interface_nominal_m, p_top_pa, dx_m,
                 cumulus_seconds=None, dx_column_m=None, cell_area_m2=None,
                 surface=None, statics=None, cumulus_active_column=None,
                 rk_momentum=None):
        # ---- pure validation FIRST (no device allocation before it) ----
        if not isinstance(n_levels, numbers.Integral) or n_levels < 3:
            raise ValueError(
                f"n_levels must be an integer >= 3, got {n_levels!r}")
        if not isinstance(n_columns, numbers.Integral) or n_columns < 1:
            raise ValueError(
                f"n_columns must be a positive integer, got {n_columns!r}")
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime (UTC)")
        nz, ncol = int(n_levels), int(n_columns)
        chosen = _selection_assignments(physics)
        cadences = _resolve_cadences(
            dt=dt, radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cu_physics=int(chosen.get("cu_physics", 0) or 0))
        p_top = _require_positive("p_top_pa", p_top_pa)
        dx = _require_positive("dx_m", dx_m)
        fnm, fnp = _interface_weights(z_interface_nominal_m)
        if fnm.shape[0] != nz:
            raise ValueError(
                f"z_interface_nominal_m has {fnm.shape[0] + 1} interfaces "
                f"but n_levels={nz} needs {nz + 1}")
        top_nominal = float(np.asarray(
            z_interface_nominal_m, dtype=np.float64)[-1])
        cfg = _selection_config(chosen, nz=nz, ncol=ncol, ny=1, dx=dx,
                                ztop=top_nominal, cadences=cadences)
        from woof.config import radiation_scheme_ids
        from woof.core.held_suarez_mesh import HeldSuarezMeshMomentum
        if radiation_scheme_ids(cfg)[0] == 31:
            if not isinstance(rk_momentum, HeldSuarezMeshMomentum):
                raise ValueError("ra_lw_physics=31 requires RK-stage edge momentum damping; pass a HeldSuarezMeshMomentum binding")
        elif rk_momentum is not None:
            raise ValueError("an RK momentum binding requires ra_lw_physics=31")
        self.rk_momentum = rk_momentum
        plan = _plan_from_config(cfg, chosen, cadences)
        lat = np.asarray(latitude_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(longitude_deg, dtype=np.float32).reshape(-1)
        if lat.shape != (ncol,) or lon.shape != (ncol,):
            raise ValueError(
                "latitude_deg/longitude_deg must have one value per "
                f"column ({ncol}), got {lat.shape}/{lon.shape}")
        terrain = np.asarray(terrain_height_m, dtype=np.float32).reshape(-1)
        if terrain.shape != (ncol,):
            raise ValueError(
                f"terrain_height_m must have shape ({ncol},), got "
                f"{terrain.shape}")
        dx_column = None
        if dx_column_m is not None:
            dx_column = np.asarray(dx_column_m, dtype=np.float32).reshape(-1)
            if dx_column.shape != (ncol,):
                raise ValueError(
                    f"dx_column_m must have one value per column ({ncol},), "
                    f"got {dx_column.shape}")
            if not np.all(np.isfinite(dx_column)) or np.any(dx_column <= 0.0):
                raise ValueError("dx_column_m must be finite and positive")
        from woof.core.gf import cumulus_active_column as _cu_active
        active_column = _cu_active(cumulus_active_column, ncol, ("gf" if int(cfg.cu_physics or 0) == 3 else f"cu_physics={int(cfg.cu_physics or 0)}"))
        area = None
        if cell_area_m2 is not None:
            area = np.asarray(cell_area_m2, dtype=np.float64).reshape(-1)
            if (area.shape != (ncol,) or not np.all(np.isfinite(area))
                    or np.any(area <= 0)):
                raise ValueError(
                    "cell_area_m2 must be finite and positive with one "
                    "value per column")
        # Surface inputs: the engine's own initialize_physics keywords.
        from woof.core.physics import initialize_physics
        accepted = set(inspect.signature(initialize_physics).parameters)
        seam_bound = {"state", "cfg", "radiation_start_time",
                      "radiation_latitude", "radiation_longitude",
                      "terrain_drag_static", "radiation", "cumulus"}
        given = {} if surface is None else dict(surface)
        unknown = sorted(name for name in given
                         if name not in accepted or name in seam_bound)
        if unknown:
            raise ValueError(
                f"surface names {unknown}, which initialize_physics does "
                f"not take from a column caller; it takes "
                f"{sorted(accepted - seam_bound)}")
        n_soil = plan.n_soil_levels
        for name in ("soil_temperature", "soil_moisture"):
            if not n_soil:
                continue
            value = given.get(name)
            if value is None:
                raise ValueError(
                    f"surface[{name!r}] is required: this land surface "
                    f"integrates a {n_soil}-level soil column and a "
                    "defaulted profile is substituted state")
            if tuple(np.shape(value)) != (n_soil, ncol):
                raise ValueError(
                    f"surface[{name!r}] must have shape ({n_soil}, {ncol}) "
                    f"for this land surface, got {tuple(np.shape(value))}")
        # Orographic statistics: exactly the selection's field set.
        need = plan.static_fields
        supplied = {} if statics is None else dict(statics)
        missing = [name for name in need if name not in supplied]
        extra = sorted(name for name in supplied if name not in need)
        if missing or extra:
            raise ValueError(
                f"statics must carry exactly {list(need)} for this "
                f"selection (missing {missing}, extra {extra}); a drag on "
                "zero statistics is a drag switched off without saying so")
        oro = {}
        for name in need:
            value = np.asarray(supplied[name], dtype=np.float32).reshape(-1)
            if value.shape != (ncol,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"static {name} must be finite with one value per "
                    f"column ({ncol},), got {value.shape}")
            oro[name] = value.reshape(1, ncol)
        znu = _nominal_znu(z_interface_nominal_m, p_top)
        znw = _nominal_znw(z_interface_nominal_m, p_top)
        if 99 in radiation_scheme_ids(cfg) and not (
                np.all(np.diff(znw) < 0.0) and np.all(znu > 0.0)):
            # Named breakage: GFDL-Eta builds its CO2 and ozone tables on
            # this eta pair (gfdleta.py GFDLTables); a layer clipped to
            # eta 0 has zero thickness and the longwave rate goes
            # non-finite.  It happens when p_top_pa is above the standard
            # atmosphere's pressure at the nominal top.
            raise ValueError(
                "GFDL-Eta radiation needs a strictly decreasing nominal eta: "
                f"p_top_pa={p_top} is above the standard-atmosphere "
                "pressure of the nominal top, so the upper layers clip to "
                "eta 0; pass a p_top_pa at or below that pressure")

        self._cfg = cfg
        self._plan = plan
        self._cadences = cadences
        self._dt = float(cadences["dt"])
        self._nz, self._ncol = nz, ncol
        self._start_time = start_time
        mp_physics = int(cfg.mp_physics)
        from woof.core.physics_inventory import microphysics_scratch_slots
        self._surface_slots = tuple(microphysics_scratch_slots(mp_physics))
        # The frozen seam's restart code keys two scheme carriers on its
        # scheme names; reuse those names where it has them.
        self._microphysics_scheme = next(
            (name for name, value in _MP_PHYSICS_BY_SCHEME.items()
             if value == mp_physics), f"mp_physics={mp_physics}")
        self._has_graupel = any(
            component == "graupelnc" for component, _ in self._surface_slots)
        self._species = plan.species
        # Grid-free scheme state (layout "vectors") restarts with the rest:
        # Eta Ferrier's restart stamp decides whether ETANEWinit reruns.
        self._state_restart_fields = tuple(
            dict.fromkeys(plan.state_fields + tuple(plan.layout["vectors"])
                          + ("h_diabatic",)))
        self._identity = dict(
            plan.identity,
            n_levels=nz, n_columns=ncol, dt=self._dt,
            stepra=cadences["stepra"], stepbl=cadences["stepbl"],
            stepcu=cadences["stepcu"],
            start_time=start_time.isoformat(), p_top_pa=p_top, dx_m=dx,
            dx_column=("per-column" if dx_column is not None else "scalar"),
            **({} if active_column is None else {
                "gf_active_column": (f"per-column: {int(active_column.sum())}"
                                     f" of {ncol} active")}),
            surface=sorted(given), physics=dict(plan.identity["physics"]),
            species=list(plan.species),
            static_fields=list(plan.static_fields))

        # ---- device construction from here on --------------------------
        import cupy as cp
        self._cp = cp
        self._cell_area_m2 = None if area is None else cp.asarray(area)
        state = _GenericColumnState(cfg, fnm, fnp, p_top, terrain, cp,
                                    znu=znu, znw=znw, layout=plan.layout,
                                    latitude_deg=lat)
        self._state = state
        self._species_buffers = {
            name: getattr(state, name) for name in self._species}

        def on_grid(values):
            if values is None or np.ndim(values) == 0 or isinstance(
                    values, (str, bytes)):
                return values
            if not hasattr(values, "__cuda_array_interface__"):
                values = np.asarray(values)
            if values.shape[-1] != ncol:
                return values
            return values.reshape(values.shape[:-1] + (1, ncol))

        kwargs = {name: on_grid(value) for name, value in given.items()}
        kwargs.setdefault("glw", None)
        self._driver = initialize_physics(
            state, cfg, **kwargs,
            radiation_start_time=start_time,
            radiation_latitude=lat.reshape(1, ncol),
            radiation_longitude=lon.reshape(1, ncol),
            terrain_drag_static=(dict(oro, HGT_M=terrain.reshape(1, ncol))
                                 if oro else None))
        self.seaice_optics_seed = _seed_ruc_fractional_seaice_optics(
            self._driver, cfg, lat, start_time)
        self._cumulus_momentum = {"pair": None, "seen": False}
        _capture_cumulus_momentum(self._driver, self._cumulus_momentum)

        self._aerosol_init = None
        if mp_physics in _FIRST_CONTACT:
            # initialize_physics ran the scheme's domain construction on
            # zero placeholder buffers with no geometry; that product
            # describes no column the seam will run, so every field it
            # could have written goes back to the allocator's value and
            # the real step runs at first contact (frozen seam docstring).
            for name, row in plan.layout["fields"].items():
                if row["view"] is None and name not in plan.radii:
                    getattr(state, name)[...] = row["fill"]

        shape3 = (nz, 1, ncol)
        iface3 = (nz + 1, 1, ncol)
        # The advective forcing lanes a cumulus scheme reads.  The driver
        # binds them to the state's own pair when the allocator made one
        # for the selected scheme; the seam fills whichever buffers the
        # driver reads, and allocates a pair only when there is none.
        for lane in ("gf_rthdynten", "gf_rqvdynten"):
            if getattr(self._driver, lane, None) is None:
                setattr(self._driver, lane,
                        cp.zeros(shape3, dtype=cp.float32))
        self._gf_rthdynten = self._driver.gf_rthdynten
        self._gf_rqvdynten = self._driver.gf_rqvdynten
        if dx_column is not None:
            self._driver.gf_dx_column = cp.asarray(
                dx_column.reshape(1, ncol))
        if active_column is not None:
            self._driver.gf_active_column = cp.asarray(
                active_column.reshape(1, ncol))
        # THE STATE'S ARRAYS ARE THE STATE'S, FOR GOOD.  A scheme may bind
        # state arrays when the driver is built (NSSL's production
        # coordinator does, and checks their identity on every call), so
        # nothing here ever points a state attribute at another array:
        # theta and pressure are the state's own thp/p, and both phases
        # copy the caller's fields in (phase 2 copies the results back).
        self._in = {name: cp.zeros(shape3, dtype=cp.float32)
                    for name in ("u", "v", "temperature", "exner", "dz",
                                 "rho")}
        self._in["theta"] = state.thp
        self._in["pressure"] = state.p
        self._in.update({name: cp.zeros(iface3, dtype=cp.float32)
                         for name in ("p_interface", "z_interface")})
        self._atmosphere = None
        state.prepared_physics_atmosphere = self._supplied_atmosphere
        # A-grid winds for a post-RK scheme that reads winds from the
        # state (an ARW state holds them on C-grid faces and such a scheme
        # destaggers them; these columns have no faces).
        state.mass_point_winds = lambda: (self._in["u"], self._in["v"])
        self._out = {name: cp.zeros((nz, ncol), dtype=cp.float32)
                     for name in _OUTPUT_BUFFERS}
        self._dw_out = None
        #: Seam-owned rate buffers for species outside the six masses,
        #: created the first time a phase-one scheme forces one.  Not on
        #: the restart manifest: every phase 1 rewrites them before a read.
        self._scalar_out = {}
        self.last_scalar_rates = MappingProxyType({})
        self._step_index = 0
        self._pending_phase = _PHASE1
        state.elapsed_seconds = 0.0

    # ------------------------------------------------------------------
    @property
    def plan(self) -> ColumnPhysicsPlan:
        return self._plan

    @property
    def species(self) -> tuple:
        return self._species

    @property
    def physics(self) -> dict:
        """Every selector the seam's RunConfig resolved, by field name."""
        return dict(self._plan.physics)

    @property
    def n_soil_levels(self) -> int:
        return int(self._plan.n_soil_levels)

    @property
    def last_ruc_census(self):
        census = getattr(self._driver, "last_ruc_census", None)
        return None if census is None else dict(census)

    def radii(self) -> dict:
        """The scheme's effective radii as live ``[level, column]`` views."""
        nz, ncol = self._nz, self._ncol
        return {name: getattr(self._state, name).reshape(nz, ncol)
                for name in self._plan.radii}

    def soil_state(self) -> dict:
        """Device copies of the soil column (``[level, column]``)."""
        f = self._driver.fields
        ncol = self._ncol
        return {name: f[name].reshape(f[name].shape[0], ncol).copy()
                for name in ("tslb", "smois") if name in f}

    def _species_arrays(self, phase, species) -> dict:
        """The selection's exact species set, both directions refused."""
        given = {} if species is None else dict(species)
        missing = [name for name in self._species
                   if given.get(name) is None]
        extra = sorted(name for name in given if name not in self._species)
        if missing or extra:
            raise ValueError(
                f"{phase}: this selection transports {self._species} "
                f"(missing {missing}, not its state {extra}); a "
                "zero-defaulted species is substituted state and an "
                "accepted extra one is caller state silently dropped")
        return {name: given[name] for name in self._species}

    # ------------------------------------------------------------------
    def run_phase1(self, *, dt, u, v, theta, pressure, pressure_interface,
                   z_interface, w, rho_dry, species, exner=None,
                   rthdynten=None, rqvdynten=None):
        """Run the due phase-1 chain; return the held raw tendency set.

        The frozen seam's phase 1 with the species passed as one mapping
        (exactly ``self.species``).  Read-only on every input.
        """
        if self.rk_momentum is not None and not self.rk_momentum.bound:
            raise RuntimeError("Held–Suarez RK-stage edge momentum damping is not bound")
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "run_phase1 called out of order: the previous step's "
                "run_phase2 has not run (strict phase1 -> phase2 "
                "alternation)")
        if float(dt) != self._dt:
            raise ValueError(
                f"phase-1 dt={dt} does not match the constructor model "
                f"step {self._dt}; the seam owns the sub-cadences")
        from woof.core import constants as _constants
        cp = self._cp
        nz, ncol = self._nz, self._ncol
        level, iface = (nz, ncol), (nz + 1, ncol)
        scalars = self._species_arrays("run_phase1", species)
        for name, value in (("u", u), ("v", v), ("theta", theta),
                            ("pressure", pressure), ("rho_dry", rho_dry),
                            *scalars.items()):
            self._require_input(name, value, level)
        for name, value in (("pressure_interface", pressure_interface),
                            ("z_interface", z_interface), ("w", w)):
            self._require_input(name, value, iface)
        for name, value in (("exner", exner), ("rthdynten", rthdynten),
                            ("rqvdynten", rqvdynten)):
            if value is not None:
                self._require_input(name, value, level)

        state, buf = self._state, self._in

        def v3(a):
            return a.reshape(a.shape[0], 1, ncol)

        buf["u"][...] = v3(u)
        buf["v"][...] = v3(v)
        buf["theta"][...] = v3(theta)
        buf["pressure"][...] = v3(pressure)
        buf["p_interface"][...] = v3(pressure_interface)
        buf["z_interface"][...] = v3(z_interface)
        state.w[...] = v3(w)
        # NEGATIVE-QV LAW (frozen seam): clamp in seam scratch only.
        cp.maximum(v3(scalars["qv"]), cp.float32(0.0), out=state.qv)
        for name in self._species:
            if name != "qv":
                getattr(state, name)[...] = v3(scalars[name])
        if exner is None:
            cp.power(buf["pressure"] / cp.float32(_constants.P0),
                     cp.float32(_constants.RCP), out=buf["exner"])
        else:
            buf["exner"][...] = v3(exner)
        cp.multiply(buf["theta"], buf["exner"], out=buf["temperature"])
        cp.subtract(buf["z_interface"][1:], buf["z_interface"][:-1],
                    out=buf["dz"])
        cp.multiply(cp.float32(1.0) + state.qv, v3(rho_dry), out=buf["rho"])
        # The state's own thermodynamic carriers, for a scheme that reads
        # them directly rather than through the phy_prep product (the UW
        # PBL forms its density from state.alt): alt = 1/rho_dry and, with
        # phb == 0, php = z*g -- the same two fills phase 2 makes.
        cp.divide(cp.float32(1.0), v3(rho_dry), out=state.alt)
        cp.multiply(buf["z_interface"], cp.float32(_constants.G),
                    out=state.php)
        if rthdynten is None:
            self._gf_rthdynten[...] = cp.float32(0.0)
        else:
            self._gf_rthdynten[...] = v3(rthdynten)
        if rqvdynten is None:
            self._gf_rqvdynten[...] = cp.float32(0.0)
        else:
            self._gf_rqvdynten[...] = v3(rqvdynten)

        # The supplied phy_prep product, with _prepare_atmosphere's own
        # zero semantics for a category the scheme set does not carry.
        absent = {}
        for name in ("qi", "qs"):
            if getattr(state, name, None) is None:
                absent[name] = state.scratch((nz, 1, ncol), f"physics_{name}")
                absent[name][...] = 0.0
        self._atmosphere = {
            "theta": buf["theta"], "temperature": buf["temperature"],
            "pressure": buf["pressure"],
            "p_interface": buf["p_interface"], "exner": buf["exner"],
            "u": buf["u"], "v": buf["v"],
            "z_interface": buf["z_interface"], "dz": buf["dz"],
            "qv": state.qv, "qc": state.qc,
            "qi": absent.get("qi", getattr(state, "qi", None)),
            "qs": absent.get("qs", getattr(state, "qs", None)),
            "rho": buf["rho"],
        }
        state.elapsed_seconds = float(
            np.float64(self._step_index) * np.float64(self._dt))
        driver, cfg = self._driver, self._cfg
        before = dict(driver.call_counts)
        captured = self._cumulus_momentum
        captured["pair"] = None
        try:
            tendencies = type(self)._PHASE1_ORCHESTRATION(driver, state, cfg)
        finally:
            self._atmosphere = None
        after = driver.call_counts
        ran = {name: after.get(name, 0) > before.get(name, 0)
               for name in ("radiation", "ysu", "cumulus")}
        out = self._out
        out["dtheta"][...] = tendencies.rtheta.reshape(nz, ncol)
        for name, component in _FIXED_MASS_RATES:
            value = (None if component is None
                     else getattr(tendencies, component, None))
            if value is None:
                out[name][...] = cp.float32(0.0)
            else:
                out[name][...] = value.reshape(nz, ncol)
        # Momentum: the PBL slot's raw A-grid rates (after any gwd_opt
        # drag), which every PBL scheme leaves in driver.pbl_raw_rates and
        # which persist across non-due steps; zero with the slot off.
        raw = driver.pbl_raw_rates or {}
        for name in ("du", "dv"):
            if name in raw:
                out[name][...] = raw[name].reshape(nz, ncol)
            else:
                out[name][...] = cp.float32(0.0)
        momentum = captured["pair"]
        if momentum is not None:
            cp.add(out["du"], momentum[0].reshape(nz, ncol), out=out["du"])
            cp.add(out["dv"], momentum[1].reshape(nz, ncol), out=out["dv"])
            captured["seen"] = True
        elif captured["seen"]:
            raise RuntimeError(
                "the selected cumulus scheme returns a momentum pair and "
                "did not run this step; its previous pair would be "
                "applied as this step's.  Run it on the model step "
                "(cumulus_seconds=None)")
        out["h_diabatic"][...] = state.h_diabatic.reshape(nz, ncol)

        # Rates for species outside the six masses, exactly as the ARW
        # step reads them (PhysicsTendencies.extra_scalars).  The driver
        # holds them mass-coupled by chm == 1.0 on this state, so they ARE
        # the raw rates.
        extra = getattr(driver.pbl_tendencies, "extra_scalars", None) or {}
        fixed = {"qv", "qc", "qr", "qi", "qs", "qg"}
        rates = {}
        for name, value in extra.items():
            if name in fixed:
                continue
            if name not in self._species:
                raise RuntimeError(
                    f"a phase-one scheme returned a rate for {name!r}, "
                    f"which this selection does not transport "
                    f"({self._species})")
            if not ran["ysu"]:
                raise RuntimeError(
                    f"a held rate for {name!r} arrived on a step with no "
                    "PBL call; it would be the previous step's")
            buffer = self._scalar_out.get(name)
            if buffer is None:
                buffer = self._scalar_out[name] = cp.zeros(
                    (nz, ncol), dtype=cp.float32)
            buffer[...] = value.reshape(nz, ncol)
            rates[name] = buffer
        self.last_scalar_rates = MappingProxyType(rates)
        # A w rate, for a closure that forces vertical velocity.  Coupled
        # by chf = c1f*mu + c2f == 1.0 on this state, so it is the raw
        # rate, exactly as the mass-level rates are.
        dw = getattr(tendencies, "rw", None)
        if dw is not None:
            if self._dw_out is None:
                self._dw_out = cp.zeros((nz + 1, ncol), dtype=cp.float32)
            self._dw_out[...] = dw.reshape(nz + 1, ncol)
        result = MpasColumnSuiteTendencies(
            du=out["du"], dv=out["dv"], dtheta=out["dtheta"],
            dqv=out["dqv"], dqc=out["dqc"], dqr=out["dqr"],
            dqi=out["dqi"], dqs=out["dqs"], dqg=out["dqg"],
            h_diabatic=out["h_diabatic"],
            step_index=self._step_index,
            elapsed_seconds=state.elapsed_seconds,
            radiation_ran=ran["radiation"], surface_pbl_ran=ran["ysu"],
            cumulus_ran=ran["cumulus"],
            scalar_rates=self.last_scalar_rates,
            dw=None if dw is None else self._dw_out)
        self._pending_phase = _PHASE2
        return result

    # ------------------------------------------------------------------
    def run_phase2(self, *, theta, pressure, z_interface, w, rho_dry,
                   species, u=None, v=None,
                   refl_10cm_due: bool = False) -> dict:
        """Run the selected microphysics in place on the caller's arrays.

        ``u``/``v`` are the post-RK A-grid winds for a scheme that reads
        winds (the bin scheme does); passed as a pair or not at all, and
        without them such a scheme sees the phase-1 winds.

        ``theta`` and every transported species are updated in the
        caller's arrays (copied into the state's own arrays, stepped, and
        copied back: see the constructor on why the state's arrays never
        change identity).  ``rho_dry`` and the post-RK ``w`` are always
        bound (WRF hands its microphysics driver the current ``w`` and
        ``rho``); a scheme that reads neither ignores them.  The receipt
        carries the per-call surface fields the scheme's driver arm binds
        (``plan.surface_receipts``).
        """
        if self._pending_phase != _PHASE2:
            raise RuntimeError(
                "run_phase2 called out of order: run_phase1 must run "
                "first (strict phase1 -> phase2 alternation)")
        from woof.core import constants as _constants
        cp = self._cp
        nz, ncol = self._nz, self._ncol
        level, iface = (nz, ncol), (nz + 1, ncol)
        scalars = self._species_arrays("run_phase2", species)
        for name, value in (("theta", theta), ("pressure", pressure),
                            ("rho_dry", rho_dry), *scalars.items()):
            self._require_input(name, value, level)
        for name, value in (("z_interface", z_interface), ("w", w)):
            self._require_input(name, value, iface)
        if (u is None) != (v is None):
            raise ValueError(
                "run_phase2: u and v are a pair; one without the other "
                "would step a scheme on winds from two different times")
        state, cfg, driver = self._state, self._cfg, self._driver

        def v3(a):
            return a.reshape(a.shape[0], 1, ncol)

        if u is not None:
            self._require_input("u", u, level)
            self._require_input("v", v, level)
            self._in["u"][...] = v3(u)
            self._in["v"][...] = v3(v)
        # Copy in (thb == 0, so thp IS full theta; phb == 0, so php is
        # z*g; alt = 1/rho_dry): every array the scheme reads is the
        # state's own, the same object it was when the driver was built.
        state.w[...] = v3(w)
        state.thp[...] = v3(theta)
        state.p[...] = v3(pressure)
        for name, value in scalars.items():
            getattr(state, name)[...] = v3(value)
        cp.divide(cp.float32(1.0), v3(rho_dry), out=state.alt)
        cp.multiply(v3(z_interface), cp.float32(_constants.G),
                    out=state.php)
        receipt = {}
        first_contact = _FIRST_CONTACT.get(int(cfg.mp_physics))
        if first_contact is not None and self._aerosol_init is None:
            self._aerosol_init = getattr(self, first_contact)(z_interface)
        if int(cfg.mp_physics) == 96:
            # MADWRF_MP's scheme CASE is empty (dycore.py, the solve_em
            # order): only the common mp_zero_out hook acts, and no
            # microphysics diagnostic result exists to accept.
            if refl_10cm_due:
                raise ValueError(
                    "refl_10cm_due with mp_physics=96: MADWRF_MP runs no "
                    "scheme, so nothing computes a reflectivity")
            if cfg.mp_zero_out:
                from woof.core.microphysics import microphysics_zero_out
                microphysics_zero_out(state, cfg)
        elif int(cfg.mp_physics) != 0:
            diagnostics = type(self)._PHASE2_MICROPHYSICS(
                state, cfg, self._dt, refl_10cm_due=refl_10cm_due)
            driver.accept_microphysics(diagnostics)
            for name in self._plan.surface_receipts:
                value = getattr(diagnostics, name, None)
                if value is not None:
                    receipt[name] = value.reshape(ncol).copy()
            if refl_10cm_due:
                from woof.core.refl import consume_refl_10cm
                receipt["refl_10cm"] = (
                    consume_refl_10cm(state).reshape(nz, ncol).copy())
        elif refl_10cm_due:
            raise ValueError(
                "refl_10cm_due with mp_physics=0: no scheme computes a "
                "reflectivity")
        # Copy out: the caller's theta and species now hold the scheme's
        # result.  A scheme that raised leaves the caller's arrays as they
        # were handed in.
        v3(theta)[...] = state.thp
        for name, value in scalars.items():
            v3(value)[...] = getattr(state, name)
        self._step_index += 1
        state.elapsed_seconds = float(
            np.float64(self._step_index) * np.float64(self._dt))
        self._pending_phase = _PHASE1
        return receipt

    def _thompson_first_contact(self, z_interface=None) -> dict:
        return super()._thompson_first_contact()

    def _tempo_first_contact(self, z_interface) -> dict:
        """The frozen seam's TEMPO first contact (its run_phase2 block)."""
        cp = self._cp
        state, cfg = self._state, self._cfg
        if not bool(cp.any(state.nwfa)) or not bool(cp.any(state.nifa)):
            raise ValueError(
                "TEMPO column seam requires supplied water-friendly and "
                "ice-friendly aerosol initial fields; an absent "
                "climatology must not become a synthetic real forecast")
        if cfg.tempo_surface_emission_before:
            area = self._cell_area_m2
            if area is None:
                raise ValueError(
                    "native TEMPO surface emission initialization requires "
                    "cell areas (cell_area_m2)")
            from woof.core.tempo import _runtime
            _runtime(cfg, self._nz).initialize_mpas_surface_emission(
                state.nwfa[0], z_interface[1] - z_interface[0], area,
                state.nwfa2d)
            emission = "native MPAS area and depth scaling"
        else:
            state.nwfa2d[...] = state.nwfa[0] * cp.float32(0.000196)
            emission = "lowest-level times 0.000196"
        return {"ccn": False, "in": False, "source": "supplied",
                "surface_emission": emission}

    def accumulated_precipitation(self) -> dict:
        """Bucket copies (mm): the scheme's own surface slots plus RAINC."""
        cp = self._cp
        ncol = self._ncol
        state, driver = self._state, self._driver
        out = {}
        for component, slot in self._surface_slots:
            if component.endswith("nc"):
                out[component.upper()] = state.scratch(
                    (1, ncol), slot).reshape(ncol).copy()
        rainc = driver.rainc
        out["RAINC"] = (cp.zeros(ncol, dtype=cp.float32)
                        if rainc is None else rainc.reshape(ncol).copy())
        return out


def run_mpas_column_physics(**kwargs) -> MpasColumnPhysics:
    """Construct the generic MPAS column-physics seam (keyword-only)."""
    return MpasColumnPhysics(**kwargs)

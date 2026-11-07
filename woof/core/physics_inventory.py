"""What a configuration's physics ALLOCATES and PUBLISHES -- no runtime.

Configuration predicates and name tables only, split out of
:mod:`woof.core.physics` because that module's body imports cupy and
these answers must not cost a GPU runtime to ask.  The VRAM estimator
(:func:`woof.core.preflight.physics_array_shapes`) is the consumer that
forced the split: ``woof domain`` -- a command that integrates nothing
on a card -- reached ``import cupy`` through this metadata and refused
every CPU-only install, with ``--card`` declared and the answer already
in hand (measured by the 2.5.0 persona walks).

Everything here is re-exported by :mod:`woof.core.physics`, whose
docstrings carry the WRF line-number authority; import from either, but
from HERE when the caller must stay importable without cupy.  The
runtime-free property is held by
``tests/test_wizard_sizing_authority.py``, which runs the wizard in an
interpreter where cupy does not resolve.
"""

from __future__ import annotations

from types import MappingProxyType

from woof.core.mp_column_schemes import (TABLE_SCHEMES,
                                         table_schemes_carrying)

from woof.config import (ACM2_PBL_SCHEME, BOULAC_PBL_SCHEME, P3_MP_PHYSICS, KEPS_PBL_SCHEME, MRF_PBL_SCHEME, GBM_PBL_SCHEME, EEPS_PBL_SCHEME,
                          RunConfig, SASE_PBL_SCHEME, UW_PBL_SCHEME, radiation_enabled)


RADIATION_BOUNDARY_FLUXES = tuple(
    side+suffix for side in ("sw","lw")
    for suffix in ("upt","uptc","dnt","dntc","upb","upbc","dnb","dnbc"))

# Registry.EM_COMMON:1565-1572, allocated by any_cu_used.
CUMULUS_HISTORY_RATES = (
    "rucuten", "rvcuten", "rthcuten", "rqvcuten", "rqrcuten",
    "rqccuten", "rqscuten", "rqicuten")
CUMULUS_HISTORY_SPECIES = {
    "rqvcuten": "qv", "rqrcuten": "qr", "rqccuten": "qc",
    "rqscuten": "qs", "rqicuten": "qi"}


def radiation_boundary_flux_names(cfg):
    """Boundary fluxes produced by the selected RRTMG-family radiation.

    Selector 4 computes WRF's all-sky and clear-sky top/bottom fluxes on
    both engines: the legacy RRTMG transcription as RRTMG_LWRAD/
    RRTMG_SWRAD do, RTE+RRTMGP from its all-sky solve and a clear-sky solve
    on the same gas optics.  Every other scheme computes none.
    """
    from woof.config import (RRTMG_VARIANT_LEGACY, RRTMG_VARIANT_RTE_RRTMGP,
                              radiation_scheme_ids)
    lw,sw=radiation_scheme_ids(cfg)
    sides = {side for side,selector in (("lw",lw),("sw",sw))
             if int(selector)==3 or (int(selector)==4 and cfg.ra_rrtmg_variant in (RRTMG_VARIANT_LEGACY,RRTMG_VARIANT_RTE_RRTMGP))}
    return tuple(name for name in RADIATION_BOUNDARY_FLUXES if name[:2] in sides)

def diagnostic_boundary_flux_names(cfg):
    """Held Registry flux carriers consumed by native RASM/solar.

    WRF allocates and cold-initializes these Registry fields even when
    the selected radiation does not write them. Keep those zeros as held
    state; do not substitute SWDOWN or infer an unselected producer.
    """
    names = set(radiation_boundary_flux_names(cfg))
    if cfg.mean_diag or cfg.diurnal_diag:
        names.update(('swdnb','lwupb','swupb','swupt','swdnt','lwupt','lwdnt'))
    elif cfg.solar_diagnostics:
        names.update(('swdnb','swdnt'))
    return tuple(name for name in RADIATION_BOUNDARY_FLUXES if name in names)

#: HORIZONTAL EDDY-VISCOSITY DIAGNOSTIC (cfg.hmix_k_diag): the (momentum,
#: scalar) history names each horizontal mixing producer publishes under.
#:
#: Named for the PRODUCER, not for the diagnostic, because that is the
#: whole point of the field.  ``km_opt = 4`` publishes WRF's own Registry
#: names for the 2-D Smagorinsky viscosities it computes; SASE publishes
#: scheme-qualified names for its governed horizontal diffusivity.  Both
#: are m2 s-1 on the mass grid and both are the coefficient of the same
#: down-gradient horizontal flux, so a run that removes one producer and
#: installs the other can be compared field to field on the channel it
#: swapped -- which is the measurement that decides whether "SASE
#: supplies the mixing the km_opt operator would otherwise apply" is
#: true, rather than leaving it as an assertion.
_HMIX_K_DIAG_NAMES: dict[str, tuple[str, str]] = {
    "smagorinsky": ("XKMH", "XKHH"),
    "sase": ("SASE_KMH", "SASE_KHH"),
}


def hmix_k_diag_names(cfg: RunConfig) -> tuple[str, ...]:
    """The horizontal-K history names this configuration can publish.

    EMPTY when the run has no horizontal mixing producer at all (the
    acknowledged ``km_opt = 0`` control).  Deliberately empty rather than
    a pair of zero fields: an absent variable cannot be misread as a
    measured zero, and "this file has no horizontal viscosity variable"
    is the strongest available statement that this run ran no horizontal
    mixing operator.
    """
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
        return _HMIX_K_DIAG_NAMES["sase"]
    if cfg.km_opt == 4:
        return _HMIX_K_DIAG_NAMES["smagorinsky"]
    return ()


def physics_enabled(cfg: RunConfig) -> bool:
    """Whether any non-timesplit physics scheme is configured."""
    return bool(cfg.obs_nudge_opt or cfg.tmn_update or cfg.sst_update or cfg.sst_skin or radiation_enabled(cfg) or cfg.sf_sfclay_physics
                or cfg.sf_surface_physics or getattr(cfg, "sf_ocean_physics", 0)
                or cfg.bl_pbl_physics
                or cfg.cu_physics or cfg.solar_diagnostics or getattr(cfg, "shcu_physics", 0) or getattr(cfg, "ifire", 0) == 2
                or getattr(cfg, "wind_farm_scheme", "off") != "off")


def physics_driver_required(cfg: RunConfig) -> bool:
    """Whether setup must attach a persistent :class:`PhysicsDriver`.

    Microphysics is advanced after RK3 rather than through the non-timesplit
    tendency path selected by :func:`physics_enabled`.  It still needs the
    driver for accumulated precipitation and the output-due REFL_10CM
    handoff, so an mp-only domain must receive the same persistent attachment
    as a domain with radiation, surface, PBL, or cumulus physics.
    """
    return bool(cfg.mp_physics or cfg.tmn_update or physics_enabled(cfg)
                or getattr(cfg, 'afwa_diag_opt', 0) or cfg.mean_diag
                or cfg.diurnal_diag or cfg.output_diagnostics or cfg.lightning_option)


def terrain_drag_array_shapes(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """TerrainDrag's domain-sized coefficients and statistics, held on GPU."""
    from woof.static.orographic import GWD_FIELDS

    shape = (int(cfg.ny), int(cfg.nx))
    shapes = {}
    if int(getattr(cfg, "topo_wind", 0) or 0):
        shapes.update({f"terrain_drag/{name}": shape
                       for name in ("ctopo", "ctopo2")})
    for name in GWD_FIELDS.get(int(getattr(cfg, "gwd_opt", 0) or 0), ()):
        shapes[f"terrain_drag/gwd/{name}"] = shape
    return shapes


def terrain_drag_transient_shapes(cfg: RunConfig
                                  ) -> dict[str, tuple[int, ...]]:
    """Per-call and cold-start terrain-drag workspace, all four-byte words.

    Physics supplies contiguous float32 inputs and int32 KPBL, so coercion
    aliases them.  Drag builds one mass-height field with one expression
    intermediate, and packs the four directional planes for each statistic
    and scale.  Its existing momentum targets are updated in place.  GSL
    under SASE also diagnoses two surface planes without changing its
    surface-layer carriers.  Topographic cold start holds three work planes
    (input statistic, zero placeholder, laplacian), while each YSU call
    holds two output planes.  The topo_wind=1 row-slab loader also holds
    terrain, landmask and two coefficient outputs over its one-row context
    on each side, so its envelope is seven context planes.
    """
    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    shapes = {}
    topo = int(getattr(cfg, "topo_wind", 0) or 0)
    if topo:
        shapes["terrain_drag/topo_work"] = (
            (7, ny + 2, nx) if topo == 1 else (3, ny, nx))
    option = int(getattr(cfg, "gwd_opt", 0) or 0)
    if option:
        shapes["terrain_drag/column_heights"] = (2, nz, ny, nx)
        scales = 2 if option == 3 else 1
        shapes["terrain_drag/directional_statistics"] = (8 * scales, ny, nx)
        if option == 3 and int(cfg.bl_pbl_physics) == SASE_PBL_SCHEME:
            shapes["terrain_drag/sase_boundary"] = (2, ny, nx)
    return shapes


#: The one PBL selector whose driver path (``PhysicsDriver._run_ysu``, the
#: ``1`` row of the PBL dispatch table in woof/core/physics.py) creates the
#: retained ``last_ysu`` output dict.
YSU_PBL_SCHEME = 1


def physics_retains_ysu_output(cfg: RunConfig) -> bool:
    """Whether the raw YSU output dict crosses a model-step boundary.

    Positive ``bldt`` keeps the historical diagnostic object untouched.
    Only YSU creates that object: MYJ, MYNN, Shin-Hong and SASE leave
    ``last_ysu`` None (SASE's raw rates and diagnostics have their own
    canonical buffers), so pricing it for them over-counts about ten
    arrays and can refuse a run that fits at the margin.
    At ``bldt == 0`` every configured PBL call is immediately consumed by
    :meth:`PhysicsDriver._run_ysu`, so retaining the raw rates duplicates the
    coupled PBL tendencies without serving a later reader.
    """
    return bool(cfg.bl_pbl_physics == YSU_PBL_SCHEME and cfg.bldt > 0.0)


def physics_reuses_pbl_composition(cfg: RunConfig) -> bool:
    """Whether the composed tendency target can be the fresh PBL stack.

    This is deliberately narrower than ``stepbl == 1``: every positive-bldt
    configuration retains the historical allocation path.  With active YSU
    and literal ``bldt == 0``, every PBL call overwrites the coupled rates before
    every composition, so radiation/cumulus can be accumulated into it once
    without corrupting a value needed by the next step.
    """
    return bool(cfg.bl_pbl_physics and cfg.bldt == 0.0
                and (radiation_enabled(cfg) or cfg.cu_physics or getattr(cfg,"shcu_physics",0)))


#: ``mp_physics`` values whose WRF Registry ``moist`` package carries QI,
#: and therefore the values for which the PBL driver returns an ``rqi``
#: tendency the run must hold.  THREE sites read it and all three must
#: agree, or the ``woof check --alloc`` measurement stops covering true
#: runtime residency: physics.py's ``_pbl_optional_tendency_components``
#: (what the run allocates), ``preflight.physics_array_shapes`` (what the
#: estimate prices) and ``preflight._materialize_physics`` (what the
#: measurement constructs).  They used to be three literal tuples, and
#: mp=16 reached production having moved only two of them.
from woof.microphysics_schemes import NAMED_SCHEMES as _NAMED_MP_SCHEMES
from woof.microphysics_schemes import SCHEMES_BY_ID as _NAMED_MP_SCHEMES_BY_ID
PBL_RQI_MICROPHYSICS = (2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 50, 51, 52, 53, 55, 56, 96, 106) + TABLE_SCHEMES + tuple(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values() if "qi" in _s.ice_mass_species)


#: Microphysics producers with an output-due radar reflectivity carrier.
REFL_10CM_MICROPHYSICS = tuple(sorted(
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 38, 40, 50, 51, 52, 53, 55, 56, 106) + TABLE_SCHEMES)) + tuple(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values() if _s.native_reflectivity)
# Kessler has no stock microphysics reflectivity writer. The default
# diagnostic remains available; strict forecast history retains the
# initialized stock slot (module_mp_kessler.F, no REFL_10CM argument).
WRF_INACTIVE_REFL_MICROPHYSICS = frozenset({1,11,96})


#: The raw PBL fields needed to repeat mass coupling after a grid move.
PBL_RAW_RATE_NAMES = ("du", "dv", "dtheta", "dqv", "dqc", "dqi")
PBL_SHARED_FORCING = {"dtheta": "gf_rthblten", "dqv": "gf_rqvblten"}

# surface_driver integrates the final land-surface flux once per due call,
# with atmospheric DT rather than the longer land-surface integration DTBL.
SURFACE_ENERGY_ACCUMULATORS = {"achfx": "hfx", "aclhf": "lh", "acgrdflx": "grdflx"}
PBL_HISTORY_RATES = {"rublten": "du", "rvblten": "dv", "rthblten": "dtheta",
                     "rqvblten": "dqv", "rqcblten": "dqc"}


def history_tendency_rows(cfg=None) -> bool:
    """Whether this process carries WRF's 3-D PBL and cumulus tendency rows.

    RUBLTEN..RQCBLTEN (RQIBLTEN under MYNN) and the RxxCUTEN rates are WRF
    Registry restart state, not default history; the exact chain carries them
    so a strict (GPUWM_WRF_EXACT=1) wrfout can be compared word for word with a
    recording that wrote them.  A default process allocates none of them.

    THE BREAKAGE THIS PREVENTS (2.9.1 integration): carried in every process,
    they added up to a dozen 3-D float32 planes per domain to the resident
    envelope (+1.0 GB, 6.6 %, on the streamed-admission check case), and the
    16 GB headline domain that fits resident in 2.9.0 no longer did.
    """
    from woof.wrf_exact import ENABLED
    return bool(ENABLED or getattr(cfg,'acc_phy_tend',0))


def pbl_history_rates(cfg) -> dict[str, str]:
    """The selected PBL producer's raw history rates (strict processes only,
    see :func:`history_tendency_rows`; a default process gets ``{}``).

    MYNN and ACM2 assign RQIBLTEN independently of whether a prognostic
    ice scalar consumes it (module_pbl_driver.F:1682 for MYNN and
    module_bl_acm.F:555 for ACM2). Keep its actual producer record on the
    same mass-coupling clock as the other rates.
    """
    if not history_tendency_rows(cfg):
        return {}
    rates = dict(PBL_HISTORY_RATES)
    if int(getattr(cfg, "bl_pbl_physics", 0)) in (5, ACM2_PBL_SCHEME):
        rates["rqiblten"] = "dqi"
    return rates
SCM_SURFACE_FORCING_FIELDS = (
    "hfx_force", "lh_force", "tsk_force", "hfx_force_tend",
    "lh_force_tend", "tsk_force_tend")


def scm_forcing_fields(cfg=None) -> tuple[str, ...]:
    """The SCM surface forcing scalars this process carries.

    HFX_FORCE and its siblings are WRF single-column-model controls. The
    ideal SCM surface advances these actual scalar carriers; the exact
    chain otherwise carries them, held at their initial
    or restored words, so a strict (GPUWM_WRF_EXACT=1) wrfout lists every
    record a recording wrote.  A default process carries none of them.

    THE BREAKAGE THIS PREVENTS (2.9.1 GPU shard leg at e3e6b7dae): carried
    in every process, the six 0-d scalars reached consumers that admit only
    gridded arrays, so default ensembles refused at launch ("driver/fields/
    hfx_force has no admitted member-column shape"), prepared-store scatter
    and nest relocation raised on shape (), and wrfout gained six records
    2.9.0 never wrote.
    """
    return (SCM_SURFACE_FORCING_FIELDS
            if int(getattr(cfg, "sf_sfclay_physics", 0)) == 89
            or history_tendency_rows() else ())


def pbl_raw_rate_names(cfg: RunConfig) -> tuple[str, ...]:
    """Rates held only when a PBL call can be skipped after relocation."""
    if not cfg.bl_pbl_physics or cfg.bldt == 0.0:
        return ()
    names = (PBL_RAW_RATE_NAMES if cfg.mp_physics in PBL_RQI_MICROPHYSICS
             else PBL_RAW_RATE_NAMES[:-1])
    # SASE returns a physical half-level vertical acceleration too.
    # Relocation couples it onto z faces using the transplanted mass.
    return names + (("dw",) if cfg.bl_pbl_physics == SASE_PBL_SCHEME else ())


def microphysics_scratch_slots(
        mp_physics: int) -> tuple[tuple[str, str], ...]:
    """Driver diagnostic component -> canonical persistent scratch slot.

    ``mp_physics=28`` shares mp=8's row, and the authority for that is WRF's
    own driver arm rather than mp=8's spelling: ``CASE (THOMPSONAERO)``
    (``phys/module_microphysics_driver.F:1029``) calls ``mp_gt_driver`` with
    RAINNC (:1085), RAINNCV (:1086), SNOWNC (:1087), SNOWNCV (:1088),
    GRAUPELNC (:1089), GRAUPELNCV (:1090) and SR (:1091) and with NO hail
    argument -- the identical seven ``CASE (THOMPSON)`` binds at :1253-:1259.
    woof's aerosol adapter writes the same seven canonical scratch slots
    (``woof/core/microphysics_aerosol.py:263-269``), so the driver aliases
    them here instead of allocating a private zero-filled copy set.
    """
    from woof.physics_registry import consumer_row_for_selector
    native = consumer_row_for_selector("microphysics", "native_column", mp_physics)
    if native:
        return tuple(native["surfaces"].items())
    if mp_physics == 1:
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_kessler_sr"))
    if mp_physics in (5, 15):
        # Ferrier-Aligo.  THREE slots: the driver arm ``CASE
        # (FER_MP_HIRES)`` binds RAINNC, RAINNCV and SR and nothing else
        # (module_microphysics_driver.F:2613-2626).  FER_HIRES puts all
        # surface precipitation into RAINNC and says how much of it fell
        # frozen through SR (module_mp_fer_hires.F:967-974).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"))
    if mp_physics in (3, 11):
        # CAMMGMP and WSM3 have rain/snow/SR, with no graupel.
        # WSM3: the driver arm binds RAINNC/RAINNCV, SNOWNC/SNOWNCV and SR
        # and no graupel argument (module_microphysics_driver.F:2241-2245).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"))
    if mp_physics in (2, 6, 8, 10, 16, 28, 29, 30, 38, 40) + TABLE_SCHEMES or mp_physics in _NAMED_MP_SCHEMES_BY_ID:
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"),
                ("graupelnc", "mp_graupelnc"),
                ("graupelncv", "mp_graupelncv"))
    if mp_physics == 106:
        # WSM6R.  THREE slots: the driver arm ``CASE (WSM6RSCHEME)`` binds
        # RAINNC and RAINNCV only (module_microphysics_driver.F:2362-2372);
        # SR is the surface contract's zero (woof/core/wsm6.py), and
        # microphysics_scheme_sr_available says 106 has none.
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"))
    if mp_physics in (7, 9, 18, 24, 26, 27, 56):
        # WSM7 (mp=24) binds the same nine in CASE (WSM7SCHEME)
        # (module_microphysics_driver.F:2401-2408: RAIN/RAINNCV, SNOW/
        # SNOWNCV, SR, GRAUPEL/GRAUPELNCV, HAIL/HAILNCV).
        # Milbrandt-Yau shares NSSL's nine-slot row because its WRF driver
        # arm binds the same nine: RAINNC/RAINNCV (:1868-1869),
        # SNOWNC/SNOWNCV (:1870-1871), HAILNC/HAILNCV (:1872-1873),
        # GRAUPELNC/GRAUPELNCV (:1874-1875) and SR (:1876) in
        # module_microphysics_driver.F's CASE (MILBRANDT2MOM).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"),
                ("graupelnc", "mp_graupelnc"),
                ("graupelncv", "mp_graupelncv"),
                ("hailnc", "mp_hailnc"),
                ("hailncv", "mp_hailncv"),
                ("sr", "mp_sr"))
    from woof.core.column_schemes import COLUMN_SCHEMES
    if mp_physics in COLUMN_SCHEMES:
        return tuple((name.removeprefix("mp_"), name) for name in COLUMN_SCHEMES[mp_physics].surface)
    if mp_physics in (14,) + P3_MP_PHYSICS:
        # WDM5 and P3 carry the five rain/snow slots, without graupel.
        # P3 driver arms: module_microphysics_driver.F:1590-1595, :1638-1642.
        # because P3 has a single ice category whose rime fraction spans
        # what other schemes split into snow, graupel and hail.  Listing
        # graupel here would allocate a canonical accumulator that stayed
        # zero forever and let output claim a graupel field P3 never has.
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"))
    # A table-admitted column scheme (woof.core.column_microphysics)
    # writes exactly the slots its registry row names, so the canonical
    # mapping is read off that row rather than restated per scheme.
    from woof.physics_registry import consumer_row_for_selector
    row = consumer_row_for_selector("microphysics", "column_kernel",
                                    int(mp_physics))
    if isinstance(row, dict):
        return tuple((slot[len("mp_"):], slot)
                     for slot in row["surface_slots"])
    if mp_physics == 95:
        # Eta Ferrier.  THREE slots: the driver arm ``CASE (ETAMPNEW)``
        # binds RAINNC, RAINNCV and SR and nothing else
        # (module_microphysics_driver.F:2577-2594); ETAMP_NEW folds snow
        # into one surface precipitation total, RAINNC, and reports its
        # frozen share only through SR (module_mp_etanew.F:547-554).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"))
    return ()


# ---------------------------------------------------------------------------
# Scheme output/state NAME tables, hoisted from their runtime modules
# (sfclay, mynn_sfclay, mynn_pbl_runtime -- each imports cupy at module
# scope) so the estimator can price their allocations without a GPU
# runtime.  Each runtime module re-imports its table, so there is still
# exactly one spelling of every inventory.
# ---------------------------------------------------------------------------

SFCLAY_OUTPUTS = (
    # USTM is Registry.EM_COMMON:1954 state ("U* IN SIMILARITY THEORY
    # WITHOUT VCONV"), unconditional and restart-carried, and SFCLAY1D
    # writes it beside UST on every column (module_sf_sfclay.F:799-804).
    "znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol", "regime",
    "psim", "psih", "fm", "fh", "lh", "u10", "v10", "th2", "t2",
    "q2", "chs", "chs2", "cqs2", "flhc", "flqc", "qgh", "rmol",
    "wspd", "br", "gz1oz0", "cpm", "ck", "cka", "cd", "cda",
)

MYNN_SURFACE_OUTPUTS = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
    "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
    "qstar", "cpm",
    # module_sf_mynn.F:436 declares ZNT INTENT(INOUT); whichever leaf the
    # isftcflx arm selects (:635/:641/:643/:647) rewrites it on every water
    # column and the new value must persist into the next step.
    "znt",
)

#: WRF Registry ``mynnscheme`` prognostic/carried 3-D state, in the spelling
#: ``module_pbl_driver.F`` binds.  ``el_pbl``/``sh3d``/``sm3d`` keep WRF's
#: names rather than the solver's ``el``/``sh``/``sm`` so the restart
#: manifest and wrfout read the same identifiers WRF writes.
MYNN_PBL_STATE_3D = (
    "qke", "tsq", "qsq", "cov", "el_pbl", "sh3d", "sm3d",
    "qc_bl", "qi_bl", "cldfra_bl",
)

#: Per-column plume diagnostics the wrapper exports (``:1698-1699``).
MYNN_PBL_DIAGNOSTICS_2D = ("maxwidth", "maxmf", "ztop_plume")

#: Integer per-column diagnostic; kept apart because it is int32.
MYNN_PBL_DIAGNOSTICS_INT_2D = ("ktop_plume",)


#: The Pleim-Xiu surface layer (``sf_sfclay_physics = 7``), in the argument
#: order of ``pxsfclay.cu`` (woof/core/pxsfclay.py launches it).
#:
#: * inputs: the lowest-level U, V, T, THETA, QV, P and layer depth DZ8W,
#:   and PSFC, TSK, PBLH, MAVAIL and XLAND (module_surface_driver.F:2163);
#: * state: WRF's INOUT words (module_sf_pxsfclay.F:138-196) and the OUT
#:   pair U10/V10.  Every name is already in :data:`SFCLAY_OUTPUTS`, so the
#:   scheme allocates nothing the MM5 layers do not; it writes no FM, FH,
#:   USTM, TH2, T2 or Q2 (those planes keep their cold-start zeros, which
#:   is why the fm/fh-binding PBLs are refused with it);
#: * sea: the open-water words PXSFCLAY_SEAICE_WRAPPER hands the post-LSM
#:   blend (module_surface_driver.F:6926-6970), stored in the ``<name>_sea``
#:   carriers the RUC seam already holds.
PX_SFCLAY_INPUTS = ("u1", "v1", "t1", "th1", "qv1", "p1", "dz8w", "psfc",
                    "tsk", "pblh", "mavail", "xland")
PX_SFCLAY_STATE = ("regime", "hfx", "qfx", "lh", "mol", "rmol", "qsfc",
                   "gz1oz0", "wspd", "br", "psim", "psih", "znt", "zol",
                   "ust", "cpm", "chs2", "cqs2", "chs", "flhc", "flqc",
                   "qgh", "u10", "v10")
PX_SFCLAY_SEA = ("chs", "chs2", "cpm", "cqs2", "flhc", "flqc", "hfx", "lh",
                 "qfx", "qgh", "qsfc", "tsk")

#: The Eta similarity surface layer's (``sf_sfclay_physics = 2``) WRF INOUT

#: The Eta similarity surface layer's (``sf_sfclay_physics = 2``) WRF INOUT
#: state, in the argument order of the ``myjsfc_column`` kernel that
#: woof/core/myjsfc.py launches (hoisted from there, which imports cupy).
MYJ_SFCLAY_INOUT = ("ust", "znt", "thz0", "qz0", "uz0", "vz0", "qsfc",
                    "akhs", "akms")

#: The Eta layer's pure outputs, in the same kernel's argument order.
#: ``rib`` is the field Noah reads as its bulk Richardson number, the same
#: slot the MM5 surface layers fill through ``br``
#: (woof/core/physics.py::_run_noah).
MYJ_SFCLAY_OUTPUTS = ("rmol", "ct", "pblh", "rib", "chs", "chs2", "cqs2",
                      "hfx", "qfx", "lh", "flhc", "flqc", "qgh", "cpm",
                      "u10", "v10", "t2", "th2", "tshltr", "th10", "q2",
                      "qshltr", "q10", "pshltr", "u10e", "v10e")

#: Every persistent 2-D surface field ``initialize_physics`` holds for the
#: Eta layer, in allocation order: the kernel roster above plus the four
#: Registry fields its driver path reads or publishes (``ustm``, ``wspd``,
#: ``ch``, ``mixht``).  MYJSFC's Z0BASE is WRF's Registry ``Z0``, the
#: land-surface field every run already carries, not a plane of its own.
#: The allocation and the VRAM estimate
#: (:func:`woof.core.preflight.physics_field_names_2d`) both read THIS
#: tuple; while the estimate restated nothing for the selector, 17 of these
#: planes were allocated and never priced.
MYJ_SFCLAY_FIELDS_2D = tuple(dict.fromkeys((
    *MYJ_SFCLAY_INOUT, *MYJ_SFCLAY_OUTPUTS,
    "ustm", "wspd", "ch", "mixht")))

#: MYJ's carried 3-D PBL state (``bl_pbl_physics = 2``): WRF's TKE_MYJ and
#: EL_MYJ, allocated once by ``initialize_physics`` for this selector only.
#: Read by the allocation and by
#: :func:`woof.core.preflight.physics_array_shapes`, so the two stay one
#: list; the estimate used to omit both, 2*nz planes per domain.
MYJ_PBL_STATE_3D = ("tke_myj", "el_myj")

#: EEPS's (``bl_pbl_physics = 16``) PBL-owned copy of its TKE and
#: dissipation rate, WRF's PEK_PBL/PEP_PBL (Registry.EM_COMMON:1051-1052,
#: ``hr``: history and restart).  Allocated by ``initialize_physics`` for
#: this selector only, at EEPSINIT's minpek/minpep, and priced by the VRAM
#: estimate from this same tuple.
EEPS_PBL_STATE_3D = ("pek_pbl", "pep_pbl")

# Re-export the CPU-only transport metadata for existing physics consumers.
from woof.core.pbl_scalar_inventory import (
    PBL_TRANSPORTED_SCALARS, pbl_transported_scalars)

#: The QNSE PBL (``bl_pbl_physics = 4``) carries the SAME two columns under
#: the same names: module_bl_qnsepbl.F's dummies are MYJ's (TKE read as
#: 2*TKE at :333 and rewritten at :429, EL_MYJ INTENT(OUT) at :188), bound
#: by the driver to the same tke_pbl/el_pbl state (module_pbl_driver.F:
#: 1549-1550).  Only the cold start differs: QNSEPBLINIT seeds TKE with
#: EPSQ2L = 0.01 (module_bl_qnsepbl.F:1357), not MYJ's EPSQ2 = 0.2.
QNSE_TKE_COLD_START = 0.01

# QNSESFCINIT, WRF v4.6.1 module_sf_qnsesfc.F:1113-1119.
# Cold-start overrides only. Restart restoration subsequently restores the
# carried words. The EM build leaves Z0 unchanged; NMM land-use reduction
# is outside this EM-core engine's selectable dynamics.
ETA_SURFACE_INITIAL_STATE = {4: {"ust": 0.1}}

#: The EDMF shallow mass flux beside QNSE (``mfshconv = 1``), allocated by
#: ``initialize_physics`` for that pairing only and priced by the VRAM
#: estimate from these same tuples, all (nz) mass levels:
#:
#: * carried: RC_MF, the updraft's diagnosed cloud water, read back as the
#:   condensate split of QV on the next call (module_bl_mfshconvpbl.F:
#:   188-189) and Registry ``r`` (restart);
#: * published: the EDKF set (:271-281), every level rewritten each call.
MFSHCONV_STATE_3D = ("rc_mf",)
MFSHCONV_OUTPUTS_3D = ("massflux_edkf", "entr_edkf", "detr_edkf", "thl_up",
                       "thv_up", "rt_up", "rv_up", "rc_up", "u_up", "v_up",
                       "frac_up")

#: The UW moist-turbulence PBL's (``bl_pbl_physics = 9``) persistent fields,
#: allocated by ``initialize_physics`` for that selector only and priced by
#: the VRAM estimate from these same tuples.  Interface fields carry nz+1
#: levels, as WRF's kms:kme arrays do for this scheme.
#:
#: * carried, (nz+1): WRF's EXCH_M / EXCH_H through kte+1, read back as
#:   kvm_in/kvh_in on every step after the first
#:   (module_bl_camuwpbl_driver.F:436-441, :551-556);
#: * carried, (ny, nx): the residual surface stress TAURESX2D/TAURESY2D
#:   (driver.F:465-470, :621-622);
#: * published, (nz+1): TKE_PBL, TURBTYPE3D, SMAW3D (driver.F:742-748);
#: * published, (ny, nx): TPERT2D, QPERT2D, WPERT2D (driver.F:754-756);
#: * held, (nz): the radiation step's CLDFRA, which WRF's radiation driver
#:   writes on due steps and the scheme reads on every PBL step
#:   (module_radiation_driver.F:1309-1332, module_pbl_driver.F:1940).
UWPBL_STATE_FULL = ("uw_kvm", "uw_kvh")
UWPBL_STATE_2D = ("tauresx2d", "tauresy2d")
UWPBL_DIAGNOSTICS_FULL = ("tke_pbl", "turbtype3d", "smaw3d")
UWPBL_DIAGNOSTICS_2D = ("tpert2d", "qpert2d", "wpert2d")
UWPBL_HELD_3D = ("uw_cldfra",)

#: The TEMF surface layer (``sf_sfclay_physics = 10``,
#: phys/module_sf_temfsfclay.F), in the argument order of the
#: ``temf_sfclay_column`` kernel woof/core/temf_sfclay.py launches.
#: Inputs are lowest-model-level and surface fields; ``hd_temf`` is the TEMF
#: PBL's dry thermal top (read for w*).
TEMF_SFCLAY_INPUTS = ("u1", "v1", "th1", "qv1", "p1", "pi1", "rho1", "z1",
                      "ht", "psfc", "znt", "mavail", "tsk", "fcor",
                      "hd_temf")
#: Read and rewritten: the first level of the PBL's TE_TEMF (floored at
#: TEmin in place, :256), HFX (read for w*, :258) and WM_TEMF (averaged
#: with its previous value, :338-340).
TEMF_SFCLAY_CARRIED = ("te1", "hfx", "wm_temf")
#: Written.  ``exch_temf`` is WRF's per-step EXCH_TEMF, which
#: module_first_rk_step_part1.F:127 holds as a local and hands the TEMF PBL.
TEMF_SFCLAY_COLUMN_OUTPUTS = ("ust", "qsfc", "chs", "chs2", "cqs2", "cpm", "qgh",
                       "qfx", "lh", "flhc", "flqc", "u10", "v10", "th2",
                       "t2", "q2", "exch_temf")
#: The persistent 2-D planes ``initialize_physics`` adds for the selector
#: beyond the shared surface-layer set: WM_TEMF (Registry state, restart
#: field, TEMFSFCLAYINIT cold start 0) and EXCH_TEMF.  Read by the
#: allocation and the VRAM estimate alike.
TEMF_SFCLAY_FIELDS_2D = ("wm_temf", "exch_temf")
#: MRF's (``bl_pbl_physics = 99``) one field of its own: WRF's HOL, the PBL
#: height over the Monin-Obukhov length the scheme writes on every call
#: (module_bl_mrf.F:855) and nothing reads back.  Allocated by
#: ``initialize_physics`` for that selector only and priced from this tuple.
MRF_PBL_DIAGNOSTICS_2D = ("hol",)
#: BouLac's (``bl_pbl_physics = 8``) persistent fields, allocated by
#: ``initialize_physics`` for that selector only and priced by the VRAM
#: estimate from this same tuple: WRF's TKE_PBL (prognostic, INOUT, carried
#: across steps and restarts; module_bl_boulac.F:108, :250, :360) and EL_PBL
#: (BouLac's mixing length DLK, rewritten every call, :361).  EXCH_H and
#: EXCH_M are the shared slot fields every PBL scheme writes.
#: Native restart diagnostics, retained even when a checkpoint falls
#: between PBL calls. The kernel's short output names map to Registry names.
BOULAC_FLUX_FIELDS = {name: name + "_tur" for name in ("wu", "wv", "wt", "wq")}
BOULAC_STATE_3D = ("tke_pbl", "el_pbl", *BOULAC_FLUX_FIELDS.values())
#: WRF's cold start, BOULACINIT (module_bl_boulac.F:940-954): TKE_PBL =
#: 0.0001 (also the scheme's floor ``temin``), EL_PBL zero.
BOULAC_COLD_START = {name: (1.0e-4 if name == "tke_pbl" else 0.0)
                    for name in BOULAC_STATE_3D}
#: The BEP handoff BouLac's column reads under sf_urban_physics 2/3, in
#: kernel argument order (``sf_bep`` carries nz + 1 faces).  ``dlg_bep`` is
#: INOUT: WRF writes the urban-blended ground distance back (:357-359).
BOULAC_BEP_INPUTS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
                     "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
                     "dlg_bep", "dl_u_bep", "vl_bep", "sf_bep")
#: The launcher's per-call mass-level outputs in kernel argument order (the
#: five tendencies, EL_PBL as DLK, EXCH_H, EXCH_M), then the four turbulent
#: fluxes WRF publishes as WU_TUR..WQ_TUR, held here so the launcher
#: allocates from them and the preflight prices from them.
BOULAC_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                         "rqcblten", "el_pbl", "exch_h", "exch_m")
BOULAC_FLUX_OUTPUTS = ("wu", "wv", "wt", "wq")
#: Threads per block of the BouLac column launch.
BOULAC_BLOCK = 128

#: The UW launcher's per-call output roster (woof/core/uwpbl.py
#: ``uwpbl_step``), held HERE so the launcher allocates from it and the
#: preflight prices from it: one copy, nothing to drift.  Mass-level (nz)
#: float32 tendencies; the interface (nz+1) outputs are
#: :data:`UWPBL_DIAGNOSTICS_FULL`; the surface (ny, nx) float32 outputs,
#: plus the int32 ``kpbl2d`` (WRF's KPBL, one-based).
UWPBL_MASS_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                      "rqcblten", "rqiblten", "rqniblten")
UWPBL_SURFACE_OUTPUTS = ("tpert2d", "qpert2d", "wpert2d", "pblh2d")

#: Threads per block of the UW column launch.
UWPBL_BLOCK = 64

#: CAM's automatic arrays, per column, in binary64 and int32 slots per
#: interface level (nk + 1).  Every allocation in the device transcription
#: is taken at the top of its routine, unconditionally (uwpbl_eddy.cuh,
#: uwpbl_caleddy.cuh, uwpbl_zisocl.cuh, uwpbl_vdiff.cuh and the column
#: driver), so the high-water mark is a function of the level count alone.
#: MEASURED 2026-09-30 with a counting build of the same headers over every
#: column of the oracle fixtures: 109 * (nk + 1) - 74 binary64 slots and
#: 5 * (nk + 1) - 2 int32 slots at nk = 35, 44 and 61, the branch-probe
#: columns included.  The coefficients are those slopes with the negative
#: intercepts dropped, so the pool is never short; the device still
#: reports an overflow and the launcher refuses it by name.
UWPBL_R8_SLOTS_PER_LEVEL = 109
UWPBL_I4_SLOTS_PER_LEVEL = 5

#: Device bytes one launch's two pools may hold before the domain is walked
#: in column chunks.  At 50 levels one column costs 45.5 KB, so a chunk is
#: about 17,000 columns.
UWPBL_WORKSPACE_BUDGET_BYTES = 768 * 1024 * 1024


def uwpbl_workspace_slots(nk: int) -> tuple[int, int]:
    """``(r8_slots, i4_slots)`` one column of ``nk`` mass levels needs."""
    levels = int(nk) + 1
    return (UWPBL_R8_SLOTS_PER_LEVEL * levels,
            UWPBL_I4_SLOTS_PER_LEVEL * levels)


def uwpbl_chunk_columns(nk: int, ncols: int,
                        budget_bytes: int = UWPBL_WORKSPACE_BUDGET_BYTES
                        ) -> int:
    """Columns per UW launch so the two pools fit ``budget_bytes``."""
    r8, i4 = uwpbl_workspace_slots(nk)
    per_column = 8 * r8 + 4 * i4
    chunk = max(1, int(budget_bytes) // per_column)
    if chunk >= UWPBL_BLOCK:
        chunk = chunk // UWPBL_BLOCK * UWPBL_BLOCK
    return int(min(chunk, max(int(ncols), 1)))


def uwpbl_workspace_bytes(nk: int, ncols: int,
                          budget_bytes: int = UWPBL_WORKSPACE_BUDGET_BYTES
                          ) -> int:
    """Device bytes the UW launcher's pools hold for one call.

    The pools are sized to one chunk and reused across the domain's
    chunks, plus the one int32 overflow word.
    """
    r8, i4 = uwpbl_workspace_slots(nk)
    chunk = uwpbl_chunk_columns(nk, ncols, budget_bytes)
    return int(chunk * (8 * r8 + 4 * i4) + 4)


#: The TEMF pair (bl_pbl_physics = 10 with sf_sfclay_physics = 10, WRF
#: v4.6.1 module_bl_temf.F and module_sf_temfsfclay.F; woof/core/temf.py).
#: Its carried state is the Registry's (Registry.EM_COMMON:1097-1115,
#: packages temfpblscheme :3172 and temfsfcscheme :3135), allocated by
#: ``initialize_physics`` for the pair only and priced by the VRAM estimate
#: from these same tuples.  Every 3-D field is on TEMF's own grid (level 1
#: the surface, level k the WRF mass level k-1) with nz levels, which is
#: every level the scheme writes.  WRF's INTENT(OUT) arrays keep last step's
#: words wherever a branch does not write them (the updraft fields in a
#: stable column, MF_TEMF's top level), so all of them are carried, not
#: rebuilt: they are restart state.
TEMF_PBL_STATE_3D = ("te_temf", "kh_temf", "km_temf", "shf_temf",
                     "qf_temf", "uw_temf", "vw_temf", "wupd_temf",
                     "mf_temf", "thup_temf", "qtup_temf", "qlup_temf",
                     "cf3d_temf")
TEMF_PBL_STATE_2D = ("hd_temf", "lcl_temf", "hct_temf", "cfm_temf")
#: The surface layer's own carried velocity scale (WM_TEMF), averaged with
#: its previous value every call (module_sf_temfsfclay.F:338).
TEMF_SFCLAY_STATE_2D = ("wm_temf",)
#: The surface kernel's WRF INOUT set, kernel argument order; ``te1`` is
#: the surface level of TE_TEMF, which the layer floors at TEMIN (:256).
TEMF_SFCLAY_INOUT = ("hfx", "qfx", "lh", "ust", "te1", "wm_temf", "qsfc")
#: The surface kernel's pure outputs, kernel argument order.  EXCH_TEMF is
#: a local of WRF's first_rk_step_part1 (never state), published here as a
#: per-call field the PBL does not read.
TEMF_SFCLAY_OUTPUTS = ("flhc", "flqc", "exch_temf", "chs", "chs2", "cqs2",
                       "u10", "v10", "th2", "t2", "q2", "qgh", "cpm")

#: Threads per block of the TEMF PBL column launch (kernels/temf.cu
#: TEMF_LANES), and the column workspace's slots per level
#: (TEMF_WS_SLOTS): one global (nz+2)-word column per slot per lane.
TEMF_BLOCK = 64
TEMF_WS_SLOTS = 49
#: Device bytes one PBL launch's workspace may hold before the domain is
#: walked in column chunks.  At 50 levels a column costs 10.2 KB, so a
#: chunk is about 26,000 columns.
TEMF_WORKSPACE_BUDGET_BYTES = 256 * 1024 * 1024


def temf_workspace_floats(nz: int, columns: int) -> int:
    """Words for whole blocks of ``columns`` columns of ``nz`` levels."""
    blocks = (int(columns) + TEMF_BLOCK - 1) // TEMF_BLOCK
    return blocks * TEMF_WS_SLOTS * (int(nz) + 2) * TEMF_BLOCK


def temf_chunk_columns(nz: int, ncols: int,
                       budget_bytes: int = TEMF_WORKSPACE_BUDGET_BYTES) -> int:
    """Columns per TEMF PBL launch so its workspace fits ``budget_bytes``."""
    per_column = 4 * TEMF_WS_SLOTS * (int(nz) + 2)
    chunk = max(TEMF_BLOCK, int(budget_bytes) // per_column)
    chunk = chunk // TEMF_BLOCK * TEMF_BLOCK
    return int(min(chunk, max(int(ncols), 1)))


def temf_workspace_bytes(nz: int, ncols: int,
                         budget_bytes: int = TEMF_WORKSPACE_BUDGET_BYTES
                         ) -> int:
    """Device bytes the TEMF PBL launcher's workspace holds for one call."""
    chunk = temf_chunk_columns(nz, ncols, budget_bytes)
    return 4 * temf_workspace_floats(nz, chunk)

#: The UW shallow cumulus (``shcu_physics = 2``, woof/core/uwshcu.py)
#: launcher's field rosters, held HERE so the launcher, the physics slot
#: and the VRAM estimate read one list.  Inputs in kernel argument order:
#: mass (nz), interface (nz+1) and surface fields; ``dlf_in``/``cmfmc_in``/
#: ``rliq_in``/``cutop_in``/``cubot_in`` are the deep-convection parts the
#: driver adds the shallow ones to (module_shcu_camuwshcu_driver.F:405,
#: :469, :481-494).
UWSHCU_MASS_INPUTS = ("u", "v", "t", "qv", "qc", "qi", "qnc", "qni", "p",
                      "z", "pi", "dlf_in")
UWSHCU_FULL_INPUTS = ("p8w", "z_at_w", "tke_pbl", "cmfmc_in")
UWSHCU_SURFACE_INPUTS = ("pblh", "ht", "cutop_in", "cubot_in", "rliq_in")
UWSHCU_MASS_OUTPUTS = ("rushten", "rvshten", "rthshten", "rqvshten",
                       "rqcshten", "rqishten", "rqcnshten", "rqinshten",
                       "dlf", "dlf2", "shfrc3d", "evapcsh", "icwmrsh",
                       "rprdsh")
UWSHCU_FULL_OUTPUTS = ("cmfmc", "cmfmc2", "cmfsl", "cmflq")
UWSHCU_SURFACE_OUTPUTS = ("pratesh", "cutop", "cubot", "rliq", "rliq2")
#: The scheme's carried state: WRF's CUSH (convective scale height), read
#: as the next step's first-guess scale height (module_shcu_camuwshcu.F:
#: 1495, 2343-2344) and restart-carried like any Registry state.
UWSHCU_STATE_2D = ("cush",)

#: Threads per block of the UW shallow cumulus column launch.
UWSHCU_BLOCK = 64

#: compute_uwshcu's automatic arrays per column, in binary64 slots per
#: interface level (nk + 1).  The device transcription takes every view
#: unconditionally at the top of the column (uwshcu.cu uwsc_take_views:
#: 28 layer + 8 interface; uwshcu_column.cuh uwsc_column: 80 layer + 27
#: interface), so the high-water mark is 143 * nk + 35 binary64 words; 150
#: per interface level covers it with margin and the device still refuses
#: an overflow by name.  No int32 workspace.
UWSHCU_R8_SLOTS_PER_LEVEL = 150
UWSHCU_WORKSPACE_BUDGET_BYTES = 512 * 1024 * 1024


def uwshcu_workspace_slots(nk: int) -> tuple[int, int]:
    """``(r8_slots, i4_slots)`` one column of ``nk`` mass levels needs."""
    return (UWSHCU_R8_SLOTS_PER_LEVEL * (int(nk) + 1), 0)


def uwshcu_chunk_columns(nk: int, ncols: int,
                         budget_bytes: int = UWSHCU_WORKSPACE_BUDGET_BYTES
                         ) -> int:
    """Columns per UW shallow cumulus launch so the pool fits the budget."""
    r8, _ = uwshcu_workspace_slots(nk)
    chunk = max(1, int(budget_bytes) // (8 * r8 + 4))
    if chunk >= UWSHCU_BLOCK:
        chunk = chunk // UWSHCU_BLOCK * UWSHCU_BLOCK
    return int(min(chunk, max(int(ncols), 1)))


def uwshcu_workspace_bytes(nk: int, ncols: int,
                           budget_bytes: int = UWSHCU_WORKSPACE_BUDGET_BYTES
                           ) -> int:
    """Device bytes the UW shallow cumulus pool holds for one call."""
    r8, _ = uwshcu_workspace_slots(nk)
    chunk = uwshcu_chunk_columns(nk, ncols, budget_bytes)
    return int(chunk * (8 * r8 + 4) + 4)


# Hoisted from woof.core.microphysics (module-scope cupy) for the
# same reason as the tables above: the preflight scratch registry
# prices these snapshot slots on installs with no GPU runtime.
def ring_guard_row(mp_physics: int) -> dict[str, list[str]]:
    """The registry's ``consumers.ring_guard`` row for ``mp_physics``.

    ``state_fields`` are the 3-D state arrays the guard captures around a
    microphysics call; ``surface_slots`` the (ny, nx) accumulator and
    diagnostic slots the scheme writes.  A scheme without a row is refused
    by name: pricing a ring the guard would then capture differently is
    exactly the under-budgeting the allocation gate exists to stop.
    """

    from woof.physics_registry import consumer_rows_by_selector

    row = consumer_rows_by_selector("microphysics", "ring_guard").get(
        int(mp_physics))
    if not isinstance(row, dict):
        raise ValueError(
            f"mp_physics={mp_physics} has no consumers.ring_guard row in "
            "woof/physics_registry_v2.json, so its specified-zone ring "
            "cannot be priced; give the option its row in "
            "tools/build_registry.py and regenerate the registry")
    return {"state_fields": list(row["state_fields"]),
            "surface_slots": list(row["surface_slots"])}


def ring_guard_state_fields() -> tuple[str, ...]:
    """The union, over every implemented scheme, of the ring guard's rows.

    The family ``woof.core.microphysics`` captures around a call, in row
    order: presence guards at the capture site keep a scheme unaffected by
    another scheme's names, so the union is what lets a scheme's registry
    moment names be captured at all.

    Hoisted here for the same reason :func:`ring_guard_row` was -- that
    module imports cupy at module scope, so on a card-free install (a CPU
    runner, a preflight-only install) the union could not be read at all,
    and the every-scheme coverage that says the captured family and the
    priced family are ONE family could not be checked anywhere.
    """

    from woof.physics_registry import consumer_rows_by_selector

    rows = consumer_rows_by_selector("microphysics", "ring_guard")
    names: list[str] = []
    for mp in sorted(rows):
        if not isinstance(rows[mp], dict):
            continue
        for name in rows[mp]["state_fields"]:
            if name not in names:
                names.append(name)
    return tuple(names)


def spec_zone_ring_slices(ny: int, nx: int, sz: int):
    """Non-overlapping index tuples covering exactly the ring WRF's
    clipped microphysics tiles exclude.

    WRF (1-based, ide/jde staggered ends): tiles run
    ``its = ids+sz .. ide-1-sz``, ``jts = jds+sz .. jde-1-sz``
    (solve_em.F:3631-3639, :4040-4048;
    module_microphysics_driver.F:870-879).  On woof's 0-based (ny, nx)
    mass grid the surviving tile is ``sz .. nx-1-sz`` x ``sz .. ny-1-sz``
    and the excluded ring is ``i < sz or i > nx-1-sz or j < sz or
    j > ny-1-sz``.  The leading Ellipsis makes each tuple apply to
    (ny, nx) and (nz, ny, nx) arrays alike.  Degenerate domains
    (``2*sz >= ny`` or ``nx`` -- WRF's clip leaves an empty tile) are
    covered without overlap.

    Hoisted from :mod:`woof.core.microphysics` (module-scope cupy), which
    re-exports it: a moving nest's host-side accumulation carry
    (:mod:`woof.core.physics_continuation`) reads the same ring and must
    import on an install with no GPU runtime.
    """
    n_lo = min(sz, ny)
    n_hi = max(ny - sz, n_lo)
    e_lo = min(sz, nx)
    e_hi = max(nx - sz, e_lo)
    return (
        (Ellipsis, slice(0, n_lo), slice(None)),          # south rows
        (Ellipsis, slice(n_hi, ny), slice(None)),         # north rows
        (Ellipsis, slice(n_lo, n_hi), slice(0, e_lo)),    # west columns
        (Ellipsis, slice(n_lo, n_hi), slice(e_hi, nx)),   # east columns
    )


def spec_zone_ring_save_slots(cfg: RunConfig) -> dict[str, tuple[int, ...]]:
    """Preflight-registry helper: every ``mp_ring_save_*`` snapshot slot the
    ring guard creates for this config, with its exact shape.

    Mirrors :func:`_capture_spec_zone_ring` and
    :func:`spec_zone_ring_slices`: one slot per (captured array, non-empty
    ring edge).  Empty when the guard is off (periodic/open, sz = 0, or no
    microphysics).  Consumed by
    ``woof.core.preflight.scratch_slot_registry`` so the completeness and
    allocation gates see the family with true sizes.
    """
    if not (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)):
        return {}
    sz = int(cfg.spec_zone)
    if sz <= 0 or cfg.mp_physics == 0 or not cfg.moist:
        return {}
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    # The scheme's snapshot family comes from the REGISTRY's row for it
    # (``consumers.ring_guard``, tools/build_registry.py), the same row
    # woof.core.microphysics captures the union of, so the priced family
    # and the captured family cannot drift from each other.  The if-chain
    # this replaces priced the condensate and radii arm for
    # (6, 8, 9, 10, 16, 28) and not for 18, so an NSSL-2 nested run was
    # under-priced and its nine number/volume moments were left advancing
    # in the ring WRF's clipped tiles never touch.
    ring = ring_guard_row(int(cfg.mp_physics))
    fields = list(ring["state_fields"])
    # Every scheme (Kessler's rain-only fallback included) stashes due
    # reflectivity into the same persistent refl_10cm slot, and the guard
    # captures that slot unconditionally once it exists -- so the family
    # must be enumerated for every scheme or the first due-reflectivity
    # call would allocate unbudgeted mp_ring_save_refl_10cm_* buffers
    # behind the allocation gate.
    volume_slots = ["refl_10cm"]
    surface_slots = list(ring["surface_slots"])
    n_lo = min(sz, ny)
    n_hi = max(ny - sz, n_lo)
    e_lo = min(sz, nx)
    e_hi = max(nx - sz, e_lo)
    edge_dims = ((n_lo, nx), (ny - n_hi, nx),
                 (n_hi - n_lo, e_lo), (n_hi - n_lo, nx - e_hi))
    slots: dict[str, tuple[int, ...]] = {}
    for key in fields + volume_slots:
        for index, (rows, cols) in enumerate(edge_dims):
            if rows and cols:
                slots[f"mp_ring_save_{key}_{index}"] = (nz, rows, cols)
    for key in surface_slots:
        for index, (rows, cols) in enumerate(edge_dims):
            if rows and cols:
                slots[f"mp_ring_save_{key}_{index}"] = (rows, cols)
    return slots


# ---------------------------------------------------------------------------
# The YSU per-thread column workspace, priced without a runtime
# ---------------------------------------------------------------------------
# These live HERE and not in :mod:`woof.core.ysu` for the reason this
# module exists: ysu.py imports cupy at module scope, and the VRAM
# estimator prices the YSU workspace on installs with no GPU runtime
# (``woof domain --card`` on a CuPy-less box reached ``import cupy``
# through exactly this pricing and refused, caught by the publish test
# job's replay before the 2.5.3 tag).  ysu.py re-exports them, so the
# launcher and the kernel-pin test keep their one authority.
#
# YSUWS_SLOTS must match ysu.cu's YSUWS_SLOTS / YSUWS_LANES;
# tests/test_ysu_workspace.py re-derives them from the .cu source and
# fails if either side moves alone.
YSUWS_SLOTS = 18

#: Launch block, and the tile's granularity.  ysu.cu indexes the
#: workspace by the thread's lane within its block, so a tile is always a
#: whole number of blocks.  Pinned against the kernel's ``YSUWS_LANES``.
YSU_BLOCK = 32

#: Blocks per SM the tile is sized for.  MEASURED, not assumed -- see
#: docs/kernel_local_memory_bounds.md for the sweep this came from.  The
#: device query in :func:`woof.core.ysu.ysu_tile_columns` only ever
#: lowers it.  development branch (b28c20509) doubled it to queue a second
#: wave; 2.8.2 restores 16 because the doubled workspace is priced on every
#: YSU domain (313,344,000 more bytes on an RTX 5090 at nz = 50) and turned
#: trees that fit into refusals (tests/test_streamed_admission.py,
#: tests/test_auto_search_host_floor.py) for a 0.06 to 0.3 percent
#: forecast-hour gain.
YSU_TILE_BLOCKS_PER_SM = 16


def ysu_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: ysu.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so
    the unit of allocation is one block's region, not one column's.

    The per-slot extent is ``nz + 1`` rather than the kernel's compile-time
    ``KMAX``: ``zq`` is the one array indexed at ``nz``, and unlike the
    compile-time frame this is allocated when ``nz`` is known.  A 49-level
    run therefore holds 50 levels of arrays where the frame had to hold
    128.
    """
    blocks = (int(columns) + YSU_BLOCK - 1) // YSU_BLOCK
    return blocks * YSUWS_SLOTS * (int(nz) + 1) * YSU_BLOCK


# EEPS's native work-array inventory and bounded tile geometry.
EEPS_WS_SLOTS = 22
EEPS_BLOCK = 128
EEPS_TILE_BLOCKS_PER_SM = 4


def eeps_workspace_floats(nz: int, columns: int) -> int:
    """Coalesced workspace, rounded up to the launch block."""
    blocks = (int(columns) + EEPS_BLOCK - 1) // EEPS_BLOCK
    return blocks * EEPS_BLOCK * EEPS_WS_SLOTS * (int(nz) + 2)
#: The GFS nonlocal PBL's (``bl_pbl_physics = 3``) per-call output roster
#: (woof/core/gfs_pbl.py ``gfs_pbl_step``), in kernel argument order: the
#: six BL_GFS tendencies, (nz, ny, nx) float32.  The scheme carries no
#: state of its own between calls; it publishes PBLH/KPBL and rewrites the
#: surface layer's UST and WSPD (module_bl_gfs.F:463-475).
GFS_PBL_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                          "rqcblten", "rqiblten")
GFS_PBL_SURFACE_OUTPUTS = ("ust", "pblh", "kpbl")
#: Launch block of ``gfs_pbl_columns``; a chunk is a whole number of blocks.
GFS_PBL_BLOCK = 64
#: binary64 workspace slots per level per column: DEL, PRSI, THETA, DKU,
#: DKT and the tridiagonal AU/A1/A2/A3/A4 (kernels/gfs_pbl.cu
#: GFS_WS_SLOTS), each nz+1 long.  tests/test_gfs_pbl_integration.py
#: re-reads the define from the kernel source.
GFS_PBL_WS_SLOTS = 10
#: VRAM the launcher may hold for the workspace at once; larger domains are
#: walked in column chunks.
GFS_PBL_WORKSPACE_BUDGET_BYTES = 256 * 1024 * 1024


def gfs_pbl_workspace_slots(nz: int) -> int:
    """binary64 words of workspace per column at ``nz`` levels."""
    return GFS_PBL_WS_SLOTS * (int(nz) + 1)


def gfs_pbl_chunk_columns(nz: int, ncols: int,
                          budget_bytes: int = GFS_PBL_WORKSPACE_BUDGET_BYTES
                          ) -> int:
    """Columns per launch: the budget's worth, whole blocks, at least one."""
    per_column = 8 * gfs_pbl_workspace_slots(nz)
    chunk = max(1, int(budget_bytes) // per_column)
    if chunk >= GFS_PBL_BLOCK:
        chunk = chunk // GFS_PBL_BLOCK * GFS_PBL_BLOCK
    return max(1, min(chunk, int(ncols)))


def gfs_pbl_workspace_bytes(nz: int, ncols: int,
                            budget_bytes: int = GFS_PBL_WORKSPACE_BUDGET_BYTES
                            ) -> int:
    """Device bytes one GFS PBL call holds for its workspace."""
    return 8 * gfs_pbl_workspace_slots(nz) * gfs_pbl_chunk_columns(
        nz, ncols, budget_bytes)


# These constants mirror the kernel geometry for runtime-free memory pricing.
SHWS_SLOTS = 50
SHINHONG_BLOCK = 32
SHINHONG_TILE_BLOCKS_PER_SM = 16


def shinhong_workspace_floats(nz: int, columns: int) -> int:
    """Floats for whole blocks with 1-based levels and a top sentinel."""
    blocks = (int(columns) + SHINHONG_BLOCK - 1) // SHINHONG_BLOCK
    return blocks * SHWS_SLOTS * (int(nz) + 2) * SHINHONG_BLOCK


#: The GBM moist TKE PBL's (``bl_pbl_physics = 12``) carried (nz) fields
#: and their cold start, allocated by ``initialize_physics`` for that
#: selector only and priced by the VRAM estimate from this same table.
#: ``tke_pbl`` is PROGNOSTIC: read at the top of every call and rewritten
#: with the first GBM2D pass's TKE (module_bl_gbmpbl.F:247, :276), so a
#: restart must carry it; gbmpblinit seeds it at 0.001 (:1252).  ``el_pbl``
#: and ``exch_tke`` are INTENT(OUT), rewritten by every call, zero being the
#: Registry default.  EXCH_M/EXCH_H are the slot's shared ``exch_m``/
#: ``exch_h``.
GBMPBL_STATE_3D = MappingProxyType(
    {"tke_pbl": 0.001, "el_pbl": 0.0, "exch_tke": 0.0})
#: Mass-level float32 outputs one GBM call allocates (woof/core/gbmpbl.py
#: ``gbmpbl_step``), held here so the launcher allocates from it and the
#: preflight prices from it.  The surface outputs are PBLH and UST
#: (float32) and the int32 KPBL.
GBMPBL_MASS_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                       "rqcblten", "rqiblten", "exch_m", "exch_h",
                       "exch_tke", "el_pbl")
GBMPBL_SURFACE_OUTPUTS = ("pblh", "ust", "kpbl")
#: Kernel geometry (GBWS_LANES / GBWS_SLOTS in kernels/gbmpbl.cu) and the
#: launch tile, mirrored for runtime-free memory pricing.
GBMPBL_BLOCK = 32
GBMPBL_WS_SLOTS = 49
GBMPBL_TILE_BLOCKS_PER_SM = 8


def gbmpbl_workspace_floats(nz: int, columns: int) -> int:
    """Float32 words of GBM workspace: 1-based levels to kte+2."""
    blocks = (int(columns) + GBMPBL_BLOCK - 1) // GBMPBL_BLOCK
    return blocks * GBMPBL_WS_SLOTS * (int(nz) + 3) * GBMPBL_BLOCK
# The MRF PBL (bl_pbl_physics = 99) column workspace, priced without a
# runtime for the same reason as YSU's.  MRFWS_SLOTS and MRF_BLOCK must
# match kernels/mrf.cu's MRFWS_SLOTS / MRFWS_LANES / MRF_BLOCK;
# tests/test_mrf_port.py re-reads them from the source.  Eleven column
# arrays (ZQ, ZA, DZA, THVX, XKZM, XKZH and the five tridiagonal
# operands), 1-based in MRF's own K/KK with ZQ(KLP1), so nz + 2 per slot.
MRFWS_SLOTS = 11
MRF_BLOCK = 32
#: YSU's measured blocks-per-SM tile, which the device occupancy query in
#: woof.core.mrf.mrf_tile_columns only ever lowers.
MRF_TILE_BLOCKS_PER_SM = 16


def mrf_workspace_floats(nz: int, columns: int) -> int:
    """Floats for whole blocks of columns: 1-based levels plus ZQ(nz+1)."""
    blocks = (int(columns) + MRF_BLOCK - 1) // MRF_BLOCK
    return blocks * MRFWS_SLOTS * (int(nz) + 2) * MRF_BLOCK

#: The K-epsilon PBL (``bl_pbl_physics = 17``, WRF v4.7.1
#: module_bl_keps.F).  Its carried fields, ArWen name -> WRF name
#: (Registry.EM_COMMON:1231-1243 and the kepsscheme package, :3176),
#: allocated by ``initialize_physics`` for that selector only and priced by
#: the VRAM estimate from this same table.  All are (nz, ny, nx):
#:
#: * prognostic: TKE_PBL, DISS_PBL, TPE_PBL, read back every call
#:   (module_bl_keps.F:278-280);
#: * carried, never read after the reset: PR_PBL (overwritten from the
#:   surface layer before any use, :1043) and the three advected copies
#:   TKE_ADV, DISS_ADV, TPE_ADV (read only when the compile-time
#:   ``bl_keps_adv`` is 1; v4.7.1 ships 0) -- carried so a restart is exact;
#: * published: WU_TUR, WV_TUR, WT_TUR; WQ_TUR is INTENT(OUT) in WRF and
#:   never assigned, so it stays the zero it is allocated as.
KEPS_FIELDS = {"keps_tke": "tke_pbl", "keps_diss": "diss_pbl",
               "keps_tpe": "tpe_pbl", "keps_pr": "pr_pbl",
               "keps_tke_adv": "tke_adv", "keps_diss_adv": "diss_adv",
               "keps_tpe_adv": "tpe_adv"}
#: The launcher's state keys (WRF names), in kernel argument order.
KEPS_STATE_3D = ("tke_pbl", "diss_pbl", "tpe_pbl", "pr_pbl", "tke_adv",
                 "diss_adv", "tpe_adv")
KEPS_STATE_2D = ("pblh",)

# WRF Registry.EM_COMMON:3176. These scalar-package fields are transported
# even though the shipped column closure does not read the transport copies.


def pbl_advected_scalars(cfg) -> tuple[str, ...]:
    return tuple(name for name, _ in PBL_ADVECTED_SCALARS.get(int(cfg.bl_pbl_physics), ()))

KEPS_DIAGNOSTICS_3D = ("wu_tur", "wv_tur", "wt_tur", "wq_tur")
#: Kernel geometry (kernels/keps.cu KE_SLOTS, the launch block, the resident
#: tile the workspace is sized to), mirrored for runtime-free pricing and
#: held equal to the kernel by tests/test_keps_integration.py.
KEPS_SLOTS = 32
KEPS_BLOCK = 64
# Allow queued column blocks as well as resident blocks. A four-block cap
# makes a small grid remainder repeat the entire sequential column in a
# persistent thread, even when the card has room for that remainder.
KEPS_TILE_BLOCKS_PER_SM = 8


def keps_workspace_floats(nz: int, tile: int) -> int:
    """Floats of the column workspace for ``tile`` threads in flight:
    KEPS_SLOTS rows of nz+2 one-based levels per thread."""
    return KEPS_SLOTS * (int(nz) + 2) * int(tile)

# ACM2 (bl_pbl_physics=7) column workspace geometry: kernels/acm2pbl.cu is
# compiled with these as ACM2_BLOCK / ACM2WS_SLOTS (woof/core/acm2pbl.py),
# and the preflight prices the same numbers without the GPU runtime.
ACM2WS_SLOTS = 27
ACM2_BLOCK = 32
ACM2_TILE_BLOCKS_PER_SM = 16


def acm2_workspace_floats(nz: int, columns: int) -> int:
    """Floats for whole blocks: 27 column arrays of levels 0..nz+1 (WRF's
    1-based arrays plus the kte+1 element ACM/ACMM touch)."""
    blocks = (int(columns) + ACM2_BLOCK - 1) // ACM2_BLOCK
    return blocks * ACM2WS_SLOTS * (int(nz) + 2) * ACM2_BLOCK


__all__ = [
    "spec_zone_ring_save_slots", "spec_zone_ring_slices",
    "MYJ_PBL_STATE_3D", "MYJ_SFCLAY_FIELDS_2D", "MYJ_SFCLAY_INOUT",
    "MRF_PBL_DIAGNOSTICS_2D",
    "GFS_PBL_BLOCK", "GFS_PBL_COLUMN_OUTPUTS", "GFS_PBL_SURFACE_OUTPUTS",
    "GFS_PBL_WORKSPACE_BUDGET_BYTES", "GFS_PBL_WS_SLOTS",
    "gfs_pbl_chunk_columns", "gfs_pbl_workspace_bytes",
    "gfs_pbl_workspace_slots",
    "QNSE_TKE_COLD_START", "MFSHCONV_STATE_3D", "MFSHCONV_OUTPUTS_3D",

    "BOULAC_STATE_3D", "BOULAC_COLD_START", "BOULAC_FLUX_FIELDS", "BOULAC_BEP_INPUTS",
    "BOULAC_COLUMN_OUTPUTS", "BOULAC_FLUX_OUTPUTS", "BOULAC_BLOCK",
    "UWPBL_STATE_FULL", "UWPBL_STATE_2D", "UWPBL_DIAGNOSTICS_FULL",
    "UWPBL_DIAGNOSTICS_2D", "UWPBL_HELD_3D", "UWPBL_MASS_OUTPUTS",
    "UWPBL_SURFACE_OUTPUTS", "UWPBL_BLOCK", "UWPBL_R8_SLOTS_PER_LEVEL",
    "UWPBL_I4_SLOTS_PER_LEVEL", "UWPBL_WORKSPACE_BUDGET_BYTES",
    "uwpbl_workspace_slots", "uwpbl_chunk_columns", "uwpbl_workspace_bytes",
    "TEMF_PBL_STATE_3D", "TEMF_PBL_STATE_2D", "TEMF_SFCLAY_STATE_2D",
    "TEMF_SFCLAY_INOUT", "TEMF_SFCLAY_OUTPUTS", "TEMF_BLOCK", "TEMF_WS_SLOTS",
    "TEMF_WORKSPACE_BUDGET_BYTES", "temf_workspace_floats",
    "temf_chunk_columns", "temf_workspace_bytes",

    "UWSHCU_MASS_INPUTS", "UWSHCU_FULL_INPUTS", "UWSHCU_SURFACE_INPUTS",
    "UWSHCU_MASS_OUTPUTS", "UWSHCU_FULL_OUTPUTS", "UWSHCU_SURFACE_OUTPUTS",
    "UWSHCU_STATE_2D", "UWSHCU_BLOCK", "UWSHCU_R8_SLOTS_PER_LEVEL",
    "UWSHCU_WORKSPACE_BUDGET_BYTES", "uwshcu_workspace_slots",
    "uwshcu_chunk_columns", "uwshcu_workspace_bytes",
    "MYJ_SFCLAY_OUTPUTS",
    "TEMF_SFCLAY_INPUTS", "TEMF_SFCLAY_CARRIED", "TEMF_SFCLAY_COLUMN_OUTPUTS",
    "TEMF_SFCLAY_FIELDS_2D",
    "MYNN_PBL_DIAGNOSTICS_2D", "MYNN_PBL_DIAGNOSTICS_INT_2D",
    "MYNN_PBL_STATE_3D", "MYNN_SURFACE_OUTPUTS",
    "EEPS_PBL_STATE_3D", "PBL_TRANSPORTED_SCALARS", "pbl_transported_scalars",
    "PBL_RQI_MICROPHYSICS", "PHYSICS_SLOT_DISPATCH", "REFL_10CM_MICROPHYSICS",
    "UnroutedPhysicsSelectorError", "resolve_physics_slot",
    "SFCLAY_OUTPUTS",
    "SHWS_SLOTS", "SHINHONG_BLOCK", "SHINHONG_TILE_BLOCKS_PER_SM",
    "shinhong_workspace_floats",
    "GBMPBL_STATE_3D", "GBMPBL_MASS_OUTPUTS", "GBMPBL_SURFACE_OUTPUTS",
    "GBMPBL_BLOCK", "GBMPBL_WS_SLOTS", "GBMPBL_TILE_BLOCKS_PER_SM",
    "gbmpbl_workspace_floats",
    "MRFWS_SLOTS", "MRF_BLOCK", "MRF_TILE_BLOCKS_PER_SM",
    "mrf_workspace_floats",

    "KEPS_BLOCK", "KEPS_DIAGNOSTICS_3D", "KEPS_FIELDS", "KEPS_SLOTS",
    "KEPS_STATE_2D", "KEPS_STATE_3D", "KEPS_TILE_BLOCKS_PER_SM",
    "keps_workspace_floats",

    "ACM2WS_SLOTS", "ACM2_BLOCK", "ACM2_TILE_BLOCKS_PER_SM",
    "acm2_workspace_floats",
    "YSUWS_SLOTS", "YSU_BLOCK", "YSU_TILE_BLOCKS_PER_SM",
    "hmix_k_diag_names",
    "microphysics_scratch_slots", "physics_driver_required",
    "physics_enabled", "physics_retains_ysu_output",
    "physics_reuses_pbl_composition", "ysu_workspace_floats",
]


#: WRF selector value -> the ``PhysicsDriver`` method that runs THAT scheme.
#: Zero means the slot is off and maps to ``None``.  Every dispatch in
#: :meth:`PhysicsDriver.compute` goes through this table; a value that is
#: absent raises :class:`UnroutedPhysicsSelectorError` instead of falling
#: through to whichever scheme happens to be wired.  ``_run_sfclay``
#: additionally re-dispatches on the exact value internally, so the two
#: MM5 spellings and MYNN cannot be confused with each other either.
PBL_OPTIONAL_TENDENCIES = {901: ("rqr",)}
PBL_ADVECTED_SCALARS = {
    901: (("tke_edmf", 0.2),),
    EEPS_PBL_SCHEME: tuple((name, 0.0) for name in
                           PBL_TRANSPORTED_SCALARS[EEPS_PBL_SCHEME]),
    17: (("keps_tke_adv", 0.0), ("keps_diss_adv", 0.0), ("keps_tpe_adv", 0.0)),
}
PBL_SCALAR_NAMES = tuple(name for rows in PBL_ADVECTED_SCALARS.values() for name, _ in rows)

PHYSICS_SLOT_DISPATCH: dict[str, dict[int, str | None]] = {
    "sf_sfclay_physics": {
        0: None,
        1: "_run_sfclay",     # revised MM5
        # Eta similarity (WRF v4.6.1 module_sf_myjsfc.F).  It gets its OWN
        # runner rather than a fourth _run_sfclay arm because it publishes
        # a different set: AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 and no MOL, ZOL,
        # PSIM/PSIH, REGIME, GZ1OZ0 or WSPD.  woof.config.
        # validate_myj_pairing refuses it with any PBL but MYJ for exactly
        # that reason.
        2: "_run_myj_sfclay",
        # GFS (WRF v4.6.1 module_sf_gfs.F, woof/core/sf_gfs.py).  Its own
        # runner for the Eta layer's reason: it publishes the shared
        # exchange set (CHS/CHS2/CQS2/FLHC/FLQC/QGH/QSFC/HFX/QFX/LH, BR,
        # PSIM/PSIH, GZ1OZ0, WSPD, U10/V10) and carries ZNT/UST, but no
        # MOL, ZOL, FM/FH, REGIME or T2/Q2/TH2 -- SF_GFS has none.
        3: "_run_gfs_sfclay",
        4: "_run_myj_sfclay",
        5: "_run_sfclay",     # MYNN surface layer
        10: "_run_temf_sfclay",
        89: "_run_ideal_scm_sfclay",

        # Pleim-Xiu (WRF v4.6.1 module_sf_pxsfclay.F).  Its own runner for
        # the Eta layer's reason: it publishes a different set (no FM/FH,
        # USTM, TH2/T2/Q2) and under RUC it is WRF's PXSFCLAY_SEAICE_WRAPPER,
        # not the MM5 wrapper the _run_sfclay arm reproduces.
        7: "_run_px_sfclay",
        91: "_run_sfclay",    # classic MM5
    },
    "sf_ocean_physics": {0: None, 1: "_run_surface_ocean", 2: "_run_surface_ocean"},
    "sf_surface_physics": {
        0: None,
        # Five-layer thermal diffusion (WRF v4.6.1 module_sf_slab.F), graded
        # word for word against the byte-unmodified module
        # (tests/test_slab_wrf461_oracle.py).
        1: "_run_slab",
        2: "_run_noah",       # Noah LSM
        3: "_run_ruc",        # RUC LSM
        4: "_run_noahmp",     # Noah-MP LSM
        7: "_run_pxlsm",      # Pleim-Xiu, two-layer soil
        8: "_run_column_land_surface",
    },
    "bl_pbl_physics": {
        0: None,
        1: "_run_ysu",        # YSU
        # MYJ (WRF v4.6.1 module_bl_myjpbl.F), Mellor-Yamada level 2.5 as
        # extended by Janjic.  Unlike every other row here the scheme does
        # its OWN implicit vertical diffusion, so _run_myj_pbl receives
        # finished tendencies rather than diffusivities.  Its surface
        # pairing is enforced in woof.config.validate_myj_pairing, which
        # is WRF's own fatal at module_physics_init.F:3770-3772.
        2: "_run_myj_pbl",
        # GFS nonlocal PBL (WRF v4.6.1 module_bl_gfs.F, GFSSCHEME): Hong-Pan
        # K profile with its OWN implicit diffusion in binary64, so like MYJ
        # and UW _run_gfs_pbl receives finished tendencies.  It reads the
        # surface layer's UST/PSIM/PSIH/GZ1OZ0/WSPD/BR and rewrites UST and
        # WSPD (module_bl_gfs.F:463-466); woof.config.validate_gfs_pbl_config
        # admits the surface layers that publish that set.
        3: "_run_gfs_pbl",
        # QNSE (WRF v4.6.1 module_bl_qnsepbl.F) with its EDMF shallow mass
        # flux (module_bl_mfshconvpbl.F, mfshconv = 1).  MYJ's column with
        # QNSE's mixing length, stability functions and TKE budget, and like
        # MYJ it does its own implicit diffusion and returns finished
        # tendencies.  Its surface pairing (WRF's own fatal,
        # module_physics_init.F:3790-3791) is woof.config.
        # validate_qnse_config.
        4: "_run_qnse_pbl",
        5: "_run_mynn_pbl",   # MYNN EDMF
        10: "_run_temf_pbl",
        # Shin-Hong scale-aware (WRF v4.6.1 module_bl_shinhong.F).  The
        # scheme has NO RunConfig knobs on purpose: WRF's
        # shinhong_tke_diag namelist is deliberately not imported.  The
        # TKE chain is a pure passenger diagnostic -- the tendencies
        # never read it, proven on the pinned source (the case-27/case-1
        # oracle pair in tests/test_shinhong_wrf461_parity.py pins
        # OFF/ON bitwise-identical tendencies) -- so ArWen computes it
        # every step under scheme 11 (_run_shinhong passes tke_diag=1
        # unconditionally) and publishes it as state.e_sgs, the field
        # the D1 gray-zone instrument scores.
        11: "_run_shinhong",
        # BouLac (WRF v4.6.1 module_bl_boulac.F, BOULACSCHEME).  Like MYJ it
        # performs its own implicit diffusion and returns finished
        # tendencies; it carries its own TKE_PBL, and under BEP/BEM its
        # column takes the urban source terms itself.
        BOULAC_PBL_SCHEME: "_run_boulac",
        # UW moist turbulence (WRF v4.7.1 module_bl_camuwpbl_driver.F,
        # CAMUWPBLSCHEME).  Like MYJ it performs its own implicit
        # diffusion and returns finished tendencies; unlike every other
        # row it computes in binary64 (CAM's real(r8)) and reads the
        # radiation step's held RTHRATENLW and CLDFRA.
        UW_PBL_SCHEME: "_run_uwpbl",
        # GBM moist TKE (WRF v4.6.1 module_bl_gbmpbl.F, GBMPBLSCHEME).
        # Like MYJ and UW it does its own implicit diffusion and returns
        # finished tendencies; it reads the held RTHRATEN and the MM5
        # surface layer's GZ1OZ0/PSIM (woof.config.validate_gbmpbl_config).
        GBM_PBL_SCHEME: "_run_gbmpbl",
        # EEPS, TKE + dissipation (WRF v4.6.1 module_bl_eepsilon.F,
        # EEPSSCHEME).  Its own implicit diffusion, finished tendencies;
        # its TKE/epsilon pair is a dycore-transported scalar
        # (PBL_TRANSPORTED_SCALARS) that the column reads and rewrites.
        EEPS_PBL_SCHEME: "_run_eeps",
        # MRF (WRF v4.6.1 module_bl_mrf.F, MRFSCHEME), the Hong-Pan nonlocal
        # PBL YSU descends from.  Its own runner because its coupling is
        # its own: it reads the MM5 layers' RAW PSIM/PSIH (not YSU's fm/fh
        # pair), the temperature rather than theta, and rewrites UST.
        # Graded word for word against WRF's compiled module
        # (tests/test_mrf_wrf461_parity.py).
        MRF_PBL_SCHEME: "_run_mrf",

        # K-epsilon (WRF v4.7.1 module_bl_keps.F, KEPSSCHEME).  It performs
        # its own implicit diffusion and returns finished tendencies, like
        # MYJ and UW; it carries TKE, dissipation and TPE across calls.
        KEPS_PBL_SCHEME: "_run_keps",

        # ACM2 (WRF v4.6.1 module_bl_acm.F, ACMPBLSCHEME): the Pleim
        # combined local/non-local closure.  Like MYJ and UW it performs its
        # own semi-implicit mixing (sub-stepped, with the bordered band
        # solver MATRIX on free-convection columns and TRI elsewhere) and
        # returns finished tendencies.
        ACM2_PBL_SCHEME: "_run_acm2",
        # SASE: not a WRF scheme and deliberately outside WRF's selector
        # namespace, so it can never collide with one WRF adds later.
        SASE_PBL_SCHEME: "_run_sase",
        901: "_run_satmedmf",
    },
}


class UnroutedPhysicsSelectorError(NotImplementedError):
    """A configured surface/LSM/PBL selector VALUE has no routed scheme.

    The driver used to branch on the truthiness of ``sf_surface_physics``
    and ``bl_pbl_physics``, so *any* nonzero value ran Noah and YSU.
    Unlocking a new selector without adding a routing row would therefore
    have produced plausible numbers from the wrong scheme with no error.
    This is that error.
    """

    def __init__(self, selector: str, value, routed):
        self.selector = selector
        self.value = value
        self.routed = tuple(sorted(routed))
        super().__init__(
            f"{selector}={value!r} has no scheme routed to it in the woof "
            "physics driver; refusing to substitute another scheme. Routed "
            f"values: {list(self.routed)}. Adding the selector to "
            "woof/config.py and woof/physics_compat.py is NOT sufficient: "
            f"add a {selector} row to PHYSICS_SLOT_DISPATCH "
            "(woof/core/physics_inventory.py) naming the PhysicsDriver method that "
            "runs that exact scheme.")


#: Land-surface schemes after which WRF's surface driver calls the ordinary
#: ``SFCDIAGS`` (module_surface_driver.F:2574-2583 for slab and
#: :2983-2998 for Noah). RUC instead calls
#: ``SFCDIAGS_RUCLSM`` in :func:`ruc_lsm_step`; Noah-MP unconditionally
#: selects its water/ice flux diagnostic or its own vegetation/bare-ground
#: 2-m fields in :func:`noahmp_lsm_step`.  This is a per-scheme decision, not
#: a property of "an LSM ran", and all three post-LSM paths overwrite the
#: surface layer's earlier T2/Q2/TH2 exactly where WRF does.
LAND_SURFACE_SFCDIAGS_SCHEMES = frozenset({1, 2})


def sase_switch_value(cfg, name: str) -> bool:
    """Read one SASE switch, falling back to RunConfig's SHIPPED default.

    Every switch here is a per-run key whose default is declared once, in
    :class:`~woof.config.RunConfig`.  A configuration object that does
    not carry the key must land on that declared default and nothing
    else: a literal fallback written at the call site is a second
    declaration of the same default, and a second declaration drifts.
    It already had.  ``sase_additive_dissipation`` ships ON, and the
    driver read it with a literal ``False`` fallback, so any object
    without the field would have dropped the dissipation channel without
    saying so -- the opposite of what a default-on remedy means.  The
    fallback is now read off the dataclass, so the call site cannot
    disagree with ``woof/config.py`` about what ships.

    RunConfig itself always carries all three keys, so for a RunConfig
    this is ``getattr`` and nothing more.
    """
    return bool(getattr(cfg, name, getattr(RunConfig, name)))


def resolve_physics_slot(selector: str, value) -> str | None:
    """Resolve one selector VALUE to its runner method, or fail loudly."""
    table = PHYSICS_SLOT_DISPATCH[selector]
    try:
        key = int(value)
    except (TypeError, ValueError):
        raise UnroutedPhysicsSelectorError(selector, value, table) from None
    if key not in table:
        raise UnroutedPhysicsSelectorError(selector, key, table)
    return table[key]


# WRF G3 restart carriers and convtrans_prep accumulators. Shape tokens
# resolve through cfg; the same rows drive allocation and memory review.
CONVTRANS_LAYOUT = {
    **{name: (("nz", "ny", "nx"), "f4", "convtrans") for name in
       ("gd_cloud_a", "qc_cu", "gd_cloud2_a", "qi_cu", "gd_cldfr")},
    "raincv_a": (("ny", "nx"), "f4", "convtrans"),
    "raincv_b": (("ny", "nx"), "f4", "convtrans"),
    "g3_diag_clock": ((4,), "i4", "convtrans"),
}
from woof.core.kfcup_contract import KFCUP_STATE_ROWS

CUMULUS_STATE_LAYOUTS = {1: {
    name: (("nz", "ny", "nx"), "f4", "cu_rad_feedback") for name in
    ("qc_cu", "qi_cu", "cldfra_dp", "cldfra_sh", "cldfra_cu")
}, 3: CONVTRANS_LAYOUT, 10: {
    row["name"]: ((("nz", "ny", "nx") if row["shape"] == "mass" else ("ny", "nx")),
                  "i4" if row["dtype"] == "int32" else "f4", None)
    for row in KFCUP_STATE_ROWS
}, 5: {
    "g3_apr": ((8, "ny", "nx"), "f4", None),
    "g3_xf_ens": (("ensdim", "ny", "nx"), "f4", None),
    "g3_pr_ens": (("ensdim", "ny", "nx"), "f4", None),
    "g3_lev": ((9, "nz", "ny", "nx"), "f4", None),
    "g3_sfc": ((7, "ny", "nx"), "f4", None),
    "g3_isfc": ((4, "ny", "nx"), "i4", None),
    **CONVTRANS_LAYOUT,
}, 14: {
    "hbot": (("ny", "nx"), "f4", None),
    "htop": (("ny", "nx"), "f4", None),
    "ksas_rucuten": (("nz", "ny", "nx"), "f4", None),
    "ksas_rvcuten": (("nz", "ny", "nx"), "f4", None),
}, 94: {
    "sas_hbot": (("ny", "nx"), "f4", None),
    "sas_htop": (("ny", "nx"), "f4", None),
}}
# WRF forces cu_diag on for option 93. These arrays use the existing GPU
# convtrans_prep implementation and the ordinary fields restart/nest path.
CUMULUS_STATE_LAYOUTS[93] = {
    "gd_lev": ((6, "nz", "ny", "nx"), "f4", None),
    "gd_sfc": ((13, "ny", "nx"), "f4", None),
    "gd_isfc": ((5, "ny", "nx"), "i4", None),
    "gd_band": (("ny", "nx"), "i4", None),
    **{
    name: (dims, dtype, None)
    for name, (dims, dtype, _flag) in CONVTRANS_LAYOUT.items()
    },
}
CUMULUS_HISTORY_LAYOUT = {
    **{row["name"]: (row["name"], None) for row in KFCUP_STATE_ROWS},
    "sas_hbot": ("sas_hbot", None), "sas_htop": ("sas_htop", None),
    "hbot": ("hbot", None), "htop": ("htop", None),
    "gd_cloud": ("g3_lev", 7), "gd_cloud2": ("g3_lev", 8),
    **{name: (name, None) for name in
       ("gd_cloud_a", "qc_cu", "gd_cloud2_a", "qi_cu", "gd_cldfr", "raincv_a", "raincv_b",
        "cldfra_dp", "cldfra_sh", "cldfra_cu")},
}

def cumulus_state_layout(cfg):
    def active(flag):
        if flag == "convtrans":
            return cfg.cu_rad_feedback or (cfg.cu_physics == 5 and cfg.cu_diag)
        return flag is None or getattr(cfg, flag)
    return {
        name: (tuple(getattr(cfg, dim) if isinstance(dim,str) else dim for dim in dims), dtype)
        for name,(dims,dtype,flag) in CUMULUS_STATE_LAYOUTS.get(cfg.cu_physics,{}).items()
        if active(flag)
    }


def solar_split_history_expected(cfg):
    """Schemes with a native surface split, or the model's SW interpolation."""
    from woof.config import radiation_scheme_ids
    shortwave = radiation_scheme_ids(cfg)[1]
    return bool(shortwave and (shortwave in (4, 5) or cfg.swint_opt == 1))
def registered_ice_mass_for_radiation(state):
    """WRF module_radiation_driver.F:1284-1304, in package order.

    Native multi-category packages publish their mass slots in the registry.
    Existing packages return their ordinary ice carrier without arithmetic.
    """
    names = getattr(state, "_registered_mass_slots", ())
    if not names:
        ice = getattr(state, "qi", None)
        second = getattr(state, "qi2", None)
        return ice if second is None else ice + second
    import numpy as np
    first = getattr(state, names[0])
    if isinstance(first, np.ndarray):
        xp = np
    else:
        import cupy as xp
    out = state.scratch(first.shape, "native_column_radiation_ice")
    out[...] = first
    for name in names[1:]:
        xp.add(out, getattr(state, name), out=out)
    return out

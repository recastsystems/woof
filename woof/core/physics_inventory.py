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

from woof.config import (RunConfig, SASE_PBL_SCHEME, UW_PBL_SCHEME,
                          radiation_enabled)


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
    if cfg.ra_rrtmg_variant not in (RRTMG_VARIANT_LEGACY,
                                    RRTMG_VARIANT_RTE_RRTMGP):
        return ()
    lw,sw=radiation_scheme_ids(cfg)
    sides = {side for side,selector in (("lw",lw),("sw",sw))
             if int(selector)==4}
    return tuple(name for name in RADIATION_BOUNDARY_FLUXES if name[:2] in sides)

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
    return bool(radiation_enabled(cfg) or cfg.sf_sfclay_physics
                or cfg.sf_surface_physics or getattr(cfg, "sf_ocean_physics", 0)
                or cfg.bl_pbl_physics
                or cfg.cu_physics or getattr(cfg, "ifire", 0) == 2
                or getattr(cfg, "wind_farm_scheme", "off") != "off")


def physics_driver_required(cfg: RunConfig) -> bool:
    """Whether setup must attach a persistent :class:`PhysicsDriver`.

    Microphysics is advanced after RK3 rather than through the non-timesplit
    tendency path selected by :func:`physics_enabled`.  It still needs the
    driver for accumulated precipitation and the output-due REFL_10CM
    handoff, so an mp-only domain must receive the same persistent attachment
    as a domain with radiation, surface, PBL, or cumulus physics.
    """
    return bool(cfg.mp_physics or physics_enabled(cfg))


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
    and literal ``bldt == 0``, ``_run_ysu`` replaces the PBL stack before
    every composition, so radiation/cumulus can be accumulated into it once
    without corrupting a value needed by the next step.
    """
    return bool(cfg.bl_pbl_physics and cfg.bldt == 0.0
                and (radiation_enabled(cfg) or cfg.cu_physics))


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
PBL_RQI_MICROPHYSICS = (2, 4, 5, 6, 8, 9, 10, 16, 18, 24, 28, 30, 50, 106) + tuple(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values() if "qi" in _s.ice_mass_species)


#: Microphysics producers with an output-due radar reflectivity carrier.
REFL_10CM_MICROPHYSICS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 16, 18, 24, 28, 50, 106) + tuple(
    _s.mp_id for _s in _NAMED_MP_SCHEMES.values())
# Kessler has no stock microphysics reflectivity writer. The default
# diagnostic remains available; strict forecast history retains the
# initialized stock slot (module_mp_kessler.F, no REFL_10CM argument).
WRF_INACTIVE_REFL_MICROPHYSICS = frozenset({1})


#: The raw PBL fields needed to repeat mass coupling after a grid move.
PBL_RAW_RATE_NAMES = ("du", "dv", "dtheta", "dqv", "dqc", "dqi")
PBL_SHARED_FORCING = {"dtheta": "gf_rthblten", "dqv": "gf_rqvblten"}

# surface_driver integrates the final land-surface flux once per due call,
# with atmospheric DT rather than the longer land-surface integration DTBL.
SURFACE_ENERGY_ACCUMULATORS = {"achfx": "hfx", "aclhf": "lh", "acgrdflx": "grdflx"}
PBL_HISTORY_RATES = {"rublten": "du", "rvblten": "dv", "rthblten": "dtheta",
                     "rqvblten": "dqv", "rqcblten": "dqc"}


def history_tendency_rows() -> bool:
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
    return bool(ENABLED)


def pbl_history_rates(cfg) -> dict[str, str]:
    """The selected PBL producer's raw history rates (strict processes only,
    see :func:`history_tendency_rows`; a default process gets ``{}``).

    MYNN's wrapper assigns RQIBLTEN independently of whether a prognostic
    ice scalar consumes it (module_pbl_driver.F:1682). Keep its actual
    producer record on the same mass-coupling clock as the other rates.
    """
    if not history_tendency_rows():
        return {}
    rates = dict(PBL_HISTORY_RATES)
    if int(getattr(cfg, "bl_pbl_physics", 0)) == 5:
        rates["rqiblten"] = "dqi"
    return rates
SCM_SURFACE_FORCING_FIELDS = (
    "hfx_force", "lh_force", "tsk_force", "hfx_force_tend",
    "lh_force_tend", "tsk_force_tend")


def scm_forcing_fields() -> tuple[str, ...]:
    """The SCM surface forcing scalars this process carries.

    HFX_FORCE and its siblings are WRF single-column-model controls with no
    admitted producer; the exact chain carries them, held at their initial
    or restored words, so a strict (GPUWM_WRF_EXACT=1) wrfout lists every
    record a recording wrote.  A default process carries none of them.

    THE BREAKAGE THIS PREVENTS (2.9.1 GPU shard leg at e3e6b7dae): carried
    in every process, the six 0-d scalars reached consumers that admit only
    gridded arrays, so default ensembles refused at launch ("driver/fields/
    hfx_force has no admitted member-column shape"), prepared-store scatter
    and nest relocation raised on shape (), and wrfout gained six records
    2.9.0 never wrote.
    """
    return SCM_SURFACE_FORCING_FIELDS if history_tendency_rows() else ()


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
    if mp_physics == 1:
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_kessler_sr"))
    if mp_physics == 5:
        # Ferrier-Aligo.  THREE slots: the driver arm ``CASE
        # (FER_MP_HIRES)`` binds RAINNC, RAINNCV and SR and nothing else
        # (module_microphysics_driver.F:2613-2626).  FER_HIRES puts all
        # surface precipitation into RAINNC and says how much of it fell
        # frozen through SR (module_mp_fer_hires.F:967-974).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"))
    if mp_physics == 3:
        # WSM3: the driver arm binds RAINNC/RAINNCV, SNOWNC/SNOWNCV and SR
        # and no graupel argument (module_microphysics_driver.F:2241-2245).
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"))
    if mp_physics in (2, 6, 8, 10, 16, 28, 30) or mp_physics in _NAMED_MP_SCHEMES_BY_ID:
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
    if mp_physics in (9, 18, 24):
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
    if mp_physics == 50:
        # P3 one-category.  FIVE slots, not mp=6/8's seven: the driver arm
        # ``CASE (P3_1CATEGORY)`` binds RAINNC/RAINNCV/SR/SNOWNC/SNOWNCV and
        # NO graupel argument (module_microphysics_driver.F:1590-1595),
        # because P3 has a single ice category whose rime fraction spans
        # what other schemes split into snow, graupel and hail.  Listing
        # graupel here would allocate a canonical accumulator that stayed
        # zero forever and let output claim a graupel field P3 never has.
        return (("rainnc", "mp_rainnc"),
                ("rainncv", "mp_rainncv"),
                ("sr", "mp_sr"),
                ("snownc", "mp_snownc"),
                ("snowncv", "mp_snowncv"))
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
#: lowers it.  lane/282-bw-physics (b28c20509) doubled it to queue a second
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


# These constants mirror the kernel geometry for runtime-free memory pricing.
SHWS_SLOTS = 50
SHINHONG_BLOCK = 32
SHINHONG_TILE_BLOCKS_PER_SM = 16


def shinhong_workspace_floats(nz: int, columns: int) -> int:
    """Floats for whole blocks with 1-based levels and a top sentinel."""
    blocks = (int(columns) + SHINHONG_BLOCK - 1) // SHINHONG_BLOCK
    return blocks * SHWS_SLOTS * (int(nz) + 2) * SHINHONG_BLOCK


__all__ = [
    "spec_zone_ring_save_slots", "spec_zone_ring_slices",
    "MYJ_PBL_STATE_3D", "MYJ_SFCLAY_FIELDS_2D", "MYJ_SFCLAY_INOUT",
    "UWPBL_STATE_FULL", "UWPBL_STATE_2D", "UWPBL_DIAGNOSTICS_FULL",
    "UWPBL_DIAGNOSTICS_2D", "UWPBL_HELD_3D", "UWPBL_MASS_OUTPUTS",
    "UWPBL_SURFACE_OUTPUTS", "UWPBL_BLOCK", "UWPBL_R8_SLOTS_PER_LEVEL",
    "UWPBL_I4_SLOTS_PER_LEVEL", "UWPBL_WORKSPACE_BUDGET_BYTES",
    "uwpbl_workspace_slots", "uwpbl_chunk_columns", "uwpbl_workspace_bytes",
    "MYJ_SFCLAY_OUTPUTS",
    "MYNN_PBL_DIAGNOSTICS_2D", "MYNN_PBL_DIAGNOSTICS_INT_2D",
    "MYNN_PBL_STATE_3D", "MYNN_SURFACE_OUTPUTS",
    "PBL_RQI_MICROPHYSICS", "PHYSICS_SLOT_DISPATCH", "REFL_10CM_MICROPHYSICS",
    "UnroutedPhysicsSelectorError", "resolve_physics_slot",
    "SFCLAY_OUTPUTS",
    "SHWS_SLOTS", "SHINHONG_BLOCK", "SHINHONG_TILE_BLOCKS_PER_SM",
    "shinhong_workspace_floats",
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
PBL_ADVECTED_SCALARS = {901: (("tke_edmf", 0.2),)}
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
        5: "_run_sfclay",     # MYNN surface layer
        91: "_run_sfclay",    # classic MM5
    },
    "sf_ocean_physics": {0: None, 1: "_run_surface_ocean", 2: "_run_surface_ocean"},
    "sf_surface_physics": {
        0: None,
        2: "_run_noah",       # Noah LSM
        3: "_run_ruc",        # RUC LSM
        4: "_run_noahmp",     # Noah-MP LSM
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
        5: "_run_mynn_pbl",   # MYNN EDMF
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
        # UW moist turbulence (WRF v4.7.1 module_bl_camuwpbl_driver.F,
        # CAMUWPBLSCHEME).  Like MYJ it performs its own implicit
        # diffusion and returns finished tendencies; unlike every other
        # row it computes in binary64 (CAM's real(r8)) and reads the
        # radiation step's held RTHRATENLW and CLDFRA.
        UW_PBL_SCHEME: "_run_uwpbl",
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
#: ``SFCDIAGS`` (module_surface_driver.F:2983-2998).  RUC instead calls
#: ``SFCDIAGS_RUCLSM`` in :func:`ruc_lsm_step`; Noah-MP unconditionally
#: selects its water/ice flux diagnostic or its own vegetation/bare-ground
#: 2-m fields in :func:`noahmp_lsm_step`.  This is a per-scheme decision, not
#: a property of "an LSM ran", and all three post-LSM paths overwrite the
#: surface layer's earlier T2/Q2/TH2 exactly where WRF does.
LAND_SURFACE_SFCDIAGS_SCHEMES = frozenset({2})


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
CUMULUS_STATE_LAYOUTS = {5: {
    "g3_apr": ((8, "ny", "nx"), "f4", None),
    "g3_xf_ens": (("ensdim", "ny", "nx"), "f4", None),
    "g3_pr_ens": (("ensdim", "ny", "nx"), "f4", None),
    "g3_lev": ((9, "nz", "ny", "nx"), "f4", None),
    "g3_sfc": ((7, "ny", "nx"), "f4", None),
    "g3_isfc": ((4, "ny", "nx"), "i4", None),
    **{name: (("nz", "ny", "nx"), "f4", "cu_diag") for name in
       ("gd_cloud_a", "qc_cu", "gd_cloud2_a", "qi_cu", "gd_cldfr")},
    "raincv_a": (("ny", "nx"), "f4", "cu_diag"),
    "raincv_b": (("ny", "nx"), "f4", "cu_diag"),
    "g3_diag_clock": ((4,), "i4", "cu_diag"),
}}
CUMULUS_HISTORY_LAYOUT = {
    "gd_cloud": ("g3_lev", 7), "gd_cloud2": ("g3_lev", 8),
    **{name: (name, None) for name in
       ("gd_cloud_a", "qc_cu", "gd_cloud2_a", "qi_cu", "gd_cldfr", "raincv_a", "raincv_b")},
}

def cumulus_state_layout(cfg):
    return {name: (tuple(getattr(cfg, dim) if isinstance(dim,str) else dim for dim in dims), dtype)
            for name,(dims,dtype,flag) in CUMULUS_STATE_LAYOUTS.get(cfg.cu_physics,{}).items()
            if flag is None or getattr(cfg,flag)}


def solar_split_history_expected(cfg):
    """Schemes with a native surface split, or the model's SW interpolation."""
    from woof.config import radiation_scheme_ids
    shortwave = radiation_scheme_ids(cfg)[1]
    return bool(shortwave and (shortwave in (4, 5) or cfg.swint_opt == 1))

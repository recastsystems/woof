from __future__ import annotations
from dataclasses import field

from typing import NamedTuple
import math
import tomllib
from dataclasses import KW_ONLY, dataclass, fields, field as _config_field
from pathlib import Path
from typing import ClassVar
from woof.surface_ocean import ocean_namelist_defaults
from woof.sfire_config import FireRunFields, validate_fire_config

from woof.physics_compat import (
    RRTMG_VARIANT_LEGACY,
    RRTMG_VARIANT_RTE_RRTMGP,
    WRF_RRTMG_COMPATIBILITY_TOKENS,
    WRF_RRTMG_LEGACY,
    WRF_RRTMG_SUBSTITUTION_TOKENS,
    require_ready_wrf_physics,
    validate_resolved_physics_vertical_levels,
)
from collections.abc import Mapping as _Mapping
from importlib import import_module as _import_module

# The closure's two cross-layer constants.  Deliberately NOT
# woof.verify.sase_ref: that tree is developer verification and the
# standalone CPU preprocessing distribution omits it.
from woof.core import sase_limits as _sase_limits
from woof import correct_mode as _correct_mode
from woof.core.wdm6_constants import (WDM_HAIL_SCHEMES, WDM_SCHEMES,
                                       WDM_SCHEME_NAMES)

DEFAULT_COLUMN_CHUNK = 3125


class ConfigurationRefused(ValueError):
    """An explicit validator refuses settings by naming their breakage.

    Programming errors remain their original exception type. Callers catching
    ValueError remain compatible; exact replay can distinguish a refusal.
    """


class UnsupportedConfiguration(ConfigurationRefused, NotImplementedError):
    """A named unsupported configuration, also caught by existing callers."""



#: THE SURFACE-RADIATION CARRIER POLICY vocabulary.  Defined in the CONFIG
#: module rather than beside the contract it governs
#: (woof.core.radiation_carriers), because the config-load refusal has to
#: be reachable from the standalone RW-WPS preprocessing wheel, which
#: carries woof.config and does not carry woof.core.  The contract
#: imports these names from here; the arrow never points the other way.
SURFACE_RADIATION_POLICY_REQUIRED = "required"
SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO = "wrf_compat_zero"
SURFACE_RADIATION_POLICIES = (SURFACE_RADIATION_POLICY_REQUIRED,
                              SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO)


def validate_surface_radiation_policy(policy: str) -> str:
    """Return the policy, or refuse a value that is not one of the two."""
    if policy not in SURFACE_RADIATION_POLICIES:
        raise ConfigurationRefused(
            f"surface_radiation_policy={policy!r} is not a policy.  The "
            f"choices are {SURFACE_RADIATION_POLICY_REQUIRED!r} (the "
            "default: every carrier a land-surface scheme reads must have "
            "a producer, checked immediately before the scheme consumes "
            f"it) and {SURFACE_RADIATION_POLICY_WRF_COMPAT_ZERO!r} (the "
            "declared escape: unsourced carriers are consumed at their "
            "allocation fill, which reproduces pre-1.9 behaviour and is an "
            "experimental forcing rather than a valid real-case "
            "configuration).")
    return policy


@dataclass(frozen=True)
class RunConfig:
    def __post_init__(self):
        if self._legacy_h_sca is None:
            object.__setattr__(self, "_legacy_h_sca", self.h_sca_adv_order is None)
        if self.h_sca_adv_order is None:
            object.__setattr__(self, "h_sca_adv_order", 2)
        if self.windturbines_spec != "none":
            if self.wind_farms or self.wind_farm_scheme not in ("off", "fitch_legacy"):
                raise ValueError("windturbines_spec is the WRF V3.5 scheme and cannot be combined with windfarm_opt or wind_farms")
            object.__setattr__(self, "wind_farm_scheme", "fitch_legacy")
        # TOML may spell waves_options as an inline table. Keep the frozen
        # config hashable and give JSON and TOML the same restart identity.
        if isinstance(self.waves_options, dict):
            import json
            object.__setattr__(self, "waves_options", json.dumps(self.waves_options, sort_keys=True))
        if self.wind_farms:
            if self.wind_farms == "off":
                object.__setattr__(self, "wind_farm_scheme", "off")
                object.__setattr__(self, "wind_farm_table", "")
            else:
                object.__setattr__(self, "wind_farm_scheme", "mav" if self.wind_farm_scheme == "mav" else "fitch")
                object.__setattr__(self, "wind_farm_table", self.wind_farms)
        # The PWP reference profiles (development branch) arrive as lists
        # from TOML, the namelist importer and callers; store tuples so the
        # frozen configuration stays hashable.
        for name in ("ocean_z", "ocean_t", "ocean_s", "press_levels", "z_levels"):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        # Resolve named microphysics before all consumers inspect its id.
        # Keep this in the same initializer as the chemistry shortcut.
        if not isinstance(self.mp_physics, int) or isinstance(self.mp_physics, bool):
            from woof.microphysics_schemes import resolve_mp_physics
            object.__setattr__(self, "mp_physics", resolve_mp_physics(self.mp_physics))
        # Resolve the generation after named microphysics so partial
        # number mixing has the same meaning through either run door.
        if self.bl_pbl_physics == 5 and self.bl_mynn_version == "wrf_461":
            changed = any(getattr(self, k) != v for k, v in MYNN_PBL_OPTION_IDENTITY.items())
            partial = self.bl_mynn_mixscalars == 1 and self.mp_physics in MYNN_QN_FLAG_SPECIES and self.mp_physics not in MYNN_QN_FAMILY_SCHEMES and bool(MYNN_QN_FLAG_SPECIES[self.mp_physics])
            if changed or self.bl_mynn_mixlength == 0 or self.tke_budget or partial:
                object.__setattr__(self, "bl_mynn_version", "wrf_471")
        # WRF resets its default shallow mass-flux switch outside QNSE.
        if self.bl_pbl_physics != 4 and self.mfshconv == 1:
            object.__setattr__(self, "mfshconv", 0)
            object.__setattr__(self, "_mfshconv_reset", True)
        # Resolve this public shortcut before arena, cache and restart
        # identities inspect chem_sets. It is inert for existing configs.
        if not isinstance(self.chem_sets, str) or not isinstance(self.fire_smoke, bool):
            return
        selected = tuple(name.strip() for name in self.chem_sets.split(","))
        profiles = ("sfire_smoke", "sfire_smoke_mixed")
        if self.fire_smoke or any(name in selected for name in profiles):
            from woof.chem_table import chem_names
            names = chem_names(self.chem_sets)
            if not any(name in names for name in profiles):
                names = (*names, "sfire_smoke")
            object.__setattr__(self, "chem_sets", ",".join(names))
            object.__setattr__(self, "fire_smoke", True)

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    ztop: float
    dt: float
    run_seconds: float
    # WRF's model-clock timestep.  Normally identical to ``dt``; the
    # the reference case compatibility integrator sets this to 60 s while advancing
    # eight uniform 7.5 s dynamics steps.  Clock-defined coefficients such
    # as Davies relaxation and sixth-order diffusion use this value.
    clock_dt: float = 0.0
    p_surf: float = 1.0e5
    time_step_sound: int = 4
    epssm: float = 0.1
    smdiv: float = 0.1
    khdif: float = 0.0
    kvdif: float = 0.0
    damp_opt: int = 0
    zdamp: float = 5000.0
    dampcoef: float = 0.2
    output_interval_s: float = 300.0
    case: str = ""
    # --- Phase 2 options (defaults preserve Phase 1 behavior exactly) ---
    hybrid_opt: int = 0          # 0/1 = B(eta)=eta (Phase 1); 2 = WRF cubic-B hybrid
    etac: float = 0.2
    moist: bool = False
    # 0 off, 1 Kessler, 2 Purdue Lin, 6 WSM6, 8 Thompson, 10 Morrison, 18 NSSL,
    # 28 Thompson aerosol-aware (Registry/Registry.EM_COMMON:3036),
    # 38 Thompson hail-aware (Registry/Registry.EM_COMMON:3037).
    # Named WOOF schemes are given by name and resolve to their
    # WOOF-side id (woof.microphysics_schemes) in __post_init__.
    mp_physics: int = 0
    moist_adv_opt: int = 1       # PD limiter on when moist
    # Non-Chem EM Registry.EM's eight dimensionless passive tracers.
    tracer_opt: int = _config_field(default=0, kw_only=True)
    tracer_adv_opt: int = _config_field(default=1, kw_only=True)
    tracer_mix2_off: bool = _config_field(default=False, kw_only=True)
    tracer_mix6_off: bool = _config_field(default=False, kw_only=True)
    tracer_pblmix: int = _config_field(default=1, kw_only=True)
    # Native EM trajectories, with the point namelist sealed into the run.
    traj_opt: int = _config_field(default=0, kw_only=True)
    dm_has_traj: bool = _config_field(default=False, kw_only=True)
    num_traj: int = _config_field(default=1000, kw_only=True)
    trajectory_input: str = _config_field(default="", kw_only=True)
    # WRF no_mp_heating (Registry.EM_COMMON:2630, default 0): 1 disables
    # microphysics latent heating entirely -- moist_physics_finish_em then
    # leaves theta untouched and zeroes h_diabatic
    # (module_big_step_utilities_em.F:5770-5782).
    no_mp_heating: int = 0
    # WRF mp_tend_lim (Registry.EM_COMMON:2642, default 10. K/s): clamp on
    # the per-step microphysics theta increment, |mpten| <= mp_tend_lim*dt
    # (module_big_step_utilities_em.F:5706-5707).
    mp_tend_lim: float = 10.0
    diff_6th_opt: int = 0
    diff_6th_factor: float = 0.12
    # WRF diff_6th_slopeopt / diff_6th_thresh (Registry.EM_COMMON:2858-2859,
    # defaults 0 / 0.10; the reference namelist sets slopeopt=1): >= 1
    # tapers every 6th-order face flux by slopedamp = MAX(1 -
    # dzmax/(thresh*9.81*dx), 0) with dzmax the BASE-state (phb) face
    # geopotential jump at the field's own level, msf-scaled per face
    # (sixth_order_diffusion, module_big_step_utilities_em.F:6487-6501 x /
    # :6569-6583 y).  Default 0 keeps the frozen untapered operator
    # bitwise; flat (1-D phb) states are untapered either way.
    diff_6th_slopeopt: int = 0
    diff_6th_thresh: float = 0.10
    km_opt: int = 1
    c_s: float = 0.25
    w_damping: int = 0
    open_x: bool = False         # False = periodic (Phase 1 default)
    open_y: bool = False
    # --- Phase 3 (Task 8): WRF real-data base state and lateral BCs. ---
    base_temp: float = 290.0
    specified: bool = False
    spec_bdy_width: int = 5
    spec_zone: int = 1
    relax_zone: int = 4
    spec_exp: float = 0.0
    # WRF external-mode divergence damping (module_small_step_em.F mudf;
    # namelist emdiv, WRF Registry default 0.01).  woof defaults to 0.0 so
    # the frozen Phase 1 acoustic path is bitwise unchanged; open-boundary
    # cases set the WRF value (em_quarter_ss: 0.01) -- the filter damps the
    # column-integrated (external) mode that open boundaries excite.
    emdiv: float = 0.0
    # WRF horizontal scalar/W transport and rhs_ph selector (native default 5).
    # Omission retains the historical idealized phi=2/transport=5 path.
    # An explicit order selects that native order for phi and transport.
    h_sca_adv_order: int | None = None
    _legacy_h_sca: bool | None = _config_field(default=None, kw_only=True, repr=False)
    terrain_opt: int = 0         # 0 = flat; 1 = bell hill
    hill_height: float = 100.0
    hill_halfwidth: float = 10000.0
    # --- Phase 3 (Task 3): map projection.  0 = idealized/none (map factors
    # 1, Coriolis parameters 0 â€” the bitwise Phase 2 state); nonzero uses
    # the WRF convention: 1 = Lambert conformal, 2 = polar stereographic,
    # 3 = Mercator (grids built by woof.static.projection).  The dycore
    # consumes the projection only through the DomainState fields
    # msft/msfu/msfv/f/e/sina/cosa (set_map_coriolis).
    map_proj: int = 0
    # --- Phase 3 (Task 9): WRF MM5 surface layer.  0 preserves the
    # pre-physics path; 1 is revised SFCLAY and 91 is classic MM5 SFCLAY.
    sf_sfclay_physics: int = 0
    # --- Phase 3 (Task 10): land surface model.  0 = none (Phase 1/2
    # idealized default, bitwise unchanged), 2 = Noah LSM (WRF
    # sf_surface_physics numbering; kernels/noah.cu + woof/core/noah.py).
    # Wired into the step loop by the physics driver (Task 12).
    sf_surface_physics: int = 0
    # --- Phase 3 (Task 11): planetary boundary layer physics.  The physics
    # driver arrives in Task 12; this option selects its PBL registry entry.
    # 0 = none, 1 = YSU, 5 = MYNN, 11 = Shin-Hong, 900 = SASE.
    bl_pbl_physics: int = 0
    # WRF v4.6.1 Registry.EM_COMMON default; consumes radiative heating.
    ysu_topdown_pblmix: int = 1
    # --- Phase 4 (Task 1): non-timesplit radiation and cumulus slots.
    # ra_physics=90 is the Phase-3 analytic clear-sky proxy; 4 selects
    # RTE+RRTMGP.  cu_physics=1 is Kain-Fritsch; cu_physics=3 is
    # Grell-Freitas (scale-aware; the sig=(1-frh)^2 taper is the scheme's
    # own, so per-domain admission carries no dx gate).
    ra_physics: int = 0
    cu_physics: int = 0
    shcu_physics: int = _config_field(default=0, kw_only=True)
    # WRF Registry.EM_COMMON:2447-2449, prescribed CAM aerosol modes (#/kg).
    accum_mode: float = _config_field(default=1000.0e6, kw_only=True)
    aitken_mode: float = _config_field(default=300.0e6, kw_only=True)
    coarse_mode: float = _config_field(default=0.2e6, kw_only=True)
    radt_minutes: float = 12.0
    cudt_minutes: float = 5.0
    # Phase-3 compatibility spelling.  A positive radt continues to
    # override radt_minutes; new configurations use radt_minutes.  bldt
    # remains the surface-layer/LSM/PBL interval.  Zero means every step.
    radt: float = 0.0
    bldt: float = 0.0
    # WRF hypsometric_opt (Registry.EM_COMMON default 2): selects the
    # hydrostatic al/p diagnostic form in calc_p_rho_phi
    # (dyn_em/module_big_step_utilities_em.F:1025-1052) and the matching
    # real-init geopotential construction (module_initialize_real.F:3811,
    # :3962, :3997).  1 = the currently validated d(phi)/d(eta) form
    # (woof's frozen behavior, bitwise-inert default); 2 = the reference
    # WRF run's log-pressure form al = dphi/phm/LOG(pfd/pfu) - alb.
    # Idealized WRF cases always force 1 (share/input_wrf.F:1038); woof's
    # idealized initializers likewise stay on the opt-1 recurrences.
    hypsometric_opt: int = 1
    # --- Phase 4 (Task 8): restart-file write cadence in seconds (WRF
    # restart_interval, minutes there).  0 disables writing.  A positive
    # interval must be a whole multiple of dt (validated by the case run
    # schedule); resuming FROM a restart file is the `woof run --restart`
    # flag, not a config key, so a resumed run keeps its config identical
    # to the writer's (the restart header enforces that identity).
    restart_interval_s: float = 0.0
    # --- Phase 5 (Task 1): multi-domain experiment schema, two INERT
    # defaults.  WRF &bdy_control ``nested`` (the bundle namelist.input
    # runs specified=T,F,F,F / nested=F,T,T,T): False on every legacy
    # single-domain path; the experiment loader (woof/experiment.py)
    # sets it True on child domains and validates it mutually exclusive
    # with ``specified``.  ``grid_id`` (WRF &domains grid_id) is 1 on
    # every legacy path; per-domain ids come from the experiment loader.
    nested: bool = False
    grid_id: int = 1
    # WRF v4.6.1 Registry.EM_COMMON:2898 defaults top_lid=.false. (open top,
    # live calc_coef_w/advance_w top row).  woof defaults to the RIGID lid:
    # the 2026-07-18 production probes showed the open-top branch NaNs a
    # the reference case 2-dom run within 15 sim-min (iso-opentop_only_2dom) while the
    # rigid branch is byte-identical to the verified pre-flag model
    # (rigid-check vs pacc-base).  Open top remains available for the
    # WRF-fidelity stabilization campaign; flip only with a full-leg
    # stability + falsification receipt.
    top_lid: bool = True
    # WRF applies calc_cq whenever its moist Registry carries water vapor,
    # including passive vapor with microphysics off. The driver bypasses
    # dry states. The former prebound-launch guard is retired by the
    # MP0/MP1/MP6 device regression and stable moist forecast replay.
    # Explicit False remains a verification counterfactual.
    moist_cq: bool = True
    # WRF v4.6.1 Registry.EM_COMMON:2663-2666 default: Morrison dense ice
    # is hail (1); 0 retains the explicit graupel branch.  Appended to
    # preserve positional RunConfig compatibility.  The scheme selects
    # AG/BG/RHOG in module_mp_morr_two_moment.F:337-382.
    morr_rimed_ice: int = 1
    # WRF v4.6.1 WSM6 ``hail_opt``.  WSM6 defaults to graupel; setting 1
    # selects the denser/faster hail intercept, density and fall-speed
    # coefficients in module_mp_wsm6.  Kept scheme-specific so a WSM6
    # trajectory cannot be restarted under a different rimed-ice mode.
    wsm6_hail_opt: int = 0
    # WRF v4.6.1 ``ccnty`` (Registry.EM_COMMON:2429, scalar default 2):
    # the NTU (mp_physics=56) aerosol background AERO_CONST builds its
    # CCN spectrum and cut-off tables from -- 1 marine, 2 clean
    # continental, 3 average continental, 4 urban (Whitby 1978).
    ntu_ccnty: int = _config_field(default=2, kw_only=True)
    # WRF-native split radiation selection.  ``-1/-1`` preserves the
    # historical woof ``ra_physics`` aggregate exactly; new configurations
    # set BOTH values explicitly and leave ra_physics=0, or restate the
    # aggregate on both streams, which resolves to the same pair.  WRF
    # commonly pairs
    # RRTM LW (1) and Dudhia SW (1), while the existing RTE+RRTMGP adapter is
    # retained as the coupled 4/4 option.
    ra_lw_physics: int = -1
    ra_sw_physics: int = -1
    # WRF module_ra_sw controls.  Both defaults are Registry.EM_COMMON's
    # values and are trajectory-bound even when Dudhia is inactive so a
    # future scheme switch cannot inherit an implicit policy.
    icloud: int = 1
    swrad_scat: float = 1.0
    # Appended port-lane schema.  ``none`` means option 4 was selected as
    # woof's native RTE+RRTMGP algorithm.  The named token is emitted only by
    # the WRF namelist importer when WRF RRTMG 4/4 is intentionally mapped to
    # that modern solver.  It is part of restart/config identity, so the
    # substitution cannot disappear between launch and resume.
    wrf_rrtmg_compatibility: str = "none"
    # The soil geometry REQUESTED, in WRF's namelist spelling.  It is not the
    # count anything allocates from: soil_layer_count(cfg) resolves that from
    # sf_surface_physics and refuses a request the selected scheme does not
    # define, so this default is Noah's four only in the sense that WRF's
    # namelist has a default too.  A scheme with more than one geometry (RUC:
    # six or nine) therefore has to state which, rather than inheriting a
    # number that belongs to a different scheme -- which is exactly the bug
    # this stopped being.  Frozen at 4 by tests/data/config_freeze_golden.json.
    num_soil_layers: int = 4
    # Which implementation serves a resolved 4/4 RRTMG radiation request.
    # The default keeps every existing configuration on woof's modern
    # RTE+RRTMGP adapter, byte-identically.  "rrtmg_legacy" selects the
    # exact port of WRF v4.6.1's bundled RRTMG and FAILS CLOSED at physics
    # setup until its LW/SW compute kernels land -- it never silently
    # falls back to RTE+RRTMGP.  Trajectory-bound through config identity.
    ra_rrtmg_variant: str = "rte-rrtmgp"
    # GPUWM-specific one-way nest microphysics transition.  The default
    # ("same-scheme-only") is the unset value: a same-scheme edge forces as
    # WRF does, and a mixed edge resolves to the versioned closure the pair
    # takes (woof.core.microphysics_transition), recorded in the coupler
    # receipt as requested versus effective policy.  Naming one of the two
    # mixed ids pins it; naming the id of another edge is refused.
    nest_microphysics_transition: str = "same-scheme-only"
    # WRF MM5 surface-layer options, both scalar in Registry.EM_COMMON
    # (dimension column 1, defaults 0).  The surface-layer kernel already
    # implements every branch: isftcflx selects the Garratt (1) or
    # Donelan (2) water-point heat/moisture roughness in
    # woof/core/kernels/sfclay.cu:356/:371/:429, and iz0tlnd selects the
    # Chen-Zhang (1) or fixed-0.1 (2) land thermal roughness at :399-402,
    # with woof/core/sfclay.py enforcing {0,1,2} for each.  Only the
    # configuration path was missing, so these expose an implemented
    # capability rather than adding one.  Appended to preserve positional
    # RunConfig compatibility.
    isftcflx: int = 0
    iz0tlnd: int = 0
    # Land-surface options whose branches exist in Noah AND RUC.
    # ``usemonalb`` leaves real.exe's monthly ALBEDO12M background albedo
    # in place instead of LANDUSE.TBL's seasonal row (WRF landuse_init,
    # module_physics_init.F:1611; woof/core/landuse.py initialize_landuse)
    # and selects it inside Noah (woof/core/kernels/noah.cu:1112, :1117);
    # ``rdlai2d`` keeps the monthly LAI12M field interpolated to the start
    # date instead of the VEGPARM table value: Noah at noah.cu:1036, :1111,
    # :1116, RUC's soilvegin at woof/core/kernels/ruc.cu:171, :178, :292
    # and the fused driver (tools/ruc_fused/driver_body.cuh).  Operational
    # HRRR runs both true (hrrr_wrf.nl:148-149).  ``opt_thcnd`` selects the
    # Johansen (1) or McCumber-Pielke (2) soil thermal conductivity at
    # noah.cu:181 and exists in Noah only.  Each default reproduces the
    # previously pinned launcher value, so frozen trajectories are bitwise
    # unchanged.  Appended to preserve positional RunConfig compatibility.
    usemonalb: bool = False
    rdlai2d: bool = False
    opt_thcnd: int = 1
    # MYNN EDMF options. Mixing length accepts 1 and 2; scalar plume mixing
    # accepts 0 and 1. The remaining knobs retain the WRF configuration
    # recorded by MYNN_PBL_OPTION_IDENTITY. Every choice reaches the driver
    # by name and binds restart identity.
    bl_mynn_closure: float = 2.6
    bl_mynn_cloudpdf: int = 2
    bl_mynn_mixlength: int = 1
    bl_mynn_edmf: int = 1
    bl_mynn_edmf_mom: int = 1
    bl_mynn_edmf_tke: int = 0
    # 0 is the Registry default of the MYNN generation this solver ports
    # (WRF v4.6.1 Registry.EM_COMMON:2479, ``rconfig integer
    # bl_mynn_mixscalars namelist,physics max_domains 0``); the V3.9 line
    # HRRR runs (gsd_41) has no such key and mixes its aerosols through
    # scalar_pblmix=1 instead.  WRF 4.7.x flipped the default to 1
    # (Registry.EM_COMMON:2482) with the MYNN-EDMF submodule, a generation
    # this solver does not carry; the namelist importer names that
    # divergence on every MYNN import that omits the key.  1 mixes WRF's
    # qn family (nc/ni/nwfa/nifa) when the microphysics carries all of it
    # (mp=28), is inert under a scheme WRF gives no number species (mp
    # 0/1/6: WRF's flag-gated skips), and is refused under a scheme WRF
    # would mix in part (Thompson's qni and the like, MYNN_QN_FLAG_SPECIES)
    # because this port's solve runs the whole family or nothing
    # (``mynn_mixscalars_active``).
    bl_mynn_mixscalars: int = 0
    bl_mynn_cloudmix: int = 1
    bl_mynn_mixqt: int = 0
    bl_mynn_output: int = 0
    bl_mynn_tkeadvect: bool = False
    icloud_bl: int = 1
    # Noah-MP option identity.  Same contract as the MYNN block above: each
    # default is the value the ported column was validated at against the
    # byte-unmodified module_sf_noahmplsm.F, and each is the ONLY value woof
    # honours.  Read by woof/core/physics.py:_run_noahmp, which forwards
    # them to woof/core/noahmp_runtime.py so the refusal for an unported
    # branch comes from the routine that would have had to implement it.
    #
    # These are WRF's ``NoahMP_OPTIONS`` namelist names verbatim.  The
    # enumeration below is accurate about what "validated" means: every value
    # is the WRF Registry default AND the value the four whole-column oracle
    # fixtures were generated at, and no other value of any of them has been
    # measured -- not approximated, not measured.
    dveg: int = 4
    opt_crs: int = 1
    opt_btr: int = 1
    opt_run: int = 3
    opt_sfc: int = 1
    opt_frz: int = 1
    opt_inf: int = 1
    opt_rad: int = 3
    opt_alb: int = 2
    snicar_snowshape_opt: int = _config_field(default=3, kw_only=True)
    snicar_snowbc_intmix: bool = _config_field(default=True, kw_only=True)
    snicar_snowdust_intmix: bool = _config_field(default=False, kw_only=True)
    snicar_bc_ppb: float = _config_field(default=0.0, kw_only=True)
    snicar_dust_ppm: float = _config_field(default=0.0, kw_only=True)
    snicar_ice_density: float = _config_field(default=900.0, kw_only=True)
    snicar_bubble_radius_um: float = _config_field(default=1000.0, kw_only=True)
    opt_snf: int = 1
    opt_tbot: int = 2
    opt_stc: int = 1
    opt_gla: int = 1
    opt_rsf: int = 1
    opt_soil: int = 1
    opt_pedo: int = 1
    opt_crop: int = 0
    opt_irr: int = 0
    opt_irrm: int = 0
    opt_infdv: int = 0
    opt_tdrn: int = 0
    wtddt: float = _config_field(default=30.0, kw_only=True)  # WRF physics minutes
    soiltstep: float = 0.0
    noahmp_output: int = 1
    noahmp_acc_dt: float = 0.0
    # RUC's weighted category parameters, Registry.EM_COMMON:2535-2537.
    # WRF defaults both switches to zero. At one the runtime consumes the
    # full LANDUSEF / SOILCTOP category fractions, including irrigation.
    # flag_sm_adj stays pinned by RUC_OPTION_IDENTITY_EVIDENCE.  spp_lsm
    # (Registry/registry.stoch:241) is validated by validate_spp_config.
    mosaic_lu: int = 0
    mosaic_soil: int = 0
    flag_sm_adj: int = 0
    spp_lsm: int = 0
    # WRF 4.7.1 NWP history windows. The GPU driver samples first-RK UH,
    # holds phy_prep pressure/density and updates every native NWP field.
    nwp_diagnostics: int = 0
    # Lane-K option exposure.  These fields are appended so positional
    # RunConfig construction keeps its historical order, and every default is
    # the behavior the assembled tree had before the option became settable.
    #
    # isfflx is WRF's surface heat/moisture-flux gate.  The ported MM5 and
    # MYNN surface-layer paths implement 0/1; WRF's prescribed-flux value 2
    # also needs the diffusion-side forcing contract and is not admitted.
    isfflx: int = 1
    # Legacy RRTMG's two implemented ozone sources: wrapper O3DATA (0) and
    # the CAM climatology / parent-routed o3rad field (2).  RTE+RRTMGP keeps
    # its established climatological-ozone identity at 2.
    o3input: int = 2
    # WRF's module_physics_init gate around the microphysics effective-radius
    # scheme table.  Only legacy RRTMG owns that wrapper branch.
    use_mp_re: int = 1
    # WRF &physics seaice_albedo_default (Registry default 0.65): RUC's live
    # CASE(RUCLSMSCHEME) sea-ice albedo replacement, and the sea-ice albedo
    # Noah's seaice_noah sets under seaice_albedo_opt = 0.
    seaice_albedo_default: float = 0.65
    # Noah LSMINIT snow-albedo source.  True preserves the supplied geogrid
    # SNOALB field; false transcribes WRF's MAXALB(IVGTYP)*0.01 branch.
    rdmaxalb: bool = True
    # --- km_opt=3 (3-D Smagorinsky) LES closure knobs, WRF's own names.
    # Appended last to preserve positional construction (the config-freeze
    # discipline); each default is WRF's Registry default, so every frozen
    # km_opt=1/4 trajectory is unchanged.
    #
    # WRF mix_isotropic (Registry.EM_COMMON:2896, "0=anistropic,
    # 1=isotropic", default 0): selects smag_km's mixing-length branch for
    # km_opt=3 -- separate horizontal sqrt(dx*dy) / vertical dz lengths at
    # 0, the single (dx*dy*dz)^(1/3) length at 1.  The em_les reference
    # namelist sets 1.
    mix_isotropic: int = 0
    # WRF mix_upper_bound (Registry.EM_COMMON:2897, default 0.1): the
    # non-dimensional K cap K <= mix_upper_bound * len^2 / dt applied per
    # direction inside smag_km (module_diffusion_em.F:1890-1927).
    mix_upper_bound: float = 0.1
    # WRF tke_heat_flux / tke_drag_coefficient (Registry.EM_COMMON:
    # 2900-2901, defaults 0., per-domain): the prescribed-flux lower
    # boundary consumed by vertical_diffusion_2's SELECT CASE(isfflx)
    # blocks under diff_opt=2 with the PBL off -- constant kinematic heat
    # flux (K m s-1) for isfflx in {0,2}, constant drag coefficient for
    # isfflx=0 (module_diffusion_em.F:4155-4200, :4286-4305).
    tke_heat_flux: float = 0.0
    tke_drag_coefficient: float = 0.0
    # WRF c_k (Registry.EM_COMMON:2863, default 0.15): the TKE-closure
    # exchange-coefficient constant K = c_k * sqrt(e) * l for km_opt=2.
    # The em_les reference namelist sets 0.10 (README.les: recommended
    # with LES).  Appended last per the config-freeze discipline.
    c_k: float = 0.15
    # WRF tke_upper_bound (Registry.EM_COMMON:2899, default 1000., per
    # domain): the ceiling in ``bound_tke`` (dyn_em/module_em.F:2490-2520),
    # which clamps the prognostic TKE into [0, tke_upper_bound] over the
    # whole mass grid after EVERY rk_update_scalar pass
    # (dyn_em/solve_em.F:2434-2440) -- on periodic domains too, not only
    # at lateral boundaries.
    tke_upper_bound: float = 1000.0
    # Report-only per-step TKE budget accumulation (km_opt=2): writes the
    # term-by-term decomposition into diagnostic scratch and reduces it on
    # device once per step.  Trajectory-inert by construction -- it reads
    # model state and writes only its own slots -- so it is a restart-
    # boundary-adjustable diagnostic toggle exactly like nwp_diagnostics
    # (woof/checkpoint_identity.py CONFIG_DIAGNOSTIC_FIELDS).
    tke_budget: int = 0
    # --- Aerosol-aware Thompson (mp_physics=28) aerosol-source selectors ---
    #
    # Both defaults are WRF's own Registry defaults, and that is essential
    # rather than cosmetic: woof/physics_compat.py's
    # _SINGLE_DOMAIN_RUNTIME_SWITCHES rows are compared for EXACT equality by
    # the prepared-forecast runner, so a nonzero default here would silently
    # change every shipped profile.
    #
    # aer_init_opt -- Registry/Registry.EM_COMMON:2656, default 0.  WRF
    # declares it ``derived`` (real.exe sets it from use_aero_icbc /
    # use_rap_aero_icbc), not ``namelist``: 0 = no IC/BC aerosol, 1 = climo,
    # 2 = first guess.  ArWen exposes it as a settable field ONLY so that a
    # request can be checked against the selected source. ArWen ports
    # the CLIMATOLOGY branch (1) and takes it BY DEFAULT on
    # real data through mp28_aerosol_source='auto'; this field stays at
    # WRF's Registry default because the prepared-forecast runner
    # compares the switch rows for exact equality.  2 (first guess) has
    # requires an explicit analyzed-aerosol selection.
    aer_init_opt: int = 0
    # wif_input_opt -- Registry/registry.new3d_wif:17, default 0.
    # 0 = do not process the Water/Ice Friendly aerosol input from metgrid;
    # 1 = use_wif_input; 2 = use_wif_input_bc, which additionally allocates
    # the black-carbon scalar qnbca.
    # 1 is ported as the climatology pair with aer_init_opt=1 and is the
    # default real-data route via mp28_aerosol_source='auto'; 2 wants the
    # nbca species, which does not exist in the port, and fails closed.
    wif_input_opt: int = 0
    # --- SASE closure knobs (bl_pbl_physics = SASE_PBL_SCHEME only).
    # Appended last, per the config-freeze discipline in
    # tests/test_config_freeze.py.  Every one is FAIL-CLOSED on its
    # NON-DEFAULT value: the default is admitted under every PBL scheme,
    # so no existing configuration moves, and the non-default is refused
    # anywhere it would name a seam that does not exist.
    #
    # Split subgrid-flux diagnostic (output-only, per domain).  True adds
    # four face-registered history fields recording the closure's own
    # vertical subgrid fluxes with the conditional-venting channel
    # separated from the K_v implicit-diffusion channel.  It reads no
    # prognostic and writes none, so the prognostic state is bitwise
    # identical either way; per domain because the cost scales with the
    # grid (four planes per frame).
    sase_flux_diag: bool = False
    # Moist-N2 substitution (physics selector, run-wide).  True is the
    # closure as built: the saturated Brunt-Vaisala N^2_m replaces the dry
    # N^2 at the stability lengths, the subgrid-energy buoyancy source and
    # the K_v/K_h stability suppression.  False consumes the dry N^2 at
    # all three points, isolating the diffusion channel from the venting
    # channel.  Not a per-domain override: every SASE domain of a tree
    # runs one variant, so an on/off pair compares two whole trees.  On a
    # tree that mixes SASE with another PBL scheme the [shared] value
    # reaches the SASE domains only (SASE_FAIL_CLOSED_DEFAULTS).
    sase_moist_n2: bool = True
    # Stable-limb dissipation decoupling (physics selector, run-wide).
    # True rides the Deardorff lambda << Delta dissipation coefficient
    # where the stability length binds the dissipation length, instead of
    # the neutral-wall value.  DEFAULT FALSE FOR A MEASURED REASON, not
    # merely for caution: with it True the closure's own registered
    # stable-boundary-layer calibration gate exits its observation band
    # (tests/test_sase.py::
    # test_jet_decoupling_stable_dissipation_exits_obs_band pins that
    # RED).  What is falsified is the PAIR of stable-limb coefficients,
    # which enter the stability ratio jointly; neither may be
    # re-registered or tuned alone.  Do not flip this default without
    # that joint re-registration.
    sase_stable_dissipation: bool = False
    # SASE S3-12 ADDITIVE e^{3/2} DISSIPATION CHANNEL.  True ADDS the
    # second, grid-scale member of Deardorff's length-dependent
    # coefficient (C_ED = 0.51) to whichever base the key above selects,
    # divided by a STATE-INDEPENDENT reference length.  Where the
    # stability length binds, HEAD's subgrid-energy equation is exactly
    # homogeneous in e and has no equilibrium amplitude at all -- below a
    # critical Richardson number of 0.1647 the energy grows exponentially
    # with nothing in the closure to stop it (measured on real data:
    # 1.62 m2/s2 at 13.1 km, still doubling every ~3 min at t+60 min).
    # This channel carries no e in its length, so it gives that limb a
    # finite, attracting fixed point.
    #
    # It is the OPPOSITE MOVE from the key above and composes with it:
    # sase_stable_dissipation LOWERS the coefficient (and is RED on the
    # closure's own stable-boundary-layer gate for exactly that reason),
    # this one ADDS to it, so dissipation is nowhere weaker than the
    # default path's.  Both registered calibration gates hold with it
    # True and the jet gate gains margin; the lake gate's margin
    # narrows from 28% to 9% and that is recorded at the fixture.
    # DEFAULT TRUE since the 2026-08-17 real-data confirmation leg
    # (fixed-means-default): the 08-01 operational configuration flown
    # through the shipped front door with this channel on.  The RED legs
    # that pin the historical un-channeled formulation set
    # additive_dissipation=False explicitly, exactly as the S3-6j legs
    # pin apply_drag=False (authority module docstring, S3-12 section).
    sase_additive_dissipation: bool = True
    # --- Horizontal-mixing diagnostic and the km_opt = 0 acknowledgement.
    #
    # Horizontal eddy-viscosity diagnostic (output-only, per domain).  True
    # adds the horizontal eddy viscosities the run's OWN mixing producer
    # used, under the name of that producer: XKMH/XKHH for the km_opt = 4
    # Smagorinsky operator, SASE_KMH/SASE_KHH for the SASE closure's
    # governed horizontal diffusivity.  Both are m2 s-1 on the mass grid
    # and mean the same thing, so a run that removes one producer and adds
    # the other can be MEASURED on the channel it swapped rather than
    # argued about.  Reads no prognostic and writes none, so a restart or
    # a prepared cache may see it change
    # (woof/checkpoint_identity.py CONFIG_DIAGNOSTIC_FIELDS).
    hmix_k_diag: bool = False
    # Expert acknowledgement admitting km_opt = 0 with a PBL scheme that
    # does not itself produce horizontal mixing -- i.e. a run with NO
    # horizontal mixing operator at all.  Must be set to the exact id
    # KM_OPT_ZERO_ACK; anything else (including True) is refused.  See
    # KM_OPT_ZERO_ACK for what it means and why the default refuses.
    km_opt_zero_acknowledgement: str = ""
    # --- LES-nest inflow turbulence seeding (P3, per-domain, TOML-only).
    #
    # CPM-style cell-blocked theta perturbation of the child's rolling
    # nest-boundary VALUE tables on the inflow-side relax-zone rows,
    # refreshed on the FORCE cadence (woof/core/inflow_perturbation.py;
    # every constant pinned in docs/superpowers/receipts/les/
    # INFLOW-GENERATOR-ACCEPTANCE-V2.md, registered before data).  An
    # ArWen-over-WRF extension (PROVENANCE.md D10): stock v4.6.1 ships
    # perturb_bdy (stoch-package boundary-tendency patterns), not a
    # cell-perturbation path, and has no namelist column for these, so a
    # config using them cannot round-trip to a namelist -- exactly the
    # per-domain isfflx situation.  Default OFF; the OFF trajectory is
    # gated byte-identical to a build without the mechanism.
    inflow_perturbation: bool = False
    # RNG seed; part of every draw's Philox key (deterministic,
    # counter-based -- same seed, same bytes, every card).
    inflow_perturbation_seed: int = 0
    # Multiplies the pinned Eckert-number amplitude.  0.0 skips every
    # table write and is the registered zero-amplitude negative control
    # (acceptance gate G2): bitwise-identical to OFF.
    inflow_perturbation_amplitude_scale: float = 1.0
    # "inflow" perturbs the flow_dep_bdy inflow faces (the mechanism);
    # "outflow" perturbs the complementary faces instead -- the
    # registered AC-P3.4 mutation control, selected by configuration.
    inflow_perturbation_faces: str = "inflow"
    # WRF v4.6.1 Registry/Registry.EM_COMMON:2889, verbatim:
    #
    #   rconfig   logical  moist_mix6_off   namelist,dynamics  max_domains
    #       .false. rh  "moist_mix6_off"
    #       "de-activate 6th-order horizontal filter for moisture"  ""
    #
    # WRF's own switch, in WRF's own spelling, with WRF's own default: the
    # importer must round-trip a stock namelist that sets it, and inventing
    # a second name for a knob WRF already has is the defect commit a0ef9d29
    # reverted for isfflx.  It is per domain because the Registry column is
    # max_domains.
    #
    # WHAT IT GATES, EXACTLY: rk_scalar_tend calls sixth_order_diffusion on
    # the moist array under ``(diff_6th_opt .NE. 0) .and. (.not. mix6_off)``
    # (dyn_em/module_em.F:1421, reached from dyn_em/solve_em.F:2230 with
    # config_flags%moist_mix6_off).  It touches the MOIST array only --
    # theta keeps its filter, and TKE has its own tke_mix6_off -- so woof
    # gates exactly the moist rows of the diff6 row set and nothing else.
    #
    # Default .false. is WRF's, so every frozen trajectory is bitwise
    # unchanged and diff_6th_opt = 0 runs make it inert either way.  It is
    # divergence-ledger entry L4 (woof/physics_mode.py, PROVENANCE.md):
    # a configuration-policy patch WRF can also express, which is why it is
    # measured against observations rather than asserted.
    moist_mix6_off: bool = False
    # WRF v4.6.1 Registry defaults for the Grell-family namelist keys
    # (Registry.EM_COMMON:2544,2546).  clos_choice=0 is the 16-member
    # ensemble closure, the arm the GF oracle covers; 1..16 run one member
    # alone (GF_CLOSURE_MEMBERS); ishallow toggles CUP_gf_sh, both arms
    # oracle-covered.
    # Read only where cu_physics = 3, which no frozen configuration
    # selects, so appending them cannot move a frozen trajectory.
    clos_choice: int = 0
    ishallow: int = 0
    # WRF v4.6.1 NSSL variant selectors
    # (Registry.EM_COMMON:2420-2425).  Since v4.5 every NSSL configuration
    # is mp_physics=18 plus these flags; the deprecated scheme IDs
    # 17/19/21/22 are namelist-import spellings that
    # share/module_check_a_mundo.F:3382-3421 rewrites onto them.  ``-1``
    # takes WRF's consistency-pass defaults (:3433-3455), which resolve to
    # the shipped two-moment + hail + predicted-CCN + graupel/hail-volume
    # lane.  Resolution and the ported-combination gate live in
    # woof/core/nssl2_contract.py; unported combinations refuse loudly at
    # validation instead of substituting a nearby branch.
    #
    # Appended last, as every new field must be, so positional construction
    # of every existing RunConfig is unchanged.
    nssl_2moment_on: int = -1
    nssl_hail_on: int = -1
    nssl_ccn_on: int = -1
    nssl_density_on: int = -1
    nssl_3moment: int = 0

    # ---- WDM6 (mp_physics = 16), appended after the NSSL block at the 1.9
    # assembly; both lanes append last, so every existing RunConfig's
    # positional construction is unchanged either way.
    # so positional construction of every existing RunConfig is unchanged.
    #
    # WDM6's half of WRF's SHARED ``hail_opt`` namelist key.  WRF reaches
    # wdm6init with ``config_flags%hail_opt`` (module_physics_init.F:
    # 4582-4584) -- the same scalar mp_wsm6_init reads at :4487 -- and the
    # branch sets the same five rimed-ice constants.  woof keeps the two
    # SEPARATE because the config field is the restart-identity record of
    # which scheme ran under which rimed ice; the importer writes both from
    # the one namelist key, so no namelist can make them disagree.
    #
    # ``wdm6_ccn_conc`` is WRF ``ccn_conc`` (Registry.EM_COMMON:2664,
    # default 1.0E8 # m-3).  Its ONLY effect in module_mp_wdm6.F is the
    # first-step fill of the CCN array (:220-227): ``wdm62D`` and
    # ``wdm6init`` both take ccn0 ``intent(in)`` and never read it.  woof
    # performs that fill once at state allocation, so this value is part of
    # the initial condition and therefore trajectory-bound.
    #
    # Both are SCHEME-SCOPED: nothing outside the mp=16 path reads either,
    # and ``woof.core.model.restart_identity_payload`` drops them from the
    # identity of every run that does not select WDM6, on the same
    # absent-stays-absent convention the ``perturbation`` block and the
    # per-domain ``spawn`` declaration already use.  That is what keeps
    # every pre-WDM6 experiment's fingerprint byte-identical.
    wdm6_hail_opt: int = 0
    wdm6_ccn_conc: float = 1.0e8

    # APPENDED LAST, and deliberately so: the config-freeze discipline
    # requires new fields at the end so positional construction of every
    # frozen case is unchanged.
    #
    # THE SURFACE-RADIATION CARRIER POLICY (woof/core/radiation_carriers.py).
    # "required" is the default for every run, real-data or idealised: a
    # radiative carrier a land-surface scheme reads must have a producer,
    # checked immediately before the LSM consumes it.  "wrf_compat_zero" is
    # the declared escape -- unsourced carriers are consumed at their
    # allocation fill, reproducing pre-1.9 behaviour.  It is never selected
    # automatically, it labels every carrier it touches in the run receipt
    # and the output metadata, and it is an EXPERIMENTAL FORCING rather
    # than a valid configuration for a real case.  Not a WRF namelist key:
    # WRF has no carrier provenance to declare.
    surface_radiation_policy: str = "required"

    # mp=28 WIF aerosol-climatology dataset (change record: appended with
    # the development branch commit onto this line).  Filesystem path to
    # WRF's global monthly QNWFA/QNIFA dataset in WPS intermediate format
    # (QNWFA_QNIFA_SIGMA_MONTHLY.dat as distributed).  Consumed ONLY when
    # aer_init_opt=1 AND wif_input_opt=1 select the ported climatology
    # ingest (woof/ingest/wif_climatology.py); the empty default runs not
    # one new instruction, and validate_aerosol_source_options refuses a
    # set path whose selectors do not consume it, so it can never ride
    # along silently.  A PATH, not a table row, because the dataset is a
    # user-provided 225 MB artifact, not something this repository ships.
    wif_climatology_path: str = ""
    # Which mp=28 aerosol initial state this run wants (change record:
    # appended with the development branch commit onto this line).  "auto",
    # the default, resolves WRF's monthly WIF climatology and uses it; when
    # the machine has no copy it downloads the pinned one, and when that
    # fails it refuses by name (acquire_wif_climatology).  It no longer
    # falls back to thompson_init's synthetic profile with a warning (a real-data
    # mp=28 run, 2026-10-23).  "climatology" is the same
    # commitment, stated.  "synthetic" selects that profile deliberately, which is
    # what an idealized case or a reproduction of a run before that change
    # wants.  Inert under every other mp_physics.
    mp28_aerosol_source: str = "auto"
    #: mp_physics=50/51 (P3): which P3 implementation integrates the column.
    #: ``"cuda"`` is the DEFAULT and is the shipping path -- the device
    #: kernels in woof/core/kernels/p3.cu, launched one step at a time
    #: (``"cuda"`` selects the level-parallel arm; ``"fused"`` is the
    #: three-launch composition proven byte-identical to it on all twelve
    #: p3-fortref fixtures).  ``"reference"`` selects the CPU float32
    #: transcription in woof/core/p3.py, which exists so every device
    #: result stays checkable and so the scheme runs where there is no
    #: CuPy; it is orders slower and is not a production selection.
    #:
    #: SCHEME-SCOPED (woof.core.model.SCHEME_SCOPED_RUN_FIELDS[50/51]) with an
    #: off-scheme refusal below, in this same commit, for the reason the
    #: mp=28 pair records: an unscoped RunConfig field moves EVERY
    #: experiment fingerprint and makes every existing checkpoint refuse to
    #: resume, for a field the run never reads.
    p3_backend: str = "cuda"
    #: Run cu_physics = 16's kernels with cu_physics = 6's DEEP closure:
    #: fixed ztau = 2400 s (module_cu_tiedtke.F:105) and the moisture-
    #: convergence first guess (:855-866) in place of scale_fac scaling and
    #: the geometric 0.1*zmfmax. This is an emulation inside cu16, not
    #: WRF's original cu6 scheme. It does not reproduce module_cu_tiedtke.F.
    #: Select cu_physics=6 for the separate original-scheme implementation.
    #: False leaves every cu16 result bit-identical.
    #:
    #: APPENDED LAST, as every field since the freeze is.  d0c23dad0 (2.6.4)
    #: declared it between ``isftcflx`` and ``iz0tlnd``, which is the one
    #: placement tests/test_config_freeze.py exists to refuse; it moves
    #: here on the line after 2.6.5.  The move changes no hash and no trajectory:
    #: every RunConfig digest sorts its keys (woof/io/restart.py
    #: ``_canonical_json``, woof/core/model.py ``experiment_fingerprint``),
    #: the freeze goldens compare as dicts, and no constructor in the tree
    #: builds a RunConfig positionally.  It is NOT scheme-scoped in
    #: woof.core.model.SCHEME_SCOPED_RUN_FIELDS, whose rows are keyed by
    #: mp_physics: a cumulus-keyed row is a mechanism this tree does not
    #: have yet, so the field binds every experiment fingerprint the way
    #: the unscoped fields before the mp=28 pair did (the anchors in
    #: tests/test_water_overlay.py re-pin for it, with the attribution
    #: measured there).
    ntiedtke_tiedtke_closure: bool = False
    # --- adaptive time step (WRF Registry.EM_COMMON:2269-2281) ----------
    #
    # SCOPE IS NOT UNIFORM ACROSS THIS BLOCK, and the split is upstream's.
    # target_cfl, target_hcfl, max_step_increase_pct and the three
    # min/max/starting pairs are declared `max_domains` -- genuinely per
    # domain, and listed in experiment._DOMAIN_RUN_OVERRIDES.  The three
    # ABOVE them are declared scope `1`: ONE scalar for the whole run, no
    # matter how many domains.  They are deliberately absent from that
    # tuple, which is what makes "[shared] only" enforced rather than
    # documented.
    #
    # Every one of these is registered in
    # ingest.prepared_cache.DEFAULT_TOLERANT_IDENTITY_FIELDS in the same
    # commit that adds it here.  A field that joins RunConfig without
    # joining that table refuses every tree prepared before it -- that has
    # now happened four times in this package's history, and the table's
    # own comment says so.
    use_adaptive_time_step: bool = False
    step_to_output_time: bool = True      # Registry default is .true.
    adaptation_domain: int = 1
    target_cfl: float = 1.2
    target_hcfl: float = 0.84
    max_step_increase_pct: int = 5
    #: -1 means "unset"; WRF substitutes a grid-spacing value in
    #: start_em.F:939-953 -- NINT(4*min(dx,dy)/1000) for starting_time_step,
    #: 8*dx(km) for max_time_step and 3*dx(km) for min_time_step.  Each carries a `_den` companion
    #: because upstream takes them as exact rationals Sn/Sd, and 0 there
    #: means "the value is whole seconds" rather than a zero denominator.
    starting_time_step: int = -1
    starting_time_step_den: int = 0
    max_time_step: int = -1
    max_time_step_den: int = 0
    min_time_step: int = -1
    min_time_step_den: int = 0
    # APPENDED LAST, deliberately: this field was declared between
    # ``etac`` and ``moist`` on development branch, which shifted
    # 150 fields' positional index and is what
    # tests/test_config_freeze.py::test_new_fields_are_reviewed_defaults_
    # appended_last exists to prevent (same correction the New Tiedtke
    # closure took in 2.6.4).  Nothing positional constructs RunConfig in
    # this tree today, which is the property the discipline preserves.
    #: This domain's OWN full-level eta ladder, ``nz + 1`` interfaces running
    #: strictly from 1.0 at the surface to 0.0 at the model top.  ``None``
    #: means the domain inherits whatever ladder its source carries, which is
    #: what every run did before per-domain ladders existed and what a bare
    #: config still does.
    #:
    #: An offline child (``woof downscale``) uses this to carry a DEEPER
    #: ladder than the archived parent it is built from -- the LES
    #: downscaling case, where the child wants the levels and not just the
    #: columns.  Naming a level count alone would be a trap: with no ladder
    #: the child would silently get ``make_vertical_coord``'s uniform default
    #: (woof/core/grid.py), which is a different atmosphere from its
    #: parent's stretched one.  ``p_top``, ``hybrid_opt`` and ``etac`` stay
    #: SHARED with the parent and are deliberately absent here; see
    #: woof/vertical_remap.py :: require_shared_column_basis for why the
    #: remap is only well posed when they agree.
    eta_levels: tuple[float, ...] | None = None
    # APPENDED LAST, after eta_levels, on the same discipline.  The two
    # lateral-boundary keys a downscaled child needs because its parent is
    # much coarser than it is (woof/downscale.py _derive_child_run_config).
    #: Davies relaxation time scale, in seconds, on the first relaxed row:
    #: the relaxation pulls that row toward the boundary state with this
    #: e-folding time, and the pull weakens linearly across the zone
    #: (fcx = ramp / relax_timescale_s, gcx = ramp / (5 relax_timescale_s)).
    #: 0 keeps WRF's recipe, fcx = 0.1 / dt and gcx = 1 / (50 dt), which is
    #: the same law with the time scale tied to 10 of the domain's OWN
    #: steps, so it moves whenever the step does.  A derived child sets it
    #: in seconds instead: the time a 20 m/s flow takes to cross one child
    #: cell (woof/downscale.py CHILD_RELAX_CROSSING_SPEED, where the arm
    #: measurement that chose it is recorded).  A nest reads it too, in
    #: WRF's nested FP32 operation order.
    relax_timescale_s: float = 0.0
    #: Whether a specified domain relaxes ``w`` toward its boundary table
    #: in the relaxation zone and takes the table's ``w`` on the specified
    #: rows, as a WRF nest does.  False keeps WRF's root-domain rule: ``w``
    #: is never relaxed, and the specified rows copy the first interior
    #: row (zero gradient), which carries an updraft that forms inside the
    #: relaxation zone straight onto the boundary.  Needs a ``w`` table,
    #: which an offline child's parent history always carries.
    relax_w: bool = False
    # APPENDED LAST, after relax_w, on the same discipline.
    #: Under the adaptive clock, the fewest acoustic substeps per step this
    #: domain takes.  The clock derives the count from its live step every
    #: root step (WRF's ``time_step_sound = 0`` rule,
    #: :func:`woof.core.adaptive_clock.wrf_num_sound_steps`), which gives 4
    #: at 1 km for any step under about 3.3 s; this is a floor under that
    #: count.  0, the default, leaves WRF's derived count untouched.  The
    #: steep-terrain rules (:mod:`woof.acoustic_adaptation`,
    #: :mod:`woof.terrain_clock`) set it on each adaptive domain whose
    #: substep count they raise, because without it the clock put their
    #: six substeps back to four: a 1 km forecast under a 67 m/s
    #: crest-level jet stopped at model second 135 on the adaptive clock
    #: with its step ceiling honoured, and ran three hours once the six
    #: substeps reached the dynamics.  Under a fixed clock nothing reads
    #: it; ``time_step_sound`` is the count there.  One of
    #: :data:`woof.core.model.ADAPTIVE_TIMESTEP_RUN_FIELDS`, so it drops
    #: out of every fixed-clock identity and is controller policy under an
    #: adaptive one.
    min_time_step_sound: int = 0
    #: Minimise nested cell-steps with a shorter root step. Changes answers.
    adaptive_nest_lattice: bool = False
    # APPENDED LAST, after adaptive_nest_lattice, on the same discipline.
    #: WRF's urban canopy selector (``&physics sf_urban_physics``,
    #: Registry.EM_COMMON:2489).  0 = no urban model: city cells are the
    #: land-surface scheme's bulk urban category, which is WRF's default
    #: and every configuration before this field.  1 = the single-layer
    #: UCM (phys/module_sf_urban.F), 2 = BEP (module_sf_bep.F), 3 = BEP+BEM
    #: (module_sf_bep_bem.F).  :func:`validate_urban_config` is the law.
    sf_urban_physics: int = 0
    #: ``&physics use_wudapt_lcz``: 1 reads URBPARM_LCZ.TBL and maps the
    #: eleven Local Climate Zone categories to urban types 1-11; 0 reads
    #: URBPARM.TBL (three types).  Read only when ``sf_urban_physics > 0``.
    use_wudapt_lcz: int = 0
    #: ``&physics num_urban_hi`` (Registry.EM_COMMON:2534): the bin count of
    #: the building-height histogram HI_URB2D.  WRF's default 15.
    num_urban_hi: int = 15
    #: WRF &physics slope_rad (Registry.EM_COMMON, max_domains, default 0):
    #: 1 hands the land surface the shortwave on the local slope, and
    #: removes its direct beam where topo_shading finds the sun hidden
    #: (module_surface_driver.F:1920-1936, TOPO_RAD_ADJ).  WRF applies it
    #: only with a longwave scheme on, and so does woof
    #: (:mod:`woof.core.topo_radiation`).
    slope_rad: int = 0
    #: WRF &physics topo_shading (max_domains, default 0): 1 computes the
    #: terrain shadow mask at every radiation call (toposhad); only the
    #: slope_rad adjustment reads it.
    topo_shading: int = 0
    #: WRF &physics shadlen (scalar, default 25000. m): how far toward the
    #: sun the shadow search looks.
    shadlen: float = 25000.0
    # APPENDED LAST, after the urban and topographic-radiation fields.
    #: WRF ``zadvect_implicit`` (WRF 4.3 and later, dynamics namelist,
    #: Registry default 0): 1 runs the implicit-explicit vertical advection
    #: of Wicker and Skamarock (2020) on the last RK3 substep (:mod:`woof.
    #: core.ieva`; its w solve's two boundary terms take consistent units, a
    #: declared divergence from WRF 4.7.1, A179).  0 leaves every step as is.
    zadvect_implicit: int = 0
    # APPENDED LAST, after zadvect_implicit (A165).
    #: WRF ``w_crit_cfl`` (dynamics namelist, one value for every domain,
    #: Registry default 1.0): the vertical Courant number ``w_damp``
    #: (``w_damping = 1``) measures its excess from, and, under
    #: ``zadvect_implicit = 1``, where it starts (WRF 4.7.1
    #: module_big_step_utilities_em.F w_damp; without zadvect_implicit it
    #: starts at w_beta = 1).  WRF's comments recommend 2.0 with
    #: zadvect_implicit.  1.0 runs exactly the limiter woof always ran.
    w_crit_cfl: float = 1.0
    # APPENDED LAST, after w_crit_cfl, on the same discipline.
    #: Noah land-use tiles (WRF v4.7.1 Registry.EM_COMMON:2536).
    #: Default off preserves the dominant-category surface column.
    sf_surface_mosaic: int = 0
    #: Tile count, read only when mosaic is on (Registry.EM_COMMON:2537).
    mosaic_cat: int = 3
    #: Where the urban canopy runs under Noah mosaic with
    #: ``sf_urban_physics = 1``; one of :data:`MOSAIC_URBAN_CANOPY_RULES`.
    #: "dominant" is WRF v4.7.1: urban_var_init zeroes FRC_URB2D wherever a
    #: cell's dominant category is not urban (module_sf_urban.F:2820-2821),
    #: so the urban tile of a mostly rural cell runs as vegetation.
    #: "every_tile" gives that tile its urban type's own fraction instead.
    #: Per domain; read only when mosaic and urban option 1 are both on.
    mosaic_urban_canopy: str = "dominant"
    # Appended to preserve existing positional RunConfig construction.
    # 1 follows model coordinate surfaces; 2 is the existing metric form.
    diff_opt: int = 2
    mix_full_fields: bool = True
    # Appended after the existing diffusion selectors, preserving their
    # positional RunConfig construction.
    #: WRF ``&physics topo_wind`` (Registry.EM_COMMON, max_domains, default
    #: 0): the sub-grid terrain correction of the surface wind, under YSU
    #: (:mod:`woof.core.terrain_drag`).  1 is Jimenez and Dudhia (2012):
    #: the first-level drag times ln(sqrt(VAR_SSO)) weighted toward stable
    #: columns, tapered off on hill tops, where the 10 m wind blends toward
    #: the first-level wind; 2 scales the drag by (VAR*0.4/200+1.175)^2,
    #: capped at 1.575^2.  0 runs every column exactly as before.
    topo_wind: int = 0
    #: WRF ``&dynamics gwd_opt`` (max_domains, default 0): orographic drag
    #: added to the PBL momentum tendencies (:mod:`woof.core.terrain_drag`).
    #: 1 is the KIM gravity-wave drag with flow blocking (module_bl_gwdo.F);
    #: 3 the GSL drag suite (module_bl_gwdo_gsl.F: large-scale gravity-wave
    #: drag and blocking, small-scale gravity-wave drag, turbulent
    #: orographic form drag, each tapered by grid length).  0 adds nothing.
    gwd_opt: int = 0

    #: Numerical generation of WRF's implicit vertical advection.
    #: wrf_471 preserves the existing module_ieva_em operator; wrf_legacy
    #: selects module_advect_em's current-mass solve and one-sided split.
    #: Both retain the declared lower-w-boundary unit correction.
    zadvect_implicit_variant: str = "wrf_471"

    # WRF's post-PBL scalar diffusion, separate from MYNN plume transport.
    scalar_pblmix: int = 0

    # Operational namelist spelling for analyzed water/ice-friendly number
    # mixing ratios. The ingest is generic and reads canonical source rows.
    use_rap_aero_icbc: bool = False

    # WRF CLM lake model. Registry default 0 preserves prescribed water
    # temperature; option 1 evolves lake water, sediment, ice and snow.
    sf_lake_physics: int = 0
    use_lakedepth: int = 1
    lakedepth_default: float = 50.0
    lake_min_elev: float = 5.0
    # Appended to preserve positional construction. Enabled WRF parameter
    # consumers require member-owned patterns before physics runs.
    spp_conv: int = 0
    spp_pbl: int = 0
    #: Which WRF lineage's LSMRUC SOILPROP sets soil-water diffusivity and
    #: hydraulic conductivity (woof.core.ruc_tier.RUC_SOILPROP_FORMS).
    #: ``wrf_45`` (WRF v4.0-4.5 and the operational RAP/HRRR branch):
    #: normalised by the moisture above the residual.  ``wrf_461`` (WRF
    #: v4.6.1): total moisture over porosity, 2.5 to 8 times the water
    #: diffusivity in dry soil, measured to wet a dry top soil level from
    #: below by +0.026 m3/m3 in one afternoon hour.  Appended after the
    #: SPP consumer switches.
    ruc_soilprop: str = "wrf_45"
    # APPENDED LAST, after ruc_soilprop, on the same discipline.
    #: WRF ``&physics fractional_seaice`` (Registry.EM_COMMON, default 0).
    #: Selects WHICH of WRF's two sea-ice thresholds the RUC seam runs
    #: (module_surface_driver.F:1365-1368 of the HRRR v4.1.21 fork): 0 keeps
    #: XICE_THRESHOLD = 0.5, the value every RUC run before this field was
    #: pinned to, so frozen trajectories are unchanged; 1 runs 0.02, HRRR's
    #: operational value, so every cell with 0.02 <= XICE <= 1 takes the ice
    #: column, the open-water surface-layer call and the post-LSM blend the
    #: seam already carries, and the CLM lake reads the same threshold.  The
    #: blend machinery runs under both values: 0 is the previous pin, not
    #: WRF's non-fractional surface treatment, and the RUC receipt names the
    #: threshold that ran.  Read by the RUC LSM only (sf_surface_physics=3).
    fractional_seaice: int = 0
    # Prescribed analysis stream. WRF's sst_update remains 0 unless selected.
    #: Which generation of the MYNN surface layer (sf_sfclay_physics = 5)
    #: runs (woof.core.mynn_surface.MYNN_SFCLAY_VARIANTS).  ``wrf_461``:
    #: WRF v4.6.1 module_sf_mynn.F.  ``gsl_wrf39``: the GSL WRF 3.9 fork's
    #: module (NOAA-EMC/HRRR v4.1.21), a 5-pass secant z/L search that gives
    #: up to 5 Ri / 8 Ri, z/L capped at 50, the Richardson number clamped at
    #: 50 after step 1, zt in the heat log numerators and z0/L as psih's
    #: lower limit.  A configuration whose ``[fetch] source`` names an
    #: operational-fork model (hrrr, hrrr-native, hrrr-prs, rap,
    #: rap-native) and omits this key takes ``gsl_wrf39`` at load
    #: (woof.physics_source_defaults.GENERATION_SELECTORS); a written
    #: value, including ``wrf_461``, is kept.  Appended last.
    mynn_sfclay_variant: str = "wrf_461"
    #: Whether the engine's measured terrain rules may rewrite this
    #: domain's clock at launch.  ``"local_face"`` is the default (below).
    #: ``"measured"`` (the default until 2.8.8, and every configuration
    #: before this field): the terrain clock
    #: (:mod:`woof.terrain_clock`) divides the step or raises the acoustic
    #: substep count where its measured map saw a longer step stop under
    #: the domain's steepest slope, crest height and crest-level wind, and
    #: the steep-ground substep rule (:mod:`woof.acoustic_adaptation`)
    #: raises four substeps to six where its map says four fail.
    #: ``"pinned"``: these launch rules preserve ``dt`` and
    #: ``time_step_sound``; a fixed clock runs exactly as configured.
    #: A selected adaptive controller still updates its live clock.
    #: Both rules still read the domain and write what they
    #: would have done into the run's receipt (``terrain_clock`` and
    #: ``acoustic_substeps``, each row with ``clock = "pinned"`` and an
    #: ``advice`` entry), and the run prints that advice, but nothing is
    #: applied.  The off-centering floor is not a clock rule and is read
    #: either way: a chosen ``epssm`` the measured map holds no substep
    #: count at is refused by name in both modes, because WRF stops on it
    #: too.  Pinned is what an operational WRF namelist means by its clock:
    #: the step and count it names are the ones that integrate.  The
    #: measured maps were taken on generated bell ridges and never on an
    #: operational grid, so on such a grid their verdict is advice, and the
    #: run's own stability evidence is the referee.  Nothing in preparation
    #: reads it; a prepared tree carries the same state under either value.
    #: ``"local_face"`` (the default from 2.8.8): the terrain clock reads
    #: every terrain face with the crest and crest-level wind in a
    #: neighbourhood around it, a measured wind margin and its own measured
    #: rows (:mod:`woof.terrain_clock_local`) instead of the domain-wide
    #: steepest slope, highest ground and strongest wind, and it is never
    #: worse than ``"measured"``: where the face-by-face reading is
    #: BEYOND_MEASURED or would run a shorter first or longest step, the
    #: grid takes the ``"measured"`` decision and its receipt and run line
    #: say so (:func:`woof.terrain_clock_local.never_worse`).  Breakage
    #: the default answers: the domain-wide reading put a Plains jet on a
    #: mountain face hundreds of km away and halved NCAR's published CONUS
    #: steps (72 s to 36 s at 12 km, 15 s to 7.5 s at 2.5 km), which ran
    #: clean at NCAR's own steps (CLOCK-CHECK-NCAR-2026-10-06, steps 3 to
    #: 7).  The substep rule reads as under ``"measured"``.  A checkpoint
    #: echo or fingerprint without ``terrain_clock`` was written under
    #: ``"measured"`` and still reads so; ``"local_face"`` binds
    #: (:func:`woof.io.restart.configuration_echo`).
    terrain_clock: str = "local_face"
    #: Which WRF source's sixth-order filter runs (diff_6th_opt > 0;
    #: woof.core.dycore.DIFF6_FORMS).  ``wrf_461`` (WRF v4.6.1
    #: sixth_order_diffusion and rk_scalar_tend): moisture, scalar and TKE
    #: rows take ``diff_6th_factor`` on the RK3 first-stage step dt/3, and
    #: the loops stop three points short of a specified or nested edge.
    #: ``noaa_wrf39`` (the NOAA-EMC WRFV3.9 fork, HRRR v4.1.21
    #: dyn_em/solve_em.F:2418/:2680/:2974, module_em.F:1217/:1409,
    #: module_big_step_utilities_em.F:6596-6632): moisture and scalar rows
    #: take ``diff_6th_factor2`` and every scalar row the full step, and
    #: the filter runs to the domain edge reading zero-gradient halo copies.
    #: Appended last.
    diff_6th_form: str = "wrf_461"
    #: The fork's second filter factor for the moist and scalar arrays
    #: (fork Registry.EM_COMMON:2629, namelist ``&dynamics``, per domain).
    #: ``None`` takes the fork Registry default 0.04 under ``noaa_wrf39``;
    #: WRF v4.6.1 has no such key, so a value is refused under ``wrf_461``.
    diff_6th_factor2: float | None = None
    #: WRF ``mp_zero_out`` (Registry.EM_COMMON:2553, both sources; routine
    #: phys/module_microphysics_zero_out.F identical in both): 0 off, 1
    #: zero non-vapour species under ``mp_zero_out_thresh``, 2 also floor
    #: vapour at zero; the outermost ring is floored at zero either way.
    mp_zero_out: int = 0
    mp_zero_out_thresh: float = 1.0e-8
    #: WRF v4.6.1 ``mp_zero_out_all`` (Registry.EM_COMMON:2554): 1 also
    #: applies the pass to the scalar (number) arrays.  The NOAA WRFV3.9
    #: fork has no switch and always does (solve_em.F:4078-4141), which is
    #: v4.6.1's 1.
    mp_zero_out_all: int = 0
    #: Source form of advance_w's upper-wind limiter. The NOAA WRFV3.9
    #: fork damps saved stage winds above 110 m/s inside zdamp on every
    #: acoustic substep (module_small_step_em.F:1703-1735). WRF v4.6.1
    #: has no such limiter. Appended to preserve positional construction.
    upper_wind_limiter_form: str = "wrf_461"
    # Appended after all staged fields, on the same positional
    # discipline.  WRF ``&dynamics`` advection orders (Registry.EM_COMMON,
    # max_domains): the vertical face-flux ladder of module_advect_em.F.
    #: WRF ``v_sca_adv_order``: the vertical order of every scalar (theta,
    #: moisture, scalars, TKE) AND of w, because WRF's advect_w keys its
    #: vertical order on the scalar order (module_advect_em.F
    #: ``vert_order = config_flags%v_sca_adv_order`` in advect_w).  3 is
    #: the ladder every earlier run took (2nd order one face in from the
    #: eta boundaries, flux3 between).  5 is WRF's ``vert_order == 5``
    #: ladder: faces kts+1 and ktf 2nd order, kts+2 and ktf-1 flux3,
    #: kts+3 .. ktf-2 flux5 (the operational HRRR value, which runs it
    #: with ``zadvect_implicit = 1``).
    v_sca_adv_order: int = 3
    #: WRF ``v_mom_adv_order``: the same ladder for u and v.
    v_mom_adv_order: int = 3
    #: WRF ``h_mom_adv_order``, declaration only: the u, v and w
    #: horizontal stencils are WRF's flux5, so 5 is the only value the
    #: kernels carry; the field exists so a namelist's dynamics block
    #: maps one to one and any other value is refused by name.
    h_mom_adv_order: int = 5
    #: Which WRF lineage's LSMRUC irrigation rule holds root-layer soil
    #: moisture up after SFCTMP (woof.core.ruc_mosaic.IRRIGATION_FORMS).
    #: ``wrf_45`` (WRF v4.0-4.5 and the operational RAP/HRRR branch): a
    #: hard floor scaled by the cell's cropland fraction, gated on leaf area,
    #: whatever mosaic_lu says (LANDUSEF fractions when the run carries them,
    #: the dominant category as the whole cell when it does not).
    #: ``wrf_461`` (WRF v4.6.1, under mosaic_lu=1): a per-step relaxation
    #: to the full 1.1 x wilting point for any cell with any crop fraction
    #: and a greenness factor above 0.75. This is the generic default;
    #: importing the operational namelist or using its recipe selects wrf_45.
    #: Appended last.
    ruc_irrigation: str = "wrf_461"
    #: LSMRUC's cold start of the ground vapour (woof.core.ruc_tier.
    #: RUC_QVG_COLD_START_FORMS).  ``wrf`` (public WRF, the default): an
    #: invalid QVG starts from saturation at the skin times moisture
    #: availability.  ``air`` (the operational RAP/HRRR branch's fallback):
    #: from the lowest-level vapour, measured to pull the skin toward the
    #: air's dewpoint on the first steps.  Read only on a cold start.
    #: Appended last.
    ruc_qvg_cold_start: str = "wrf"
    #: How SFCDIAGS_RUCLSM writes T2, TH2 and Q2 (woof.core.ruc_tier.
    #: RUC_2M_DIAGNOSTIC_FORMS).  ``flux`` (public WRF, the default) or
    #: ``log_profile`` (the operational RAP/HRRR branch's logarithmic
    #: profile where the air is warmer or moister than the surface).
    #: Appended last.
    ruc_2m_diagnostic: str = "flux"
    #: Which WRF lineage's RUC snow scheme runs (woof.core.ruc_tier.
    #: RUC_SNOW_FORMS).  ``wrf_45`` (WRF v4.0-4.5, carried by the
    #: operational RAP/HRRR branch): constant snow
    #: conductivity, cover from depth over a critical depth, fresh-snow
    #: albedo from the depth on the ground, a step-independent melt cap.
    #: ``wrf_461`` (WRF v4.6.1): the Sturm conductivity, the blended
    #: depth/roughness cover rebuilt after the snow column, and the
    #: density-gated melt limits. The generic default is wrf_461; the
    #: operational namelist importer and recipe explicitly select wrf_45.
    #: Appended last.
    ruc_snow: str = "wrf_461"
    #: Which generation of the MYNN boundary-layer scheme the column solver
    #: runs (woof.core.mynn_pbl_gpu.MYNN_VERSIONS).  ``wrf_461`` is WRF
    #: v4.6.1 module_bl_mynn.F.  ``gsd_41`` is the GSD MYNN v4.1 carried by
    #: the NOAA-EMC WRF 3.9 branch (tag v4.1.21 of its repository), ported
    #: row by row: a downward surface vapour flux (dew, frost) reaches the
    #: vapour equation instead of being deleted, and mixing length option 2
    #: takes that generation's constants, caps and blend.  Rows not yet
    #: ported run the v4.6.1 form under either name (docs/dev/
    #: mynn-gsd41.md lists them).  Appended last.
    bl_mynn_version: str = "wrf_461"
    #: ``gsd_41`` mixing length option 2 converts the interface q to TKE as
    #: 0.5*q (module_bl_mynn.F v4.1.21:995) where its own option 1 and every
    #: later generation use 0.5*q**2.  True takes that line as written;
    #: False (default) takes the squared form, so the default fork run
    #: carries the fix.  Read only under bl_mynn_version = "gsd_41" with
    #: bl_mynn_mixlength = 2.  Appended last.
    bl_mynn_gsd41_unsquared_qtke: bool = False
    #: Heat reconstruction and negative-condensate handling only. wrf_461
    #: (default) uses the mixed condensate and conserves water through the
    #: existing moisture repair. gsd_41 takes the fork's two defined defects
    #: together: pre-mixing condensate in heat and tendency-only condensate
    #: clipping (module_bl_mynn.F:3417-3421, 3449-3453, 3487-3497).
    #: The defect form requires bl_mynn_version=gsd_41 and is never selected
    #: automatically by an importer or recipe. Appended after staged fields.
    bl_mynn_cloud_tendency_form: str = "wrf_461"

    # WRF &physics radiation-driver options beyond the pinned RRTMG
    # envelope (Registry.EM_COMMON:2389-2390; both single-valued, both
    # default 0, which is WRF's own and leaves every earlier trajectory
    # byte-identical).  Operational HRRR v4 runs swint_opt = 1 and
    # aer_opt = 3 (hrrr_wrf.nl lines 114 and 156).
    #   swint_opt = 1: the surface shortwave between radiation calls is a
    #   per-column power-law fit in the solar zenith cosine, refitted at
    #   every radiation call and evaluated on every step
    #   (woof.core.swint; module_radiation_driver.F
    #   update_swinterp_parameters / interp_sw_radiation).  0 holds the
    #   last radiation call's fluxes for the whole interval.
    #   aer_opt = 3: the legacy RRTMG shortwave sees per-band aerosol
    #   optics built from the Thompson water- and ice-friendly aerosol
    #   numbers (woof.core.rrtmg_aerosol_optics; the driver's gt_aod plus
    #   module_ra_aerosol.F's calc_aerosol_rrtmg_sw).  0 runs zero
    #   aerosol.  1 (the ECMWF six-type climatology, iaer = 6) and 2 (the
    #   aer_aod550/angexp/ssa/asy auxinput path) run on the legacy RRTMG
    #   shortwave only (validate_radiation_driver_options).
    swint_opt: int = 0
    aer_opt: int = 0
    #: Sun-angle land albedo (WRF radiation driver's ALBSOL/ALBBCKSOL).
    #: 0 preserves the previous albedo carrier. 1 updates the two fields
    #: on radiation steps and supplies ALBSOL to shortwave and the RUC LSM.
    alb_sol: int = 0
    #: Which generation of the aerosol-aware Thompson microphysics
    #: (mp_physics = 28) runs (woof.core.thompson_aerosol_launch.
    #: THOMPSON_VERSIONS).  ``wrf_461`` (WRF v4.6.1, the default): the
    #: graupel number diagnosed from graupel content alone, the ice-to-snow
    #: size 300 microns.  ``wrf_39_noaa`` (the operational WRF 3.9 fork,
    #: NOAA-EMC/HRRR v4.1.21): the graupel intercept from graupel content
    #: and supercooled rain, non-increasing downward, the ice-to-snow size
    #: 200 microns, its own lookup tables
    #: (woof.physics_compat.thompson_fork_table_root).  An imported WRF
    #: namelist that carries the fork's own keys selects it, and so does a
    #: configuration whose ``[fetch] source`` names an operational-fork
    #: model (hrrr*, rap*) under mp_physics = 28 on every domain when this
    #: key is omitted (woof.physics_source_defaults.GENERATION_SELECTORS);
    #: a written value is kept.  Appended last.
    #: Classic mp=8 also registers ``wrf_44``: WRF v4.4's complete
    #: column statements and coefficient records. A WRF-folder import
    #: selects from the matching model TITLE; an explicit value wins.
    #: Inputs without registered provenance retain wrf_461.
    thompson_version: str = "wrf_461"
    #: How melting snow falls under thompson_version = "wrf_39_noaa".
    #: ``blend`` (the default): the rain-share blend of WRF v4.6.1, which the
    #: fork itself carries commented out as the upstream bug fix.
    #: ``wrf_39_noaa``: the fork's live expression, vts_boost 1.5 above 0 C
    #: and a fall speed divided by (T - 273.15) just above +0.1 C, singular
    #: there.  It is what the operational model integrates, so the HRRR
    #: namelist importer, the shipped mp28 recipes and a configuration
    #: whose ``[fetch] source`` names an operational-fork model (hrrr*,
    #: rap*) under the fork generation on every domain select it when this
    #: key is omitted (woof.physics_source_defaults.GENERATION_SELECTORS):
    #: the fork generation with the v4.6.1 snow fall is a physics HRRR
    #: never runs, and on the Iowa 2024-05-21 HRRR crop it kept three
    #: quarters of the first-hour gap to HRRR that the pair closes
    #: (WOOF-FIX-PROGRAM-2026-10-06, woof-hour1-spinup).  A written value
    #: is kept; the generic default stays ``blend``.  Appended last.
    thompson_fork_snow_fall: str = "blend"

    #: Source form of the legacy RRTMG cloud wrapper. WRF v4.6.1 is the
    #: frozen default; noaa_wrf39 selects the fork's liquid/ice shortwave
    #: radius defaults and full snow ice fraction, plus Thompson's fork
    #: cold-start effective radii. Appended to preserve positional calls.
    rrtmg_cloud_optics_form: str = "wrf_461"
    #: Explicit, immutable prescribed layer-smoke input for legacy SW.
    #: Empty keeps the existing no-smoke trajectory. This is a forcing
    #: sensitivity input, not an advected or emitted smoke chemistry state.
    rrtmg_smoke_manifest: str = ""
    #: Output-only surface energy carriers for reconstructed net radiation.
    #: The history writer reads live fields; no physics kernel reads this.
    #: Appended after rrtmg_smoke_manifest, ahead of the chem block (which
    #: stays the last positional block) and the keyword-only fire block.
    surface_energy_diag: bool = False
    #: WRF &time_control ``cycling`` (Registry default .false.): the start
    #: carries a previous forecast's scheme state, so the first MYNN call
    #: keeps the input QKE (unless its lowest-level maximum is below
    #: 0.0002, module_bl_mynn.F:4007-4018 of the NOAA-EMC/HRRR v4.1.21
    #: fork) and the input QC_BL and CLDFRA_BL, which the first radiation
    #: call and the first cloud decay also read. Both MYNN generations
    #: carry these fields; wrf_461 repairs the source's lost-carry zeroing.
    #: Appended after surface_energy_diag (2.8.8).
    cycling: bool = False
    #: Where the aer_opt = 3 shortwave aerosol type comes from.  ``rural``
    #: (the default): type 1 everywhere, the HRRR fork's PARAMETER
    #: (module_radiation_driver.F:769); every earlier configuration runs it.
    #: ``landmask``: the NOAA MPAS fork RRFS v2 runs, type 1 where xland is
    #: 1 and maritime type 3 where it is 2
    #: (mpas_atmphys_driver_radiation_sw.F:583-586 at RRFSx/MPAS-Model
    #: 1d6e370e).  Appended after cycling, ahead of the chem block, which
    #: stays the last positional block.
    rrtmg_aer3_type_source: str = "rural"
    # MYNN number-species mixing split by family, the MYNN-EDMF namelist
    # pair (NCAR/MYNN-EDMF c942418 module_bl_mynnedmf.F90: qni :4975 and
    # qnc :5064 gate on bl_mynn_mixnumcon, qnwfa :5151, qnifa :5241 and
    # qnbca :5331 on bl_mynn_mixaerosols).  RRFS v2 runs mixscalars=0,
    # mixaerosols=1, mixnumcon=0 (rrfs-workflow 45ace1bf
    # parm/hrrrv5/namelist.atmosphere:61-63).  Each key selects which of
    # this solver's own qn solves are coupled (MYNN_QN_MIX_KEYS); the
    # solve arithmetic stays the selected bl_mynn_version's.  Defaults 0:
    # every existing configuration mixes exactly what it mixed before.
    # They follow rrtmg_aer3_type_source and sit before the chem block,
    # the last of the positional fields (tests/test_chem_config.py holds
    # chem, then the keyword-only fire block, at the end).
    bl_mynn_mixaerosols: int = 0
    bl_mynn_mixnumcon: int = 0
    # Optional TEMPO coupling arms (named microphysics TEMPO only): the ML
    # subgrid-cloud radii and the pre-physics aerosol surface emission.
    # Off is the historical adapter behaviour.  Appended after the mixing
    # keys, ahead of the chem block.
    tempo_ml_for_bl_nc: bool = False
    tempo_surface_emission_before: bool = False
    #: Optional directory of generation-specific Thompson table records.
    #: The WRF-folder door supplies its own directory for a known generation.
    #: Appended at None after the TEMPO arms, ahead of the chem block, so
    #: positional construction keeps every earlier field where it was
    #: (tests/test_config_freeze.py); a header at None drops it.
    thompson_table_dir: str | None = None
    # --- 2.9.0 (golden/wrf-fixes): appended after thompson_table_dir, ahead
    # of the chem block (which stays the last positional block) and the
    # keyword-only fire block, so no earlier positional index moves.
    #: WRF scalar-array advection: 0 ordinary, 1 positive definite.
    #: Earlier forecasts retain 1 when this selector is omitted.
    scalar_adv_opt: int = 1
    #: Source numerical generation of the single-layer canopy.
    urban_ucm_generation: str = "wrf_471"
    #: -1 retains the corresponding URBPARM entry.
    urban_treeoption: int = -1
    urban_ttscheme: int = -1
    urban_fvg: float = -1.0
    #: Explicit URBPARM directory. Its resolved table bytes bind restart.
    urban_table_dir: str = ""
    # WRF 4.7/4.8 upstream correctness fixes, default-on (fixed means
    # default).  Each keeps the previous generation selectable for
    # reproduction; a checkpoint header without the key ran the previous
    # generation, so continuing it under the corrected default is refused
    # by name (woof/io/restart.py ECHO_ABSENT_RUN_VALUES).
    #: WRF 4.8 radconst/calc_coszen (julian - 0.5, 0.040849); ``wrf_461``
    #: reproduces julian - 1 and 0.04089.
    solar_generation: str = "wrf_480"
    #: WRF 4.8 YSU capped top-down entrainment, min(0.2*8*tmp1, 0.4);
    #: ``wrf_461`` reproduces 0.2*8*tmp1 + 0.2 on 2.8.9's kernel.
    ysu_version: str = "wrf_480"
    #: Snow-aging call placement only, not the complete Noah-MP version.
    #: ``wrf_470`` ages at night; ``wrf_461`` reproduces daylight-only aging.
    noahmp_snow_age: str = "wrf_470"
    #: Impervious SLUCM stores: WRF 4.8 allows evaporation during weak rain.
    #: Both generations retain the WRF 4.7.1 rho_air/1000 unit correction.
    urban_water_balance: str = "wrf_480"
    #: aer_opt = 2 (user AOD/Angstrom/ssa/asymmetry aerosol, WRF
    #: module_radiation_driver.F aerosol_in branch).  Appended after
    #: thompson_table_dir, ahead of the chem block, so positional
    #: construction keeps every earlier field where it was; a run that
    #: does not select aer_opt = 2 drops them from its identity.
    aer_type: int = 1
    aer_aod550_opt: int = 1
    aer_angexp_opt: int = 1
    aer_ssa_opt: int = 1
    aer_asy_opt: int = 1
    aer_aod550_val: float = 0.12
    aer_angexp_val: float = 1.3
    aer_ssa_val: float = 0.85
    aer_asy_val: float = 0.9
    aer_auxinput15: str = ''
    aer_auxinput15_interval: int = 0
    #: WRF lightning diagnostic package (lightning_option, Registry.EM_COMMON
    #: :3332-3357).  Appended after the aer_opt = 2 controls, ahead of the
    #: chem block, so positional construction keeps every earlier field where
    #: it was (the lane first inserted them mid-class); off reads none.
    lightning_option: int = 0
    lightning_dt: float = 0.0
    lightning_start_seconds: float = 0.0
    flashrate_factor: float = 1.0
    iccg_method: int = 0
    iccg_prescribed_num: float = 0.0
    iccg_prescribed_den: float = 1.0
    cellcount_method: int = 0
    cldtop_adjustment: float = 0.0
    do_radar_ref: int = 0

    #: Turbulent Prandtl number of the km_opt = 1 constants: xkhh =
    #: khdif/prandtl and xkhv = kvdif/prandtl (isotropic_km,
    #: module_diffusion_em.F:1756-1767), read by
    #: dycore.launch_wrf_isotropic_km.  WRF fixes it at prandtl = 1./3.0
    #: (share/module_model_constants.F:86) and has no namelist knob for
    #: it, so it is a class constant, not a field: no TOML key, no
    #: namelist import, no configuration digest or restart identity entry.
    constant_k_prandtl: ClassVar[float] = 1.0 / 3.0

    # Correct-mode mechanism keys (woof/correct_mode.py; physics.mode =
    # "correct" sets every one), appended last ahead of the chem block,
    # which stays the last positional block.  From 2.9.0 correct mode is the
    # default (fixed means default): each key defaults ON, except in a
    # strict WRF process (GPUWM_WRF_EXACT=1), which forces the matched
    # arithmetic (correct_mode.CORRECT_KEYS_DEFAULT).  False is the WRF
    # v4.6.1 / MPAS-A v8.4.2 arithmetic, still selectable, and named by every
    # parity and identity proof; a checkpoint header without the key ran it,
    # so continuing that checkpoint under the new default is refused by name
    # (woof/io/restart.py ECHO_ABSENT_RUN_VALUES and
    # CORRECT_MODE_RESTART_BREAK_NOTICE).
    # WB5: WSM6 books the below-ground part of the straddling source cell
    # with the remap's own piecewise-linear profile (kernels/wsm6.cu,
    # WSM6_PLM_SURFACE) so column water + surface precipitation closes.
    wsm6_conservative_surface: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    # WB4: New Tiedtke's water tendencies are per unit of its moist frame
    # mass dp/g = m_dry*(1+qv); scale them by (1+qv) onto dry mass and hand
    # the scheme a specific-humidity vapour forcing (woof/core/ntiedtke.py).
    ntiedtke_dry_mass_coupling: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    # WB5 in WDM6 (2.9.0, development branch): mp_wdm6.F carries the same
    # sedimentation routine and books the below-ground part of the straddling
    # cell at the flat cell mean; the correction books it with the remap's
    # own piecewise-linear profile (kernels/wdm6.cu, WDM6_PLM_SURFACE).
    wdm6_conservative_surface: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    # WB1 in classic Thompson (2.9.0, development branch) and aerosol-aware
    # Thompson (mp 28, 2.9.1): the cloud water the fallout takes out of the
    # lowest level is booked in its own total, CLOUDNC
    # (woof/core/thompson_cloud_booking.py), instead of vanishing.
    # Rain and the atmosphere are bit-identical with the key off.
    thompson_cloud_surface_booking: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    # WB5 in WSM3, WSM5 and WSM7 (2.9.1): module_mp_wsm3.F, module_mp_wsm5.F
    # and module_mp_wsm7.F carry their own copies of WSM6's nislfv_rain_plm
    # and its defect (the below-ground part of the straddling cell booked at
    # the flat cell mean); each correction books it with the remap's own
    # piecewise-linear profile (kernels/wsm3.cu, wsm5.cu, wsm7.cu,
    # WSM3/WSM5/WSM7_PLM_SURFACE).  More keys of finding WB5.
    wsm3_conservative_surface: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    wsm5_conservative_surface: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    wsm7_conservative_surface: bool = _correct_mode.CORRECT_KEYS_DEFAULT
    # HN1 (2.9.7): a hydrometeor mass handed over without its number
    # (prep from another model, woof downscale, the specified ring) gets
    # the number its mass gives under the child scheme's relations, and the
    # specified ring forces a number back with its mass
    # (woof/ingest/handover_numbers.py).  Off keeps real.exe's zeros and
    # WRF's tendency-only scalar ring.
    # Keyword-only, so the positional constructor is unchanged.
    hydrometeor_handover_numbers: bool = _config_field(
        default=_correct_mode.CORRECT_KEYS_DEFAULT, kw_only=True)
    # WRF's Pollard and three-dimensional PWP oceans. New PWP controls
    # are keyword-only so the existing positional constructor is stable.
    sf_ocean_physics: int = 0
    ocean_levels: int = _config_field(default=30, kw_only=True)
    ocean_analysis: str = _config_field(default="", kw_only=True)
    ocean_analysis_source: str = _config_field(default="rtofs", kw_only=True)
    ocean_mld_climatology: str = _config_field(default="woa18", kw_only=True)
    omdt: float = _config_field(default=1.0, kw_only=True)
    # Profiles are tuples: RunConfig is frozen and hashable (a list field
    # made hash(cfg) raise TypeError for every configuration); __post_init__
    # normalizes a list from TOML or a direct constructor call.
    ocean_z: tuple[float, ...] = _config_field(default=(), kw_only=True)
    ocean_t: tuple[float, ...] = _config_field(default=(), kw_only=True)
    ocean_s: tuple[float, ...] = _config_field(default=(), kw_only=True)
    oml_hml0: float = ocean_namelist_defaults()["oml_hml0"]
    oml_gamma: float = ocean_namelist_defaults()["oml_gamma"]
    oml_relaxation_time: float = ocean_namelist_defaults()["oml_relaxation_time"]

    # Appended after the existing selectors.
    # The chem block (smoke, GOCART-lite aerosols, CAMS-carried gases;
    # the smoke and air-quality design, section 1.5).  Every key is
    # inert while ``chem_sets`` is empty, which is the default: no table is
    # read, nothing is allocated and no launch runs, so a default run is the
    # engine it was.  :func:`validate_chem_config` is the law.
    #: The chem door: the species sets this run carries, by name (files under
    #: woof/data/chem/sets/), comma-separated in order ("smoke,gocart_lite").
    #: Empty = chem off.  A STRING, not a tuple: RunConfig is echoed into
    #: JSON receipts, checkpoint headers and identity documents that are
    #: compared with the live config, and a tuple comes back from JSON as a
    #: list and would never match itself.  A TOML array is accepted by both
    #: loaders and joined.  woof.chem_table.chem_names parses it.
    # WRF v4.6.1 Registry default for cugd_avedx (Registry.EM_COMMON:2543),
    # "number of grid boxes over which subsidence is spread".  Read only
    # where cu_physics = 5 (Grell 3D): WRF's cumulus driver hands it to
    # G3DRV and conv_grell_spread3d and to nothing else
    # (module_cumulus_driver.F:1250, :1640).  1 is no spreading; > 1 turns
    # on G3's high-resolution arm (3 x 3 forcing average, 3 x 3 subsidence
    # spreading).  No frozen configuration selects 5, so appending it
    # cannot move a frozen trajectory.
    cugd_avedx: int = _config_field(default=1, kw_only=True)
    maxiens: int = _config_field(default=1, kw_only=True)
    maxens: int = _config_field(default=3, kw_only=True)
    maxens2: int = _config_field(default=3, kw_only=True)
    maxens3: int = _config_field(default=16, kw_only=True)
    ensdim: int = _config_field(default=144, kw_only=True)
    cu_diag: int = _config_field(default=0, kw_only=True)
    convtrans_avglen_m: float = _config_field(default=30.0, kw_only=True)
    # --- New SAS (cu_physics = 96) namelist knobs --------------------------
    # Keyword-only, after G3's block and ahead of the chem and fire blocks
    # (which stay last).  Both are WRF's own
    # &physics keys at WRF's own Registry defaults: nsas_dx_factor
    # (Registry.EM_COMMON:2493, scope 1, default 0 = the fixed
    # trigger/closure velocities; 1 = the 250/dx scaling of
    # module_cu_nsas.F:567-576) and sas_pgcon (:2651, default 0.55, the
    # convectively forced pressure-gradient factor cu_nsas passes as pgcon).
    # nsas_dx_factor is read at 96; sas_pgcon is shared by 94 and 96.
    # Per-control reader sets drive validation and identity pruning.
    # shalconv is read only at 94. Defaults disappear from other schemes
    # so their existing configuration and restart identities stay fixed.
    nsas_dx_factor: int = _config_field(default=0, kw_only=True)
    sas_pgcon: float = _config_field(default=0.55, kw_only=True)
    #: CU_SAS optional ARW shallow switch; WRF defaults to 1.
    shalconv: int = _config_field(default=1, kw_only=True)

    # WRF's scalar Thompson cache controls. They change startup IO only;
    # the column kernels always consume the verified native table words.
    force_read_thompson: bool = _config_field(default=False, kw_only=True)
    write_thompson_tables: bool = _config_field(default=True, kw_only=True)
    write_thompson_mp38table: bool = _config_field(default=False, kw_only=True)
    chem_sets: str = ""
    #: External chem sources enabled for this run, by name (files under
    #: woof/data/chem/sources/), comma-separated in priority order.
    chem_sources: str = ""
    #: WRF ``&dynamics chem_adv_opt`` (Registry.EM_COMMON:2881, default 1):
    #: 1 = positive-definite final RK stage, 2 = WRF's monotonic
    #: ``advect_scalar_mono`` (woof.core.chem_advect_mono); 0, 3 and 4 are
    #: refused at the door.
    chem_adv_opt: int = 1
    #: WRF ``&dynamics chem_mix2_off``/``chem_mix6_off``
    #: (Registry.EM_COMMON:2888, :2894): switch the 2nd-order horizontal
    #: mixing / 6th-order filter off for the chem array only.
    chem_mix2_off: bool = False
    chem_mix6_off: bool = False
    #: WRF ``&chem chemdt`` (registry.chem:3778), minutes; 0 = every step.
    chemdt: float = 0.0
    #: WRF ``&chem kemit`` (registry.chem:3780, default 9): the emission
    #: layer count.
    kemit: int = 9
    #: WRF ``&chem biomass_burn_opt``/``plumerisefire_frq``
    #: (registry.chem:3853-3854; minutes).
    biomass_burn_opt: int = 0
    plumerisefire_frq: int = 180
    #: WRF ``&chem dust_opt``/``seas_opt``/``dmsemis_opt``
    #: (registry.chem:3844, :3851, :3850).
    dust_opt: int = 0
    seas_opt: int = 0
    dmsemis_opt: int = 0
    #: WRF ``&chem wetscav_onoff`` (registry.chem:3822): <0 large-scale
    #: washout (wetdep_ls), 1 aqueous scavenging (refused: needs a mechanism).
    wetscav_onoff: int = 0
    #: WRF ``&chem chem_conv_tr`` (registry.chem:3816, default 1): subgrid
    #: convective transport of chem.  Read only with a cumulus scheme on.
    chem_conv_tr: int = 1
    #: WRF ``&chem vertmix_onoff`` (registry.chem:3832, default 1).
    vertmix_onoff: int = 1
    #: WRF ``&chem aer_ra_feedback``/``aer_op_opt`` (registry.chem:3863-3864).
    aer_ra_feedback: int = 0
    aer_op_opt: int = 1
    #: WRF AFWA dust tuning (registry.chem:3887-3890).
    dust_alpha: float = 1.0
    dust_gamma: float = 1.0
    dust_smtune: float = 1.0
    dust_ustune: float = 1.0
    #: WRF ``&chem mynn_chem_vertmx`` (registry.chem:3793).
    mynn_chem_vertmx: bool = False
    #: ArWen: fire emission timing.  ``trailing_24h_dcycle`` (the default,
    #: HRRR-Smoke's method: the previous 24 h of fire detections with a
    #: diurnal cycle, woof.core.chem_fire.TRAILING_MODE),
    #: ``daily_mean_dcycle`` (RRFS ebb_dcycle 2) or ``observed_hourly``
    #: (hindcast only).  ``persistence_hourly`` is retired (RETIRED_MODES).
    fire_emission_mode: str = "trailing_24h_dcycle"
    #: ArWen: plume-rise fire properties, ``frp`` (HRRR-Smoke) or ``landuse``
    #: (WRF-Chem plumerise_driver).
    plume_fire_properties: str = "frp"
    #: ArWen: aerosol to aerosol-aware Thompson. ``auto`` diagnoses numbers
    #: when chemistry has a composition IC/BC source; ``none`` opts out.
    aerosol_mp_coupling: str = "auto"
    #: WRF ``&chem simple_dir_fdb`` (HRRR v4.1.21 fork, registry.smoke): 1 =
    #: the transported smoke's layer AOD (smoke/module_add_emiss_burn.F:194,
    #: 1e-6 x 4.5 m2/g x ug/kg x dry density x dz) joins the aer_opt = 3
    #: shortwave AOD, MIN(3, layer AOD) (phys/module_radiation_driver.F:
    #: 979-984), at every radiation call.  Keyword-only so the positional
    #: API is unchanged.
    simple_dir_fdb: int = _config_field(default=0, kw_only=True)
    #: HRRR v4.1.21 fork ``&chem enh_vermix`` (smoke/dry_dep_driver.F:
    #: 178-195): the chem vertical mixing's eddy diffusivity is raised over
    #: burning columns (FRP > 1 MW: levels kts+1..kts+4 to MAX(1.1 K, 10);
    #: diurnal FRP > 250 MW: kts+1..kts+5 to MAX(1.1 K, 20)).
    enh_vermix: bool = _config_field(default=False, kw_only=True)
    #: HRRR v4.1.21 fork ``&chem flam_part`` (smoke/module_plumerise1.F:141):
    #: the flaming fraction of the fire emission lifted into the plume
    #: (the rest stays in the lowest layer), constant over every burning
    #: column.  -1 keeps the GSL plume's own fraction (0.85 / 0.9 by FRP,
    #: woof/core/kernels/chem_plumerise.cu).
    flam_part: float = _config_field(default=-1.0, kw_only=True)
    #: HRRR v4.1.21 fork ``&chem aer_drydep_opt``: 111 is the smoke dry
    #: deposition velocity (smoke/dry_dep_driver.F:244-256, ust/500 with the
    #: unstable and humid corrections, capped at 0.5 m/s) as the lower
    #: boundary of the chem vertical mixing, for aerosol rows with no dry
    #: deposition scheme of their own.  0 = those rows deposit nothing.
    aer_drydep_opt: int = _config_field(default=0, kw_only=True)
    # The WRF v4.7.1 fire (SFIRE) namelist, appended keyword-only so the
    # positional API and the order of every field above are unchanged.
    # Names, types and defaults are woof.sfire_config.FireRunFields'
    # (tests/test_sfire_config_fields.py holds the two equal).
    _: KW_ONLY
    # Independent opt-in TC bundle. Defaults are omitted from stock identities.
    tc_flux_curve: str = "off"
    tc_spray: str = "off"
    tc_spray_partition: float = 1.
    tc_wave_roughness: bool = False
    tc_wave_source: str = "fetch"
    tc_fetch_m: float = 100000.0
    tc_horizontal_length_m: float = 0.0
    nfmc: int = 5
    fmoist_run: bool = False
    fmoist_interp: bool = False
    fmoist_only: bool = False
    fmoist_freq: int = 0
    fmoist_dt: float = 600.0
    fmep_decay_tlag: float = 999999.0
    ifire: int = 0
    fire_boundary_guard: int = 8
    fire_num_ignitions: int = 0
    fire_ignition_ros1: float = 0.01
    fire_ignition_start_lon1: float = 0.0
    fire_ignition_start_lat1: float = 0.0
    fire_ignition_end_lon1: float = 0.0
    fire_ignition_end_lat1: float = 0.0
    fire_ignition_radius1: float = 0.0
    fire_ignition_start_time1: float = 0.0
    fire_ignition_end_time1: float = 0.0
    fire_ignition_ros2: float = 0.01
    fire_ignition_start_lon2: float = 0.0
    fire_ignition_start_lat2: float = 0.0
    fire_ignition_end_lon2: float = 0.0
    fire_ignition_end_lat2: float = 0.0
    fire_ignition_radius2: float = 0.0
    fire_ignition_start_time2: float = 0.0
    fire_ignition_end_time2: float = 0.0
    fire_ignition_ros3: float = 0.01
    fire_ignition_start_lon3: float = 0.0
    fire_ignition_start_lat3: float = 0.0
    fire_ignition_end_lon3: float = 0.0
    fire_ignition_end_lat3: float = 0.0
    fire_ignition_radius3: float = 0.0
    fire_ignition_start_time3: float = 0.0
    fire_ignition_end_time3: float = 0.0
    fire_ignition_ros4: float = 0.01
    fire_ignition_start_lon4: float = 0.0
    fire_ignition_start_lat4: float = 0.0
    fire_ignition_end_lon4: float = 0.0
    fire_ignition_end_lat4: float = 0.0
    fire_ignition_radius4: float = 0.0
    fire_ignition_start_time4: float = 0.0
    fire_ignition_end_time4: float = 0.0
    fire_ignition_ros5: float = 0.01
    fire_ignition_start_lon5: float = 0.0
    fire_ignition_start_lat5: float = 0.0
    fire_ignition_end_lon5: float = 0.0
    fire_ignition_end_lat5: float = 0.0
    fire_ignition_radius5: float = 0.0
    fire_ignition_start_time5: float = 0.0
    fire_ignition_end_time5: float = 0.0
    fire_ignition_start_x1: float = 0.0
    fire_ignition_start_y1: float = 0.0
    fire_ignition_end_x1: float = 0.0
    fire_ignition_end_y1: float = 0.0
    fire_ignition_start_x2: float = 0.0
    fire_ignition_start_y2: float = 0.0
    fire_ignition_end_x2: float = 0.0
    fire_ignition_end_y2: float = 0.0
    fire_ignition_start_x3: float = 0.0
    fire_ignition_start_y3: float = 0.0
    fire_ignition_end_x3: float = 0.0
    fire_ignition_end_y3: float = 0.0
    fire_ignition_start_x4: float = 0.0
    fire_ignition_start_y4: float = 0.0
    fire_ignition_end_x4: float = 0.0
    fire_ignition_end_y4: float = 0.0
    fire_ignition_start_x5: float = 0.0
    fire_ignition_start_y5: float = 0.0
    fire_ignition_end_x5: float = 0.0
    fire_ignition_end_y5: float = 0.0
    fire_lat_init: float = 0.0
    fire_lon_init: float = 0.0
    fire_ign_time: float = 0.0

    #: WRF 4.7.1 fog settling: 0 off, 1 Duynkerke, 2 FogDES.
    grav_settling: int = 0
    #: WRF 4.7.1 KF trigger: 1 vertical velocity, 2 moisture advection, 3 RH.
    kfeta_trigger: int = 1
    fire_shape: int = 0
    fire_sprd_mdl: int = 1
    fire_crwn_hgt: float = 15.0
    fire_ext_grnd: float = 50.0
    fire_ext_crwn: float = 50.0
    fire_sfc_flx: int = 0
    fire_heat_peak: float = 0.0
    fire_tg_ub: float = 1000.0
    fire_smk_scheme: int = 0
    fire_smk_peak: float = 0.0
    fire_smk_ext: float = 50.0
    fire_wind_height: float = 6.096
    fire_fuel_read: int = -1
    fire_fuel_cat: int = 1
    fire_fmc_read: int = 1
    fire_print_msg: int = 0
    fire_print_file: int = 0
    fire_fuel_left_method: int = 1
    fire_fuel_left_irl: int = 2
    fire_fuel_left_jrl: int = 2
    fire_grows_only: int = 1
    fire_upwinding: int = 9
    fire_upwind_split: int = 0
    fire_viscosity: float = 0.4
    fire_lfn_ext_up: float = 1.0
    fire_topo_from_atm: int = 1
    fire_advection: int = 1
    fire_test_steps: int = 0
    fire_const_time: float = -1.0
    fire_const_grnhfx: float = 0.0
    fire_const_grnqfx: float = 0.0
    fire_atm_feedback: float = 1.0
    fire_mountain_type: int = 0
    fire_mountain_height: float = 500.0
    fire_mountain_start_x: float = 100.0
    fire_mountain_start_y: float = 100.0
    fire_mountain_end_x: float = 100.0
    fire_mountain_end_y: float = 100.0
    delt_perturbation: float = 0.0
    xrad_perturbation: float = 0.0
    yrad_perturbation: float = 0.0
    zrad_perturbation: float = 0.0
    hght_perturbation: float = 0.0
    stretch_grd: bool = True
    stretch_hyp: bool = False
    z_grd_scale: float = 0.4
    sfc_full_init: bool = False
    sfc_lu_index: int = 28
    sfc_tsk: float = 285.0
    # WRF Registry.EM_COMMON:2562,2575; check_a_mundo.F:2306-2312.
    sfc_tmn: float = 285.0
    fire_read_lu: bool = False
    fire_read_tsk: bool = False
    fire_read_tmn: bool = False
    fire_read_atm_ht: bool = False
    fire_read_fire_ht: bool = False
    fire_read_atm_grad: bool = False
    fire_read_fire_grad: bool = False
    sfc_vegfra: float = 0.5
    sfc_canwat: float = 0.0
    sfc_ivgtyp: int = 18
    sfc_isltyp: int = 7
    fire_lsm_reinit: bool = True
    fire_lsm_reinit_iter: int = 1
    fire_upwinding_reinit: int = 4
    fire_is_real_perim: bool = False
    fire_lsm_band_ngp: int = 4
    fire_lsm_zcoupling: bool = False
    fire_lsm_zcoupling_ref: float = 50.0
    fire_tracer_smoke: float = 0.02
    fire_viscosity_bg: float = 0.4
    fire_viscosity_band: float = 0.5
    fire_viscosity_ngp: int = 2
    fire_slope_factor: float = 1.0
    fs_array_maxsize: int = 100000
    fs_firebrand_gen_lim: int = 0
    fs_firebrand_gen_dt: int = 5
    fs_firebrand_gen_levels: int = 5
    fs_firebrand_gen_maxhgt: int = 50
    fs_firebrand_gen_levrand: bool = False
    fs_firebrand_gen_levrand_seed: int = 1
    fs_firebrand_gen_mom3d_dt: int = 4
    fs_firebrand_gen_prop_diam: float = 10.0
    fs_firebrand_gen_prop_effd: float = 10.0
    fs_firebrand_gen_prop_temp: float = 900.0
    fs_firebrand_gen_prop_tvel: float = 0.0
    fs_firebrand_dens: float = 513000.0
    fs_firebrand_dens_char: float = 299000.0
    fs_firebrand_max_life_dt: int = 200
    fs_firebrand_land_hgt: float = 0.15
    fuel_crosswalk: bool = False
    trackember: bool = False
    sr_x: int = 0
    sr_y: int = 0
    fire_static: str = ""
    fire_fuel_namelist: str = ""
    fire_smoke: bool = False
    # Ocean door TOML; empty preserves uncoupled configurations.
    ocean_config: str = ""
    waves: str = "off"
    waves_options: str = ""

    # Auto enables hail diagnostics on convection-allowing moist grids.
    hailcast_opt: int = -1
    haildt: float = 60.0

    # Prognostic zero-layer ice defaults on in domains containing ocean ice.
    seaice_prognostic: bool = True
    seaice_initial_thickness: float = 1.0  # declared prior, not an analysis
    seaice_new_ice_thickness: float = 0.5
    seaice_ocean_heat_flux: float = 2.0  # W m-2, positive ocean to ice

    # Automatic eclipse dimming; 0 is the undimmed counterfactual.
    ra_sw_eclipse: int = 1

    # Observed lake state, table-driven and default-on where covered.
    lake_state_source: str = "auto"
    lake_state_path: str = ""
    lake_state_accept_licence: str = ""

    # APPENDED LAST, on the config-freeze discipline.
    #: WRF ``&physics seaice_threshold`` (Registry.EM_COMMON, real,
    #: default 100 K).  A real.exe (initialisation) rule only, never read by
    #: the model: share/module_soil_pre.F adjust_for_seaice_post (HRRR
    #: v4.1.21 fork :245-246, :341-342) turns every WATER cell whose input
    #: skin temperature is below it into sea ice -- ice land use and soil
    #: category, LANDMASK 1, TMN 271.4 K, the ice soil column -- whatever
    #: its analysed XICE says.  Under ``fractional_seaice = 1`` the cell's
    #: XICE is left as analysed (possibly 0); under 0 it is set to 1.  The
    #: default 100 K converts nothing, so every run before this field is
    #: unchanged; HRRR v4 runs 271.4.
    seaice_threshold: float = 100.0

    # HRRR v4 output (development branch: e55b68accb, bd4a1c2090,
    # f80ac46aeb), APPENDED LAST and keyword-only.  The lane first inserted
    # these six after nwp_diagnostics, in the positional block: that shifted
    # every positional RunConfig field after it and moved the pre-mosaic
    # checkpoint identity (tests/test_noah_mosaic_config.py, 2.9.4 pass
    # c4a7d932).  Every default is off, so no earlier run changes.
    # WRF precipitation accumulation windows, minutes (0 disables).
    prec_acc_dt: float = 0.0
    prec_acc_dt1: float = 0.0
    gsd_diagnostics: int = 0
    reset_interval1: int = 60
    # Optional second history cadence, seconds from simulation start.
    history_interval2_s: float = 0.0
    history_interval_change_s: float = 0.0

    #: Per-run capacity decision, retained through preparation and receipts.
    smoke_off_reason: str = _config_field(default="", kw_only=True)
    # HRRR v4 climate statistics (development branch 3aec2abd31), appended
    # last and keyword-only so the positional constructor is unchanged.
    # Registry.EM_COMMON:2084,2087-2088; windows in minutes.
    diag_print: int = _config_field(default=0, kw_only=True)
    output_diagnostics: int = _config_field(default=0, kw_only=True)
    diag_int: int = _config_field(default=15, kw_only=True)
    wind_int: int = _config_field(default=5, kw_only=True)
    # Archive nest-integrated precipitation in a fixed frame across moves
    # (tc/nest); False reproduces the historical parent-seeded diagnostic
    # counters.  Appended last, keyword-only: it sat positionally after
    # `nested` and shifted every later positional argument.
    nest_accumulation_archive: bool = _config_field(default=True, kw_only=True)
    # WRF &physics ghg_input: 0 = fixed gases (CO2 a function of year),
    # 1 = time-varying CO2/N2O/CH4/CFC-11/CFC-12 from the CAMtr SSP245
    # table (woof.core.camtr_gases), re-interpolated at every radiation
    # call as WRF's RRTMG and RRTM longwave wrappers do; one tree-wide
    # value, inert on a domain whose radiation reads no gases (as in WRF).  The WRF v4 Registry
    # default is 1; the namelist importer writes 1 for a namelist named as
    # V4 that omits the key.  The engine's own default stays 0 so
    # configurations authored before the key keep their gases.  Appended
    # last, keyword-only (2.9.10 fix batch 2): it sat positionally after
    # o3input and shifted every later positional argument; 0 is absent from
    # checkpoint and document identities (woof.checkpoint_identity).
    ghg_input: int = _config_field(default=0, kw_only=True)
    #: Constant Cartesian domain translation (m/s); see docs/moving-frame.md.
    umove: float = _config_field(default=0.0, kw_only=True)
    vmove: float = _config_field(default=0.0, kw_only=True)
    # Optional turbine forcing. Keyword-only preserves every existing positional
    # RunConfig constructor; inactive settings are absent from checkpoint identity.
    wind_farm_scheme: str = _config_field(default="off", kw_only=True)
    wind_farms: str = _config_field(default="", kw_only=True)
    wind_farm_table: str = _config_field(default="", kw_only=True)
    windfarm_tke_factor: float = _config_field(default=0.25, kw_only=True)
    windfarm_ij: int = _config_field(default=0, kw_only=True)
    windfarm_wake_model: int = _config_field(default=2, kw_only=True)
    windfarm_overlap_method: int = _config_field(default=4, kw_only=True)
    windfarm_induction: int = _config_field(default=0, kw_only=True)
    windfarm_deg: float = _config_field(default=0.0, kw_only=True)
    wind_farm_max_dx_m: float = _config_field(default=3000.0, kw_only=True)
    # Native WRF station diagnostics. Empty path preserves prior outputs.
    tslist: str = _config_field(default="", kw_only=True)
    ts_buf_size: int = _config_field(default=200, kw_only=True)
    max_ts_locs: int = _config_field(default=5, kw_only=True)
    max_ts_level: int = _config_field(default=15, kw_only=True)
    tslist_unstagger_winds: bool = _config_field(default=False, kw_only=True)
    # WRF V3.5's native analytic scheme; distinct from modern Fitch tables.
    windturbines_spec: str = _config_field(default="none", kw_only=True)
    td_turbgridid: int = _config_field(default=-1, kw_only=True)
    td_hubheight: float = _config_field(default=100.0, kw_only=True)
    td_diameter: float = _config_field(default=60.0, kw_only=True)
    td_stdthrcoef: float = _config_field(default=0.158, kw_only=True)
    td_cutinspeed: float = _config_field(default=4.0, kw_only=True)
    td_cutoutspeed: float = _config_field(default=27.0, kw_only=True)
    td_power: float = _config_field(default=2.0, kw_only=True)
    td_turbpercell: float = _config_field(default=1.0, kw_only=True)
    td_ewfx: int = _config_field(default=0, kw_only=True)
    td_ewfy: int = _config_field(default=0, kw_only=True)
    td_pwfx: int = _config_field(default=1, kw_only=True)
    td_pwfy: int = _config_field(default=1, kw_only=True)
    # HAILCAST generation (woof.core.hail.HAILCAST_KERNELS).  "wrf_39_noaa":
    # the operational HRRR v4.1.21 branch's HAILCAST (WRFV3.9), what NOAA's
    # HRRR posts as surface HAIL; "wrf_461": WRF v4.6.1's, whose 2019-2022
    # ventilation and melting changes melt most stones before they reach the
    # ground.  Output-only (it reads the state and writes its own fields);
    # keyword-only, appended last, absent from documents and checkpoints at
    # its default (woof.checkpoint_identity.HAILCAST_DEFAULT_RUN_KEYS).
    hailcast_version: str = _config_field(default="wrf_39_noaa", kw_only=True)
    # WRF carries theta_m = theta * (1 + Rv/Rd*qv) when selected.
    # Engine default stays dry; WRF-file defaults are resolved at import.
    use_theta_m: int = _config_field(default=0, kw_only=True)
    # Native turbulence companions. Other operators require separate ports.
    # Keyword-only additions preserve all existing positional constructors.
    tke_adv_opt: int = _config_field(default=1, kw_only=True)
    sfs_opt: int = _config_field(default=0, kw_only=True)
    m_opt: int = _config_field(default=0, kw_only=True)
    tke_mix2_off: bool = _config_field(default=False, kw_only=True)
    tke_mix6_off: bool = _config_field(default=False, kw_only=True)
    # WRF Registry/registry.avgflx, per-domain history-window flux means.
    do_avgflx_em: int = _config_field(default=0, kw_only=True)
    do_avgflx_cugd: int = _config_field(default=0, kw_only=True)
    # WRF KF-CuP controls, consumed only by cu_physics=10. Appended and
    # keyword-only to preserve existing positional RunConfig constructors.
    numbins: int = _config_field(default=1, kw_only=True)
    thbinsize: float = _config_field(default=1.0, kw_only=True)
    rbinsize: float = _config_field(default=1.0, kw_only=True)
    mindeepfreq: float = _config_field(default=1.0, kw_only=True)
    minshallowfreq: float = _config_field(default=1.0, kw_only=True)
    shallowcu_forced_ra: bool = _config_field(default=False, kw_only=True)

    # WRF GCE switches; keyword-only preserves existing positional callers.
    gsfcgce_hail: int = _config_field(default=0, kw_only=True)
    gsfcgce_2ice: int = _config_field(default=0, kw_only=True)

    # Append after every E field so earlier positional constructors remain valid.
    # The preceding KW_ONLY marker also keeps this new selector keyword-only.
    sst_skin: int = 0
    sst_update: int = 0
    surface_analysis: str = ""
    tmn_update: int = 0
    lagday: int = 150
    ua_phys: bool = False

    # WRF 4.7.1 irrigation packages: channel=1, drip=2, sprinkler=3.
    sf_surf_irr_scheme: int = 0
    irr_daily_amount: float = 0.0
    irr_start_hour: int = 0
    irr_num_hours: int = 0
    irr_start_julianday: int = 0
    irr_end_julianday: int = 0
    irr_freq: int = 1
    irr_ph: int = 0

    # WRF 4.7.1 prescribed sea-ice controls (Registry.EM_COMMON:2636-2642).
    seaice_albedo_opt: int = 0
    seaice_snowdepth_opt: int = 0
    seaice_thickness_opt: int = 0
    seaice_thickness_default: float = 3.0
    seaice_snowdepth_min: float = 0.001
    seaice_snowdepth_max: float = 1.0e10


    # WRF GFDL CO2 initialization, consumed only by radiation 99.
    co2tf: int = _config_field(default=1, kw_only=True)
    # WRF &physics ifsnow (Registry default 1): snow-cover effects in the
    # slab scheme (sf_surface_physics=1), which caps the skin and the top
    # soil layer at 273.16 K under snow.  Read by woof/core/physics.py
    # _run_slab; nonzero enables snow caps (WRF-exact retains WRF behavior).
    ifsnow: int = _config_field(default=1, kw_only=True)
    # WRF aerosol coupling; keyword-only preserves positional constructors.
    aercu_opt: int = _config_field(default=0, kw_only=True)
    # BMJ cloud feedback is opt-in and does not shift positional arguments.
    bmj_rad_feedback: bool = _config_field(default=False, kw_only=True)
    # CAM cache timing must not shift the historical positional fields.
    cam_abs_freq_s: float = _config_field(default=21600.0, kw_only=True)

    #: WRF ``&physics mfshconv`` (Registry.EM_COMMON:2485, default 1): the
    #: EDMF shallow mass flux of module_bl_mfshconvpbl.F, run before the
    #: QNSE PBL by module_pbl_driver.F:1519-1536.  Read under
    #: bl_pbl_physics = 4 only: WRF resets it to 0 under every other PBL
    #: (share/module_check_a_mundo.F:1715-1730), and so it is inert there
    #: (:func:`qnse_mfshconv_active`).  The Registry default is the default
    #: here, so a QNSE run that names nothing runs WRF's mass flux.
    #: Appended last.
    mfshconv: int = _config_field(default=1, kw_only=True)
    aercu_fct: float = _config_field(default=1.0, kw_only=True)
    kf_edrates: int = _config_field(default=0, kw_only=True)

    afwa_bad_data_check: int = _config_field(default=0, kw_only=True)
    afwa_diag_opt: int = _config_field(default=0, kw_only=True)
    afwa_therm_opt: int = _config_field(default=0, kw_only=True)
    afwa_ptype_opt: int = _config_field(default=0, kw_only=True)
    afwa_vil_opt: int = _config_field(default=0, kw_only=True)
    afwa_radar_opt: int = _config_field(default=0, kw_only=True)
    afwa_vis_opt: int = _config_field(default=0, kw_only=True)
    afwa_icing_opt: int = _config_field(default=0, kw_only=True)
    afwa_buoy_opt: int = _config_field(default=0, kw_only=True)
    afwa_severe_opt: int = _config_field(default=0, kw_only=True)
    afwa_ptype_ccn_tmp: float = _config_field(default=264.15, kw_only=True)
    afwa_ptype_tot_melt: float = _config_field(default=50.0, kw_only=True)

    bucket_mm: float = _config_field(default=-1., kw_only=True)
    bucket_j: float = _config_field(default=-1., kw_only=True)
    acc_phy_tend: int = _config_field(default=0, kw_only=True)

    # Native level output is opt-in and appended keyword-only.
    p_lev_diags: int = _config_field(default=0, kw_only=True)
    z_lev_diags: int = _config_field(default=0, kw_only=True)
    p_lev_diags_dfi: int = _config_field(default=0, kw_only=True)
    z_lev_diags_dfi: int = _config_field(default=0, kw_only=True)
    press_levels: tuple[float, ...] = _config_field(default=(), kw_only=True)
    z_levels: tuple[float, ...] = _config_field(default=(), kw_only=True)
    use_tot_or_hyd_p: int = _config_field(default=2, kw_only=True)
    extrap_below_grnd: int = _config_field(default=1, kw_only=True)
    p_lev_missing: float = _config_field(default=-999., kw_only=True)
    z_lev_missing: float = _config_field(default=-999., kw_only=True)
    auxhist22_interval_s: float = _config_field(default=0., kw_only=True)
    auxhist22_begin_s: float = _config_field(default=0., kw_only=True)
    auxhist22_end_s: float = _config_field(default=0., kw_only=True)
    auxhist22_outname: str = _config_field(default="auxhist22_d<domain>_<date>", kw_only=True)
    frames_per_auxhist22: int = _config_field(default=1, kw_only=True)
    auxhist23_interval_s: float = _config_field(default=0., kw_only=True)
    auxhist23_begin_s: float = _config_field(default=0., kw_only=True)
    auxhist23_end_s: float = _config_field(default=0., kw_only=True)
    auxhist23_outname: str = _config_field(default="auxhist23_d<domain>_<date>", kw_only=True)
    frames_per_auxhist23: int = _config_field(default=1, kw_only=True)
    p_lev_interval_s: float = _config_field(default=0., kw_only=True)
    z_lev_interval_s: float = _config_field(default=0., kw_only=True)

    # Startup-only WRF DFI controls. Stop offsets retain the analysis's UTC
    # reference; they do not shift the requested forecast end time.
    dfi_opt: int = _config_field(default=0, kw_only=True)
    dfi_savehydmeteors: int = _config_field(default=0, kw_only=True)
    dfi_nfilter: int = _config_field(default=7, kw_only=True)
    dfi_cutoff_seconds: int = _config_field(default=3600, kw_only=True)
    dfi_bckstop_seconds: int = _config_field(default=0, kw_only=True)
    dfi_fwdstop_seconds: int = _config_field(default=0, kw_only=True)
    dfi_write_filtered_input: bool = _config_field(default=True, kw_only=True)
    dfi_write_dfi_history: bool = _config_field(default=False, kw_only=True)
    time_step_dfi: int = _config_field(default=-1, kw_only=True)
    moist_adv_dfi_opt: int = _config_field(default=0, kw_only=True)
    dfi_constant_bc: bool = _config_field(default=False, kw_only=True)
    dfi_time_dim: int = _config_field(default=1000, kw_only=True)

    #: WRF FARMS can replace the surface SWDOWN carrier, independently of heating.
    couple_farms: bool = _config_field(default=False, kw_only=True)
    cu_rad_feedback: bool = _config_field(default=False, kw_only=True)
    rrtmg_aer3_source: str = _config_field(default="operational_fork", kw_only=True)
    #: WRF solve_em: midpoint RK2 or the established three-stage RK3.
    #: Keyword-only so positional construction keeps its historical order.
    rk_ord: int = _config_field(default=3, kw_only=True)
    #: WRF standard momentum transport (1), or final-stage WENO (3).
    momentum_adv_opt: int = _config_field(default=1, kw_only=True)

    # Stock WRF module_diag_cl uses auxhist3's interval in minutes. Zero
    # retains the operational fork's explicitly configured diag/wind path.
    auxhist3_interval: int = _config_field(default=0, kw_only=True)
    solar_diagnostics: int = _config_field(default=0, kw_only=True)
    mean_diag: int = _config_field(default=0, kw_only=True)
    diurnal_diag: int = _config_field(default=0, kw_only=True)
    mean_freq: int = _config_field(default=0, kw_only=True)
    mean_interval: int = _config_field(default=0, kw_only=True)


    # WRF observation FDDA; native Registry defaults, inert when off.
    obs_nudge_opt: int = 0
    max_obs: int = 0
    fdda_start: float = 0.0
    fdda_end: float = 0.0
    obs_nudge_wind: int = 0
    obs_coef_wind: float = 0.0
    obs_nudge_temp: int = 0
    obs_coef_temp: float = 0.0
    obs_nudge_mois: int = 0
    obs_coef_mois: float = 0.0
    obs_nudge_pstr: int = 0
    obs_coef_pstr: float = 0.0
    obs_no_pbl_nudge_uv: int = 0
    obs_no_pbl_nudge_t: int = 0
    obs_no_pbl_nudge_q: int = 0
    obs_sfc_scheme_horiz: int = 0
    obs_sfc_scheme_vert: int = 0
    obs_max_sndng_gap: float = 20.0
    obs_nudgezfullr1_uv: float = 50.0
    obs_nudgezrampr1_uv: float = 50.0
    obs_nudgezfullr2_uv: float = 50.0
    obs_nudgezrampr2_uv: float = 50.0
    obs_nudgezfullr4_uv: float = -5000.0
    obs_nudgezrampr4_uv: float = 50.0
    obs_nudgezfullr1_t: float = 50.0
    obs_nudgezrampr1_t: float = 50.0
    obs_nudgezfullr2_t: float = 50.0
    obs_nudgezrampr2_t: float = 50.0
    obs_nudgezfullr4_t: float = -5000.0
    obs_nudgezrampr4_t: float = 50.0
    obs_nudgezfullr1_q: float = 50.0
    obs_nudgezrampr1_q: float = 50.0
    obs_nudgezfullr2_q: float = 50.0
    obs_nudgezrampr2_q: float = 50.0
    obs_nudgezfullr4_q: float = -5000.0
    obs_nudgezrampr4_q: float = 50.0
    obs_nudgezfullmin: float = 50.0
    obs_nudgezrampmin: float = 50.0
    obs_nudgezmax: float = 3000.0
    obs_sfcfact: float = 1.0
    obs_sfcfacr: float = 1.0
    obs_dpsmx: float = 7.5
    obs_rinxy: float = 0.0
    obs_rinsig: float = 0.0
    obs_twindo: float = 0.0
    obs_npfi: int = 0
    obs_ionf: int = 1
    obs_idynin: int = 0
    obs_dtramp: float = 0.0
    obs_prt_max: int = 1000
    obs_prt_freq: int = 1000
    obs_ipf_in4dob: bool = False
    obs_ipf_errob: bool = False
    obs_ipf_nudob: bool = False
    obs_ipf_init: bool = True
    obs_scl_neg_qv_innov: int = 0
    obs_inputfile: str = ""
    obs_domain_id: int = 1
    obs_parent_grid_ratio: int = 1
    obs_levelidn: int = 0
    obs_truelat1: float = 0.0
    obs_truelat2: float = 0.0
    obs_stand_lon: float = 0.0

    # Native PX controls, Registry.EM_COMMON:2615-2616,2747.
    # Keyword-only additions preserve earlier positional constructors.
    pxlsm_smois_init: int = _config_field(default=1, kw_only=True)
    pxlsm_modis_veg: int = _config_field(default=0, kw_only=True)
    pxlsm_soil_nudge: int = _config_field(default=0, kw_only=True)
    sgfdda_interval_m: int = _config_field(default=180, kw_only=True)
    sgfdda_inname: str = _config_field(default="sgfdda_d<domain>", kw_only=True)
    io_form_sgfdda: int = _config_field(default=2, kw_only=True)
    # NSSL coefficients append keyword-only; legacy positional construction is unchanged.
    nssl_cccn: float = _config_field(default=0.5e9, kw_only=True)
    nssl_alphah: float = _config_field(default=0.0, kw_only=True)
    nssl_alphahl: float = _config_field(default=1.0, kw_only=True)
    nssl_icdx: int = _config_field(default=6, kw_only=True)
    nssl_icdxhl: int = _config_field(default=6, kw_only=True)
    # Native latitude/longitude hydrostatic topology. Keyword-only additions
    # preserve every legacy positional constructor and checkpoint identity.
    polar: bool = _config_field(default=False, kw_only=True)
    fft_filter_lat: float = _config_field(default=45.0, kw_only=True)
    # WRF 4.7.1 radiation-driver fixed-step phase (Registry.EM_COMMON:2573).
    ra_call_offset: int = _config_field(default=0, kw_only=True)
    madwrf_opt: int = _config_field(default=0, kw_only=True)
    madwrf_cldinit: int = _config_field(default=0, kw_only=True)
    madwrf_dt_relax: float = _config_field(default=60.0, kw_only=True)
    madwrf_dt_nudge: float = _config_field(default=60.0, kw_only=True)
    # 0-1, 0-2 and 0-3 km AGL updraft-helicity maxima over each history
    # interval (UP_HELI_MAX01/02/03, woof/core/uh_layers.py), folded beside
    # UP_HELI_MAX by the nwp_diagnostics = 1 step.  Appended last,
    # keyword-only and off by default, so every earlier run is unchanged.
    uh_layers: int = _config_field(default=0, kw_only=True)

@dataclass(frozen=True)
class _DfiBackwardRunConfig(RunConfig):
    """Internal WRF DFI_BCK phase, built from an admitted forecast config.

    This type is never constructed by a document or namelist loader. WRF
    deliberately sets diff_opt=0, km_opt=1 and negative dt in this phase.
    Ordinary configurations keep their existing diffusion admission.
    """

#: The ways a domain's clock meets the measured terrain rules
#: (:attr:`RunConfig.terrain_clock`).
TERRAIN_CLOCK_MODES: tuple[str, ...] = ("measured", "pinned", "local_face")

#: The restart doors' accuracy about the changed default, beside the
#: existing refusal (the mismatch already names the breakage: a run whose
#: clock can differ from the checkpoint's).  Said when a checkpoint
#: written under "measured" (an absent key) meets a run under
#: "local_face", because from 2.8.8 a configuration that leaves
#: terrain_clock unset reads "local_face".
TERRAIN_CLOCK_RESTART_BREAK_NOTICE = (
    "note: the checkpoint was integrated under the measured terrain clock "
    "(terrain_clock = \"measured\", the default before 2.8.8) and this run "
    "reads terrain_clock = \"local_face\", the default from 2.8.8, which "
    "can give a domain a longer step. To resume this checkpoint, write "
    "terrain_clock = \"measured\" explicitly; otherwise restart from t = 0.")

#: The restart doors' accuracy about the 2.9.0 correct-mode default, beside
#: the existing refusal (the mismatch already names the breakage: a WSM6 or
#: New Tiedtke trajectory continued under other water-budget arithmetic).
#: Said when a checkpoint written in exact mode (an absent key, the only
#: arithmetic before 2.9.0) meets a run whose scheme reads a correct-mode key
#: that is on, because from 2.9.0 a configuration that leaves the keys unset
#: reads physics.mode = "correct".
CORRECT_MODE_RESTART_BREAK_NOTICE = (
    "note: the checkpoint was integrated with the matched (exact) "
    "arithmetic for its scheme (every release before 2.9.0, and 2.9.0 for "
    "WSM3, WSM5 and WSM7, which its correct mode did not reach) and this "
    "run reads the correct-mode default, whose water budgets differ. To "
    "resume this checkpoint, set every correct-mode key its "
    "scheme reads to false explicitly (wsm6_conservative_surface, "
    "wdm6_conservative_surface, wsm3_conservative_surface, "
    "wsm5_conservative_surface, wsm7_conservative_surface, "
    "ntiedtke_dry_mass_coupling, hydrometeor_handover_numbers); otherwise "
    "restart from t = 0.")

#: The same accuracy for the 2.8.9 WRF-folder default (D8): a fixed-step
#: WRF namelist now imports with terrain_clock = "pinned", WRF's own
#: acoustic count, so a checkpoint integrated under the terrain clock from
#: an earlier import of the same folder meets a run whose count can differ.
TERRAIN_CLOCK_PINNED_RESTART_NOTICE = (
    "note: the checkpoint was integrated under the terrain clock, and this "
    "run reads terrain_clock = \"pinned\": from 2.8.9 a fixed-step WRF "
    "namelist runs the acoustic substep count wrf.exe runs. To resume this "
    "checkpoint, put ! gpuwm-physics-selectors-v1: "
    "{\"terrain_clock\":\"local_face\"} (or the clock the checkpoint names) "
    "first in namelist.input; otherwise restart from t = 0.")


G3_RUN_DEFAULTS = {"cugd_avedx": 1, "maxiens": 1, "maxens": 3,
                   "maxens2": 3, "maxens3": 16, "ensdim": 144,
                   "cu_diag": 0, "convtrans_avglen_m": 30.0}


def validate_g3_dimensions(maxiens, maxens, maxens2, maxens3, ensdim, avedx=1):
    """WRF's explicit member references determine the minimum array sizes."""
    for name, value in zip(("maxiens", "maxens", "maxens2", "maxens3", "ensdim"),
                           (maxiens, maxens, maxens2, maxens3, ensdim)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if maxens < 3:
        raise ValueError("maxens must be at least 3: massflx_stats reads cap member 3")
    if maxens3 < 16:
        raise ValueError("maxens3 must be at least 16: cup_forcing_ens_3d writes member 16")
    if maxens2 < 2 or (avedx > 1 and maxens2 < 3):
        raise ValueError("maxens2 must hold edtc member 2, and edtc member 3 when cugd_avedx > 1")
    if ensdim < maxiens * maxens * maxens2 * maxens3:
        raise ValueError("ensdim must hold maxiens*maxens*maxens2*maxens3: the closure writes would exceed XF_ENS/PR_ENS")


def drop_inert_g3_fields(values: dict) -> None:
    """Scheme-only fields cannot change unrelated configuration identities."""
    if values.get("cu_physics", 0) != 5:
        for name in G3_RUN_DEFAULTS:
            if name == "convtrans_avglen_m" and values.get("cu_physics") == 3 and values.get("cu_rad_feedback"):
                continue
            values.pop(name, None)


#: WRF vertical advection orders the kernels carry (module_advect_em.F
#: ``vert_order`` branches 2 through 6).
VERTICAL_ADV_ORDERS = (2, 3, 4, 5, 6)


def advection_order_refusal(*, v_sca_adv_order, v_mom_adv_order,
                            h_mom_adv_order) -> str | None:
    """Why the advection orders cannot run as set, or None.

    Shared by :func:`validate_run_config` and the namelist importer.
    The breakage each refusal prevents: a vertical order the kernels do
    not carry would silently run the 3rd-order ladder under a namelist
    that says otherwise, and a horizontal momentum order other than 5
    would run WRF's flux5 under a namelist that asked for flux3/flux6.
    """
    for name, value in (("v_sca_adv_order", v_sca_adv_order),
                        ("v_mom_adv_order", v_mom_adv_order)):
        if isinstance(value, bool) or value not in VERTICAL_ADV_ORDERS:
            return (f"{name} = {value!r}: WRF's vertical flux ladders "
                    "accept integer orders 2 through 6; another value "
                    "does not select a native advection operator.")
    if isinstance(h_mom_adv_order, bool) or h_mom_adv_order not in (0, 2, 3, 4, 5, 6):
        return (f"h_mom_adv_order = {h_mom_adv_order!r}: the u, v and w "
                "horizontal stencils carry WRF orders 0, 2, 3, 4, 5 and 6 only "
                "(woof/core/kernels/advection.cu), so another order would "
                "run flux5 under a namelist that asked for something else.")
    return None


#: Canonical Noah-MP defaults, with the evidence for each identity.
#: Additional accepted values live in NOAHMP_OPTION_VALUES below.
#:
#: Read the second column as the evidence, not as a slogan.  "fixture" means
#: the four whole-column ``noahmp-sflx.csv`` cases were generated with that
#: value and the port is bitwise against them; "dead-proved" means the option
#: kills a code region and the kill is proved in
#: :mod:`woof.core.noahmp_sflx`'s module docstring out of the source rather
#: than assumed; "unmeasured" means no other value has been tried at all.
#: Evidence for an alternative is stated explicitly in its row.
NOAHMP_OPTION_IDENTITY_EVIDENCE: dict[str, tuple[object, str]] = {
    "dveg": (4, "fixture; prescribed table vegetation by default. "
                "Values 1/3/7/8/9 also have real whole-module replay and forecast "
                "proof; dynamic 2/5/6 run native CARBON/CO2FLUX and persistent pools"),
    "opt_crs": (1, "fixture; Ball-Berry default. Jarvis CANRES/CALHUM has a real "
                   "WRF whole-module replay and forecast proof"),
    "opt_btr": (1, "fixture; Noah by default. CLM (2) and SSiB (3) have real WRF "
                   "whole-module replays"),
    "opt_run": (3, "fixture; Schaake96.  Kills GROUNDWATER, "
                   "SHALLOWWATERTABLE, ZWTEQ, VIC, XAJ and DVIC -- "
                   "dead-proved"),
    "opt_sfc": (1, "fixture; SFCDIF1 default. Native SFCDIF2 reaches both flux "
                   "tiles and has a real whole-module replay"),
    "opt_frz": (1, "fixture; NY06 supercooled liquid"),
    "opt_inf": (1, "fixture; NY06 frozen-soil permeability"),
    "opt_rad": (3, "fixture; gap = 1 - FVEG"),
    "opt_alb": (2, "fixture; CLASS default. BATS (1) has a real WRF "
                   "whole-module replay. SNICAR (3) is written from papers, not bitwise to WRF; "
                   "layered adding-doubling, generated optics and aging, four grain shapes and glacier ice"),
    "snicar_snowshape_opt": (3, "paper reconstruction; four He grain habits"),
    "snicar_snowbc_intmix": (True, "paper reconstruction; He internal BC fit"),
    "snicar_snowdust_intmix": (False, "paper reconstruction; He internal dust fit"),
    "opt_snf": (1, "fixture; Jordan91 default. Values 2-5 have real "
                   "whole-module replay and forecast proof, plus complete "
                   "native ATM threshold/species checks"),
    "opt_tbot": (2, "fixture; Noah lower boundary at ZBOT"),
    "opt_stc": (1, "fixture; semi-implicit snow/soil temperature"),
    "opt_gla": (1, "default glacier path; GLA2 has a strict real native glacier "
                   "module replay, complete Greenland forecast and resident lifecycle proof"),
    "opt_rsf": (1, "fixture; Sakaguchi/Zeng by default. Values 2/3/4 have "
                   "real WRF whole-module replay and forecast proof"),
    "opt_soil": (1, "fixture; uniform category by default. Depth categories and "
                    "native SR2006 pedotransfer reach the resident parameter gather"),
    "opt_pedo": (1, "native PEDOTRANSFER_SR2006 when opt_soil=3; inactive with "
                    "table-category soils"),
    "opt_crop": (0, "inactive default; CROP1 executes native crop calendar, "
                    "phenology and carbon growth with strict real replay and forecast proof; "
                    "GECROS2 is opt-in: native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF; "
                    "measured active-crop step cost is 1.4283x CROP1 on a shared GPU"),
    "opt_irr": (0, "inactive default; native triggers 1/2/3 carry resident "
                   "amounts, counters and complete irrigation water/energy coupling"),
    "opt_irrm": (0, "inactive with IRR0; mapped fractions at 0 and complete "
                    "sprinkler, micro/drip and flood procedures at 1/2/3"),
    "opt_infdv": (0, "inactive outside RUN8; methods 1/2/3 implement the complete "
                     "native Philip, Green-Ampt and Smith-Parlange infiltration"),
    "opt_tdrn": (0, "complete native simple and Hooghoudt tile drainage at 1/2; inactive when run is not 3 or mapped fraction is below the native threshold"),
    "soiltstep": (0.0, "native decimated soil clock and nine persisted ACC arrays; "
                       "zero retains the original per-step default"),
    "noahmp_output": (1, "native output level 3 carries the complete water and energy accumulator block"),
    "noahmp_acc_dt": (0.0, "native accumulator reset interval in minutes; zero leaves accumulation inactive"),
}

#: Derived for compatibility with all four admission doors. Every current
#: option now has a consumer or a concrete retained-branch refusal.
NOAHMP_OPTIONS_WITHOUT_CONSUMER: frozenset[str] = frozenset(
    name for name, (_value, why) in NOAHMP_OPTION_IDENTITY_EVIDENCE.items()
    if why.startswith("declared only")
)

#: The enforced form of the table above: field -> the only accepted value.
NOAHMP_OPTION_IDENTITY: dict[str, object] = {
    name: value for name, (value, _why) in
    NOAHMP_OPTION_IDENTITY_EVIDENCE.items()
}

# Alternatives with a native real-column replay. Keep the canonical identity
# above unchanged: it is also the default and the default restart seal.
NOAHMP_OPTION_VALUES: dict[str, tuple[int, ...]] = {
    "opt_crop": (0,1,2),
    "opt_gla": (1,2),
    "noahmp_output": (0,1,2,3),
    "opt_tdrn": (0,1,2),
    "opt_irr": (0,1,2,3),
    "opt_irrm": (0,1,2,3),
    "opt_soil": (1, 2, 3, 4),
    "opt_sfc": (1, 2),
    "opt_crs": (1, 2),
    "snicar_snowshape_opt": (1, 2, 3, 4),
    "snicar_snowbc_intmix": (True, False),
    "snicar_snowdust_intmix": (True, False),
    "opt_inf": (1, 2),
    "opt_run": (1, 2, 3, 4, 5, 6, 7, 8),
    "opt_infdv": (0, 1, 2, 3),
    "opt_rad": (1, 2, 3),
    "opt_frz": (1, 2),
    "opt_stc": (1, 2, 3),
    "opt_tbot": (1, 2),
    "dveg": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    "opt_btr": (1, 2, 3),
    "opt_rsf": (1, 2, 3, 4),
    "opt_alb": (1, 2, 3),
    "opt_snf": (1, 2, 3, 4, 5),
}


#: The MYNN PBL option identity woof admits, field -> the only accepted
#: value.  ``validate_run_config`` refuses anything else *before* a run
#: starts, so the driver's own refusal is a second line rather than the
#: first. Options with multiple implemented values are validated below.
MYNN471_OPTION_VALUES = {
    "bl_mynn_closure": (2.5, 2.6, 3.0), "bl_mynn_cloudpdf": (0, 1, 2),
    "bl_mynn_edmf": (0, 1), "bl_mynn_edmf_mom": (0, 1),
    "bl_mynn_edmf_tke": (0, 1), "bl_mynn_cloudmix": (0, 1),
    "bl_mynn_mixqt": (0, 1), "bl_mynn_output": (0, 1), "icloud_bl": (0, 1),
}
MYNN_PBL_OPTION_IDENTITY: dict[str, object] = {
    "bl_mynn_closure": 2.6,
    "bl_mynn_cloudpdf": 2,
    "bl_mynn_edmf": 1,
    "bl_mynn_edmf_mom": 1,
    "bl_mynn_edmf_tke": 0,
    # bl_mynn_mixscalars left this pin at the W4 full admission (mf-close2
    # Stage B): it is validated by its own block in validate_run_config --
    # {0,1}; 1 mixes when the microphysics carries the qn family and is
    # inert otherwise (mynn_mixscalars_active below).
    "bl_mynn_cloudmix": 1,
    "bl_mynn_mixqt": 0,
    "bl_mynn_output": 0,
    "icloud_bl": 1,
}


#: Microphysics schemes whose state carries the WHOLE qn family this
#: port's mixscalars solve runs (nc/ni/nwfa/nifa, qnbca as an exact zero):
#: aerosol-aware Thompson (28) and aerosol-aware TEMPO (900, whose row
#: transports nc/ni/nwfa/nifa beside ng/volg; NOAA MPAS 1d6e370e
#: mpas_atmphys_todynamics.F:394-405 couples MYNN's nc/nifa/nwfa rates
#: for mp_tempo with config_tempo_aerosolaware, the only TEMPO value the
#: engine carries).  The solve admits the key with all five of WRF's flags
#: true or not at all (mynn_pbl_gpu.py qn_flags; the anchored oracle
#: fixture family was generated at exactly that combo).
MYNN_QN_FAMILY_SCHEMES: tuple[int, ...] = (28, 29, 38, 900)

#: Which number species each MYNN mixing key couples, by state name.  The
#: family solve runs every qn species whenever any key is on; a species no
#: selected key names keeps WRF's skipped-solve answer, a zero tendency
#: (``qn2 = qn``).  Every solve reads only its own column and plume flux,
#: so dropping one changes no other species' rate by a bit.
#: ``bl_mynn_mixscalars`` keeps its WRF v4.6.1 meaning (the whole qn
#: family, module_bl_mynn.F:4654-4860); the other two are the MYNN-EDMF
#: split (see RunConfig.bl_mynn_mixaerosols).  A new key is a row here.
MYNN_QN_MIX_KEYS: dict[str, tuple[str, ...]] = {
    "bl_mynn_mixscalars": ("nc", "ni", "nwfa", "nifa"),
    "bl_mynn_mixnumcon": ("nc", "ni"),
    "bl_mynn_mixaerosols": ("nwfa", "nifa"),
}

#: State species name -> WRF qn flag name, for the partial-scheme refusal.
_MYNN_QN_STATE_TO_FLAG: dict[str, str] = {
    "nc": "qnc", "ni": "qni", "nwfa": "qnwfa", "nifa": "qnifa",
}

#: The number species WRF v4.6.1's MYNN mixes under ``bl_mynn_mixscalars=1``
#: for each ported microphysics selector: the scalar members of the
#: scheme's Registry package that one of mynn_tendencies' five flag-gated
#: solves reads (``bl_mynn_mixscalars > 0 .AND. FLAG_QNI`` at :4654, qnc
#: :4695, qnwfa :4736, qnifa :4778, qnbca :4820).  Registry.EM_COMMON
#: package lines: thompson (8) scalar:qni,qnr; milbrandt2mom (9)
#: qnc,qnr,qni,...; morr_two_moment (10) qni,...; wdm6scheme (16)
#: qnn,qnc,qnr; nssl_2mom (18) qni,... through nssl2mconc; thompsonaero
#: (28) qni,qnr,qnc,qnwfa,qnifa,qnbca; p3_1category (50) qni,...;
#: p3_1category_nc (51) qnc,qni.
#: p3_1cat_3mom (53) qnc,qni,qnr,qir,qib,qzi (:3041).  The
#: fast_khain_lynn_shpund (30, Registry/registry.sbm:318)
#: qnn,qnc,qnr,qni,qns,qng, so F_QNC and F_QNI are true for it.  The
#: schemes with no number moment (kessler 1, lin 2, fer_mp_hires 5, wsm3 3, wsm5 4,
#: wsm6 6, wsm6r 106, and 0) carry none: there every
#: flag is false, ``mynn_bl_driver`` hands the solves zero columns and the
#: key is WRF's no-op.  Any scheme not listed here is a new table row, not
#: a new code path.
MYNN_QN_FLAG_SPECIES: dict[int, tuple[str, ...]] = {
    2: (),  # Six Lin masses, no prognostic number species.
    4: (),  # Single-moment package carries no number concentrations.
    0: (), 1: (), 3: (), 5: (), 6: (), 13: (), 15: (), 24: (), 106: (),
    8: ("qni",), 9: ("qnc", "qni"), 10: ("qni",), 14: ("qnc",), 26: ("qnc",), 27: ("qnc",), 16: ("qnc",),
    18: ("qni",), 28: ("qnc", "qni", "qnwfa", "qnifa", "qnbca"),
    # WRF 4.7.1 Registry.EM_COMMON:3053 gives RCON the same number family.
    29: ("qnc", "qni", "qnwfa", "qnifa", "qnbca"),
    30: ("qnc", "qni"), 50: ("qni",),
    51: ("qnc", "qni"), 52: ("qnc", "qni"), 53: ("qnc", "qni"),
    # TEMPO aerosol-aware: f_nc/f_ni/f_nwfa/f_nifa true, no nbca (NOAA
    # MPAS 1d6e370e mpas_atmphys_driver_pbl.F:1119 stages nc/nifa/nwfa
    # together; the engine's TEMPO row carries no nbca scalar).
    900: ("qnc", "qni", "qnwfa", "qnifa"),
    38: ("qnc", "qni", "qnwfa", "qnifa", "qnbca"),
    # WRF 4.7.1 Registry.EM_COMMON:3027,3031,3047-3051.
    7: (), 11: ("qnc", "qni"), 40: ("qnc", "qni"),
    55: ("qni",), 56: ("qnc", "qni"), 95: (), 96: (), 97: (),
}


def _mynn_qn_requested_species(cfg) -> tuple[str, ...]:
    """State species the MYNN mixing keys ask for, in family order.

    The union over :data:`MYNN_QN_MIX_KEYS` of every key set to 1, before
    any scheme or PBL gate.  ``getattr`` defaults keep duck-typed
    configurations that predate the split keys reading as key-off.
    """
    wanted = set()
    for key, species in MYNN_QN_MIX_KEYS.items():
        if int(getattr(cfg, key, 0)) == 1:
            wanted.update(species)
    return tuple(name for name in MYNN_QN_MIX_KEYS["bl_mynn_mixscalars"]
                 if name in wanted)


def mynn_mixed_number_species(cfg) -> tuple[str, ...]:
    """The number species MYNN couples on this configuration (state names).

    Non-empty only under MYNN itself (bl_pbl_physics=5) with a scheme in
    ``MYNN_QN_FAMILY_SCHEMES``; then it is the union of the species each
    selected key names (:data:`MYNN_QN_MIX_KEYS`).  The family solve runs
    for every qn species and the coupling keeps only these, so the rate of
    a species no key selects is WRF's skipped-solve zero.
    """
    if int(cfg.bl_pbl_physics) != 5:
        return ()
    if getattr(cfg, "bl_mynn_version", "wrf_461") == "wrf_471":
        available = MYNN_QN_FLAG_SPECIES.get(int(cfg.mp_physics), ())
        return tuple(name for name in _mynn_qn_requested_species(cfg)
                     if _MYNN_QN_STATE_TO_FLAG[name] in available)
    if int(cfg.mp_physics) not in MYNN_QN_FAMILY_SCHEMES:
        return ()
    return _mynn_qn_requested_species(cfg)


def mynn_mixscalars_active(cfg) -> bool:
    """Whether any MYNN number-species mixing key mixes on this configuration.

    WRF v4.6.1 gates every scalar solve on the key AND the species flag
    (``phys/module_bl_mynn.F`` mynn_tendencies: ``bl_mynn_mixscalars > 0
    .AND. FLAG_QNI`` at :4654 and the same shape for qnc :4695, qnwfa
    :4736, qnifa :4778, qnbca :4820), and ``mynn_bl_driver`` zeroes each
    column whose flag is false.  MYNN-EDMF splits the key by family
    (``bl_mynn_mixnumcon``/``bl_mynn_mixaerosols``, MYNN_QN_MIX_KEYS).
    This port's solve runs the whole family or nothing, so mixing is
    active only under MYNN itself with a scheme in
    ``MYNN_QN_FAMILY_SCHEMES`` and at least one key on; the runtime
    stages the family, the preflight prices its storage and the
    every-step restart invariant applies in exactly that case.  Inactive
    is one of two things, which ``validate_run_config`` tells apart with
    ``MYNN_QN_FLAG_SPECIES``: a scheme WRF gives no number species, where
    the key is WRF's no-op and the run records it and mixes nothing; or a
    scheme WRF would mix in part, which is refused by name.
    """
    return bool(mynn_mixed_number_species(cfg))


def mynn_mixscalars_unported_species(cfg) -> tuple[str, ...]:
    """The number species WRF would mix under these keys that the port cannot.

    Empty when every key is off, when mixing is active (the scheme carries
    the whole family) and when the microphysics carries none of the
    requested species (WRF's no-op).  Non-empty names the concrete
    divergence the validator refuses: WRF runs the listed flag-gated
    solves on this scheme (Thompson's qni, WDM6's qnc, ...) and this
    port's solve cannot run on a state that lacks the rest of the family,
    so a run that recorded the key would claim a mixing it does not
    perform.
    """
    if int(cfg.bl_pbl_physics) != 5:
        return ()
    if getattr(cfg, "bl_mynn_version", "wrf_461") == "wrf_471" or int(cfg.mp_physics) in MYNN_QN_FAMILY_SCHEMES:
        return ()
    requested = {_MYNN_QN_STATE_TO_FLAG[name]
                 for name in _mynn_qn_requested_species(cfg)}
    return tuple(name for name in MYNN_QN_FLAG_SPECIES.get(
        int(cfg.mp_physics), ()) if name in requested)


def mynn_mixscalars_driver_value(cfg) -> int:
    """The ``bl_mynn_mixscalars`` the MYNN driver is handed.

    The fixture-anchored driver admits 1 only with all five qn flags true
    (the solves it then runs are exactly WRF's).  The inert case -- key 1,
    a microphysics with no number species -- is WRF's five skipped solves
    on zeroed columns, which leave every tendency the driver exports
    exactly as the key-0 path does, so the driver is handed 0 and runs
    that path bit for bit.  The partial case never reaches the driver
    (``validate_run_config`` refuses it, ``mynn_mixscalars_unported_species``).
    """
    return 1 if mynn_mixscalars_active(cfg) else 0


#: The RUC option identity woof admits, field -> (only accepted value, what
#: pins it).  Same contract as the Noah-MP block above: ``validate_run_config``
#: refuses anything else BEFORE a run starts, and
#: :func:`woof.core.physics.PhysicsDriver._run_ruc` reads every one off the
#: configuration by name so the registry's citation of that file is true.
#:
#: The mosaic switches are validated separately as 0/1, and spp_lsm by
#: validate_spp_config. The remaining control needs soil-adjustment input
#: state the runtime does not carry. Other restrictions are in
#: RUC_RUNTIME_RESTRICTIONS.
RUC_OPTION_IDENTITY_EVIDENCE: dict[str, tuple[object, str]] = {
    "flag_sm_adj": (0, "no consumer; share/module_soil_pre.F:2063 reads this "
                       "inside init_soil_ruc, i.e. in real.exe, to adjust a "
                       "Noah-derived RUC soil state.  woof has no RUC soil "
                       "ingest for it to adjust, so the knob is pinned at 0 "
                       "to keep it that way rather than accepted and "
                       "ignored"),
}

#: The enforced form of the table above: field -> the only accepted value.
RUC_OPTION_IDENTITY: dict[str, object] = {
    name: value for name, (value, _why) in
    RUC_OPTION_IDENTITY_EVIDENCE.items()
}


#: The scalars WRF v4.6.1's ``scalar_pblmix`` diffuses for each ported
#: microphysics selector, by their woof state names.  diff4d
#: (phys/module_pbl_driver.F:2668-2671) runs over every member of the
#: ``scalar`` array except QNS, QNR, QNG, QNH, QT, QVOLG and QKE_ADV, so a
#: row is the scheme's Registry.EM_COMMON scalar package less those:
#: thompson (8) qni,qnr -> qni; milbrandt2mom (9) qnc,qnr,qni,qns,qng,qnh ->
#: qnc,qni; morr_two_moment (10) qni,qns,qnr,qng -> qni (its nc is
#: diagnostic, not a scalar member); wdm6scheme (16) qnn,qnc,qnr ->
#: qnn,qnc; nssl_2mom (18, nssl2mconc + nssl_ccn_opt + nssl_hailvol)
#: qndrop,qnr,qni,qns,qng,qnh,qnn,qvolg,qvolh -> qndrop,qni,qnn,qvolh;
#: thompsonaero (28) -> qnc,qni,qnwfa,qnifa (the engine carries no qnbca);
#: p3_1category (50) qni,qnr,qir,qib -> qni,qir,qib; TEMPO (900)
#: nc,nr,ni,ng,volg,nwfa,nifa -> nc,ni,nwfa,nifa.  The packages with no
#: scalar member, and Ferrier (5), whose one scalar QT diff4d skips, mix
#: nothing: WRF's diff4d loop runs and changes no field.  A scheme with no
#: row (Fast-SBM's bin scalars) is refused until it has one; a new row is
#: table work, not a new code path.
SCALAR_PBLMIX_SPECIES: dict[int, tuple[str, ...]] = {
    0: (), 1: (), 2: (), 3: (), 4: (), 5: (), 6: (), 24: (), 106: (),
    8: ("ni",), 9: ("nc", "ni"), 10: ("ni",), 16: ("nn", "nc"),
    18: ("qndrop", "qni", "qnn", "qvolh"),
    28: ("nc", "ni", "nwfa", "nifa"),
    38: ("nc", "ni", "nwfa", "nifa"),
    50: ("ni", "qir", "qib"),
    900: ("nc", "ni", "nwfa", "nifa"),
}


def scalar_pblmix_species(cfg) -> tuple[str, ...]:
    """The state fields ``scalar_pblmix=1`` diffuses for ``cfg``'s microphysics."""
    return SCALAR_PBLMIX_SPECIES[int(cfg.mp_physics)]


def scalar_pblmix_exchange(cfg) -> dict | None:
    """The registry's EXCH_H producer row for ``cfg``'s PBL scheme, or None."""
    from woof.physics_registry import consumer_row_for_selector
    return consumer_row_for_selector(
        "pbl", "scalar_diffusion_exchange", cfg.bl_pbl_physics)


def scalar_pblmix_mixes(cfg) -> bool:
    """Whether scalar_pblmix=1 runs WRF's post-PBL diff4d for ``cfg``.

    False for no PBL and SASE (the registry row declares them no-ops, as
    WRF does not call its PBL driver without a PBL scheme).
    """
    if cfg.scalar_pblmix != 1:
        return False
    row = scalar_pblmix_exchange(cfg)
    return bool(row and row["writes_exch_h"])


def validate_scalar_pblmix_consumer(cfg) -> None:
    """scalar_pblmix=1 is admitted where its diff4d tendencies are carried.

    WRF runs diff4d on the selected scheme's EXCH_H after any PBL scheme
    (module_pbl_driver.F:2251-2259, v4.6.1); the registry row
    ``scalar_diffusion_exchange`` names each producer.  The solve carries
    each microphysics' own diffused scalars (:data:`SCALAR_PBLMIX_SPECIES`);
    only a scheme with no row there is refused, since WRF would diffuse
    scalars this run does not know it carries.  Until U-E4 every scheme but
    mp=28 was refused here, which refused every microphysics choice on the
    HRRR suite.

    Shared with plan review (woof.physics_registry), which calls it on the
    resolved settings so a plan is not offered that this refuses.
    """
    if cfg.scalar_pblmix != 1:
        return
    row = scalar_pblmix_exchange(cfg)
    if row is None:
        raise UnsupportedConfiguration(
            f"scalar_pblmix=1 with bl_pbl_physics={cfg.bl_pbl_physics}: the "
            "registry declares no EXCH_H producer row for that scheme; "
            "running it would omit the requested vertical-mixing tendencies.")
    if row["writes_exch_h"] and int(cfg.mp_physics) not in SCALAR_PBLMIX_SPECIES:
        raise UnsupportedConfiguration(
            f"scalar_pblmix=1 with mp_physics={cfg.mp_physics}: WRF diff4d "
            "would mix that package's nonprecipitating scalars on the PBL "
            "scheme's exch_h, and woof.config.SCALAR_PBLMIX_SPECIES has no "
            "row naming them for this scheme, so the run would omit those "
            "vertical-mixing tendencies.")


def validate_mynn_generation_spp(cfg) -> None:
    """The stochastic MYNN kernels exist for the WRF v4.6.1 generation only.

    Shared with plan review (woof.physics_registry) like
    :func:`validate_scalar_pblmix_consumer`.
    """
    if cfg.bl_mynn_version == "gsd_41" and cfg.spp_pbl == 1:
        raise ConfigurationRefused(
            "bl_mynn_version='gsd_41' with spp_pbl=1: the stochastic MYNN "
            "kernels are specialised from the WRF v4.6.1 source only, so "
            "the gsd_41 rows (dew flux, mixing length) would be absent from "
            "every perturbed step while the run claimed them.")


def validate_spp_config(cfg) -> None:
    """Require an actual physics consumer for each requested SPP pattern."""
    consumers = {
        "spp_conv": (("cu_physics", 3),),
        "spp_pbl": (("bl_pbl_physics", 5), ("sf_sfclay_physics", 5)),
        "spp_lsm": (("sf_surface_physics", 3),),
    }
    for name, required in consumers.items():
        value = getattr(cfg, name)
        if type(value) is not int or value not in (0, 1):
            raise ConfigurationRefused(f"{name} must be integer 0 or 1, got {value!r}")
        if value:
            for selector, expected in required:
                if getattr(cfg, selector) != expected:
                    raise ConfigurationRefused(
                        f"{name}=1 requires {selector}={expected}; the selected "
                        "scheme has no consumer for this parameter pattern")


# --------------------------------------------------------------------------
# WRF selector schema.
#
# These tables answer only "is this a WRF selector value woof's schema
# knows, with a coherent soil geometry".  Whether the scheme is EXECUTABLE is
# a separate question owned by woof/physics_compat.py, which runs first in
# validate_run_config and emits the complete fail-closed port receipt.
# Keeping the two apart is what makes an eventual admission a one-line change
# in physics_compat rather than a scatter of numeric literals here, and it
# stops an in-port scheme collapsing onto a generic "must be 0 or 2" message
# that says nothing about what is actually missing.
# --------------------------------------------------------------------------

#: WRF sf_sfclay_physics values in woof's schema.
SURFACE_LAYER_SCHEMES = (0, 1, 2, 3, 4, 5, 7, 10, 89, 91)
#: WRF sf_surface_physics values in woof's schema.
LAND_SURFACE_SCHEMES = (0, 1, 2, 3, 4, 7, 8)

#: WRF ``sf_urban_physics`` values in woof's schema: 0 none, 1 single-layer
#: UCM, 2 BEP, 3 BEP+BEM.
URBAN_PHYSICS_SCHEMES = (0, 1, 2, 3)
#: The model module each urban option runs, loaded by NAME
#: (:mod:`woof.core.urban_driver`), so an option whose module is absent
#: from this build is refused at the door rather than failing as an
#: ImportError on the first surface step.
URBAN_MODEL_MODULES = {1: "woof.core.urban_ucm", 2: "woof.core.urban_bep",
                       3: "woof.core.urban_bem"}
#: The land-surface schemes WRF couples an urban model to: Noah calls the
#: UCM inside its column loop and BEP/BEP_BEM after it
#: (module_sf_noahdrv.F:1317, :1603-1790); Noah-MP calls ``noahmp_urban``
#: from the surface driver (module_surface_driver.F:3184).  RUC and the
#: no-LSM path never call one.
URBAN_LAND_SURFACE_SCHEMES = (2, 4)
#: The PBL schemes WRF runs with BEP/BEP_BEM that woof has: YSU (1), MYJ
#: (2, through ``myjurb``), BouLac (8, with distributed BEP sources)
#: and K-epsilon (17, with the native blended surface-flux boundary).
#: is admitted on WRF's own terms, which differ from YSU's and MYJ's: its
#: driver receives the BEP source terms and reads none of them (it sets the
#: urban surface and volume fractions to one, module_bl_keps.F:287-291), so
#: the city reaches it as the grid-blended HFX, QFX and UST the BEP coupling
#: writes (HFX_URB + (1-f)*HFX_RURAL, module_sf_noahdrv.F:1739-1752) --
#: the column totals of the urban heat, moisture and momentum exchange,
#: deposited at the first level instead of distributed over the building
#: heights.  tools/keps_wrf461_oracle proves the unread terms word for word.
URBAN_BEP_PBL_SCHEMES = (1, 2, 8, 17)
#: WRF's only admitted ``num_urban_hi`` in this build.
URBAN_HEIGHT_BINS = 15


def urban_model_in_build(option: int) -> bool:
    """True when the model module for urban ``option`` ships in this build."""
    import importlib.util

    name = URBAN_MODEL_MODULES.get(int(option))
    if name is None:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


URBAN_WATER_BALANCE_DEFAULT = "wrf_480"
URBAN_WATER_BALANCE_GENERATIONS = {
    "wrf_471": {"code": 0, "algorithm": "urban-slucm-wrf-v4.7.1-v1"},
    "wrf_480": {"code": 1, "algorithm": "urban-slucm-wrf-v4.7.1-water-balance-v4.8.0-v1"},
}


def validate_urban_config(cfg) -> None:
    """The ``sf_urban_physics`` law, each refusal naming what it prevents.

    Everything here is inert at the default ``sf_urban_physics = 0``: the two
    companion keys are then read by nothing, so no value of them can make a
    run wrong and none is refused.
    """
    option = getattr(cfg, "sf_urban_physics", 0)
    generation = getattr(cfg, "urban_water_balance", URBAN_WATER_BALANCE_DEFAULT)
    if generation not in URBAN_WATER_BALANCE_GENERATIONS:
        raise ValueError(
            f"urban_water_balance={generation!r} must be one of "
            f"{tuple(URBAN_WATER_BALANCE_GENERATIONS)}: an unknown generation "
            "cannot decide whether weak rainfall suppresses evaporation.")
    lcz = getattr(cfg, "use_wudapt_lcz", 0)
    bins = getattr(cfg, "num_urban_hi", URBAN_HEIGHT_BINS)
    for name, value in (("sf_urban_physics", option),
                        ("use_wudapt_lcz", lcz), ("num_urban_hi", bins)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationRefused(
                f"{name} must be an integer, got {value!r}: WRF declares it "
                "an integer namelist key (Registry.EM_COMMON:2489, 2534).")
    if option not in URBAN_PHYSICS_SCHEMES:
        raise ConfigurationRefused(
            "sf_urban_physics must be 0 (none), 1 (single-layer UCM), 2 (BEP) "
            f"or 3 (BEP+BEM), WRF numbering, got {option}.")
    if option == 0:
        return
    generation = getattr(cfg, "urban_ucm_generation", "wrf_471")
    if generation not in ("wrf_471", "wrf_480"):
        raise ValueError("urban_ucm_generation must be wrf_471 or wrf_480")
    if generation != "wrf_471" and option != 1:
        raise ValueError("urban_ucm_generation=wrf_480 requires sf_urban_physics=1; the new BEP/BEM generations are not implemented")
    if generation == "wrf_480" and getattr(cfg, "sf_surface_mosaic", 0):
        raise ValueError("wrf_480 with sf_surface_mosaic=1: the composed Noah mosaic canopy kernel does not carry TVG/TVGL/SMG or vegetated-ground flux feedback")
    for name in ("urban_treeoption", "urban_ttscheme"):
        value = getattr(cfg, name, -1)
        if type(value) is not int or value not in (-1, 0, 1):
            raise ValueError(f"{name} must be -1 (table), 0 or 1")
    fvg = getattr(cfg, "urban_fvg", -1.0)
    if isinstance(fvg, bool) or not isinstance(fvg, (int, float)) or not (fvg == -1 or 0 <= fvg <= 1):
        raise ValueError("urban_fvg must be -1 (table) or between 0 and 1")
    if generation == "wrf_471" and (fvg != -1 or any(getattr(cfg, n, -1) != -1 for n in ("urban_treeoption", "urban_ttscheme"))):
        raise ValueError("vegetated-ground and tree controls require urban_ucm_generation=wrf_480; wrf_471 does not compute their fluxes")
    if lcz not in (0, 1):
        raise ConfigurationRefused(
            f"use_wudapt_lcz must be 0 (URBPARM.TBL, 3 urban types) or 1 "
            f"(URBPARM_LCZ.TBL, 11 LCZ types), got {lcz}: urban_param_init "
            "opens one of those two tables and nothing else.")
    if bins != URBAN_HEIGHT_BINS:
        raise ConfigurationRefused(
            f"num_urban_hi={bins}: the BEP/BEM kernels in this build index "
            f"HI_URB2D with WRF's default {URBAN_HEIGHT_BINS}-bin stride, and "
            "the gridded morphology (URB_PARAM) that alone could fill another "
            "bin count is not ingested, so any other count would read past "
            "the array the kernels were built for.")
    lsm = int(cfg.sf_surface_physics)
    if lsm not in URBAN_LAND_SURFACE_SCHEMES:
        raise ConfigurationRefused(
            f"sf_urban_physics={option} needs sf_surface_physics=2 (Noah) or "
            f"4 (Noah-MP), got sf_surface_physics={lsm}: WRF's RUC and no-LSM "
            "paths never call an urban model, so the selector would be "
            "silently ignored and city cells would run as bulk urban land. "
            "Select sf_surface_physics=2 or 4, or set sf_urban_physics=0.")
    if option in (2, 3):
        pbl = int(cfg.bl_pbl_physics)
        if pbl not in URBAN_BEP_PBL_SCHEMES:
            raise ConfigurationRefused(
                f"sf_urban_physics={option} (BEP{'+BEM' if option == 3 else ''})"
                f" needs bl_pbl_physics=1 (YSU), 2 (MYJ), 8 (BouLac) or 17 (K-epsilon), "
                f"got bl_pbl_physics={pbl}: BEP injects its drag, heat and "
                "TKE sources through the PBL implicit solve or, for "
                "K-epsilon, through the grid-blended surface fluxes. "
                "Other PBL schemes would drop them. Select "
                "bl_pbl_physics=1, 2, 8 or 17, or set sf_urban_physics=1 (the "
                "single-layer UCM feeds any PBL through its surface fluxes).")
    if not urban_model_in_build(option):
        raise ConfigurationRefused(
            f"sf_urban_physics={option}: option {option}'s kernel is not in "
            f"this build ({URBAN_MODEL_MODULES[option]} is absent), so no "
            "urban model would run and the selector would be silently "
            "ignored.")

#: ``cu_physics`` values whose scheme consumes the dycore's PURE ADVECTIVE
#: theta/qv forcing pair -- WRF's ``RTHFTEN``/``RQVFTEN``.
#:
#: A TABLE, not a branch.  The dycore's export, the state allocation, the
#: VRAM projection and the restart inventory all read this one set, so
#: admitting a further consumer (WRF's G3, GD or NTiedtke, each of which
#: takes the same two arguments from ``module_cumulus_driver.F``) is one
#: entry here and nothing else.
#:
#: Grell-Freitas (3) is the entry today.  Kain-Fritsch (1) is deliberately
#: absent: WRF's ``module_cumulus_driver.F`` KFETASCHEME arm passes no
#: RTHFTEN/RQVFTEN at all, so allocating the pair for a KF run would price
#: two full [nz, ny, nx] arrays nothing reads.
#:
#: THE FOLD TRAP, recorded where the table is: WRF's cumulus driver
#: pre-folds ``RTHRATEN + RTHBLTEN`` into ``RTHFTEN`` at
#: ``module_cumulus_driver.F:867`` for G3SCHEME and NTIEDTKESCHEME ONLY.
#: GFSCHEME is not in that list -- GF sums the advective, radiative and
#: boundary-layer lanes itself.  The dycore therefore exports PURE
#: ADVECTION, and any scheme added to this set that expects the pre-folded
#: form must do that fold in its own adapter rather than moving the
#: export, or it double-counts the heating GF must not see twice.
#
# 16 JOINS IT, AND THE FOLD STAYS IN THE ADAPTER. WRF's cumulus driver
# pre-folds RTHRATEN + RTHBLTEN into RTHFTEN at
# module_cumulus_driver.F:867 for G3SCHEME and NTIEDTKESCHEME only. GF is
# not in that list and sums the three lanes itself, so ArWen's dycore
# exports PURE ADVECTION. Moving the export to do the fold would
# double-count radiative and PBL heating for Grell-Freitas and move
# run_myj -- the baseline every intensity number is graded against. The
# fold therefore belongs in the New Tiedtke adapter and nowhere else.
#
# 5 JOINS IT ON THE SAME TERMS AS 16.  Grell 3D is the other scheme in
# WRF's fold list; woof/core/g3.py Grell3D does the fold in its adapter.
CUMULUS_ADVECTIVE_FORCING_SCHEMES = frozenset({3, 5, 6, 16, 93})


def consumes_advective_forcing(cfg) -> bool:
    """Shared allocation/export predicate, including KF's conditional reader."""
    return (cfg.cu_physics in CUMULUS_ADVECTIVE_FORCING_SCHEMES
            or (cfg.cu_physics == 1 and getattr(cfg, "kfeta_trigger", 1) == 2))


def validate_fog_kfeta_config(cfg) -> None:
    """Native fog and KF reader laws, shared by plan review and execution."""
    if type(cfg.grav_settling) is not int or cfg.grav_settling not in (0, 1, 2):
        raise ConfigurationRefused("grav_settling must be integer 0, 1 or 2")
    if cfg.grav_settling and cfg.mp_physics in (28, 29):
        raise ConfigurationRefused(
            "grav_settling>0 with mp_physics=28/29 is disabled by native WRF "
            "4.7.1: its microphysics already settles cloud water "
            "(share/module_check_a_mundo.F:2474-2487); set grav_settling=0")
    if type(cfg.kfeta_trigger) is not int or cfg.kfeta_trigger not in (1, 2, 3):
        raise ConfigurationRefused("kfeta_trigger must be integer 1, 2 or 3")
    if cfg.kfeta_trigger != 1 and cfg.cu_physics != 1:
        raise ConfigurationRefused(
            "kfeta_trigger=2/3 requires cu_physics=1: native WRF 4.7.1 "
            "only consumes these triggers in KF_eta_PARA "
            "(phys/module_cu_kfeta.F:991-1017; run/README.namelist:868)")

#: Cumulus schemes whose tendencies are theta, qv, qc and (where the moist
#: package carries QI) qi only -- no rain or snow category.  New Tiedtke
#: (cu_ntiedtke_post_run) and Grell-Devenyi (GRELLDRV routes cold
#: condensate to RQICUTEN at 258 K, module_cu_gd.F:350-366).
CUMULUS_QC_QI_SCHEMES = frozenset({5, 6, 16, 93})
#: ``bl_pbl_physics`` value selecting the SASE closure.
#:
#: SASE is an ArWen-only scheme -- a Scale-Adaptive Stress-Energetics
#: closure with no WRF v4.6.1 counterpart -- so it cannot take a WRF
#: number without claiming to be a transcription of something.  WRF's own
#: ``bl_pbl_physics`` namespace runs to 99 (99 = MRF), so a value above it
#: can never collide with a scheme WRF adds later, and it deliberately
#: falls outside :data:`woof.wrf461_compatibility.PBL_OPTIONS` so the
#: WRF PBL/surface-layer compatibility matrix -- which states what *WRF*
#: admits -- is not consulted about a scheme WRF does not have.  The
#: readiness authority is the registry entry and the constraints below.
SASE_PBL_SCHEME = 900
CCPP_TKE_EDMF_PBL_SCHEME = 901

#: Deepest column the SASE vertical solve accepts, read from the closure's
#: own authority module so the device define, the launcher's rejection and
#: this admission check cannot drift apart.  ``sase_ref`` is NumPy-only, so
#: importing it here costs the config layer no CuPy dependency.
SASE_MAX_NZ = _sase_limits.MAX_COLUMN_LEVELS

#: The SASE knobs that are fail-closed on their non-default value, each
#: with the default it is judged against: :func:`validate_run_config`
#: refuses a non-default value on a domain that does not run SASE.  The
#: domain-tree loader (woof.experiment) reads the same table, so a
#: ``[shared]`` value on a tree that mixes SASE with another PBL scheme
#: reaches the SASE domains and leaves the others on these defaults.
SASE_FAIL_CLOSED_DEFAULTS = {
    "sase_flux_diag": False,
    "sase_moist_n2": True,
    "sase_stable_dissipation": False,
}

#: The MYNN surface layer's selector.  No RULE reads it any more -- the
#: SASE pairing that did is admitted (the closure reads ust/hfx/qfx/wspd
#: and this layer publishes all four) -- but it is not dead vocabulary:
#: it is the spelling of this layer's number, and the three places that
#: dispatch on it (``woof/core/physics.py``: the MYNN surface result,
#: its sea-point twin, and the surface-field allocation) read it from
#: here rather than carrying a bare 5.
MYNN_SFCLAY_SCHEME = 5

#: WRF's Eta similarity surface layer (``sf_sfclay_physics``) and the MYJ
#: PBL (``bl_pbl_physics``).  They share the number 2 in WRF's two
#: namespaces and they are a PAIR, not two independently selectable
#: schemes: ``phys/module_physics_init.F:3770-3772`` fatals a MYJ PBL
#: whose surface layer did not set ``isfc = 2``, and the Eta surface layer
#: is the only ``sf_sfclay_physics`` value woof ports that does
#: (:3169).  :func:`validate_myj_pairing` is that law.
MYJ_SFCLAY_SCHEME = 2
MYJ_PBL_SCHEME = 2

#: WRF's GFS surface layer (``sf_sfclay_physics = 3``, GFSSFCSCHEME,
#: phys/module_sf_gfs.F), ported in binary64 and graded word for word
#: against tools/sfgfs_wrf461_oracle (woof/core/sf_gfs.py).  Like the Eta
#: layer it sets ``isfc = 2`` (module_physics_init.F:3186-3199), so WRF
#: admits it under the MYJ PBL; woof does not, for the structural reason
#: :func:`validate_myj_pairing` states.
GFS_SFCLAY_SCHEME = 3

#: WRF's PXSFCSCHEME (``Registry.EM_COMMON:3134``, ``pxsfcscheme
#: sf_sfclay_physics==7``): the Pleim-Xiu surface layer, ported from WRF
#: v4.6.1 ``phys/module_sf_pxsfclay.F`` with the fractional sea-ice wrapper
#: (woof/core/pxsfclay.py, graded word for word against
#: tools/pxsfclay_wrf461_oracle).  WRF initialises ``isfc = 7`` for it
#: (phys/module_physics_init.F:3153-3155), which YSU, Shin-Hong, MRF and GBM
#: (isfc must be 1), MYNN (5, 1 or 2) and MYJ/GFS (2) fatal; the ACM2 and
#: CAMUW PBLs and PBL-off accept it.  No new validator carries that law:
#: the YSU/MYNN/Shin-Hong cells are rows of
#: woof.wrf461_compatibility.PBL_SURFACE_LAYER_AUTHORITY (refused through
#: require_ready_wrf_physics), MYJ is refused by
#: :func:`validate_myj_pairing`, and UW and SASE read only UST/HFX/QFX
#: (/WSPD), which this layer publishes.
PX_SFCLAY_SCHEME = 7

#: WRF's CAMUWPBLSCHEME: the University of Washington moist-turbulence
#: PBL of CAM5 (Bretherton and Park 2009), ported from WRF v4.7.1
#: ``phys/module_bl_camuwpbl_driver.F`` and the CAM modules it calls
#: (woof/core/uwpbl.py, graded against tools/uwpbl_wrf471_oracle).  WRF
#: has no surface-layer pairing law for it (module_physics_init.F:3825-3832
#: carries only the BEP/BEM urban fatal): it reads UST, HFX and QFX, which
#: every ported surface layer publishes.  :func:`validate_uwpbl_config`.
UW_PBL_SCHEME = 9

#: WRF's GBMPBLSCHEME: the Grenier-Bretherton-McCaa moist TKE PBL, ported
#: from WRF v4.6.1 ``phys/module_bl_gbmpbl.F`` (woof/core/gbmpbl.py,
#: graded word for word against tools/gbmpbl_wrf461_oracle).  WRF's one
#: law for it is the surface-layer fatal at module_physics_init.F:3892-3893
#: (``isfc .ne. 1``): only the MM5 surface layers (1 and 91) set isfc = 1.
#: :func:`validate_gbmpbl_config`.
GBM_PBL_SCHEME = 12
#: The surface layers WRF admits under GBM (module_physics_init.F:3892).
GBM_PBL_SURFACE_LAYERS = (1, 91)
#: WRF's GFSSCHEME: the GFS nonlocal PBL (Hong and Pan 1996), ported from
#: WRF v4.6.1 ``phys/module_bl_gfs.F`` (woof/core/gfs_pbl.py, graded
#: against tools/gfs_pbl_wrf461_oracle).  It reads the surface layer's UST,
#: PSIM, PSIH, GZ1OZ0, WSPD and BR (module_pbl_driver.F:1416-1418) and forms
#: FM = GZ1OZ0 - PSIM, FH = GZ1OZ0 - PSIH from them.
#: :func:`validate_gfs_pbl_config` admits the surface layers that publish
#: that set; :data:`GFS_PBL_SURFACE_LAYERS` names them.
GFS_PBL_SCHEME = 3

#: Surface layers whose published set carries every word BL_GFS reads, with
#: the SF_GFS definitions (GZ1OZ0 = ln(z1/z0), PSIM/PSIH the integrated
#: stability corrections, so GZ1OZ0 - PSIM is the momentum denominator):
#: WRF v4.6.1 requires isfc == 2 (module_physics_init.F:3742). SF_GFS
#: meets that law and publishes the full boundary. MYJSFC omits these
#: stability fields, and GFDL is not implemented.
GFS_PBL_SURFACE_LAYERS = (GFS_SFCLAY_SCHEME,)

#: WRF's QNSEPBLSCHEME and QNSESFCSCHEME (Quasi-Normal Scale Elimination;
#: ``bl_pbl_physics = 4``, ``sf_sfclay_physics = 4``).  Like MYJ's pair
#: they share one number in WRF's two namespaces and they are a PAIR:
#: ``phys/module_physics_init.F:3790-3791`` fatals a QNSE PBL whose surface
#: layer did not set ``isfc = 4``, and only QNSESFCSCHEME sets it (:3171).
#: The PBL is woof/core/kernels/qnsepbl.cu (graded against
#: tools/qnse_wrf461_oracle); :func:`validate_qnse_config` is the law.
QNSE_PBL_SCHEME = 4
QNSE_SFCLAY_SCHEME = 4

#: WRF's BOULACSCHEME: the Bougeault-Lacarrere TKE closure, ported from WRF
#: v4.6.1 ``phys/module_bl_boulac.F`` (woof/core/boulac.py, graded word for
#: word against tools/boulac_wrf461_oracle).  WRF's law for it is its init
#: case (module_physics_init.F:3814-3816): the surface layer must set isfc 1
#: (revised or classic MM5) or 2 (Eta), which is the PBL/surface-layer
#: matrix (woof.wrf461_compatibility) for the MM5 cells and
#: :func:`validate_myj_pairing` for the Eta cell.  It is one of WRF's three
#: BEP/BEM PBL schemes (:data:`URBAN_BEP_PBL_SCHEMES`).
BOULAC_PBL_SCHEME = 8

#: The PBL schemes the Eta surface layer runs under, each mapped to the
#: carried field that holds WRF's TKE_PBL for it.  WRF's surface driver hands
#: MYJSFC ``tke_pbl`` (module_surface_driver.F:2211) and MYJSFC's PBLH scan
#: reads it (module_sf_myjsfc.F:263-277), so the Eta layer needs a PBL scheme
#: that carries one: MYJ's TKE_MYJ, or BouLac's own TKE (WRF admits BouLac
#: with isfc = 2, module_physics_init.F:3815).  The driver reads the scan's
#: column through this table and :func:`validate_myj_pairing` admits the
#: Eta layer under exactly its keys.
ETA_LAYER_TKE_FIELDS = {MYJ_PBL_SCHEME: "tke_myj",
                        BOULAC_PBL_SCHEME: "tke_pbl"}

#: WRF's KEPSSCHEME: the K-epsilon PBL of Zonato et al. (2022), prognostic
#: TKE, dissipation rate and turbulent potential energy, ported from WRF
#: v4.7.1 ``phys/module_bl_keps.F`` (woof/core/keps.py). Historical
#: synthetic v4.6.1 evidence is separate from the native real-input replay.
#: WRF's bl_init has no case for it, so WRF places no pairing law
#: on it; ArWen's own is :func:`validate_keps_config`, each refusal naming
#: what would break.
KEPS_PBL_SCHEME = 17

#: Microphysics selectors whose state carries WRF's ``P_QNI`` (cloud-ice
#: number), mapped to ArWen's name for that species.  The UW scheme mixes
#: the ice number and returns RQNIBLTEN, which WRF adds to exactly that
#: scalar (module_physics_addtendc.F:966-970, the adv_moist_cond default
#: arm); for every other scheme WRF hands the PBL its zero dummy slot and
#: the tendency goes nowhere.  Registry.EM_COMMON package lines: thompson
#: (8) and thompsonaero (28) scalar:qni; milbrandt2mom (9) qni; morr_two_
#: moment (10) qni; nssl_2mom (18) qni; p3_1category (50) and
#: p3_1category_nc (51) qni.
UW_PBL_ICE_NUMBER_SPECIES = {8: "ni", 9: "ni", 10: "ni", 11: "ni", 18: "qni",
                             28: "ni", 38: "ni", 50: "ni", 51: "ni", 52: "ni", 53: "ni"}

#: WRF's MRFSCHEME (Registry.EM_COMMON:3177, ``mrfscheme bl_pbl_physics==99``):
#: the Hong-Pan (1996) nonlocal PBL, YSU's predecessor, ported from WRF
#: v4.6.1 ``phys/module_bl_mrf.F`` (woof/core/mrf.py, kernels/mrf.cu) and
#: graded word for word against WRF's own compiled module
#: (tools/mrf_wrf461_oracle).  Its surface-layer law is WRF's own
#: (module_physics_init.F:3715-3717, isfc=1: the two MM5 layers), carried
#: as cells of :mod:`woof.wrf461_compatibility`; the BEP/BEM urban fatal
#: beside it (:3718-3719) is :data:`URBAN_BEP_PBL_SCHEMES`.
MRF_PBL_SCHEME = 99

#: WRF's EEPSSCHEME: the TKE + TKE-dissipation (e-epsilon) PBL of Zhang,
#: Wang and Xue (2020, MWR 148:1121-1145), ported from WRF v4.6.1
#: ``phys/module_bl_eepsilon.F`` (woof/core/eeps_pbl.py, with synthetic
#: regression fixtures in tools/eeps_wrf461_oracle). It carries WRF's advected scalar
#: pair PEK_ADV/PEP_ADV (Registry.EM_COMMON:1053-1054), transported by the
#: dycore with the moist scalars.  WRF's init law is a surface-layer list
#: and the BEP/BEM fatal (module_physics_init.F:3871-3876):
#: :func:`validate_eeps_config`.
EEPS_PBL_SCHEME = 16

#: The surface layers WRF v4.6.1 lets EEPS run over: ``isfc`` 1 (revised
#: MM5), 2 (Eta), 5 (MYNN) or 91 (classic MM5),
#: module_physics_init.F:3872-3873. Eta independently owns a zero TKE_PBL
#: column under EEPS (Registry myjsfcscheme), separate from PEK_ADV.
EEPS_SURFACE_LAYERS = (1, 2, 5, 91)

# Eta owns its native zero TKE column under EEPS.
ETA_LAYER_TKE_FIELDS[EEPS_PBL_SCHEME] = "tke_myj"

#: WRF's ACMPBLSCHEME: the Asymmetric Convective Model version 2 (Pleim
#: 2007), ported from the byte-pinned WRF v4.6.1 ``phys/module_bl_acm.F``
#: (woof/core/acm2pbl.py, graded word for word against
#: tools/acm2_wrf461_oracle).  WRF's pairing law is isfc 1 or 7
#: (module_physics_init.F:3728-3730: the MM5 surface layers or Pleim-Xiu),
#: transcribed as the WRF matrix's (7, sfclay) cells in
#: woof.wrf461_compatibility; its BEP/BEM fatal (:3731-3732) is
#: URBAN_BEP_PBL_SCHEMES.
ACM2_PBL_SCHEME = 7

#: ``bl_pbl_physics`` values in woof's schema.  0/1/2/5/7/9/11 are WRF's --
#: 2 is MYJ (module_bl_myjpbl.F), 7 ACM2 (:data:`ACM2_PBL_SCHEME`), 9 the
#: UW moist-turbulence PBL (:data:`UW_PBL_SCHEME`) and 11 is Shin-Hong
#: (module_bl_shinhong.F, ported at max ULP 0 against the byte-frozen WRF
#: v4.6.1 module; runtime wired by _run_shinhong in
#: woof/core/physics.py) -- and 900 is :data:`SASE_PBL_SCHEME`,
#: ArWen-only (see there).
PBL_SCHEMES = (0, 1, MYJ_PBL_SCHEME, GFS_PBL_SCHEME, QNSE_PBL_SCHEME, 5, ACM2_PBL_SCHEME, BOULAC_PBL_SCHEME, UW_PBL_SCHEME, 10, 11, GBM_PBL_SCHEME, EEPS_PBL_SCHEME, KEPS_PBL_SCHEME, MRF_PBL_SCHEME, SASE_PBL_SCHEME, CCPP_TKE_EDMF_PBL_SCHEME)

# Bidirectional coupling rows: the run door and registry read one reason.
PHYSICS_SELECTOR_COUPLINGS = (
    ("bl_pbl_physics", 10, "sf_sfclay_physics", 10,
     "TEMF shares TE_TEMF and HD_TEMF between its PBL and surface layer; "
     "another pairing leaves that carried total energy and thermal height "
     "without their required writer (WRF v4.6.1 "
     "share/module_check_a_mundo.F:2294-2299)"),
)


def selector_coupling_refusal(settings):
    for left, lv, right, rv, reason in PHYSICS_SELECTOR_COUPLINGS:
        a = int(settings.get(left, 0)) == lv
        b = int(settings.get(right, 0)) == rv
        if a != b:
            source, target, value = (left, right, rv) if a else (right, left, lv)
            return f"{source}={settings[source]} requires {target}={value}: {reason}"
    return None


def validate_selector_couplings(cfg):
    refusal = selector_coupling_refusal(vars(cfg))
    if refusal:
        raise ValueError(refusal)

# 0 = none, 1 = Kain-Fritsch, 3 = Grell-Freitas, 5 = Grell 3D, 6 = original
# Tiedtke, 16 = New Tiedtke.
#
# 16 IS WRF'S OWN NUMBER for NTIEDTKESCHEME (module_cumulus_driver.F), not
# an ArWen allocation. It became selectable once all six of the things
# tests/test_ntiedtke_phase2_gates.py names landed in one edit: the cudt
# law below, the advective forcing pair, clock.py's cumulus calendar,
# preflight's kernel-module table and its column-workspace term, and
# physics.py's optional tendency components. Each was a refusal that cost
# nothing while 16 was unreachable and would have been a silent wrong
# answer the moment it was not.
# These selectors share the KainFritsch adapter constructor. MSKF is in
# the physics family but owns an independent registry-backed adapter.
KF_ETA_CU_SCHEMES = frozenset({1, 99})
KF_FAMILY_CU_SCHEMES = KF_ETA_CU_SCHEMES | {11}
#
# 5 IS WRF'S OWN NUMBER for G3SCHEME, the Grell 3D ensemble scheme
# (phys/module_cu_g3.F), transcribed in woof/core/kernels/g3.cu and held
# word for word against tools/g3_wrf461_oracle.
#
# 96 IS WRF'S OWN NUMBER for NSASSCHEME, New SAS (YSU) with its built-in
# NSCV shallow convection (module_cu_nsas.F). It is admitted as a row of
# woof.core.column_cumulus.COLUMN_CUMULUS_SCHEMES -- one table row, no
# adapter module -- graded word for word against WRF v4.6.1's own object
# by tests/test_nsas_column_oracle.py.
CU_SCHEMES = (0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 94, 95, 96, 99)

SAS_CU_PHYSICS = 94
def cumulus_adaptive_cudt_zero_site(cu_physics: int) -> str | None:
    """The WRF adaptive-step authority for the KSAS admission rule."""
    if int(cu_physics) == 14:
        return "share/module_check_a_mundo.F:2141-2160"
    return None

COLUMN_KERNEL_CU_SCHEMES = (2,)

#: cu_physics -> the scheme's name, for the refusals and receipts below.
CU_SCHEME_LABELS = {0: "off", 1: "Kain-Fritsch", 3: "Grell-Freitas",
                    16: "New Tiedtke", 93: "Grell-Devenyi"}

#: The Grell-family cumulus schemes: the ones WRF's cumulus driver hands
#: the ensemble dimensions (maxiens, maxens, maxens2, maxens3, ensdim) and
#: for which check_a_mundo forces cu_diag = 1 (share/module_check_a_mundo.F
#: :2276-2286).  GRELL_FAMILY_DEFAULTS' clos_choice/ishallow below belong
#: to Grell-Freitas alone: GRELLDRV takes neither (it hard-codes ichoice =
#: 0, module_cu_gd.F:176, and has no shallow arm).
GRELL_FAMILY_CU_SCHEMES = frozenset({3, 93})

#: WRF v4.6.1's Registry defaults for the Grell ensemble dimensions
#: (Registry.EM_COMMON:2538-2542), the only values GRELLDRV can run.
GRELL_ENSEMBLE_DEFAULTS = {"maxiens": 1, "maxens": 3, "maxens2": 3,
                           "maxens3": 16, "ensdim": 144}

#: Why each ensemble dimension cannot move from its default in
#: Grell-Devenyi, by the breakage it would cause in module_cu_gd.F.
GRELL_ENSEMBLE_BREAKAGE = {
    "maxiens": ("GRELLDRV runs one outer ensemble member (iens = 1, "
                "module_cu_gd.F:177) and massflx_stats averages over "
                "ensdim/maxens3 members, so another value averages "
                "members that were never computed"),
    "maxens": ("CUP_enss zeroes and fills xaa0_ens(i,1..3) by hand "
               "(module_cu_gd.F:1132-1134) and cup_forcing_ens copies "
               "closures from the ne = 2 slab, so another value reads "
               "unset members"),
    "maxens2": ("cup_output_ens divides the summed iedt members by "
                "maxens2 while massflx_stats is handed the dimensions in "
                "swapped order (module_cu_gd.F:3584), so only the square "
                "3 x 3 shape is consistent"),
    "maxens3": ("cup_forcing_ens writes closures xff_ens3(1..16) into an "
                "array dimensioned maxens3 (module_cu_gd.F:2662-2760), so "
                "a smaller value writes past its end"),
    "ensdim": ("xf_ens/pr_ens are dimensioned ensdim and indexed up to "
               "maxiens*maxens*maxens2*maxens3 = 144, and massflx_stats "
               "averages ensdim/maxens3 members"),
}


def grell_ensemble_refusal(key: str, value: object) -> str | None:
    """The breakage a non-default Grell ensemble dimension causes, or None."""
    default = GRELL_ENSEMBLE_DEFAULTS[key]
    if not isinstance(value, bool) and isinstance(value, int) and value == default:
        return None
    return (f"{key}={value!r} is refused for Grell-Devenyi (cu_physics=93): "
            f"{GRELL_ENSEMBLE_BREAKAGE[key]}. Leave it at the WRF Registry "
            f"default {default}.")

#: The cumulus schemes that read ``sas_pgcon`` (WRF's convectively forced
#: pressure-gradient factor, Registry.EM_COMMON:2651) and ``nsas_dx_factor``
#: (:2493).  The SAS family's other members join their rows when ported.
SAS_PGCON_SCHEMES = frozenset({94, 96})
NSAS_DX_FACTOR_SCHEMES = frozenset({96})

#: Grell-Freitas, the cumulus scheme with the PBL requirement below.
GRELL_FREITAS_CU_PHYSICS = 3

#: Grell 3D (WRF's G3SCHEME), the scheme that reads ``cugd_avedx``.
GRELL_3D_CU_PHYSICS = 5

#: Cumulus schemes whose WRITTEN band is anchored to the domain's lateral
#: edges even on a periodic axis.  Grell 3D's conv_grell_spread3d writes only
#: ids+4..ide-5 in both axes whatever the boundary type
#: (module_cu_g3.F:5211-5229, woof.core.g3.spread_band), so on a periodic
#: domain the four columns beside each domain edge get no convective
#: tendency.  A streamed tile cannot reproduce that: its window wraps across
#: the domain edge and the band it computes is the WINDOW's, not the
#: domain's.  Measured (RTX 5090, 96x80 periodic, 2x2 and 3x3 tiles):
#: rthcuten/rqvcuten, the G3 ensemble carriers and everything radiation and
#: the dycore take from them differ from the resident run, while Kain-Fritsch
#: and New SAS stream bit-exact on the same gate.  On a non-periodic axis the
#: tile window is clamped to the domain, so a window edge that is a domain
#: edge carries the same band and a seam's band falls inside the halo.
#: :func:`woof.core.streaming.attach` refuses these schemes on a periodic
#: axis by this table; a new scheme with the same property joins it here.
CU_DOMAIN_EDGE_BAND_SCHEMES = frozenset({GRELL_3D_CU_PHYSICS})

#: The Grell-family cumulus schemes: each reads some of the keys in
#: :data:`GRELL_FAMILY_DEFAULTS`, as :data:`GRELL_FAMILY_READS` lists.
GRELL_FAMILY_CU_PHYSICS = frozenset({GRELL_FREITAS_CU_PHYSICS,
                                     GRELL_3D_CU_PHYSICS})

#: Which Grell-family keys each scheme reads, from WRF v4.6.1's cumulus
#: driver: GFDRV takes ichoice and ishallow_g3 (module_cumulus_driver.F:
#: 1296-1305); G3DRV takes those and cugd_avedx (:1250-1253).  A key a
#: selected scheme does not read must stay at its Registry default.
GRELL_FAMILY_READS = {
    GRELL_FREITAS_CU_PHYSICS: ("clos_choice", "ishallow"),
    GRELL_3D_CU_PHYSICS: ("clos_choice", "ishallow", "cugd_avedx"),
}

#: Cumulus schemes that run on the model step with no NCA hold: their
#: RAINCV is a per-call amount the driver adds once, so cudt_minutes is
#: pinned to 0 (WRF accumulates the held PRATEC every step between calls,
#: module_physics_addtendc.F:2141, which this contract would not repeat).
CU_MODEL_STEP_SCHEMES = frozenset({GRELL_FREITAS_CU_PHYSICS,
                                   GRELL_3D_CU_PHYSICS, 16})

#: Cumulus schemes that read the radiation-time net surface shortwave GSW:
#: Grell 3D lowers its capping depth to 25 hPa wherever GSW < 1 W m-2
#: (module_cu_g3.F:933).  The physics driver keeps the GSW plane, and
#: preflight prices it, for these as for RUC and the chem readers.
CU_GSW_READING_SCHEMES = frozenset({GRELL_3D_CU_PHYSICS, 93})

#: The stock cumulus callable for each admitted ``cu_physics``, by dotted
#: path.  ``initialize_physics`` resolves the selector through this table
#: rather than a branch per scheme, so admitting a scheme is a row here and
#: in :data:`CU_SCHEMES`.
CU_STOCK_CALLABLES = {
    14: "woof.core.sas.SasCumulus",

    7: "woof.core.cam5_runtime.CAMDeepConvection",
    1: "woof.core.kf.KainFritsch",
    10: "woof.core.kf.KFCumulusPotential",

    11: "woof.core.column_cumulus.MskfColumnCumulus",

    99: "woof.core.kf.KainFritsch",
    GRELL_FREITAS_CU_PHYSICS: "woof.core.gf.GrellFreitas",
    GRELL_3D_CU_PHYSICS: "woof.core.g3.Grell3D",
    6: "woof.core.tiedtke.Tiedtke",
    16: "woof.core.ntiedtke.NewTiedtke",
    94: "woof.core.sas.GfsSas",
    93: "woof.core.gf.GrellDevenyi",
}

#: The Grell-family keys, each with the WRF Registry default it is judged
#: against: :func:`validate_run_config` refuses a non-default value on a
#: domain whose ``cu_physics`` is not :data:`GRELL_FREITAS_CU_PHYSICS`.
#: The domain-tree loader (woof.experiment) reads the same table, so a
#: ``[shared]`` value on a tree that mixes Grell-Freitas with another
#: cumulus choice reaches the Grell-Freitas domains and leaves the others
#: on these defaults.
GRELL_FAMILY_DEFAULTS = {
    "clos_choice": 0,
    "ishallow": 0,
    "cugd_avedx": 1,
}

#: The SAS-family keys and their WRF Registry defaults, on the Grell
#: table's terms: read only where cu_physics is in SAS_PGCON_SCHEMES /
#: NSAS_DX_FACTOR_SCHEMES, so on a tree that mixes New SAS with another
#: cumulus choice a [shared] value reaches the SAS domains and leaves the
#: others on these defaults.
SAS_FAMILY_DEFAULTS = {
    "shalconv": 1,
    "nsas_dx_factor": 0,
    "sas_pgcon": 0.55,
}

#: Grell-Freitas closure selector, ``clos_choice`` (WRF's ``ichoice``).
#: ``cup_forcing_ens_3d`` (module_cu_gf_deep.F:2373-2720, gf.cu
#: ``gfd_cup_forcing_ens_3d``, woof/verify/gf_deep_ref.py) fills a
#: closure array of ``maxens3 = 16`` members, ``xf_ens(1:16)``.  0 takes
#: the ensemble mean of all sixteen; 1..16 copies member ``ichoice`` into
#: every slot, so the mean IS that member.  Each value in this range is
#: a code path WRF defines and the port carries.
GF_CLOSURE_MEMBERS = 16

#: The four closure families of ``cup_forcing_ens_3d``, named by what
#: each member's mass flux is computed from.  Members of one family
#: share their forcing; 14 and 15 differ from their siblings only in a
#: coefficient (14 scales the cloud-base mass flux by BETAJB once more,
#: 15 floors the precipitation efficiency at 1e-3 instead of 1e-5).
GF_CLOSURE_FAMILIES: tuple[tuple[tuple[int, ...], str], ...] = (
    ((1, 2, 3, 16),
     "the rate the large-scale forcing builds the cloud work function, "
     "(aa1 - aa0)/dt"),
    ((4, 5, 6, 14),
     "the grid-scale vertical mass flux (omega) at cloud base"),
    ((7, 8, 9, 15),
     "the column moisture convergence over the precipitation efficiency"),
    ((10, 11, 12, 13),
     "removal of the cloud work function aa1 over the adjustment time "
     "tau_ecmwf"),
)


def gf_clos_choice_refusal(value: object) -> str | None:
    """The breakage an out-of-range ``clos_choice`` would cause, or None.

    One wording for every door (``validate_run_config``, the namelist
    importer, the registry row).  Admitted: 0..16.  Refused: anything
    else, because the value has no meaning in WRF's closure code and the
    port would run one of two wrong things with no word said.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return (f"clos_choice must be an integer 0..{GF_CLOSURE_MEMBERS}, "
                f"got {value!r}.")
    if 0 <= value <= GF_CLOSURE_MEMBERS:
        return None
    if value > GF_CLOSURE_MEMBERS:
        why = (f"cup_forcing_ens_3d copies member clos_choice of the "
               f"{GF_CLOSURE_MEMBERS}-member closure array xf_ens(1:"
               f"{GF_CLOSURE_MEMBERS}) into every slot, so "
               f"clos_choice={value} reads past the end of that array")
    else:
        why = ("a negative value takes neither of cup_forcing_ens_3d's "
               "two branches (0 = ensemble mean, 1..16 = one member): the "
               "run would average the ensemble with its negative-forcing "
               "guard (the ichoice = 0 branch) skipped")
    return (f"clos_choice={value} is refused: {why}. Choose 0 (the "
            f"16-member ensemble mean, the WRF Registry default) or one "
            f"member 1..{GF_CLOSURE_MEMBERS}.")


def gf_clos_choice_advice(value: int) -> str | None:
    """The implemented-unverified warning for a single-member closure.

    None for 0, the ensemble mean, which is the arm compared bitwise
    against WRF v4.6.1 (tests/test_gf_driver_parity.py,
    tests/test_gf_gfdrv_cuda.py).
    """
    if value == 0:
        return None
    family = next(what for members, what in GF_CLOSURE_FAMILIES
                  if value in members)
    return (f"clos_choice={value} runs Grell-Freitas on closure member "
            f"{value} alone (its family computes cloud-base mass flux from "
            f"{family}) instead of the 16-member ensemble mean. It is WRF "
            "v4.6.1's own code path and the kernel matches the float32 "
            "reference on it, but only clos_choice=0 has been compared "
            "against a WRF run, so this closure is implemented but not "
            "verified against WRF.")

#: Every radiation selector value ``validate_run_config`` admits, as
#: importable tuples for the same reason :data:`MP_PHYSICS_ACCEPTED` is
#: one: the gate that proves every ACCEPTED scheme can be named by the
#: checkpoint writer (tests/test_checkpoint_scheme_identity_gate.py)
#: iterates what the loader accepts, and a menu spelled only inside a
#: refusal string cannot be iterated.  The refusal sentences below still
#: recite the names; these tuples are the membership they test.
RA_PHYSICS_ACCEPTED = (0, 4, 90)
RA_LW_PHYSICS_ACCEPTED = (0, 1, 3, 4, 5, 7, 31, 90, 99)
RA_SW_PHYSICS_ACCEPTED = (0, 1, 2, 3, 4, 5, 7, 90, 99)

#: Every other WRF v4.6.1/v4.7.1 radiation selector value, with the reason it
#: is refused.  A value is never run as another scheme: the forecast and its
#: receipt would name radiation that was never computed.  This is metadata
#: only; admitting one means a ported scheme, its menu entry above and its
#: registry row (tools/build_registry.py), never an edit here alone.
#:
#: Each row is a public sentence (the refusal prints it): the WRF scheme's
#: name and package, and what is missing.  Port bookkeeping (lane names,
#: batch branches) stays out of it.
RA_UNPORTED_REASONS: dict[str, dict[int, str]] = {
    "ra_lw_physics": {
        14: ("WRF RRTMG-K longwave is not ported; RRTMG (4, with "
             "ra_rrtmg_variant = rrtmg_legacy for WRF's transcription) is the "
             "nearest scheme and is a different k-distribution"),
        24: ("WRF fast RRTMG longwave (module_ra_rrtmg_lwf) is a separately "
             "coded solver that is not ported; ra_lw_physics = 4 with "
             "ra_rrtmg_variant = rrtmg_legacy runs WRF's RRTMG transcription"),
    },
    "ra_sw_physics": {
        14: ("WRF RRTMG-K shortwave is not ported; RRTMG (4, with "
             "ra_rrtmg_variant = rrtmg_legacy for WRF's transcription) is the "
             "nearest scheme and is a different k-distribution"),
        24: ("WRF fast RRTMG shortwave (module_ra_rrtmg_swf) is a separately "
             "coded solver that is not ported; ra_sw_physics = 4 with "
             "ra_rrtmg_variant = rrtmg_legacy runs WRF's RRTMG transcription"),
    },
}

#: Every other WRF v4.6.1-v4.8.0 cumulus, PBL, surface-layer and land-surface
#: selector value, with the reason it is refused -- the same law as
#: :data:`RA_UNPORTED_REASONS`, for the same reason.
#:
#: Named breakage: until 2.9.1 these values were refused with only the menu
#: of implemented numbers ("cu_physics = 4: no woof mapping (implemented:
#: [0, 1, 3, 5, 6, 16, 96])"), so a user whose WRF configuration uses BMJ,
#: ACM2, Pleim-Xiu or the slab LSM could not tell an unported scheme from a
#: typing error.  A row is retired when its scheme is admitted (its menu
#: entry and dispatch row land in the same edit); a WRF value with neither a
#: menu entry nor a row fails tests/test_wrf_unported_scheme_reasons.py.
WRF_UNPORTED_SCHEME_REASONS: dict[str, dict[int, str]] = {
    "cu_physics": {
        4: ("scale-aware SAS (WRF scalesasscheme) is not ported in this "
            "release; New SAS (96) is a different SAS generation"),
        98: ("the WRF v4.7 cumulus package ducuscheme is not ported in this "
             "release"),
    },
    "bl_pbl_physics": {
        6: ("WRF v4.6.1 to v4.8.0 define no bl_pbl_physics = 6 (the old MYNN "
            "level-3 selector), and wrf.exe stops on it ('The pbl option "
            "does not exist'); WRF selects the level-3 closure as "
            "bl_pbl_physics = 5 with bl_mynn_closure = 3.0, selecting the "
            "source-pinned WRF 4.7.1 MYNN generation"),
        98: ("the WRF v4.7 surface-drag boundary-layer package "
             "(surfdragscheme) is not ported in this release"),
    },
    "sf_sfclay_physics": {
    },
    "sf_surface_physics": {
        5: ("CLM4 (WRF clmscheme) is not ported in this release"),
        6: ("CTSM (WRF ctsmscheme) needs NCAR's external CTSM library, which "
            "this engine does not carry"),
    },
    **RA_UNPORTED_REASONS,
}

#: Appended to every unported-scheme refusal: the engine never runs a nearby
#: scheme under the requested selector.
WRF_UNPORTED_NOTHING_SUBSTITUTED = (
    "Nothing is substituted for it; choose one of the implemented values")


#: The aer_opt = 2 user-aerosol controls: read only under aer_opt = 2.
AER_OPT2_RUN_KEYS = ("aer_type", "aer_aod550_opt", "aer_angexp_opt",
                     "aer_ssa_opt", "aer_asy_opt", "aer_aod550_val",
                     "aer_angexp_val", "aer_ssa_val", "aer_asy_val",
                     "aer_auxinput15", "aer_auxinput15_interval")


def drop_inactive_radiation_port_keys(values: dict) -> None:
    """Drop, in place, the lightning and aer_opt = 2 RunConfig keys that the
    run does not read (lightning_option = 0; aer_opt != 2).

    One rule for every document that echoes a RunConfig -- the checkpoint
    echo and digest, the resume comparison and the public experiment
    document -- so a run that selects neither keeps the bytes it had before
    the fields existed, and a 2.8.9 checkpoint still resumes.
    """
    if values.get("ra_call_offset", 0) == 0:
        values.pop("ra_call_offset", None)
    from woof.core.madwrf_contract import drop_inactive_madwrf
    drop_inactive_madwrf(values)
    from woof.gfdleta_contract import inert_gfdleta_fields
    for name in inert_gfdleta_fields(values):
        values.pop(name, None)
    from woof.lightning_contract import drop_default_lightning_selectors
    drop_default_lightning_selectors(values)
    if values.get("aer_opt", 0) != 2:
        for key in AER_OPT2_RUN_KEYS:
            values.pop(key, None)


def wrf_unported_reason(key: str, value: int) -> str:
    """The named reason a WRF physics selector value is refused, or ''.

    Covers cumulus, PBL, surface layer, land surface and radiation
    (:data:`WRF_UNPORTED_SCHEME_REASONS`); microphysics has its own tables
    (:func:`unported_mp_scheme_refusal`)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    return WRF_UNPORTED_SCHEME_REASONS.get(key, {}).get(number, "")


def wrf_unported_refusal_tail(key: str, value: int) -> str:
    """`` <reason>. Nothing is substituted ...`` for an unported WRF value,
    or ``""`` when the value has no row (a number WRF does not define)."""
    why = wrf_unported_reason(key, value)
    if not why:
        return ""
    return f" {why[0].upper()}{why[1:]}. {WRF_UNPORTED_NOTHING_SUBSTITUTED}."


def unported_scheme_reasons(key: str) -> dict[int, str]:
    """Every WRF value of one physics selector that is refused by name, with
    its reason: the table the namelist contract publishes so a site can
    show the reason before a box starts."""
    if key == "mp_physics":
        return {value: text
                for table in (_P3_UNPORTED_VARIANTS, _WDM_UNPORTED_SIBLINGS,
                              _MP_UNPORTED_SCHEMES)
                for value, text in table.items()}
    return dict(WRF_UNPORTED_SCHEME_REASONS.get(key, {}))


def radiation_unported_reason(key: str, value: int) -> str:
    """The named reason a WRF radiation selector value is refused, or ''."""
    return RA_UNPORTED_REASONS.get(key, {}).get(int(value), "")

#: The exact id :attr:`RunConfig.km_opt_zero_acknowledgement` must carry.
#:
#: ONE SENTENCE, the one the refusal prints: ``km_opt = 0`` with a PBL
#: scheme that produces no horizontal mixing of its own leaves the run
#: with NO horizontal mixing operator, so the only thing that can damp
#: grid-scale horizontal structure is the sixth-order numerical filter --
#: and only when ``diff_6th_opt`` is nonzero, which is its own switch and
#: not km_opt's.  That is normally refused because it is what a mis-set
#: switch looks like rather than what a forecast wants.
#:
#: WHY IT IS ADMITTED AT ALL.  This is a real configuration, not an
#: impossible one: it is WRF's own ``diff_opt = 0``, and it is the
#: control the SASE attribution needs.  SASE is admitted at ``km_opt =
#: 0`` because its closure supplies the horizontal mixing the operator
#: would otherwise apply.  That admission, on its own, made the
#: single-variable control UNWRITABLE -- every SASE run necessarily
#: changed the PBL scheme AND removed the Smagorinsky operator, so no
#: run could say which of the two a difference came from.  Holding the
#: PBL scheme fixed and removing only the mixing is the missing cell.
#:
#: THIS IS NOT A GATE WIDENED TO PASS A TEST.  The gate's job is to stop
#: an unwitting misconfiguration, and it still does: the default still
#: refuses, the refusal still names the missing producer, and nothing
#: passes by accident.  What changed is that a user who states in
#: writing that the absent operator is the point can now write the run.
#: The id is a literal string rather than a boolean precisely so it
#: cannot be reached by a stray ``= true`` or an inherited default.
KM_OPT_ZERO_ACK = "no-horizontal-mixing-operator-v1"
#: sf_surface_physics -> the soil-layer counts that scheme defines.
#:
#: This is WRF's own table.  ``share/module_check_a_mundo.F`` subroutine
#: ``set_physics_rconfigs`` (:3546-3577) *resolves* ``num_soil_layers`` from
#: ``sf_surface_physics(1)`` -- it overwrites whatever the namelist asked for
#: -- and calls ``wrf_error_fatal`` on a scheme it has no entry for.  woof
#: follows that shape in :func:`soil_layer_count`, with one deliberate
#: difference recorded there.
#:
#: No count is written here.  Each value is READ from the module that owns the
#: scheme, so RUC's shipped nine is stated once, next to the level-depth
#: tables the kernel selects between -- ``__constant__ real
#: ruc_soil_layer_depth[RUC_NZS]``, whose ``#if RUC_NZS == 9`` /
#: ``#elif == 6`` arms are the two ``init_soil_depth_3`` tabulates.  (It read
#: ``[9]``, singular, while the forecast column was pinned to nine; that pin
#: is gone -- see docs/wrf_ruc_runtime_admission.md.)
#:
#: * Noah (2) and Noah-MP (4) share ``init_soil_depth_2``
#:   (``share/module_soil_pre.F:795`` and ``:807``, which is itself fatal at
#:   any other count), so both read
#:   :data:`woof.core.noah.NUM_SOIL_LAYERS`.
#: * RUC (3) is the only multi-geometry scheme.  ``init_soil_depth_3``
#:   (:1153-1194) tabulates zs for 6 and for 9 and is fatal at 4 or 5;
#:   :data:`woof.core.ruc_contract.WRF_SUPPORTED_NUM_SOIL_LAYERS` is that
#:   pair, in WRF's order.  Which of the two woof has actually validated is
#:   a separate question, answered by :func:`validated_soil_layer_count` and
#:   deliberately NOT encoded as this tuple's order -- the schema states what
#:   the scheme defines, not what woof has finished.
#:
#: DIVERGENCE, deliberate: WRF gives NOLSMSCHEME five layers
#: (``module_check_a_mundo.F:3547-3548``, a slab-scheme inheritance).  woof's
#: ``sf_surface_physics=0`` allocates no soil state at all, so the number is
#: only the length of a wrfout axis no field is written on; it is held at
#: Noah's so the frozen idealized-case wrfout headers (hill2d, igw, straka,
#: wk82, moist_bubble) keep byte identity.  Moving it would change five gated
#: files to describe an axis that stays empty.
class _LandSurfaceSoilLayers(_Mapping):
    """The table above, resolved per scheme on first ask.

    A lazy Mapping rather than a dict for one concrete reason: RUC's contract
    module is forecast-side, and the RW-WPS preprocessing wheel stages
    ``woof/config.py`` but deliberately not ``woof/core/ruc_contract.py``
    (``tools/build_rw_wps_release.py``, which fails the staging if config
    carries an import the wheel cannot resolve).  Reading RUC's geometry at
    config-import time would therefore make woof.config unimportable in that
    distribution, and the alternative -- writing 6 and 9 here instead --
    is the very duplication this table exists to prevent.  Deferring the read
    to the first question *about scheme 3* keeps both properties at once.

    ``_PROVIDERS`` names modules and attributes, never counts.
    """

    #: scheme -> (module, attribute holding every count WRF's own generator
    #: tabulates for that scheme, attribute holding the single count woof has
    #: validated -- the same attribute when the scheme defines one geometry).
    _PROVIDERS = {
        0: ("woof.core.noah", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        # init_soil_depth_1: five layers, fatal at any other count.
        1: ("woof.core.slab", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        2: ("woof.core.noah", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        3: ("woof.core.ruc_contract", "WRF_SUPPORTED_NUM_SOIL_LAYERS",
            "NUM_SOIL_LAYERS"),
        4: ("woof.core.noah", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        7: ("woof.core.pxlsm", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
        8: ("woof.core.ssib", "NUM_SOIL_LAYERS", "NUM_SOIL_LAYERS"),
    }

    def __init__(self):
        self._resolved: dict[int, tuple[tuple[int, ...], int]] = {}

    def _read(self, scheme: int) -> tuple[tuple[int, ...], int]:
        try:
            return self._resolved[scheme]
        except KeyError:
            pass
        module_name, counts_attr, validated_attr = self._PROVIDERS[scheme]
        module = _import_module(module_name)
        counts = getattr(module, counts_attr)
        counts = ((int(counts),) if isinstance(counts, int)
                  else tuple(int(value) for value in counts))
        validated = int(getattr(module, validated_attr))
        if validated not in counts:
            raise AssertionError(
                f"{module_name}.{validated_attr}={validated} is not one of "
                f"{module_name}.{counts_attr}={counts}; the scheme's own "
                "module disagrees with itself about its soil geometry")
        entry = (counts, validated)
        self._resolved[scheme] = entry
        return entry

    def __getitem__(self, scheme: int) -> tuple[int, ...]:
        return self._read(scheme)[0]

    def validated(self, scheme: int) -> int:
        """The one geometry woof has validated for ``scheme``.

        Kept apart from ``__getitem__`` on purpose: the schema states what the
        SCHEME defines (RUC: WRF's 6 and 9, in WRF's order), which is a
        different fact from what woof has finished, and encoding the second
        as the first tuple's order would quietly conflate them.
        """
        return self._read(scheme)[1]

    def __iter__(self):
        return iter(self._PROVIDERS)

    def __len__(self) -> int:
        return len(self._PROVIDERS)


LAND_SURFACE_SOIL_LAYERS = _LandSurfaceSoilLayers()


def validated_soil_layer_count(sf_surface_physics: int) -> int:
    """The soil geometry woof has validated for one land-surface scheme."""
    return LAND_SURFACE_SOIL_LAYERS.validated(int(sf_surface_physics))


#: The wrfout soil axis for a run that selected no land-surface scheme.  Named
#: so :mod:`woof.io.wrfout` can default its ``soil_layers`` argument without
#: carrying a soil-geometry constant of its own.
NO_LAND_SURFACE_SOIL_LAYERS = LAND_SURFACE_SOIL_LAYERS[0][0]
#: The soil axis real.exe writes with no land-surface scheme: WRF's
#: ``set_physics_rconfigs`` sets NOLSMSCHEME to five layers
#: (share/module_check_a_mundo.F:3548-3549) and overwrites any namelist value.
#: The WRF namelist door resolves this number, so a ``sf_surface_physics = 0``
#: namelist agrees with the ``wrfinput_d01`` its own real.exe wrote (93 runs of
#: the 2026-10-07 combo sweep were refused on 5 against 4).  Idealised configs
#: keep :data:`NO_LAND_SURFACE_SOIL_LAYERS` for the frozen headers named above.
WRF_NO_LAND_SURFACE_SOIL_LAYERS = 5
#: Human-readable names used only in validation messages.
_LAND_SURFACE_NAMES = {
    0: "none", 1: "the five-layer thermal diffusion (slab) scheme",
    2: "Noah LSM", 3: "RUC LSM", 4: "Noah-MP LSM"}


def soil_layer_count(cfg: RunConfig) -> int:
    """Resolved soil-layer count for the configured land-surface scheme.

    Every soil allocation, VRAM count, output dimension and restart shape
    reads this.  It is a *resolver*, not an accessor: it consults
    :data:`LAND_SURFACE_SOIL_LAYERS` for the scheme actually selected, so a
    ``RunConfig`` built in code rather than loaded through
    :func:`load_config` -- which is how most tools and tests build one, and
    which runs no validation -- cannot quietly hand a nine-layer scheme
    Noah's four.  That mattered concretely: the VRAM preflight sizes TSLB,
    SMOIS, SH2O and SMCREL from this number, and on this hardware an
    understated preflight is a correctness failure, not an estimate.

    DIVERGENCE from WRF, deliberate: ``set_physics_rconfigs`` *overwrites* a
    namelist request that disagrees with the scheme and only logs it at debug
    level, so a namelist asking Noah for nine layers runs on four with no
    error.  woof refuses instead.  Silently reinterpreting a soil geometry
    is how a run produces plausible, wrong soil and no receipt of it.
    """
    scheme = int(cfg.sf_surface_physics)
    if scheme == 0:
        requested = cfg.num_soil_layers
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("num_soil_layers must be a positive integer to preserve supplied inactive soil geometry")
        # With no LSM there is no scheme-specific soil geometry to override
        # the initializer. Keep its declared extent for I/O and restart.
        return requested
    defined = LAND_SURFACE_SOIL_LAYERS.get(scheme)
    if defined is None:
        # module_check_a_mundo.F:3573-3576, the same fail-closed default.
        raise ValueError(
            f"sf_surface_physics={scheme} has no associated number of soil "
            "levels; a land-surface scheme must declare its soil geometry "
            "in LAND_SURFACE_SOIL_LAYERS before it can be sized, written "
            "or checkpointed.")
    requested = int(cfg.num_soil_layers)
    if requested in defined:
        return requested
    name = _LAND_SURFACE_NAMES.get(scheme, f"sf_surface_physics={scheme}")
    expected = " or ".join(str(value) for value in defined)
    detail = ("" if len(defined) == 1 else
              f"  {name} defines more than one soil geometry, so it has no "
              "default to inherit: woof's validated geometry is "
              f"{LAND_SURFACE_SOIL_LAYERS.validated(scheme)} and must be "
              "requested explicitly, rather than taken from a RunConfig "
              "default that belongs to another scheme.")
    raise ValueError(
        f"num_soil_layers must be {expected} for {name} "
        f"(sf_surface_physics={scheme}), got {requested}: the scheme's soil "
        "level depths exist only for the counts it defines, so any other "
        "count has no soil column to size, write or checkpoint. Set "
        f"num_soil_layers={defined[0]}, or select the land-surface scheme "
        f"that defines {requested} layers.{detail}")


def radiation_scheme_ids(cfg: RunConfig) -> tuple[int, int]:
    """Resolve effective WRF ``(LW, SW)`` radiation scheme IDs.

    The two spellings may BOTH be written as long as they say the same
    thing.  ``ra_physics=N`` means "N on both streams", so ``ra_physics=4``
    beside ``ra_lw_physics=4``/``ra_sw_physics=4`` is one selection
    written twice, not two selections: the resolved pair is (4, 4) either
    way, every run-path consumer reaches the pair through THIS resolver
    (woof/core/physics.py, preflight, model, runtime, the composition
    layer), and the two places that echo ``cfg.ra_physics`` raw --
    woof/io/restart.py's ``resolved_schemes`` block and
    woof/verify/chaos_envelope.py's domain identity -- then print the
    same number the run used.  The receipt hazard that a wholesale
    refusal was written for is closed by the AGREEMENT, not by the
    refusal.  This matters beyond taste: the WRF namelist importer emits
    the aggregate spelling, so a caller who then writes the split pair
    onto an imported configuration was refused for restating it.

    What is still refused is a CONTRADICTION -- the aggregate naming one
    engine while the split pair names another -- because there the run
    and the receipts really would disagree and nothing in the
    configuration says which was meant.
    """
    lw = int(getattr(cfg, "ra_lw_physics", -1))
    sw = int(getattr(cfg, "ra_sw_physics", -1))
    aggregate = int(getattr(cfg, "ra_physics", 0))
    if lw == -1 and sw == -1:
        return aggregate, aggregate
    if (lw == -1) != (sw == -1):
        raise ValueError(
            "ra_lw_physics and ra_sw_physics must both be explicit or both "
            "be -1 (legacy ra_physics compatibility)")
    if aggregate != 0 and (lw, sw) != (aggregate, aggregate):
        raise ValueError(
            "the two radiation spellings contradict each other: "
            f"ra_physics={aggregate} selects that engine on BOTH streams "
            f"while ra_lw_physics={lw}/ra_sw_physics={sw} selects another "
            "pair, so the run and the receipts would name different "
            "radiation. Set ONE spelling: keep the split pair and set "
            f"ra_physics=0, or drop ra_lw_physics/ra_sw_physics to -1 and "
            "keep ra_physics.")
    return lw, sw


def bmj_feedback_radiation_supported(pair: tuple[int, int]) -> bool:
    """Whether both radiation streams consume BMJ clouds, or are disabled.

    Each physical adapter merges the carried clouds before its optical
    inputs are packed, including when it is used in a mixed composition.
    The analytic clear-sky adapter (90) has no cloud optical inputs.
    """
    return all(selector in (0, 1, 4, 5) for selector in pair)


def radiation_scheme_ids_from_settings(settings) -> tuple[int, int]:
    """:func:`radiation_scheme_ids`, asked of a settings MAP.

    The rule above is the engine's: ``ra_lw_physics`` and
    ``ra_sw_physics`` of -1 mean "not stated here", and the resolved pair
    lives in the combined ``ra_physics``.  Three readers outside this
    module re-implemented it against a switch map and all three copied
    it wrong in the same way -- they fell back to the combined value only
    when a split key was ABSENT, so a suite on the aggregate radiation
    option, which states -1 explicitly, was read as scheme -1 on both
    streams.  One of those readers then attached the acknowledgement for
    "nothing computes downward longwave" to a suite running full
    RTE+RRTMGP longwave.

    So the map-shaped question gets an answer HERE, beside the rule it
    is asking about, and the menu reader and both runners call it.

    Deliberately tolerant where :func:`radiation_scheme_ids` refuses: a
    picker classifies whatever it is handed, including a map with one
    half stated, and a display path must not raise.  A materialized
    RunConfig still meets the strict form before step 0.
    """

    def _value(name, default):
        value = settings.get(name, default)
        return default if value is None else int(value)

    combined = _value("ra_physics", 0)
    longwave = _value("ra_lw_physics", -1)
    shortwave = _value("ra_sw_physics", -1)
    return (combined if longwave < 0 else longwave,
            combined if shortwave < 0 else shortwave)


def radiation_enabled(cfg: RunConfig) -> bool:
    """Whether either resolved radiation component is active."""
    return any(radiation_scheme_ids(cfg))


def effective_radt_minutes(cfg) -> float:
    """The radiation cadence this configuration actually runs, in minutes.

    ``radt`` is the Phase-3 compatibility spelling declared above: a
    POSITIVE one overrides ``radt_minutes``, and zero means "not stated
    here", which is what a domain row that never mentions radiation
    carries.  Every integrator resolves it that way -- physics.py,
    clock.py, rrtmgp.py, dudhia.py, analytic_radiation.py and
    radiation_composition.py all spell ``radt if radt > 0 else
    radt_minutes`` -- so an EMITTER that wrote the raw field published
    ``radt = 0``, radiation every model step, for a configuration that
    means twelve minutes.  Named here, beside the two fields, so the
    next emitter asks rather than picks one.
    """
    return cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes


# --------------------------------------------------------------------------
# P3: the ported mp_physics values (50, 51, 53) and the unported siblings.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class P3Variant:
    """One WRF v4.6.1 P3 package, as data.

    WRF selects P3's configurations inside ONE Fortran module by what its
    driver passes to ``mp_p3_wrapper_wrf``: ``nc_3d`` present switches on
    ``log_predictNc`` (module_mp_p3.F:819-820) and ``qzi1_3d`` present
    switches on ``log_3momentIce`` (:822-823); ``N_ICECAT`` is the number
    of ice categories.  Every consumer that used to spell ``mp_physics ==
    50`` asks this table instead, so a further P3 package is a new row
    here, not a new code path.
    """

    mp_physics: int
    #: the Registry package name and its Registry.EM_COMMON line
    package: str
    registry_line: int
    label: str
    #: the driver passes ``nc_3d`` (prognostic droplet number, CCN
    #: activation, module_mp_p3.F:3313-3339)
    predict_nc: bool
    #: the driver passes ``qzi1_3d`` (three-moment ice, the 3momI table)
    three_moment_ice: bool
    #: restart algorithm identity (woof.checkpoint_identity)
    algorithm_identity: str
    n_cat: int = 1

    @property
    def extra_species(self) -> tuple[str, ...]:
        """Transported fields beyond mp=50's qi/ni/nr/qir/qib, in WRF's
        scalar-package order (``qnc`` first, ``qzi`` last,
        Registry.EM_COMMON:3039-3041), in woof's spelling."""
        return ((("nc",) if self.predict_nc else ())
                + (("qzi",) if self.three_moment_ice else ())
                + (("qi2", "ni2", "qir2", "qib2") if self.n_cat > 1 else ()))


#: The P3 packages woof runs.  50 is ``p3_1category`` (specified droplet
#: number, two-moment ice; module_microphysics_driver.F:1557-1602).  53 is
#: ``p3_1cat_3mom`` (Registry.EM_COMMON:3041): the driver passes BOTH
#: ``nc_3d=qnc_curr`` and ``QZI1_3d=qzi_curr`` with ``N_ICECAT=1``
#: (module_microphysics_driver.F:1710-1746), and ``p3_init`` reads the
#: 3momI table (module_physics_init.F:4574-4575).
P3_VARIANTS: dict[int, P3Variant] = {
    50: P3Variant(
        mp_physics=50, package="p3_1category", registry_line=3038,
        label="P3 one-category", predict_nc=False, three_moment_ice=False,
        algorithm_identity=(
            "p3-one-category-wrf-v4.6.1-v2-2mom-ice-specified-nc-"
            "diagnosed-ssat-rime-mass-volume-transported-real-clamp-product")),
    51: P3Variant(mp_physics=51, package='p3_1category_nc', registry_line=3039,
        label='P3 one-category, predicted droplet number', predict_nc=True, three_moment_ice=False,
        algorithm_identity='p3-one-category-wrf-v4.6.1-v2-2mom-ice-predicted-nc-ccn-activation-2mode-derf-diagnosed-ssat-rime-mass-volume-nc-transported-real-clamp-product', n_cat=1),
    52: P3Variant(mp_physics=52, package='p3_2category', registry_line=3040,
        label='P3 two-category', predict_nc=True, three_moment_ice=False,
        algorithm_identity='p3-two-category-wrf-v4.6.1-v1-prognostic-nc-iceice-merge-table2-v5.3', n_cat=2),
    53: P3Variant(
        mp_physics=53, package="p3_1cat_3mom", registry_line=3041,
        label="P3 one-category three-moment ice", predict_nc=True,
        three_moment_ice=True,
        algorithm_identity=(
            "p3-one-category-wrf-v4.6.1-v2-3mom-ice-predicted-nc-"
            "diagnosed-ssat-rime-mass-volume-qnc-qzi-transported-real-clamp-product")),
}

#: Every ``mp_physics`` value the P3 adapter (woof/core/p3.py) runs.
P3_MP_PHYSICS: tuple[int, ...] = tuple(P3_VARIANTS)


def p3_variant(mp_physics) -> P3Variant | None:
    """The P3 row for ``mp_physics``, or None for any other scheme."""
    try:
        return P3_VARIANTS.get(int(mp_physics))
    except (TypeError, ValueError):
        return None


def is_p3(mp_physics) -> bool:
    return p3_variant(mp_physics) is not None

def p3_predicts_nc(mp_physics) -> bool:
    row = p3_variant(mp_physics)
    return bool(row is not None and row.predict_nc)


_P3_UNPORTED_VARIANTS = {}

#: The mp_physics schema menu, as one sentence, so every out-of-schema
#: value is refused in the same shape.  Held apart from the raise sites
#: because there are two of them: the generic unknown-value branch and the
#: P3 sibling branch below, which states its own missing physics FIRST and
#: then recites this menu.  A refusal that recites the menu is a VALUE
#: refusal; a refusal that does not is about a combination.  That
#: distinction is essential -- tools/report_physics_composition_walk.py
#: separates the two kinds by exactly this recitation -- so a new mp value
#: is added here once and both branches stay in the same form.
_MP_PHYSICS_SCHEMA_MENU = (
    "mp_physics must be 0 (off), 1 (Kessler), 2 (Purdue Lin), 3 (WSM3), "
    "4 (WSM5), 5 (Ferrier-Aligo), 6 (WSM6), 7 (Goddard four-ice), 8 (Thompson), 9 "
    "(Milbrandt-Yau two-moment), 10 (Morrison two-moment), 13 (SBU-YLin five-class with riming intensity), 14 (WDM5), 26 (WDM7), 27 (UDM), 16 (WDM6 "
    "double-moment warm rain), 18 (NSSL two-moment), 24 (WSM7 "
    "single-moment with hail), 28 (Thompson aerosol-aware), 29 (RCON), 30 "
    "(FAST_SBM spectral bin), 38 (Thompson hail-aware), 40 (Morrison aerosol), 50 (P3 one-category), 51 (P3 predicted Nc), 52 (P3 two-category), 53 (P3 three-moment ice), 55 (ISHMAEL), 56 (NTU multi-moment), 95 (Eta Ferrier), 96 (MAD-WRF passive cloud), 97 (Goddard GCE three-ice), 106 (WSM6R, the "
    "revised WSM6), or \"tempo\" (TEMPO, WOOF id 900)"
)

#: Every ``mp_physics`` value the config loader admits, as the importable
#: authority the accepted-implies-builds instrument iterates
#: (tests/test_mp_accepted_builds.py).  This tuple and the menu sentence
#: above are two spellings of ONE schema: ``validate_run_config`` tests
#: membership against this tuple and recites the menu when it refuses, so
#: a new scheme is added in both places in the same edit or the loader
#: and its refusal text disagree in front of a user.
MP_PHYSICS_ACCEPTED = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 50, 51, 52, 53, 55, 56, 95, 96, 97, 106, 900)



class ThompsonAerosolMember(NamedTuple):
    """One member of WRF's aerosol-aware Thompson family.

    WRF v4.6.1 runs ONE module, module_mp_thompson.F, for both members, and
    the driver arms differ only in the arguments they hand mp_gt_driver
    (module_microphysics_driver.F CASE(THOMPSONAERO) and CASE(THOMPSONGH)).
    The hail-aware member adds the prognostic graupel number and volume
    (Registry package thompsongh: scalar qng, qvolg on top of thompsonaero's
    set), nine graupel density bins and their nine-slab rain-graupel tables
    (thompson_init :459-466).  Everything the aerosol-aware port owns --
    the prognostic droplet number, both aerosol tracers, their sources and
    the WIF climatology -- applies to both, so every reader that used to ask
    ``mp_physics in THOMPSON_AEROSOL_FAMILY`` asks membership here, and a reader that needs the
    graupel volume asks ``hail_aware``.
    """

    package: str
    hail_aware: bool
    #: State species beyond the aerosol-aware set, in woof's spelling.
    graupel_species: tuple[str, ...]


#: The aerosol-aware Thompson family, keyed by ``mp_physics``.
THOMPSON_AEROSOL_FAMILY: dict[int, ThompsonAerosolMember] = {
    28: ThompsonAerosolMember("thompsonaero", False, ()),
    29: ThompsonAerosolMember("rcon_mp_scheme", False, ()),
    38: ThompsonAerosolMember("thompsongh", True, ("ng", "qvolg")),
}


def thompson_aerosol_aware(mp_physics) -> bool:
    """Whether ``mp_physics`` is a member of the aerosol-aware family."""
    try:
        return int(mp_physics) in THOMPSON_AEROSOL_FAMILY
    except (TypeError, ValueError):
        return False


def thompson_hail_aware(mp_physics) -> bool:
    """Whether ``mp_physics`` carries the hail-aware graupel (mp=38)."""
    member = THOMPSON_AEROSOL_FAMILY.get(int(mp_physics or 0))
    return bool(member and member.hail_aware)


_WDM_UNPORTED_SIBLINGS = {}


#: Other WRF microphysics packages with no port, one row each, for the same
#: reason the P3 and WDM tables are rows: the refusal names what is missing,
#: and retiring a row when its port lands is table work.
_MP_UNPORTED_SCHEMES = {
    # The rest of WRF v4.6.1's microphysics packages with no port.  Named
    # breakage: until 2.9.1 each was refused as "no ratified woof mapping"
    # plus a list of numbers, which does not say whether the number is a
    # typing error or an unported scheme.
    12: (
        "mp_physics=12 (Milbrandt-Yau three-moment, package milbrandt3mom) "
        "is not ported. It predicts a third moment (reflectivity) for each "
        "precipitating class; the two-moment Milbrandt-Yau scheme "
        "(mp_physics=9) carries no third moment and is not run in its place."
    ),
    32: (
        "mp_physics=32 (full Khain-Lynn spectral bin microphysics, package "
        "full_khain_lynn) is not ported. It carries mass and number in "
        "dozens of size bins per hydrometeor class; the fast spectral bin "
        "scheme (mp_physics=30) is a different, reduced scheme and is not "
        "run in its place."
    ),
    98: (
        "mp_physics=98 (large-scale condensation, package lscondscheme) is "
        "not ported. It removes supersaturation as instantaneous rain with "
        "no cloud or ice species; no woof scheme runs it, and none is run "
        "in its place."
    ),
    99: (
        "mp_physics=99 (modified Kessler, package mkesslerscheme) is not "
        "ported. Its warm-rain rates differ from Kessler (mp_physics=1), "
        "which is not run in its place."
    ),
}


def unported_mp_scheme_refusal(value: int) -> str | None:
    """The by-name refusal for any unported WRF mp_physics package, or
    ``None`` when ``value`` is not one: the P3 siblings, the WDM siblings
    and :data:`_MP_UNPORTED_SCHEMES`, each its own missing physics followed
    by the schema menu."""

    for table in (_P3_UNPORTED_VARIANTS, _WDM_UNPORTED_SIBLINGS,
                  _MP_UNPORTED_SCHEMES):
        if value in table:
            return f"{table[value]} {_MP_PHYSICS_SCHEMA_MENU}, got {value}."
    return None


def unported_p3_variant_refusal(value: int) -> str:
    """A P3 sibling's refusal: its own missing physics, then the menu.

    Remaining entries in ``_P3_UNPORTED_VARIANTS`` name real WRF packages
    whose physics is unavailable. Each entry owes its specific reason;
    the message ends with the admitted schema menu. MP52 has its own
    CUDA category implementation and is no longer an unported sibling.
    """

    return (f"{_P3_UNPORTED_VARIANTS[value]} "
            f"{_MP_PHYSICS_SCHEMA_MENU}, got {value}.")


# --------------------------------------------------------------------------
# mp_physics = 28 aerosol source.  This WAS "the one deliberate deviation,
# published"; it is now a resolved input with a named fallback.
#
# RETIREMENT RECORD (development branch, 2026-08-27).  The constant
# ``MP28_AEROSOL_SOURCE_DEVIATION`` used to live here.  It said ArWen took
# its mp=28 aerosol initial state from thompson_init's synthetic CCN/IN
# profile because "ArWen has no QNWFA/QNIFA ingest lane", and warned that a
# whole-forecast comparison against a WIF-initialized WRF run was NOT
# equivalent.  Both halves of that sentence were true when it was written
# and neither is true now: ``woof/ingest/wif_climatology.py`` ports the
# ingest and is oracle-measured against real.exe, and it is what a real-data
# mp=28 run uses BY DEFAULT.  Fixing a defect retires its guards, so the
# constant is gone rather than softened -- a deviation notice that survives
# the deviation is how a tree accumulates warnings nobody can act on.
#
# What replaces it is two constants, because there are now two states and
# they are not the same claim:
#   * MP28_AEROSOL_SOURCE_DEFAULT   -- what a run that found the data did.
#   * MP28_AEROSOL_SYNTHETIC_FALLBACK -- what a run that did not found
#     instead, printed loudly and by name, because a forecast initialized
#     from a synthetic profile is a scientifically different forecast.
# --------------------------------------------------------------------------

#: What a real-data ``mp_physics = 28`` run uses for its aerosol initial
#: state, published as a named constant rather than a comment for the reason
#: the retired deviation was: a receipt can print a constant, and cannot
#: print a source comment.
#:
#: This is the DEFAULT and it is not a flag.  ``real.exe`` reaches the same
#: state through ``use_aero_icbc = .true.`` -> ``aer_init_opt = 1`` with
#: ``wif_input_opt = 1``, reading the global monthly dataset metgrid routes
#: through ``constants_name``; WRF in fact FATALs the other configuration
#: (``dyn_em/module_initialize_real.F:2735-2736``, ``'wif_input_opt=0 but
#: mp_physics=28'``).  ArWen now agrees with WRF by construction instead of
#: documenting why it did not.
MP28_AEROSOL_SOURCE_DEFAULT = (
    "mp_physics=28 takes its aerosol initial state from WRF's global "
    "monthly water/ice-friendly aerosol climatology "
    "(QNWFA_QNIFA_SIGMA_MONTHLY.dat), ported as "
    "woof/ingest/wif_climatology.py: metgrid's four_pt bilinear horizontal "
    "interpolation (METGRID.TBL:885-1150), real.exe's monthly_interp_to_date "
    "temporal weighting (module_initialize_real.F:8029-8095), and real.exe's "
    "vert_interp onto the dry eta pressure (:2452/:2519), with the surface "
    "emission qnwfa2d=w_wif(:,1,:)*0.000196*(50/z1) (:4530-4547). This is "
    "the same dataset, the same operators and the same stage order real.exe "
    "runs under use_aero_icbc=.true. (aer_init_opt=1, wif_input_opt=1), "
    "matched to it to 1e-5. The dataset is located by "
    "woof.ingest.wif_climatology.resolve_wif_climatology and the run "
    "receipt carries its path and SHA-256."
)

#: What a run got INSTEAD when the dataset could not be found, and why that
#: matters enough to be shouted rather than logged.
#:
#: The fallback is not a degraded version of the default.  It is WRF's own
#: synthetic boundary-layer-following CCN/IN profile
#: (``phys/module_mp_thompson.F:493-551``), a real initial condition that
#: WRF installs whenever the aerosol fields arrive empty -- but a forecast
#: started from it is a different forecast, measurably (see the lane/
#: wif-default before/after), and anyone comparing it to a WIF-initialized
#: WRF run is comparing two different experiments.  So it is REPORTED, in
#: the run receipt, by name, with the reason the dataset was not found.
MP28_AEROSOL_SYNTHETIC_FALLBACK = (
    "FALLBACK IN USE -- this mp_physics=28 run was NOT initialized from "
    "aerosol data. WRF's global monthly QNWFA/QNIFA climatology could not "
    "be located, so the aerosol initial state is thompson_init's SYNTHETIC "
    "CCN/IN profile (phys/module_mp_thompson.F:493-551), the profile WRF "
    "installs when the fields arrive empty. That is a scientifically "
    "different initial condition from the default: it is not measured "
    "aerosol, it is a two-parameter analytic profile keyed on the height of "
    "model level 1, and a whole-forecast comparison against a "
    "WIF-initialized WRF run is NOT equivalent and must not be reported as "
    "one. Set wif_climatology_path, $WOOF_WIF_CLIMATOLOGY, or "
    "$WOOF_WIF_CLIMATOLOGY_ROOT to WRF's QNWFA_QNIFA_SIGMA_MONTHLY.dat, or "
    "run from a directory holding it, to take the default path."
)

#: The dataset precondition for an EXTERNALLY FORCED mp=28 domain, asked of
#: every source.
#:
#: WHERE IT FIRES, and why not everywhere.  This is a question about the
#: MACHINE -- is a 225 MB dataset installed here -- not about the
#: configuration, so it is not part of ``validate_run_config``'s battery.
#: A battery that answers it refuses a NAMELIST IMPORT, which emits a TOML
#: and runs no forecast, for a file the import does not read and cannot be
#: handed: ``import_namelists`` has no ``mp28_aerosol_source`` parameter, so
#: the one way out that needs no download was unreachable at the door the
#: refusal fired at, and the user never got the TOML they would have set it
#: in.  It is measured by :func:`mp28_aerosol_lateral_forcing_precondition`
#: and raised by :func:`validate_experiment_preparation` at the doors that
#: commit to building a forecast, before either of them fetches a byte
#: (``woof go``'s stage composer, ``woof run``'s experiment dispatch),
#: with :func:`validate_run_preparation` kept as the floor inside
#: ``woof.ingest.real.initialize_real``; the registry reports it at plan
#: review from its own row, and the option panel reports it against the
#: draft.  Every one of those doors can carry the ways out below.
#:
#: What was here before was a per-route table: ``woof/hrrr_route_inputs.py``
#: subtracted 28 from the one route spelled "hrrr" and no other route was
#: checked at all.  It was wrong in both directions.  It refused hrrr runs
#: that HAD the dataset -- correct runs -- because the premise it named ("no
#: aerosol boundary species on this stream") stopped being true when
#: ``woof/ingest/wif_climatology.py`` landed and nwfa/nifa joined
#: ``lateral_bc.COUPLED_SCALAR_STATE_FIELDS``; WRF does not take aerosol
#: from the driving model either, it reads the same monthly dataset through
#: ``constants_name``.  And it let every other specified-BC route --
#: era5_direct, gfs_direct, mapped_direct, a bare ``woof run`` -- walk into
#: the real defect with no refusal.
#:
#: The real condition is DATASET-conditioned and source-independent: with no
#: climatology the aerosol falls back to thompson_init's synthetic profile,
#: ``aerosol_from_input`` is False, and nwfa/nifa revert to flow-dependent
#: boundaries with zero inflow -- the measured depletion in
#: docs/public/PHYSICS.md, a front at 1.00454 of the wind, the whole domain's
#: initial aerosol gone after L/U.  A domain with no EXTERNAL inflow face
#: is not exposed to it, which is why the check is on ``specified`` rather
#: than on mp=28 alone.
#:
#: THE MEASUREMENT IS HERE AND NOT IN THE SENTENCE.  The front reaches
#: 1.00454 of the wind speed, so a 100 km nest in a 20 m/s flow has lost its
#: initial aerosol in 83 minutes and nothing NaNs, trips a bound or reports it
#: (docs/public/wrf-comparison/mp28-column-evidence.md).  A reader at a door
#: needs the breakage and the way out; the numbers are why the refusal
#: exists, which is a question the source answers.
#:
#: AND WHY ``nested`` IS NOT PART OF THE CONDITION.  A child does not take
#: zero-inflow boundaries for these two fields: ``nwfa`` and ``nifa`` are
#: members of ``lateral_bc.COUPLED_SCALAR_STATE_FIELDS``, so a nest edge
#: carries them from the parent with the generic scalar coupling, whatever
#: the parent's aerosol source was.  The depletion is carried by
#: ``_external_scalar_boundary_fields`` (``lateral_bc.py``), the
#: specified-BC inventory, and an idealized tree -- a periodic root with
#: children and no ``specified`` anywhere -- has no external face at all.
#: Refusing it named a breakage that path does not have, which is the half
#: of the gate law that says a refusal must name a breakage it PREVENTS.
MP28_AEROSOL_LATERAL_FORCING_PRECONDITION = (
    "mp_physics=28 on a domain with external lateral boundaries "
    "(specified) needs WRF's monthly WIF aerosol climatology "
    "(QNWFA_QNIFA_SIGMA_MONTHLY.dat): without it nwfa/nifa take zero-inflow "
    "boundaries and aerosol-free air drains the domain's aerosol, to zero in "
    "clear air and to WRF's aerosol floor where the scheme runs, with "
    "nothing in the output saying so. Set [shared] "
    "wif_climatology_path or $WOOF_WIF_CLIMATOLOGY, stage it with `woof "
    "fetch-tables --wif`, or select [shared] mp28_aerosol_source='synthetic' "
    "to take the synthetic profile deliberately."
)

#: The three values of :attr:`RunConfig.p3_backend`.  "cuda" and "fused"
#: are the same device kernels composed into nine launches and three; they
#: are byte-identical on every p3-fortref fixture (evidence/
#: p3-cuda-20260829).  "reference" is the CPU float32 transcription.
P3_BACKENDS = ("cuda", "fused", "reference")

#: The supported values of :attr:`RunConfig.mp28_aerosol_source`.
#:
#: ``auto`` is the default and does the correct thing without being asked.
#: The other two exist because a default that cannot be pinned is a default
#: nobody can reproduce: ``climatology`` refuses rather than degrades, and
#: ``synthetic`` is how an idealized or deliberately data-free run NAMES the
#: fallback instead of arriving at it by accident.
MP28_AEROSOL_SOURCES = ("auto", "climatology", "synthetic", "analysis")

#: The only implemented value of each mp=28 aerosol-source selector, and the
#: named capability a user would need for anything else.  Read by
#: :func:`validate_aerosol_source_options`, by the namelist importer's
#: aerosol-key sweep, and by the mp=28 admission test -- one table, so a
#: refusal message and a namelist refusal cannot drift apart.
MP28_AEROSOL_SOURCE_OPTIONS: dict[str, tuple[int, str, str]] = {
    "aer_init_opt": (
        0,
        "Registry/Registry.EM_COMMON:2656",
        "aer_init_opt=2 is real.exe's FIRST-GUESS WIF branch and requires an "
        "explicit analyzed selection (mp28_aerosol_source='analysis' or "
        "use_rap_aero_icbc=true), with both metgrid QNWFA/QNIFA fields on every frame; "
        "otherwise an absent pair could silently take climatology. The CLIMATOLOGY branch "
        "(aer_init_opt=1 with wif_input_opt=1) IS ported "
        "(woof/ingest/wif_climatology.py) and is what a real-data mp=28 "
        "run does by default",
    ),
    "wif_input_opt": (
        0,
        "Registry/registry.new3d_wif:17",
        "wif_input_opt=2 additionally allocates the black-carbon scalar "
        "qnbca (Registry/registry.new3d_wif:82) and there is no nbca "
        "species anywhere in the mp=28 port; wif_input_opt=1 activates the "
        "use_wif_input package (monthly WIF levels per species, :80) and IS "
        "ported as the climatology pair with aer_init_opt=1, the default "
        "route for real-data mp=28",
    ),
}


def validate_radiation_driver_options(cfg: RunConfig) -> None:
    """Radiation-driver options: the implemented values, by name.

    * ``swint_opt`` is 0, 1 or 2.  1 needs a shortwave scheme: the between-call
      fit rewrites SWDOWN, SWDDIR, SWDDIF, SWDDNI and GSW on every step,
      and with ``ra_sw_physics = 0`` it would overwrite a declared or
      externally supplied SWDOWN with the fit of a flux nothing computed.
    * ``aer_opt`` is 0, 1, 2 or 3. 1 is the root-domain six-type climatology and 2 the user AOD path, both for legacy RRTMG shortwave.  3 is the Thompson-coupled branch, which WRF
      gates on the water-friendly aerosol package being present
      (``F_QNWFA``) and on RRTMG shortwave; it is transcribed for the
      legacy RRTMG shortwave composition only.  Any other pairing would
      run zero aerosol under a configuration whose receipt says aerosol-
      aware radiation, which is the silent substitution this refuses.  1 is the unimplemented ECMWF six-type aerosol climatology.
      2 reads user aerosol values through the WRF spectral producer.
    """
    if type(cfg.ra_call_offset) is not int or cfg.ra_call_offset not in (-1, 0):
        raise ConfigurationRefused(
            f"ra_call_offset={cfg.ra_call_offset!r}: must be integer -1 "
            "(old radiation call phase) or 0 (new radiation call phase)")
    alb_sol = cfg.alb_sol
    if type(alb_sol) is not int or alb_sol not in (0, 1):
        raise ConfigurationRefused(
            f"alb_sol={alb_sol!r}: must be integer 0 (unchanged land "
            "albedo) or 1 (sun-angle-dependent ALBSOL/ALBBCKSOL)")
    cloud_form = cfg.rrtmg_cloud_optics_form
    if cloud_form not in ("wrf_461", "noaa_wrf39"):
        raise ConfigurationRefused(
            f"rrtmg_cloud_optics_form={cloud_form!r} must be 'wrf_461' "
            "or 'noaa_wrf39': the source forms carry different cloud "
            "radius defaults and snow ice fractions, so an unknown name "
            "cannot select either")
    if type(cfg.cu_rad_feedback) is not bool:
        raise ConfigurationRefused("cu_rad_feedback must be a logical")
    if int(cfg.icloud) == 3:
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
        if (not cfg.moist or int(cfg.mp_physics) == 0
                or not legacy_cloud_fraction_flags(int(cfg.mp_physics))[0]):
            raise ConfigurationRefused(
                "Can not use icloud = 3 option, missing QC or QI field.")
    swint = cfg.swint_opt
    if isinstance(swint, bool) or swint not in (0, 1, 2):
        raise ConfigurationRefused(
            f"swint_opt={swint!r}: WRF defines 0 (hold the radiation call's "
            "surface shortwave for the interval), 1 (fit and interpolate), "
            "and 2 (FARMS); nothing else is a WRF value")
    if type(cfg.couple_farms) is not bool:
        raise ConfigurationRefused("couple_farms must be a logical")
    if cfg.couple_farms and swint != 2:
        raise ConfigurationRefused("couple_farms requires swint_opt=2")
    if cfg.rrtmg_aer3_source not in ("operational_fork", "wrf_461"):
        raise ConfigurationRefused("rrtmg_aer3_source must be operational_fork or wrf_461")
    if cfg.rrtmg_aer3_source == "wrf_461" and (
            cfg.rrtmg_smoke_manifest or cfg.simple_dir_fdb or
            cfg.rrtmg_aer3_type_source != "rural"):
        raise ConfigurationRefused("stock WRF aerosol type 1 has no prescribed smoke or landmask type branch")
    if swint == 2 and cfg.aer_opt == 3 and cfg.rrtmg_aer3_source != "wrf_461":
        raise ConfigurationRefused(
            "FARMS aer_opt=3 needs stock WRF gt_aod and surface aerosol diagnostics; "
            "the existing HRRR aerosol form has a different layer floor")
    aer = cfg.aer_opt
    if isinstance(aer, bool) or aer not in (0, 1, 2, 3):
        raise ConfigurationRefused(
            f"aer_opt={aer!r}: WRF defines 0, 1, 2 and 3 (Registry.EM_COMMON"
            ":2389); nothing else is a WRF value")
    lw, sw = radiation_scheme_ids(cfg)
    if aer == 1:
        if int(sw) != 4 or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY:
            raise ValueError("aer_opt=1 requires legacy RRTMG shortwave: RRTMGP has no checked six-type aerosol mapping and would discard or mis-map these optics")
        if cfg.nested:
            raise ValueError("aer_opt=1 on a resident nest needs the domain-1 aerod parent forcing; child-grid climatology would change WRF's root-only aerosol routing")
    if aer == 2:
        from woof.core.aer_opt2 import validate_options, OPTION_KEYS, VALUE_KEYS
        validate_options(aer_type=cfg.aer_type,
                         options=tuple(getattr(cfg,k) for k in OPTION_KEYS),
                         values=tuple(getattr(cfg,k) for k in VALUE_KEYS))
        if int(sw) != 4 or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY:
            raise ValueError("aer_opt=2 requires legacy RRTMG shortwave (ra_sw_physics=4); another SW solver would discard its aerosol optics")
        if any(getattr(cfg,k)==2 for k in OPTION_KEYS) and not cfg.aer_auxinput15:
            raise ValueError("aer_opt=2 gridded aerosol options require auxinput15_inname; missing AOD5502D/ANGEXP2D/AERSSA2D/AERASY2D would replace the requested optics")
        if cfg.aer_auxinput15_interval < 0:
            raise ValueError("aer_auxinput15_interval must be nonnegative seconds")
    if alb_sol == 1 and int(sw) == 0:
        raise ConfigurationRefused(
            "alb_sol=1 needs a shortwave scheme (ra_sw_physics != 0): "
            "ALBSOL/ALBBCKSOL are updated at radiation calls and otherwise "
            "would remain their allocation values")
    smoke = cfg.rrtmg_smoke_manifest
    if type(smoke) is not str:
        raise ConfigurationRefused("rrtmg_smoke_manifest must be a file path string; empty disables the forcing")
    if smoke and (int(sw) != 4 or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY
                  or aer != 3 or int(cfg.mp_physics) not in THOMPSON_AEROSOL_FAMILY):
        raise ConfigurationRefused(
            "rrtmg_smoke_manifest requires legacy RRTMG shortwave "
            "(ra_sw_physics=4, ra_rrtmg_variant='rrtmg_legacy'), aer_opt=3 "
            "and mp_physics=28: another path would discard the prescribed "
            "smoke or omit its Thompson aerosol background")
    type_source = cfg.rrtmg_aer3_type_source
    if type_source not in ("rural", "landmask"):
        raise ValueError(
            f"rrtmg_aer3_type_source={type_source!r} must be 'rural' (the "
            "HRRR fork's type 1 everywhere) or 'landmask' (the NOAA MPAS "
            "fork's type 1 over land, maritime type 3 over water)")
    if type_source != "rural":
        if aer != 3 or int(sw) != 4 or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY:
            raise ValueError(
                "rrtmg_aer3_type_source='landmask' selects the aer_opt=3 "
                "shortwave aerosol type and needs aer_opt=3 with legacy RRTMG "
                "shortwave (ra_sw_physics=4, ra_rrtmg_variant="
                f"'{RRTMG_VARIANT_LEGACY}'); got aer_opt={aer}, "
                f"ra_sw_physics={int(sw)}, ra_rrtmg_variant="
                f"{cfg.ra_rrtmg_variant!r}, where the selector would be "
                "recorded and never read")
        if smoke:
            raise ValueError(
                "rrtmg_aer3_type_source='landmask' with rrtmg_smoke_manifest: "
                "prescribed smoke follows the HRRR fork's rural optics and the "
                "NOAA MPAS driver has no smoke arm "
                "(mpas_atmphys_driver_radiation_sw.F:573-683), so the pair has "
                "no source behavior to reproduce")
    if cloud_form != "wrf_461" and (
            4 not in (int(lw), int(sw))
            or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY):
        raise ConfigurationRefused(
            "rrtmg_cloud_optics_form='noaa_wrf39' needs the legacy RRTMG "
            "path (ra_lw_physics=4 or ra_sw_physics=4 with "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}'); got "
            f"ra_lw_physics={int(lw)}, ra_sw_physics={int(sw)}, "
            f"ra_rrtmg_variant={cfg.ra_rrtmg_variant!r}, where the "
            "selected cloud wrapper would be silently discarded")
    if swint == 1 and int(sw) == 0:
        raise ConfigurationRefused(
            "swint_opt=1 needs a shortwave scheme (ra_sw_physics != 0): the "
            "between-call fit rewrites SWDOWN, SWDDIR, SWDDIF, SWDDNI and "
            "GSW on every step and would overwrite a declared or external "
            "SWDOWN with the fit of a flux nothing computed")
    if swint == 1 and int(sw) != 4:
        raise ConfigurationRefused(
            f"swint_opt=1 is wired for the RRTMG shortwave (ra_sw_physics=4), "
            "which hands the fit its surface direct beam SWDDIR; with "
            f"ra_sw_physics={int(sw)} WRF first forms SWDDIR by the "
            "Ruiz-Arias split (module_radiation_driver.F:2329-2360), which "
            "is not wired ahead of the fit here, so the direct-beam fit "
            "would run on a zero SWDDIR")
    if aer == 3:
        from woof.microphysics_schemes import SCHEMES_BY_ID
        named = SCHEMES_BY_ID.get(int(cfg.mp_physics))
        carries_numbers = (int(cfg.mp_physics) in THOMPSON_AEROSOL_FAMILY or
                           (named is not None and {"nwfa", "nifa"} <= set(named.aerosol_species)))
        if not carries_numbers:
            raise ValueError(
                f"aer_opt=3 needs water-friendly and ice-friendly aerosol numbers "
                f"(Thompson aerosol-aware mp_physics=28, or a named aerosol-aware "
                f"scheme that carries nwfa and nifa); "
                f"with mp_physics={int(cfg.mp_physics)} "
                "there is no QNWFA/QNIFA and WRF's driver would run zero "
                "aerosol under a configuration claiming aerosol-aware "
                "radiation")
        if int(sw) != 4 or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY:
            raise ConfigurationRefused(
                "aer_opt=3 is transcribed for the legacy RRTMG shortwave "
                f"composition only (ra_sw_physics=4 with ra_rrtmg_variant="
                f"'{RRTMG_VARIANT_LEGACY}'); got ra_sw_physics={int(sw)}, "
                f"ra_rrtmg_variant={cfg.ra_rrtmg_variant!r}, where the "
                "aerosol optics would be silently discarded")


def validate_aerosol_source_options(cfg: RunConfig) -> None:
    """Fail closed on any mp=28 aerosol-source selector WOOF cannot honour.

    THE DEFAULT MOVED (development branch).  ``(aer_init_opt, wif_input_opt)``
    at WRF's Registry defaults ``(0, 0)`` no longer MEANS "synthetic
    profile"; it means "WOOF chooses", and what WOOF chooses is the
    climatology whenever the dataset resolves.  The two namelist fields keep
    WRF's Registry defaults on purpose -- ``physics_compat``'s
    ``_SINGLE_DOMAIN_RUNTIME_SWITCHES`` rows are compared for EXACT equality
    by the prepared-forecast runner, and a nonzero default here would move
    every shipped profile for a reason that has nothing to do with profiles.
    The ArWen-side selection therefore lives in its own field,
    :attr:`RunConfig.mp28_aerosol_source`, and ``(1, 1)`` remains the
    namelist-level way to demand the same thing.

    ``aer_init_opt=2`` requires an explicit analyzed source selection so
    missing QNWFA/QNIFA cannot silently take climatology. Still refused is
    ``wif_input_opt=2`` (allocates qnbca, a species the port does not have).
    Those are unimplemented capabilities, not defaults, so they refuse by
    name rather than being reinterpreted.  Everything asked here is a
    property of the CONFIGURATION and is true on every machine, because
    this battery runs at every door including the ones that only translate
    a configuration; whether the dataset those selectors name is installed
    is asked by :func:`validate_run_preparation` at the run door.
    The check stays unconditional on
    ``mp_physics``: under any other scheme both keys are inert in WRF too,
    and accepting an inert nonzero here would let a configuration that MEANS
    something under mp=28 ride silently into an mp=28 restart or nest.
    """
    source = str(getattr(cfg, "mp28_aerosol_source", "auto") or "auto")
    if bool(cfg.use_rap_aero_icbc) and int(cfg.mp_physics) not in THOMPSON_AEROSOL_FAMILY:
        raise ConfigurationRefused("use_rap_aero_icbc requires mp_physics=28 or mp_physics=38; other schemes carry no QNWFA/QNIFA to initialize or force")
    if source not in MP28_AEROSOL_SOURCES:
        raise ConfigurationRefused(
            f"mp28_aerosol_source={source!r} is not one of "
            f"{MP28_AEROSOL_SOURCES}; 'auto' (the default) uses WRF's "
            "monthly WIF aerosol climatology when it resolves and announces "
            "the synthetic fallback by name when it does not, 'climatology' "
            "refuses rather than falling back, and 'synthetic' selects the "
            "fallback deliberately; 'analysis' requires both analyzed "
            "aerosol fields on every forcing frame")
    selected = (int(cfg.aer_init_opt), int(cfg.wif_input_opt))
    path = str(getattr(cfg, "wif_climatology_path", "") or "")
    analyzed = bool(cfg.use_rap_aero_icbc) or source == "analysis"
    if analyzed:
        if source not in ("auto", "analysis") or (path and not cfg.use_rap_aero_icbc):
            raise ConfigurationRefused("analyzed aerosol IC/BC cannot also select climatology or synthetic aerosol")
        if selected[0] not in (0, 1, 2) or selected[1] not in (0, 1):
            raise ConfigurationRefused("analyzed aerosol IC/BC requires aer_init_opt in 0,1,2 and wif_input_opt in 0,1; black carbon input is not implemented")
        return
    if selected == (0, 1) and source == "synthetic" and not path:
        # Native zero initialization followed by thompson_init's analytic
        # profiles, explicitly selected by the stock-namelist importer.
        return
    if selected == (1, 1):
        # The namelist spelling of mp28_aerosol_source='climatology' --
        # real.exe's use_aero_icbc=.true. state.  It no longer requires an
        # explicit path, because the resolver has a search order; it does
        # still require that the search SUCCEED, which the resolver enforces
        # with explicit_required=True.  Nothing about the PAIR is checked
        # here, and the pair is not a way past the dataset precondition
        # either: the namelist spelling of the request takes exactly the
        # same run-door battery the ArWen spelling takes
        # (:func:`validate_run_preparation`), which is why this branch
        # needs no check of its own.  One tuple, one door.
        return
    if selected != (0, 0):
        for name, (only, citation, why) in MP28_AEROSOL_SOURCE_OPTIONS.items():
            value = getattr(cfg, name)
            if value in (only, 1):
                continue
            raise UnsupportedConfiguration(
                f"{name}={value!r} is not implemented; WOOF honours "
                f"{name}={only!r} (WRF Registry default, {citation}) and "
                f"{name}=1 as the ported climatology pair. {why}.")
        raise UnsupportedConfiguration(
            f"aer_init_opt={cfg.aer_init_opt!r}/wif_input_opt="
            f"{cfg.wif_input_opt!r} is a MIXED aerosol-source selection. "
            "real.exe's climatology route derives both together "
            "(use_aero_icbc=.true. -> aer_init_opt=1, and the use_wif_input "
            "package -> wif_input_opt=1); one without the other names half "
            "an input path, and WOOF refuses rather than guessing which "
            "half was meant. Use aer_init_opt=1 with wif_input_opt=1, or "
            "leave both at 0 and let mp28_aerosol_source decide.")
    if path and source == "synthetic":
        raise ConfigurationRefused(
            "wif_climatology_path names WRF's monthly WIF aerosol dataset "
            "but mp28_aerosol_source='synthetic' selects thompson_init's "
            "synthetic profile instead, so nothing would read it; refusing "
            "to carry a dataset path no code would open rather than "
            "silently ignoring it")


def mp28_aerosol_lateral_forcing_precondition(cfg) -> str | None:
    """Why this configuration cannot be externally forced HERE, or None.

    THE ONE SPELLING of the dataset precondition, and a MEASUREMENT
    rather than a posture: it resolves the dataset through the module
    that owns the search and returns the sentence, so the run door can
    raise it, plan review can report it, and the option panel can print
    it beside a cell without any of the three writing its own version.

    ``mp28_aerosol_source='synthetic'`` answers None: that is the
    operator naming the fallback, which is the way out the sentence
    offers.  ``'climatology'`` is strict at ingest as well; here it is the
    same answer, earlier.
    """
    if int(getattr(cfg, "mp_physics", 0)) not in THOMPSON_AEROSOL_FAMILY:
        return None
    surface_from_monthly = bool(getattr(cfg, "use_rap_aero_icbc", False))
    if not bool(getattr(cfg, "specified", False)) and not surface_from_monthly:
        return None
    source = str(getattr(cfg, "mp28_aerosol_source", "auto") or "auto")
    if source == "analysis" and not surface_from_monthly:
        # Presence is checked against every decoded frame at initialization.
        # A climatology install cannot establish that analyzed pair.
        return None
    if source == "synthetic":
        return None
    from woof.ingest.wif_climatology import (
        MissingWifClimatologyDataset, resolve_wif_climatology)

    try:
        resolution = resolve_wif_climatology(
            str(getattr(cfg, "wif_climatology_path", "") or "") or None)
    except MissingWifClimatologyDataset as named:
        # A NAMED DATASET THAT IS NOT ONE IS A REFUSAL, NOT A TRACEBACK.
        # The resolver raises rather than returning unresolved when a
        # human chose the path -- a typo and a truncated download are its
        # two cases -- and every door here converts only ValueError, so an
        # operator who pointed the setting at a half-downloaded file got
        # `MissingWifClimatologyDataset` and a stack from `woof run`,
        # `woof go` and `woof run-plan --resolve` alike.  The resolver
        # already names the file, the breakage and the way out, so its
        # sentence IS the refusal: restating it here would be a second
        # copy that drifts from the one the ingest door raises.
        return str(named)
    if resolution.resolved:
        return None
    if surface_from_monthly:
        return ("use_rap_aero_icbc keeps analyzed three-dimensional aerosol, but its "
                "operational two-dimensional surface emission needs the monthly WIF "
                "dataset. Stage it with woof fetch-tables --wif or set "
                "wif_climatology_path; substituting an analyzed surface number "
                "would change the emission source. Searched: "
                + ", ".join(resolution.candidates) + ".")
    return (MP28_AEROSOL_LATERAL_FORCING_PRECONDITION
            + " Searched: " + ", ".join(resolution.candidates) + ".")


#: The machine preconditions a run door asks OF A ``RunConfig``, in the
#: order it asks them.
#:
#: WHAT IT IS AND IS NOT.  A configuration is portable: the same TOML is
#: valid on the laptop that wrote it and on the node that runs it, and
#: ``validate_run_config`` decides exactly that.  Whether a 225 MB dataset
#: is INSTALLED is a different question with a different answer per
#: machine, and mixing the two made a format conversion
#: (``import_namelists``) refuse for a missing download.  Every row here
#: is measured by a function taking a ``RunConfig`` and returning the
#: sentence or None.  :func:`validate_experiment_preparation` raises them
#: at the doors that commit to building a forecast, BEFORE they fetch
#: anything -- ``woof go``'s stage composer, ``woof run``'s experiment
#: dispatch and ``woof run-plan``'s resolution, which is the route the
#: desktop launches through; :func:`validate_run_preparation` is the
#: per-domain floor the real initializer keeps;
#: ``woof.physics_registry.validate_physics_plan`` reports the same row
#: at plan review from the registry's own declaration; ``woof check``
#: and ``woof.companion_physics.availability`` report it without
#: changing a verdict, because both can be asked about a machine that is
#: not this one.
#:
#: IT IS NOT THE WHOLE CENSUS OF INSTALL QUESTIONS, and it does not claim
#: to be.  A question a ``RunConfig`` cannot express belongs to the door
#: that loads the thing: a scheme's own table set is asked by its staging
#: preflight and its loader (Thompson's by
#: ``woof.prepared_domain_tree_forecast._verify_thompson_assets`` ->
#: ``woof.table_assets.require_thompson_tables``, RTE+RRTMGP's by
#: ``woof.core.rrtmgp._table``), each naming the missing member and the
#: command that stages it.  The registry publishes both kinds side by
#: side in a plan report's ``install_state``
#: (``woof.physics_registry.INSTALL_STATE_CODES`` names the codes and
#: the door per code), and ``tests/test_authority_agreement.py::
#: test_every_install_state_code_is_raised_by_a_run_door`` walks them, so
#: "the registry alone asks this" cannot become "nothing else asks this".
def microphysics_lookup_table_precondition(cfg) -> str | None:
    """Report missing external tables before preparation or a first step.

    Hail-aware Thompson may generate missing caches under its native
    write controls. That work belongs to preparation asset acquisition.
    ISHMAEL has no generated fallback.
    """
    mp = int(getattr(cfg, "mp_physics", 0))
    if mp == 55:
        from woof.core.ishmael import require_tables
        try:
            require_tables()
        except FileNotFoundError as error:
            return str(error)
    elif mp == 38:
        from pathlib import Path
        from woof.physics_compat import thompson_hail_table_root
        from woof.core.thompson_contract import HAIL_TABLE_ASSETS, HAIL_MP38_TABLE_FILE
        from woof.table_assets import classify_assets
        root = Path(thompson_hail_table_root())
        _valid, invalid, absent = classify_assets(root, HAIL_TABLE_ASSETS)
        if invalid:
            return "mp_physics=38 has invalid Thompson tables: " + "; ".join(invalid)
        missing = {asset.filename for asset in absent}
        if missing and getattr(cfg, "force_read_thompson", False):
            return (f"mp_physics=38 force_read_thompson=true needs verified tables in {root}: "
                    + ", ".join(sorted(missing)))
        if HAIL_MP38_TABLE_FILE in missing and not getattr(cfg, "write_thompson_mp38table", False):
            return (f"mp_physics=38 needs {HAIL_MP38_TABLE_FILE} in {root}; "
                    "stage the verified native table with WOOF_THOMPSON_HAIL_TABLE_ROOT "
                    "or set write_thompson_mp38table=true to generate it during preparation")
    return None


RUN_PREPARATION_PRECONDITIONS = (
    mp28_aerosol_lateral_forcing_precondition,
    microphysics_lookup_table_precondition,
)


def run_preparation_preconditions(cfg, *, pending_wif: bool = False) -> tuple[str, ...]:
    """Every machine-dependent precondition ``cfg`` does not meet."""

    said = [check(cfg) for check in RUN_PREPARATION_PRECONDITIONS
            if not (pending_wif and check is mp28_aerosol_lateral_forcing_precondition)]
    return tuple(sentence for sentence in said if sentence)


def validate_run_preparation(cfg) -> None:
    """Refuse, for one domain, what this machine cannot prepare.

    Asked at every door that is BUILDING a forecast rather than
    translating a configuration -- which is why it does not live in
    ``validate_run_config``: that battery is also the namelist importer's,
    and the importer emits a TOML and runs nothing, so a dataset refusal
    there names a breakage that path cannot have and offers a way out
    (``mp28_aerosol_source``) that lives only in the TOML it refused to
    write.

    This is the per-domain floor.  The doors a user actually types call
    :func:`validate_experiment_preparation` instead, which asks it of
    every domain of an experiment before anything is spent.
    """

    acquire_run_preparation_assets(cfg)
    for sentence in run_preparation_preconditions(cfg):
        raise ConfigurationRefused(sentence)


def acquire_run_preparation_assets(cfg) -> None:
    """Download, at a door that commits to a forecast, what the floor needs.

    Only the doors that BUILD a forecast call this (never ``woof check``,
    plan review or the namelist importer, which report about a machine).
    When the WIF precondition is unmet only because the dataset is not on
    this machine, it is staged under its pin first (the same acquisition
    ``initialize_real`` performs); if that fails the precondition's own
    sentence is the refusal.  Named breakage: an mp=28 run on a box without
    the file must not depend on someone having run `woof fetch-tables
    --wif` by hand (2026-10-23).
    """

    if int(getattr(cfg, "mp_physics", 0)) == 38:
        from woof.physics_compat import thompson_hail_table_root
        from woof.thompson_density_tables import (
            prepare_thompson_hail_tables, stage_hail_table_for_run)
        # A bare install has no native hail table; stage the verified one
        # (the fetch-tables route) before WRF's cache policy (2.9.12 R11).
        hail_root = str(stage_hail_table_for_run(
            str(thompson_hail_table_root()),
            force_read=getattr(cfg, "force_read_thompson", False)))
        prepare_thompson_hail_tables(
            hail_root,
            force_read=getattr(cfg, "force_read_thompson", False),
            write_tables=getattr(cfg, "write_thompson_tables", True),
            write_mp38table=getattr(cfg, "write_thompson_mp38table", False))
    if mp28_aerosol_lateral_forcing_precondition(cfg) is None:
        return
    from woof.ingest.wif_climatology import (
        MissingWifClimatologyDataset, acquire_wif_climatology)

    try:
        acquire_wif_climatology(
            str(getattr(cfg, "wif_climatology_path", "") or "") or None)
    except MissingWifClimatologyDataset:
        return


def experiment_preparation_refusals(experiment, *, pending_wif_domains=()) -> tuple[tuple[str, str], ...]:
    """``(domain label, sentence)`` for every unmet machine precondition.

    Reports rather than raises, so a door that wants to print all of them
    -- or to answer in its own refusal type -- can.
    """

    said: list[tuple[str, str]] = []
    for index, domain in enumerate(getattr(experiment, "domains", ()) or ()):
        grid_id = getattr(domain, "grid_id", index + 1)
        try:
            label = f"d{int(grid_id):02d}"
        except (TypeError, ValueError):       # pragma: no cover - defensive
            label = f"domains[{index}]"
        run = getattr(domain, "run", None)
        if run is None:
            continue
        for sentence in run_preparation_preconditions(
                run, pending_wif=grid_id in pending_wif_domains):
            said.append((label, sentence))
    # An enabled [simulated_radar] needs a current rw_simradar and a scan
    # the host memory holds; asked here, before the fetch, rather than at
    # the forecast's first history.  Off (the default) asks nothing.
    radar = getattr(experiment, "simulated_radar", None)
    if radar is not None and getattr(radar, "enabled", False):
        from woof.simulated_radar_config import door_refusals

        said.extend(door_refusals(experiment))
    return tuple(said)


def validate_experiment_preparation(experiment, *, pending_wif_domains=()) -> None:
    """Refuse, BEFORE THE FETCH, what this machine cannot prepare.

    WHY THIS EXISTS AND NOT ONLY THE RUN DOOR.  ``initialize_real`` is the
    floor, and a floor is not a front door: on ``woof go`` it runs in the
    PREPARE stage, after the authority stage, after the whole GFS/ERA5/HRRR
    cycle has been downloaded and after the input manifest has been
    verified; on ``woof run`` it is inside the time loop.  A precondition
    about this MACHINE -- is a 225 MB dataset installed here -- is knowable
    the moment the experiment loads, and asking it there is the difference
    between one sentence and one sentence after 10-15 GB of paid transfer.
    That "PASS, fetch, then refuse" shape is exactly what the route table
    this precondition replaced was written against, and moving the question
    out of ``validate_run_config`` would have recreated it if the doors
    below did not ask.

    Every door that COMMITS to building a forecast calls this with the
    loaded experiment before it spends anything: ``woof go``'s stage
    composer (before the fetch stage is spawned), ``woof run``'s
    experiment dispatch, and ``woof run-plan``'s resolution -- which is
    the front door the desktop launches every forecast through, and which
    reached its own fetch stage first until this was wired into it.
    ``woof check``'s preflight and ``tools/battery_route_preflight.py``
    report rather than raise.  The namelist importer is deliberately NOT
    one of them.
    """

    # Only a fetch that owns WIF acquisition may name pending domains.
    # Every other precondition remains enforced; preparation itself and
    # initialization call this without a pending acquisition allowance.
    # Every other domain acquires the dataset here, before the refusal is
    # measured, so a missing download is fetched rather than refused.
    for index, domain in enumerate(getattr(experiment, "domains", ()) or ()):
        run = getattr(domain, "run", None)
        if (run is not None
                and getattr(domain, "grid_id", index + 1) not in pending_wif_domains):
            acquire_run_preparation_assets(run)
    for label, sentence in experiment_preparation_refusals(
            experiment, pending_wif_domains=pending_wif_domains):
        raise ConfigurationRefused(f"{label}: {sentence}")


#: Every switch WRF's Milbrandt-Yau path hard-codes, with the line that
#: does it.  ``mp_milbrandt2mom_driver`` fixes the first seven
#: (phys/module_mp_milbrandt2mom.F:3615-3623) and the scheme body fixes the
#: last two (:1174-1175); there is no namelist that moves any of them, so
#: mp=9 has exactly ONE identity in WRF v4.6.1 and woof ships that one.
#: woof's constant table (woof/core/milbrandt2_constants.py) is derived
#: under these settings -- CCNtype=2 fixes N_c_SM=2e8 and
#: snowSpherical=.false. selects the Brandes m(D) pair -- so a knob that
#: moved one of them would silently invalidate 154 constants.
MILBRANDT2_FIXED_IDENTITY: dict[str, tuple[object, str]] = {
    "ccntype": (2, "module_mp_milbrandt2mom.F:3615 (continental)"),
    "precip_diag": (True, "module_mp_milbrandt2mom.F:3618"),
    "sedi": (True, "module_mp_milbrandt2mom.F:3619"),
    "warmphase": (True, "module_mp_milbrandt2mom.F:3620"),
    "autoconv": (True, "module_mp_milbrandt2mom.F:3621"),
    "icephase": (True, "module_mp_milbrandt2mom.F:3622"),
    "snow": (True, "module_mp_milbrandt2mom.F:3623"),
    "snow_spherical": (False, "module_mp_milbrandt2mom.F:1174"),
    "prim_ice_nucl": (1, "module_mp_milbrandt2mom.F:1175 (Meyers+contact)"),
}


def validate_milbrandt2_options(cfg: RunConfig) -> None:
    """Fail closed on an mp=9 identity switch woof cannot honour.

    THE PINNED IDENTITY.  WRF exposes no namelist for any of
    :data:`MILBRANDT2_FIXED_IDENTITY`, so there is nothing to validate on
    :class:`RunConfig` today -- the entry exists so that the day someone
    adds an ``mp9_*`` knob, the pin is already written down beside the
    Fortran line that owns it (the MYNN pattern).  The loop below refuses
    any such attribute that appears and disagrees.

    THE RTE+RRTMGP REFUSAL THIS FUNCTION USED TO CARRY IS RETIRED, with
    the defect it guarded.  It refused mp_physics=9 against the 4/4 pair
    on every ``ra_rrtmg_variant`` but ``rrtmg_legacy`` because
    ``woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME`` had no row for 9, and
    since the variant defaults to RTE+RRTMGP that refused every bare mp=9
    run.  The row now exists -- ``9: "milbrandt2"``, the scheme's own
    radii from the block WRF ships commented out
    (module_mp_milbrandt2mom.F:3351-3378) evaluated over the transported
    number moments -- so there is no pairing left to refuse.  What
    remains is the generic gate in ``woof.core.rrtmgp.cloud_optics_scheme``
    (an UNJUDGED selector still fails closed) and plan review's
    consumer-row check, both reached from validate_run_config before
    step 0.  The MYNN composition walk and tests/test_rrtmgp_coupling.py
    measure that the pairing now validates.
    """
    if cfg.mp_physics != 9:
        return
    for name, (only, citation) in MILBRANDT2_FIXED_IDENTITY.items():
        if not hasattr(cfg, name):
            continue
        value = getattr(cfg, name)
        if value == only:
            continue
        raise UnsupportedConfiguration(
            f"{name}={value!r} is not implemented for mp_physics=9; WRF "
            f"hard-codes {name}={only!r} ({citation}) and woof's constant "
            "table is derived under that identity, so a different value "
            "would silently invalidate it.")


# validate_p3_radiation RETIRED 2026-08-29, with the defect it guarded.
#
# It refused mp_physics=50 against the 4/4 RTE+RRTMGP pair because
# woof.core.rrtmgp._MP_CLOUD_OPTICS_SCHEME had no row for 50 and the run
# died at its first radiation call.  That row now exists -- ``50: "p3"``,
# transcribed from WRF's own P3 coupling: has_reqc=1, has_reqi=1,
# has_reqs=0 (phys/module_physics_init.F:1022-1024 then :1033 in v4.7.1,
# :1021-1023 then :1032 in v4.6.1), cal_cldfra1's P3 arm
# (module_radiation_driver.F:3879-3887) and the wrappers' P3 species remap
# (module_ra_rrtmg_lw.F:12250-12261, module_ra_rrtmg_sw.F:10851-10863).
#
# The refusal's OTHER half was worse than the thing it refused: it named
# ra_rrtmg_variant='rrtmg_legacy' as a way through, and that arm raised
# too -- woof.core.rrtmg_legacy._MP_DECLARES_RADII had no row for 50
# either, so the advised remedy also died at the first radiation call,
# after the prepare.  Both arms are wired now (the legacy adapter asks
# has_reqs separately, which is what its wrapper's P3 branch has always
# needed), so there is no pairing left to refuse and no remedy left to
# name.  A guard that outlives its defect refuses working configurations,
# which is what the 7,500 m mesh floor did.


#: Tables whose keys merge into :class:`RunConfig` field by field.
_RUN_CONFIG_TABLES = ("grid", "dynamics", "run", "fire")

#: Every table a RunConfig TOML may legally carry.
#:
#: ``[tiles]`` is here and NOT in :data:`_RUN_CONFIG_TABLES`, because it is
#: not a RunConfig table: it is an EXECUTION choice whose entire claim is
#: that it changes nothing about the forecast
#: (:func:`woof.core.streaming.identity_payload_entry`), so its keys belong
#: to :class:`woof.core.streaming.StreamingOptions` rather than to the
#: config whose fields a restart identity binds.  Read with
#: :func:`load_streaming_options`.
#:
#: It was absent from both lists until now, which is the defect: the
#: experiment TOML the multi-domain front doors read has accepted ``[tiles]``
#: since 2.2.0, and this schema -- the one ``woof downscale`` hands to the
#: offline child -- refused the block outright as an unknown table.  So the
#: one route whose domains are MOST likely to outgrow the card, a child
#: refined out of an archived parent, was the one route that could not ask
#: to stream.
#: ``[output]`` joins it on exactly the same terms.  It selects which
#: history variables reach the product tape
#: (:mod:`woof.io.history_selection`) and changes no number the model
#: computes, so it is not a RunConfig table either and does not enter a
#: restart identity -- a run that trimmed its history must resume from a
#: checkpoint written by one that did not, and the checkpoint stream is
#: a different file written from model state.
_KNOWN_TABLES = (*_RUN_CONFIG_TABLES, "tiles", "devices", "output", "simulated_radar")

#: ``[static]`` is known to ONE RunConfig-TOML route, the one that builds
#: static geography from a RunConfig: a downscaled child's own terrain,
#: land use, smoothing and high-resolution sources
#: (:func:`woof.offline_child_geography.load_child_static_policy`, read
#: through :func:`woof.offline_child.resolve_child_run_config`).  It is
#: not a RunConfig table either: the geography is an input the child is
#: built from, not a number its restart identity binds.  Every other route
#: that reads a RunConfig TOML builds no child geography, so it keeps
#: refusing the table as unknown: accepted there, a ``[static]`` table
#: would be dropped without a word.
_CHILD_STATIC_TABLE = "static"


def load_history_selection(path: str | Path):
    """The ``[output]`` block of a RunConfig TOML, or the shared FULL object.

    Separate from :func:`load_config` and not a field on what it returns,
    for the reason :func:`load_streaming_options` gives: ``RunConfig``'s
    fields bind into every restart identity, and which variables the
    HISTORY tape carries must not.  Both readers go through the same
    config authority, so they read the same bytes of the same file.
    """
    import io

    from woof.config_authority import read_config_authority
    from woof.io.history_selection import HistorySelection

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    return HistorySelection.from_mapping(raw.get("output"), source=str(path))


def load_device_options(path: str | Path):
    """Read execution options separately so they never bind RunConfig identity."""
    import io
    from woof.config_authority import read_config_authority
    from woof.core.devices import DeviceOptions
    raw = tomllib.load(io.BytesIO(read_config_authority(path).payload))
    return DeviceOptions.from_mapping(raw.get("devices"), source=str(path))


def load_streaming_options(path: str | Path):
    """The ``[tiles]`` block of a RunConfig TOML, or the shared OFF object.

    Separate from :func:`load_config` and not a field on what it returns:
    ``RunConfig``'s fields are bound into every restart identity, and
    ``[tiles]`` is the one surface that must NOT bind -- a checkpoint
    written resident has to resume streamed and back again, which is the
    operation the mode exists for.  Both readers go through the same config
    authority, so they read the same bytes of the same file.
    """
    import io

    from woof.config_authority import read_config_authority
    from woof.core.streaming import StreamingOptions

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    return StreamingOptions.from_mapping(raw.get("tiles"), source=str(path))


#: Keys whose value names a FILE the run opens for reading.  A relative
#: value resolves from the working directory first, else from the config
#: file's own directory and its ancestors, nearest first.
#:
#: The concrete breakage: a config that names its dataset relatively --
#: the documented way to keep a case folder self-contained -- opened only
#: from the one directory the author happened to be standing in.  From
#: anywhere else ``resolve_wif_climatology`` built the path against the
#: process CWD, found nothing, and refused with "there is no such file",
#: naming a path the user never wrote.  A config that travels with its
#: data has to be loadable from any working directory or it is not a
#: config, it is a bookmark.
#:
#: Working directory FIRST, deliberately, and not the config-dir-only
#: rule ``case_data._resolve_path`` uses: this value is scheme-scoped
#: into the run fingerprint (``core.model.SCHEME_SCOPED_RUN_FIELDS`` maps
#: mp=28 to it), so a checkout whose relative path already resolves must
#: keep resolving to the same string, or restarts and experiment
#: identities move underneath runs that were never reconfigured.
CONFIG_RELATIVE_FILE_KEYS: tuple[str, ...] = (
    "wif_climatology_path", "rrtmg_smoke_manifest",
    "fire_static", "fire_fuel_namelist",
    "ocean_config", "wind_farm_table", "wind_farms", "windturbines_spec",
    "tslist",
)


def _anchor_config_file_paths(merged: dict, config_path: str | Path,
                              *, keys=CONFIG_RELATIVE_FILE_KEYS) -> None:
    """Rewrite relative file values that resolve only beside the config.

    A value that is empty, absolute, or already a file from the working
    directory is left exactly as written.  One that resolves nowhere is
    also left as written: the consumer's own refusal names it, and
    rewriting it would only change which path the refusal prints.
    """

    base = Path(config_path).resolve().parent
    for key in keys:
        value = merged.get(key, "")
        if not value or not isinstance(value, str):
            continue
        if key == "windturbines_spec" and value in ("none", "ideal"):
            continue
        rel = Path(value)
        if rel.is_absolute() or rel.is_file():
            continue
        for home in (base, *base.parents):
            candidate = home / rel
            if candidate.is_file():
                merged[key] = str(candidate.resolve())
                break


def load_config(path: str | Path, *, accept_epssm_auto: bool = False,
                child_static: bool = False,
                native_fire_ideal: bool = False) -> RunConfig:
    """The RunConfig a legacy ``[grid]``/``[dynamics]``/``[run]`` TOML names.

    ``accept_epssm_auto`` admits ``epssm = "auto"`` (:data:`EPSSM_AUTO`),
    which then starts on ``RunConfig.epssm`` like an unset key. A derived
    child may carry ``epssm = { auto = 0.5 }`` to retain the inherited
    model-chosen value and its automatic label. Only a
    door that carries the label to the off-centering floor passes it
    (:func:`woof.offline_child.child_epssm_is_auto`); elsewhere the
    string stays refused, because a door that read "auto" as a plain 0.1
    would refuse that 0.1 over steep ground as if the user had chosen it.

    ``child_static`` admits the ``[static]`` table of a downscaled child's
    config (:data:`_CHILD_STATIC_TABLE`); only the offline child route,
    which builds the geography that table describes, passes it.
    """
    import io

    from woof.config_authority import read_config_authority

    authority = read_config_authority(path)
    raw = tomllib.load(io.BytesIO(authority.payload))
    known_tables = (_KNOWN_TABLES + (_CHILD_STATIC_TABLE,) if child_static
                    else _KNOWN_TABLES)
    # The loader's own refusals, before any door acts on this file.
    # Refuse missing random consumers before acting on the configuration.
    from woof.ensemble_admission import (
        OVERLAY_MARKERS, overlay_table_refusal, refuse_configured_random,
        table_not_honoured)
    refuse_configured_random(raw)
    if "ensemble" in raw:
        # No door that loads a RunConfig opens an ensemble session: the
        # legacy case runners and the downscaled child integrate one
        # forecast.  2.8.4 refused the table here as unknown.  An
        # experiment config may carry it now, so here it is refused by
        # name instead of read and dropped.
        table = raw["ensemble"]
        markers = OVERLAY_MARKERS & set(table) if isinstance(table, dict) else ()
        raise ValueError(
            overlay_table_refusal(markers) if markers else table_not_honoured(
                f"config file {path}", "a [grid]/[dynamics]/[run] config"))
    unknown_tables = [name for name in raw if name not in known_tables]
    if unknown_tables:
        hint = ""
        if _CHILD_STATIC_TABLE in unknown_tables:
            hint = (" [static] belongs to a downscaled child's config "
                    "(`woof downscale --child-config`), the one RunConfig "
                    "route that builds the geography it describes.")
        raise ValueError(
            f"unknown table(s)/top-level key(s) {unknown_tables} in config "
            f"file {path}; known tables: {list(_KNOWN_TABLES)}.{hint}"
        )
    # Validated here as well as in load_streaming_options, and discarded:
    # [tiles] is refused key by key by StreamingOptions on the
    # [relocation]/[perturbation] precedent -- a misspelled knob that
    # silently does nothing is how a run gets configured for a mode it is
    # not in -- and a caller that reads only the RunConfig must not be the
    # reason a typo survives admission.
    tiles = load_streaming_options(path)
    devices = load_device_options(path)
    from woof.core.devices import validate_device_road
    validate_device_road(devices, tiles)
    # Same treatment for [output], and for the same reason: a misspelled
    # history_drop that silently does nothing is how a run comes to write
    # the full inventory under the name of your selection.
    load_history_selection(path)
    from woof.simulated_radar_config import load_options
    if load_options(path).enabled:
        raise ValueError("[simulated_radar] needs an [experiment]/[[domain]] run "
                         "or the simulated-radar replay command; legacy case runners "
                         "do not expose a common durable-history stream")
    # And for [static]: a misspelled smoother is how a child's terrain gets
    # built under the default with your value's name on it.
    if _CHILD_STATIC_TABLE in raw:
        from woof.offline_child_geography import parse_child_static_table
        parse_child_static_table(raw[_CHILD_STATIC_TABLE], source=str(path))
    known_keys = {f.name for f in fields(RunConfig)}
    merged: dict = {}
    key_table: dict[str, str] = {}
    for table in _RUN_CONFIG_TABLES:
        entries = raw.get(table, {})
        unknown_keys = [key for key in entries if key not in known_keys]
        if unknown_keys:
            raise ValueError(
                f"unknown key(s) {unknown_keys} in table [{table}] of config "
                f"file {path}; known keys: {sorted(known_keys)}."
            )
        for key in entries:
            if key in key_table:
                raise ValueError(
                    f"duplicate key {key!r} appears in both "
                    f"[{key_table[key]}] and [{table}] of config file "
                    f"{path}; each key belongs in exactly one table."
                )
            key_table[key] = table
        merged.update(entries)
    _anchor_config_file_paths(merged, path)
    choice = merged.get("epssm")
    if not accept_epssm_auto and (
            choice == EPSSM_AUTO or isinstance(choice, dict)):
        raise ValueError(
            'epssm automatic choices need the child configuration loader; '
            'discarding the label could skip the steep-terrain acoustic floor. '
            'Use the downscale child door or a numeric epssm value.')
    if accept_epssm_auto:
        if choice == EPSSM_AUTO:
            del merged["epssm"]
        elif isinstance(choice, dict):
            if set(choice) != {EPSSM_AUTO} or isinstance(
                    choice[EPSSM_AUTO], bool) or not isinstance(
                    choice[EPSSM_AUTO], (int, float)):
                raise ValueError(
                    'epssm automatic inheritance must be { auto = NUMBER }; '
                    'a different table would lose the inherited acoustic setting')
            merged["epssm"] = choice[EPSSM_AUTO]
    # TOML arrays arrive as lists; RunConfig is frozen and hashable, so the
    # ladder is stored as a tuple.  Normalized here rather than in
    # validate_run_config, which returns its argument unchanged.
    if isinstance(merged.get("eta_levels"), list):
        merged["eta_levels"] = tuple(merged["eta_levels"])
    join_chem_name_lists(merged)
    apply_chem_set_defaults(merged)
    from woof.surface_ocean import apply_ocean_defaults
    apply_ocean_defaults(merged)
    return validate_run_config(RunConfig(**merged),
                               native_fire_ideal=native_fire_ideal)


#: RunConfig's chem name lists: comma-separated strings on RunConfig, and
#: accepted from TOML either as that string or as an array of names.
CHEM_NAME_LIST_KEYS = ("chem_sets", "chem_sources")


def join_chem_name_lists(values: dict) -> None:
    """Join TOML arrays of chem names into RunConfig's string, in place."""
    for key in CHEM_NAME_LIST_KEYS:
        value = values.get(key)
        if isinstance(value, (list, tuple)):
            if not all(isinstance(v, str) for v in value):
                raise ValueError(f"{key} must name chem sets/sources as "
                                 f"strings, got {value!r}")
            values[key] = ",".join(value)


def apply_chem_set_defaults(values: dict) -> dict:
    """Fill the chem keys a configuration leaves unset from its sets' rows.

    A set whose reference scheme runs a process by default says so in its
    file (``namelist_defaults``): the smoke set's reference, NOAA GSL's
    RRFS-SD smoke code (the descendant of HRRR-Smoke), washes smoke out and
    refreshes the plume every hour by default, where WRF-Chem's registry
    defaults (``wetscav_onoff = 0``, ``plumerisefire_frq = 180``) would run
    the smoke path with no washout and a three-hour plume.  So a bare
    ``chem_sets = "smoke"`` runs the scheme it names; a key the
    configuration states is never touched, so ``wetscav_onoff = 0`` still
    turns washout off.  ``values`` is the loader's merged key table, after
    :func:`join_chem_name_lists`; changed in place and returned.  Shared by
    both loaders.

    Refused (naming the breakage): a default for a key outside the chem
    block (a species set must not retune the dynamics), and two active sets
    defaulting one key to different values (the run would take whichever set
    happened to be listed first).  Nothing else records the filled keys:
    the resolved values are in the RunConfig every receipt already echoes.
    """
    sets_text = values.get("chem_sets")
    if not sets_text:
        return values
    from woof.chem_table import catalog, chem_names
    try:
        names = chem_names(sets_text, "chem_sets")
    except ValueError:
        return values       # validate_chem_config names the bad list
    sets = catalog().sets
    chosen: dict[str, tuple[object, str]] = {}
    for name in names:
        row = sets.get(name)
        if row is None:
            continue        # validate_chem_config refuses an unknown set
        for key, value in row.namelist_defaults.items():
            if key not in CHEM_RUN_FIELDS:
                raise ValueError(
                    f"chem set {name!r} defaults {key!r}, which is not a "
                    "chem key; a species set may only default the chem "
                    f"block ({', '.join(CHEM_RUN_FIELDS)}), or choosing a "
                    "set would silently retune the rest of the model.")
            if key in values:
                continue    # stated: the configuration's own choice
            if key in chosen and chosen[key][0] != value:
                raise ValueError(
                    f"chem sets {chosen[key][1]!r} and {name!r} default "
                    f"{key} to {chosen[key][0]!r} and {value!r}; set {key} "
                    "in the configuration, or the run would take whichever "
                    "set is listed first.")
            chosen.setdefault(key, (value, name))
    for key, (value, _name) in chosen.items():
        values.setdefault(key, value)
    return values

def apply_route_chem_sources(raw: dict, fetch_table) -> dict:
    """Enable the boundary sources a run's meteorological source brings.

    A boundary source row may name the ``[fetch]`` sources it belongs to
    (``default_for_fetch_sources``): HRRR-Smoke's 3-D smoke for a run that
    starts from HRRR (``hrrr``, ``hrrr-prs``).  HRRR begins every run from
    its previous run's smoke and takes smoke in at its edges; a WOOF run
    started from the same HRRR with smoke on and no smoke in the air or at
    the edges would forecast an atmosphere HRRR says is smoky as clean.  So
    when the config's ``[shared] chem_sets`` carries a row whose boundary
    list names such a source, and its ``[fetch] source`` is one of the
    source's, the source joins ``chem_sources`` (after the stated ones).

    ``raw`` is the experiment's parsed tables (changed in place, its
    ``[shared]`` table only); ``fetch_table`` the advisory ``[fetch]`` table
    or None.  Both experiment loaders call this before building, from the
    same file, so preparation and forecast resolve the same sources.
    """
    shared = raw.get("shared")
    source = (fetch_table or {}).get("source") if isinstance(fetch_table, _Mapping) else None
    if not isinstance(shared, dict) or not source or not shared.get("chem_sets"):
        return raw
    from woof.chem_table import catalog, chem_names
    values = dict(shared)
    join_chem_name_lists(values)
    try:
        sets = chem_names(values.get("chem_sets"), "chem_sets")
        stated = list(chem_names(values.get("chem_sources", ""), "chem_sources"))
    except ValueError:
        return raw      # validate_chem_config names the bad list
    table = catalog()
    named = {ref["source"] for row in table.species.values()
             if any(s in sets for s in row.sets)
             for ref in row.boundary}
    added = [name for name in sorted(named)
             if name in table.sources and name not in stated
             and source in table.sources[name].default_for_fetch_sources]
    if added:
        shared["chem_sources"] = ",".join(stated + added)
    return raw

#: Every RunConfig field of the chem block, in declaration order: inert
#: while ``chem_sets`` is empty, which is what lets a checkpoint or a
#: prepared state written before the block existed be read at defaults.
CHEM_RUN_FIELDS = (
    "chem_sets", "chem_sources", "chem_adv_opt", "chem_mix2_off",
    "chem_mix6_off", "chemdt", "kemit", "biomass_burn_opt",
    "plumerisefire_frq", "dust_opt", "seas_opt", "dmsemis_opt",
    "wetscav_onoff", "chem_conv_tr", "vertmix_onoff", "aer_ra_feedback",
    "aer_op_opt", "dust_alpha", "dust_gamma", "dust_smtune", "dust_ustune",
    "mynn_chem_vertmx", "fire_emission_mode", "plume_fire_properties",
    "aerosol_mp_coupling", "simple_dir_fdb", "enh_vermix", "flam_part",
    "aer_drydep_opt")

#: Every RunConfig field of the fire (SFIRE) block, in declaration order,
#: appended keyword-only after CHEM_RUN_FIELDS: inert while ``ifire`` is 0
#: (no FireCoupler is built and no fire static is read, woof/core/physics.py),
#: which is what lets a checkpoint or prepared state written before the
#: block existed be read at defaults.
from woof.sfire_config import FIRE_CONFIG_FIELDS as FIRE_RUN_FIELDS  # noqa: E402


def inert_fire_fields(values) -> tuple[str, ...]:
    """The fire keys an identity or public document omits for values.

    With ifire 0 no fire key is read by the forecast, so the block is
    left out and every fire-off identity keeps the bytes it had before the
    block existed.  fire_smoke is the one exception: it also selects the
    bulk-smoke chem profile without a fire grid, so a True value stays (it is
    bound through chem_sets as well).  With ifire 2 nothing is omitted.
    """
    get = values.get if hasattr(values, "get") else (lambda k, d=None: getattr(values, k, d))
    if int(get("ifire", 0) or 0) != 0:
        return ()
    return tuple(name for name in FIRE_RUN_FIELDS
                 if not (name == "fire_smoke" and get(name, False)))


#: SASE's structural requirements: attribute -> (admitted value, why).
#:
#: These are the closure's ACTUAL dependencies, and each one is a thing
#: the code reads or a thing that would be double-counted -- not a record
#: of which suite it happened to be smoke-tested with.  The lane that
#: built SASE admitted exactly one physics combination (one microphysics
#: id, one radiation id, one land-surface id) because a development lane
#: wants its evidence ladder narrow.  Those three are NOT dependencies:
#: the closure never reads a hydrometeor tendency, a radiative flux or a
#: soil layer, and pinning them would refuse combinations that work while
#: claiming a coupling that does not exist.  Re-scoping the gate to the
#: real dependency set is what makes the scheme experimentable; every
#: genuine constraint below is kept, and the ones that are about
#: correctness (double-counted mixing, the missing surface fluxes) refuse
#: rather than warn.
_SASE_REQUIREMENTS: tuple[tuple[str, object, str], ...] = (
    ("km_opt", 0,
     "SASE computes its own horizontal mixing from the closure's own "
     "diffusivities, so a km_opt mixing operator would double-count it"),
    ("khdif", 0.0,
     "constant-K diffusion may not silently stack on the SASE mixing"),
    ("kvdif", 0.0,
     "constant-K diffusion may not silently stack on the SASE mixing"),
)


def validate_temf_pairing(cfg: RunConfig, *, dfi_opt: int = 0) -> None:
    """WRF v4.6.1 module_check_a_mundo.F:2287-2330 TEMF fatals."""
    pbl, sfc = int(cfg.bl_pbl_physics), int(cfg.sf_sfclay_physics)
    if pbl == 10 and sfc != 10:
        raise ValueError(
            "Using bl_pbl_physics=10 requires sf_sfclay_physics=10. "
            "WRF v4.6.1 share/module_check_a_mundo.F:2292: "
            "the TEMF PBL reads EXCH_TEMF and WM_TEMF from its surface layer.")
    if sfc == 10 and pbl != 10:
        raise ValueError(
            "Using sf_sfclay_physics=10 requires bl_pbl_physics=10. "
            "WRF v4.6.1 share/module_check_a_mundo.F:2297: "
            "the TEMF surface layer reads TE_TEMF and HD_TEMF from its PBL.")
    if pbl == 10 and int(dfi_opt) != 0:
        raise ValueError(
            "DFI not available for bl_pbl_physics=10. "
            "WRF v4.6.1 share/module_check_a_mundo.F:2324: "
            "digital filtering is forbidden with the TEMF pair.")

def validate_myj_pairing(cfg: RunConfig) -> None:
    """Refuse a MYJ half without its other half, in either direction.

    WRF's own law, transcribed: ``phys/module_physics_init.F:3770-3772``
    is ``CASE (MYJPBLSCHEME) / if(isfc .ne. 2) CALL wrf_error_fatal
    ('module_physics_init: use myjsfc scheme for this pbl option')``, and
    of the surface layers woof ports only ``MYJSFCSCHEME`` sets
    ``isfc = 2`` (:3169).  So the MYJ PBL with anything else is a WRF
    fatal, and this is that fatal.

    THE REVERSE DIRECTION IS ARWEN'S, and it is stated as such.  WRF
    admits the Eta surface layer under several PBLs it also ports (the
    ``isfc .ne. 2`` guards at :3742 and :3756 belong to other schemes);
    EEPS consumes its published fluxes and preserves the independently
    allocated Eta TKE column. Other ported PBLs need a missing PAIR,
    named exactly:
    ``FM``/``FH``.  Those are the full similarity denominators
    ln(z/z0)-psi that WRF's PBL driver binds as ``PSIM``/``PSIH``
    (module_pbl_driver.F:1228); only the MM5 surface layers fill them
    (``SFCLAY_OUTPUTS``, woof/core/physics_inventory.py) and
    module_sf_myjsfc.F computes them nowhere.  YSU and Shin-Hong bind
    ``fm``/``fh`` directly and reconstruct ``zol = br*fm^2/fh``, so under
    the Eta layer they would divide by an allocated zero -- finite,
    plausible, and wrong.  The message names that pair and nothing else:
    the seven-field list this refusal used to print was partly FALSE
    (``rmol`` is in ``MYJ_SFCLAY_OUTPUTS`` and ``wspd`` is filled in the
    Eta layer's own post-call block), and a reason a reader can refute is
    how a correct refusal gets deleted.

    The PBL-OFF cell is refused separately and for its own reason: with
    the slot off nothing reads fm/fh at all, but the Eta layer's own PBLH
    scan reads the carried ``TKE_MYJ`` column, which is allocated for the
    MYJ and EEPS PBL selectors here.

    Urban: WRF sends the MYJ PBL through ``myjurb`` when
    ``sf_urban_physics`` is 2 or 3 (BEP/BEM, module_physics_init.F:
    3775-3781, phys/module_bl_myjurb.F).  woof does the same
    (``PhysicsDriver._run_myj_pbl`` routes to ``woof.core.myjurb``), and
    :func:`validate_urban_config` refuses the option while that module is
    not in the build.
    """
    pbl = int(cfg.bl_pbl_physics)
    sfclay = int(cfg.sf_sfclay_physics)
    if pbl == MYJ_PBL_SCHEME and sfclay == GFS_SFCLAY_SCHEME:
        # Not WRF's fatal: the GFS layer sets isfc=2 too
        # (module_physics_init.F:3186-3199), so WRF v4.6.1 runs this pair.
        # It is refused for what it would compute, not for what WRF says.
        raise ValueError(
            f"bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) cannot run on "
            f"sf_sfclay_physics={GFS_SFCLAY_SCHEME} (GFS). WRF v4.6.1 admits "
            "the pair (the GFS layer sets isfc=2, "
            "phys/module_physics_init.F:3186-3199), but the MYJ PBL's every "
            "implicit solve takes AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower "
            "boundary and SF_GFS writes none of them "
            "(phys/module_sf_gfs.F:9-18), so the PBL would integrate on the "
            "allocated zeros -- finite, plausible and wrong. Select "
            "sf_sfclay_physics=2 (Eta similarity) with MYJ, or the UW PBL "
            "(bl_pbl_physics=9) or no PBL with the GFS layer.")
    if pbl == MYJ_PBL_SCHEME and sfclay != MYJ_SFCLAY_SCHEME:
        raise ConfigurationRefused(
            f"bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) requires "
            f"sf_sfclay_physics={MYJ_SFCLAY_SCHEME} (Eta similarity), got "
            f"sf_sfclay_physics={sfclay}. WRF v4.6.1 fatals this exact "
            "pairing at phys/module_physics_init.F:3770-3772 ('use myjsfc "
            "scheme for this pbl option'): the MYJ PBL's every implicit "
            "solve takes AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as its lower boundary "
            "and only the Eta surface layer produces them. No substitution "
            "is applied.")
    # A DRY MYJ RUN IS ADMITTED.  The refusal that stood here read WRF's
    # `PRESENT(qv_curr) .AND. PRESENT(qc_curr)` fatal
    # (module_pbl_driver.F:1500-1513) as MYJ's own law; it is the OPTIONAL
    # argument plumbing every scheme arm in that driver carries, and in WRF
    # the presence of qv/qc is decided by the mp_physics package, not by the
    # PBL selector.  In woof a dry state is not an absent state:
    # woof/core/physics.py _atmosphere_for_physics hands every PBL seam the
    # persistent zero ``physics_dry_qv``/``physics_dry_qc`` planes.  Deleting
    # this arm alone did not make the run reachable: ``initialize_physics``
    # refused EVERY dry PBL scheme one layer later, so admitting MYJ here
    # bought a configuration that loads and cannot start.  That blanket
    # driver refusal goes with this arm, and the whole ported slot is now
    # measured dry rather than argued -- YSU, MYJ, MYNN and Shin-Hong each
    # initialise and step on a dry column, stay finite, and mix, with exch_h
    # and the PBL top both growing from the cold state
    # (tests/test_myj_port.py::
    # test_a_dry_pbl_run_reaches_the_driver_and_mixes).  MYJ's dry limit is
    # clean rather than 0/0:
    # MIXLEN's GH degenerates to dth*rdz and is floored at EPSGH, the qv/qc
    # rows solve against zero surface boundary values (qsfc/qz0/thz0/chklowq
    # all cold-start at zero) so rqvblten/rqcblten are exactly zero, and
    # nothing consumes them on a dry run.  If dry admission is ever to be
    # denied it is a property of the PBL SLOT, not of one scheme: it belongs
    # as a column on the PBL option rows the registry publishes, read by one
    # loop here, not as a branch.
    if sfclay == MYJ_SFCLAY_SCHEME and pbl == 0:
        raise ConfigurationRefused(
            f"sf_sfclay_physics={MYJ_SFCLAY_SCHEME} (Eta similarity) is "
            f"admitted with bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) or "
            f"{EEPS_PBL_SCHEME} (EEPS) or {BOULAC_PBL_SCHEME} (BouLac), got "
            "bl_pbl_physics=0 (PBL off). This cell has its own reason: the "
            "Eta layer's PBLH scan is MYJSFC's own TKE scan "
            "(module_sf_myjsfc.F:263-277), so the surface call itself reads "
            "the carried TKE_MYJ column -- and that column is allocated for "
            f"the bl_pbl_physics={MYJ_PBL_SCHEME} selector alone "
            "(woof/core/physics.py allocates tke_myj/el_myj there, at the "
            "EPSQ2 cold start MIXLEN's LPBL scan needs), so with the PBL "
            "slot off the first surface step has no TKE column to scan. "
            f"Select bl_pbl_physics={MYJ_PBL_SCHEME} or {BOULAC_PBL_SCHEME} "
            "to run the Eta layer, "
            "or sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5), "
            "which carry no TKE column and run with the PBL slot off.")
    if sfclay == MYJ_SFCLAY_SCHEME and pbl == UW_PBL_SCHEME:
        # The generic arm below names the FM/FH pair, which the UW scheme
        # never reads (it takes UST/HFX/QFX only), so it gets its own
        # breakage: the Eta layer's surface call itself scans TKE_MYJ.
        raise ConfigurationRefused(
            f"sf_sfclay_physics={MYJ_SFCLAY_SCHEME} (Eta similarity) is "
            f"admitted with bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) or "
            f"{BOULAC_PBL_SCHEME} (BouLac), got "
            f"bl_pbl_physics={UW_PBL_SCHEME} (UW). The UW scheme reads "
            "only UST, HFX and QFX from its surface layer, but the Eta "
            "layer's PBLH scan is MYJSFC's own TKE scan "
            "(module_sf_myjsfc.F:263-277) over the carried TKE_MYJ column, "
            "which MYJ advances and EEPS preserves as a separate zero "
            "column: under UW the "
            "surface call would scan a column no scheme writes. Select "
            "sf_sfclay_physics=1 (revised MM5), 91 (classic MM5) or 5 "
            "(MYNN) or 3 (GFS) with the UW PBL.")
    # BouLac under the Eta layer is WRF-legal (isfc = 2,
    # module_physics_init.F:3815) and runs: BouLac reads only UST/HFX/QFX of
    # the surface set, which the Eta layer writes, and it carries the
    # TKE_PBL column the Eta layer's PBLH scan reads (ETA_LAYER_TKE_FIELDS).
    if sfclay == MYJ_SFCLAY_SCHEME and pbl not in ETA_LAYER_TKE_FIELDS:
        raise ConfigurationRefused(
            f"sf_sfclay_physics={MYJ_SFCLAY_SCHEME} (Eta similarity) is "
            f"admitted with bl_pbl_physics={MYJ_PBL_SCHEME} (MYJ) or "
            f"{BOULAC_PBL_SCHEME} (BouLac) or {EEPS_PBL_SCHEME} (EEPS), got "
            f"bl_pbl_physics={pbl}. The Eta surface layer publishes no FM/FH "
            "-- the full similarity denominators ln(z/z0)-psi that WRF's PBL "
            "driver binds as PSIM/PSIH (module_pbl_driver.F:1228) and that "
            "only the MM5 surface layers fill (SFCLAY_OUTPUTS in "
            "woof/core/physics_inventory.py); module_sf_myjsfc.F computes "
            "them nowhere. YSU and Shin-Hong bind fm/fh directly and "
            "reconstruct zol = br*fm^2/fh from them, so under the Eta layer "
            "they would divide by an allocated zero -- finite, plausible and "
            "wrong. Producing fm/fh out of the Eta layer's tabulated psi "
            "would be new physics, not a table row. Select "
            "sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5) for "
            f"those schemes, or bl_pbl_physics={MYJ_PBL_SCHEME} or "
            f"{EEPS_PBL_SCHEME} or {BOULAC_PBL_SCHEME} for this surface layer.")


def validate_gfs_surface_pairing(cfg: RunConfig) -> None:
    """SF_GFS has no RMOL output for MYNN or EEPS."""
    if int(cfg.sf_sfclay_physics) == GFS_SFCLAY_SCHEME and int(cfg.bl_pbl_physics) in (5, 16):
        raise ValueError(
            "sf_sfclay_physics=3 (GFS) publishes no RMOL; the selected PBL "
            "reads RMOL and would use the allocated zero. Select the paired "
            "bl_pbl_physics=3 (GFS), or a surface layer that writes RMOL.")


def validate_gfs_pbl_config(cfg: RunConfig) -> None:
    """Admission for the GFS nonlocal PBL (bl_pbl_physics=3).

    The scheme's lower boundary is six words of its surface layer: UST,
    PSIM, PSIH, GZ1OZ0, WSPD and BR (module_pbl_driver.F:1416-1418), from
    which it forms FM = GZ1OZ0 - PSIM and the surface stress
    (karman*wspd/FM)**2 (module_bl_gfs.F:282-321).  A surface layer that
    does not write GZ1OZ0/PSIM leaves FM meaningless and the stress,
    UST and the momentum solve's surface row unbounded.  So the
    admitted layers are :data:`GFS_PBL_SURFACE_LAYERS`, which publish the
    set with SF_GFS's definitions.

    WRF requires isfc == 2 (module_physics_init.F:3742-3743).
    SF_GFS is the implemented layer that also writes every lower-boundary
    word this solver reads. The archived MM5 exception is retired.
    """
    if int(cfg.bl_pbl_physics) != GFS_PBL_SCHEME:
        return
    sfclay = int(cfg.sf_sfclay_physics)
    if sfclay not in GFS_PBL_SURFACE_LAYERS:
        raise ValueError(
            f"bl_pbl_physics={GFS_PBL_SCHEME} (GFS) requires a surface "
            "layer that publishes UST, PSIM, PSIH, GZ1OZ0, WSPD and BR, got "
            f"sf_sfclay_physics={sfclay}. The scheme forms FM = GZ1OZ0 - "
            "PSIM and the surface stress (karman*wspd/FM)**2 from them "
            "(module_bl_gfs.F:282-321), so a layer that does not write them "
            "leaves FM meaningless: WRF's own isfc == 2 pairing with the Eta "
            "layer, which writes none of GZ1OZ0/PSIM/PSIH, reached +Inf UST "
            "and NaN fluxes within two model minutes and then a "
            "segmentation fault (tools/gfs_pbl_wrf461_oracle/README.md). "
            "Select sf_sfclay_physics=3 (GFS).")
    urban = int(getattr(cfg, "sf_urban_physics", 0) or 0)
    if urban in (2, 3):
        raise ValueError(
            f"bl_pbl_physics={GFS_PBL_SCHEME} (GFS) cannot run with "
            f"sf_urban_physics={urban} (BEP/BEM). WRF v4.6.1 fatals this "
            "pairing (module_physics_init.F:3744-3745: 'use ysu (option1), "
            "myj (option 2), or boulac (option 8) with BEP/BEM urban "
            "scheme'): BL_GFS has no arm for the multi-layer canopy's drag "
            "and heat source terms, so the urban momentum and heat "
            "exchange would be dropped. Select bl_pbl_physics=1 (YSU) or "
            "2 (MYJ) with BEP/BEM, or sf_urban_physics=1 with the GFS PBL.")


def validate_uwpbl_config(cfg: RunConfig) -> None:
    """Admission for the UW moist-turbulence PBL (bl_pbl_physics=9).

    WRF's own law for the scheme is one fatal, the BEP/BEM urban pairing
    (module_physics_init.F:3825-3827), and it is reproduced here for the
    day ``sf_urban_physics`` exists in this schema.  Any surface layer that
    publishes UST/HFX/QFX is admitted; the Eta-layer cell is refused in
    :func:`validate_myj_pairing` for its own reason.  The ArWen-side
    refusals each name what would break: no surface layer (the scheme's
    whole lower boundary would be the allocated zeros) and the held
    ice-number tendency under a positive PBL cadence.  A dry state is
    admitted, as for every scheme in the PBL slot: in WRF the moisture
    arrays are the microphysics package's to provide, not the PBL's to
    demand, and the scheme reads the driver's persistent zero planes
    (total water at CAM's 1e-30 floor, no condensate).
    """
    if int(cfg.bl_pbl_physics) != UW_PBL_SCHEME:
        return
    if not cfg.sf_sfclay_physics:
        raise ConfigurationRefused(
            f"bl_pbl_physics={UW_PBL_SCHEME} (UW) requires a surface-layer "
            "scheme (sf_sfclay_physics != 0): the scheme's lower boundary "
            "is the friction velocity and the surface heat and moisture "
            "fluxes (UST, HFX, QFX; module_pbl_driver.F:1937-1939), and "
            "with the slot off nothing writes them, so the surface stress "
            "and both fluxes would stay the allocated zeros for the whole "
            "run. Select sf_sfclay_physics=1 (revised MM5), 91 (classic "
            "MM5), 5 (MYNN) 3 (GFS) or 7 (Pleim-Xiu).")
    urban = int(getattr(cfg, "sf_urban_physics", 0) or 0)
    if urban in (2, 3):
        raise ConfigurationRefused(
            f"bl_pbl_physics={UW_PBL_SCHEME} (UW) cannot run with "
            f"sf_urban_physics={urban} (BEP/BEM). WRF v4.7.1 fatals this "
            "pairing (module_physics_init.F:3826-3827: 'use ysu (option1), "
            "myj (option 2), or boulac (option 8) with BEP/BEM urban "
            "scheme'): the UW scheme has no arm for the multi-layer "
            "canopy's drag and heat source terms (the A_*_BEP/B_*_BEP "
            "coefficients), so the urban momentum and heat exchange would "
            "be dropped. Select bl_pbl_physics=1 (YSU) or 2 (MYJ) with "
            "BEP/BEM, or sf_urban_physics=1 (the single-layer UCM, which "
            "acts through the surface fluxes) with the UW PBL.")
    if cfg.bldt != 0.0 and int(cfg.mp_physics) in UW_PBL_ICE_NUMBER_SPECIES:
        raise UnsupportedConfiguration(
            f"bl_pbl_physics={UW_PBL_SCHEME} (UW) with mp_physics="
            f"{cfg.mp_physics} requires bldt=0, got {cfg.bldt!r}. The "
            "scheme's cloud-ice number tendency (RQNIBLTEN) is held "
            "outside the restart TENDENCY_COMPONENTS manifest and is "
            "restart-exact only when recomputed every step "
            "(woof/core/physics.py PhysicsTendencies.scalar_for, the "
            "MYNN mixscalars precedent); a positive cadence would make a "
            "mid-interval restart silently drop it. bldt=0 is WRF's "
            "default and its recommendation.")


def validate_gbmpbl_config(cfg: RunConfig) -> None:
    """Require WRF's MM5 layer and its cloud-water and cloud-ice species.

    Missing surface inputs corrupt surface stress. Missing condensate
    species discard the corresponding GBM tendency and water budget.
    """
    if int(cfg.bl_pbl_physics) != GBM_PBL_SCHEME:
        return
    sfclay = int(cfg.sf_sfclay_physics)
    if sfclay not in GBM_PBL_SURFACE_LAYERS:
        raise ValueError(
            f"bl_pbl_physics={GBM_PBL_SCHEME} (GBM) requires "
            "sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5), got "
            f"sf_sfclay_physics={sfclay}. WRF v4.6.1 fatals every other "
            "pairing at phys/module_physics_init.F:3892-3893: the scheme "
            "builds its friction velocity and surface stress from the MM5 "
            "layers' GZ1OZ0 and PSIM (module_bl_gbmpbl.F:539); the Eta "
            "layer publishes neither and with the slot off nothing does, so "
            "u* would be formed from allocated zeros.")
    from woof.core.physics_inventory import PBL_RQI_MICROPHYSICS
    if not cfg.moist or int(cfg.mp_physics) not in PBL_RQI_MICROPHYSICS:
        raise ValueError(
            "bl_pbl_physics=12 (GBM) requires F_QC and F_QI, with cloud-water "
            "and cloud-ice state (module_pbl_driver.F:1992-2018). "
            "The scheme mixes total water with saturation adjustment and "
            "diffuses ice separately; absent species would discard RQCBLTEN "
            "or RQIBLTEN and their condensate budget. Select a microphysics "
            "scheme that carries cloud water and cloud ice (moist=true), or "
            "another PBL.")


def validate_eeps_config(cfg: RunConfig) -> None:
    """Admission for the EEPS PBL (bl_pbl_physics=16).

    WRF's own law for the scheme (module_physics_init.F:3871-3876) is a
    surface-layer list (:data:`EEPS_SURFACE_LAYERS`) and the BEP/BEM urban
    fatal; the surface list is in the shared compatibility table, and the
    urban fatal is reproduced here. The Eta-layer cell preserves its own
    Registry-allocated zero TKE column.
    The one ArWen-side refusal is a dry state: EEPS's TKE and dissipation
    are advected scalars (PEK_ADV/PEP_ADV), and the dycore transports
    scalars only on a moist state.  Any microphysics is admitted: a scheme
    without cloud ice hands the column the driver's zero QI plane, which the
    column reads as an exact 0 (no melting, no ice diffusion) and whose
    RQIBLTEN the coupling does not apply.
    """
    if int(cfg.bl_pbl_physics) != EEPS_PBL_SCHEME:
        return
    # WRF's surface-layer law lives in PBL_SURFACE_LAYER_AUTHORITY and
    # require_ready_wrf_physics emits its complete shared refusal receipt.
    # Keep only the EEPS state/canopy constraints here.
    urban = int(getattr(cfg, "sf_urban_physics", 0) or 0)
    if urban in (2, 3):
        raise ValueError(
            f"bl_pbl_physics={EEPS_PBL_SCHEME} (EEPS) cannot run with "
            f"sf_urban_physics={urban} (BEP/BEM). WRF v4.6.1 fatals this "
            "pairing (module_physics_init.F:3874-3875: 'use myj (option 2) "
            "or boulac (option 8) with BEP/BEM urban scheme'): EEPS has no "
            "arm for the multi-layer canopy's drag and heat source terms, "
            "so the urban momentum and heat exchange would be dropped. "
            "Select sf_urban_physics=1 (the single-layer UCM, which acts "
            "through the surface fluxes) with EEPS.")
    if not cfg.moist:
        raise ValueError(
            f"bl_pbl_physics={EEPS_PBL_SCHEME} (EEPS) requires a moist "
            "state (moist=true): the scheme's TKE and dissipation rate are "
            "WRF scalar-array species (PEK_ADV/PEP_ADV, "
            "Registry.EM_COMMON:1053-1054) that the dycore advects between "
            "PBL calls, and a dry state runs no scalar transport, so the "
            "pair would never move.")
def qnse_mfshconv_active(cfg: RunConfig) -> bool:
    """Whether the EDMF shallow mass flux runs: ``mfshconv = 1`` under the
    QNSE PBL only.  WRF resets the switch to 0 for every other PBL
    (share/module_check_a_mundo.F:1715-1730, a NOTE, not a fatal), so a
    value carried under another scheme is inert here exactly as there."""
    return (int(cfg.bl_pbl_physics) == QNSE_PBL_SCHEME
            and int(getattr(cfg, "mfshconv", 1)) == 1)


def validate_qnse_config(cfg: RunConfig) -> None:
    """Admission for the QNSE PBL (bl_pbl_physics=4) and its mass flux.

    WRF's law, transcribed: ``CASE (QNSEPBLSCHEME)`` in
    phys/module_physics_init.F:3789-3793 fatals any surface layer that did
    not set ``isfc = 4`` ('use qnsesfc scheme for this pbl option') and
    the BEP/BEM urban schemes.  ``mfshconv`` is WRF's Registry integer
    (Registry.EM_COMMON:2485, default 1); the PBL driver tests it with
    ``.EQ.1`` (module_pbl_driver.F:1519).
    """
    mfshconv = int(getattr(cfg, "mfshconv", 1))
    if mfshconv not in (0, 1):
        raise ValueError(
            f"mfshconv must be 0 or 1, got {mfshconv}. WRF's PBL driver "
            "runs the EDMF mass flux only when the switch .EQ. 1 "
            "(module_pbl_driver.F:1519), so any other value silently means "
            "off there; refusing it keeps a typed 2 from running a "
            "different PBL than the one asked for.")
    if int(cfg.sf_sfclay_physics) == QNSE_SFCLAY_SCHEME and int(cfg.bl_pbl_physics) != QNSE_PBL_SCHEME:
        raise ValueError("sf_sfclay_physics=4 requires bl_pbl_physics=4: the QNSE surface layer scans carried TKE_PBL and publishes AKHS/AKMS, while other PBL paths do not consume its boundary state or write its TKE column.")
    if int(cfg.bl_pbl_physics) != QNSE_PBL_SCHEME:
        return
    sfclay = int(cfg.sf_sfclay_physics)
    if sfclay != QNSE_SFCLAY_SCHEME:
        raise ValueError(
            f"bl_pbl_physics={QNSE_PBL_SCHEME} (QNSE) requires "
            f"sf_sfclay_physics={QNSE_SFCLAY_SCHEME} (QNSE surface layer), "
            f"got sf_sfclay_physics={sfclay}. WRF v4.6.1 fatals this exact "
            "pairing at phys/module_physics_init.F:3790-3791 ('use qnsesfc "
            "scheme for this pbl option'): the QNSE PBL's implicit solves "
            "take AKHS/AKMS/THZ0/QZ0/UZ0/VZ0 as their lower boundary and "
            "only the QNSE surface layer produces them for it. No "
            "substitution is applied.")
    urban = int(getattr(cfg, "sf_urban_physics", 0) or 0)
    if urban in (2, 3):
        raise ValueError(
            f"bl_pbl_physics={QNSE_PBL_SCHEME} (QNSE) cannot run with "
            f"sf_urban_physics={urban} (BEP/BEM). WRF v4.6.1 fatals this "
            "pairing (module_physics_init.F:3792-3793: 'use ysu (option1), "
            "myj (option 2), or boulac (option 8) with BEP/BEM urban "
            "scheme'): QNSEPBL has no arm for the multi-layer canopy's drag "
            "and heat source terms, so the urban exchange would be dropped. "
            "Select sf_urban_physics=1 (the single-layer UCM) or 0.")

#: The surface layers whose published fields cover what the K-epsilon PBL
#: reads from its surface layer: UST, HFX, QFX, BR, ZNT and the stability
#: functions PSIM/PSIH (module_pbl_driver.F:1880-1902 binds the surface
#: driver's own psim/psih, not YSU's fm/fh).
KEPS_SURFACE_LAYERS = (1, 5, 91)


def validate_keps_config(cfg: RunConfig) -> None:
    """Admission for the K-epsilon PBL (bl_pbl_physics=17).

    WRF v4.7.1 has no law for the scheme to transcribe: bl_init's SELECT
    CASE has no KEPSSCHEME arm, so every surface layer and every urban
    option reaches it.  The refusals below are WOOF's and each names the
    breakage it prevents:

    * no surface layer, or the Eta layer: the scheme forms z/L from
      br*psim*psim/psih and its first-level gradient from sflux/ust
      (module_bl_keps.F:660, :1074); with the slot off UST is the allocated
      zero and the gradient divides by it, and the Eta layer publishes no
      PSIM/PSIH/BR (and is refused for every PBL but MYJ already, in
      :func:`validate_myj_pairing`);
    * gwd_opt: the drag suite reads the PBL top level KPBL, which this
      scheme never writes (WRF leaves it at whatever the last writer
      left), so the drag would be distributed against a stale level.

    The scheme receives the model DT at every bldt cadence, matching
    module_pbl_driver.F:1882.
    """
    if int(cfg.bl_pbl_physics) != KEPS_PBL_SCHEME:
        return
    sfclay = int(cfg.sf_sfclay_physics)
    if sfclay not in KEPS_SURFACE_LAYERS and sfclay != MYJ_SFCLAY_SCHEME:
        raise ValueError(
            f"bl_pbl_physics={KEPS_PBL_SCHEME} (K-epsilon) requires a "
            "surface layer that publishes UST, HFX, QFX, BR, ZNT, PSIM and "
            f"PSIH, got sf_sfclay_physics={sfclay}: the scheme's "
            "first-level temperature gradient divides the surface buoyancy "
            "flux by UST and its z/L divides by PSIH (module_bl_keps.F:660, "
            ":1074), and with no surface layer both are the allocated zeros, "
            "so the first K-epsilon call would put a NaN in every tendency "
            "of every column. Select sf_sfclay_physics=1 (revised MM5), 91 "
            "(classic MM5) or 5 (MYNN).")
    if int(getattr(cfg, "gwd_opt", 0) or 0):
        raise ValueError(
            f"gwd_opt={cfg.gwd_opt} cannot run with bl_pbl_physics="
            f"{KEPS_PBL_SCHEME} (K-epsilon): the drag suite distributes its "
            "stress up to the PBL top level KPBL, and the K-epsilon scheme "
            "never writes KPBL (module_bl_keps.F has no KPBL argument), so "
            "the drag would be placed against a level no scheme computed. "
            "Set gwd_opt=0, or use a PBL scheme that writes KPBL.")
    if int(cfg.cu_physics) == GRELL_FREITAS_CU_PHYSICS:
        raise ValueError(
            f"cu_physics={GRELL_FREITAS_CU_PHYSICS} (Grell-Freitas) cannot "
            f"run with bl_pbl_physics={KEPS_PBL_SCHEME} (K-epsilon): the "
            "trigger's temperature and moisture excesses and the shallow "
            "arm read the PBL top level KPBL, which the K-epsilon scheme "
            "never writes, so GF would see the allocated zero for the whole "
            "run. Select cu_physics=1 (Kain-Fritsch) or 16 (New Tiedtke), "
            "or a PBL scheme that writes KPBL.")


def validate_sase_config(cfg: RunConfig) -> None:
    """Admission for the SASE closure and its three knobs.

    Warn-not-block applies to *maturity*, never to coherence: an
    experimental scheme still refuses a configuration it cannot accurately
    execute.  What is refused here is only what the closure genuinely
    needs -- see :data:`_SASE_REQUIREMENTS` for why the development
    lane's wider whitelist is not reproduced.
    """
    validate_myj_pairing(cfg)
    sase = cfg.bl_pbl_physics == SASE_PBL_SCHEME
    if sase:
        for name, admitted, why in _SASE_REQUIREMENTS:
            value = getattr(cfg, name)
            if value != admitted:
                raise ConfigurationRefused(
                    f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) is not "
                    f"admitted with {name}={value!r}: {why}. Set "
                    f"{name}={admitted!r}.")
        if not cfg.sf_sfclay_physics:
            # Not a taste question: the closure reads u*, the heat and
            # moisture fluxes and the gust-corrected wind speed from the
            # surface layer, and its lower boundary condition is those
            # four fields.  With the slot off they do not exist.
            raise ConfigurationRefused(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) requires a "
                "surface-layer scheme (sf_sfclay_physics != 0): the "
                "closure's lower boundary condition is the surface "
                "layer's friction velocity, heat and moisture fluxes and "
                "gust-corrected wind speed. Select sf_sfclay_physics=1 "
                "(revised MM5) or 91 (classic MM5), which are the "
                "surface layers SASE's registry option declares.")
        # SASE WITH THE MYNN SURFACE LAYER IS ADMITTED.  The refusal that
        # stood here read WRF's 16-cell isfc matrix as covering the pair;
        # it does not and cannot.  That matrix is an isfc-class check
        # inside module_physics_init's PBL SELECT CASE, and SASE is not a
        # WRF package at all -- bl_pbl_physics=900 is outside
        # PBL_OPTIONS, so wrf461_compatibility.pbl_surface_layer_verdict
        # RAISES for it rather than returning a verdict.  Intersecting the
        # MYNN option's WRF-transcribed pbl list with SASE's is arithmetic
        # on two tables, not a physical reason.  The physical question is
        # whether SASE's four lower-boundary fields exist, and the MYNN
        # surface layer publishes all four: ust, hfx, qfx and wspd are in
        # MYNN_SURFACE_OUTPUTS, and MynnSurfaceResult is allocated on
        # sf_sfclay_physics=5 alone, independent of the PBL selector.
        # The pairing is unmeasured, which is maturity: this module's own
        # rule is that maturity warns and coherence refuses, so it carries
        # SASE's registry maturity warnings and no refusal.
        if not cfg.moist:
            raise ConfigurationRefused(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) requires "
                "moist=true: the closure mixes water vapour, cloud water "
                "and cloud ice alongside potential temperature and forms "
                "its stability from the saturated Brunt-Vaisala "
                "frequency, which a dry run cannot supply. Set "
                "moist=true, or select bl_pbl_physics=1, 2, 5 or 11 (YSU, "
                "MYJ, MYNN or Shin-Hong), which run dry.")
        if cfg.nz > SASE_MAX_NZ:
            # A compile-time bound, not a policy: the vertical solve keeps
            # three FP64 columns of this depth in per-thread local memory.
            raise ConfigurationRefused(
                f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE) is limited to "
                f"nz <= {SASE_MAX_NZ}, got nz={cfg.nz}: the closure's "
                "implicit vertical solve carries its tridiagonal columns "
                "in per-thread local memory at that fixed depth.")
    # The three knobs are fail-closed on their NON-default value only, so
    # every existing configuration keeps validating unchanged.
    for name, what in (
            ("sase_flux_diag",
             "the split subgrid-flux diagnostic records the SASE venting "
             "and K_v channels, which no other turbulence path computes"),
            ("sase_moist_n2",
             "the moist-N2 substitution it disables exists only inside "
             "the SASE closure, so switching it off elsewhere would "
             "disable nothing"),
            ("sase_stable_dissipation",
             "the stable-limb dissipation coefficient it decouples lives "
             "in the SASE analytic decay substep, so setting it "
             "elsewhere would decouple nothing")):
        default = SASE_FAIL_CLOSED_DEFAULTS[name]
        value = getattr(cfg, name)
        if type(value) is not bool:
            raise ConfigurationRefused(f"{name} must be boolean, got {value!r}.")
        if value != default and not sase:
            raise ConfigurationRefused(
                f"{name}={value!r} requires bl_pbl_physics="
                f"{SASE_PBL_SCHEME} (SASE), got "
                f"bl_pbl_physics={cfg.bl_pbl_physics}: {what} -- a key "
                "that names a seam this run does not have would read as "
                "a setting that took effect.")
    # sase_additive_dissipation left the fail-closed loop when its
    # default flipped (False -> True, 2026-08-16, 1a0e8a7f8): the loop's
    # charter is "every existing configuration keeps validating
    # unchanged", and artifacts written under the old default RECORD
    # ``sase_additive_dissipation = false`` beside non-SASE PBLs -- every
    # 2.4.x restart header and every child TOML 2.4.x downscale rendered.
    # Refusing the old default broke `woof downscale` of a 2.4.1
    # archive on this tree (MEASURED 2026-08-17, masked as a VRAM-fit
    # refusal).  Off-SASE the knob is inert in BOTH positions, so only
    # the type is held; on SASE both positions are legal channels.
    value = cfg.sase_additive_dissipation
    if type(value) is not bool:
        raise ConfigurationRefused(
            f"sase_additive_dissipation must be boolean, got {value!r}.")


#: ``name -> (predicate, what the value has to be)`` for the dynamics
#: coefficients.  Each predicate is the range the *scheme* is defined
#: on, not a taste bound: outside it the solver is not being used, it is
#: being misused, and the answer that comes back is confident garbage.
#:
#: These keys reached ``forecast validity PASS`` at exit 0 through both
#: loaders in v1.4.0 -- ``epssm = 5.0`` moved 2 m temperature 2.2 K in
#: two simulated hours, and ``dampcoef = -5.0`` turned the upper damping
#: layer into an energy source.  Every value the wizard emits, every
#: value in configs/, and every WRF Registry default passes.
_DYNAMICS_RANGES: dict[str, tuple] = {
    # Forward off-centring weight of the vertically implicit acoustic
    # solve (module_small_step_em.F): a weight in (0, 1].
    "epssm": (lambda v: 0.0 < v <= 1.0,
              "an acoustic off-centring weight in (0, 1] "
              "(WRF Registry default 0.1)"),
    # 3-D divergence damping coefficient; 0 disables it.
    "smdiv": (lambda v: v >= 0.0,
              "a non-negative divergence-damping coefficient "
              "(WRF Registry default 0.1)"),
    # External-mode divergence damping (mudf); 0 disables it.
    "emdiv": (lambda v: v >= 0.0,
              "a non-negative external-mode damping coefficient "
              "(WRF Registry default 0.01)"),
    # Rayleigh/implicit-w damping strength.  Negative is ANTI-damping:
    # the layer injects the energy it exists to remove.
    "dampcoef": (lambda v: v >= 0.0,
                 "a non-negative damping rate in 1/s "
                 "(WRF Registry default 0.2); a negative coefficient is "
                 "anti-damping, and the layer adds energy instead of "
                 "removing it"),
    # Depth of that layer below the model top; _damp_factors divides by
    # it, so 0 is a division by zero and negative inverts the profile.
    "zdamp": (lambda v: v > 0.0,
              "a positive damping-layer depth in m "
              "(WRF Registry default 5000)"),
    # Smagorinsky constant.
    "c_s": (lambda v: v > 0.0,
            "a positive Smagorinsky constant (WRF default 0.25)"),
    # Model-top height: the damping profile is measured down from it.
    "ztop": (lambda v: v > 0.0,
             "a positive model-top height in m"),
    "p_surf": (lambda v: v > 0.0,
               "a positive reference surface pressure in Pa"),
    "base_temp": (lambda v: v > 0.0,
                  "a positive base-state reference temperature in K"),
    # Exponential lateral-sponge exponent (WRF spec_exp).
    "spec_exp": (lambda v: v >= 0.0,
                 "a non-negative lateral-sponge exponent "
                 "(WRF Registry default 0.0)"),
    # Davies time scale on the first relaxed row; 0 is WRF's 10 x dt.
    "relax_timescale_s": (lambda v: v >= 0.0,
                          "a non-negative relaxation time scale in s "
                          "(0 keeps WRF's 10 x dt)"),
    "diff_6th_factor": (lambda v: v >= 0.0,
                        "a non-negative 6th-order diffusion factor "
                        "(WRF Registry default 0.12)"),
}

#: Dynamics selectors, with the values this tree IMPLEMENTS.
#: A selector that is a real WRF choice but unimplemented here is the
#: worst of the three cases: every damping site tests ``damp_opt == 3``,
#: so ``damp_opt = 2`` used to run with the damping layer switched off
#: while the config said it was on.
_DYNAMICS_CHOICES: dict[str, tuple] = {
    "upper_wind_limiter_form": (("wrf_461", "noaa_wrf39"),
        "'wrf_461' (no upper-wind limiter) or 'noaa_wrf39' (the "
        "WRFV3.9 fork's 110 m/s limiter in the damping layer)"),
    "diff_6th_form": (("wrf_461", "noaa_wrf39"),
                      "'wrf_461' (WRF v4.6.1 sixth_order_diffusion) or "
                      "'noaa_wrf39' (the NOAA-EMC WRFV3.9 fork: a second "
                      "factor on the full step for moisture and scalars, "
                      "filtered to the domain edge); the two filter "
                      "moisture by different amounts, so an unknown name "
                      "cannot select either"),
    "mp_zero_out": ((0, 1, 2),
                    "0 (off), 1 (zero non-vapour species below "
                    "mp_zero_out_thresh) or 2 (also floor vapour at "
                    "zero), WRF module_microphysics_zero_out; another "
                    "value would run as 0 in WRF's routine"),
    "mp_zero_out_all": ((0, 1),
                        "0 (moist array only, WRF v4.6.1 default) or 1 "
                        "(also the scalar number arrays, as the NOAA "
                        "WRFV3.9 fork always does)"),
    "solar_generation": (("wrf_480", "wrf_461"),
                         "WRF 4.8 corrected solar phase or WRF 4.6.1 reproduction; unknown phases change COSZEN"),
    "zadvect_implicit_variant": (("wrf_471", "wrf_legacy"),
                                 "'wrf_471' (module_ieva_em) or "
                                 "'wrf_legacy' (module_advect_em); the "
                                 "split and column masses differ, so an "
                                 "unknown variant cannot be substituted"),
    "damp_opt": ((0, 3),
                 "0 (no upper damping) or 3 (the Klemp-Dudhia-Hassiotis "
                 "implicit w-only damper); WRF's damp_opt 1 (diffusive) "
                 "and 2 (Rayleigh relaxation to the base state) are not "
                 "implemented here and would silently run undamped"),
    "w_damping": ((0, 1),
                  "0 (off) or 1 (WRF's per-stage vertical-velocity "
                  "limiter)"),
    "zadvect_implicit": ((0, 1),
                         "0 (explicit vertical advection) or 1 (WRF's "
                         "implicit-explicit vertical advection on the last "
                         "RK substep); WRF 4.7.1 runs 2 and 3 exactly as 1, "
                         "their other-substep variants being commented out "
                         "of CHK_IEVA, so write 1"),
}


def w_crit_cfl_refusal(w_crit_cfl, *, w_damping: int,
                       zadvect_implicit: int) -> str | None:
    """Why ``w_crit_cfl`` cannot run beside these switches, or None (A165).

    One authority for :func:`_validate_dynamics_coefficients` and the
    namelist importer.  WRF 4.7.1 ``w_damp`` starts at ``w_crit_cfl``
    under ``zadvect_implicit`` and at w_beta = 1 without it, and measures
    the excess from ``w_crit_cfl`` either way
    (module_big_step_utilities_em.F:2601-2607, :2689).
    """
    try:
        value = float(w_crit_cfl)
    except (TypeError, ValueError):
        return "must be a number (WRF Registry default 1.0)."
    if (isinstance(w_crit_cfl, bool) or not math.isfinite(value)
            or value <= 0.0):
        # w_damp acts where vert_cfl > its onset: at zero or below, under
        # zadvect_implicit, that is every interior w level of every column.
        return ("must be a positive vertical Courant number (WRF Registry "
                "default 1.0; WRF recommends 2.0 with zadvect_implicit = "
                "1): at zero or below w_damp drags every vertical motion, "
                "not just the fast ones.")
    if int(w_damping) == 1 and not int(zadvect_implicit) and value > 1.0:
        return ("with w_damping = 1 and zadvect_implicit = 0 turns WRF's "
                "w-damping into an amplifier: without zadvect_implicit it "
                "starts at vertical Courant number 1 but measures the excess "
                "from w_crit_cfl, so between 1 and w_crit_cfl it pushes w "
                "along its own direction.  Set zadvect_implicit = 1 (WRF "
                "recommends w_crit_cfl = 2.0 there) or w_crit_cfl = 1.0 "
                "(WRF's setting without it).")
    return None


def _validate_dynamics_coefficients(cfg: RunConfig) -> None:
    """Refuse dynamics coefficients outside the range they are defined on.

    The value checks the fleet found missing.  Kept in one table so the
    next coefficient added to :class:`RunConfig` has an obvious place to
    declare what it means, rather than joining the set of keys that
    reach the solver unexamined.
    """

    for name, (ok, wanted) in _DYNAMICS_RANGES.items():
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or not ok(value):
            raise ConfigurationRefused(
                f"{name} = {value!r} must be {wanted}.")
    for name, (allowed, wanted) in _DYNAMICS_CHOICES.items():
        value = getattr(cfg, name)
        if value not in allowed:
            raise ConfigurationRefused(f"{name} must be {wanted}, got {value!r}.")
    if not isinstance(cfg.time_step_sound, int) \
            or isinstance(cfg.time_step_sound, bool) \
            or cfg.time_step_sound < 1:
        raise ConfigurationRefused(
            "time_step_sound must be a positive number of acoustic "
            "substeps per dynamics step (WRF Registry default 4), got "
            f"{cfg.time_step_sound!r}.")
    # The damping layer is measured DOWN from the model top; a layer
    # deeper than the column damps every level, including the ground.
    if cfg.damp_opt == 3 and cfg.zdamp >= cfg.ztop > 0.0:
        raise ConfigurationRefused(
            f"zdamp = {cfg.zdamp} is the depth of the upper damping "
            f"layer below the model top ztop = {cfg.ztop}, so it has to "
            "be smaller than it; as written the damper covers the whole "
            "column.")
    why = w_crit_cfl_refusal(cfg.w_crit_cfl, w_damping=cfg.w_damping,
                             zadvect_implicit=cfg.zadvect_implicit)
    if why is not None:
        raise ConfigurationRefused(f"w_crit_cfl = {cfg.w_crit_cfl!r} {why}")


#: WRF v4.7.1 topo_wind and gwd_opt values ported
#: (:mod:`woof.core.terrain_drag`).
TOPO_WIND_VALUES = (0, 1, 2)
GWD_OPT_VALUES = (0, 1, 3)
#: The bl_pbl_physics values whose scheme reads CTOPO/CTOPO2 in WRF v4.6.1
#: (module_pbl_driver.F passes them to YSU and to Shin-Hong only).
TOPO_WIND_PBL_SCHEMES = (1, 11)


def terrain_drag_refusal(*, topo_wind, gwd_opt, bl_pbl_physics,
                         sf_urban_physics=0) -> str | None:
    """Why ``topo_wind``/``gwd_opt`` cannot run as set, or None.

    Shared by :func:`validate_run_config` and the namelist importer.
    """
    for name, value, allowed, what in (
            ("topo_wind", topo_wind, TOPO_WIND_VALUES,
             "0 off, 1 Jimenez-Dudhia, 2 the VAR form"),
            ("gwd_opt", gwd_opt, GWD_OPT_VALUES,
             "0 off, 1 the KIM drag, 3 the GSL drag suite; WRF's 2 is not "
             "ported")):
        if isinstance(value, bool) or value not in allowed:
            return f"{name} = {value!r} is not one of {allowed} ({what})"
    if topo_wind and int(bl_pbl_physics) not in TOPO_WIND_PBL_SCHEMES:
        return (f"topo_wind = {topo_wind} acts only through the surface drag "
                "of the two PBL schemes that read CTOPO/CTOPO2 (WRF v4.6.1: "
                "YSU, bl_ysu.F90:1308-1314, and Shin-Hong, "
                "module_bl_shinhong.F:1387 and :1472-1473), and "
                f"bl_pbl_physics = {bl_pbl_physics} is neither: the "
                "coefficients would be built and nothing would read them, so "
                "the run would differ from topo_wind = 0 in its receipt only.  "
                "Select bl_pbl_physics = 1 or 11, or use gwd_opt, which works "
                "under every PBL scheme")
    if topo_wind and int(sf_urban_physics) in (2, 3):
        return (f"topo_wind = {topo_wind} scales YSU's first-level drag, "
                "and under sf_urban_physics = "
                f"{sf_urban_physics} (BEP) that drag is removed for the "
                "declared rural-drag divergence (kernels/ysu.cu "
                "ysu_column_bep), so there is no drag left to scale")
    if gwd_opt and int(bl_pbl_physics) == 0:
        return (f"gwd_opt = {gwd_opt} adds its drag inside WRF's PBL driver "
                "(module_pbl_driver.F), which a run without a PBL scheme "
                "never calls, and it reads the PBL height and top level the "
                "scheme writes")
    return None


#: PBL schemes that produce horizontal mixing of their OWN.
#:
#: The whole content of the ``km_opt = 0`` question: with ``km_opt = 0``
#: the dycore runs no horizontal mixing operator, so whether the run has
#: horizontal mixing at all depends entirely on whether something else
#: produces it.  Exactly one scheme in this tree does -- SASE, whose
#: governed horizontal diffusivity serves its stress, its subgrid-energy
#: transport and its scalar channel.  YSU, MYNN and PBL-off are all
#: vertical-only, and on any of them ``km_opt = 0`` means no horizontal
#: mixing operator anywhere.
HMIX_PRODUCING_PBL_SCHEMES: tuple[int, ...] = (SASE_PBL_SCHEME,)


def constant_k_mixing_active(cfg) -> bool:
    """Whether ``km_opt = 1`` runs WRF's diff_opt = 2 mixing package.

    WRF's ``km_opt = 1`` is not a separate operator: ``calculate_km_kh``
    calls ``isotropic_km`` (module_diffusion_em.F:1718-1772), which fills
    xkmh = khdif, xkmv = kvdif, xkhh = khdif/prandtl and xkhv =
    kvdif/prandtl, and the SAME ``horizontal_diffusion_2`` and (PBL off)
    ``vertical_diffusion_2`` that serve km_opt 2-4 then mix every dry,
    moist, chem and tracer row once per step from the time-t fields
    (module_first_rk_step_part2.F:419-1100).

    The package is skipped only where WRF's own words would all be zero:
    both constants zero AND either a PBL scheme on (vertical_diffusion_2
    is not called) or no surface forcing ``vertical_diffusion_2`` could
    apply.  Skipping then changes nothing but the sign of an exact zero
    tendency word, and it keeps the default configuration (km_opt = 1,
    khdif = kvdif = 0) free of a full-grid package that computes zeros.
    """
    if getattr(cfg, "diff_opt", 2) == 0 or int(getattr(cfg, "km_opt", 0)) != 1:
        return False
    if cfg.khdif != 0.0 or cfg.kvdif != 0.0:
        return True
    if getattr(cfg, "diff_opt", 2) != 2 or cfg.bl_pbl_physics != 0:
        return False
    isfflx = getattr(cfg, "isfflx", 1)
    heat = getattr(cfg, "tke_heat_flux", 0.0)
    drag = getattr(cfg, "tke_drag_coefficient", 0.0)
    if isfflx == 0:
        return heat != 0.0 or drag != 0.0
    if isfflx == 2 and heat != 0.0:
        return True
    return getattr(cfg, "sf_sfclay_physics", 0) != 0


def wrf_mixing_package_active(cfg) -> bool:
    """Whether the dycore builds WRF's once-per-step mixing tendencies.

    km_opt 2/3/4/5 always; km_opt 1 under :func:`constant_k_mixing_active`.
    """
    return (int(getattr(cfg, "km_opt", 0)) in (2, 3, 4, 5)
            or constant_k_mixing_active(cfg))


def km_opt_zero_producer(cfg: RunConfig) -> str | None:
    """What supplies horizontal mixing at ``km_opt = 0``, or None."""
    if cfg.bl_pbl_physics == SASE_PBL_SCHEME:
        return (f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE), whose closure "
                "supplies the mixing the km_opt operator would otherwise "
                "apply")
    return None


def validate_km_opt(cfg: RunConfig) -> None:
    """Admission for ``km_opt``, shared by the loaders and the dycore.

    ONE function so the loader's refusal and the dycore's refusal cannot
    drift: the dycore's is the fail-closed one (it is what actually
    decides whether a mixing operator runs), and a config that got past
    the loader must get past it too.

    ``km_opt = 0`` is admitted on two distinct grounds, and they are not
    the same kind of thing:

    * A PBL scheme in :data:`HMIX_PRODUCING_PBL_SCHEMES` supplies the
      horizontal mixing itself.  Nothing is missing, so nothing is
      acknowledged -- the operator is redundant, and running it too
      would double-count.
    * :data:`KM_OPT_ZERO_ACK`, written out in full by the user.  Here
      the horizontal mixing operator really is absent and the run really
      does have none, which is the RESEARCH CONTROL the acknowledgement
      exists for and the thing the default refusal exists to prevent
      happening by accident.  See :data:`KM_OPT_ZERO_ACK`.
    """
    if isinstance(cfg, _DfiBackwardRunConfig):
        if (cfg.dt < 0 and cfg.diff_opt == 0 and cfg.km_opt == 1 and
                not any((cfg.mp_physics, cfg.ra_physics, cfg.ra_lw_physics,
                         cfg.ra_sw_physics, cfg.sf_surface_physics,
                         cfg.sf_sfclay_physics, cfg.bl_pbl_physics,
                         cfg.cu_physics, cfg.sf_urban_physics, cfg.damp_opt,
                         cfg.gwd_opt, cfg.diff_6th_opt))):
            return
        raise ConfigurationRefused('invalid internal backward DFI phase configuration')
    if isinstance(cfg.diff_opt, bool) or not isinstance(cfg.diff_opt, int) or cfg.diff_opt not in (1, 2):
        raise ConfigurationRefused(
            f"diff_opt must be 1 (model-coordinate horizontal diffusion) "
            f"or 2 (metric stress/scalar diffusion), got {cfg.diff_opt}")
    if not isinstance(cfg.mix_full_fields, bool):
        raise ConfigurationRefused("mix_full_fields must be a boolean")
    if cfg.diff_opt == 1 and cfg.km_opt not in (2, 4):
        raise ConfigurationRefused(
            f"diff_opt=1 requires km_opt=2 or 4; km_opt={cfg.km_opt} "
            "does not supply the implemented model-coordinate operator's "
            "exchange coefficients")
    ack = cfg.km_opt_zero_acknowledgement
    producer = km_opt_zero_producer(cfg)
    if not isinstance(ack, str):
        raise ConfigurationRefused(
            "km_opt_zero_acknowledgement must be the exact string "
            f"{KM_OPT_ZERO_ACK!r} or absent, got {ack!r}.")
    if ack and ack != KM_OPT_ZERO_ACK:
        raise ConfigurationRefused(
            f"km_opt_zero_acknowledgement={ack!r} is not the "
            f"acknowledgement id. Write it exactly: "
            f"km_opt_zero_acknowledgement = {KM_OPT_ZERO_ACK!r}. It is a "
            "literal id and not a boolean so that it cannot be reached "
            "by a stray `= true`.")
    if cfg.km_opt not in (0, 1, 2, 3, 4, 5):
        # Checked BEFORE the acknowledgement's placement, so an
        # unimplemented selector reports itself rather than reporting
        # that the acknowledgement is in the wrong place: the id opens
        # km_opt = 0 and nothing else, and cannot shadow this.
        raise ConfigurationRefused(
            f"km_opt must be 1 (constant K via khdif/kvdif), 2 (1.5-order "
            f"prognostic TKE), 3 (3-D Smagorinsky), or 4 "
            f"(2-D Smagorinsky), or 5 (SMS-3DTKE), got {cfg.km_opt}. 0 is admitted with "
            f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE), whose closure "
            "supplies the mixing the km_opt operator would otherwise "
            f"apply, or with km_opt_zero_acknowledgement = "
            f"{KM_OPT_ZERO_ACK!r}.")
    if ack and cfg.km_opt != 0:
        raise ConfigurationRefused(
            f"km_opt_zero_acknowledgement is set with km_opt={cfg.km_opt}, "
            "which runs a horizontal mixing operator; the acknowledgement "
            "admits km_opt = 0 and acknowledges nothing here. A key that "
            "names a seam this run does not have would read as a setting "
            "that took effect.")
    if ack and producer is not None:
        raise ConfigurationRefused(
            "km_opt_zero_acknowledgement is set with "
            f"{producer.split(',')[0]}, which already supplies the "
            "horizontal mixing; km_opt = 0 is admitted here without any "
            "acknowledgement, and the id would record an absence this "
            "run does not have.")
    if cfg.tke_adv_opt != 1:
        raise ConfigurationRefused("tke_adv_opt must be 1: only WRF positive-definite TKE transport is implemented")
    for name in ("tke_mix2_off", "tke_mix6_off"):
        if not isinstance(getattr(cfg, name), bool):
            raise ConfigurationRefused(f"{name} must be a boolean")
    if isinstance(cfg.rk_ord, bool) or cfg.rk_ord not in (2, 3):
        raise ConfigurationRefused("rk_ord must be 2 or 3: WRF solve_em's supported Runge-Kutta solvers")
    if isinstance(cfg.momentum_adv_opt, bool) or cfg.momentum_adv_opt not in (1, 3):
        raise ConfigurationRefused("momentum_adv_opt must be 1 (standard) or 3 (WENO)")
    if cfg.sfs_opt != 0:
        raise ConfigurationRefused("sfs_opt must be 0: nonlinear backscatter stresses are not implemented")
    if isinstance(cfg.m_opt, bool) or cfg.m_opt not in (0, 1):
        raise ConfigurationRefused("m_opt must be 0 (off) or 1 (WRF stress history fields)")
    if cfg.km_opt == 5:
        # WRF 4.6.1 module_check_a_mundo.F:341-402,1351-1364.
        if cfg.diff_opt != 2:
            raise ConfigurationRefused("km_opt=5 requires diff_opt=2: SMS-3DTKE uses metric stresses and implicit vertical diffusion")
        if cfg.bl_pbl_physics != 0:
            raise ConfigurationRefused("km_opt=5 requires bl_pbl_physics=0: SMS-3DTKE supplies the boundary-layer turbulence closure, so a PBL scheme would mix the same boundary layer a second time. Set bl_pbl_physics=0, or select another km_opt.")
        if cfg.sf_sfclay_physics not in (0, 1, 5, 91):
            raise ConfigurationRefused("km_opt=5 requires sf_sfclay_physics in {0,1,5,91}, the surface flux interfaces supported by WRF SMS-3DTKE")
        if cfg.cu_physics == 11:
            raise ConfigurationRefused("km_opt=5 cannot use cu_physics=11: WRF's scale-aware Kain-Fritsch scheme requires a separate PBL scheme")
    if cfg.km_opt == 3 and cfg.bl_pbl_physics != 0:
        # The registry says the same thing from the other side
        # (physics_registry_v2.json#/components/turbulence/options/
        # smagorinsky-3d declares required_settings bl_pbl_physics=0 and
        # requires_components pbl=[off]), and until this refusal existed
        # the two authorities disagreed about 3,360 of the 46,080
        # component combinations tests/test_authority_agreement.py
        # enumerates: the registry refused and validate_run_config said
        # OK.  The registry was right.  What makes km_opt=3 different
        # from km_opt=4 is that it computes a genuinely SEPARATE
        # vertical exchange pair (kmv/khv) rather than reusing the
        # horizontal one, and that pair is applied by
        # vertical_diffusion_2, which is PBL-off-gated exactly as the
        # km_opt=2 branch below describes.  So with a PBL scheme on, the
        # vertical half of the closure the user selected does not run at
        # all and the selection quietly means something narrower than
        # its name.  Refusing is the no-silent-override rule.
        raise ConfigurationRefused(
            "km_opt=3 (3-D Smagorinsky) is admitted with "
            "bl_pbl_physics=0 only. Its vertical exchange pair "
            "(kmv/khv) is applied by vertical_diffusion_2, which is "
            "PBL-off-gated, so with a PBL scheme on only the horizontal "
            "half of the closure would run and the run would not be the "
            "3-D closure it names. Select bl_pbl_physics=0 for an LES "
            "domain, or km_opt=4 (2-D Smagorinsky), which is the "
            "horizontal-only closure every PBL-on template pins.")
    if cfg.km_opt == 2:
        if cfg.diff_opt == 2 and cfg.bl_pbl_physics != 0:
            raise ConfigurationRefused(
                "km_opt=2 (prognostic TKE) is admitted with "
                "bl_pbl_physics=0 only. Its vertical TKE self-diffusion "
                "and its surface TKE forcing are applied by "
                "vertical_diffusion_2, which is PBL-off-gated because a "
                "PBL scheme is already the column's vertical closure and "
                "running both would double-count vertical mixing. With a "
                "PBL scheme on, TKE would be produced and dissipated "
                "column-locally with no vertical redistribution and no "
                "surface source, so the run would not be the "
                "prognostic-TKE closure it names. Select "
                "bl_pbl_physics=0 for an LES domain (this is a per-domain "
                "setting: an LES child under a PBL-on parent is "
                "admitted), or km_opt=4 (2-D Smagorinsky), the "
                "horizontal-only closure every PBL-on template pins. "
                "km_opt=3 is NOT an alternative with a PBL scheme on: it "
                "is refused three lines above for the same gate."
            )
        # km_opt=2 on a NEST child is no longer refused here.  This
        # function sees one domain at a time and cannot see the parent,
        # and the parent is what decides whether the question is even
        # live: WRF gives tke no ``i`` (nest-interpolation) and no ``f``
        # (feedback) Registry flag (Registry.EM_COMMON:312), so a child
        # cold-starts its own TKE and never returns it.  Under a parent
        # that carries no TKE at all there is nothing to interpolate and
        # nothing to feed back, and that case has now been run and scored
        # (7 h, 250 m child under a km_opt=4 parent, status PASS; see
        # docs/superpowers/receipts/les/nested-les-km2-2026-08-02.md).
        # A km_opt=2 child under a km_opt=2 parent does the same thing
        # (the parent's TKE stays on the parent); woof.experiment, the
        # only place that knows the parent, admits that tree with a
        # not-yet-verified warning.
    # mix_full_fields = false under diff_opt 2 is admitted at either value.
    # A refusal used to stand here ("this run would mix full fields while
    # its configuration says otherwise").  It retired with the reading of
    # the operational HRRR source (NOAA-EMC/HRRR v4.1.21, WRFV3.9): WRF's
    # perturbation branch subtracts the 1-D base-state wind profiles
    # u_base/v_base from du/dz and dv/dz in cal_deform_and_div
    # (dyn_em/module_diffusion_em.F:842-860 and :1017-1035) and the 1-D
    # t_base/qv_base profiles from theta and qv in vertical_diffusion_2
    # (:3694-3712 and :3778-3792), which module_first_rk_step_part2.F:
    # 925-927 calls only with bl_pbl_physics = 0.  real.exe never assigns
    # any of the four profiles (Registry.EM_COMMON:372 and :1247-1249 name
    # them "BASE STATE ... IN IDEALIZED CASES"; only the
    # module_initialize_<ideal>.F routines and module_force_scm.F write
    # them) and WRF's allocation zero-fills them
    # (frame/wrf_num_bytes_between.c:43-48 through
    # frame/module_alloc_space.h:139, head grid setinitval 3 at
    # frame/module_domain.F:782), so for every real-data initialization
    # the perturbation operand is ``((u - 0) - u(k-1)) + 0``: the
    # full-field arithmetic to the bit (the sign of an exact zero aside:
    # compiled WRF cal_deform_and_div, byte-identical in the fork and
    # v4.7.1, differs between the two branches in 7 of 966,854 tensor
    # words over the 14 deformation fixtures, all of them signed zeros in
    # D23 of one zero-flow fixture; lane 286-mix-full-fields receipt; with
    # a nonzero profile 237,319 D13/D23 words move).  The metric
    # operator therefore IS WRF's operator at either value for real-data
    # runs, and the value is recorded as declared.  The one input that
    # would differ, an ideal.exe wrfinput whose base-state profile is not
    # zero, is refused by name at the wrfinput door
    # (woof/ingest/wrfinput.py, require_zero_base_state_profiles).  WRF
    # calls cal_deform_and_div under diff_opt 1 too
    # (module_first_rk_step_part2.F:448), but under diff_opt 1 only
    # smag_km (km_opt 3, refused just above) reads D13/D23: tke_rhs runs
    # only under diff_opt 2 (:888).  diff_opt 1 with no PBL scheme also
    # subtracts u_base/v_base/qv_base at either value, scaled by kvdif
    # (module_em.F:844-853, :1381-1385), and kvdif > 0 is refused for
    # km_opt 2-4 (the khdif/kvdif check in validate_config).
    # base_state_profiles_read names every one of those paths, so the
    # door stays exact if either refusal retires.  The
    # engine's own idealized cases (``woof verify``) run fixed
    # configurations with mix_full_fields = true and take no override.
    if cfg.km_opt in (1, 2, 3, 4, 5):
        return
    if producer is None and ack != KM_OPT_ZERO_ACK:
        # The sixth-order filter is gated on diff_6th_opt ALONE
        # (woof/core/dycore.py: include_diff6 = cfg.diff_6th_opt > 0,
        # and the tendency rows on ``km_opt not in (2, 3, 4) and
        # diff_6th_opt <= 0``), so km_opt = 0 does not turn it off.  The
        # refusal states the run it is actually refusing: with the filter
        # on there IS something acting on horizontal structure, and
        # saying otherwise would be a false reason a reader can refute.
        if cfg.diff_6th_opt > 0:
            filter_clause = (
                "the dycore skips the whole mixing package at km_opt = 0, "
                "and the only thing left acting on horizontal structure is "
                "the sixth-order numerical filter, which this run does have "
                f"on (diff_6th_opt={cfg.diff_6th_opt}; that filter is gated "
                "on diff_6th_opt alone, independent of km_opt) -- a 2dx "
                "noise filter, not a mixing closure, and not a substitute "
                "for one")
        else:
            filter_clause = (
                "the dycore skips the whole mixing package at km_opt = 0, "
                "and diff_6th_opt is 0 on this run as well, so not even the "
                "sixth-order numerical filter is on")
        raise ConfigurationRefused(
            f"km_opt = 0 runs NO horizontal mixing operator, and "
            f"bl_pbl_physics={cfg.bl_pbl_physics} produces none of its "
            "own, so this run would carry NO explicit horizontal "
            f"mixing at all: {filter_clause}. That is "
            "refused by default because it is what a mis-set switch "
            "looks like. It is also a legitimate research control -- it "
            "is WRF's own diff_opt = 0, and it is the only way to vary "
            "the PBL closure while holding the horizontal mixing fixed "
            "at none. To run it deliberately, write the acknowledgement "
            f"out in full: km_opt_zero_acknowledgement = "
            f"{KM_OPT_ZERO_ACK!r}. Otherwise set km_opt = 4 (2-D "
            f"Smagorinsky) or 1 (constant K via khdif/kvdif), or select "
            f"bl_pbl_physics={SASE_PBL_SCHEME} (SASE), which supplies "
            "horizontal mixing from its own closure.")


#: Stability limit of an EXPLICIT horizontal Laplacian on the shortest
#: representable mode.  Forward-in-time diffusion of a 2-grid-interval
#: wave multiplies it by ``1 - 4*K*dt/dx^2`` each step, so ``K*dt/dx^2``
#: above 1/4 flips the sign and above 1/2 grows the wave: a diffusion
#: operator that amplifies what it exists to remove.
EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT = 0.25

#: The second threshold on the same per-step factor, and a different
#: statement: between 1/4 and 1/2 the factor is negative but still inside
#: the unit circle, so the 2-grid-interval mode flips sign each step and
#: shrinks; past 1/2 its magnitude exceeds one and the mode GROWS without
#: bound.  Both are reported, because "inverts and decays" and "amplifies
#: until the run aborts" deserve different sentences.
EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT = 0.5

#: The mixing selectors that build SEPARATE horizontal and vertical
#: exchange coefficients when ``mix_isotropic = 0``: 2 (prognostic TKE)
#: and 3 (3-D Smagorinsky).  1 (constant K) and 4 (2-D Smagorinsky) build
#: one coefficient on the horizontal grid spacing and are not exposed.
_ANISOTROPIC_LENGTH_KM_OPTS = (2, 3)


def selects_anisotropic_w_mixing(*, km_opt: int, mix_isotropic: int,
                                 dx: float, dy: float) -> bool:
    """Is this domain on the exposed path, depth aside?

    The SELECTOR half of the criterion, asked without a layer depth.
    :func:`anisotropic_w_mixing_ratio` folds the two halves together and
    answers ``None`` for both "not on the path" and "on the path but the
    depth could not be resolved", and those are opposite readings: the
    first is genuinely nothing to say, the second is the criterion's own
    subject with the number missing.  Splitting the selector out is what
    lets :func:`anisotropic_w_mixing_advice` tell them apart.
    """

    if int(km_opt) not in _ANISOTROPIC_LENGTH_KM_OPTS:
        return False
    if int(mix_isotropic) != 0:
        return False
    return min(float(dx), float(dy)) > 0.0


def anisotropic_w_mixing_ratio(*, km_opt: int, mix_isotropic: int,
                               mix_upper_bound: float, dx: float, dy: float,
                               dz_max: float) -> float | None:
    """Worst-case ``K*dt/dx^2`` the horizontal w operator can be handed.

    With ``mix_isotropic = 0`` the mixing lengths are per-axis, so the
    VERTICAL exchange coefficient is both built and capped on the layer
    depth -- ``xkm_v <= mix_upper_bound * dz^2 / dt`` -- while the
    horizontal diffusion of ``w`` differences over the horizontal grid
    spacing.  Nothing in that chain compares the coefficient against the
    horizontal spacing, so the largest ratio the operator can be asked to
    integrate is

        mix_upper_bound * (dz_max / min(dx, dy))^2

    which is ``dt``-invariant: the cap carries ``1/dt`` and the ratio
    carries ``dt``, so a shorter step buys nothing.  Where the ratio
    exceeds :data:`EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT` the operator can
    amplify a 2-grid-interval mode in ``w`` instead of damping it.

    Returns ``None`` when the configuration is not on that path at all --
    an isotropic mixing length, or a selector that builds a single
    coefficient on the horizontal spacing -- so a caller can distinguish
    "not applicable" from "applicable and small".

    ``None`` ALSO comes back for a domain that IS on the path but whose
    ``dz_max`` is non-finite, because no number exists to return and
    fabricating one would be worse.  That case is not silence: it is the
    subject of the criterion with the depth missing, and
    :func:`anisotropic_w_mixing_advice` separates it out (via
    :func:`selects_anisotropic_w_mixing`) and says so in words.  A caller
    reading this function's ``None`` alone must not read it as a pass.
    """

    if not selects_anisotropic_w_mixing(
            km_opt=km_opt, mix_isotropic=mix_isotropic, dx=dx, dy=dy):
        return None
    if not math.isfinite(float(dz_max)):
        return None
    horizontal = min(float(dx), float(dy))
    return float(mix_upper_bound) * (float(dz_max) / horizontal) ** 2


#: The phrase every no-number mixing advisory carries, so a reader
#: skimming a door and a test asserting on one are matching the same
#: string.  It is upper-case for the same reason the tier verbs are: the
#: distinction a reader must not miss is that there is no ratio here, and
#: that absence is the finding rather than the absence of one.
UNRESOLVED_ANISOTROPIC_DEPTH_MARK = "NO LAYER DEPTH COULD BE RESOLVED"


def _unresolved_depth_advice(*, where: str, km_opt: int, horizontal: float,
                             ladder: str) -> str:
    """The advisory for an exposed domain whose depths will not resolve.

    Separate wording from the over-the-limit sentence because it is a
    separate statement: that one reports a number against a threshold,
    this one reports that no number can be produced at all.  Both remedies
    are named, and they are not the same pair -- lowering
    ``mix_upper_bound`` cannot help when the quantity it multiplies is
    unknown, so what this one offers instead is making the ladder
    resolvable.
    """

    provenance = f" ({ladder})" if ladder else ""
    return (
        f"{where} runs km_opt = {int(km_opt)} with mix_isotropic = 0 at "
        f"dx = {horizontal:g} m, so its horizontal mixing of w is handed "
        f"a coefficient built on the LAYER DEPTH -- but "
        f"{UNRESOLVED_ANISOTROPIC_DEPTH_MARK} for this grid"
        f"{provenance}, so mix_upper_bound*(dz_max/dx)^2 cannot be "
        f"evaluated and the criterion has no number to report. The "
        f"absence of a ratio is NOT a pass: this domain is on the exposed "
        f"path and nothing here can say whether it is over the limit "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT} or under it. Declare "
        f"eta_levels and p_top explicitly, or bring the model top inside "
        f"the analytic base state's representable range, so the depths "
        f"resolve and the criterion can run; or set mix_isotropic = 1 on "
        f"this domain, which builds one length from (dx*dy*dz)^(1/3) and "
        f"takes it off the per-axis path entirely, making the question "
        f"moot. ADVISORY, not a refusal: the load and the run proceed.")


def anisotropic_w_mixing_advice(*, where: str, km_opt: int,
                                mix_isotropic: int, mix_upper_bound: float,
                                dx: float, dy: float, dz_max: float,
                                ladder: str = "", forced: bool = False,
                                ) -> tuple[float | None, str | None]:
    """``(ratio, one-sentence advisory)`` for the exposed-mixing check.

    ``forced`` says the configuration WROTE ``mix_isotropic = 0`` -- as
    opposed to inheriting it -- and appends the override state to the
    over-the-limit sentence.  Since the auto-switch (project ruling, 2026-08-16;
    :func:`auto_mix_isotropic_selection`), a domain that reaches this
    advisory over the limit through the experiment loader can ONLY have
    written the value: an unset ``mix_isotropic`` resolves to 1 there
    and never lands here.  The caller supplies the flag because only it
    knows which loader the config came through.

    One wording, every door.  :func:`warn_anisotropic_w_mixing` prints
    it at config load; ``woof check`` repeats it in its advisory list
    (``woof.core.preflight.anisotropic_w_mixing_advisories``), because
    the load-time line scrolls past hours before the run that pays for
    it and the preflight report is where a reader is actually looking.

    The advisory is ``None`` when the configuration is not on the
    exposed path, or is on it and under the limit.  The ratio comes back
    either way, so a caller can record the number whether or not there
    was anything to say.

    A NON-FINITE ``dz_max`` ON AN OTHERWISE-EXPOSED DOMAIN GETS ITS OWN
    ADVISORY, and the ratio stays ``None``.  This is the third outcome,
    added 2026-08-09: the domain selected the per-axis mixing length, but
    its layer depths could not be resolved, so the criterion has a
    subject and no number.  Before this branch existed the ``None`` from
    :func:`anisotropic_w_mixing_ratio` collapsed that case back into "not
    applicable" and every door went quiet -- the same silence-reads-as-a-
    pass hole the resolved-ladder work had just closed one layer up.  No
    number is invented: the sentence says the depth is unresolvable, why
    (``ladder`` carries the provenance), and what to declare instead.

    The sentence is TIERED, because the criterion has two thresholds and
    they mean different things: above
    :data:`EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT` the 2-grid-interval mode
    inverts each step and still decays; above
    :data:`EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT` it grows.  A
    reader who sees one number against one limit cannot tell 0.3 from
    4.2, and 4.2 is the tier that aborted a run.

    ``ladder`` is a provenance clause for ``dz_max``, appended verbatim
    when non-empty (``woof.experiment.ExposedMixing.ladder`` supplies
    it).  A depth the caller RESOLVED for a config that wrote no eta
    interfaces must not read the same as one the config declared, and
    the number is otherwise indistinguishable.
    """

    ratio = anisotropic_w_mixing_ratio(
        km_opt=km_opt, mix_isotropic=mix_isotropic,
        mix_upper_bound=mix_upper_bound, dx=dx, dy=dy, dz_max=dz_max)
    if ratio is None and not math.isfinite(float(dz_max)) \
            and selects_anisotropic_w_mixing(
                km_opt=km_opt, mix_isotropic=mix_isotropic, dx=dx, dy=dy):
        return None, _unresolved_depth_advice(
            where=where, km_opt=km_opt,
            horizontal=min(float(dx), float(dy)), ladder=ladder)
    if ratio is None or ratio <= EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT:
        return ratio, None
    horizontal = min(float(dx), float(dy))
    admitted = (EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
                * (horizontal / float(dz_max)) ** 2)
    if ratio > EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT:
        tier = (
            f"AMPLIFIES a 2dx mode instead of damping it -- past "
            f"{EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT} the per-step "
            f"factor 1 - 4*K*dt/dx^2 has magnitude above one, so "
            f"grid-scale vertical velocity grows out of the field the "
            f"operator exists to smooth until a health bound stops the "
            f"run")
    else:
        tier = (
            f"INVERTS a 2dx mode every step instead of damping it; its "
            f"magnitude still decays below "
            f"{EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT}, which is "
            f"where the mode starts to grow instead")
    provenance = f" ({ladder})" if ladder else ""
    advice = (
        f"{where} runs km_opt = {int(km_opt)} with mix_isotropic = 0 at "
        f"dx = {horizontal:g} m over base-state layers up to "
        f"{float(dz_max):.1f} m deep{provenance}: "
        f"mix_upper_bound*(dz_max/dx)^2 = "
        f"{ratio:.3g} exceeds the explicit horizontal diffusion limit "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}, so the horizontal "
        f"mixing of w {tier}. The remedy is mix_isotropic = 1 on this "
        f"domain: it builds ONE length from (dx*dy*dz)^(1/3) and caps "
        f"the vertical coefficient against the horizontal one, which "
        f"takes the domain off this path entirely rather than moving it "
        f"nearer the limit. Lowering mix_upper_bound below "
        f"{admitted:.3g} also satisfies the criterion, but it weakens "
        f"the subgrid mixing everywhere to fix a horizontal-operator "
        f"problem. COST OF THE REMEDY, STATED HERE SO IT IS NOT "
        f"DISCOVERED LATER: mix_isotropic is inside the RunConfig "
        f"fingerprint woof/io/restart.py writes as "
        f"configuration_sha256, so a checkpoint written under "
        f"mix_isotropic = 0 CANNOT be resumed under 1 -- take it at "
        f"t = 0, not part-way through a campaign you intend to restart. "
        f"ADVISORY, not a "
        f"refusal: {ratio:.3g} is the WORST case the cap admits and a "
        f"flow that never reaches it never sees this, which is why "
        f"trees above the limit have completed -- but above the limit "
        f"nothing guarantees the operator's own stability, and what it "
        f"costs when it does bite is an abort late in a long run rather "
        f"than a number you can inspect")
    if forced:
        advice += (
            ". OVERRIDE STATE: mix_isotropic = 0 is this configuration's "
            "EXPLICIT setting, so the automatic isotropic selection "
            "stands aside and the anisotropic form runs, at the ratio "
            "above; delete the key (or write \"auto\") to let the model "
            "select the stable length here")
    return ratio, advice


def warn_anisotropic_w_mixing(*, where: str, km_opt: int, mix_isotropic: int,
                              mix_upper_bound: float, dx: float, dy: float,
                              dz_max: float, ladder: str = "",
                              forced: bool = False) -> float | None:
    """Advise when the anisotropic mixing length exceeds the explicit limit.

    A warning and not a refusal, RE-RULED 2026-08-09 after the shipped
    configs were cleaned: the ratio is the WORST case the cap admits and
    the criterion overestimates what a flow reaches.  The receipt-backed
    leg of that is the 250 m nested tree, which completed five
    multi-hour runs at 0.702 -- 2.8x the limit -- with no sign of the
    failure mode.  Beside it sits one weaker observation, recorded in
    prose rather than in an instrument: the attempt #2b post-mortem
    notes 1.55 at the failing cell where the criterion read 4.23
    (``configs/les_tornado_100m_mayfield_20211210_attempt3.toml:232``,
    the only place that number is written down).  Refusing at load would
    also make the frozen records under ``configs/`` unloadable, which is
    the same as deleting the ability to reproduce a committed crash.

    What it is not is a taste bound -- above the limit the operator's
    own stability is no longer guaranteed by anything, and the failure
    mode is grid-scale vertical velocity that grows out of a field which
    should be being smoothed.  So it stays advisory, it says the number
    and the tier, and ``tests/test_shipped_configs_mixing_stability.py``
    keeps the shipped set out of the exposed state entirely.

    THE DEFAULT IS THE AUTO-SWITCH (project ruling, 2026-08-16; supersedes the
    recipe-only half of the 2.5.0 "where the default moved" ruling).
    "Fixed means default" makes an opt-in remedy for a correctness
    criterion a workaround, and an advisory that a bare config scrolls
    past was exactly that.  A domain that leaves ``mix_isotropic`` unset
    (or writes ``"auto"``) and violates this criterion now RUNS
    ``mix_isotropic = 1`` -- resolved by the experiment loader
    (``woof.experiment.resolve_auto_mix_isotropic``), announced by
    :func:`auto_mix_isotropic_selection` at load and by ``woof check``,
    with the fingerprint consequence named at both restart doors
    (:data:`MIX_ISOTROPIC_RESTART_BREAK_NOTICE`).  The wizard's
    :func:`woof.domain_wizard.gray_zone_advisory` recipe keeps naming
    ``mix_isotropic = 1`` explicitly, as before.

    So this function fires only for a configuration that WROTE
    ``mix_isotropic = 0`` (or one on a route the auto-switch does not
    resolve, e.g. a wrapped legacy RunConfig), and that asymmetry is the
    ruling: an explicit setting is kept, in the danger zone too, and
    what it gets is this warning -- the instability by name, the
    measured ratio, and (``forced``) the override state.  Refusing would
    make the frozen crash records unloadable; flipping a WRITTEN key
    would mutate a physics selector the user chose and orphan its
    checkpoints (``mix_isotropic`` is inside ``configuration_sha256``).

    Returns the computed ratio so a caller can record the number whether
    or not it warned.  ``None`` covers two cases and a caller must not
    conflate them: not applicable, and applicable with an unresolvable
    layer depth -- the second one warns, and the warning is the only
    place that distinction is visible from here.
    """

    ratio, advice = anisotropic_w_mixing_advice(
        where=where, km_opt=km_opt, mix_isotropic=mix_isotropic,
        mix_upper_bound=mix_upper_bound, dx=dx, dy=dy, dz_max=dz_max,
        ladder=ladder, forced=forced)
    if advice is None:
        return ratio
    from woof.explain import warn
    warn(
        advice,
        why="With mix_isotropic = 0 the vertical exchange coefficient is "
            "built and capped on the LAYER DEPTH (xkm_v <= "
            "mix_upper_bound*dz^2/dt) and is then the coefficient the "
            "horizontal diffusion of w differences over the HORIZONTAL "
            "spacing. Nothing compares the two, so the reachable "
            "K*dt/dx^2 is mix_upper_bound*(dz_max/dx)^2 -- independent "
            "of dt, because the cap carries 1/dt and the ratio carries "
            "dt. An explicit Laplacian multiplies a 2-grid-interval mode "
            "by 1 - 4*K*dt/dx^2 per step, so beyond 1/4 the sign flips "
            "and beyond 1/2 the mode grows. mix_isotropic = 1 builds one "
            "length from (dx*dy*dz)^(1/3) and caps the vertical "
            "coefficient against the horizontal one, which is why it is "
            "the usual choice where layers are much deeper than the grid "
            "is wide. This is an advisory: the ratio is the worst case "
            "the cap admits, not a value this flow is required to reach.")
    return ratio


#: The TOML sentinel for "let the model choose the mixing length": the
#: same meaning as leaving ``mix_isotropic`` unset, writable so a config
#: can SAY it is deferring rather than merely omit the key.  Only the
#: experiment loader consumes it; ``RunConfig.mix_isotropic`` itself
#: stays the resolved WRF integer (0/1) everywhere downstream.
MIX_ISOTROPIC_AUTO = "auto"

#: The TOML sentinel for "let the model choose the off-centering": the same
#: meaning as leaving ``epssm`` unset, and what ``woof import-namelist``
#: writes where WRF's Registry default fills a domain the namelist did not
#: assign.  The domain starts on ``RunConfig.epssm`` (WRF's 0.1) and the
#: off-centering floor (:func:`woof.acoustic_adaptation.offcentering_floor`)
#: raises it over ground that value was not measured stable on.  Only the
#: experiment loader consumes it; ``RunConfig.epssm`` stays a number.
EPSSM_AUTO = "auto"


def auto_mix_isotropic_selection(*, where: str, ratio: float,
                                 ladder: str = "") -> str:
    """The one loud line for an auto-selected isotropic mixing length.

    One wording, every door -- exactly the discipline of
    :func:`anisotropic_w_mixing_advice`: the experiment loader prints it
    (through :func:`woof.explain.warn`) at the shared config load every
    front door runs at model start, and ``woof check`` repeats it in
    its advisory list so the report says what the run WILL do, not that
    something is wrong.  It names what happened and why: the criterion
    value, the limit it exceeds, and that isotropic mixing was selected
    -- plus the escape hatch and the restart consequence, so neither is
    discovered later.
    """

    provenance = f" ({ladder})" if ladder else ""
    return (
        f"{where} leaves mix_isotropic UNSET on a grid that violates the "
        f"anisotropic-mixing criterion -- mix_upper_bound*(dz_max/dx)^2 "
        f"= {ratio:.3g} exceeds the explicit horizontal diffusion limit "
        f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}{provenance} -- so this "
        f"run SELECTS ISOTROPIC MIXING: mix_isotropic = 1, the single "
        f"(dx*dy*dz)^(1/3) length, runs on this domain. Write "
        f"mix_isotropic = 0 explicitly to keep the anisotropic form "
        f"(the instability advisory then applies); a checkpoint written "
        f"under the old anisotropic default will not bit-continue under "
        f"this selection.")


#: The restart doors' one-line accuracy about the changed default: said
#: when a checkpoint integrated under ``mix_isotropic = 0`` meets a run
#: that selects 1, because a bare hash/field mismatch does not tell a
#: reader that the DEFAULT moved under them.  A notice beside the
#: existing refusal, not a refusal of its own -- the mismatch machinery
#: already names the concrete breakage (a different trajectory).
MIX_ISOTROPIC_RESTART_BREAK_NOTICE = (
    "note: the checkpoint was integrated under anisotropic mixing "
    "(mix_isotropic = 0) and this run selects ISOTROPIC mixing "
    "(mix_isotropic = 1). A config that leaves mix_isotropic unset now "
    "auto-selects the isotropic length where mix_upper_bound*"
    "(dz_max/dx)^2 exceeds "
    f"{EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT}, and a trajectory cannot "
    "bit-continue across that change. To resume this checkpoint, write "
    "mix_isotropic = 0 explicitly on the domain (the instability "
    "advisory then applies); otherwise restart from t = 0.")


def _adaptive_interval(whole: int, den: int, name: str):
    """One of WRF's ``<name>`` / ``<name>_den`` rational pairs, or None.

    ``-1`` is upstream's "unset" for all three pairs and means different
    things per field, so it resolves to ``None`` here and each caller says
    what its own unset means.  ``start_em.F:939-953`` substitutes
    ``NINT(4*MIN(dx,dy)/1000)`` for the starting step, ``8*`` for the max
    and ``3*`` for the min.  (An earlier version of this docstring said
    ``6*dx`` for the starting step; that is the acoustic ``dt <= 6*dx``
    constraint, a different rule, and it was wrong here.)

    ``den = 0`` is upstream's "the value is whole seconds", NOT a zero
    denominator: ``adapt_timestep_em.F:132-136`` branches on
    ``starting_time_step_den .EQ. 0`` and passes ``Sd=1``, and
    ``:186-214`` does the same for the two limits.  Otherwise the pair is
    a genuine rational ``Sn/Sd`` -- ESMF holds time intervals as exact
    rationals, so a step like 100/3 s stays exact over thousands of steps
    where a float would drift.  That is the same reason WOOF's clock is
    an integer tick lattice, which is why the pair maps onto it cleanly.
    """
    from fractions import Fraction

    if whole == -1 and den == 0:
        return None
    if den < 0:
        raise ValueError(
            f"{name}_den must be >= 0 (0 means '{name} is whole seconds'), "
            f"got {den}.")
    if den == 0:
        return Fraction(whole)
    return Fraction(whole, den)


def _validate_adaptive_time_step(cfg: RunConfig) -> None:
    """Refuse incoherent adaptive-timestep combinations at admission.

    Every refusal here is one that would otherwise present as a physics
    failure hours into a run rather than as a configuration error in one
    line -- which is the same argument
    ``_check_run_length_on_step_grid`` makes for the run length.
    """
    if cfg.target_cfl <= 0.0:
        raise ConfigurationRefused(
            f"target_cfl must be positive, got {cfg.target_cfl}: it is the "
            "vertical Courant number the controller steers toward "
            "(WRF Registry default 1.2), and calc_dt divides by it.")
    if cfg.target_hcfl <= 0.0:
        raise ConfigurationRefused(
            f"target_hcfl must be positive, got {cfg.target_hcfl}: it is "
            "the horizontal Courant number the controller steers toward "
            "(WRF Registry default 0.84).")
    if cfg.max_step_increase_pct < 0:
        raise ConfigurationRefused(
            "max_step_increase_pct must be >= 0, got "
            f"{cfg.max_step_increase_pct}: it bounds growth against the "
            "PREVIOUS step as 1 + pct/100, so a negative value asks the "
            "step to shrink every time it is allowed to grow.")
    if cfg.adaptation_domain != 1:
        # THE REFUSAL NAMES THE BREAKAGE.  Upstream's `adaptation_domain`
        # selects which domain's CFL drives the tree's step; woof drives
        # every domain from its own measured CFL and divides the parent's
        # step down.  The child-driven arm exists in
        # adaptive_timestep.nest_dt_from_parent (adapt_using_child=True)
        # and has no caller, so any value but 1 was accepted, validated,
        # written into three identity tables and the checkpoint header,
        # and then did nothing at all -- the run adapted parent-first and
        # said no word about it.  Refused rather than warned, because a
        # user who sets this is asking for a different trajectory.
        raise ConfigurationRefused(
            f"adaptation_domain must be 1, got {cfg.adaptation_domain}: "
            "woof adapts every domain from its own measured CFL and "
            "divides the parent step down, so the child-driven selection "
            "upstream's values above 1 request is not wired to the "
            "driver.  Accepting the value would run a parent-driven "
            "tree while the configuration said otherwise.")

    min_dt = _adaptive_interval(cfg.min_time_step, cfg.min_time_step_den,
                                "min_time_step")
    max_dt = _adaptive_interval(cfg.max_time_step, cfg.max_time_step_den,
                                "max_time_step")
    start_dt = _adaptive_interval(cfg.starting_time_step,
                                  cfg.starting_time_step_den,
                                  "starting_time_step")
    for value, name in ((min_dt, "min_time_step"), (max_dt, "max_time_step"),
                        (start_dt, "starting_time_step")):
        if value is not None and value <= 0:
            raise ConfigurationRefused(
                f"{name} resolves to {float(value):g} s, which is not a "
                "positive interval; -1 with a zero denominator is the way "
                "to leave it unset.")
    if min_dt is not None and max_dt is not None and min_dt > max_dt:
        # Upstream applies max first and min second, so the minimum
        # silently wins and the run integrates above max_time_step for its
        # whole length.  Refused by name here instead.
        raise ConfigurationRefused(
            f"min_time_step ({float(min_dt):g} s) exceeds max_time_step "
            f"({float(max_dt):g} s).  WRF clamps max first and min second "
            "(adapt_timestep_em.F:186-214), so this does not error there "
            "-- the minimum wins and the model runs above the maximum it "
            "was told to respect.")
    if start_dt is not None:
        if max_dt is not None and start_dt > max_dt:
            raise ConfigurationRefused(
                f"starting_time_step ({float(start_dt):g} s) exceeds "
                f"max_time_step ({float(max_dt):g} s).")
        if min_dt is not None and start_dt < min_dt:
            raise ConfigurationRefused(
                f"starting_time_step ({float(start_dt):g} s) is below "
                f"min_time_step ({float(min_dt):g} s).")
    floor = getattr(cfg, "min_time_step_sound", 0)
    if (not isinstance(floor, int) or isinstance(floor, bool)
            or floor < 0):
        raise ConfigurationRefused(
            "min_time_step_sound must be a whole number of acoustic "
            f"substeps, 0 or more, got {floor!r}: it is the fewest "
            "substeps the adaptive clock may take, and 0 leaves WRF's "
            "count.")
    if floor % 2:
        raise ConfigurationRefused(
            f"min_time_step_sound must be even, got {floor}: the adaptive "
            "clock runs at least this many substeps, and RK3 stage 2 runs "
            "time_step_sound//2 acoustic substeps of dt/time_step_sound, "
            "which only integrates to dt/2 for even counts.")

    if not isinstance(cfg.nest_accumulation_archive, bool):
        raise ConfigurationRefused('nest_accumulation_archive must be a boolean')
    if not isinstance(cfg.adaptive_nest_lattice, bool):
        raise ConfigurationRefused("adaptive_nest_lattice must be true or false")
    if cfg.adaptive_nest_lattice and not cfg.use_adaptive_time_step:
        raise ConfigurationRefused("adaptive_nest_lattice needs use_adaptive_time_step = true")
    if cfg.terrain_clock not in TERRAIN_CLOCK_MODES:
        raise ConfigurationRefused(
            f"terrain_clock={cfg.terrain_clock!r} is not one of "
            f"{list(TERRAIN_CLOCK_MODES)}: \"measured\" lets the terrain "
            "clock and the steep-ground substep rule rewrite the clock at "
            "launch where their measured maps saw a step stop; \"pinned\" "
            "runs dt and time_step_sound exactly as configured and records "
            "what those rules would have done as advice in the receipt; "
            "\"local_face\" (the default) reads every terrain face with the "
            "crest and crest-level wind around it and is never worse than "
            "\"measured\" (woof/terrain_clock_local.py)")
    if not cfg.use_adaptive_time_step:
        return

    # From here on the feature is ON, so the refusals are about whether it
    # can actually be run rather than about whether the numbers cohere.
    if not cfg.step_to_output_time:
        # woof.experiment._check_cadence requires every cadence to be a
        # whole number of steps, and with a varying dt only the
        # step-to-time mechanism can keep that true.  WRF's own Registry
        # default is .true. for the same reason.
        raise ConfigurationRefused(
            "use_adaptive_time_step needs step_to_output_time = true.  "
            "woof requires every cadence -- history_interval_s, the "
            "relocation cadence and track interval, the radiation and "
            "cumulus calls -- to be a whole number of steps, and a "
            "varying dt can only keep that promise by landing exactly on "
            "the output times.  WRF defaults it true "
            "(Registry.EM_COMMON:2278); turning it off here would make "
            "the first cadence check fail mid-run instead of now.")


#: ``mosaic_urban_canopy``: where Noah mosaic runs the urban canopy model
#: (``sf_surface_mosaic = 1`` with ``sf_urban_physics = 1``), rule -> what it
#: does.  WRF's rule is the default; the town rule is an option until it
#: beats WRF's against station observations over several nights.
MOSAIC_URBAN_CANOPY_RULES = {
    "dominant": (
        "WRF v4.7.1: the canopy runs only in cells whose dominant land-use "
        "category is urban; an urban tile in a mostly rural cell runs as "
        "vegetation, because urban_var_init sets its urban fraction to 0 "
        "(module_sf_urban.F:2820-2821)"),
    "every_tile": (
        "every land cell with an urban tile runs the canopy in that tile at "
        "its urban type's own fraction (URBPARM FRC_URB, the largest urban "
        "tile's type); cells whose dominant category is urban are WRF's"),
}
MOSAIC_URBAN_CANOPY_DEFAULT = "dominant"


def validate_noah_mosaic_config(cfg) -> None:
    """Refuse selectors that would skip Noah or integrate the wrong surface."""
    option = getattr(cfg, "sf_surface_mosaic", 0)
    if isinstance(option, bool) or not isinstance(option, int) or option not in (0, 1):
        raise ConfigurationRefused(
            "sf_surface_mosaic must be an integer 0 or 1: neither Noah arm "
            "would run, so the land surface would not be integrated "
            "(module_surface_driver.F:2662,2778).")
    canopy = getattr(cfg, "mosaic_urban_canopy", MOSAIC_URBAN_CANOPY_DEFAULT)
    if not isinstance(canopy, str) or canopy not in MOSAIC_URBAN_CANOPY_RULES:
        raise ConfigurationRefused(
            f"mosaic_urban_canopy must be one of "
            f"{sorted(MOSAIC_URBAN_CANOPY_RULES)}, got {canopy!r}: no other "
            "rule says where the urban canopy runs under Noah mosaic.")
    if canopy != MOSAIC_URBAN_CANOPY_DEFAULT:
        if option != 1:
            raise ConfigurationRefused(
                f"mosaic_urban_canopy = {canopy!r} needs sf_surface_mosaic = "
                "1: without land-use tiles a cell has no urban tile for the "
                "canopy to run in, so the setting would be ignored.")
        if getattr(cfg, "sf_urban_physics", 0) != 1:
            raise ConfigurationRefused(
                f"mosaic_urban_canopy = {canopy!r} needs sf_urban_physics = "
                "1: it says where the single-layer urban canopy runs inside "
                "the tiles, and with no canopy model there is nothing for it "
                "to place (options 2 and 3 cannot run with mosaic at all).")
    if option == 0:
        return
    if cfg.sf_surface_physics != 2:
        raise ConfigurationRefused(
            "sf_surface_mosaic=1 requires sf_surface_physics=2: every other "
            "LSM silently ignores the switch (module_surface_driver.F:2662).")
    count = cfg.mosaic_cat
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ConfigurationRefused(
            "mosaic_cat must be an integer >= 1: otherwise Noah has no tile "
            "to integrate (Registry.EM_COMMON:2537).")
    if getattr(cfg, "sf_urban_physics", 0) in (2, 3):
        raise ConfigurationRefused(
            "mosaic option cannot work with urban options 2 and 3 "
            "(WRF module_check_a_mundo.F:505-518).")


#: ``chem_adv_opt`` values and the module each needs (DESIGN 2.2).  1 is the
#: positive-definite final stage the moist path already ports; 2 is WRF's
#: monotonic limiter (``advect_scalar_mono``), admitted only when its module
#: is in the build.
CHEM_ADV_OPT_MODULES = {1: None, 2: "woof.core.chem_advect_mono"}
FIRE_EMISSION_MODES = ("trailing_24h_dcycle", "daily_mean_dcycle",
                       "observed_hourly")
PLUME_FIRE_PROPERTIES = ("frp", "landuse")
AEROSOL_MP_COUPLINGS = ("auto", "none", "diagnose", "emission")


def _module_in_build(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def validate_chem_config(cfg) -> None:
    """The chem law (DESIGN 1.5 and 9), each refusal naming its breakage.

    Inert at the default ``chem_sets = ()``: the other chem keys are then read
    by nothing, exactly as WRF reads none of ``&chem`` with ``chem_opt = 0``,
    so no value of them can make a run wrong and none is refused.
    """
    from woof.chem_table import ChemTableError, chem_names

    if getattr(cfg, "simple_dir_fdb", 0) and not getattr(cfg, "chem_sets", ""):
        raise ConfigurationRefused(
            "simple_dir_fdb=1 with chem_sets empty: there is no transported "
            "smoke for the shortwave to read, so the switch would do nothing.")
    for name in ("chem_sets", "chem_sources"):
        value = getattr(cfg, name, "")
        if not isinstance(value, str):
            raise ConfigurationRefused(
                f"{name} must be a comma-separated string of names, got "
                f"{value!r}: RunConfig is echoed into JSON identity documents, "
                "where only a string survives the round trip unchanged.")
        try:
            chem_names(value, name)
        except ChemTableError as error:
            raise ConfigurationRefused(str(error)) from error
    sets = chem_names(cfg.chem_sets, "chem_sets")
    sources = chem_names(cfg.chem_sources, "chem_sources")
    if not sets:
        return
    from woof.chem_table import load_sets, process_in_build

    try:
        table = load_sets(sets, sources)
    except ChemTableError as error:
        raise ConfigurationRefused(str(error)) from error
    for name in ("chem_adv_opt", "kemit", "biomass_burn_opt",
                 "plumerisefire_frq", "dust_opt", "seas_opt", "dmsemis_opt",
                 "wetscav_onoff", "chem_conv_tr", "vertmix_onoff",
                 "aer_ra_feedback", "aer_op_opt", "simple_dir_fdb",
                 "aer_drydep_opt"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationRefused(f"{name} must be an integer (WRF declares it an "
                             f"integer namelist key), got {value!r}.")
    option = cfg.chem_adv_opt
    if option not in CHEM_ADV_OPT_MODULES:
        raise ConfigurationRefused(
            f"chem_adv_opt={option}: only 1 (positive definite) and 2 "
            "(monotonic) are defined for this build; WRF's 0 (unlimited), 3 "
            "(WENO) and 4 (WENO positive definite) are not transcribed, so "
            "the chem array would be advected by an operator nobody ported.")
    module = CHEM_ADV_OPT_MODULES[option]
    if module is not None and not _module_in_build(module):
        raise ConfigurationRefused(
            f"chem_adv_opt={option}: option {option}'s kernel is not in this "
            f"build ({module} is absent), so the monotonic limiter would be "
            "silently replaced by the positive-definite one.")
    if cfg.chem_conv_tr != 0 and int(getattr(cfg, "cu_physics", 0)) != 0:
        raise ConfigurationRefused(
            f"chem_conv_tr={cfg.chem_conv_tr} with cu_physics="
            f"{cfg.cu_physics}: WRF-Chem's subgrid convective transport of "
            "chem (grelldrvct) is not transcribed, so with a cumulus scheme "
            "on, chem would stay under the cloud base the scheme vents.  Set "
            "chem_conv_tr = 0 (WRF's own option) to run without it.")
    if cfg.wetscav_onoff > 0:
        raise ConfigurationRefused(
            f"wetscav_onoff={cfg.wetscav_onoff}: aqueous scavenging needs a "
            "gas-phase and cloud-chemistry mechanism, which this program does "
            "not carry by design; wetscav_onoff < 0 selects the large-scale "
            "washout (wetdep_ls).")
    if cfg.dust_opt not in (0, 1, 3):
        raise ConfigurationRefused(
            f"dust_opt={cfg.dust_opt}: only 1 (GOCART) and 3 (AFWA) are "
            "transcribed; 4 (University of Cologne) and the others are not, "
            "so no dust would be emitted under the name of a scheme.")
    if cfg.dmsemis_opt != 0:
        raise ConfigurationRefused(
            f"dmsemis_opt={cfg.dmsemis_opt}: no DMS seawater concentration "
            "source row exists, so the DMS flux would read an absent field.")
    if cfg.mynn_chem_vertmx and not _module_in_build(
            "woof.core.mynn_chem_mix"):
        raise ConfigurationRefused(
            "mynn_chem_vertmx = true: the MYNN chem mixing kernel is not in "
            "this build (woof.core.mynn_chem_mix is absent), so chem would be "
            "mixed by vertmx under a flag that says it is not.")
    from woof.core.chem_fire import RETIRED_MODES
    if cfg.fire_emission_mode in RETIRED_MODES:
        raise ConfigurationRefused(RETIRED_MODES[cfg.fire_emission_mode])
    for name, value, allowed in (
            ("fire_emission_mode", cfg.fire_emission_mode,
             FIRE_EMISSION_MODES),
            ("plume_fire_properties", cfg.plume_fire_properties,
             PLUME_FIRE_PROPERTIES),
            ("aerosol_mp_coupling", cfg.aerosol_mp_coupling,
             AEROSOL_MP_COUPLINGS)):
        if value not in allowed:
            raise ConfigurationRefused(f"{name}={value!r}: must be one of {allowed}.")
    from woof.core.chem_mp_coupling import effective_mode
    coupling_mode = effective_mode(cfg)
    if coupling_mode != "none" and int(cfg.mp_physics) not in THOMPSON_AEROSOL_FAMILY:
        raise ConfigurationRefused(
            f"aerosol_mp_coupling={cfg.aerosol_mp_coupling!r} needs "
            f"mp_physics=28 (aerosol-aware Thompson), got {cfg.mp_physics}: "
            "no other scheme carries the nwfa/nifa fields it writes.")
    if coupling_mode == "emission":
        raise ConfigurationRefused(
            "aerosol_mp_coupling='emission' (GSL rrfs_smoke_wrapper.F90:"
            "632-655: emitted smoke, sea-salt and dust number added to "
            "nwfa/nifa at kemit) is not in this build: no process adds it, "
            "so Thompson would see none of the emitted aerosol while the "
            "configuration says it does; use 'diagnose' or 'none'.")
    if (coupling_mode == "diagnose"
            and (int(getattr(cfg, "aer_init_opt", 0)) != 0
                 or int(getattr(cfg, "wif_input_opt", 0)) != 0)):
        raise ConfigurationRefused(
            "aerosol_mp_coupling='diagnose' with the WIF climatology "
            "(aer_init_opt/wif_input_opt): two aerosol sources for one number "
            "field, and the second would overwrite the first every step.")
    missing = [key for key in table.processes if not process_in_build(key)]
    if missing:
        from woof.chem_table import CHEM_PROCESS_MODULES

        raise ConfigurationRefused(
            "chem_sets " + repr(list(sets)) + " name process(es) not in this "
            "build: " + ", ".join(
                f"{key} ({CHEM_PROCESS_MODULES[key]} is absent)"
                for key in missing)
            + "; the rows naming them would silently skip that physics.")
    from woof.chem_table import CHEM_PROCESS_MODULES
    import importlib

    for key in table.processes:
        module = importlib.import_module(CHEM_PROCESS_MODULES[key])
        hook = getattr(module, "refusal", None)
        reason = hook(cfg) if hook is not None else None
        if reason:
            raise ConfigurationRefused(f"chem process {key}: {reason}.")
    if (coupling_mode == "diagnose"
            and not table.rows_for("coupling.thompson")):
        raise ConfigurationRefused(
            f"aerosol_mp_coupling='diagnose' with chem_sets {list(sets)}: no "
            "active row feeds get_niwfa (coupling.thompson), so nwfa/nifa "
            "would stay the scheme's own under a flag that says the chem "
            "species set them; select a set with dust, sea-salt, sulfate or "
            "organic-carbon rows, or set aerosol_mp_coupling = 'none'.")
    claims: dict[str, set[str]] = {}
    for row in table.rows:
        if row.exclusive_group is None:
            continue
        group, _, claimant = row.exclusive_group.partition(":")
        claims.setdefault(group, set()).add(claimant or row.name)
    for group, claimants in claims.items():
        if len(claimants) > 1:
            raise ConfigurationRefused(
                f"chem_sets {list(sets)} activate {sorted(claimants)} as "
                f"claimants of one source field ({group!r}): the same "
                "emitted mass would be added twice.")
    catalog_sources = _catalog_sources()
    if (table.rows_for("chem.sulfur")
            and not any(catalog_sources[name].kind == "oxidant"
                        for name in sources)):
        raise ConfigurationRefused(
            "the sulfur rows (chem.sulfur) need an oxidant source (OH, H2O2, "
            "NO3) in chem_sources, and none is enabled: SO2 would never "
            "oxidize and sulfate would be emission-only.")
    if cfg.aer_ra_feedback != 0 and not table.rows_for("optics.gocart"):
        raise ConfigurationRefused(
            f"aer_ra_feedback={cfg.aer_ra_feedback} with no active row naming "
            "optics.gocart: radiation would be handed aerosol optics nothing "
            "computes.")
    if cfg.aer_ra_feedback != 0:
        lw, sw = radiation_scheme_ids(cfg)
        if any(int(value) not in (0, 4) for value in (lw, sw)) or not (lw or sw):
            raise ConfigurationRefused(
                "aer_ra_feedback=1 needs RRTMG or RRTMGP radiation "
                "(ra_lw_physics/ra_sw_physics=4): another solver would discard "
                "the chemistry aerosol optical properties.")
        if cfg.aer_opt != 0 or cfg.rrtmg_smoke_manifest:
            raise ConfigurationRefused(
                "aer_ra_feedback=1 owns the aerosol optical properties; "
                "aer_opt climatology/user/Thompson optics or a prescribed smoke "
                "manifest would overwrite or double count that same aerosol. "
                "Use aer_opt=0 with prognostic chemistry feedback.")
    if cfg.simple_dir_fdb not in (0, 1):
        raise ConfigurationRefused(
            f"simple_dir_fdb={cfg.simple_dir_fdb}: the HRRR fork defines 0 (off) "
            "and 1 (smoke AOD into the aer_opt = 3 shortwave) only.")
    if cfg.simple_dir_fdb:
        smoke_rows = [row for row in table.transported if row.feeds_shortwave_aod]
        if len(smoke_rows) != 1:
            raise ConfigurationRefused(
                "simple_dir_fdb=1 needs exactly one transported chem row whose "
                "radiation.shortwave declares the smoke layer AOD (chem_sets = \"smoke\"); "
                f"this selection has {[r.name for r in smoke_rows]}, so the "
                "shortwave would read no smoke or pick one row arbitrarily.")
        lw, sw = radiation_scheme_ids(cfg)
        if (int(sw) != 4 or cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY
                or cfg.aer_opt != 3):
            raise ConfigurationRefused(
                "simple_dir_fdb=1 adds smoke to the legacy RRTMG aer_opt = 3 "
                "shortwave AOD (the HRRR fork's only smoke coupling); with "
                f"ra_sw_physics={sw}, ra_rrtmg_variant={cfg.ra_rrtmg_variant!r}, "
                f"aer_opt={cfg.aer_opt} no solver would read it.")
        if cfg.aer_ra_feedback or cfg.rrtmg_smoke_manifest:
            raise ConfigurationRefused(
                "simple_dir_fdb=1 with aer_ra_feedback or a prescribed smoke "
                "manifest: the shortwave would count the same plume twice "
                "(the transported smoke's AOD and the chem optics or the "
                "prescribed profile); keep one.")
    if cfg.aer_drydep_opt not in (0, 111):
        raise ConfigurationRefused(
            f"aer_drydep_opt={cfg.aer_drydep_opt}: only the HRRR fork's 111 "
            "(smoke deposition velocity in the vertical mixing) is a run "
            "field; other aerosol dry deposition is a row's drydep scheme.")
    if cfg.aer_drydep_opt == 111 and not (
            cfg.vertmix_onoff > 0 and table.rows_for("mixing.vertmx")):
        raise ConfigurationRefused(
            "aer_drydep_opt=111 deposits through the chem vertical mixing "
            "(dry_dep_driver.F:256), and this run mixes no row "
            "(vertmix_onoff or no mixing.vertmx row): nothing would deposit.")
    if not isinstance(cfg.enh_vermix, bool):
        raise ConfigurationRefused(f"enh_vermix must be a boolean, got {cfg.enh_vermix!r}.")
    if cfg.enh_vermix and not (table.rows_for("emission.fire")
                               and table.rows_for("mixing.vertmx")):
        raise ConfigurationRefused(
            "enh_vermix raises the mixing over burning columns from the "
            "fire FRP; this run has no emission.fire or mixing.vertmx row, "
            "so the switch would read no FRP or mix nothing.")
    flam = cfg.flam_part
    if isinstance(flam, bool) or not isinstance(flam, (int, float)) or not (
            flam == -1.0 or 0.0 <= flam <= 1.0):
        raise ConfigurationRefused(
            f"flam_part={flam!r}: a flaming fraction in [0, 1] (the fork "
            "refuses > 1, smoke_init.F:70), or -1 for the plume's own.")
    if flam != -1.0 and not table.rows_for("plumerise.freitas"):
        raise ConfigurationRefused(
            "flam_part splits the plume-rise injection; this run has no "
            "plumerise.freitas row, so it would split nothing.")
    if cfg.aer_ra_feedback and cfg.ra_rrtmg_variant == RRTMG_VARIANT_RTE_RRTMGP:
        missing = [row.name for row in table.rows_for("optics.gocart")
                   if not row.optics or row.optics.get("merra_type") is None]
        if missing:
            raise ConfigurationRefused(
                "RRTMGP aerosol feedback needs native MERRA optical table "
                f"metadata for rows {missing}: these masses would otherwise "
                "be omitted from the radiation spectrum.")
    # aer_op_opt = 0 is WRF-Chem's "no optical averaging": with the feedback
    # off nothing consumes an optical mixing rule, so only 2-4 (Maxwell-
    # Garnett, exact Mie) or a feedback run with no rule can be handed the
    # wrong one.  The bulk smoke row gained optics.gocart metadata
    # (development branch, 177a585dd32), and HRRR's own namelist (smoke,
    # aer_op_opt = 0, feedback off) must still import.
    if table.rows_for("optics.gocart") and (
            cfg.aer_op_opt not in (0, 1)
            or (cfg.aer_op_opt == 0 and cfg.aer_ra_feedback)):
        raise ConfigurationRefused(
            f"aer_op_opt={cfg.aer_op_opt}: only WRF-Chem's volume "
            "approximation (aer_op_opt = 1, module_optical_averaging.F) is "
            "ported; the Maxwell-Garnett and exact-Mie mixing rules are not, "
            "so the optics would silently use the volume rule.")
    # Emission frames.  The one frame provider in this build
    # (woof.chem_emission_frames, through woof.chem_source_netcdf.ingest)
    # serves extensive NetCDF sources on a cell-sum remap, the RAVE fire
    # rows.  An enabled emission source it cannot serve (EDGAR's annual
    # flux densities: no fetch route carries its archives and no
    # intensive-field remap feeds the frames) passed the door and stopped
    # the run at its first chem step, after a whole preparation, with "no
    # source frames to read them from".
    unserved = sorted({ref["source"] for row in table.rows
                       for ref in row.emissions
                       if ref["source"] in sources
                       and catalog_sources[ref["source"]].kind == "emission"
                       and catalog_sources[ref["source"]].remap
                       not in ("cell_sum", "cell_sum_split")})
    if unserved:
        raise ConfigurationRefused(
            f"chem_sources enables emission source(s) {unserved}, but no "
            "emission frame provider in this build can supply them (frames "
            "come from extensive NetCDF sources on a cell-sum remap, such as "
            "the RAVE fire rows), so the run would stop at its first chem "
            "step; leave them out of chem_sources and the rows naming them "
            "take no emission from them.")
    # A data-store source (a row with an acquisition block, CAMS through the
    # ADS) fills its rows on every forcing frame (woof.chem_source_init),
    # and the root's sealed forcing carries them as WRF-Chem's
    # have_bcs_chem rows.  Any other boundary source has no frame fill in
    # this build, so its rows would take their default inflow instead.
    boundary_sources = sorted({ref["source"] for row in table.rows
                               for ref in row.boundary
                               if ref["source"] in sources
                               and catalog_sources[ref["source"]].acquisition
                               is None})
    if boundary_sources:
        raise ConfigurationRefused(
            f"chem_sources enables boundary source(s) {boundary_sources} for "
            "active rows, but no forcing-frame fill for them is in this "
            "build, so the rows would take their default inflow instead.")


def _catalog_sources():
    from woof.chem_table import catalog

    return catalog().sources


CUMULUS_PARAMETER_IDENTITY_RULES = {'kf_edrates': (0, (11, 99)), 'shcu_physics': (0, ()), 'accum_mode': (1.e9, ()), 'aitken_mode': (3.e8, ()), 'coarse_mode': (2.e5, ()), 'nsas_dx_factor': (0, (96,)), 'sas_pgcon': (0.55, (94, 96)),
    'numbins': (1, (10,)), 'thbinsize': (1.0, (10,)), 'rbinsize': (1.0, (10,)),
    'mindeepfreq': (1.0, (10,)), 'minshallowfreq': (1.0, (10,)),
    'shallowcu_forced_ra': (False, (10,)), 'shalconv': (1, (94,)), 'bmj_rad_feedback': (False, ())}


def validate_seaice_parameters(cfg):
    """Configuration-only validation, also available in preprocessing installs."""
    if type(cfg.seaice_prognostic) is not bool:
        raise ValueError("seaice_prognostic must be boolean")
    for key in ("seaice_initial_thickness", "seaice_new_ice_thickness"):
        value = getattr(cfg, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(cfg.seaice_ocean_heat_flux):
        raise ValueError("seaice_ocean_heat_flux must be finite, positive into ice")
def validate_irrigation_config(cfg) -> None:
    """WRF check_a_mundo:530-562 and executable denominator checks."""
    names = ('sf_surf_irr_scheme', 'irr_start_hour', 'irr_num_hours',
             'irr_start_julianday', 'irr_end_julianday', 'irr_freq', 'irr_ph')
    for name in names:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{name} must be an integer')
    option = cfg.sf_surf_irr_scheme
    if option not in (0, 1, 2, 3):
        raise ValueError('sf_surf_irr_scheme must be 0, 1=channel, 2=drip, 3=sprinkler')
    if not option:
        return
    if option == 1 and cfg.sf_surface_physics != 2:
        raise ValueError('channel irrigation (sf_surf_irr_scheme=1) requires Noah (sf_surface_physics=2); other LSMs do not consume IRRIGATION_CHANNEL')
    if cfg.irr_num_hours <= 0:
        raise ValueError('irr_num_hours must be greater than zero; irrigation divides the daily water amount by this value')
    if cfg.irr_freq <= 0:
        raise ValueError('irr_freq must be greater than zero; irrigation timing uses it as a modulo divisor')
    if cfg.irr_ph not in (0, 1, 2):
        raise ValueError('irr_ph must be 0, 1 or 2; other values leave WRF phase undefined')
    if not 0 <= cfg.irr_start_hour <= 23 or cfg.irr_num_hours > 24:
        raise ValueError('irrigation requires irr_start_hour in 0..23 and irr_num_hours in 1..24; larger windows exceed WRF single-day wrap arithmetic')
    if not math.isfinite(cfg.irr_daily_amount) or cfg.irr_daily_amount < 0:
        raise ValueError('irr_daily_amount must be finite and nonnegative')
    if option == 1 and (cfg.sf_surface_mosaic or cfg.sf_urban_physics):
        raise ValueError('channel irrigation currently requires dominant-category Noah without urban coupling; the tile and urban soil-water kernels do not carry IRRIGATION_CHANNEL')
    if option == 2 and cfg.sf_surface_physics == 0:
        raise ValueError('drip irrigation requires an LSM to consume its RAINBL water')
    if option == 3 and (not cfg.moist or not cfg.mp_physics):
        raise ValueError('sprinkler irrigation requires moist state and active microphysics to consume its QR water')


def validate_run_config(cfg: RunConfig, *, native_fire_ideal: bool = False) -> RunConfig:
    """The RunConfig invariant battery, shared by BOTH loaders.

    Historically these checks lived inline in :func:`load_config`; the
    Phase-5 experiment path (woof/experiment.py ``build_experiment``)
    constructs per-domain RunConfigs directly and MUST apply the same
    battery (p5t1 review F1) -- same checks, same messages, so a
    hand-typed ``mp_physics = 55`` or odd ``time_step_sound`` in an
    experiment TOML fails exactly as it always has on the legacy path.
    Returns ``cfg`` unchanged on success.
    """
    from woof.dfi_control import validate_dfi
    validate_dfi(cfg)

    from woof.fddaobs.config import validate as validate_observation_nudging
    validate_observation_nudging(cfg)
    from woof.core.tslist import validate_config as validate_tslist_config
    validate_tslist_config(cfg)
    from woof.moving_frame import validate_moving_frame
    validate_moving_frame(cfg)
    from woof.tc.config import validate_tc
    validate_tc(cfg)
    from woof.core.passive_tracer_contract import validate_passive_tracers
    validate_passive_tracers(cfg)
    from woof.core.madwrf_contract import validate_madwrf
    validate_madwrf(cfg)
    from woof.core.trajectory_contract import validate as validate_trajectory
    validate_trajectory(cfg)
    from woof.lightning_contract import validate_lightning
    validate_lightning(cfg)
    if type(cfg.hailcast_opt) is not int or cfg.hailcast_opt not in (-1, 0, 1):
        raise ConfigurationRefused("hailcast_opt must be -1 (automatic), 0 or 1")
    if not math.isfinite(cfg.haildt) or not 0 <= cfg.haildt <= 60:
        raise ConfigurationRefused("haildt must be between 0 (every step) and 60 seconds")
    if cfg.hailcast_version not in ("wrf_39_noaa", "wrf_461"):
        raise ConfigurationRefused(
            f"hailcast_version={cfg.hailcast_version!r} must be 'wrf_39_noaa' "
            "(operational HRRR v4) or 'wrf_461' (WRF v4.6.1)")
    validate_fire_config(cfg)
    if not isinstance(cfg.ocean_config, str):
        raise ValueError("ocean_config must name a TOML ocean contract")
    from woof.ocean.settings import resolve as resolve_waves
    waves_enabled, _ = resolve_waves(cfg)
    if waves_enabled and not cfg.sf_sfclay_physics:
        raise ValueError("waves require an active surface layer to supply 10m winds")
    if cfg.ocean_config and not cfg.sf_sfclay_physics:
        raise ValueError("ocean_config requires an active surface layer to supply 10m winds and friction velocity")
    if type(cfg.tmn_update) is not int or cfg.tmn_update not in (0, 1):
        raise ValueError("tmn_update must be integer 0 or 1: no other TMN evolution is implemented")
    if type(cfg.lagday) is not int or cfg.lagday < 1:
        raise ValueError("lagday must be a positive integer: TLAG needs a nonempty daily history")
    if cfg.tmn_update == 1 and cfg.lagday != 150:
        raise ValueError("tmn_update=1 requires lagday=150 (WRF share/module_check_a_mundo.F:2306-2312): other history lengths change the deep-soil mean")
    if type(cfg.sst_update) is not int or cfg.sst_update not in (0, 1):
        raise ValueError("sst_update must be integer 0 or 1")
    if not isinstance(cfg.surface_analysis, str):
        raise ValueError("surface_analysis must be a manifest path string")
    if cfg.surface_analysis and not cfg.sst_update:
        raise ValueError("surface_analysis requires sst_update=1; otherwise the supplied SST and ice would never be read")
    if type(cfg.surface_energy_diag) is not bool:
        raise ConfigurationRefused("surface_energy_diag must be boolean")
    validate_irrigation_config(cfg)
    why = terrain_drag_refusal(
        topo_wind=cfg.topo_wind, gwd_opt=cfg.gwd_opt,
        bl_pbl_physics=cfg.bl_pbl_physics,
        sf_urban_physics=cfg.sf_urban_physics)
    if why is not None:
        raise ConfigurationRefused(why)
    validate_fog_kfeta_config(cfg)
    validate_radiation_driver_options(cfg)
    for name in ("slope_rad", "topo_shading"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or value not in (0, 1):
            raise ConfigurationRefused(
                f"{name} must be 0 or 1 (WRF Registry.EM_COMMON: 1 turns "
                f"it on), got {value!r}")
    if not (math.isfinite(float(cfg.shadlen)) and float(cfg.shadlen) > 0.0):
        raise ConfigurationRefused(
            f"shadlen must be a finite positive length in metres (WRF's "
            f"default is 25000.), got {cfg.shadlen!r}: the shadow search "
            "steps int(shadlen/dx + 1) cells toward the sun")
    if cfg.eta_levels is not None:
        # Shape and monotonicity only.  p_top is NOT a RunConfig field, so
        # the pressure-dependent half of the eta contract cannot run here;
        # it runs where the model top is actually known (the offline child
        # reads it off the parent archive).
        eta = tuple(float(value) for value in cfg.eta_levels)
        if len(eta) != int(cfg.nz) + 1:
            raise ConfigurationRefused(
                f"eta_levels has {len(eta)} interfaces but nz={cfg.nz} "
                f"requires {int(cfg.nz) + 1}: the ladder and the level count "
                "have to describe one grid, and a mismatch here would reach "
                "make_vertical_coord as a shape error after the run had "
                "already been prepared.")
        if not all(math.isfinite(value) for value in eta):
            raise ConfigurationRefused("eta_levels must all be finite")
        if (eta[0] != 1.0 or eta[-1] != 0.0
                or any(b >= a for a, b in zip(eta, eta[1:]))):
            raise ConfigurationRefused(
                "eta_levels must decrease strictly from 1.0 at the surface "
                f"to 0.0 at the model top, got {eta[0]!r} .. {eta[-1]!r}: a "
                "non-monotone ladder folds the coordinate and the reference "
                "dry pressure stops decreasing with height.")
    if cfg.time_step_sound % 2 != 0:
        raise ConfigurationRefused(
            f"time_step_sound must be even, got {cfg.time_step_sound}: RK3 "
            "stage 2 runs time_step_sound//2 acoustic substeps of dt/"
            "time_step_sound, which only integrates to dt/2 for even values "
            "(WRF requires even time_step_sound)."
        )
    if cfg.nz < 4:
        raise ConfigurationRefused(
            f"nz must be >= 4 (the vertical stencil width), got {cfg.nz}: "
            "the surface extrapolation coefficients (cf1/cf2/cf3) and the "
            "k=0 pressure-gradient stencil read the three lowest half "
            "levels."
        )
    hybrid_choices = (0, 1, 2, 3) if native_fire_ideal else (0, 1, 2)
    if cfg.hybrid_opt not in hybrid_choices:
        raise ConfigurationRefused(
            f"hybrid_opt must be 0 or 1 (B(eta) = eta, the Phase 1 sigma "
            f"coordinate) or 2 (WRF v4 cubic-B hybrid), got "
            f"{cfg.hybrid_opt}. The native fire-ideal initializer also "
            "implements 3 (WRF sine-squared hybrid); the generic vertical "
            "initializer does not produce that coefficient set."
        )
    if native_fire_ideal and (cfg.ifire != 2 or cfg.map_proj != 0):
        raise ConfigurationRefused(
            "fire-ideal requires ifire=2 and map_proj=0: its sounding and "
            "Cartesian coordinates initialize the native ideal fire domain")
    if cfg.map_proj not in (0, 1, 2, 3):
        raise ConfigurationRefused(
            f"map_proj must be 0 (idealized/none), 1 (Lambert conformal), "
            f"2 (polar stereographic), or 3 (Mercator), got {cfg.map_proj}."
        )
    # ---- WRF selector schema, then port readiness ------------------
    # Order matters. A value outside woof's schema (sf_surface_physics=8,
    # a RUC request carrying Noah's four layers) is nonsense in WRF terms
    # and is reported as such.  Only a schema-COHERENT request reaches
    # require_ready_wrf_physics, which then emits one complete receipt for
    # every unfinished component instead of failing on the first number.
    validate_temf_pairing(cfg)
    if cfg.sf_sfclay_physics == 89 and cfg.bl_pbl_physics != 0:
        raise ConfigurationRefused(
            "sf_sfclay_physics=89 requires bl_pbl_physics=0: WRF 4.7.1 "
            "module_physics_init.F:3140-3223 assigns no isfc value to "
            "IDEALSCMSFCSCHEME before the PBL compatibility checks")
    if cfg.sf_sfclay_physics == 89 and cfg.sf_surface_physics != 0:
        raise ConfigurationRefused(
            "sf_sfclay_physics=89 requires sf_surface_physics=0: the native "
            "prescribed-flux surface leaves UST/CHS/CHS2/CQS2 unchanged, "
            "so an active LSM has no surface producer for those inputs")
    if cfg.sf_sfclay_physics not in SURFACE_LAYER_SCHEMES:
        raise ConfigurationRefused(
            "sf_sfclay_physics must be 0 (off), 1 (revised MM5), 2 (Eta "
            "similarity, the MYJ pair), 3 (GFS), 5 (MYNN), 7 (Pleim-Xiu), "
            "10 (TEMF), 89 (ideal SCM prescribed flux), or 91 (classic MM5), got "
            f"{cfg.sf_sfclay_physics}."
            + wrf_unported_refusal_tail("sf_sfclay_physics",
                                        cfg.sf_sfclay_physics)
        )
    if cfg.sf_surface_physics == 7:
        for name in ("pxlsm_soil_nudge", "pxlsm_modis_veg"):
            if getattr(cfg, name) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1")
        if cfg.pxlsm_smois_init < 0:
            raise ValueError("pxlsm_smois_init must be nonnegative")
        if cfg.pxlsm_soil_nudge and cfg.sgfdda_interval_m <= 0:
            raise ValueError("PX soil nudging needs a positive sgfdda_interval_m")
        if cfg.pxlsm_soil_nudge and (cfg.io_form_sgfdda != 2 or not cfg.sgfdda_inname):
            raise ConfigurationRefused("PX soil nudging requires native NetCDF auxinput9 (io_form_sgfdda=2) and sgfdda_inname")
    if cfg.sf_surface_physics not in LAND_SURFACE_SCHEMES:
        raise ConfigurationRefused(
            "sf_surface_physics must be 0 (none), 1 (five-layer thermal "
            "diffusion), 2 (Noah LSM), 3 (RUC LSM), 4 (Noah-MP LSM), or 7 (Pleim-Xiu LSM), "
            "WRF numbering, got "
            f"{cfg.sf_surface_physics}."
            + wrf_unported_refusal_tail("sf_surface_physics",
                                        cfg.sf_surface_physics)
        )
    # One implementation of the soil-geometry rule, not two.  This used to
    # repeat the table lookup inline while soil_layer_count did no lookup at
    # all, which meant a validated config and a code-built one disagreed
    # about the same question.  Calling the resolver is what makes them agree.
    soil_layer_count(cfg)
    if cfg.bl_pbl_physics not in PBL_SCHEMES:
        raise ConfigurationRefused(
            f"bl_pbl_physics must be 0 (none), 1 (YSU), "
            f"{MYJ_PBL_SCHEME} (MYJ), {GFS_PBL_SCHEME} (GFS), {QNSE_PBL_SCHEME} (QNSE), 5 (MYNN), {ACM2_PBL_SCHEME} (ACM2), {BOULAC_PBL_SCHEME} (BouLac), {UW_PBL_SCHEME} (UW), "
            f"10 (TEMF), 11 (Shin-Hong), {GBM_PBL_SCHEME} (GBM), {EEPS_PBL_SCHEME} (EEPS), {KEPS_PBL_SCHEME} (K-epsilon), {MRF_PBL_SCHEME} (MRF), {SASE_PBL_SCHEME} (SASE, experimental), or "
            f"{CCPP_TKE_EDMF_PBL_SCHEME} (CCPP sa-TKE-EDMF, the GFS/HAFS PBL), "
            f"got {cfg.bl_pbl_physics}."
            + wrf_unported_refusal_tail("bl_pbl_physics", cfg.bl_pbl_physics)
        )
    if cfg.bl_pbl_physics == CCPP_TKE_EDMF_PBL_SCHEME:
        if not cfg.moist:
            raise ValueError("CCPP TKE-EDMF requires moist=true; a dry state cannot supply its moisture and cloud buoyancy")
        if cfg.nz < 8:
            raise ValueError("CCPP TKE-EDMF requires nz >= 8; shorter columns cannot supply its mass-flux searches")
        if cfg.sf_sfclay_physics not in (1, 91):
            raise ValueError("CCPP TKE-EDMF requires sf_sfclay_physics=1 or 91; its stability closure divides by the fm/fh similarity denominators those layers publish")
        if cfg.bldt != 0:
            raise ValueError("CCPP TKE-EDMF requires bldt=0; positive cadence would retain a TKE tendency that the current extra-scalar restart stream does not serialize")
    validate_selector_couplings(cfg)
    validate_gbmpbl_config(cfg)
    validate_sase_config(cfg)
    validate_gfs_pbl_config(cfg)
    validate_gfs_surface_pairing(cfg)
    validate_uwpbl_config(cfg)
    validate_eeps_config(cfg)
    validate_qnse_config(cfg)

    validate_keps_config(cfg)
    validate_urban_config(cfg)
    validate_chem_config(cfg)
    require_ready_wrf_physics(
        mp_physics=cfg.mp_physics,
        sf_sfclay_physics=cfg.sf_sfclay_physics,
        bl_pbl_physics=cfg.bl_pbl_physics,
        sf_surface_physics=cfg.sf_surface_physics,
        num_soil_layers=cfg.num_soil_layers,
        # The grid width the readiness authority needs for the one blocker that
        # is about throughput rather than a missing branch.  nx*ny is the whole
        # grid: the land fraction is a property of the ingested landmask, not
        # of the configuration, so a fail-closed rail uses the worst case.
        columns=int(cfg.nx) * int(cfg.ny))
    if cfg.isftcflx not in (0, 1, 2):
        raise ConfigurationRefused(
            "isftcflx must be 0 (standard MM5 water-point roughness), 1 "
            "(Garratt), or 2 (Donelan), got "
            f"{cfg.isftcflx}."
        )
    if cfg.iz0tlnd not in (0, 1, 2):
        raise ConfigurationRefused(
            "iz0tlnd must be 0 (standard CZIL), 1 (Chen-Zhang thermal "
            "roughness), or 2 (fixed CZIL 0.1), got "
            f"{cfg.iz0tlnd}."
        )
    if cfg.opt_thcnd not in (1, 2):
        raise ConfigurationRefused(
            "opt_thcnd must be 1 (Johansen soil thermal conductivity) or 2 "
            "(McCumber-Pielke, clamped for soiltyp 4), got "
            f"{cfg.opt_thcnd}."
        )
    if type(cfg.sst_skin) is not int or cfg.sst_skin not in (0, 1):
        raise ValueError("sst_skin must be integer 0 or 1")
    if cfg.isfflx not in (0, 1, 2):
        raise ConfigurationRefused(
            "isfflx must be 0 (prescribed tke_drag_coefficient/"
            "tke_heat_flux under the PBL-off Smagorinsky path; surface-"
            "scheme fluxes off), 1 (surface-scheme fluxes on), or 2 "
            "(surface-scheme drag/moisture with prescribed tke_heat_flux), "
            f"got {cfg.isfflx}."
        )
    # km_opt=1 under diff_opt=2 runs the same vertical_diffusion_2 surface
    # arms as km_opt 2-4 (constant_k_mixing_active), so it consumes the
    # prescribed forcing too.
    _prescribed_flux_consumer = cfg.bl_pbl_physics == 0 and (
        cfg.km_opt in (2, 3, 4, 5) or constant_k_mixing_active(cfg))
    if cfg.isfflx == 0 and cfg.sf_sfclay_physics == 0 \
            and not _prescribed_flux_consumer:
        raise ConfigurationRefused(
            "isfflx=0 has no consumer when sf_sfclay_physics=0 unless the "
            "PBL-off turbulence path (km_opt=2/3/4, or 1 under diff_opt=2; "
            "bl_pbl_physics=0) is "
            "active to take the prescribed tke_drag_coefficient/"
            "tke_heat_flux forcing; woof otherwise implements the gate "
            "in its MM5 and MYNN surface-layer paths."
        )
    if cfg.isfflx == 2 and not _prescribed_flux_consumer:
        raise ConfigurationRefused(
            "isfflx=2 (prescribed tke_heat_flux) is consumed only by "
            "WRF's diff_opt=2 vertical_diffusion_2 path, which woof "
            "runs under km_opt=2/3/4 (or 1 under diff_opt=2) with "
            "bl_pbl_physics=0 "
            "(module_diffusion_em.F:4286-4305); enable that path or "
            "choose isfflx 0/1."
        )
    if cfg.o3input not in (0, 2):
        raise ConfigurationRefused(
            "o3input must be 0 (legacy-RRTMG wrapper O3DATA) or 2 "
            f"(CAM climatology), got {cfg.o3input}."
        )
    if cfg.ghg_input not in (0, 1):
        raise ConfigurationRefused(
            "ghg_input must be 0 (fixed RRTMG gases) or 1 (CAMtr "
            f"time-varying gases), got {cfg.ghg_input}.")
    if cfg.use_mp_re not in (0, 1):
        raise ConfigurationRefused(
            "use_mp_re must be 0 (legacy-RRTMG calculated radii) or 1 "
            f"(use the WRF microphysics scheme table), got {cfg.use_mp_re}."
        )
    # The carrier policy is validated HERE as well as at the seam that
    # consumes it, so a misspelled policy is refused at config load with
    # the two choices named rather than three minutes into a run.
    #
    # The vocabulary lives in THIS module, not in woof.core, and the
    # import direction is the reason: the standalone RW-WPS preprocessing
    # wheel carries woof.config and does NOT carry woof.core, so a
    # config-load refusal that reached into the core would hand a
    # standalone user an ImportError instead of the sentence written for
    # them.  That is the exact defect that withdrew 1.8.8.
    validate_surface_radiation_policy(cfg.surface_radiation_policy)
    from woof.noahmp_generation import snow_age_generation
    snow_age_generation(cfg.noahmp_snow_age)
    validate_seaice_parameters(cfg)
    if (not math.isfinite(cfg.seaice_albedo_default)
            or not 0.0 <= cfg.seaice_albedo_default <= 1.0):
        raise ConfigurationRefused(
            "seaice_albedo_default must be finite and in [0, 1], got "
            f"{cfg.seaice_albedo_default}."
        )
    if (cfg.seaice_albedo_default != 0.65
            and cfg.sf_surface_physics not in (2, 3, 4)):
        raise ConfigurationRefused(
            "a nondefault seaice_albedo_default is implemented only by "
            "Noah and Noah-MP (sf_surface_physics=2/4, seaice_noah) and RUC LSM "
            "(sf_surface_physics=3), got "
            f"sf_surface_physics={cfg.sf_surface_physics}."
        )
    if type(cfg.rdmaxalb) is not bool:
        raise ConfigurationRefused(
            f"rdmaxalb must be boolean, got {cfg.rdmaxalb!r}."
        )
    if isinstance(cfg.ifsnow, bool) or not isinstance(cfg.ifsnow, int):
        raise ValueError(f"ifsnow must be an integer, got {cfg.ifsnow!r}.")
    if cfg.sf_surface_physics == 1:
        from woof.core.slab import require_admitted_ifsnow
        require_admitted_ifsnow(cfg.ifsnow)
    elif cfg.ifsnow != 1:
        raise ValueError(
            "ifsnow is read only by the slab scheme (sf_surface_physics=1); "
            f"with sf_surface_physics={cfg.sf_surface_physics} a nondefault "
            "value would be accepted and change nothing. Noah, RUC and "
            "Noah-MP always run their own snow physics.")
    if not cfg.rdmaxalb and cfg.sf_surface_physics != 2:
        raise ConfigurationRefused(
            "rdmaxalb=false is implemented by Noah LSMINIT and requires "
            "sf_surface_physics=2, got "
            f"sf_surface_physics={cfg.sf_surface_physics}."
        )
    validate_noah_mosaic_config(cfg)
    if cfg.ua_phys and int(cfg.sf_surface_physics) != 2:
        raise ValueError("ua_phys=true requires sf_surface_physics=2: other land schemes do not apply Noah canopy burial and canopy heat flux")
    if cfg.sf_surface_physics not in (2, 3) and (
            cfg.usemonalb or cfg.rdlai2d):
        raise ConfigurationRefused(
            "usemonalb/rdlai2d are implemented by the Noah LSM "
            "(sf_surface_physics=2) and the RUC LSM (sf_surface_physics=3); "
            "Noah-MP and the slab carry no branch that reads the monthly "
            "albedo or LAI field, so the flag would be accepted and ignored. "
            f"The selected land-surface model is sf_surface_physics="
            f"{cfg.sf_surface_physics}."
        )
    if cfg.sf_surface_physics != 2 and cfg.opt_thcnd != 1:
        raise ConfigurationRefused(
            "opt_thcnd is a Noah LSM option (the soil thermal conductivity "
            "branch at woof/core/kernels/noah.cu:181) and requires "
            "sf_surface_physics=2; the selected land-surface model is "
            f"sf_surface_physics={cfg.sf_surface_physics}."
        )
    if (type(cfg.fractional_seaice) is not int
            or cfg.fractional_seaice not in (0, 1)):
        raise ConfigurationRefused(
            f"fractional_seaice={cfg.fractional_seaice!r} must be the integer "
            "0 (sea-ice threshold 0.5) or 1 (threshold 0.02), WRF's two "
            "values (module_surface_driver.F fractional_seaice select)."
        )
    if (isinstance(cfg.seaice_threshold, bool)
            or not isinstance(cfg.seaice_threshold, (int, float))
            or not math.isfinite(float(cfg.seaice_threshold))
            or not 0.0 < float(cfg.seaice_threshold) <= 400.0):
        raise ConfigurationRefused(
            f"seaice_threshold={cfg.seaice_threshold!r} must be a finite "
            "skin temperature in K, 0 < value <= 400 (WRF &physics "
            "seaice_threshold, a REAL; the default 100 converts no water "
            "cell to ice).")
    if (float(cfg.seaice_threshold) > 170.0
            and cfg.sf_surface_physics not in (1, 2, 3, 4, 8)):
        raise ConfigurationRefused(
            f"seaice_threshold={cfg.seaice_threshold!r} converts open water "
            "colder than it to sea ice in the soil ingest of Noah, RUC, "
            "Noah-MP and SSiB (woof/ingest/soil.py preprocess_noah_soil, real.exe's "
            "adjust_for_seaice_post); sf_surface_physics="
            f"{cfg.sf_surface_physics} has no such ingest, so the cells would "
            "stay open water and the requested rule would silently not run.")
    if cfg.fractional_seaice == 1 and cfg.sf_surface_physics not in (2, 3, 4, 7, 8):
        raise ConfigurationRefused("fractional_seaice=1 needs a land model with WRF's ice/water surface coupling (sf_surface_physics=2/3/4/7/8)")
    for key, allowed in (("seaice_albedo_opt", (0,1,2)),
                         ("seaice_snowdepth_opt", (0,1)),
                         ("seaice_thickness_opt", (0,1))):
        value = getattr(cfg, key)
        if type(value) is not int or value not in allowed:
            raise ConfigurationRefused(f"{key} must be one of {allowed}")
        if value and cfg.sf_surface_physics not in (2,4):
            raise ConfigurationRefused(f"{key}={value} requires Noah or Noah-MP, which call WRF seaice_noah")
    for key in ("seaice_thickness_default", "seaice_snowdepth_min", "seaice_snowdepth_max"):
        value = getattr(cfg, key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
            raise ConfigurationRefused(f"{key} must be finite and positive")
    if (cfg.sf_surface_physics not in (2, 4) and
            (cfg.seaice_thickness_default != 3.0 or
             cfg.seaice_snowdepth_min != .001 or
             cfg.seaice_snowdepth_max != 1.e10)):
        raise ConfigurationRefused(
            "prescribed sea-ice thickness and snow-depth bounds require "
            "Noah or Noah-MP, which call WRF seaice_noah")
    if cfg.seaice_snowdepth_max < cfg.seaice_snowdepth_min:
        raise ConfigurationRefused("seaice_snowdepth_max must be >= seaice_snowdepth_min")
    # Surface coupling.  The registry has always refused these two through
    # land_surface/pbl requires_components while this battery accepted them,
    # and woof/core/physics.py initialize_physics then raised at driver
    # construction -- so the configuration was dead either way, but the two
    # authorities disagreed about when, and a launcher that trusts the registry
    # and a launcher that trusts validate_run_config saw different answers.
    # tests/test_authority_agreement.py is what found it and what keeps them
    # together; the initialize_physics checks stay as the guard for a directly
    # constructed driver.
    if cfg.sf_surface_physics and not cfg.sf_sfclay_physics:
        raise ConfigurationRefused(
            f"sf_surface_physics={cfg.sf_surface_physics} requires a surface "
            "layer (sf_sfclay_physics != 0) for its exchange coefficients: "
            "Noah reads CHS/CHS2/CQS2/QGH/RIB, Noah-MP and RUC read the same "
            "seam, and with sf_sfclay_physics=0 those fields are allocated "
            "and stay identically zero -- Noah-MP's write-back then divides "
            "by chs2/cqs2 behind a substitute-1.0 guard, so the run "
            "degenerates silently instead of failing. Select "
            "sf_sfclay_physics=1 (revised MM5) or 91 (classic MM5), which "
            "run under any PBL; sf_sfclay_physics=2 (Eta), 5 (MYNN) and 7 "
            "(Pleim-Xiu) write the same seam but each is admitted only with "
            "the PBLs that read what it publishes (2 with bl_pbl_physics=2, "
            "5 with 5, 9, 900 or 0, 7 with 7, 9, 900 or 0)."
        )
    # NO GENERIC "a PBL scheme requires moist=true" RULE HERE, and its
    # absence is a decision.  Audit R-025 read one off the state
    # allocation -- woof/core/state.py allocates qv/qc/qr under
    # cfg.moist and every ported closure writes RQVBLTEN/RQCBLTEN --
    # and this door carried it for part of one candidate.  As a rule
    # over the whole slot it is false: a dry state is not an absent
    # state.  The physics seam hands every closure the persistent zero
    # physics_dry_qv / physics_dry_qc planes, the moisture rows solve
    # against zero surface boundary values so their tendencies are
    # exactly zero, and with no moist scalar advance nothing consumes
    # them.  A dry km_opt=4 + YSU + classic-MM5 plan is admitted at
    # this door and reaches the driver -- tests/test_config.py's
    # test_km_opt4_admits_pbl_off_vertical_diffusion is the pin -- and
    # a generic rule here would refuse it.
    #
    # THE ONE PER-SCHEME DRY REFUSAL THAT DOES STAND is SASE's, raised
    # where its reason lives rather than by a rule typed here: the
    # saturated Brunt-Vaisala stability it forms and the condensate
    # rows it mixes, refused by validate_sase_config.  MYJ's dry refusal
    # is gone (see validate_myj_pairing): YSU, MYJ, MYNN and Shin-Hong
    # each run dry, which tests/test_myj_port.py's
    # test_a_dry_pbl_run_reaches_the_driver_and_mixes pins.
    if cfg.cu_physics and not cfg.moist:
        raise ConfigurationRefused(
            f"cu_physics={cfg.cu_physics} requires moist=true: the cumulus "
            "schemes are moist convective schemes and woof/core/physics.py "
            "initialize_physics refuses a cumulus scheme on a dry DomainState "
            "(state.qv is None). Set moist=true, or select cu_physics=0. The "
            "registry says the same thing through the option's "
            "required_settings."
        )
    if cfg.cu_physics == 10:
        if type(cfg.numbins) is not int or not 1 <= cfg.numbins <= 46340:
            raise ConfigurationRefused("KF-CuP numbins must be an integer in [1,46340]")
        for key in ('thbinsize', 'rbinsize'):
            value = getattr(cfg, key)
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationRefused(f"KF-CuP {key} must be positive and finite")
        for key in ('mindeepfreq', 'minshallowfreq'):
            value = getattr(cfg, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigurationRefused(f"KF-CuP {key} must be finite in [0,1]")
        if type(cfg.shallowcu_forced_ra) is not bool:
            raise ConfigurationRefused("KF-CuP shallowcu_forced_ra must be boolean")
    if cfg.cu_physics == 93:
        for name in GRELL_ENSEMBLE_DEFAULTS:
            reason = grell_ensemble_refusal(name, getattr(cfg, name))
            if reason:
                raise ConfigurationRefused(reason)
        if type(cfg.cu_diag) is not int or cfg.cu_diag not in (0, 1):
            raise ConfigurationRefused("cu_diag must be 0 or 1")
        if (not math.isfinite(cfg.convtrans_avglen_m)
                or cfg.convtrans_avglen_m * 60. < cfg.dt):
            raise ConfigurationRefused("convtrans_avglen_m must span at least one model step: "
                                      "convtrans_prep would evaluate MOD(ktau,0)")
    elif cfg.cu_physics == 5:
        validate_g3_dimensions(cfg.maxiens, cfg.maxens, cfg.maxens2,
                               cfg.maxens3, cfg.ensdim, cfg.cugd_avedx)
        if type(cfg.cu_diag) is not int or cfg.cu_diag not in (0, 1):
            raise ConfigurationRefused("cu_diag must be 0 or 1")
        if (not math.isfinite(cfg.convtrans_avglen_m)
                or cfg.convtrans_avglen_m * 60. < cfg.dt):
            raise ConfigurationRefused("convtrans_avglen_m must span at least one model step: "
                             "convtrans_prep would evaluate MOD(ktau,0)")
    elif any(getattr(cfg, name) != default for name, default in
             G3_RUN_DEFAULTS.items() if name != "cugd_avedx" and not
             (name == "convtrans_avglen_m" and cfg.cu_physics == 3 and cfg.cu_rad_feedback)):
        raise ConfigurationRefused("G3 ensemble dimensions and cu_diag are read only by cu_physics=5; "
                         "other schemes would discard the requested ensemble or diagnostic window")
    if cfg.cu_physics == 3 and cfg.cu_rad_feedback and (
            not math.isfinite(cfg.convtrans_avglen_m) or cfg.convtrans_avglen_m * 60. < cfg.dt):
        raise ConfigurationRefused("convtrans_avglen_m must span one model step: native MOD(ktau,0) is undefined")
    if cfg.cu_physics in GRELL_FAMILY_CU_PHYSICS:
        if cfg.cu_physics == GRELL_FREITAS_CU_PHYSICS and not cfg.bl_pbl_physics:
            raise ConfigurationRefused(
                "cu_physics=3 (Grell-Freitas) requires a PBL scheme: the "
                "trigger's temperature/moisture excesses and the shallow "
                "arm read KPBL and the surface fluxes the PBL stack "
                "maintains; with bl_pbl_physics=0 the engine has no KPBL "
                "to hand the scheme. Select a PBL scheme, e.g. "
                "bl_pbl_physics=1 (YSU); note GF also requires "
                "cudt_minutes=0, so set both in one edit."
            )
        if (cfg.cu_physics == GRELL_FREITAS_CU_PHYSICS
                and cfg.cudt_minutes != 0.0):
            raise ConfigurationRefused(
                "cu_physics=3 (Grell-Freitas) requires cudt_minutes=0: GF "
                "runs on the model step (WRF's usual GF configuration, "
                "STEPCU=1) and carries no NCA hold; cudt is a Kain-Fritsch "
                "cadence knob."
            )
        if cfg.cu_physics == GRELL_3D_CU_PHYSICS and cfg.cudt_minutes != 0.0:
            raise ValueError(
                "cu_physics=5 (Grell 3D) requires cudt_minutes=0: woof "
                "adds the scheme's per-call RAINCV once per call, while WRF "
                "adds its held PRATEC every step between calls "
                "(module_physics_addtendc.F:2141), so a cadence above one "
                "step would drop that rain. Run it on the model step, "
                "WRF's usual G3 configuration."
            )
        refusal = gf_clos_choice_refusal(cfg.clos_choice)
        if refusal is not None:
            raise ConfigurationRefused(refusal)
        if cfg.cu_physics == GRELL_FREITAS_CU_PHYSICS:
            advice = gf_clos_choice_advice(cfg.clos_choice)
            if advice is not None:
                from woof.explain import warn_once
                warn_once(f"gf-clos-choice-{cfg.clos_choice}", advice)
        if cfg.ishallow not in (0, 1):
            raise ConfigurationRefused(
                f"ishallow must be 0 or 1, got {cfg.ishallow}.")
        if (isinstance(cfg.cugd_avedx, bool)
                or not isinstance(cfg.cugd_avedx, int)):
            raise ValueError(
                f"cugd_avedx must be an integer (WRF Registry scalar, "
                f"Registry.EM_COMMON:2543), got {cfg.cugd_avedx!r}.")
        for name, default in GRELL_FAMILY_DEFAULTS.items():
            if (name not in GRELL_FAMILY_READS[cfg.cu_physics]
                    and getattr(cfg, name) != default):
                readers = ", ".join(
                    f"cu_physics={c}" for c, keys in
                    sorted(GRELL_FAMILY_READS.items()) if name in keys)
                raise ValueError(
                    f"{name}={getattr(cfg, name)!r} is refused with "
                    f"cu_physics={cfg.cu_physics}: WRF v4.6.1's cumulus "
                    f"driver hands {name} only to {readers}, so this "
                    f"scheme would run without the requested setting. "
                    f"Leave it at the Registry default ({default}) or "
                    f"select a scheme that reads it.")
    elif any(getattr(cfg, name) != default
             for name, default in GRELL_FAMILY_DEFAULTS.items()):
        raise ConfigurationRefused(
            "clos_choice/ishallow/cugd_avedx are Grell-family keys read "
            "only where cu_physics is 3 or 5; set them with the scheme or "
            "leave the Registry defaults (0/0/1)."
        )

    # PLACED AFTER THE GRELL CHAIN ABOVE, NOT INSIDE IT.  The first
    # version of this sat between `if cfg.cu_physics == 3:` and its
    # `elif cfg.clos_choice != 0 or cfg.ishallow != 0:`, which silently
    # re-parented that elif onto the 16 test -- so every cu_physics = 3
    # config carrying ishallow = 1 was refused, which is most of the
    # campaign's.  Every unit test passed: none exercises a GF config
    # with a non-default Grell key through validate_run_config.  The
    # BASELINE RE-RUN caught it, which is the whole reason the group is
    # re-run rather than argued inert.
    if (cumulus_adaptive_cudt_zero_site(cfg.cu_physics) is not None
            and cfg.use_adaptive_time_step and cfg.cudt_minutes != 0.0):
        raise ConfigurationRefused(
            "cu_physics=14 with use_adaptive_time_step requires cudt_minutes=0: "
            "WRF resets cudt for KSAS: its closure is formed over dt*stepcu and cannot be held "
            "across variable steps (share/module_check_a_mundo.F:2141-2160).")
    if cfg.cu_physics == 6 and cfg.cudt_minutes != 0.0:
        raise ValueError(
            "cu_physics=6 requires cudt_minutes=0: the original scheme "
            "returns per-call precipitation and carries no NCA hold; "
            "holding it would reapply that precipitation each step.")
    if cfg.cu_physics == 16:
        # NEW TIEDTKE PINS cudt_minutes = 0 for the same reason GF does,
        # and one it does not share.
        #
        # Shared: it runs on the model step and carries no NCA hold. cudt
        # is a Kain-Fritsch cadence knob.
        #
        # Its own: cudt_minutes DEFAULTS TO 5.0, so a config that merely
        # selects 16 and says nothing else would hold the scheme's rates
        # for five minutes -- for a scheme whose RAINCV is rn/stepcu, a
        # per-call rate with no persistence (module_cu_ntiedtke.F:506-509,
        # verified against the real routine rather than read). The held
        # rate would be reapplied every step of the hold and the
        # precipitation would be five times what the scheme produced.
        if cfg.cudt_minutes != 0.0:
            raise ConfigurationRefused(
                "cu_physics=16 (New Tiedtke) requires cudt_minutes=0: the "
                "scheme runs on the model step and carries no NCA hold, "
                "and its RAINCV is a per-call rate (rn/stepcu) with no "
                "persistence -- a five-minute hold would reapply it every "
                "step. cudt is a Kain-Fritsch cadence knob."
            )
        # NEW TIEDTKE DOES NOT REQUIRE A PBL SCHEME.  The refusal that
        # stood here was cloned from Grell-Freitas ("same requirement GF
        # states, for the same reason") and the reason does not transfer.
        # GF's is real: it reads fields["kpbl"] into a ONE-BASED column
        # index and divides by t[kpbl], so with the slot off it indexes
        # slot 0 of an uninitialised workspace.  New Tiedtke reads no
        # kpbl anywhere.  Its two PBL-named inputs are defined with the
        # slot off: hfx/qfx come from the SURFACE stack (the surface
        # layer and the LSM, which woof/core/physics.py dispatches
        # independently of bl_pbl_physics), and gf_rthblten/gf_rqvblten
        # are the zero-allocated advective-forcing lanes, which is
        # exactly WRF's own RTHBLTEN=0 fold in
        # module_cumulus_driver.F:879-880.  The "cumastrn:509 zdhpbl"
        # integral runs from the CLOUD-BASE index kcbot, not from a PBL
        # index, and its non-positive cases have defined fallbacks.
    if cfg.cu_physics in NSAS_DX_FACTOR_SCHEMES:
        if type(cfg.nsas_dx_factor) is not int or cfg.nsas_dx_factor not in (0, 1):
            raise ValueError(
                "nsas_dx_factor must be 0 or 1 (module_cu_nsas.F:567: 1 "
                "scales the trigger and closure velocities by 250/dx, any "
                "other value takes the fixed branch), got "
                f"{cfg.nsas_dx_factor!r}: a 2 would run the fixed branch "
                "under a name that says otherwise.")
    elif cfg.nsas_dx_factor != 0:
        raise ValueError(
            "nsas_dx_factor is read only by cu_physics=96 (New SAS); with "
            f"cu_physics={cfg.cu_physics} it would reach nothing and the run "
            "would not be the configuration it names. Leave it at 0.")
    if cfg.cu_physics == SAS_CU_PHYSICS:
        if cfg.use_adaptive_time_step:
            raise ValueError(
                "cu_physics=94 (SAS) cannot use an adaptive time step: "
                "CU_SAS closes over DT*STEPCU and holds rates for that fixed "
                "window; WRF refuses this pair (module_cumulus_driver.F:1131-1135).")
        if type(cfg.shalconv) is not int or cfg.shalconv not in (0, 1):
            raise ValueError("shalconv must be 0 or 1: CU_SAS executes the shallow branch only at 1")
    elif cfg.shalconv != 1:
        raise ValueError("shalconv is consumed only by cu_physics=94; leave it at 1 otherwise")
    if cfg.cu_physics in SAS_PGCON_SCHEMES or cfg.shcu_physics == 4:
        if not math.isfinite(float(cfg.sas_pgcon)):
            raise ValueError(
                "sas_pgcon must be finite: it multiplies every cloud "
                "momentum term of the scheme (module_cu_nsas.F:937-944), so a "
                f"non-finite value poisons RUCUTEN/RVCUTEN, got {cfg.sas_pgcon!r}.")
    elif cfg.sas_pgcon != 0.55:
        raise ValueError(
            "sas_pgcon is read only by the SAS cumulus family "
            f"(cu_physics in {sorted(SAS_PGCON_SCHEMES)}); with "
            f"cu_physics={cfg.cu_physics} it would reach nothing and the run "
            "would not be the configuration it names. Leave it at 0.55.")
    if (cfg.cu_physics in (95, 96) and cfg.use_adaptive_time_step
            and cfg.cudt_minutes != 0.0):
        # WRF's own rule (module_check_a_mundo.F:2142-2160): with an
        # adaptive step every SAS scheme runs every step.  The breakage it
        # prevents is concrete: the scheme forms its rates over
        # delt = dt*stepcu (module_cu_nsas.F:163) with the step of the CALL,
        # and holds them until the next call while the clock keeps moving,
        # so the held rate and the accumulated rain describe an interval
        # the run did not take.  WRF resets cudt to 0 and says so; the
        # namelist door does the same, and a hand-built configuration is
        # told here.
        raise ValueError(
            f"cu_physics={cfg.cu_physics} (SAS) with use_adaptive_time_step requires "
            "cudt_minutes=0, as WRF requires of every SAS scheme "
            "(module_check_a_mundo.F:2142-2160): the scheme's rates are "
            "formed over dt*stepcu at the call and held while the adaptive "
            "clock moves, so a nonzero cudt would apply them over an "
            "interval the run did not take.")
    if type(cfg.bmj_rad_feedback) is not bool:
        raise ValueError("bmj_rad_feedback must be bool")
    if cfg.bmj_rad_feedback:
        if cfg.cu_physics not in COLUMN_KERNEL_CU_SCHEMES:
            raise ValueError("bmj_rad_feedback requires cu_physics=2: other schemes do not publish BMJ cloud fields")
        if not bmj_feedback_radiation_supported(radiation_scheme_ids(cfg)):
            raise ValueError("bmj_rad_feedback requires physical cloud radiation or disabled streams: the selected radiation adapter does not merge the carried convective cloud")
    if cfg.sf_sfclay_physics not in (1, 91) and (
            cfg.isftcflx or cfg.iz0tlnd):
        raise ConfigurationRefused(
            "isftcflx/iz0tlnd are MM5 surface-layer options and require "
            "sf_sfclay_physics=1 or 91; the selected surface layer is "
            f"sf_sfclay_physics={cfg.sf_sfclay_physics}."
        )
    from woof.ysu_generations import ysu_generation
    ysu_generation(cfg.ysu_version)
    if cfg.ysu_topdown_pblmix not in (0, 1):
        raise ConfigurationRefused(
            "ysu_topdown_pblmix must be 0 or 1, got "
            f"{cfg.ysu_topdown_pblmix}."
        )
    from woof.wind_table import validate_config as validate_wind_config
    validate_wind_config(cfg)
    if type(cfg.bl_mynn_tkeadvect) is not bool:
        raise TypeError("bl_mynn_tkeadvect must be a bool")
    if type(cfg.scalar_adv_opt) is not int or cfg.scalar_adv_opt not in (0, 1):
        raise ValueError("scalar_adv_opt must be 0 (ordinary) or 1 (positive definite)")
    if cfg.bl_mynn_tkeadvect and cfg.bl_pbl_physics != 5:
        raise ValueError("bl_mynn_tkeadvect requires the MYNN bl_pbl_physics=5 owner")
    for name, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        value = getattr(cfg, name)
        if name == 'icloud_bl' and cfg.shcu_physics == 5 and value == 0 and type(value) is int:
            continue
        if type(value) is not type(admitted) or (value not in MYNN471_OPTION_VALUES[name] if cfg.bl_mynn_version == "wrf_471" else value != admitted):
            implemented = (
                f"{name} in {MYNN471_OPTION_VALUES[name]!r}"
                if cfg.bl_mynn_version == "wrf_471" else f"{name}={admitted!r}")
            raise ConfigurationRefused(
                f"{name}={value!r} is outside the admitted MYNN option "
                f"identity; woof implements {implemented} only, and "
                "no nearby branch is substituted for an unported one."
            )
    if type(cfg.bl_mynn_mixlength) is not int or cfg.bl_mynn_mixlength not in ((0, 1, 2) if cfg.bl_mynn_version == "wrf_471" else (1, 2)):
        raise ConfigurationRefused(
            "bl_mynn_mixlength must be 1 or 2 (or 0, WRF 4.7.1's, which "
            "runs bl_mynn_version = 'wrf_471'); other WRF mixing-length "
            "branches have no implementation in the column solver.")
    if cfg.bl_pbl_physics == 5 and cfg.bl_mynn_edmf > 0 and cfg.ishallow > 0:
        raise ConfigurationRefused(
            "bl_mynn_edmf > 0 requires ishallow=0 with MYNN: WRF v4.6.1 "
            "share/module_check_a_mundo.F:1769-1783 refuses concurrent "
            "MYNN mass flux and Grell shallow convection. Select ishallow=0 "
            "or a PBL configuration without MYNN mass flux.")
    if type(cfg.scalar_pblmix) is not int or cfg.scalar_pblmix not in (0, 1):
        raise ConfigurationRefused(
            f"scalar_pblmix={cfg.scalar_pblmix!r} must be the integer 0 "
            "(off) or 1 (scalar diffusion): the PBL driver diffuses the "
            "scalars only for exactly 1, so any other value would run "
            "without the mixing its configuration names.")
    if cfg.scalar_pblmix == 1:
        for key in MYNN_QN_MIX_KEYS:
            if getattr(cfg, key) == 1:
                raise ValueError(
                    f"scalar_pblmix=1 and {key}=1 cannot be combined: "
                    "WRF disables scalar_pblmix for MYNN scalar plume mixing "
                    "(module_check_a_mundo.F:2497-2511), and both would "
                    "write the same number-species rate. Select one mixing "
                    "path.")
        validate_scalar_pblmix_consumer(cfg)
        if cfg.bldt != 0.0 and scalar_pblmix_mixes(cfg):
            raise UnsupportedConfiguration(
                "scalar_pblmix=1 requires bldt=0: scalar tendencies are "
                "recomputed every step and are not stored in the restart "
                "tendency manifest; a restart between PBL calls would "
                "otherwise drop them.")
    # W4 full admission (mf-close2 Stage B): bl_mynn_mixscalars leaves the
    # single-value identity table and is admitted at {0,1}.  The 1 arm
    # mixes under the combo the anchored oracle fixture family
    # (w4-oracle-fixtures) was generated at and the runtime was wired
    # for: MYNN itself (bl_pbl_physics=5) and a scheme whose state carries
    # the whole qn family (MYNN_QN_FAMILY_SCHEMES: nc/ni/nwfa/nifa).  Under
    # a microphysics WRF gives no number species (MYNN_QN_FLAG_SPECIES
    # empty: 0/1/6) the key is inert, as it is in WRF (every solve is
    # gated on its species flag; mynn_mixscalars_active): the former
    # "requires mp_physics=28" refusal named no breakage there -- WRF
    # mixes nothing and runs -- and is retired for those schemes.  Under a
    # microphysics WRF mixes in PART (Thompson's qni, WDM6's qnc, ...) the
    # key is refused by name below: this port's solve runs all five
    # species or none, so the run would record a mixing WRF performs and
    # it does not.  The every-step cadence (bldt=0) binds the ACTIVE case
    # only: the qn tendencies are held as plain-attribute extras outside
    # the restart TENDENCY_COMPONENTS manifest, which is restart-exact only
    # when every compute() replaces them before any read
    # (woof/core/physics.py scalar_for); the inert case holds none.
    # The MYNN-EDMF split keys (bl_mynn_mixaerosols, bl_mynn_mixnumcon)
    # follow the same rules, each over its own species (MYNN_QN_MIX_KEYS).
    for key in MYNN_QN_MIX_KEYS:
        value = getattr(cfg, key)
        if value not in (0, 1) or type(value) is not int:
            raise ValueError(
                f"{key}={value!r} is outside the admitted MYNN option "
                "identity; woof implements 0 (off) and 1 (the "
                "fixture-anchored stock qn mixing of "
                f"{', '.join(MYNN_QN_MIX_KEYS[key])}) only."
            )
        if value == 1 and cfg.bl_pbl_physics != 5:
            raise NotImplementedError(
                f"{key}=1 requires bl_pbl_physics=5 (MYNN), "
                f"got bl_pbl_physics={cfg.bl_pbl_physics}. The mixscalars "
                "arms are MYNN's own qn solves (module_bl_mynn.F:4654-4860) "
                "and no other PBL scheme reads the key -- accepting it "
                "would record a mixing option no code performs."
            )
    unported = mynn_mixscalars_unported_species(cfg)
    if unported:
        keys = [key for key in MYNN_QN_MIX_KEYS if getattr(cfg, key) == 1]
        raise NotImplementedError(
            f"{'/'.join(f'{key}=1' for key in keys)} with "
            f"mp_physics={cfg.mp_physics}: WRF mixes "
            f"{', '.join(unported)} under this key on that scheme "
            "(module_bl_mynn.F mynn_tendencies, the flag-gated solves at "
            ":4654-4860), and this port's mixscalars solve runs the whole "
            "nc/ni/nwfa/nifa family or nothing (mynn_pbl_gpu.py qn_flags; "
            "config.MYNN_QN_FLAG_SPECIES). Accepting the key would record "
            "a number-concentration mixing WRF performs and this run does "
            "not.  Select mp_physics=28 or mp_physics=900 (the whole "
            f"family), {', '.join(f'{key}=0' for key in keys)}, or a "
            "microphysics with no number species, where the key is WRF's "
            "no-op and is admitted."
        )
    if mynn_mixscalars_active(cfg):
        # Restart-exact only when every compute() replaces the held qn
        # extras before any read: bldt=0, or a fixed clock whose WRF
        # STEPBL (MAX(NINT(bldt*60/dt),1), woof/core/physics.py
        # _physics_interval_seconds) is 1, where every step is a PBL step
        # (_surface_pbl_step_due).  The MPAS column seams run that second
        # form: a positive bldt equal to the model step.
        every_step = cfg.bldt == 0.0 or (
            not cfg.use_adaptive_time_step and cfg.bldt > 0.0
            and max(int(math.floor(cfg.bldt * 60.0 / cfg.dt + 0.5)), 1) == 1)
        if not every_step:
            raise NotImplementedError(
                "MYNN number-species mixing "
                f"({', '.join(mynn_mixed_number_species(cfg))}) requires "
                "bldt=0 or a PBL call every model step, got "
                f"bldt={cfg.bldt!r} min at dt={cfg.dt!r} s. "
                "The qn tendencies are held outside the restart "
                "TENDENCY_COMPONENTS manifest and are restart-exact only "
                "when recomputed every step (physics.py scalar_for, the rw "
                "precedent); a longer cadence would make a mid-interval "
                "restart silently drop them."
            )
    from woof.core.noahmp_groundwater_slab import validate as groundwater_validate
    groundwater_validate(cfg.wtddt)
    if cfg.sf_surface_physics==4 and cfg.opt_run==8 and cfg.opt_infdv==0:
        raise ConfigurationRefused("Noah-MP opt_run=8 requires opt_infdv=1,2,3: native FSUR is unassigned at zero")
    if cfg.sf_surface_physics==4 and cfg.opt_soil==4:
        from woof.core.noahmp_soil_options import SOIL4_LABEL
        from woof.explain import warn
        warn("opt_soil=4: "+SOIL4_LABEL)
    for name, (admitted, evidence) in \
            NOAHMP_OPTION_IDENTITY_EVIDENCE.items():
        value = getattr(cfg, name)
        if type(value) is not type(admitted):
            raise ConfigurationRefused(
                f"{name}={value!r} is a {type(value).__name__} where "
                f"Noah-MP takes a {type(admitted).__name__}; write "
                f"{name}={admitted!r} or another value of that type."
            )
        if name in ("soiltstep","noahmp_acc_dt"):
            from woof.core.noahmp_soil_cadence import validate
            if name=="noahmp_acc_dt":
                from woof.core.noahmp_output import validate
            validate(value)
            continue
        if value == admitted or value in NOAHMP_OPTION_VALUES.get(name, ()):
            if name == "opt_crop" and value == 2:
                from woof.explain import warn
                warn("opt_crop=2: native WRF 4.7.1 fails here; this fixes it, not bitwise to WRF. Measured active-crop step cost is 1.4283x CROP1 on a shared GPU. Cold start is sowing; emergence uses native thermal time.")
            continue
        if name in NOAHMP_OPTIONS_WITHOUT_CONSUMER:
            # No breakage to name: the knob reaches no woof code at
            # any value, so the run goes ahead and says what the
            # setting does.
            from woof.explain import warn
            warn(
                f"{name}={value!r} reaches no woof code "
                f"({evidence}), so it changes nothing this run does; "
                f"the port behaves as {name}={admitted!r}."
            )
            continue
        raise ConfigurationRefused(
            f"{name}={value!r} is outside the admitted Noah-MP option "
            f"identity; woof implements {name}={admitted!r} only "
            f"({evidence}), and no nearby branch is substituted for an "
            "unported one."
        )
    if type(cfg.sf_ocean_physics) is not int or cfg.sf_ocean_physics not in (0, 1, 2):
        raise ValueError("sf_ocean_physics must be 0 (off), 1 (Pollard) or 2 (3DPWP)")
    for name in ocean_namelist_defaults():
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite: nonfinite ocean settings corrupt mixed-layer state")
    if type(cfg.ocean_levels) is not int or not 2 <= cfg.ocean_levels <= 256:
        raise ValueError("ocean_levels must be an integer from 2 through 256: PWP needs adjacent levels and bounded per-column device storage")
    from woof.surface_ocean import validate_pwp_profile
    validate_pwp_profile(cfg)
    if cfg.sf_ocean_physics:
        if cfg.oml_gamma <= 0:
            raise ValueError("oml_gamma must be positive: OML1D divides by the lapse rate and buoyancy frequency")
        if cfg.sf_sfclay_physics == 0:
            raise ValueError("sf_ocean_physics requires surface-layer fluxes: UST, HFX and LH would otherwise stay zero")
    from woof.ingest.lake_state import source_rows
    rows = source_rows()
    if cfg.lake_state_source not in ("auto", "forcing", *rows):
        raise ConfigurationRefused(
            f"lake_state_source={cfg.lake_state_source!r} is unknown; "
            "an unknown row would leave requested lake observations unbound")
    row = rows.get(cfg.lake_state_source)
    if row is not None and row.get("research_only"):
        if cfg.lake_state_accept_licence != row["licence"]:
            raise ConfigurationRefused(
                f"lake source {row['id']} requires lake_state_accept_licence="
                f"{row['licence']!r}; restricted data cannot become a default")
        if not cfg.lake_state_path:
            raise ConfigurationRefused(
                f"lake source {row['id']} requires lake_state_path to its acquired CF NetCDF file")
    if type(cfg.sf_lake_physics) is not int or cfg.sf_lake_physics not in (0, 1):
        raise ConfigurationRefused(
            f"sf_lake_physics={cfg.sf_lake_physics!r} must be the integer 0 "
            "(off) or 1 (the WRF CLM lake model, the one lake scheme "
            "ported): the lake call, its static fields and its restart "
            "storage are selected only for exactly 1, so any other value "
            "would leave the lake columns without the lake physics its "
            "configuration names.")
    if cfg.sf_lake_physics and cfg.sf_surface_physics not in (1, 2, 3, 4, 8):
        raise ConfigurationRefused(
            "sf_lake_physics=1 requires an active land-surface driver "
            "(sf_surface_physics=2, 3 or 4): the lake call consumes its "
            "surface state and accumulated precipitation")
    if type(cfg.use_lakedepth) is not int or cfg.use_lakedepth not in (0, 1):
        raise ConfigurationRefused(
            f"use_lakedepth={cfg.use_lakedepth!r} must be the integer 0 "
            "(lakedepth_default everywhere) or 1 (the LAKE_DEPTH "
            "bathymetry): WRF defines no other depth rule, and the static "
            "preparation reads bathymetry only for exactly 1, so another "
            "value would start the lakes at depths its configuration does "
            "not name.")
    if (isinstance(cfg.lakedepth_default, bool)
            or not isinstance(cfg.lakedepth_default, (int, float))
            or not math.isfinite(cfg.lakedepth_default)):
        raise ConfigurationRefused(
            f"lakedepth_default={cfg.lakedepth_default!r} must be a finite "
            "depth in metres: it is the depth of every lake column without "
            "bathymetry, and a depth that is not a finite number makes "
            "that column's layer thicknesses, and so its temperatures, "
            "not finite.")
    if (isinstance(cfg.lake_min_elev, bool)
            or not isinstance(cfg.lake_min_elev, (int, float))
            or not math.isfinite(cfg.lake_min_elev)):
        raise ConfigurationRefused(
            f"lake_min_elev={cfg.lake_min_elev!r} must be a finite "
            "elevation in metres: without a lake mask the lake columns are "
            "the water and ice columns at or above it, and a threshold "
            "that is not a finite number selects none of them, or every "
            "one with the ocean among them.")
    if cfg.mp_physics == 38 and cfg.thompson_version != "wrf_461":
        raise ValueError("mp_physics=38 requires thompson_version='wrf_461': "
            "the fork has no prognostic ng/qvolg or density-binned hail tables")
    if cfg.thompson_version not in ("wrf_461", "wrf_39_noaa", "wrf_44"):
        raise ValueError(
            f"thompson_version={cfg.thompson_version!r} must be 'wrf_461' "
            "(WRF v4.6.1's Thompson), 'wrf_44' (classic WRF v4.4), or 'wrf_39_noaa' (the operational WRF "
            "3.9 fork's): the two size graupel, snow and ice differently, "
            "so an unknown name cannot select either.")
    for key in ("force_read_thompson", "write_thompson_tables", "write_thompson_mp38table"):
        if type(getattr(cfg, key)) is not bool:
            raise ConfigurationRefused(f"{key} must be a boolean")
    if cfg.mp_physics != 38 and (cfg.force_read_thompson or not cfg.write_thompson_tables
                                or cfg.write_thompson_mp38table):
        raise ConfigurationRefused("Thompson cache IO controls require the MP38 table loader; "
                                   "another scheme has no runtime binding for these overrides")
    if cfg.thompson_fork_snow_fall not in ("blend", "wrf_39_noaa"):
        raise ConfigurationRefused(
            f"thompson_fork_snow_fall={cfg.thompson_fork_snow_fall!r} must "
            "be 'blend' (melting snow blends with the rain fall speed) or "
            "'wrf_39_noaa' (the fork's singular expression): the two move "
            "snow below the freezing level differently.")
    if (cfg.thompson_fork_snow_fall != "blend"
            and cfg.thompson_version != "wrf_39_noaa"):
        raise ConfigurationRefused(
            "thompson_fork_snow_fall='wrf_39_noaa' is the WRF 3.9 fork's "
            "snow fall speed and only the fork generation carries it; "
            f"thompson_version={cfg.thompson_version!r} would ignore it.  "
            "Set thompson_version = 'wrf_39_noaa' as well, or leave the "
            "snow fall at 'blend'.")
    if cfg.thompson_version == "wrf_44" and cfg.mp_physics != 8:
        raise ValueError("thompson_version='wrf_44' requires mp_physics=8; its classic column has no aerosol or hail state")
    if cfg.thompson_version == "wrf_39_noaa" and cfg.mp_physics != 28:
        raise ValueError(
            f"thompson_version={cfg.thompson_version!r} is ported for the "
            f"aerosol-aware scheme (mp_physics = 28) only; mp_physics="
            f"{cfg.mp_physics} would run its WRF v4.6.1 kernels under the "
            "fork's name.  Use mp_physics = 28, or leave thompson_version "
            "at 'wrf_461'.")
    if cfg.bl_mynn_version not in ("wrf_461", "wrf_471", "gsd_41"):
        raise ConfigurationRefused(
            f"bl_mynn_version={cfg.bl_mynn_version!r} must be 'wrf_461' "
            "(WRF v4.6.1 MYNN) or 'gsd_41' (the GSD MYNN v4.1 of the "
            "NOAA-EMC WRF 3.9 branch): the two delete or keep a dew flux "
            "and mix with different lengths, so an unknown name cannot "
            "select either.")
    if type(cfg.cycling) is not bool:
        raise TypeError("cycling must be a bool")
    if cfg.bl_mynn_cloud_tendency_form not in ("wrf_461", "gsd_41"):
        raise ConfigurationRefused("bl_mynn_cloud_tendency_form must be 'wrf_461' or 'gsd_41'")
    if cfg.bl_mynn_cloud_tendency_form == "gsd_41" and cfg.bl_mynn_version != "gsd_41":
        raise ConfigurationRefused("bl_mynn_cloud_tendency_form='gsd_41' requires bl_mynn_version='gsd_41': "
                         "the generic kernel has no fork cloud tendency branch")
    if type(cfg.bl_mynn_gsd41_unsquared_qtke) is not bool:
        raise ConfigurationRefused(
            "bl_mynn_gsd41_unsquared_qtke must be true or false: it picks "
            "one of two TKE conversions inside the gsd_41 mixing length, "
            "and a non-boolean value names neither.")
    # bl_mynn_version='gsd_41' hands radiation in-cloud QC_BL.  Every
    # radiation coupling merges it as that generation does (QC_BL times
    # CLDFRA_BL where no resolved cloud is present,
    # woof.core.mynn_radiation.merge_mynn_bl_clouds), so no radiation pair
    # is refused for it any more (U-E4; the legacy-RRTMG-only refusal that
    # stood here is retired with the merge).
    if cfg.bl_mynn_version == "gsd_41" and mynn_mixscalars_active(cfg):
        raise ConfigurationRefused(
            "bl_mynn_version='gsd_41' with bl_mynn_mixscalars=1: the scalar "
            "plume transport runs the mass flux from its sibling unit, which "
            "carries the v4.6.1 shallow-cumulus cloud only, so a gsd_41 run "
            "would hand radiation grid-mean plume water as in-cloud water.")
    validate_mynn_generation_spp(cfg)
    if cfg.mynn_sfclay_variant not in ("wrf_461", "gsl_wrf39"):
        raise ConfigurationRefused(
            f"mynn_sfclay_variant={cfg.mynn_sfclay_variant!r} must be "
            "'wrf_461' (WRF v4.6.1's MYNN surface layer) or 'gsl_wrf39' "
            "(the GSL WRF 3.9 fork's): the two solve different equations "
            "for z/L, so an unknown name cannot select either.")
    if cfg.ruc_soilprop not in ("wrf_45", "wrf_461"):
        raise ConfigurationRefused(
            f"ruc_soilprop={cfg.ruc_soilprop!r} must be 'wrf_45' (the WRF "
            "v4.0-4.5 soil-water diffusivity over the moisture above the "
            "residual) or 'wrf_461' (the WRF v4.6.1 form over total "
            "porosity): the two move different water between soil levels, "
            "so an unknown name cannot select either.")
    if cfg.ruc_snow not in ("wrf_45", "wrf_461"):
        raise ConfigurationRefused(
            f"ruc_snow={cfg.ruc_snow!r} must be 'wrf_45' (the WRF v4.0-4.5 "
            "snow scheme the operational RAP/HRRR branch carries) or "
            "'wrf_461' (the WRF v4.6.1 rewrite): the two differ in snow "
            "conductivity, cover, melt and albedo, so an unknown name cannot "
            "select either.")
    if cfg.ruc_2m_diagnostic not in ("flux", "log_profile"):
        raise ConfigurationRefused(
            f"ruc_2m_diagnostic={cfg.ruc_2m_diagnostic!r} must be 'flux' "
            "(public WRF's 2 m flux form) or 'log_profile' (the operational "
            "RAP/HRRR branch's logarithmic profile): the two write different "
            "2 m values, so an unknown name cannot select either.")
    if cfg.ruc_qvg_cold_start not in ("air", "wrf"):
        raise ConfigurationRefused(
            f"ruc_qvg_cold_start={cfg.ruc_qvg_cold_start!r} must be 'air' "
            "(the ground vapour starts from the lowest-level air) or 'wrf' "
            "(from saturation at the skin times moisture availability): the "
            "two start different surface humidity, so an unknown name "
            "cannot select either.")
    if cfg.ruc_irrigation not in ("wrf_45", "wrf_461"):
        raise ConfigurationRefused(
            f"ruc_irrigation={cfg.ruc_irrigation!r} must be 'wrf_45' (the "
            "WRF v4.0-4.5 crop-fraction-scaled soil moisture floor) or "
            "'wrf_461' (the WRF v4.6.1 per-step relaxation to 1.1 x "
            "wilting point): the two add different soil water, so an "
            "unknown name cannot select either.")
    for name in ("mosaic_lu", "mosaic_soil"):
        value = getattr(cfg, name)
        if type(value) is not int or value not in (0, 1):
            raise ConfigurationRefused(
                f"{name}={value!r} must be the integer 0 (dominant "
                "category) or 1 (fractional mosaic): the RUC drivers mix "
                "by category fraction for any nonzero value, so another "
                "value would replace the dominant-category surface "
                "parameters under a selector WRF does not define.")
        if value and cfg.sf_surface_physics != 3:
            raise ConfigurationRefused(
                f"{name}=1 requires sf_surface_physics=3: only RUC reads "
                "these weighted land and soil parameters")
    validate_spp_config(cfg)
    for name, (admitted, evidence) in RUC_OPTION_IDENTITY_EVIDENCE.items():
        value = getattr(cfg, name)
        if type(value) is not type(admitted) or value != admitted:
            raise ConfigurationRefused(
                f"{name}={value!r} is outside the admitted RUC option "
                f"identity; woof implements {name}={admitted!r} only "
                f"({evidence}), and no nearby branch is substituted for an "
                "unported one."
            )
    if cfg.ra_physics not in RA_PHYSICS_ACCEPTED:
        raise ConfigurationRefused(
            "ra_physics must be 0 (off), 4 (RTE+RRTMGP), or 90 "
            f"(analytic clear-sky proxy), got {cfg.ra_physics}."
        )
    if type(cfg.ra_sw_eclipse) is not int or cfg.ra_sw_eclipse not in (0, 1):
        raise ConfigurationRefused("ra_sw_eclipse must be 0 (undimmed) or 1 (Besselian eclipse)")
    ra_lw_physics, ra_sw_physics = radiation_scheme_ids(cfg)
    if ra_lw_physics not in RA_LW_PHYSICS_ACCEPTED:
        why = radiation_unported_reason("ra_lw_physics", ra_lw_physics)
        raise ConfigurationRefused(
            "ra_lw_physics must be 0 (off), 1 (WRF RRTM), 3 (WRF CAM), 4 "
            "(RTE+RRTMGP), 5 (WRF New Goddard), 7 (WRF Fu-Liou-Gu), 31 (idealized relaxation), 90 (analytic proxy) or 99 (GFDL Eta), "
            f"got {ra_lw_physics}." + (f" {why}." if why else ""))
    if ra_sw_physics not in RA_SW_PHYSICS_ACCEPTED:
        why = radiation_unported_reason("ra_sw_physics", ra_sw_physics)
        raise ConfigurationRefused(
            "ra_sw_physics must be 0 (off), 1 (WRF Dudhia), 2 (WRF GSFC), 3 (WRF CAM), 4 "
            "(RTE+RRTMGP), 5 (WRF New Goddard), 7 (WRF Fu-Liou-Gu), 90 (analytic proxy) or 99 (GFDL Eta), "
            f"got {ra_sw_physics}." + (f" {why}." if why else ""))
    if 7 in (ra_lw_physics, ra_sw_physics):
        if ra_lw_physics == 7 and ra_sw_physics == 0:
            raise ConfigurationRefused(
                "WRF radiation_driver rejects ra_lw_physics=7 with "
                "ra_sw_physics=0: select an active ra_sw_physics "
                "(native CASE 0).")
        if ra_lw_physics == 0:
            raise ConfigurationRefused("WRF radiation_driver has no ra_lw_physics=0 CASE for ra_sw_physics=7; select an implemented ra_lw_physics.")
        if ra_lw_physics == 7 and not 4 <= cfg.nz <= 100:
            raise ConfigurationRefused(
                "WRF FLG needs 4 to 100 model layers: its three cloud groups "
                "need a layer each and its native scratch bound is nvx=100.")
    if 99 in (ra_lw_physics, ra_sw_physics) and (type(cfg.co2tf) is not int or cfg.co2tf not in (0, 1)):
        raise ConfigurationRefused("co2tf must be 0 (CONRAD) or 1 (CO2O3)")
    if 3 in (ra_lw_physics, ra_sw_physics):
        if not math.isfinite(cfg.cam_abs_freq_s) or cfg.cam_abs_freq_s < 0:
            raise ConfigurationRefused("cam_abs_freq_s must be finite and non-negative")
    if ra_sw_physics == 2 and not 2 <= cfg.nz <= 128:
        raise ConfigurationRefused('GSFC shortwave requires 2 to 128 layers; larger columns exceed its allocated CUDA workspace')
    if cfg.icloud not in (0, 1, 2, 3):
        raise ConfigurationRefused(f"icloud={cfg.icloud}: choose 0, 1, 2 or 3.")
    if not math.isfinite(cfg.swrad_scat) or cfg.swrad_scat < 0.0:
        raise ConfigurationRefused(
            "swrad_scat must be finite and non-negative, got "
            f"{cfg.swrad_scat}.")
    if cfg.wrf_rrtmg_compatibility not in (
            "none", *WRF_RRTMG_COMPATIBILITY_TOKENS):
        raise ConfigurationRefused(
            "wrf_rrtmg_compatibility must be 'none' or one of "
            f"{WRF_RRTMG_COMPATIBILITY_TOKENS}, got "
            f"{cfg.wrf_rrtmg_compatibility!r}.")
    if (cfg.wrf_rrtmg_compatibility in WRF_RRTMG_COMPATIBILITY_TOKENS
            and (ra_lw_physics, ra_sw_physics) != (4, 4)):
        raise ConfigurationRefused(
            f"wrf_rrtmg_compatibility={cfg.wrf_rrtmg_compatibility!r} "
            "requires the resolved 4/4 pair (ra_lw_physics=4, "
            f"ra_sw_physics=4), got ra_lw_physics={ra_lw_physics}, "
            f"ra_sw_physics={ra_sw_physics}: the token is the receipt of "
            "the RRTMG 4/4 radiation engines (it selects the RTE+RRTMGP "
            "snow treatment and names the WRF RRTMG lineage the run "
            "reproduces), nothing on any other radiation pair reads it, and "
            "the run receipt would describe an RRTMG run that did not "
            "happen. Set "
            "wrf_rrtmg_compatibility='none' for this radiation pair (the "
            "registry's non-RRTMG radiation options carry that value, so a "
            "plan that selects one of them resolves it), or select "
            "ra_lw_physics=4 and ra_sw_physics=4.")
    if cfg.ra_rrtmg_variant not in (
            RRTMG_VARIANT_RTE_RRTMGP, RRTMG_VARIANT_LEGACY):
        raise ConfigurationRefused(
            f"ra_rrtmg_variant must be '{RRTMG_VARIANT_RTE_RRTMGP}' or "
            f"'{RRTMG_VARIANT_LEGACY}', got {cfg.ra_rrtmg_variant!r}.")
    if (4 in (ra_lw_physics, ra_sw_physics)
            and cfg.ra_rrtmg_variant == RRTMG_VARIANT_RTE_RRTMGP):
        from woof.cloud_optics_contract import _NO_CLOUD_OPTICS_COUPLING, _CLOUD_OPTICS_REMEDY
        reason = _NO_CLOUD_OPTICS_COUPLING.get(int(cfg.mp_physics))
        if reason is not None:
            raise ValueError(f"mp_physics={cfg.mp_physics}: {reason} {_CLOUD_OPTICS_REMEDY}")
    if (4 in (ra_lw_physics, ra_sw_physics)
            and cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY):
        # ONE MESSAGE PER KNOB.  The composite that stood here printed both
        # values whichever one was off-default, so a user who set o3input=0
        # was told use_mp_re was a problem too.  Each names the operation
        # the RTE+RRTMGP arm substitutes for it, which is the fact that
        # decides whether the remedy is worth taking.
        #
        # A JUDGED cloud-optics exclusion refuses here, before step 0.
        # Breakage prevented: mp_physics=30 (FAST_SBM) on the default
        # RTE+RRTMGP arm passed this door, which the registry's
        # component-conditional refusal does not, and died at its first
        # radiation call in woof.core.rrtmgp.cloud_optics_scheme with a
        # forecast already allocated.  The row is the registry's
        # cloud_optics consumer row, generated from the adapter's own
        # tables, so a scheme that gains its coupling stops refusing with no
        # edit here; it is read from the registry rather than the adapter
        # so a preparation-only install (the RW-WPS wheel) never imports
        # the radiation module.
        if cfg.moist:
            from woof.physics_registry import consumer_row_for_selector

            optics = consumer_row_for_selector(
                "microphysics", "cloud_optics", int(cfg.mp_physics))
            if (isinstance(optics, dict)
                    and optics.get("rte_rrtmgp_coupling") is None
                    and optics.get("rte_rrtmgp_refusal")):
                raise NotImplementedError(
                    f"mp_physics={int(cfg.mp_physics)} has no RTE+RRTMGP "
                    "cloud-optics coupling, deliberately: "
                    f"{optics['rte_rrtmgp_refusal']} "
                    "For namelist import, use --rrtmg-variant "
                    f"{RRTMG_VARIANT_LEGACY}.")
        if cfg.o3input != 2:
            raise ConfigurationRefused(
                f"o3input={cfg.o3input} is not implemented by "
                f"ra_rrtmg_variant='{RRTMG_VARIANT_RTE_RRTMGP}': the "
                "O3DATA profile read it selects exists only inside the "
                "legacy RRTMG wrapper, and this arm SUBSTITUTES its own "
                "packaged ozone climatology interpolated in log-pressure "
                "for it, so honouring the switch here would be a label "
                "over different physics. Leave o3input=2 to run the "
                "packaged climatology, or select "
                f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}', which "
                "implements o3input in (0, 2).")
        if cfg.use_mp_re != 1:
            raise ConfigurationRefused(
                f"use_mp_re={cfg.use_mp_re} is not implemented by "
                f"ra_rrtmg_variant='{RRTMG_VARIANT_RTE_RRTMGP}': "
                "use_mp_re=0 selects the wrapper's temperature-diagnosed "
                "relcalc/reicalc radii, and every cloud-optics row on this "
                "arm consumes the MICROPHYSICS scheme's own effective "
                "radii instead, so the run would not be the diagnosed-radii "
                "configuration the switch names. Leave use_mp_re=1 to use "
                "the scheme's radii, or select "
                f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}', which "
                "implements use_mp_re=0.")
    if (cfg.wrf_rrtmg_compatibility in WRF_RRTMG_SUBSTITUTION_TOKENS
            and cfg.ra_rrtmg_variant != RRTMG_VARIANT_RTE_RRTMGP):
        raise ConfigurationRefused(
            f"wrf_rrtmg_compatibility={cfg.wrf_rrtmg_compatibility!r} "
            "records the RTE+RRTMGP substitution and contradicts "
            f"ra_rrtmg_variant={cfg.ra_rrtmg_variant!r}. The token is not "
            "a label: the RTE+RRTMGP arm reads it to choose its snow "
            "treatment and stamps it into the receipts and the restart "
            "algorithm identities, and a resume refuses the other 4/4 "
            "implementation by name, so a contradicting pair would carry "
            "two solvers in one run's receipts. Set "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_RTE_RRTMGP}' to run the "
            "substitution this token records, or "
            "wrf_rrtmg_compatibility='none' to run RTE+RRTMGP with no WRF "
            f"mapping receipt (or '{WRF_RRTMG_LEGACY}' with "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}' for the legacy "
            "port).")
    if (4 in (ra_lw_physics, ra_sw_physics)
            and cfg.ra_rrtmg_variant == RRTMG_VARIANT_LEGACY):
        # THE READINESS GATE, AT PLAN REVIEW.  It existed and nothing
        # called it (audit R-030): the refusal of record was the
        # RRTMGLegacyRadiation constructor, which is reached from
        # initialize_physics -- pre-integration on a cold start, but on the
        # mid-run twin rebuild (woof/core/streaming.py) it is reached with
        # a forecast already in hand.  Asking here costs seven imports and
        # thirteen file stats, and answers before anything is allocated.
        # A complete pip install from this tree can never trip it; only a
        # stripped or hand-mutilated one can, and that is exactly the
        # install whose forecast must not start.
        from woof.physics_compat import require_rrtmg_legacy_ready

        require_rrtmg_legacy_ready()
    if (cfg.wrf_rrtmg_compatibility == WRF_RRTMG_LEGACY
            and cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY):
        raise ConfigurationRefused(
            f"wrf_rrtmg_compatibility='{WRF_RRTMG_LEGACY}' records the "
            "legacy RRTMG mapping and requires "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}', got "
            f"{cfg.ra_rrtmg_variant!r}. The legacy adapter stamps this "
            "token itself and the two 4/4 implementations carry distinct "
            "restart algorithm identities, so the pair as written would "
            "label an RTE+RRTMGP run as the legacy port and make its "
            "checkpoints refuse the run that wrote them. Set "
            f"ra_rrtmg_variant='{RRTMG_VARIANT_LEGACY}' to run the legacy "
            "port, or wrf_rrtmg_compatibility='none' to run RTE+RRTMGP "
            "without a WRF mapping receipt.")
    if 4 in (ra_lw_physics, ra_sw_physics) and cfg.icloud not in (1, 2, 3):
        # NOT A PHYSICAL INCOMPATIBILITY: icloud is WRF's clear-sky coupling
        # switch, woof honours it on the 1/1 pair, and the legacy 4/4
        # preparation carries its arms.  What is missing on the 4/4 pair is
        # the anchor: every recorded RRTMG oracle case is an icloud=1 (or,
        # since golden/ports-rad, icloud=2 cal_cldfra2) case, so admitting 0
        # here would ship a clear-sky arm with nothing bitwise behind it.
        # The message says which kind of gap it is (2.9.0 round 2 restored
        # this wording after the icloud=2 port had shortened it).
        raise ConfigurationRefused(
            f"icloud={cfg.icloud} is not admitted with the 4/4 radiation "
            "pair. Neither 4/4 implementation has wired RRTMGP clear-sky "
            "cloud optics or a recorded clear-sky solve, so a run asking for "
            "clear-sky coupling would get the cloudy solve and a receipt "
            "saying icloud=0: the run would not be the configuration it "
            "names. This is an evidence gap rather than a physical one "
            "(WRF's own clear-sky arms are transcribed in the legacy "
            "preparation), and closing it means wiring the switch through "
            "both arms AND recording one longwave and one shortwave oracle "
            "case at icloud=0. Run clear-sky coupling on the 1/1 pair "
            "(ra_lw_physics=1 WRF RRTM with ra_sw_physics=1 Dudhia), which "
            "honours icloud=0 end to end, or use icloud=1 or 2 on the 4/4 "
            "pair.")
    if type(cfg.shcu_physics) is not int or cfg.shcu_physics not in (0, 1, 2, 3, 4, 5):
        raise ConfigurationRefused("shcu_physics must be 0 (off), 1 (G3 native no-op), 2 (UW), 3 (GRIMS), 4 (NSCV), or 5 (Deng)")
    from woof.shallow_cumulus import shallow_refusal
    why = shallow_refusal(cfg.shcu_physics, cfg.bl_pbl_physics,
                          cfg.bl_mynn_edmf, cfg.cu_physics,
                          icloud=cfg.icloud, icloud_bl=cfg.icloud_bl)
    if why:
        raise ConfigurationRefused(why)
    if cfg.shcu_physics in (3, 4, 5) and not cfg.moist:
        raise ConfigurationRefused("shallow convection requires a water-vapour carrier")
    if cfg.shcu_physics == 5 and not cfg.moist_cq:
        raise ConfigurationRefused("Deng requires QC and QR condensate carriers")
    if cfg.shcu_physics == 2 and not (cfg.moist and cfg.moist_cq):
        raise ConfigurationRefused("UW shallow convection requires moist condensate carriers")
    if cfg.shcu_physics == 2 and cfg.mp_physics not in (2,4,5,6,8,9,10,11,16,18,24,28,30,50,106,900):
        raise ConfigurationRefused("WRF UW shallow convection requires the QI carrier (module_shallowcu_driver.F:519-522)")
    if cfg.cu_physics == 7:
        from woof.core.camzm_contract import pbl_arm_refusal
        why = pbl_arm_refusal(cfg.bl_pbl_physics, cfg.sf_sfclay_physics)
        if why: raise ConfigurationRefused(why)
    for name, default in (("accum_mode",1.e9),("aitken_mode",3.e8),("coarse_mode",2.e5)):
        value=getattr(cfg,name)
        if not math.isfinite(value) or value < 0:
            raise ConfigurationRefused(f"{name} must be finite and nonnegative")
        if cfg.mp_physics != 11 and value != default:
            raise ConfigurationRefused(f"{name} is consumed only by CAMMGMP (mp_physics=11)")
    if cfg.cu_physics == 11 and cfg.km_opt == 5:
        raise ConfigurationRefused("cu_physics=11 cannot use km_opt=5: WRF module_check_a_mundo.F:1353-1363 forbids MSKF with 3D TKE because that closure is incompatible with its subgrid convection")
    if type(cfg.kf_edrates) is not int or cfg.kf_edrates not in (0, 1):
        raise ConfigurationRefused("kf_edrates must be 0 or 1")
    if cfg.kf_edrates and cfg.cu_physics not in (11, 99):
        raise ConfigurationRefused("kf_edrates=1 requires rate-profile carriers (cu_physics=11 or cu_physics=99)")
    if not math.isfinite(cfg.aercu_fct) or cfg.aercu_fct < 0:
        raise ConfigurationRefused("aercu_fct must be finite and nonnegative: a negative multiplier would supply negative aerosol mass to activation")
    if cfg.cu_physics not in CU_SCHEMES:
        raise ConfigurationRefused(
            f"cu_physics must be one of {CU_SCHEMES} -- 0 (off), "
            f"1 (Kain-Fritsch), 2 (Betts-Miller-Janjic), 3 (Grell-Freitas), 5 (Grell 3D), 6 (original cu6), 10 (KF-CuP), 14 (KIAPS SAS), "
            f"16 (New Tiedtke), 93 (Grell-Devenyi), 94 (GFS SAS), 95 (Old GFS SAS), 96 (New SAS) or 99 (previous KF), got {cfg.cu_physics}."
            + wrf_unported_refusal_tail("cu_physics", cfg.cu_physics)
        )
    if cfg.use_theta_m not in (0, 1):
        raise ConfigurationRefused("use_theta_m must be 0 (dry theta) or 1 (moist theta)")
    if cfg.hypsometric_opt not in (1, 2):
        raise ConfigurationRefused(
            "hypsometric_opt must be 1 (d(phi)/d(eta) hydrostatic "
            "inversion, the frozen woof form) or 2 (WRF log-pressure "
            "form, calc_p_rho_phi), got "
            f"{cfg.hypsometric_opt}."
        )
    if cfg.mp_physics in _P3_UNPORTED_VARIANTS:
        # Refuse the sibling P3 options BY NAME rather than letting them
        # fall into the generic "unknown mp_physics" message: each is a real
        # WRF v4.6.1 package (Registry.EM_COMMON:3040-3041) that woof's
        # P3 port (50, 51) does not cover, and a user who typed one
        # is owed the reason, not a claim that the number is meaningless.
        # The menu follows the reason: these are out-of-schema VALUES, and
        # a value refusal recites the menu here the same way as below.
        raise ConfigurationRefused(unported_p3_variant_refusal(cfg.mp_physics))
    if type(cfg.aercu_opt) is not int or cfg.aercu_opt not in (0, 1, 2):
        raise ConfigurationRefused("aercu_opt must be 0, 1 or 2")
    if cfg.aercu_opt > 0:
        from woof.core.morrison_aero import refuse_aercu_opt
        refuse_aercu_opt(cfg.aercu_opt, cu_physics=cfg.cu_physics, mp_physics=cfg.mp_physics)
    # RCON-SPEED-01: retire when a qualified handoff clears this measured gap.
    if cfg.mp_physics == 29:
        raise ConfigurationRefused(
            "mp_physics=29 (RCON) is ported but its release speed proof is open: "
            "the whole-forecast microphysics stage costs 1.885431x MP28; "
            "the retained native 1.053748x profile does not explain the GPU excess. "
            "A qualified speed handoff is required before forecast admission; "
            "until then select another microphysics scheme, such as mp_physics=28 "
            "(the Thompson aerosol-aware scheme it is timed against)."
        )
    if cfg.mp_physics not in MP_PHYSICS_ACCEPTED:
        # Known unported options carry a source-backed reason before the
        # general selector menu. The WDM family is admitted above.
        named = unported_mp_scheme_refusal(cfg.mp_physics)
        if named is not None:
            raise ConfigurationRefused(named)
        raise ConfigurationRefused(f"{_MP_PHYSICS_SCHEMA_MENU}, got {cfg.mp_physics}.")
    validate_milbrandt2_options(cfg)
    if cfg.mp_physics not in THOMPSON_AEROSOL_FAMILY:
        # The mp=28 aerosol-source pair is scheme-scoped
        # (woof.core.model.SCHEME_SCOPED_RUN_FIELDS), and this refusal is
        # what makes that scoping provably lossless: under any other scheme
        # the two can hold ONLY their defaults, so dropping them from the
        # restart identity discards no information.  The concrete breakage:
        # they landed unscoped at the 2.5.8 integration and moved every
        # experiment fingerprint, which would have made every existing
        # checkpoint refuse to resume for fields only mp=28 reads.
        for _mp28_key, _mp28_default in (("mp28_aerosol_source", "auto"),
                                         ("wif_climatology_path", "")):
            if getattr(cfg, _mp28_key) != _mp28_default:
                raise ConfigurationRefused(
                    f"{_mp28_key} selects the Thompson aerosol initial state "
                    "and requires " + " or ".join(f"mp_physics={_m}" for _m in sorted(THOMPSON_AEROSOL_FAMILY))
                    + f" (got mp_physics="
                    f"{cfg.mp_physics}); no other scheme reads it, and "
                    "woof refuses a stray value instead of silently "
                    "dropping it."
                )
    validate_aerosol_source_options(cfg)
    if cfg.mp_physics != 0 and not cfg.moist:
        raise ConfigurationRefused(
            f"mp_physics={cfg.mp_physics} requires moist=true; the dry "
            "state does not allocate water or hydrometeor fields."
        )
    if cfg.nest_microphysics_transition not in (
            "same-scheme-only", "mp8-to-mp18-mass-diagnosed-v1",
            "mp-edge-mass-diagnosed-v1"):
        raise ConfigurationRefused(
            "nest_microphysics_transition must be 'same-scheme-only', "
            "'mp8-to-mp18-mass-diagnosed-v1', or "
            "'mp-edge-mass-diagnosed-v1', got "
            f"{cfg.nest_microphysics_transition!r}."
        )
    if (not cfg.nested
            and cfg.nest_microphysics_transition != "same-scheme-only"):
        raise ConfigurationRefused(
            "nest_microphysics_transition may only be selected on a nested "
            "child domain"
        )
    if cfg.morr_rimed_ice not in (0, 1):
        raise ConfigurationRefused(
            "morr_rimed_ice must be 0 (graupel) or 1 (hail, the WRF "
            f"default), got {cfg.morr_rimed_ice}."
        )
    if cfg.ntu_ccnty not in (1, 2, 3, 4):
        raise ValueError(
            "ntu_ccnty must be 1 (marine), 2 (clean continental, the WRF "
            "default), 3 (average continental) or 4 (urban): NTU's "
            "AERO_CONST has no aerosol spectrum for "
            f"{cfg.ntu_ccnty}.")
    if cfg.wsm6_hail_opt not in (0, 1):
        raise ConfigurationRefused(
            "wsm6_hail_opt must be 0 (graupel, the WRF WSM6 default) or "
            f"1 (hail), got {cfg.wsm6_hail_opt}."
        )
    for option in ("tempo_ml_for_bl_nc", "tempo_surface_emission_before"):
        if type(getattr(cfg, option)) is not bool:
            raise ValueError(f"{option} must be logical; interpreting a string would select the wrong TEMPO coupling")
        if getattr(cfg, option) and cfg.mp_physics != 900:
            raise ValueError(f"{option} selects TEMPO behavior but mp_physics={cfg.mp_physics} does not execute it")
    for _correct_key in _correct_mode.CORRECTION_KEYS:
        # A string "false" from a hand-built mapping is truthy in Python and
        # would switch the correction ON while the file says off.
        if not isinstance(getattr(cfg, _correct_key), bool):
            raise ValueError(
                f"{_correct_key} must be a boolean (correct-mode key, "
                f"woof/correct_mode.py), got "
                f"{getattr(cfg, _correct_key)!r}.")
        if _correct_mode.STRICT and getattr(cfg, _correct_key):
            # Breakage: a strict 0 ULP WRF proof that ran corrected
            # arithmetic under its strict label.  Strict mode forces exact.
            raise ValueError(
                f"{_correct_key} = true in a strict WRF process "
                f"({_correct_mode.STRICT_FORCED_BY}): strict mode is the 0 "
                "ULP proof of the matched arithmetic and forces physics.mode "
                "= 'exact'.  Set it false, or run without GPUWM_WRF_EXACT.")
    _NSSL_SELECTOR_DEFAULTS = {
        "nssl_2moment_on": -1, "nssl_hail_on": -1, "nssl_ccn_on": -1,
        "nssl_density_on": -1, "nssl_3moment": 0, "nssl_cccn": 0.5e9,
        "nssl_alphah": 0.0, "nssl_alphahl": 1.0,
        "nssl_icdx": 6, "nssl_icdxhl": 6,
    }
    if cfg.mp_physics == 18:
        # Resolve and gate the variant mode exactly once at validation, so
        # a rejected combination can never reach state allocation or the
        # scheme.  resolve applies WRF's module_check_a_mundo.F:3423-3465
        # consistency pass; require_ported refuses combinations without a
        # ported numerical path (3-moment, two-moment fixed-density, and
        # WRF's undefined inactive-pointer pairings).
        from woof.core.nssl2_contract import (
            require_ported_nssl2_mode,
            resolve_nssl2_mode_for_config,
        )
        try:
            require_ported_nssl2_mode(resolve_nssl2_mode_for_config(cfg))
            from woof.core.nssl2_contract import nssl2_cccn_concentration
            nssl2_cccn_concentration(cfg.nssl_cccn)
            from woof.core.nssl2_shape import resolved_shapes
            resolved_shapes(cfg.nssl_alphah, cfg.nssl_alphahl, cfg.nssl_icdx, cfg.nssl_icdxhl)
        except (TypeError, ValueError) as exc:
            raise ConfigurationRefused(f"NSSL variant selectors: {exc}") from exc
    else:
        for _nssl_key, _nssl_default in _NSSL_SELECTOR_DEFAULTS.items():
            if getattr(cfg, _nssl_key) != _nssl_default:
                raise ConfigurationRefused(
                    f"{_nssl_key} is an NSSL selector and requires "
                    f"mp_physics=18 (got mp_physics={cfg.mp_physics}); "
                    "WRF zeroes stray NSSL flags under other schemes "
                    "(module_check_a_mundo.F:3423-3430), woof refuses "
                    "them instead of silently dropping them."
                )
    if cfg.wdm6_hail_opt not in (0, 1):
        raise ConfigurationRefused(
            "wdm6_hail_opt must be 0 (graupel, the WRF WDM6 default) or "
            f"1 (hail), got {cfg.wdm6_hail_opt}."
        )
    if not math.isfinite(cfg.wdm6_ccn_conc) or cfg.wdm6_ccn_conc <= 0.0:
        raise ConfigurationRefused(
            "wdm6_ccn_conc must be a finite positive CCN number "
            "concentration in # m-3 (WRF Registry default 1.0e8), got "
            f"{cfg.wdm6_ccn_conc}."
        )
    if cfg.mp_physics in WDM_SCHEMES and not (1.0e8 <= cfg.wdm6_ccn_conc <= 2.0e10):
        # module_mp_wdm6.F:585 clamps the CCN array to [1e8, 2e10] on every
        # entry to wdm62D, so a value outside the window is not the initial
        # condition the operator asked for -- the very first minor loop
        # discards it.  Refusing beats silently running a different run.
        raise ConfigurationRefused(
            f"wdm6_ccn_conc={cfg.wdm6_ccn_conc} lies outside the WDM family's own "
            "[1e8, 2e10] # m-3 clamp (module_mp_wdm6.F:585), so the scheme "
            "would overwrite it on the first minor time step; choose a "
            "value inside the window."
        )
    if cfg.mp_physics == 14 and cfg.wdm6_hail_opt != 0:
        raise ValueError("wdm6_hail_opt retunes a graupel category; WDM5 carries no graupel, so this setting would be ignored. Use its default zero.")
    if cfg.mp_physics in WDM_HAIL_SCHEMES and cfg.wdm6_hail_opt != 0:
        # Breakage prevented: a WDM7 run that silently ignores a hail_opt
        # the operator set.  WRF reads hail_opt for WSM6/WDM6 only
        # (module_physics_init.F:4583-4584); wdm7init takes no such argument
        # (:4586) because WDM7 carries hail as its own category and its
        # graupel is WDM6's hail_opt = 0 graupel.
        raise ValueError(
            "wdm6_hail_opt retunes WDM6's graupel into hail and has no "
            "meaning under mp_physics=26: WDM7 carries hail as a separate "
            "category and its graupel is fixed (module_physics_init.F:4586 "
            "calls wdm7init without hail_opt).  Remove wdm6_hail_opt or set "
            "it to 0."
        )
    if cfg.mp_physics == 27 and cfg.wdm6_hail_opt != 0:
        raise ConfigurationRefused("UDM carries separate hail and reads no hail_opt; keep wdm6_hail_opt=0 (module_mp_udm.F:262-277).")
    if (cfg.mp_physics == 27 and cfg.ra_rrtmg_variant != RRTMG_VARIANT_LEGACY
            and (cfg.ra_lw_physics == 4 or cfg.ra_sw_physics == 4)):
        from woof.core.udm import RADIUS_REASON
        raise ConfigurationRefused(RADIUS_REASON + " Select legacy RRTMG for its diagnosed radii.")
    if cfg.mp_physics not in WDM_SCHEMES and cfg.mp_physics != 27:
        # Off-scheme WDM6 knobs are REFUSED, not dropped, on the NSSL
        # precedent (the nssl_* selectors under mp != 18): nothing outside
        # the WDM family path reads either field, so a set value is a request
        # woof would silently ignore.  The refusal is also what makes
        # woof.core.model.SCHEME_SCOPED_RUN_FIELDS provably lossless --
        # under any other scheme these two can hold ONLY their defaults, so
        # dropping them from the restart identity discards no information.
        for _wdm6_key, _wdm6_default in (("wdm6_hail_opt", 0),
                                         ("wdm6_ccn_conc", 1.0e8)):
            if getattr(cfg, _wdm6_key) != _wdm6_default:
                raise ConfigurationRefused(
                    f"{_wdm6_key} is a WDM6 selector and requires "
                    + " or ".join(f"mp_physics={_m}" for _m in sorted({*WDM_SCHEMES, 27})) + " "
                    f"(got mp_physics={cfg.mp_physics}); per-variant restrictions still apply. "
                    "WRF reads hail_opt/ccn_conc only inside the schemes "
                    "that declare them, and woof refuses a stray value "
                    "instead of silently dropping it."
                )
    # Goddard GCE switches.  WRF stops the run on any other value
    # (module_mp_gsfcgce.F:249-262, "ihail has to be either 1 or 0",
    # "gsfcgce_2ice in namelist.input has to be 0, 1, 2, or 3"); woof
    # refuses it before allocation instead of after the first call.
    if cfg.gsfcgce_hail not in (0, 1):
        raise ValueError(
            "gsfcgce_hail must be 0 (graupel, the WRF default) or 1 (hail), "
            f"got {cfg.gsfcgce_hail}; WRF's module_mp_gsfcgce.F stops on "
            "any other value.")
    if cfg.gsfcgce_2ice not in (0, 1, 2, 3):
        raise ValueError(
            "gsfcgce_2ice must be 0 (ice, snow and graupel/hail, the WRF "
            "default), 1 (ice and snow), 2 (ice and graupel) or 3 (warm "
            f"rain only), got {cfg.gsfcgce_2ice}; WRF's module_mp_gsfcgce.F "
            "stops on any other value.")
    from woof.core.mp_column_schemes import MP_COLUMN_SCHEMES as _TABLE_ROWS
    for _table_mp, _table_row in _TABLE_ROWS.items():
        if cfg.mp_physics == _table_mp:
            continue
        for _switch in _table_row.switches:
            if getattr(cfg, _switch) != 0:
                # Off-scheme switches are REFUSED, not dropped (the WDM6
                # and NSSL precedent): nothing outside the row's scheme
                # reads them, so a set value is a request woof would
                # silently ignore, and the refusal is what makes the
                # restart identity's scheme scoping lossless.
                raise ValueError(
                    f"{_switch} is a {_table_row.name} selector and requires "
                    f"mp_physics={_table_mp} (got mp_physics="
                    f"{cfg.mp_physics}); WRF reads it only inside that "
                    "scheme, and woof refuses a stray value instead of "
                    "silently dropping it.")
    if cfg.p3_backend not in P3_BACKENDS:
        raise ConfigurationRefused(
            f"p3_backend={cfg.p3_backend!r} is not one of "
            f"{sorted(P3_BACKENDS)}: 'cuda' is the shipping device arm "
            "(woof/core/kernels/p3.cu, one kernel per step of the "
            "authority), 'fused' is the three-launch composition proven "
            "byte-identical to it, and 'reference' is the CPU float32 "
            "transcription kept for verification.  A fourth arm has to be "
            "byte-compared against 'cuda' before it can be named here.")
    if cfg.mp_physics == 52 and cfg.p3_backend == "reference":
        raise ValueError("P3 mp_physics=52 needs cuda or fused: the CPU reference "
                         "omits category-2 collection, merge and sedimentation")
    _p3_row = p3_variant(cfg.mp_physics)
    if (_p3_row is not None and _p3_row.three_moment_ice
            and cfg.p3_backend == "reference"):
        # The CPU float32 transcription (woof/core/p3.py p3_main) carries
        # the two-moment-ice configuration only; the three-moment branches
        # (Z, the mu_i closure, the 3momI table) are in the device kernels.
        # Running it for mp=53 would integrate two-moment ice under a
        # three-moment receipt and never touch qzi.
        raise ValueError(
            f"p3_backend='reference' cannot run mp_physics={cfg.mp_physics} "
            "(P3 three-moment ice): the CPU reference transcription carries "
            "two-moment ice only, so it would integrate a different scheme "
            "and leave qzi untouched. Use p3_backend='cuda' (the default) or "
            "'fused'.")
    if _p3_row is None and cfg.p3_backend != "cuda":
        # The P3 backend selector is scheme-scoped
        # (woof.core.model.SCHEME_SCOPED_RUN_FIELDS), and this refusal is
        # what makes that scoping provably lossless: under any other scheme
        # it can hold ONLY its default, so dropping it from the restart
        # identity discards no information.  The concrete breakage it
        # prevents is the one the mp=28 aerosol pair caused for real at the
        # 2.5.8 integration -- an unscoped field moved every experiment
        # fingerprint, so every existing checkpoint would have refused to
        # resume for a knob only P3 reads.
        raise ConfigurationRefused(
            f"p3_backend={cfg.p3_backend!r} selects the P3 implementation "
            f"and requires mp_physics=50 or mp_physics=51 or mp_physics=52 or mp_physics=53 (got mp_physics={cfg.mp_physics}); "
            "no other scheme reads it, and woof refuses a stray value "
            "instead of silently dropping it.")
    for name in ("history_interval2_s", "history_interval_change_s"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ConfigurationRefused(f"{name} must be finite and non-negative")
    if cfg.history_interval_change_s and not cfg.history_interval2_s:
        raise ConfigurationRefused("history_interval_change_s needs history_interval2_s")
    if cfg.history_interval2_s and cfg.history_interval2_s <= cfg.output_interval_s:
        raise ConfigurationRefused("history_interval2_s must exceed output_interval_s, as in the fork history alarm")
    _validate_adaptive_time_step(cfg)
    if cfg.no_mp_heating not in (0, 1):
        raise ConfigurationRefused(
            "no_mp_heating must be 0 (microphysics latent heating on, the "
            f"WRF default) or 1 (heating off), got {cfg.no_mp_heating}."
        )
    for key in ("do_avgflx_em", "do_avgflx_cugd"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) not in (0, 1):
            raise ConfigurationRefused(f"{key} must be the integer 0 or 1")
    if cfg.do_avgflx_cugd:
        from woof.avgflx_contract import CUGD_REFUSAL
        raise ConfigurationRefused(CUGD_REFUSAL)
    for name in ("mean_diag", "diurnal_diag", "solar_diagnostics"):
        if type(getattr(cfg,name)) is not int or getattr(cfg,name) not in (0,1):
            raise ConfigurationRefused(name+" must be 0 or 1")
    if cfg.mean_diag:
        if cfg.mean_freq not in (1,2,3,4,5) or type(cfg.mean_interval) is not int or cfg.mean_interval<=0:
            raise ConfigurationRefused("RASM mean_diag needs a positive native mean interval and frequency 1..5")
        seconds=cfg.mean_interval*{1:1,2:60,3:3600,4:86400,5:86400}[cfg.mean_freq]
        if seconds/cfg.dt < .5:
            raise ConfigurationRefused("RASM mean interval rounds to zero native steps")
    if type(cfg.diag_print) is not int or cfg.diag_print not in (0,1,2):
        raise ValueError("diag_print must be 0, 1 or 2")
    if type(cfg.output_diagnostics) is not int or cfg.output_diagnostics not in (0, 1):
        raise ValueError("output_diagnostics must be 0 or 1")
    if type(cfg.auxhist3_interval) is not int or cfg.auxhist3_interval < 0:
        raise ValueError("auxhist3_interval must be nonnegative whole minutes")
    if cfg.output_diagnostics and cfg.auxhist3_interval and cfg.auxhist3_interval * 60. / cfg.dt < .5:
        raise ValueError("auxhist3_interval rounds to zero steps; native CLWRF would take MOD by zero")
    for key in ("diag_int", "wind_int"):
        if type(getattr(cfg, key)) is not int or getattr(cfg, key) <= 0:
            raise ValueError(f"{key} must be positive whole minutes")
    if cfg.gsd_diagnostics not in (0, 1):
        raise ValueError("gsd_diagnostics must be 0 or 1; research package 2 is not carried")
    if type(cfg.reset_interval1) is not int or cfg.reset_interval1 <= 0:
        raise ValueError("reset_interval1 must be positive whole minutes")
    if cfg.gsd_diagnostics and cfg.mp_physics not in (0, 8, 28):
        raise ValueError("gsd_diagnostics hail PSD supports mp_physics 8/28; other fork PSD tables are not carried")
    for key in ("prec_acc_dt", "prec_acc_dt1"):
        value = getattr(cfg, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative minutes")
    for key in ('bucket_mm','bucket_j'):
        if not math.isfinite(getattr(cfg,key)):
            raise ValueError(key+' must be finite')
    if cfg.bucket_mm>0 and cfg.bucket_j>0:
        if any(x not in (0,4) for x in radiation_scheme_ids(cfg)):
            raise ValueError('native energy bucketing needs accumulated all-sky and clear-sky top/bottom flux carriers; these are supplied by radiation selectors 4 only')
    from woof.afwa_config import validate as validate_afwa
    validate_afwa(cfg)
    from woof.core.physics_tendency_output import validate as validate_tendencies
    validate_tendencies(cfg)
    from woof.level_output_config import validate as validate_level_output
    validate_level_output(cfg)
    if type(cfg.uh_layers) is not int or cfg.uh_layers not in (0, 1):
        raise ValueError("uh_layers must be 0 or 1")
    if cfg.uh_layers == 1 and cfg.nwp_diagnostics != 1:
        # Breakage this prevents: the maxima are folded by the step's
        # nwp_diagnostics = 1 helicity pass; without it they would be written
        # as zeros every hour, a quiet map that says no rotation anywhere.
        raise ValueError("uh_layers = 1 needs nwp_diagnostics = 1 (the step "
                         "that folds UP_HELI_MAX also folds the 0-1/0-2/0-3 km maxima)")
    if cfg.nwp_diagnostics not in (0, 1):
        raise ConfigurationRefused(
            "nwp_diagnostics must be 0 (off, the WRF default) or 1 "
            "(native per-step NWP diagnostics), got "
            f"{cfg.nwp_diagnostics}."
        )
    if cfg.nwp_diagnostics and not cfg.gsd_diagnostics:
        if cfg.output_interval_s < 60 or cfg.output_interval_s % 60 or cfg.dt > cfg.output_interval_s:
            raise ConfigurationRefused(
                "native NWP idump requires a positive whole-minute history interval at least dt; "
                "a zero idump would divide the native window mean by zero")
    # GSD owns the fork's interval windows when both switches are selected;
    # nwp_diagnostics.active() disables the stock NWP writer in that case.
    if not math.isfinite(cfg.mp_tend_lim) or cfg.mp_tend_lim <= 0.0:
        raise ConfigurationRefused(
            "mp_tend_lim must be a finite positive heating-rate clamp in "
            f"K/s (WRF Registry default 10.0), got {cfg.mp_tend_lim}."
        )
    for name in ("radt", "bldt", "radt_minutes", "cudt_minutes"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0.0:
            raise ConfigurationRefused(
                f"{name} must be a finite non-negative interval in minutes "
                f"(0 means every model step), got {value}."
            )
    if not math.isfinite(cfg.restart_interval_s) or cfg.restart_interval_s < 0.0:
        raise ConfigurationRefused(
            "restart_interval_s must be a finite non-negative interval in "
            f"seconds (0 disables restart writing), got "
            f"{cfg.restart_interval_s}."
        )
    _validate_dynamics_coefficients(cfg)
    validate_km_opt(cfg)
    for name in ("khdif", "kvdif"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0.0:
            raise ConfigurationRefused(
                f"{name} must be a finite non-negative constant-K "
                f"diffusivity, got {value}."
            )
    # Under diff_opt = 1 WRF's constant-K vertical diffusion
    # (module_em.F vertical_diffusion_u/_v/_mp) reads kvdif whatever
    # km_opt is, and this engine applies khdif/kvdif only under km_opt = 1
    # (woof/core/diffusion.py), so 2/3/4 beside a nonzero constant would
    # silently drop a term WRF runs.  Under diff_opt = 2 WRF ignores both
    # outside km_opt = 1, so WRF's own fire namelists (km_opt = 2 with
    # khdif/kvdif set) import as written (development branch).
    if (cfg.diff_opt == 1 and cfg.km_opt in (2, 3, 4)
            and (cfg.khdif > 0.0 or cfg.kvdif > 0.0)):
        raise ConfigurationRefused(
            f"diff_opt=1 with km_opt={cfg.km_opt}: khdif/kvdif are constant-K "
            "controls for km_opt=1 and cannot also be active; WRF's "
            "diff_opt=1 vertical diffusion would read kvdif, which this "
            "engine does not run outside km_opt=1.")
    if not math.isfinite(cfg.c_k) or cfg.c_k <= 0.0:
        raise ConfigurationRefused(
            f"c_k must be a finite positive TKE-closure constant, got "
            f"{cfg.c_k}."
        )
    if not math.isfinite(cfg.tke_upper_bound) or cfg.tke_upper_bound <= 0.0:
        raise ConfigurationRefused(
            f"tke_upper_bound must be a finite positive TKE ceiling in "
            f"m2 s-2 (WRF Registry default 1000.0), got "
            f"{cfg.tke_upper_bound}."
        )
    if cfg.tke_budget not in (0, 1):
        raise ConfigurationRefused(
            f"tke_budget must be 0 (off) or 1 (per-step term-by-term TKE "
            f"budget accumulation), got {cfg.tke_budget}."
        )
    if cfg.tke_budget == 1 and cfg.km_opt != 2 and cfg.bl_pbl_physics != 5:
        raise ConfigurationRefused(
            "tke_budget=1 requires km_opt=2 or bl_pbl_physics=5: the implemented budget "
            "accumulates that closure's explicit source terms. It does "
            f"not account for km_opt={cfg.km_opt}, including SMS-3DTKE's "
            "implicit transport and dissipation."
        )
    if cfg.mix_isotropic not in (0, 1):
        raise ConfigurationRefused(
            f"mix_isotropic must be 0 (anisotropic mixing lengths) or 1 "
            f"(isotropic (dx*dy*dz)^(1/3)), got {cfg.mix_isotropic}."
        )
    if not math.isfinite(cfg.mix_upper_bound) or cfg.mix_upper_bound <= 0.0:
        raise ConfigurationRefused(
            f"mix_upper_bound must be a finite positive non-dimensional K "
            f"cap, got {cfg.mix_upper_bound}."
        )
    if not math.isfinite(cfg.tke_heat_flux):
        raise ConfigurationRefused(
            f"tke_heat_flux must be a finite kinematic heat flux "
            f"(K m s-1), got {cfg.tke_heat_flux}."
        )
    if not math.isfinite(cfg.tke_drag_coefficient) \
            or cfg.tke_drag_coefficient < 0.0:
        raise ConfigurationRefused(
            f"tke_drag_coefficient must be a finite non-negative drag "
            f"coefficient, got {cfg.tke_drag_coefficient}."
        )
    if cfg.diff_6th_slopeopt not in (0, 1):
        raise ConfigurationRefused(
            f"diff_6th_slopeopt must be 0 (untapered) or 1 (WRF terrain-"
            f"slope taper), got {cfg.diff_6th_slopeopt}."
        )
    if not math.isfinite(cfg.diff_6th_thresh) or cfg.diff_6th_thresh <= 0.0:
        raise ConfigurationRefused(
            f"diff_6th_thresh must be a positive slope (m/m), got "
            f"{cfg.diff_6th_thresh}."
        )
    if isinstance(cfg.h_sca_adv_order, bool) or cfg.h_sca_adv_order not in (0, 2, 3, 4, 5, 6):
        raise ConfigurationRefused(
            f"h_sca_adv_order must be 0, 2, 3, 4, 5 or 6, got "
            f"{cfg.h_sca_adv_order}."
        )
    if cfg.h_sca_adv_order == 0 and (cfg.moist_adv_opt != 0 or cfg.scalar_adv_opt != 0):
        raise ConfigurationRefused(
            "h_sca_adv_order=0 requires ordinary moisture and scalar advection "
            "(moist_adv_opt=0, scalar_adv_opt=0): "
            "WRF 4.7.1 module_advect_em.F:7256 refuses order 0 in advect_scalar_pd.")
    # Open-boundary rhs_ph (h_sca_adv_order 5 with radiative edges) is the
    # compiled slow_geopotential_open kernel since the SFIRE port; the former
    # refusal named exactly that missing boundary-row upwinding.
    why = advection_order_refusal(
        v_sca_adv_order=cfg.v_sca_adv_order,
        v_mom_adv_order=cfg.v_mom_adv_order,
        h_mom_adv_order=cfg.h_mom_adv_order)
    if why is not None:
        raise ConfigurationRefused(why)
    if cfg.spec_zone < 1 or cfg.relax_zone < 2:
        raise ConfigurationRefused("spec_zone must be >= 1 and relax_zone must be >= 2")
    if cfg.spec_bdy_width < cfg.spec_zone + cfg.relax_zone:
        raise ConfigurationRefused(
            "spec_bdy_width must cover spec_zone + relax_zone")
    # Unsupported lateral-boundary combinations fail loudly here and again
    # in dycore.step (which also catches nonzero terrain heights and directly
    # constructed RunConfigs): their stencils wrap unconditionally across
    # a physical boundary.
    if cfg.zadvect_implicit and (cfg.open_x or cfg.open_y):
        raise UnsupportedConfiguration(
            "zadvect_implicit = 1 with radiative open boundaries is not "
            "wired: the implicit u (v) column solve on an open boundary "
            "face averages the implicit eta mass flux of the column on "
            "each side of the face, and outside an open boundary there is "
            "no column (WRF reads its halo there), so the solve would run "
            "on an invented flux.  Periodic, specified and nested "
            "boundaries are wired."
        )
    if cfg.diff_6th_opt == 1 and cfg.moist:
        raise ConfigurationRefused(
            "diff_6th_opt=1 (non-monotonic) with moist=true risks negative "
            "moisture: the unlimited 6th-order fluxes bypass the "
            "positive-definite transport limiter; use the monotonic "
            "diff_6th_opt=2."
        )
    if cfg.diff_6th_factor2 is not None:
        if cfg.diff_6th_form != "noaa_wrf39":
            raise ConfigurationRefused(
                f"diff_6th_factor2={cfg.diff_6th_factor2!r} is read only by "
                "the NOAA WRFV3.9 fork's filter (diff_6th_form = "
                "'noaa_wrf39'); WRF v4.6.1 has no second factor, so under "
                f"diff_6th_form = {cfg.diff_6th_form!r} the value would be "
                "silently ignored.")
        if (isinstance(cfg.diff_6th_factor2, bool)
                or not isinstance(cfg.diff_6th_factor2, (int, float))
                or not math.isfinite(cfg.diff_6th_factor2)
                or cfg.diff_6th_factor2 < 0.0):
            raise ConfigurationRefused(
                "diff_6th_factor2 must be a finite non-negative filter "
                f"factor (fork Registry default 0.04), got "
                f"{cfg.diff_6th_factor2!r}.")
    if (isinstance(cfg.mp_zero_out_thresh, bool)
            or not isinstance(cfg.mp_zero_out_thresh, (int, float))
            or not math.isfinite(cfg.mp_zero_out_thresh)):
        raise ConfigurationRefused(
            "mp_zero_out_thresh must be a finite mixing ratio in kg/kg "
            f"(WRF Registry default 1e-8), got {cfg.mp_zero_out_thresh!r}.")
    if not isinstance(cfg.moist_mix6_off, bool):
        raise ConfigurationRefused(
            "moist_mix6_off must be a boolean (WRF v4.6.1 declares it "
            "logical, Registry.EM_COMMON:2889), got "
            f"{cfg.moist_mix6_off!r}."
        )
    validate_resolved_physics_vertical_levels(cfg)
    # LES-nest inflow seeding (P3).  The three companion keys are
    # schema-validated even while the mechanism is off (fail-loud, the
    # D3 convention): a typo'd faces mode or a negative scale is a dead
    # configuration either way, and it should say so before a run.
    if type(cfg.inflow_perturbation) is not bool:
        raise ConfigurationRefused(
            f"inflow_perturbation must be boolean, got "
            f"{cfg.inflow_perturbation!r}.")
    if cfg.inflow_perturbation and not cfg.nested:
        raise ConfigurationRefused(
            "inflow_perturbation is a nest-boundary mechanism: it "
            "perturbs the rolling parent-forced boundary tables a "
            "NestCoupler refreshes, so it requires nested=true; this "
            "domain is not a nest child."
        )
    if not isinstance(cfg.inflow_perturbation_seed, int) \
            or type(cfg.inflow_perturbation_seed) is bool \
            or cfg.inflow_perturbation_seed < 0:
        raise ConfigurationRefused(
            "inflow_perturbation_seed must be a non-negative integer "
            "(it keys every Philox draw), got "
            f"{cfg.inflow_perturbation_seed!r}.")
    if not math.isfinite(cfg.inflow_perturbation_amplitude_scale) \
            or cfg.inflow_perturbation_amplitude_scale < 0.0:
        raise ConfigurationRefused(
            "inflow_perturbation_amplitude_scale must be finite and "
            ">= 0 (0.0 is the registered zero-amplitude control), got "
            f"{cfg.inflow_perturbation_amplitude_scale}.")
    if cfg.inflow_perturbation_faces not in ("inflow", "outflow"):
        raise ConfigurationRefused(
            "inflow_perturbation_faces must be 'inflow' (the mechanism) "
            "or 'outflow' (the registered AC-P3.4 mutation control), "
            f"got {cfg.inflow_perturbation_faces!r}.")
    # PLAN REVIEW, LAST: every scheme this configuration selects must be
    # nameable by the checkpoint writer.  It runs after the per-selector
    # value checks above so that an out-of-schema id is refused by its own
    # message and this one only ever fires for a scheme the loader ADMITS
    # and the identity tables do not know -- the shape of the 2026-09-10
    # defect, where an accepted mp_physics=9 forecast integrated for 59
    # minutes and then died writing its first hourly checkpoint.
    #
    # It comes from woof.checkpoint_identity, a top-level module holding
    # nothing but the identity tables and this gate, and NOT from
    # woof.io.restart, where the tables were born and where every reader
    # still spells them.  Plan review runs in every distribution this
    # module ships in, and the standalone RW-WPS preprocessing project
    # stages woof/*.py with only a named handful of woof/io -- restart.py
    # deliberately not among them, because it reaches the CUDA side.  An
    # import of woof.io.restart here is therefore unresolvable there, and
    # would raise ModuleNotFoundError out of the one call every front door
    # makes, on configurations it should simply have accepted.  Still
    # imported inside the function rather than at module scope, because
    # woof.checkpoint_identity resolves this module in turn; the import is
    # resolved once and cached.
    from woof.checkpoint_identity import (
        require_identifiable_checkpoint_schemes)
    require_identifiable_checkpoint_schemes(cfg)
    # ...and every OTHER consumer this run will reach must have its row for
    # every scheme selected: the vertical preflight, the nest-edge resolver,
    # the radar operator, the moment policy, the offline child, the ring
    # guard, the reflectivity operator, the stock export.  The identity gate
    # above was the first instance of this question (one table, one scheme,
    # one lost forecast); this is the general form, asked through the
    # registry's per-option consumer rows so that a scheme is rows and the
    # next missing row costs a launch, never a run.  Same module family as
    # the import above: woof.physics_registry imports nothing from woof
    # and no device library.
    from woof.physics_registry import require_consumer_rows
    require_consumer_rows(cfg)
    from woof.physics_registry import validate_column_surface_settings
    validate_column_surface_settings(cfg)
    return cfg



# ---------------------------------------------------------------------------
# [radar_heating]: radar latent heating in a deterministic forecast
# ---------------------------------------------------------------------------

from woof.config_keys import KeyRow as _KeyRow, key_rows as _key_rows  # noqa: E402

#: One row per key of the ``[radar_heating]`` table.  The table is OFF when
#: absent; present, it must name its windows.
RADAR_HEATING_KEY_ROWS = _key_rows(
    _KeyRow("windows", "string", None,
            "root of the heating windows, <root>/<YYYYmmddTHHMMZ>/ref.f32 per window end",
            required=True),
    _KeyRow("window_minutes", "number", 15.0,
            "window length in minutes; must divide active_minutes"),
    _KeyRow("active_minutes", "number", 60.0,
            "forced period in minutes from the run start, pass-through after it"),
    _KeyRow("latent_heat_period_min", "number", 20.0,
            "NOAA dt_cond: minutes over which observed condensate is taken to form"),
    _KeyRow("strict_suppression", "boolean", False,
            "force observed-no-echo points to exactly zero heating after NOAA's smoothing"),
    _KeyRow("pbl_extension", "boolean", False,
            "heat down to the PBL top where 200 hPa of the column is observed"),
    _KeyRow("mp_tend_lim", "number", 0.07,
            "microphysics heating clamp in K/s while forced (HRRR's 0.07)"),
)
RADAR_HEATING_KEYS = frozenset(RADAR_HEATING_KEY_ROWS)


@dataclass(frozen=True)
class RadarHeatingConfig:
    """The ``[radar_heating]`` table: lane 6's NOAA heating rule driven from
    reflectivity windows for the first ``active_minutes`` of a forecast.

    The forcing attaches at the run start and reads the window ending at
    ``start + n * window_minutes`` while the clock is inside window ``n``,
    for ``n = 1 .. active_minutes / window_minutes``; after that every
    microphysics step is a pass-through.  There is no arrangement key: an
    observed pre-forecast hour is "start an hour early with observed
    windows", a nowcast is "start at the analysis with nowcast windows".

    ``windows=None`` is OFF, and an OFF table leaves every document,
    fingerprint and argv byte for byte as it was.  The attach, refusals and
    receipt are :mod:`woof.da.forecast_heating`.
    """

    windows: str | None = None
    window_minutes: float = 15.0
    active_minutes: float = 60.0
    latent_heat_period_min: float = 20.0
    strict_suppression: bool = False
    pbl_extension: bool = False
    mp_tend_lim: float = 0.07

    @property
    def enabled(self) -> bool:
        return self.windows is not None

    @property
    def window_count(self) -> int:
        return int(round(float(self.active_minutes) / float(self.window_minutes)))

    def window_end_minutes(self) -> tuple[float, ...]:
        """Window end times in minutes from the run start."""
        return tuple(float(self.window_minutes) * n
                     for n in range(1, self.window_count + 1))

    def to_mapping(self) -> dict:
        return {name: getattr(self, name) for name in RADAR_HEATING_KEY_ROWS}

    @classmethod
    def from_mapping(cls, table, *, source: str,
                     base_dir: "Path | None" = None) -> "RadarHeatingConfig":
        """Validate ``table`` (absent means OFF) against the declared rows.

        A relative ``windows`` path is anchored at ``base_dir`` when one is
        given (the config file's directory), so a moved working directory
        cannot point a run at another case's windows.
        """
        if table is None:
            return RADAR_HEATING_OFF
        where = f"[radar_heating] of {source}"
        if not isinstance(table, _Mapping):
            raise ConfigurationRefused(f"{where} must be a table, got {table!r}")
        unknown = sorted(set(table) - RADAR_HEATING_KEYS)
        if unknown:
            raise ConfigurationRefused(
                f"{where} has unknown key(s) {unknown}; known: "
                f"{sorted(RADAR_HEATING_KEYS)}. No key is ignored, because a "
                "dropped heating setting would run defaults under its name")
        if "windows" not in table:
            raise ConfigurationRefused(
                f"{where} names no windows: a heating table without windows "
                "would record a heated forecast whose heating never ran")
        values = {name: row.get(table, where=where)
                  for name, row in RADAR_HEATING_KEY_ROWS.items()}
        windows = str(values["windows"]).strip()
        if not windows:
            raise ConfigurationRefused(f"{where} windows is empty")
        path = Path(windows)
        if base_dir is not None and not path.is_absolute():
            path = Path(base_dir) / path
        values["windows"] = str(path)
        for name in ("window_minutes", "active_minutes",
                     "latent_heat_period_min", "mp_tend_lim"):
            value = float(values[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigurationRefused(
                    f"{where} {name} = {values[name]!r} must be a finite "
                    "positive number")
            values[name] = value
        ratio = values["active_minutes"] / values["window_minutes"]
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ConfigurationRefused(
                f"{where}: window_minutes {values['window_minutes']:g} does not "
                f"divide active_minutes {values['active_minutes']:g}; part of "
                "the forced period would run on the model's own heating while "
                "the record says forced")
        return cls(**values)


RADAR_HEATING_OFF = RadarHeatingConfig()


def declared_key_rows() -> dict[str, dict[str, dict]]:
    """Every declared configuration key row, by the table that owns it.

    The companion of :class:`RunConfig`'s fields for the keys no typed
    field carries (:mod:`woof.config_keys`): a front end reads each key's
    TOML type, default and one-line meaning here, and the loader that owns
    the table checks values against the same row.  Imported inside the
    function because every owning module imports this one.
    """

    from woof import experiment
    from woof.case_data import CASE_DATA_KEY_ROWS
    from woof.fetch import FETCH_HINT_ROWS
    from woof.ingest.soil_downscale import INGEST_TABLE_ROWS
    from woof.simulated_radar_config import declared_key_rows as radar_key_rows

    tables = {
        "experiment": experiment._EXPERIMENT_KEY_ROWS,
        "shared": experiment._SHARED_KEY_ROWS,
        "domain": experiment._DOMAIN_KEY_ROWS,
        "relocation": experiment._RELOCATION_KEY_ROWS,
        "case_data": CASE_DATA_KEY_ROWS,
        "fetch": FETCH_HINT_ROWS,
        "ingest": INGEST_TABLE_ROWS,
        "simulated_radar": radar_key_rows(),
        "radar_heating": RADAR_HEATING_KEY_ROWS,
    }
    return {table: {name: row.to_json() for name, row in rows.items()}
            for table, rows in tables.items()}

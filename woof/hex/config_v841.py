"""Fail-closed MPAS-A v8.4.1 dry-dycore configuration.

The v8.2.3 configuration remains the default lane.  This additive subtype is
the only object that may select the v8.4.1 CPU implementation, which prevents
an old configuration from being relabelled by changing receipt text.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import warnings

import numpy as np

from .driver import DryDycoreConfig
from .errors import ConfigurationRefusal
from .species_row import species_row_for_scheme as _species_row_for_scheme

V841_SOURCE_RELEASE = "v8.4.1"
V823_SOURCE_RELEASE = "v8.2.3"
# Restatements of two digests ``woof.hex.cuda_gwdo_v841`` COMPUTES at import,
# kept here because this module is the one the frozen config admits from and
# it may not import the CUDA GWDO module to get them.
#
# THE CONTRACT DIGEST IS DERIVED, NEVER TYPED.  It hashes an authority
# document that frames a module NAME into itself
# (``"module_key": "woof.hex.cuda_gwdo_v841"``), so the 0.2.0 package rename
# moved it, with no arithmetic having changed: the cost a name-framed
# digest is designed to charge.
#
# THE BREAKAGE A STALE VALUE HERE CAUSES, measured on an RTX 3080 on
# 2026-08-28 rather than argued: ``CudaPhaseOneExecutionProvenanceV841.
# validate`` compares the provenance the adapter built against this constant,
# so the first composite step of EVERY full-physics forecast refused with
# ``external GWDO provenance gwdo_contract_sha256 changed`` while the whole
# CPU battery was green.  Re-derive with
# ``python tools/repin_source_tables.py --write``; the CPU gate that now
# holds the two sides equal is
# ``tests/test_cuda_v841_full_physics_x4.py::
# test_every_restated_contract_digest_equals_the_module_that_computes_it``.
V841_GWDO_CONTRACT_SHA256 = (
    "f3c92686b2b0f86b7076e10bd261a7897ef42b3ebd7f7a60789e2b8ee95e96ef"
)
V841_GWDO_KERNEL_SHA256 = (
    "b334506b289c6f002a2660e6c7796361e39372f6652b8e60268c1400900ab9ec"
)
V841_GWDO_EXTERNAL_COMPOSER = "PersistentTwoPhaseCudaPhysicsBackendV841"
V841_GWDO_COMPOSITION_PHASE = "before_held_tendency_coupling"
V841_GWDO_RESULT_MODULE = "woof.hex.cuda_gwdo_v841"
V841_GWDO_RESULT_CLASS = "CudaYsuGwdoResultV841"

V841_ARWEN_BUILD_COMMIT = "fa35bfbdeb4a33a1121852fad930bb3841302856"
V841_ARWEN_AGGREGATE_FACTORY = "woof.core.mpas_column_batch.run_mpas_column_batch"
V841_ARWEN_PHASE1_ORCHESTRATION = "woof.core.physics.PhysicsDriver.compute"
V841_ARWEN_SOURCE_MANIFEST = (
    (
        "woof/core/mpas_column_batch.py",
        "4da7cfbe3e12e201a2c3ab50909355f43ef2704dc839601ba889fe547f8df941",
    ),
    (
        "woof/core/physics.py",
        "e4dc59d4d38dab927e18af85bc3fb5163216f7fd659671fa8a0e8f6a9b4fc8a8",
    ),
    (
        "woof/core/microphysics.py",
        "571e0c2724fe5d2126d34d77b742b2c249c3ac91132800148c50382365de069f",
    ),
    (
        "woof/core/gf.py",
        "4e4e200c6b8cf10e09c9d7ef72820b4356c298456b0f98686ad0f617dd6e98fd",
    ),
    (
        "woof/core/kernels/gf.cu",
        "bc8b301453a413f71b5e9a117cba2a9cab8dcf9f566bc91c5a03ff2a60b47514",
    ),
    (
        "woof/core/rrtmg_legacy.py",
        "c2acafc1e3a089d04a9b271cc5ffdd5473b6af3085a22b05742de9a8063cf0ae",
    ),
)
V841_ARWEN_H_DIABATIC_APPLIED = False


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


class LegacyEpssmIgnored(UserWarning):
    """A nonzero ``config_epssm`` was read and ignored, as MPAS-A v8.4.1 does.

    The text is upstream's own (mpas_atm_dynamics_checks), so a receipt that
    records warnings says what the native model would have logged.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)
        # Warning transports reconstruct the instance with type(*args).
        # Retain the numeric constructor argument, not the formatted text.
        super().__init__(self.value)

    def __str__(self) -> str:
        return (
            f"config_epssm={self.value!r} is ignored: the specification of the "
            "off-centering parameter for the vertically implicit acoustic "
            "integration using config_epssm in the &nhyd_model namelist group "
            "is no longer supported in MPAS-A v8.4.1; the &damping options "
            "config_epssm_minimum, config_epssm_maximum, "
            "config_epssm_transition_bottom_z and config_epssm_transition_top_z "
            "set level-dependent off-centering"
        )


@dataclass(frozen=True, slots=True)
class V841DryDycoreConfig(DryDycoreConfig):
    """The first executable closed, dry, no-active-mixing v8.4.1 lane."""

    source_release: str = V841_SOURCE_RELEASE

    # v8.4.1 retains this legacy Registry sentinel but no longer consumes it.
    config_epssm: float = 0.0
    config_epssm_minimum: float = 0.1
    config_epssm_maximum: float = 0.5
    config_epssm_transition_bottom_z: float = 30_000.0
    config_epssm_transition_top_z: float = 50_000.0

    # native Registry.xml:264 config_smdiv default 0.1, and
    # mpas_atm_time_integration.F:2401 calls atm_divergence_damping_3d
    # unconditionally inside the acoustic sub-step loop.  The port's boolean has
    # no native counterpart and is descriptive only; config_smdiv is the knob.
    config_divergence_damping: bool = True
    config_smdiv: float = 0.1

    config_gpu_aware_mpi: bool = False
    config_les_model: str = "none"
    config_les_surface: str = "none"
    config_mix_scalars: bool = False
    config_mix_full: bool = True
    config_surface_heat_flux: float = 0.0
    config_surface_moisture_flux: float = 0.0
    config_surface_drag_coefficient: float = 0.0

    def validate(self, *, les_dissipation=None) -> None:
        # The v8.4.1 deformation-coefficient binary32 path IS ported
        # (woof.hex.mixing_v841 CPU authority + smagorinsky_v841_f32 CUDA
        # kernel).  Its dependent coefficients validate through the v8.4.1
        # mixing authority; the inherited v8.2.3 validator then checks the
        # remaining knobs on a neutralized no-mixing view, because its
        # split-three Smagorinsky refusal names the v8.2.3 compiled-MPAS
        # oracle, which is not this lane's authority.
        if not isinstance(self.config_monotonic, (bool, np.bool_)):
            _refuse("config_monotonic", self.config_monotonic,
                "the Registry option is logical", "config_monotonic=True or False")
        if not isinstance(self.config_apply_lbcs, (bool, np.bool_)):
            _refuse(
                "config_apply_lbcs",
                self.config_apply_lbcs,
                "the Registry option is logical",
                "config_apply_lbcs=True or False",
            )
        if not isinstance(self.config_moist_physics, (bool, np.bool_)):
            _refuse(
                "config_moist_physics",
                self.config_moist_physics,
                "the driver branch selector is logical",
                "config_moist_physics=True or False",
            )
        neutral_values = {
            field.name: getattr(self, field.name)
            for field in fields(type(self))
            if field.name in {f.name for f in fields(V841DryDycoreConfig)}
        }
        # The v8.4.1 lane owns the regional limited-area branch and the
        # moist-coefficient path (woof.hex.regional_v841; anchored by the
        # CANDIDATE-REGIONAL-DRY record set); the inherited v8.2.3 validator
        # refuses both by name, so they validate here and the base sees a
        # neutralized view.  The driver still refuses config_apply_lbcs=True
        # unless a RegionalRuntime with admitted LBC state is supplied, and
        # config_moist_physics=True without a declared index_qv.
        neutral_values["config_apply_lbcs"] = False
        neutral_values["config_moist_physics"] = False
        if not isinstance(self.config_rayleigh_damp_u, (bool, np.bool_)):
            _refuse("config_rayleigh_damp_u",self.config_rayleigh_damp_u,
                "the Registry option is logical","config_rayleigh_damp_u=True or False")
        neutral_values["config_rayleigh_damp_u"] = False
        # config_epssm is read and IGNORED, with upstream's own warning,
        # because that is what v8.4.1 does with it: mpas_atm_dynamics_checks
        # (v8.4.1 mpas_atm_time_integration.F:174-197) logs "no longer
        # supported" for ANY nonzero real and the acoustic step reads only
        # the four &damping profile knobs.  A namelist written for an
        # 8.3.1-era MPAS, which sets 0.1, must therefore be read as written.
        # Measured on the native v8.4.1 CPU build on 2026-10-03: it printed
        # this warning and ran.
        #
        # So the only value refused here is one native v8.4.1 cannot read
        # at all: a non-real (the Registry type is real).  The inherited
        # v8.2.3 range test, which guards the v8.2.3 coefficients that DO
        # consume the knob, sees a neutral 0.0, because no v8.4.1 path reads
        # the value: every v8.4.1 acoustic and coefficient path takes its
        # off-centering from woof.hex.offcentering_v841, and the drivers that
        # also carry v8.2.3 dispatch on source_release before either v8.2.3
        # read.  tests/test_epssm_v841.py holds both, by source and by a
        # bitwise run on CPU and CUDA.
        epssm = self.config_epssm
        if isinstance(epssm, (bool, np.bool_)) or not isinstance(
            epssm, (int, float, np.integer, np.floating)
        ):
            _refuse(
                "config_epssm",
                epssm,
                "the MPAS Registry type is real, and native MPAS cannot read "
                "a namelist that gives it another type",
                "a real config_epssm (v8.4.1 ignores it with a warning), or 0.0",
            )
        neutral_values["config_epssm"] = 0.0
        if self.config_horiz_mixing == "2d_smagorinsky":
            from .mixing_v841 import V841MixingConfig

            V841MixingConfig(
                config_horiz_mixing=self.config_horiz_mixing,
                config_len_disp=self.config_len_disp,
                config_visc4_2dsmag=self.config_visc4_2dsmag,
                config_smagorinsky_coef=self.config_smagorinsky_coef,
                config_del4u_div_factor=self.config_del4u_div_factor,
                config_h_ScaleWithMesh=self.config_h_ScaleWithMesh,
                config_mpas_cam_coef=self.config_mpas_cam_coef,
            ).validate()
            neutral_values["config_horiz_mixing"] = "off"
            neutral_values["config_len_disp"] = 0.0
            neutral_values["config_visc4_2dsmag"] = 0.0
            neutral_values["config_smagorinsky_coef"] = 0.0
            neutral_values["config_mpas_cam_coef"] = 0.0
        # ``slots=True`` creates a replacement class object, so zero-argument
        # super() is not reliable in methods retained from the
        # pre-replacement class namespace (CPython dataclasses issue 90562).
        neutral = V841DryDycoreConfig(**neutral_values)
        DryDycoreConfig.validate(neutral)
        if self.source_release != V841_SOURCE_RELEASE:
            _refuse(
                "source_release",
                self.source_release,
                "this configuration type is pinned to the v8.4.1 source semantics",
                f"source_release={V841_SOURCE_RELEASE!r}",
            )
        # Upstream's test is ``config_epssm /= 0.0``, so NaN warns too.
        if not float(epssm) == 0.0:
            warnings.warn(LegacyEpssmIgnored(float(epssm)), stacklevel=2)

        minimum = float(self.config_epssm_minimum)
        maximum = float(self.config_epssm_maximum)
        bottom = float(self.config_epssm_transition_bottom_z)
        top = float(self.config_epssm_transition_top_z)
        if not np.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            _refuse(
                "config_epssm_minimum",
                self.config_epssm_minimum,
                "the v8.4.1 off-centering profile must be bounded",
                "0<=config_epssm_minimum<=1",
            )
        if not np.isfinite(maximum) or not minimum <= maximum <= 1.0:
            _refuse(
                "config_epssm_maximum",
                self.config_epssm_maximum,
                "the v8.4.1 off-centering profile must be ordered and bounded",
                "config_epssm_minimum<=config_epssm_maximum<=1",
            )
        if not np.isfinite(bottom) or bottom < 0.0:
            _refuse(
                "config_epssm_transition_bottom_z",
                self.config_epssm_transition_bottom_z,
                "the computational-height transition must begin at or above zero",
                "config_epssm_transition_bottom_z>=0",
            )
        if not np.isfinite(top) or top <= bottom:
            _refuse(
                "config_epssm_transition_top_z",
                self.config_epssm_transition_top_z,
                "the computational-height transition must have positive width",
                "config_epssm_transition_top_z>config_epssm_transition_bottom_z",
            )

        if not isinstance(self.config_gpu_aware_mpi, (bool, np.bool_)):
            _refuse(
                "config_gpu_aware_mpi",
                self.config_gpu_aware_mpi,
                "the Registry option is logical",
                "config_gpu_aware_mpi=false",
            )
        if self.config_gpu_aware_mpi:
            _refuse(
                "config_gpu_aware_mpi",
                self.config_gpu_aware_mpi,
                "the closed serial CPU lane has no MPI halo transport",
                "config_gpu_aware_mpi=false",
            )
        from .mixing_v842_les import validate_les_configuration
        validate_les_configuration(self, les_dissipation=les_dissipation)

        # A zero-coefficient 2d_fixed selection is a no-mixing branch.  The
        # active v8.4.1 Smagorinsky association/order remains a separate lane.


@dataclass(frozen=True, slots=True)
class V841MpasColumnPhysicsConfig(V841DryDycoreConfig):
    """v8.4.1 CUDA dynamics plus aggregate column physics.

    This is deliberately a distinct type rather than a relaxed dry config.
    Its selector fields form the boundary contract with the persistent
    two-phase Arwen column backend: a selector the pinned engine seam cannot
    run is refused by name, because the run would record it and not run it.

    The schedule defaults are MPAS-A v8.4.2's own Registry defaults
    (:mod:`woof.hex.mpas_defaults`): order 2, four acoustic sub-steps (two
    through v8.4.1 and still selectable), three
    dynamics splits, the monotonic limiter, radiation every 1800 s.  The
    native x4 reference's schedule (order 3, six sub-steps, 600 s radiation,
    no limiter) is ``mpas_defaults.X4_RECORDED_SCHEDULE``, passed explicitly
    by the byte-identity tests and the x4 proof.  ``config_dt`` keeps a
    dataclass default of 120 s only so a bare instance is self-consistent;
    every forecast sets it from the mesh (``mpas_defaults.default_dt_seconds``)
    or from ``--dt``.
    """

    config_dt: float = 120.0
    config_time_integration_order: int = 2
    config_number_of_sub_steps: int = 4
    config_dynamics_split_steps: int = 3
    config_split_dynamics_transport: bool = True
    config_scalar_advection: bool = True
    config_monotonic: bool = True
    config_positive_definite: bool = False
    config_apvm_upwinding: float = 0.5
    config_xnutr: float = 0.2
    config_zd: float = 22_000.0
    config_terrain_following: bool | None = True

    config_moist_physics: bool = True
    config_physics_suite: str = "arwen_mpas_column"
    config_microp_scheme: str = "mp_wsm6"
    config_convection_scheme: str = "cu_grell_freitas"
    config_lsm_scheme: str = "sf_noahmp"
    config_pbl_scheme: str = "bl_ysu"
    config_sfclayer_scheme: str = "sf_monin_obukhov_rev"
    config_radt_cld_scheme: str = "cld_fraction"
    config_radt_lw_scheme: str = "rrtmg_lw"
    config_radt_sw_scheme: str = "rrtmg_sw"
    config_gwdo_scheme: str = "off"

    #: MPAS-A config_radtlw_interval = config_radtsw_interval = 00:30:00
    #: (Registry.xml:2316, 2321).
    config_radt_seconds: float = 1800.0
    #: MPAS-A config_pbl_interval and config_conv_interval default to
    #: 'none', which is every step (mpas_atmphys_manager.F:522-523,
    #: 541-542): the surface/PBL and cumulus cadences ARE config_dt.
    config_bldt_seconds: float = 120.0
    #: ``None`` when no cumulus scheme is selected.  Not "zero calls": a
    #: cadence for a scheme that is never called is a number with nothing to
    #: mean, and the sealed constructor refuses ``cumulus_seconds`` without a
    #: ``cumulus_scheme``.  See :mod:`woof.hex.convection_admission`.
    config_cudt_seconds: float | None = 120.0

    phase1_arwen_commit: str = V841_ARWEN_BUILD_COMMIT
    phase1_aggregate_factory: str = V841_ARWEN_AGGREGATE_FACTORY
    phase1_orchestration: str = V841_ARWEN_PHASE1_ORCHESTRATION
    phase1_source_manifest: tuple[tuple[str, str], ...] = V841_ARWEN_SOURCE_MANIFEST
    phase1_h_diabatic_applied: bool = V841_ARWEN_H_DIABATIC_APPLIED

    def validate(self, *, les_dissipation=None) -> None:
        # Reuse every numerical and registry check from the pinned dry lane
        # without weakening that lane's fail-closed physics refusal.
        dry_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841DryDycoreConfig)
        }
        dry_values["config_moist_physics"] = False
        dry_values["config_physics_suite"] = "none"
        V841DryDycoreConfig(**dry_values).validate(les_dissipation=les_dissipation)

        # The cumulus selection is admitted from the ruling's own module.  It
        # was ruled on 2026-08-26 that convection is switched off below 3 km,
        # so "off" is a real configuration of this lane, beside Grell-Freitas
        # and New Tiedtke.  Every other cumulus option of WOOF's physics
        # registry is registered there by the physics resolver when a run
        # selects it (convection_admission.register_engine_scheme), so the
        # one refusal left is a name nothing resolved.
        from . import convection_admission

        if self.config_convection_scheme not in (
            convection_admission.ADMITTED_CONVECTION_SCHEMES
        ):
            _refuse(
                "config_convection_scheme",
                self.config_convection_scheme,
                (
                    "no physics selection resolved this name, so no engine "
                    "cu_physics stands behind it and it would be recorded in "
                    "the receipt and never called"
                ),
                "a cumulus option of WOOF's physics registry, selected with "
                "--physics config_convection_scheme=<MPAS name, registry "
                "option id or cu_physics number>; resolved so far: "
                f"{list(convection_admission.ADMITTED_CONVECTION_SCHEMES)}",
            )
        cumulus_scheme = convection_admission.constructor_scheme(
            self.config_convection_scheme
        )

        # The cadence and the selection are one decision, so they are checked
        # as one.  A cumulus cadence with no scheme to call is a number with
        # nothing to mean, and the sealed Arwen constructor refuses the pair
        # AFTER the card is reserved -- which is the wrong place to learn it.
        if cumulus_scheme is None and self.config_cudt_seconds is not None:
            _refuse(
                "config_cudt_seconds",
                self.config_cudt_seconds,
                (
                    "no cumulus scheme is selected, so there is no closure "
                    "for this cadence to call.  SealedArwenConstructorV841 "
                    "refuses 'cumulus_seconds requires a cumulus_scheme' at "
                    "host preparation, after the card is reserved"
                ),
                "config_cudt_seconds=None with config_convection_scheme='off'",
            )
        if cumulus_scheme is not None and self.config_cudt_seconds is None:
            _refuse(
                "config_cudt_seconds",
                self.config_cudt_seconds,
                (
                    "a selected cumulus scheme is called on a cadence.  WRF "
                    "pins cudt=0 for cu_physics=3, so for Grell-Freitas that "
                    "cadence IS config_dt and is not an independent knob"
                ),
                "config_cudt_seconds=config_dt with "
                "config_convection_scheme='cu_grell_freitas'",
            )

        # RETIRED 2026-10-10:
        # the timestep and the schedule are no longer admitted from the
        # earned-anchor registry or the dycore step certificates.  What is
        # still refused is what cannot run: a physics cadence that is not a
        # whole number of steps, and a timestep whose multiples are not exact
        # in binary64 (woof.hex.dt_admission.require_cadences names both).
        # Whether the configuration matches a registered anchor is recorded
        # in the run's receipt (anchor_record / step_anchor_record).
        from . import dt_admission

        try:
            dt_admission.require_cadences(
                self.config_dt,
                radiation_seconds=self.config_radt_seconds,
                surface_pbl_seconds=self.config_bldt_seconds,
                cumulus_seconds=self.config_cudt_seconds,
                cumulus_scheme=cumulus_scheme,
            )
        except dt_admission.DtAdmissionError as error:
            _refuse(
                "config_dt",
                self.config_dt,
                str(error),
                "physics cadences that are whole multiples of config_dt, and "
                "a config_dt whose multiples are exact in binary64",
            )

        # The microphysics scheme is admitted by its species row: the row
        # sizes the scalar block, the preparation pointer table, the moist
        # coefficients, the carrier contracts and the history names, so a
        # scheme with no row would run with an inventory nobody declared.
        # A row with no engine counterparty refuses here by name rather
        # than at the first phase-one call.
        _species_row_for_scheme(self.config_microp_scheme).require_engine_scheme()

        # RETIRED 2026-10-10: the "exact" map that pinned every dynamics knob
        # to the native x4 reference's values ("the matched v8.4.1 real-x4
        # column-physics lane is exact").  The CUDA path honours the
        # schedule, the transport and limiter switches, the off-centering
        # profile, the damping and divergence-damping coefficients and the
        # radiation cadence, so none of them is refused for differing from
        # x4; mpas_defaults.x4_recorded_match records whether a run IS the
        # recorded schedule.  What stays below is what this type cannot
        # execute.
        if not (
            isinstance(self.config_moist_physics, (bool, np.bool_))
            and bool(self.config_moist_physics)
        ):
            _refuse(
                "config_moist_physics",
                self.config_moist_physics,
                "this type runs the column-physics seam every step, which "
                "forms moist coefficients from the scalar block; a dry run "
                "is V841DryDycoreConfig",
                "config_moist_physics=True",
            )
        # The frozen column batch (woof.core.mpas_column_batch) constructs
        # its radiation, land-surface, surface-layer and PBL schemes as
        # literals (woof.hex.forecast_preset.ENGINE_COLUMN_SUITE), so any
        # other selector on THIS type would be recorded and not run.  Every
        # other scheme of WOOF's registry runs through the generic column
        # seam (woof.hex.physics_suites -> woof.hex.cuda_suite_physics_v841),
        # whose configuration type records the selectors it executes.
        engine_built = {
            "config_lsm_scheme": "sf_noahmp",
            "config_pbl_scheme": ("off" if self.config_les_model != "none" and self.config_pbl_scheme == "off" else "bl_ysu"),
            "config_sfclayer_scheme": "sf_monin_obukhov_rev",
            "config_radt_cld_scheme": "cld_fraction",
            "config_radt_lw_scheme": "rrtmg_lw",
            "config_radt_sw_scheme": "rrtmg_sw",
        }
        for knob, built in engine_built.items():
            if getattr(self, knob) != built:
                _refuse(
                    knob,
                    getattr(self, knob),
                    f"THIS configuration type is the frozen column lane, "
                    f"whose batch builds {built}; on it the selector would "
                    "be recorded in the receipt and not run.  The forecast "
                    "door never builds this type for another scheme: it "
                    "resolves any option of WOOF's physics registry "
                    "(--physics, --physics-suite, --physics-namelist) to the "
                    "generic column seam and records the selectors it runs "
                    "(woof.hex.cuda_suite_physics_v841.suite_config)",
                    f"{knob}={built!r} on this type, or the selection through "
                    "the forecast door",
                )
        if self.config_gwdo_scheme != "off" and type(self) is V841MpasColumnPhysicsConfig:
            _refuse(
                "config_gwdo_scheme",
                self.config_gwdo_scheme,
                "gravity-wave drag is composed by V841MpasColumnPhysicsGwdoConfig; "
                "on this type the selector would be recorded and not run",
                "config_gwdo_scheme='off', or V841MpasColumnPhysicsGwdoConfig",
            )
        # The phase-one identity fields name the engine build and the factory
        # the seam executes; another value would make the receipt name code
        # this run does not execute.
        identity = {
            "phase1_arwen_commit": V841_ARWEN_BUILD_COMMIT,
            "phase1_aggregate_factory": V841_ARWEN_AGGREGATE_FACTORY,
            "phase1_orchestration": V841_ARWEN_PHASE1_ORCHESTRATION,
            "phase1_source_manifest": V841_ARWEN_SOURCE_MANIFEST,
            "phase1_h_diabatic_applied": V841_ARWEN_H_DIABATIC_APPLIED,
        }
        for knob, executed in identity.items():
            if getattr(self, knob) != executed:
                _refuse(
                    knob,
                    getattr(self, knob),
                    "this field names the engine build the column seam "
                    "executes; another value would make the receipt name an "
                    "engine this run does not execute",
                    f"{knob}={executed!r}",
                )
        # config_physics_suite records which named suite (hexcore.
        # physics_suites) the run was resolved from.  A name no row carries
        # would put a suite nobody declared into the receipt.
        from .physics_suites import recordable_suite_names

        if self.config_physics_suite not in recordable_suite_names():
            _refuse(
                "config_physics_suite",
                self.config_physics_suite,
                "no row of the suite table (data/profiles/"
                "mpas-physics-suites.v1.json) carries this name, so the "
                "receipt would name a suite nobody declared",
                f"config_physics_suite in {list(recordable_suite_names())}",
            )

    def recorded_lane_match(self) -> dict:
        """Whether this is the native x4 reference's own schedule (evidence)."""

        from .mpas_defaults import x4_recorded_match

        return x4_recorded_match(self)


@dataclass(frozen=True, slots=True)
class V841MpasColumnPhysicsGwdoConfig(V841MpasColumnPhysicsConfig):
    """The column-physics lane with the released external YSU gravity-wave drag."""

    config_gwdo_scheme: str = "bl_ysu_gwdo"
    gwdo_external_composer: str = V841_GWDO_EXTERNAL_COMPOSER
    gwdo_composition_phase: str = V841_GWDO_COMPOSITION_PHASE
    gwdo_contract_sha256: str = V841_GWDO_CONTRACT_SHA256
    gwdo_kernel_sha256: str = V841_GWDO_KERNEL_SHA256
    gwdo_result_module: str = V841_GWDO_RESULT_MODULE
    gwdo_result_class: str = V841_GWDO_RESULT_CLASS
    gwdo_arwen_execution_claim: bool = False

    def validate(self, *, les_dissipation=None) -> None:
        # Validate the parent through its own concrete type, changing only
        # the selector that distinguishes this subtype.
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841MpasColumnPhysicsConfig)
        }
        parent_values["config_gwdo_scheme"] = "off"
        V841MpasColumnPhysicsConfig(**parent_values).validate(les_dissipation=les_dissipation)

        exact = {
            "config_gwdo_scheme": "bl_ysu_gwdo",
            "gwdo_external_composer": V841_GWDO_EXTERNAL_COMPOSER,
            "gwdo_composition_phase": V841_GWDO_COMPOSITION_PHASE,
            "gwdo_contract_sha256": V841_GWDO_CONTRACT_SHA256,
            "gwdo_kernel_sha256": V841_GWDO_KERNEL_SHA256,
            "gwdo_result_module": V841_GWDO_RESULT_MODULE,
            "gwdo_result_class": V841_GWDO_RESULT_CLASS,
            "gwdo_arwen_execution_claim": False,
        }
        for knob, required in exact.items():
            actual = getattr(self, knob)
            if isinstance(required, bool):
                matches = isinstance(actual, (bool, np.bool_)) and bool(actual) is required
            else:
                matches = actual == required
            if not matches:
                # These fields name the drag composer, its kernel and its
                # result type -- the code this run executes for GWDO -- not
                # a tunable.  Another value would make the receipt name a
                # drag computation this run does not perform.
                _refuse(
                    knob,
                    actual,
                    "this field names the gravity-wave-drag code the run "
                    "executes (composer, kernel digest, result type); another "
                    "value would make the receipt name a drag computation this "
                    "run does not perform",
                    f"{knob}={required!r}",
                )


@dataclass(frozen=True, slots=True)
class V841MpasColumnPhysicsSmagorinskyGwdoConfig(V841MpasColumnPhysicsGwdoConfig):
    """Released v8.4.1 GWDO lane plus native 2-D Smagorinsky mixing.

    The mixing defaults are the native Registry configuration the natB-24h
    reference integrated (2d_smagorinsky, len_disp 0.0 -> nominalMinDc,
    visc4_2dsmag 0.05, smagorinsky_coef 0.125, del4u_div_factor 10.0,
    h_ScaleWithMesh true); each is a knob, not a pin.  Runs under this type
    are not bit-comparable to any mixing-off arm.
    """

    config_horiz_mixing: str = "2d_smagorinsky"
    config_len_disp: float = 0.0
    config_visc4_2dsmag: float = 0.05
    config_smagorinsky_coef: float = 0.125
    config_del4u_div_factor: float = 10.0
    config_h_ScaleWithMesh: bool = True

    def validate(self, *, les_dissipation=None) -> None:
        # Validate the GWDO-lane parent through its own
        # concrete type, changing only the mixing block back to that lane's
        # off values.
        parent_values = {
            field.name: getattr(self, field.name)
            for field in fields(V841MpasColumnPhysicsGwdoConfig)
        }
        parent_values["config_horiz_mixing"] = "off"
        parent_values["config_len_disp"] = 0.0
        parent_values["config_visc4_2dsmag"] = 0.0
        parent_values["config_smagorinsky_coef"] = 0.0
        parent_values["config_del4u_div_factor"] = 10.0
        parent_values["config_h_ScaleWithMesh"] = True
        parent_values["config_mpas_cam_coef"] = 0.0
        V841MpasColumnPhysicsGwdoConfig(**parent_values).validate(les_dissipation=les_dissipation)

        # RETIRED 2026-10-10: the pins that held every coefficient at the
        # native Registry value ("the v8.4.1 native-Registry Smagorinsky lane
        # is exact").  The CUDA horizontal kernels read all six from the
        # config, so another value runs as given; V841MixingConfig below
        # still refuses a value outside the Registry's own domain.  The type
        # selects the Smagorinsky branch, so the selector itself must say so.
        if self.config_horiz_mixing != "2d_smagorinsky":
            _refuse(
                "config_horiz_mixing",
                self.config_horiz_mixing,
                "this type's coefficients drive the 2-D Smagorinsky branch; "
                "with another selector they would be recorded and not used",
                "config_horiz_mixing='2d_smagorinsky', or "
                "V841MpasColumnPhysicsGwdoConfig for mixing off",
            )
        # The dependent-coefficient authority runs on the actual values (the
        # neutralized parent view above cannot have seen them).
        from .mixing_v841 import V841MixingConfig

        V841MixingConfig(
            config_horiz_mixing=self.config_horiz_mixing,
            config_len_disp=self.config_len_disp,
            config_visc4_2dsmag=self.config_visc4_2dsmag,
            config_smagorinsky_coef=self.config_smagorinsky_coef,
            config_del4u_div_factor=self.config_del4u_div_factor,
            config_h_ScaleWithMesh=self.config_h_ScaleWithMesh,
            config_mpas_cam_coef=self.config_mpas_cam_coef,
        ).validate()


def _upper_configuration(name, base):
    from dataclasses import make_dataclass
    def validate(self, *, les_dissipation=None):
        base.validate(self, les_dissipation=les_dissipation)
        for knob in ("config_number_cam_damping_levels","config_number_rayleigh_damp_u_levels"):
            value=getattr(self,knob)
            if type(value) is not int or value<1:
                _refuse(knob,value,"the native damping loop requires a positive integer level count","a positive integer")
        days=self.config_rayleigh_damp_u_timescale_days
        if not np.isfinite(days) or days<=0:
            _refuse("config_rayleigh_damp_u_timescale_days",days,"the Rayleigh coefficient divides by a positive timescale","a finite positive timescale")
    return make_dataclass(name,[
        ("config_number_cam_damping_levels",int,4),
        ("config_number_rayleigh_damp_u_levels",int,6),
        ("config_rayleigh_damp_u_timescale_days",float,5.0)],
        bases=(base,),namespace={"__module__":__name__,"validate":validate},frozen=True,slots=True)


V841UpperDampingDryConfig=_upper_configuration("V841UpperDampingDryConfig",V841DryDycoreConfig)
V841UpperDampingColumnConfig=_upper_configuration("V841UpperDampingColumnConfig",V841MpasColumnPhysicsConfig)
V841UpperDampingSmagorinskyConfig=_upper_configuration("V841UpperDampingSmagorinskyConfig",V841MpasColumnPhysicsSmagorinskyGwdoConfig)


__all__ = [
    "LegacyEpssmIgnored",
    "V823_SOURCE_RELEASE",
    "V841_ARWEN_AGGREGATE_FACTORY",
    "V841_ARWEN_BUILD_COMMIT",
    "V841_ARWEN_H_DIABATIC_APPLIED",
    "V841_ARWEN_PHASE1_ORCHESTRATION",
    "V841_ARWEN_SOURCE_MANIFEST",
    "V841_GWDO_COMPOSITION_PHASE",
    "V841_GWDO_CONTRACT_SHA256",
    "V841_GWDO_EXTERNAL_COMPOSER",
    "V841_GWDO_KERNEL_SHA256",
    "V841_GWDO_RESULT_CLASS",
    "V841_GWDO_RESULT_MODULE",
    "V841_SOURCE_RELEASE",
    "V841DryDycoreConfig",
    "V841MpasColumnPhysicsConfig",
    "V841MpasColumnPhysicsGwdoConfig",
    "V841MpasColumnPhysicsSmagorinskyGwdoConfig",
]

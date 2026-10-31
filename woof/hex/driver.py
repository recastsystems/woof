"""Executable closed-sphere dry MPAS-A RK3 driver.

This module wires the scalar authority kernels into the frozen v8.2.3 order of
operations.  The orchestration follows
``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:638-1105``:

* form the three large/acoustic RK timesteps;
* compute large-step tendencies;
* take the forward/backward vertically implicit acoustic substeps;
* recover the large-step variables and advect passive scalars; and
* refresh C-grid diagnostics before the next stage.

Only the closed/global, dry CPU path is admitted here.  Both coupled and split
scalar transport are wired for one dynamics split, including the native
terrain-coordinate conversion when the frozen ``zb``/``zb3`` coupling metrics
are supplied.  The frozen JW 2-D Smagorinsky filters, three-dimensional
divergence damping, and implicit upper-level w damping are linked with their
saved-Euler/acoustic ordering.  Regional integration and the remaining filter
variants refuse by their Registry/configuration names instead of being
approximated.  Receipts retain the narrow frozen no-mixing gate label only on
that exact branch; the original filter branch uses a conservative implemented
label until its local public-history gate becomes a packaged internal oracle.

Logical arrays remain ``(vertical_level, horizontal_entity)``.  Nothing in
this driver decides the eventual GPU memory layout.
"""

from __future__ import annotations

from collections import abc
from contextlib import ExitStack, contextmanager

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
from netCDF4 import Dataset, Variable
from numpy.typing import NDArray

from .acoustic import (
    AcousticStepForcing,
    AcousticStepState,
    advance_acoustic_step,
    compute_vertical_implicit_coefficients,
    convert_w_tendency_to_omega,
    edge_signs_on_cells,
)
from .acoustic_v841 import (
    advance_acoustic_step_v841,
    compute_vertical_implicit_coefficients_v841,
)
from .damping_v841 import build_v841_vertical_velocity_damping
from .diagnostics import SolveDiagnostics, compute_solve_diagnostics
from .dynamics import (
    density_tendency,
    flux3,
    mass_flux_divergence,
    pressure_gradient_euler_tendency,
    vector_invariant_momentum_tendency,
    vertical_transport_u,
)
from .dynamics_v841 import (
    V841ReferenceWindProfiles,
    precomputed_mesh_inverse_v841,
    vector_invariant_momentum_tendency_v841,
)
from .errors import ConfigurationRefusal
from .integration import RKSchedule
from .integration import accumulate_split_flux, finish_split_flux
from .integration_v841 import enforce_recovered_rw_endpoints_v841
from .mixing import (
    AppliedMixingTendencies,
    MixingConfig,
    apply_saved_euler_mixing,
    capture_rtheta_pp_old,
    compute_dry_mixing_tendencies,
    divergence_damping_3d,
)
from .offcentering_v841 import (
    AcousticOffcenteringV841,
    build_v841_acoustic_offcentering,
)
from . import rkind_libm
from .classic_netcdf import WholeReads
from .host_parallel import all_finite, run_blocks, threads as host_threads, transposed_copy
from .prep_cache import cached
from .regional_v841 import (
    RegionalRuntime,
    adjust_dynamics_relaxzone_tend,
    adjust_dynamics_speczone_tend,
    bdy_adjust_scalars,
    bdy_set_scalars,
    clamp_negative_scalars,
    compute_moist_coefficients,
    dynamics_time_offset,
    overwrite_speczone_u_ru,
    pad_cells_column,
    regional_bdy_checks,
    regional_normal_velocity,
    reset_speczone_values,
    rk_timestep_f32,
    transport_rk_timestep_f32,
    zero_speczone_w,
)
from .state import PrognosticState
from .terrain import TerrainCoupling, build_terrain_coupling, recover_velocities
from .transport import (
    AdvectionCoefficients,
    advance_scalar_transport,
    build_advection_coefficients,
)
from .vertical import VerticalGrid, build_vertical_grid


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
WHOLE_STEP_EVIDENCE = "frozen-jw-nomix-step-linked"
ORIGINAL_JW_BRANCH_EVIDENCE = "implemented-original-jw-branch"
NATIVE_SPLIT3_IMPLEMENTATION_EVIDENCE = (
    "implemented-native-dt-dynamics-split3-no-authority-claim"
)
FROZEN_SOURCE = "MPAS-A v8.2.3 mpas_atm_time_integration.F:638-1105"
V823_SOURCE_RELEASE = "v8.2.3"
V841_SOURCE_RELEASE = "v8.4.1"
V841_IMPLEMENTATION_EVIDENCE = (
    "implemented-v841-closed-dry-nomix-no-compiled-authority-claim"
)
V841_SOURCE = (
    "MPAS-A v8.4.2 tag-object=2eee75b147d83d67835cecfb30c6fb3793e3042b; "
    "commit=f34984b2c89353cf31c235b08d3b4acd96b37bb3; "
    "archive-sha256=03e2d518c2f32eccbfa84a48d2c7efa223967819f4f72856f36b87c3b1a796f5; "
    "closed-dry CPU implementation pending compiled authority"
)
NATIVE_SPLIT3_SOURCE = (
    "MPAS-A v8.2.3 mpas_atm_time_integration.F:638-1268,5971-6057"
)
SPLIT_FLUX_REDUCTION = "RKIND:first-copy,current-plus-accumulator,times-reciprocal"


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


class _RegionalEdgeShim:
    """Minimal mesh view for edge kernels on a regional mesh.

    Exposes ``cellsOnEdge`` with ring-7 stored-0 slots remapped to the
    explicit garbage-cell index, plus ``nominalMinDc`` when the base mesh
    carries it (``resolve_config_len_disp`` may consult it)."""

    def __init__(self, cells_on_edge_remapped: NDArray[np.int64], base: object) -> None:
        arrays: dict[str, NDArray[Any]] = {"cellsOnEdge": cells_on_edge_remapped}
        try:
            arrays["nominalMinDc"] = np.asarray(_mesh_array(base, "nominalMinDc"))
        except AttributeError:
            pass
        self.arrays = arrays


@dataclass(frozen=True, slots=True)
class DryDycoreConfig:
    """Works-or-refuses contract for the first executable whole-step path."""

    config_dt: float = 30.0
    config_time_integration_order: int = 3
    config_number_of_sub_steps: int = 6
    config_dynamics_split_steps: int = 1
    config_apply_lbcs: bool = False
    config_split_dynamics_transport: bool = False
    config_scalar_advection: bool = True
    config_monotonic: bool = True
    config_positive_definite: bool = True
    config_scalar_adv_order: int = 3
    config_scalar_vadv_order: int = 3
    config_coef_3rd_order: float = 0.25
    config_apvm_upwinding: float = 0.0
    config_epssm: float = 0.1
    config_moist_physics: bool = False
    config_physics_suite: str = "none"
    config_iau_option: str = "off"
    config_divergence_damping: bool = False
    config_horiz_mixing: str = "off"
    config_len_disp: float = 0.0
    # Port-safe defaults differ from the active Smagorinsky Registry defaults:
    # an inactive branch must not silently carry nonzero active-only knobs.
    config_visc4_2dsmag: float = 0.0
    config_smagorinsky_coef: float = 0.0
    config_del4u_div_factor: float = 10.0
    config_h_ScaleWithMesh: bool = True
    config_mpas_cam_coef: float = 0.0
    config_h_theta_eddy_visc2: float = 0.0
    config_v_theta_eddy_visc2: float = 0.0
    config_h_mom_eddy_visc2: float = 0.0
    config_v_mom_eddy_visc2: float = 0.0
    config_h_theta_eddy_visc4: float = 0.0
    config_h_mom_eddy_visc4: float = 0.0
    config_smdiv: float = 0.0
    config_xnutr: float = 0.0
    config_zd: float = 22_000.0
    config_vertical_mixing: bool = False
    config_rayleigh_damp_u: bool = False
    config_curvature_terms: bool = False
    config_terrain_following: bool | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DryDycoreConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            knob = unknown[0]
            _refuse(
                knob,
                raw[knob],
                "this option is not registered in the dry whole-step authority",
                f"a supported value after porting {knob!r}",
            )
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if not np.isfinite(self.config_dt) or self.config_dt <= 0.0:
            _refuse("config_dt", self.config_dt, "the timestep must be finite and positive", "config_dt>0")
        if isinstance(self.config_time_integration_order, (bool, np.bool_)) or not isinstance(
            self.config_time_integration_order, (int, np.integer)
        ) or self.config_time_integration_order not in (2, 3):
            _refuse(
                "config_time_integration_order",
                self.config_time_integration_order,
                "the MPAS split-explicit loop has only order-2 and order-3 schedules",
                "config_time_integration_order=2 or 3",
            )
        if isinstance(self.config_number_of_sub_steps, (bool, np.bool_)) or not isinstance(
            self.config_number_of_sub_steps, (int, np.integer)
        ) or self.config_number_of_sub_steps < 1:
            _refuse(
                "config_number_of_sub_steps",
                self.config_number_of_sub_steps,
                "the split-explicit acoustic loop requires a positive count",
                "config_number_of_sub_steps=4 (MPAS-A's default) or any positive integer",
            )
        if not isinstance(
            self.config_split_dynamics_transport, (bool, np.bool_)
        ):
            _refuse(
                "config_split_dynamics_transport",
                self.config_split_dynamics_transport,
                "the Registry option is logical",
                "config_split_dynamics_transport=True or False",
            )
        if (
            not isinstance(self.config_dynamics_split_steps, (int, np.integer))
            or isinstance(self.config_dynamics_split_steps, (bool, np.bool_))
            or self.config_dynamics_split_steps not in (1, 3)
        ):
            _refuse(
                "config_dynamics_split_steps",
                self.config_dynamics_split_steps,
                "the whole-step drivers implement one and three dynamics "
                "subcycles only (the v8.4.1 CUDA subcycle three; MPAS-A's "
                "default is three, Registry.xml:109)",
                "config_dynamics_split_steps=1 or 3",
            )
        if (
            self.config_dynamics_split_steps == 3
            and not self.config_split_dynamics_transport
        ):
            _refuse(
                "config_dynamics_split_steps",
                self.config_dynamics_split_steps,
                "the frozen source only activates dynamics subcycles when split transport is enabled",
                "config_split_dynamics_transport=True with config_dynamics_split_steps=3",
            )
        if (
            self.config_dynamics_split_steps == 3
            and self.config_horiz_mixing == "2d_smagorinsky"
        ):
            # This refusal is the V8.2.3 branch's, and stays: its named
            # oracle is the frozen v8.2.3 compiled endpoint, which has no
            # split-three Smagorinsky arm.  The v8.4.1 lane does not reach
            # here -- V841DryDycoreConfig validates the Smagorinsky knobs
            # through its own authority and hands the base a neutralized
            # view -- and its premise IS retired there: the
            # CANDIDATE-REGIONAL-DRY-VALUESAFE record set packages exactly
            # this combination (config_dynamics_split_steps=3 with
            # config_horiz_mixing='2d_smagorinsky', coefficient 0.125,
            # visc4 0.05, len_disp 25000) as a compiled-MPAS oracle, and
            # the regional CPU lane is pinned against it.
            _refuse(
                "config_horiz_mixing",
                self.config_horiz_mixing,
                "native split-three Smagorinsky has no packaged compiled-MPAS "
                "oracle in the v8.2.3 authority (the v8.4.1 lane has one: "
                "CANDIDATE-REGIONAL-DRY-VALUESAFE)",
                "config_horiz_mixing='off', zero-coefficient '2d_fixed', or "
                "V841DryDycoreConfig for the v8.4.1 authority",
            )
        unsupported_bools = (
            ("config_apply_lbcs", self.config_apply_lbcs, "regional driving and halo rows are absent"),
            ("config_moist_physics", self.config_moist_physics, "qtot and moist coefficients are fixed to dry values"),
            ("config_vertical_mixing", self.config_vertical_mixing, "the saved vertical Euler filters are not linked"),
            ("config_rayleigh_damp_u", self.config_rayleigh_damp_u, "the upper-level saved Euler branch is not linked"),
            ("config_curvature_terms", self.config_curvature_terms, "the optional CURVATURE compile branch is not linked"),
        )
        for knob, enabled, reason in unsupported_bools:
            if enabled:
                _refuse(knob, enabled, reason, f"{knob}=False")
        if self.config_physics_suite != "none":
            _refuse(
                "config_physics_suite",
                self.config_physics_suite,
                "this path is the no-physics dry authority",
                "config_physics_suite='none'",
            )
        if self.config_iau_option != "off":
            _refuse(
                "config_iau_option",
                self.config_iau_option,
                "incremental-analysis tendencies are not supplied",
                "config_iau_option='off'",
            )
        if self.config_horiz_mixing not in ("off", "2d_fixed", "2d_smagorinsky"):
            _refuse(
                "config_horiz_mixing",
                self.config_horiz_mixing,
                "the dry authority admits off, zero 2d_fixed, and frozen 2d_smagorinsky",
                "config_horiz_mixing='2d_smagorinsky' or a supported no-mixing value",
            )
        mixing_coefficients = (
            "config_h_theta_eddy_visc2",
            "config_v_theta_eddy_visc2",
            "config_h_mom_eddy_visc2",
            "config_v_mom_eddy_visc2",
            "config_h_theta_eddy_visc4",
            "config_h_mom_eddy_visc4",
        )
        for knob in mixing_coefficients:
            value = float(getattr(self, knob))
            if not np.isfinite(value) or value != 0.0:
                _refuse(
                    knob,
                    value,
                    "nonzero saved Euler mixing is not linked in the whole-step driver",
                    f"{knob}=0.0",
                )
        if self.config_horiz_mixing == "2d_smagorinsky":
            MixingConfig(
                config_horiz_mixing=self.config_horiz_mixing,
                config_len_disp=self.config_len_disp,
                config_visc4_2dsmag=self.config_visc4_2dsmag,
                config_smagorinsky_coef=self.config_smagorinsky_coef,
                config_del4u_div_factor=self.config_del4u_div_factor,
                config_h_ScaleWithMesh=self.config_h_ScaleWithMesh,
                config_mpas_cam_coef=self.config_mpas_cam_coef,
                config_smdiv=self.config_smdiv,
            ).validate()
        else:
            inactive_active_only = (
                ("config_len_disp", self.config_len_disp),
                ("config_visc4_2dsmag", self.config_visc4_2dsmag),
                ("config_smagorinsky_coef", self.config_smagorinsky_coef),
            )
            for knob, raw_value in inactive_active_only:
                value = float(raw_value)
                if not np.isfinite(value) or value != 0.0:
                    _refuse(
                        knob,
                        raw_value,
                        "this Smagorinsky-only knob is inactive for "
                        f"config_horiz_mixing={self.config_horiz_mixing!r} and "
                        "must not be silently ignored",
                        f"{knob}=0.0 or config_horiz_mixing='2d_smagorinsky'",
                    )
            inactive_smag_scalars = (
                ("config_del4u_div_factor", self.config_del4u_div_factor, 0.0, "positive"),
                ("config_mpas_cam_coef", self.config_mpas_cam_coef, 0.0, "zero"),
            )
            for knob, raw_value, minimum, special in inactive_smag_scalars:
                value = float(raw_value)
                invalid = not np.isfinite(value) or value < minimum
                invalid = invalid or (special == "positive" and value <= 0.0)
                invalid = invalid or (special == "zero" and value != 0.0)
                if invalid:
                    _refuse(
                        knob,
                        raw_value,
                        "the configured value is outside the frozen Registry domain",
                        f"an admitted finite {knob}",
                    )
            if not isinstance(self.config_h_ScaleWithMesh, (bool, np.bool_)):
                _refuse(
                    "config_h_ScaleWithMesh",
                    self.config_h_ScaleWithMesh,
                    "the Registry option is logical",
                    "config_h_ScaleWithMesh=True or False",
                )
        if not isinstance(self.config_divergence_damping, (bool, np.bool_)):
            _refuse(
                "config_divergence_damping",
                self.config_divergence_damping,
                "the driver branch selector is logical",
                "config_divergence_damping=True or False",
            )
        if not np.isfinite(self.config_smdiv) or self.config_smdiv < 0.0:
            _refuse(
                "config_smdiv",
                self.config_smdiv,
                "the frozen three-dimensional divergence damping coefficient is non-negative",
                "config_smdiv>=0.0",
            )
        if self.config_divergence_damping and self.config_smdiv == 0.0:
            _refuse(
                "config_smdiv",
                self.config_smdiv,
                "the enabled divergence-damping branch would be an accidental no-op",
                "config_smdiv>0.0 when config_divergence_damping=True",
            )
        if not self.config_divergence_damping and self.config_smdiv != 0.0:
            _refuse(
                "config_divergence_damping",
                self.config_divergence_damping,
                "a nonzero config_smdiv must not be silently ignored",
                "config_divergence_damping=True when config_smdiv>0.0",
            )
        if not np.isfinite(self.config_xnutr) or not 0.0 <= self.config_xnutr <= 1.0:
            _refuse(
                "config_xnutr",
                self.config_xnutr,
                "the frozen Registry bounds the implicit w-damping maximum to [0,1]",
                "0.0<=config_xnutr<=1.0",
            )
        if not np.isfinite(self.config_zd) or self.config_zd <= 0.0:
            _refuse(
                "config_zd",
                self.config_zd,
                "the gravity-wave damping start height must be finite and positive",
                "config_zd>0.0",
            )
        if self.config_scalar_adv_order != 3:
            _refuse(
                "config_scalar_adv_order",
                self.config_scalar_adv_order,
                "the admitted atmosphere whole-step uses the frozen third-order coefficients",
                "config_scalar_adv_order=3",
            )
        if self.config_scalar_vadv_order != 3:
            _refuse(
                "config_scalar_vadv_order",
                self.config_scalar_vadv_order,
                "the admitted atmosphere whole-step uses frozen third-order vertical fluxes",
                "config_scalar_vadv_order=3",
            )
        if not np.isfinite(self.config_coef_3rd_order) or self.config_coef_3rd_order < 0.0:
            _refuse(
                "config_coef_3rd_order",
                self.config_coef_3rd_order,
                "the upwind coefficient must be finite and non-negative",
                "config_coef_3rd_order=0.25",
            )
        if (
            not np.isfinite(self.config_apvm_upwinding)
            or self.config_apvm_upwinding < 0.0
        ):
            _refuse(
                "config_apvm_upwinding",
                self.config_apvm_upwinding,
                "the frozen APVM branch requires a finite non-negative coefficient",
                "config_apvm_upwinding>=0.0 (the JW authority uses 0.5)",
            )
        # A non-real value is refused by name before the range test: the
        # Registry type is real, native MPAS's namelist reader stops on any
        # other type, and np.isfinite on a string raised a bare TypeError
        # here instead of a refusal that names the knob.
        if isinstance(self.config_epssm, (bool, np.bool_)) or not isinstance(
            self.config_epssm, (int, float, np.integer, np.floating)
        ):
            _refuse(
                "config_epssm",
                self.config_epssm,
                "the MPAS Registry type is real, and native MPAS cannot read "
                "a namelist that gives it another type",
                "a real config_epssm",
            )
        if not np.isfinite(self.config_epssm) or not 0.0 <= self.config_epssm < 1.0:
            _refuse(
                "config_epssm",
                self.config_epssm,
                "off-centering must be finite in [0,1)",
                "0<=config_epssm<1",
            )


def _mixing_config(config: DryDycoreConfig) -> MixingConfig:
    """Project the whole-step Registry contract onto the filter authority."""

    return MixingConfig(
        config_horiz_mixing=config.config_horiz_mixing,
        config_len_disp=config.config_len_disp,
        config_visc4_2dsmag=config.config_visc4_2dsmag,
        config_smagorinsky_coef=config.config_smagorinsky_coef,
        config_del4u_div_factor=config.config_del4u_div_factor,
        config_h_ScaleWithMesh=config.config_h_ScaleWithMesh,
        config_mpas_cam_coef=config.config_mpas_cam_coef,
        config_smdiv=config.config_smdiv,
    )


@cached("frozen-vertical-damping")
def _frozen_vertical_damping(
    mesh: object,
    vertical_grid: VerticalGrid,
    config: DryDycoreConfig,
) -> FloatArray:
    """Build ``dss`` exactly as frozen ``mpas_atm_core.F:1230-1268``.

    Initialized MPAS files carry a zero placeholder for ``dss``; atmosphere
    core fills it from ``config_xnutr`` and ``config_zd`` at run setup.  The
    whole-step driver must therefore rebuild it rather than trusting the file.
    """

    zgrid = np.asarray(vertical_grid.zgrid)
    if zgrid.ndim != 2 or zgrid.shape[0] != vertical_grid.n_vert_levels + 1:
        raise ValueError("vertical_grid.zgrid must have shape (nVertLevels+1, nCells)")
    if zgrid.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("vertical_grid.zgrid must use float32 or float64")
    if not np.all(np.isfinite(zgrid)) or np.any(np.diff(zgrid, axis=0) <= 0.0):
        _refuse(
            "zgrid",
            "non-monotonic or non-finite",
            "the frozen damping profile requires finite increasing columns",
            "a finite strictly increasing zgrid",
        )
    density = np.asarray(_mesh_array(mesh, "meshDensity"), dtype=zgrid.dtype)
    if density.shape != (zgrid.shape[1],) or np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        _refuse(
            "meshDensity",
            f"shape={density.shape}",
            "the frozen damping scaling requires one finite positive value per cell",
            f"a positive meshDensity field with shape ({zgrid.shape[1]},)",
        )

    dtype = zgrid.dtype
    out = np.zeros((vertical_grid.n_vert_levels, zgrid.shape[1]), dtype=dtype)
    xnutr = dtype.type(config.config_xnutr)
    if xnutr == dtype.type(0.0):
        return out

    zd = dtype.type(config.config_zd)
    one_quarter = dtype.type(0.25)
    half_pi = dtype.type(0.5) * np.arccos(dtype.type(-1.0))
    # v8.4.2 core F:1310 calls the working-precision host SIN intrinsic.
    # NumPy's single-precision approximation differs from GNU sinf.
    profile_sin = np.sin
    native_profile = getattr(config, "source_release", V823_SOURCE_RELEASE) == V841_SOURCE_RELEASE
    if native_profile:
        from .native_geometry_math import sin as profile_sin
    active_count = 0
    for cell in range(zgrid.shape[1]):
        top = zgrid[-1, cell]
        density_scale = density[cell] ** one_quarter
        for level in range(vertical_grid.n_vert_levels):
            height = dtype.type(0.5) * (
                zgrid[level, cell] + zgrid[level + 1, cell]
            )
            if height > zd:
                if top == zd:
                    _refuse(
                        "config_zd",
                        config.config_zd,
                        "an active damping layer would divide by zero at the model top",
                        "config_zd below every active column top",
                    )
                phase = half_pi * (height - zd) / (top - zd)
                sine = profile_sin(phase)
                # GNU folds the literal SIN(...)**2.0 to one RKIND product.
                square = sine * sine if native_profile else sine ** dtype.type(2.0)
                out[level, cell] = xnutr * square
                out[level, cell] /= density_scale
                active_count += 1
    if active_count == 0:
        _refuse(
            "config_zd",
            config.config_zd,
            "nonzero config_xnutr has no model layer above the damping start",
            "config_zd below at least one layer midpoint",
        )
    if not np.all(np.isfinite(out)):
        _refuse(
            "config_zd",
            config.config_zd,
            "the frozen damping formula produced a non-finite coefficient",
            "a valid damping height below the model top",
        )
    return out


@dataclass(frozen=True, slots=True)
class DryReferenceState:
    """Hydrostatic base fields normally carried in MPAS diagnostic pools."""

    rho_base: FloatArray
    rho_theta_base: FloatArray
    pressure_base: FloatArray
    exner_base: FloatArray

    def validate(self, shape: tuple[int, int]) -> None:
        for name in self.__slots__:
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if value.dtype.kind != "f" or not all_finite(value):
                raise ValueError(f"{name} must be a finite floating array")
            if np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True, slots=True)
class SyntheticDryCase:
    state: PrognosticState
    vertical_grid: VerticalGrid
    reference: DryReferenceState


@dataclass(frozen=True, slots=True)
class TerrainMetrics:
    """Cell-oriented frozen ``zb``/``zb3`` omega-conversion coefficients."""

    zb_cell: FloatArray
    zb3_cell: FloatArray

    def validate(self, *, nlev: int, ncells: int, max_edges: int) -> None:
        expected = (nlev + 1, ncells, max_edges)
        for name in self.__slots__:
            value = np.asarray(getattr(self, name))
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape} != {expected}")
            # The same boolean as np.all(np.isfinite(value)), checked in blocks on host threads: 0.16 s of one
            # core per member start on the own24 zoom globe (two calls over 2 x 94 M words; host_parallel).
            if value.dtype.kind != "f" or not all_finite(value):
                raise ValueError(f"{name} must be a finite floating array")


@dataclass(frozen=True, slots=True)
class NativeVerticalData:
    vertical_grid: VerticalGrid
    terrain_metrics: TerrainMetrics


@dataclass(frozen=True, slots=True)
class DrySavedDiagnostics:
    """Exact time-level-one theta_m/Exner carried outside mass products.

    MPAS stores ``theta_m`` and diagnostic ``exner`` independently.  Rebuilding
    either from ``rho_theta`` loses float32 source-order bits, so native gates
    pass this sidecar into the first step and each result carries the next one.
    """

    theta_m: FloatArray
    exner: FloatArray
    density_perturbation: FloatArray
    rho_theta_perturbation: FloatArray
    pressure_perturbation: FloatArray
    normal_velocity: FloatArray
    vertical_velocity: FloatArray

    def validate(
        self,
        shape: tuple[int, int],
        dtype: np.dtype[Any],
        n_edges: int,
    ) -> None:
        for name in self.__slots__:
            value = np.asarray(getattr(self, name))
            if name == "normal_velocity":
                expected = (shape[0], n_edges)
            elif name == "vertical_velocity":
                expected = (shape[0] + 1, shape[1])
            else:
                expected = shape
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape} != {expected}")
            if value.dtype != dtype:
                raise TypeError(f"{name} dtype {value.dtype} != state dtype {dtype}")
            if not all_finite(value):
                raise ValueError(f"{name} must be finite")
            if name in ("theta_m", "exner") and np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True, slots=True)
class StateMetrics:
    mass: float
    theta_mass: float
    energy_proxy: float
    min_density: float
    max_density: float
    max_abs_velocity: float
    all_finite: bool


@dataclass(frozen=True, slots=True)
class StabilityBounds:
    max_mass_relative_drift: float = 5.0e-11
    max_energy_relative_drift: float = 0.25
    max_abs_velocity: float = 400.0
    min_density: float = 1.0e-12

    def validate(self) -> None:
        for name in self.__slots__:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class StepReceipt:
    evidence: str
    frozen_source: str
    source_release: str
    start_time_seconds: float
    end_time_seconds: float
    stage_acoustic_steps: tuple[int, int, int]
    dynamics_split_steps: int
    dynamics_timestep_seconds: float
    dynamics_stage_timesteps: tuple[float, float, float]
    scalar_transport_stage_timesteps: tuple[float, float, float] | None
    split_flux_reduction: str
    before: StateMetrics
    after: StateMetrics
    mass_relative_drift: float
    energy_relative_drift: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DryStepResult:
    state: PrognosticState
    receipt: StepReceipt
    saved_diagnostics: DrySavedDiagnostics


@dataclass(frozen=True, slots=True)
class _DynamicsSubcycleResult:
    state: PrognosticState
    saved_diagnostics: DrySavedDiagnostics
    diagnostics: SolveDiagnostics
    mass_flux_u: FloatArray
    mass_flux_w: FloatArray


@dataclass(frozen=True, slots=True)
class RunReceipt:
    evidence: str
    frozen_source: str
    source_release: str
    steps: int
    start_time_seconds: float
    end_time_seconds: float
    initial: StateMetrics
    final: StateMetrics
    max_mass_relative_drift: float
    max_energy_relative_drift: float
    max_abs_velocity: float
    bounds: StabilityBounds

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DryRunResult:
    state: PrognosticState
    receipt: RunReceipt
    saved_diagnostics: DrySavedDiagnostics


def _pressure_exner(
    state: PrognosticState,
    vertical: VerticalGrid,
    *,
    rgas: float,
    cp: float,
    reference_pressure: float,
    native_gnu_powf: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    dtype = state.rho.dtype
    theta = state.rho_theta / state.rho
    zz = np.asarray(vertical.zz, dtype=dtype)
    # Preserve frozen F:2833/5953 left-to-right operation order.  Regrouping
    # zz*rtheta first loses float32 bits that become next-step Exner state.
    argument = zz * dtype.type(rgas / reference_pressure) * state.rho_theta
    if np.any(argument <= 0.0):
        raise FloatingPointError("non-positive dry Exner argument")
    # RKIND pow under the pinned correctly-rounded convention; see
    # woof.hex.rkind_libm for the measured reference-implementation slop
    # this replaces.
    power_function = rkind_libm.powf_rkind
    if native_gnu_powf:
        from .native_powf import powf_native_gnu
        power_function = powf_native_gnu
    exner = power_function(argument, dtype.type(rgas / (cp - rgas)))
    pressure = zz * dtype.type(rgas) * state.rho_theta * exner
    return pressure, exner, theta


def _pressure_perturbation(
    state: PrognosticState,
    reference: DryReferenceState,
    vertical: VerticalGrid,
    *,
    rgas: float,
    cp: float,
    reference_pressure: float,
    exner: FloatArray | None = None,
    rho_theta_perturbation: FloatArray | None = None,
    native_gnu_powf: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Frozen perturbation-pressure evaluation at F:2833-2836/5953-5963."""

    _, computed_exner, theta = _pressure_exner(
        state,
        vertical,
        rgas=rgas,
        cp=cp,
        reference_pressure=reference_pressure,
        native_gnu_powf=native_gnu_powf,
    )
    dtype = state.rho.dtype
    exact_exner = computed_exner if exner is None else np.asarray(exner)
    if exact_exner.shape != state.rho.shape or exact_exner.dtype != dtype:
        raise ValueError("exact exner must match the state shape and dtype")
    exact_rtheta_perturbation = (
        state.rho_theta - reference.rho_theta_base
        if rho_theta_perturbation is None
        else np.asarray(rho_theta_perturbation)
    )
    if (
        exact_rtheta_perturbation.shape != state.rho.shape
        or exact_rtheta_perturbation.dtype != dtype
    ):
        raise ValueError("exact rho_theta_perturbation must match the state shape and dtype")
    pressure_perturbation = (
        np.asarray(vertical.zz, dtype=dtype)
        * dtype.type(rgas)
        * (
            exact_exner * exact_rtheta_perturbation
            + reference.rho_theta_base * (exact_exner - reference.exner_base)
        )
    )
    return pressure_perturbation, exact_exner, theta


def _normal_velocity(state: PrognosticState, mesh: object) -> FloatArray:
    """Recover frozen F:2863 u with its exact multiply/divide ordering."""

    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    denominator = state.rho[:, cells[:, 0]] + state.rho[:, cells[:, 1]]
    if np.any(denominator == 0.0):
        raise FloatingPointError("zero edge-density denominator")
    return state.rho.dtype.type(2.0) * state.rho_u / denominator


def _interface_velocity(
    state: PrognosticState,
    vertical: VerticalGrid,
    mesh: object | None = None,
    terrain_metrics: TerrainMetrics | None = None,
    *,
    third_order_coefficient: float = 0.25,
) -> FloatArray:
    nlev, ncells = state.rho.shape
    if terrain_metrics is not None:
        if mesh is None:
            raise ValueError("mesh is required with terrain_metrics")
        counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
        signs = edge_signs_on_cells(mesh).astype(state.rho.dtype, copy=False)
        coupling = TerrainCoupling(
            n_edges_on_cell=counts,
            edges_on_cell=edges,
            cells_on_edge=cells,
            edges_on_cell_sign=signs,
            zb_cell=np.asarray(terrain_metrics.zb_cell, dtype=state.rho.dtype),
            zb3_cell=np.asarray(terrain_metrics.zb3_cell, dtype=state.rho.dtype),
            config_coef_3rd_order=third_order_coefficient,
        )
        # Frozen F:2793-2903 includes a distinct bottom-interface diagnosis
        # using cf1/cf2/cf3.  Delegating to the terrain authority prevents the
        # tempting but wrong w[0]=0 shortcut after RK stage one.
        return recover_velocities(
            rho_zz=state.rho,
            ru=state.rho_u,
            rw=state.rho_w,
            zz=vertical.zz,
            fzm=vertical.fzm,
            fzp=vertical.fzp,
            cf1=vertical.cf1,
            cf2=vertical.cf2,
            cf3=vertical.cf3,
            coupling=coupling,
        ).w

    velocity = np.zeros((nlev + 1, ncells), dtype=state.rho.dtype)
    momentum = np.zeros((nlev + 1, ncells), dtype=state.rho.dtype)
    for level in range(1, nlev):
        density = vertical.fzm[level] * state.rho[level] + vertical.fzp[level] * state.rho[level - 1]
        metric = vertical.fzm[level] * vertical.zz[level] + vertical.fzp[level] * vertical.zz[level - 1]
        if np.any(density <= 0.0):
            raise FloatingPointError(f"non-positive interface density at level {level}")
        if np.any(metric == 0.0):
            raise FloatingPointError(f"zero interface zz metric at level {level}")
        momentum[level] = state.rho_w[level] / metric
    for level in range(1, nlev):
        density = vertical.fzm[level] * state.rho[level] + vertical.fzp[level] * state.rho[level - 1]
        velocity[level] = momentum[level] / density
    return velocity


def _rho_theta_tendency(
    mesh: object,
    state: PrognosticState,
    saved: PrognosticState,
    vertical: VerticalGrid,
    coefficients: AdvectionCoefficients,
    *,
    rk_step: int,
    third_order_coefficient: float,
    theta_m: FloatArray | None = None,
    theta_m_saved: FloatArray | None = None,
    inv_area_cell: FloatArray | None = None,
    skip_cells: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Frozen dry ``tend_theta`` loops at lines 5254-5346.

    ``skip_cells`` is the regional specified-zone set (bdyMaskCell >
    nRelaxZone): native computes these cells' tendencies through
    garbage-element gathers and then unconditionally overwrites them with
    driving tendencies (atm_bdy_adjust_dynamics_speczone_tend), so the
    skipped lanes are provably dead; skipping them keeps every gather on
    this path sentinel-free.
    """

    theta = (
        state.rho_theta / state.rho
        if theta_m is None
        else np.asarray(theta_m, dtype=state.rho.dtype)
    )
    theta_saved = (
        saved.rho_theta / saved.rho
        if theta_m_saved is None
        else np.asarray(theta_m_saved, dtype=saved.rho.dtype)
    )
    if theta.shape != state.rho.shape or theta_saved.shape != saved.rho.shape:
        raise ValueError("theta_m overrides must match their state shapes")
    nlev, ncells = theta.shape
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    area = _mesh_array(mesh, "areaCell").astype(theta.dtype, copy=False)
    dv = _mesh_array(mesh, "dvEdge").astype(theta.dtype, copy=False)
    adv = np.asarray(coefficients.adv_coefs, dtype=theta.dtype)
    adv3 = np.asarray(coefficients.adv_coefs_3rd, dtype=theta.dtype)
    adv_cells = np.asarray(coefficients.adv_cells_for_edge, dtype=np.int64)
    n_adv = np.asarray(coefficients.n_adv_cells_for_edge, dtype=np.int64)
    result = np.zeros((nlev, ncells), dtype=theta.dtype)
    source_inverse = None
    if inv_area_cell is not None:
        source_inverse = np.asarray(inv_area_cell)
        if source_inverse.dtype != theta.dtype:
            raise TypeError(
                f"inv_area_cell dtype {source_inverse.dtype} != RKIND {theta.dtype}"
            )
        if source_inverse.shape != area.shape:
            raise ValueError("inv_area_cell must have one value per cell")
        if not np.all(np.isfinite(source_inverse)):
            raise ValueError("inv_area_cell must be finite")

    if source_inverse is None:
        for edge, (cell0, cell1) in enumerate(cells_on_edge):
            count = int(n_adv[edge])
            stencil = adv_cells[edge, :count]
            sign_velocity = np.copysign(theta.dtype.type(1.0), state.rho_u[:, edge])
            weights = adv[edge, :count][None, :] + (
                theta.dtype.type(third_order_coefficient)
                * sign_velocity[:, None]
                * adv3[edge, :count][None, :]
            )
            edge_theta = np.sum(weights * theta[:, stencil], axis=1)
            flux = state.rho_u[:, edge] * edge_theta
            result[:, cell0] -= flux
            result[:, cell1] += flux
            if rk_step > 1:
                correction = (
                    dv[edge]
                    * (saved.rho_u[:, edge] - state.rho_u[:, edge])
                    * theta.dtype.type(0.5)
                    * (theta_saved[:, cell0] + theta_saved[:, cell1])
                )
                result[:, cell0] -= correction
                result[:, cell1] += correction
        result /= area[None, :]
    else:
        counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        signs = edge_signs_on_cells(mesh).astype(theta.dtype, copy=False)
        coefficient = theta.dtype.type(third_order_coefficient)
        one = theta.dtype.type(1.0)
        active_cells = np.ones(ncells, dtype=bool) if skip_cells is None else ~np.asarray(skip_cells)
        # Preserve each cell's edge-slot order and each edge's stencil order.
        # Gather only active lanes: regional padding can hold invalid indices.
        for slot in range(edges.shape[1]):
            cells = np.flatnonzero(active_cells & (counts > slot))
            edge = edges[cells, slot]
            edge_theta = np.zeros((nlev, len(cells)), dtype=theta.dtype)
            sign_velocity = np.copysign(one, state.rho_u[:, edge])
            for stencil_slot in range(adv_cells.shape[1]):
                lanes = np.flatnonzero(n_adv[edge] > stencil_slot)
                selected = edge[lanes]
                donor = adv_cells[selected, stencil_slot]
                weight = (adv[selected, stencil_slot]
                          + coefficient * sign_velocity[:, lanes] * adv3[selected, stencil_slot])
                edge_theta[:, lanes] = edge_theta[:, lanes] + weight * theta[:, donor]
            result[:, cells] = result[:, cells] - signs[cells, slot] * state.rho_u[:, edge] * edge_theta
        if rk_step > 1:
            half = theta.dtype.type(0.5)
            for slot in range(edges.shape[1]):
                cells = np.flatnonzero(active_cells & (counts > slot))
                edge = edges[cells, slot]
                cell0, cell1 = cells_on_edge[edge].T
                correction = (signs[cells, slot] * dv[edge]
                              * (saved.rho_u[:, edge] - state.rho_u[:, edge]) * half
                              * (theta_saved[:, cell1] + theta_saved[:, cell0]))
                result[:, cells] = result[:, cells] - correction
        result[:] = result * source_inverse

    vertical_flux = np.zeros((nlev + 1, ncells), dtype=theta.dtype)
    if nlev >= 2:
        level = 1
        vertical_flux[level] = state.rho_w[level] * (
            vertical.fzm[level] * theta[level] + vertical.fzp[level] * theta[level - 1]
        )
        vertical_flux[level] += (saved.rho_w[level] - state.rho_w[level]) * (
            vertical.fzm[level] * theta_saved[level]
            + vertical.fzp[level] * theta_saved[level - 1]
        )
        for level in range(2, nlev - 1):
            vertical_flux[level] = flux3(
                theta[level - 2],
                theta[level - 1],
                theta[level],
                theta[level + 1],
                state.rho_w[level],
                third_order_coefficient,
            )
            vertical_flux[level] += (saved.rho_w[level] - state.rho_w[level]) * (
                vertical.fzm[level] * theta_saved[level]
                + vertical.fzp[level] * theta_saved[level - 1]
            )
        level = nlev - 1
        if level > 1:
            vertical_flux[level] = saved.rho_w[level] * (
                vertical.fzm[level] * theta[level]
                + vertical.fzp[level] * theta[level - 1]
            )
    result -= np.asarray(vertical.rdzw, dtype=theta.dtype)[:, None] * np.diff(vertical_flux, axis=0)
    return result


def _vertical_momentum_transport(
    mesh: object,
    state: PrognosticState,
    vertical: VerticalGrid,
    coefficients: AdvectionCoefficients,
    terrain_metrics: TerrainMetrics | None = None,
    *,
    third_order_coefficient: float,
    vertical_velocity: FloatArray | None = None,
    inv_area_cell: FloatArray | None = None,
    skip_cells: "NDArray[np.bool_] | None" = None,
) -> FloatArray:
    """Dry w horizontal/vertical transport at frozen lines 5017-5168.

    ``skip_cells`` is the regional specified-zone set; those cells' w
    tendencies are unconditionally zeroed by
    atm_bdy_adjust_dynamics_speczone_tend, so skipping keeps their
    advection stencils (which reach garbage elements) from being gathered.
    """

    nlev, ncells = state.rho.shape
    w = (
        _interface_velocity(
            state,
            vertical,
            mesh,
            terrain_metrics,
            third_order_coefficient=third_order_coefficient,
        )
        if vertical_velocity is None
        else np.asarray(vertical_velocity, dtype=state.rho.dtype)
    )
    if w.shape != (nlev + 1, ncells):
        raise ValueError("vertical_velocity override has the wrong shape")
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    area = _mesh_array(mesh, "areaCell").astype(state.rho.dtype, copy=False)
    adv = np.asarray(coefficients.adv_coefs, dtype=state.rho.dtype)
    adv3 = np.asarray(coefficients.adv_coefs_3rd, dtype=state.rho.dtype)
    adv_cells = np.asarray(coefficients.adv_cells_for_edge, dtype=np.int64)
    n_adv = np.asarray(coefficients.n_adv_cells_for_edge, dtype=np.int64)
    result = np.zeros_like(state.rho_w)
    source_inverse = None
    if inv_area_cell is not None:
        source_inverse = np.asarray(inv_area_cell)
        if source_inverse.dtype != state.rho.dtype:
            raise TypeError(
                f"inv_area_cell dtype {source_inverse.dtype} != RKIND {state.rho.dtype}"
            )
        if source_inverse.shape != area.shape:
            raise ValueError("inv_area_cell must have one value per cell")
        if not np.all(np.isfinite(source_inverse)):
            raise ValueError("inv_area_cell must be finite")
    if source_inverse is None:
        for edge, (cell0, cell1) in enumerate(cells_on_edge):
            count = int(n_adv[edge])
            stencil = adv_cells[edge, :count]
            for level in range(1, nlev):
                ru_interface = (
                    vertical.fzm[level] * state.rho_u[level, edge]
                    + vertical.fzp[level] * state.rho_u[level - 1, edge]
                )
                weights = adv[edge, :count] + (
                    state.rho.dtype.type(third_order_coefficient)
                    * np.copysign(state.rho.dtype.type(1.0), ru_interface)
                    * adv3[edge, :count]
                )
                edge_w = np.sum(weights * w[level, stencil])
                flux = ru_interface * edge_w
                result[level, cell0] -= flux
                result[level, cell1] += flux
        result /= area[None, :]
    else:
        counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        signs = edge_signs_on_cells(mesh).astype(state.rho.dtype, copy=False)
        coefficient = state.rho.dtype.type(third_order_coefficient)
        one = state.rho.dtype.type(1.0)
        active_cells = np.ones(ncells, dtype=bool) if skip_cells is None else ~np.asarray(skip_cells)
        for slot in range(edges.shape[1]):
            cells = np.flatnonzero(active_cells & (counts > slot))
            edge = edges[cells, slot]
            ru_interface = (vertical.fzm[1:nlev, None] * state.rho_u[1:nlev, edge]
                            + vertical.fzp[1:nlev, None] * state.rho_u[:nlev-1, edge])
            edge_w = np.zeros_like(ru_interface)
            for stencil_slot in range(adv_cells.shape[1]):
                lanes = np.flatnonzero(n_adv[edge] > stencil_slot)
                selected = edge[lanes]
                donor = adv_cells[selected, stencil_slot]
                weight = (adv[selected, stencil_slot]
                          + coefficient * np.copysign(one, ru_interface[:, lanes])
                          * adv3[selected, stencil_slot])
                edge_w[:, lanes] = edge_w[:, lanes] + weight * w[1:nlev, donor]
            result[1:nlev, cells] = result[1:nlev, cells] - signs[cells, slot] * ru_interface * edge_w
        result[:] = result * source_inverse

    vertical_flux = np.zeros_like(state.rho_w)
    if nlev >= 2:
        level = 1
        vertical_flux[level] = (
            state.rho.dtype.type(0.25)
            * (state.rho_w[level] + state.rho_w[level - 1])
            * (w[level] + w[level - 1])
        )
        for level in range(2, nlev - 1):
            vertical_flux[level] = flux3(
                w[level - 2],
                w[level - 1],
                w[level],
                w[level + 1],
                state.rho.dtype.type(0.5)
                * (state.rho_w[level] + state.rho_w[level - 1]),
                1.0,
            )
        level = nlev - 1
        if level > 1:
            vertical_flux[level] = (
                state.rho.dtype.type(0.25)
                * (state.rho_w[level] + state.rho_w[level - 1])
                * (w[level] + w[level - 1])
            )
    for level in range(1, nlev):
        result[level] -= vertical.rdzu[level] * (
            vertical_flux[level + 1] - vertical_flux[level]
        )
    result[0] = 0.0
    result[-1] = 0.0
    return result


def _state_metrics(mesh: object, state: PrognosticState, vertical: VerticalGrid) -> StateMetrics:
    area = _mesh_array(mesh, "areaCell").astype(np.float64, copy=False)
    dzw = np.asarray(vertical.dzw, dtype=np.float64)
    volume_weight = dzw[:, None] * area[None, :]
    rho = np.asarray(state.rho, dtype=np.float64)
    rho_theta = np.asarray(state.rho_theta, dtype=np.float64)
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    rho_edge = 0.5 * (rho[:, cells[:, 0]] + rho[:, cells[:, 1]])
    velocity = np.asarray(state.rho_u, dtype=np.float64) / rho_edge
    edge_area = (
        0.5
        * _mesh_array(mesh, "dcEdge").astype(np.float64, copy=False)
        * _mesh_array(mesh, "dvEdge").astype(np.float64, copy=False)
    )
    kinetic = np.sum(0.5 * rho_edge * velocity**2 * dzw[:, None] * edge_area[None, :])
    theta_mass = float(np.sum(rho_theta * volume_weight, dtype=np.float64))
    # A positive dry-energy monitor, not a thermodynamic total-energy claim.
    energy_proxy = float(1004.5 * theta_mass + kinetic)
    arrays = (state.rho, state.rho_theta, state.rho_u, state.rho_w, state.scalars)
    finite = all(np.all(np.isfinite(value)) for value in arrays)
    return StateMetrics(
        mass=float(np.sum(rho * volume_weight, dtype=np.float64)),
        theta_mass=theta_mass,
        energy_proxy=energy_proxy,
        min_density=float(np.min(rho)),
        max_density=float(np.max(rho)),
        max_abs_velocity=float(np.max(np.abs(velocity), initial=0.0)),
        all_finite=bool(finite),
    )


def _relative_change(after: float, before: float) -> float:
    scale = max(abs(before), np.finfo(np.float64).tiny)
    return abs(after - before) / scale


# THE COST THIS REMOVES (NOAA RRFS v2 conus3km, 2,341,819 cells x 59 levels, V8X6000 2026-10-29, host-only):
# after the terrain bridge, load_mpas_initial_state spent about 7 s on one thread in whole-array NumPy
# expressions over (level, cell) and (level, edge) fields (coupled perturbations, base and saved pressure,
# Exner arguments, the w coupling, the saved-diagnostic copies).  Each is elementwise, so it is now written in
# column blocks on host threads with the same ufuncs in the same order (woof.hex.host_parallel): the same bytes.
def _columns(expression: Callable[[slice], Any], columns: int) -> FloatArray:
    """``expression(slice(None))`` for a (..., columns) result, written in column blocks on host threads."""

    probe = np.asarray(expression(slice(0, min(1, columns))))
    out = np.empty((*probe.shape[:-1], columns), dtype=probe.dtype)

    def fill(block: slice) -> None:
        out[..., block] = expression(block)

    run_blocks(fill, columns, elements=out.size)
    return out


def _threaded_copy(value: Any) -> Any:
    """``np.asarray(value).copy()`` written in blocks on host threads (the same bytes)."""

    from .host_parallel import copy_into

    array = np.asarray(value)
    out = np.empty(array.shape, dtype=array.dtype)
    copy_into(out, array)
    return out


def _logical_netcdf_field(
    variable: Variable,
    *,
    level_dimension: str,
    entity_dimension: str,
    time_index: int,
) -> FloatArray:
    """Read an MPAS field as logical ``(level, entity)`` regardless of file order."""

    dimensions = list(variable.dimensions)
    selection: list[int | slice] = [slice(None)] * len(dimensions)
    if "Time" in dimensions:
        selection[dimensions.index("Time")] = time_index
    value = np.asarray(variable[tuple(selection)])
    dimensions = [name for name in dimensions if name != "Time"]
    if set(dimensions) != {level_dimension, entity_dimension} or len(dimensions) != 2:
        raise ValueError(
            f"{variable.name} dimensions {variable.dimensions} do not contain exactly "
            f"{level_dimension!r} and {entity_dimension!r}"
        )
    level_axis = dimensions.index(level_dimension)
    entity_axis = dimensions.index(entity_dimension)
    # The same bytes as np.transpose(...).copy(), copied in column blocks on
    # threads (20.2 s of one core on the 20 km init; woof.hex.host_parallel).
    return transposed_copy(value, (level_axis, entity_axis))


class _InvariantOverlay:
    """``variables`` of an init file with the native invariant stream on top.

    A name the invariant file carries resolves there; any other name resolves
    in the init file.  With no invariant file this is the init file itself.
    """

    def __init__(self, init_dataset: Any, invariant_dataset: Any | None) -> None:
        merged = dict(init_dataset.variables)
        if invariant_dataset is not None:
            merged.update(invariant_dataset.variables)
        self.variables = merged


class _WholeVariable:
    """A netCDF4 variable whose whole array is being read on a host thread (woof.hex.classic_netcdf.WholeReads).

    Indexing it the first time indexes that array, which holds the words ``variable[...]`` returns; metadata and
    any later indexing go to the netCDF4 variable as before, and so does the first one when the classic reader
    does not serve the variable.
    """

    def __init__(self, variable: Any, reads: WholeReads, name: str) -> None:
        self._variable = variable
        self._reads = reads
        self._name = name

    def __getitem__(self, key: Any) -> Any:
        reads, self._reads = self._reads, None
        if reads is not None:
            value = reads.take(self._name, tuple(self._variable.shape), self._variable.dtype)
            if value is not None:
                return value[key]
        return self._variable[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._variable, name)


class _WholeReadVariables(abc.Mapping):
    """``dataset.variables`` whose variables in ``reads`` come from threaded whole reads (see _WholeVariable).

    THE COST THIS REMOVES (own24 zoom globe, 167,254 cells x 55 levels, hex 0.4.9 DA member leg on an RTX 5090,
    2026-10-28): the vertical-grid and initial-state loaders read about 1 GB of the init through netCDF4 one
    variable after another on the main thread, before the card's first kernel.
    """

    def __init__(self, variables: Mapping[str, Any], reads: Mapping[str, WholeReads]) -> None:
        self._variables = variables
        self._reads = reads

    def __getitem__(self, name: str) -> Any:
        variable = self._variables[name]
        reads = self._reads.get(name)
        return variable if reads is None else _WholeVariable(variable, reads, name)

    def __contains__(self, name: object) -> bool:
        return name in self._variables

    def __iter__(self) -> Any:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)


class _WholeReadDataset:
    """A dataset whose ``variables`` read the named variables whole on host threads; everything else is netCDF4's."""

    def __init__(self, dataset: Any, reads: WholeReads, names: Any) -> None:
        self._dataset = dataset
        self.variables = _WholeReadVariables(dataset.variables, {name: reads for name in names})

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


@contextmanager
def _whole_reads(dataset: Any, path: Path, names: list[str]) -> Iterator[_WholeReadDataset]:
    """``dataset`` with ``names`` read whole on host threads from now on; reads not taken are dropped at exit."""

    with WholeReads(path, names, workers=host_threads()) as reads:
        yield _WholeReadDataset(dataset, reads, names)


def _initial_state_reads(
    variables: Any,
    *,
    scalar_names: tuple[str, ...],
    normal_velocity_name: str,
    cold_start: bool = False,
) -> list[str]:
    """The state-file variables ``load_mpas_initial_state`` reads whole, in the order it reads them.

    On a cold or DA start the public ``rho``/``theta`` rebuild the coupled state even when the file also carries
    ``rho_zz``/``theta_m``, and ``ru``/``rw`` are rebuilt rather than read (rrfs/suites, RRFS F:7055-7109)."""

    native_rho = "rho_zz" in variables and not (cold_start and "rho" in variables)
    native_theta = "theta_m" in variables and not (cold_start and "theta" in variables)
    names = ["rho_zz" if native_rho else "rho"]
    names += ["theta_m"] if native_theta else ["theta", "qv"]
    names += ["exner", "rtheta_p", "rho_p", "pressure_p", normal_velocity_name, "w"]
    names += [] if cold_start else ["ru", "rw"]
    names += [*scalar_names]
    return [name for name in dict.fromkeys(names) if name in variables]


def _reference_reads(variables: Any, *, cold_start: bool) -> list[str]:
    """The reference-file variables ``load_mpas_initial_state`` reads whole."""

    names = ["rho_base", "theta_base"]
    if not cold_start:
        names += ["rtheta_base", "exner_base", "pressure_base"]
    return [name for name in names if name in variables]


def load_mpas_vertical_grid(
    path: str | Path,
    mesh: object,
    *,
    config_coef_3rd_order: float = 0.25,
    allow_regional_sentinels: bool = False,
    invariant_path: str | Path | None = None,
) -> NativeVerticalData:
    """Load the exact native MPAS ``zgrid/fzm/fzp/zz/zb`` metric set.

    This is the authority path for an initialized file; rebuilding these
    fields from ``ter`` would discard initialization-time surface smoothing.

    ``invariant_path`` is the native ``invariant`` stream.  When a run
    configures it, MPAS-A reads the vertical grid from that file and the
    ``input`` stream omits those fields (Registry.xml: the input stream lists
    ``<stream name="invariant" packages="no_invariant_stream"/>``), so every
    metric field the invariant file carries is taken from it, not from the
    init file.  RRFS v2 runs this way: its L60 invariant file and an init
    built from a different static set disagree by up to 2.7 mm in ``zgrid``
    and 8e-5 relative in ``zz``, which moved a physics-off conus12km hour
    from 2.8e-5 K to 2.0e-4 K RMS theta against native.
    """

    if not np.isfinite(config_coef_3rd_order) or config_coef_3rd_order < 0.0:
        _refuse(
            "config_coef_3rd_order",
            config_coef_3rd_order,
            "zb3 coupling requires a finite non-negative coefficient",
            "config_coef_3rd_order=0.25",
        )
    source = Path(path).expanduser().resolve(strict=True)
    invariant_source = (
        None if invariant_path is None
        else Path(invariant_path).expanduser().resolve(strict=True)
    )
    required = (
        "zgrid", "rdzw", "dzu", "rdzu", "fzm", "fzp", "zz", "zxu",
        "dss", "zb", "zb3", "cf1", "cf2", "cf3",
    )
    with ExitStack() as stack:
        init_dataset = stack.enter_context(Dataset(source, mode="r"))
        invariant_dataset = (
            None if invariant_source is None
            else stack.enter_context(Dataset(invariant_source, mode="r"))
        )
        dataset = _InvariantOverlay(init_dataset, invariant_dataset)
        missing = [name for name in required if name not in dataset.variables]
        if missing:
            knob = missing[0]
            _refuse(
                knob,
                None,
                "native vertical authority is incomplete",
                f"an initialized MPAS file containing {knob}",
            )
        # Every field below is read whole on host threads at once, from the file it resolves in (_WholeVariable).
        by_file: dict[Path, list[str]] = {}
        for name in required:
            owner = (invariant_source if invariant_dataset is not None and name in invariant_dataset.variables
                     else source)
            by_file.setdefault(owner, []).append(name)
        routes: dict[str, WholeReads] = {}
        for owner, names in by_file.items():
            reads = stack.enter_context(WholeReads(owner, names, workers=host_threads()))
            routes.update((name, reads) for name in names)
        dataset.variables = _WholeReadVariables(dataset.variables, routes)
        zgrid = _logical_netcdf_field(
            dataset.variables["zgrid"],
            level_dimension="nVertLevelsP1",
            entity_dimension="nCells",
            time_index=0,
        )
        zz = _logical_netcdf_field(
            dataset.variables["zz"],
            level_dimension="nVertLevels",
            entity_dimension="nCells",
            time_index=0,
        ).astype(zgrid.dtype, copy=False)
        zxu = _logical_netcdf_field(
            dataset.variables["zxu"],
            level_dimension="nVertLevels",
            entity_dimension="nEdges",
            time_index=0,
        ).astype(zgrid.dtype, copy=False)
        dss = _logical_netcdf_field(
            dataset.variables["dss"],
            level_dimension="nVertLevels",
            entity_dimension="nCells",
            time_index=0,
        ).astype(zgrid.dtype, copy=False)
        rdzw = np.asarray(dataset.variables["rdzw"][:], dtype=zgrid.dtype)
        dzu = np.asarray(dataset.variables["dzu"][:], dtype=zgrid.dtype)
        rdzu = np.asarray(dataset.variables["rdzu"][:], dtype=zgrid.dtype)
        fzm = np.asarray(dataset.variables["fzm"][:], dtype=zgrid.dtype)
        fzp = np.asarray(dataset.variables["fzp"][:], dtype=zgrid.dtype)
        raw_zb = np.asarray(dataset.variables["zb"][:], dtype=zgrid.dtype)
        raw_zb3 = np.asarray(dataset.variables["zb3"][:], dtype=zgrid.dtype)
        cf1 = float(np.asarray(dataset.variables["cf1"][...]))
        cf2 = float(np.asarray(dataset.variables["cf2"][...]))
        cf3 = float(np.asarray(dataset.variables["cf3"][...]))

    nlev, ncells = zz.shape
    if rdzw.shape != (nlev,) or zgrid.shape != (nlev + 1, ncells):
        raise ValueError("native vertical dimensions disagree")
    dzw = np.reciprocal(rdzw)
    zw = np.empty(nlev + 1, dtype=zgrid.dtype)
    zw[0] = 0.0
    zw[1:] = np.cumsum(dzw, dtype=zgrid.dtype)
    zu = (zw[:-1] + zw[1:]) * zgrid.dtype.type(0.5)
    rdzwp = np.zeros(nlev, dtype=zgrid.dtype)
    rdzwm = np.zeros(nlev, dtype=zgrid.dtype)
    rdzwp[1:] = dzw[:-1] / (dzw[1:] * (dzw[1:] + dzw[:-1]))
    rdzwm[1:] = dzw[1:] / (dzw[:-1] * (dzw[1:] + dzw[:-1]))
    vertical = VerticalGrid(
        zw=zw,
        dzw=dzw,
        rdzw=rdzw,
        zu=zu,
        dzu=dzu,
        rdzu=rdzu,
        rdzwp=rdzwp,
        rdzwm=rdzwm,
        fzp=fzp,
        fzm=fzm,
        ah=np.zeros(nlev + 1, dtype=zgrid.dtype),
        hx=np.zeros_like(zgrid),
        zgrid=zgrid,
        zz=zz,
        zxu=zxu,
        dss=dss,
        cf1=cf1,
        cf2=cf2,
        cf3=cf3,
        first_height_level=nlev + 1,
    )
    nedges = _mesh_array(mesh, "dcEdge").size
    if raw_zb.shape != (nedges, 2, nlev + 1) or raw_zb3.shape != raw_zb.shape:
        raise ValueError("native zb/zb3 dimensions disagree with mesh")
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    # Keep this loader on the single terrain authority used by physical-state
    # conversion.  The NetCDF metric fields are transposed to the authority's
    # logical (level, side, edge) convention while mesh connectivity is already
    # canonical zero-based.  core.F:1419-1436 scales zb3 exactly once here.
    coupling = build_terrain_coupling(
        n_edges_on_cell=counts,
        edges_on_cell=edges,
        cells_on_edge=cells_on_edge,
        zb=np.transpose(raw_zb, (2, 1, 0)),
        zb3=np.transpose(raw_zb3, (2, 1, 0)),
        config_coef_3rd_order=config_coef_3rd_order,
        array_layout="logical",
        allow_regional_sentinels=allow_regional_sentinels,
    )
    terrain = TerrainMetrics(
        zb_cell=np.asarray(coupling.zb_cell),
        zb3_cell=np.asarray(coupling.zb3_cell),
    )
    terrain.validate(nlev=nlev, ncells=ncells, max_edges=edges.shape[1])
    return NativeVerticalData(vertical_grid=vertical, terrain_metrics=terrain)


def load_mpas_initial_state(
    path: str | Path,
    mesh: object,
    vertical_grid: VerticalGrid,
    *,
    time_index: int = 0,
    scalar_names: tuple[str, ...] = (),
    reference_path: str | Path | None = None,
    terrain_metrics: TerrainMetrics | None = None,
    allow_regional_sentinels: bool = False,
    rgas: float = 287.0,
    cp: float = 1004.5,
    water_vapor_rgas: float = 461.6,
    reference_pressure: float = 100_000.0,
    return_saved_diagnostics: bool = False,
    cold_start: bool = False,
    normal_velocity_name: str = "u",
) -> (
    tuple[PrognosticState, DryReferenceState]
    | tuple[PrognosticState, DryReferenceState, DrySavedDiagnostics]
):
    """Load native MPAS prognostics/reference fields for a whole-step gate.

    ``cold_start=True`` is a forecast that starts from an initial-conditions
    file, which native MPAS-A does through atm_init_coupled_diagnostics
    (v8.4.1 mpas_atm_time_integration.F:7590-7650).  That routine always
    recomputes rho_p, rtheta_base, rtheta_p, exner, exner_base, pressure_p
    and pressure_base from rho_zz, theta_m, rho_base, theta_base and the
    model's own zz, overwriting any copy the file carries.  An init_atmosphere
    file carries pressure_p and pressure_base built on its own zz, so reading
    them puts the init program's numbers, not the model's, into the first
    step's pressure gradient and into every history ``pressure``.  The default
    (False) keeps the oracle-stream behaviour the whole-step gates pin: a
    native model dump's saved diagnostics ARE the model's, and are consumed.

    On a cold or DA start, public ``rho``/``theta`` and ``u``/``w`` rebuild
    the coupled state, even if the file also carries old diagnostic words
    (RRFS mpas_atm_time_integration.F:7055-7109). Otherwise native
    ``rho_zz``/``theta_m`` are consumed when present.  Public init files
    instead carry physical ``rho``/``theta``; those aliases are converted
    exactly as frozen lines 5889-5964, including ``theta_m``'s qv factor.  If
    diagnostic momenta ``ru``/``rw`` were included in an oracle stream they are
    consumed directly; otherwise public ``u``/``w`` are converted with frozen
    density/terrain staggering.  Base Exner and pressure are derived from
    ``rho_base/theta_base/zz`` when not materialized in the stream.
    """

    state_path = Path(path).expanduser().resolve(strict=True)
    ref_path = state_path if reference_path is None else Path(reference_path).expanduser().resolve(strict=True)
    # Every field each block reads is read whole on host threads as the block opens (_WholeVariable).
    with Dataset(state_path, mode="r") as opened, _whole_reads(opened, state_path, _initial_state_reads(
            opened.variables, scalar_names=scalar_names, normal_velocity_name=normal_velocity_name,
            cold_start=cold_start)) as dataset:
        if "rho_zz" in dataset.variables and not (cold_start and "rho" in dataset.variables):
            rho = _logical_netcdf_field(
                dataset.variables["rho_zz"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            )
        elif "rho" in dataset.variables:
            physical_rho = _logical_netcdf_field(
                dataset.variables["rho"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            )
            zz_rho = np.asarray(vertical_grid.zz, dtype=physical_rho.dtype)
            rho = _columns(lambda b: physical_rho[:, b] / zz_rho[:, b], physical_rho.shape[1])
        else:
            _refuse("rho", None, "neither native rho_zz nor public physical rho is present", "an input stream containing rho_zz or rho")
        if "theta_m" in dataset.variables and not (cold_start and "theta" in dataset.variables):
            theta_m = _logical_netcdf_field(
                dataset.variables["theta_m"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        elif "theta" in dataset.variables:
            if "qv" not in dataset.variables:
                _refuse("qv", None, "theta_m derivation requires the frozen water-vapor slot", "an input stream containing qv")
            theta_dry = _logical_netcdf_field(
                dataset.variables["theta"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
            qv = _logical_netcdf_field(
                dataset.variables["qv"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
            # ``rvord = rv/rgas`` is a compile-time REAL(RKIND) division of
            # the float32 constants (mpas_constants.F); rounding the float64
            # quotient instead lands one ulp high and was measured breaking
            # frame-0 theta bitwise identity on the regional ladder.
            rvord = rho.dtype.type(water_vapor_rgas) / rho.dtype.type(rgas)
            unit = rho.dtype.type(1.0)
            theta_m = _columns(lambda b: theta_dry[:, b] * (unit + rvord * qv[:, b]), theta_dry.shape[1])
        else:
            _refuse("theta", None, "neither native theta_m nor public theta is present", "an input stream containing theta_m or theta")
        native_exner = None
        if "exner" in dataset.variables:
            native_exner = _logical_netcdf_field(
                dataset.variables["exner"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        native_rtheta_perturbation = None
        if "rtheta_p" in dataset.variables:
            native_rtheta_perturbation = _logical_netcdf_field(
                dataset.variables["rtheta_p"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        native_density_perturbation = None
        if "rho_p" in dataset.variables:
            native_density_perturbation = _logical_netcdf_field(
                dataset.variables["rho_p"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        native_pressure_perturbation = None
        if "pressure_p" in dataset.variables:
            native_pressure_perturbation = _logical_netcdf_field(
                dataset.variables["pressure_p"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        native_normal_velocity = None
        if normal_velocity_name in dataset.variables:
            native_normal_velocity = _logical_netcdf_field(
                dataset.variables[normal_velocity_name],
                level_dimension="nVertLevels",
                entity_dimension="nEdges",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        native_vertical_velocity = None
        if "w" in dataset.variables:
            native_vertical_velocity = _logical_netcdf_field(
                dataset.variables["w"],
                level_dimension="nVertLevelsP1",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        if cold_start:
            # F:7590-7650 overwrites every one of these on a cold start.
            native_exner = None
            native_rtheta_perturbation = None
            native_density_perturbation = None
            native_pressure_perturbation = None
        # Replaced below by the exact frozen perturbation-state ordering once
        # the base fields have been read.  Keep an initial value so all public
        # input aliases have a complete local representation.
        rho_theta = _columns(lambda b: rho[:, b] * theta_m[:, b], rho.shape[1])
        cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
        gather_rho = rho
        if allow_regional_sentinels:
            # atm_init_coupled_diagnostics runs before any lbc read: the
            # garbage cell of rho_zz still holds its 0.0 allocation, so a
            # one-cell ring-7 edge sums rho_zz(present) + 0.
            gather_rho = np.concatenate(
                [rho, np.zeros((rho.shape[0], 1), dtype=rho.dtype)], axis=1
            )
            cells = np.where(cells < 0, rho.shape[1], cells)
        elif np.any(cells < 0):
            _refuse(
                "config_apply_lbcs",
                True,
                "a boundary edge was found while loading a closed-mesh state",
                "allow_regional_sentinels=True for a culled regional file",
            )
        if "ru" in dataset.variables and not cold_start:
            rho_u = _logical_netcdf_field(
                dataset.variables["ru"],
                level_dimension="nVertLevels",
                entity_dimension="nEdges",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        elif normal_velocity_name in dataset.variables:
            assert native_normal_velocity is not None
            u = native_normal_velocity
            # atm_init_coupled_diagnostics F:7561 exactly:
            # ru = 0.5 * u * (rho_zz(cell1) + rho_zz(cell2)), the half
            # binding to u BEFORE the density sum multiplies in.
            half = rho.dtype.type(0.5)

            def coupled_ru(block: slice) -> Any:
                return (half * u[:, block]) * (
                    gather_rho[:, cells[block, 0]] + gather_rho[:, cells[block, 1]]
                )

            # The same elementwise expression per edge, edge blocks on threads
            # (8.9 s of one core over 4.4 M edges at 20 km; host_parallel).
            probe = coupled_ru(slice(0, 1))
            rho_u = np.empty((probe.shape[0], cells.shape[0]), dtype=probe.dtype)

            def fill_ru(block: slice) -> None:
                rho_u[:, block] = coupled_ru(block)

            run_blocks(fill_ru, cells.shape[0], elements=rho_u.size)
        else:
            _refuse(normal_velocity_name, None, "neither named normal velocity nor diagnostic ru is present", f"an input stream containing {normal_velocity_name} or ru")

        if "rw" in dataset.variables and not cold_start:
            rho_w = _logical_netcdf_field(
                dataset.variables["rw"],
                level_dimension="nVertLevelsP1",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        elif "w" in dataset.variables:
            assert native_vertical_velocity is not None
            w = native_vertical_velocity
            interface_density = np.zeros_like(w)
            interface_zz = np.zeros_like(w)
            vertical_zz = vertical_grid.zz

            def interfaces(block: slice) -> None:
                for level in range(1, rho.shape[0]):
                    interface_density[level, block] = (
                        vertical_grid.fzm[level] * rho[level, block]
                        + vertical_grid.fzp[level] * rho[level - 1, block]
                    )
                    interface_zz[level, block] = (
                        vertical_grid.fzm[level] * vertical_zz[level, block]
                        + vertical_grid.fzp[level] * vertical_zz[level - 1, block]
                    )

            run_blocks(interfaces, w.shape[1], elements=w.size)
            # atm_init_coupled_diagnostics F:7578-7580 exactly:
            # rw = w * (interpolated rho_zz) * (interpolated zz).
            rho_w = _columns(lambda b: w[:, b] * interface_density[:, b] * interface_zz[:, b], w.shape[1])
            # Native explicitly assigns +0 at both boundaries (RRFS F:7083-7084).
            rho_w[0] = rho.dtype.type(0.0)
            rho_w[-1] = rho.dtype.type(0.0)
            zxu = np.asarray(vertical_grid.zxu)
            peaks: list[Any] = []

            def peak(block: slice) -> None:
                peaks.append(np.max(np.abs(zxu[:, block]), initial=0.0))

            run_blocks(peak, zxu.shape[-1] if zxu.ndim else 1, elements=zxu.size)
            has_terrain = np.max(np.asarray(peaks), initial=0.0) > 2.0e-14
            if has_terrain and terrain_metrics is None:
                _refuse(
                    "zb",
                    None,
                    "public w to rw conversion on this initialized grid needs zb/zb3",
                    "terrain_metrics=load_mpas_vertical_grid(...).terrain_metrics",
                )
            if terrain_metrics is not None:
                counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
                edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
                signs = edge_signs_on_cells(mesh).astype(rho.dtype, copy=False)
                terrain_metrics.validate(
                    nlev=rho.shape[0], ncells=rho.shape[1], max_edges=edges.shape[1]
                )
                from .rust_hostprep import correct_vertical_momentum

                rho_w = correct_vertical_momentum(
                    counts=counts, edges=edges, fzm=vertical_grid.fzm,
                    fzp=vertical_grid.fzp, ru=rho_u, interface_zz=interface_zz,
                    signs=signs, zb_cell=terrain_metrics.zb_cell,
                    zb3_cell=terrain_metrics.zb3_cell, rw=rho_w,
                )
        else:
            _refuse("w", None, "neither native w nor diagnostic rw is present", "an input stream containing w or rw")

        scalar_fields: list[FloatArray] = []
        for name in scalar_names:
            if name not in dataset.variables:
                _refuse(name, None, "requested scalar is absent from the input stream", f"an input stream containing {name}")
            scalar_fields.append(
                _logical_netcdf_field(
                    dataset.variables[name],
                    level_dimension="nVertLevels",
                    entity_dimension="nCells",
                    time_index=time_index,
                ).astype(rho.dtype, copy=False)
            )
        scalars = (
            np.stack(scalar_fields, axis=0)
            if scalar_fields
            else np.empty((0, *rho.shape), dtype=rho.dtype)
        )
        time_seconds = 0.0
        if "Time" in dataset.variables:
            time_value = np.asarray(dataset.variables["Time"][time_index]).reshape(-1)
            if time_value.size and np.isfinite(time_value[0]):
                time_seconds = float(time_value[0])

    with Dataset(ref_path, mode="r") as opened, _whole_reads(
            opened, ref_path, _reference_reads(opened.variables, cold_start=cold_start)) as reference_dataset:
        if "rho_base" not in reference_dataset.variables:
            _refuse(
                "rho_base",
                None,
                "the exact MPAS perturbation equations require the frozen base state",
                "reference_path containing rho_base",
            )
        rho_base = _logical_netcdf_field(
            reference_dataset.variables["rho_base"],
            level_dimension="nVertLevels",
            entity_dimension="nCells",
            time_index=time_index,
        ).astype(rho.dtype, copy=False)
        theta_base = None
        if "theta_base" in reference_dataset.variables:
            theta_base = _logical_netcdf_field(
                reference_dataset.variables["theta_base"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        if "rtheta_base" in reference_dataset.variables and not cold_start:
            rho_theta_base = _logical_netcdf_field(
                reference_dataset.variables["rtheta_base"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        elif theta_base is not None:
            rho_theta_base = _columns(lambda b: rho_base[:, b] * theta_base[:, b], rho_base.shape[1])
        else:
            _refuse("theta_base", None, "rtheta_base derivation requires theta_base", "reference_path containing rtheta_base or theta_base")
        if theta_base is None:
            theta_base = rho_theta_base / rho_base

        # F:5937-5947 does not form full rtheta as rho_zz*theta_m.  Preserve
        # its operation order: the difference is measurable in float32 and
        # feeds Exner/acoustic coefficients.
        ncol = rho.shape[1]
        rho_perturbation = (
            _columns(lambda b: rho[:, b] - rho_base[:, b], ncol)
            if native_density_perturbation is None
            else native_density_perturbation
        )
        rho_theta_perturbation = (
            _columns(lambda b: theta_m[:, b] * rho_perturbation[:, b]
                     + rho_base[:, b] * (theta_m[:, b] - theta_base[:, b]), ncol)
            if native_rtheta_perturbation is None
            else native_rtheta_perturbation
        )
        rho_theta = _columns(lambda b: rho_theta_perturbation[:, b] + rho_theta_base[:, b], ncol)

        zz = np.asarray(vertical_grid.zz, dtype=rho.dtype)
        if "exner_base" in reference_dataset.variables and not cold_start:
            exner_base = _logical_netcdf_field(
                reference_dataset.variables["exner_base"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        else:
            exner_scale = rho.dtype.type(rgas / reference_pressure)
            exner_base = rkind_libm.powf_rkind(
                _columns(lambda b: zz[:, b] * exner_scale * rho_theta_base[:, b], ncol),
                rho.dtype.type(rgas / (cp - rgas)),
            )
        if "pressure_base" in reference_dataset.variables and not cold_start:
            pressure_base = _logical_netcdf_field(
                reference_dataset.variables["pressure_base"],
                level_dimension="nVertLevels",
                entity_dimension="nCells",
                time_index=time_index,
            ).astype(rho.dtype, copy=False)
        else:
            gas = rho.dtype.type(rgas)
            pressure_base = _columns(
                lambda b: zz[:, b] * gas * exner_base[:, b] * rho_theta_base[:, b], ncol)
    state = PrognosticState(
        rho=rho,
        rho_theta=rho_theta,
        rho_u=rho_u,
        rho_w=rho_w,
        scalars=scalars,
        time_seconds=time_seconds,
    )
    state.validate(
        n_cells=_mesh_array(mesh, "areaCell").size,
        n_edges=_mesh_array(mesh, "dcEdge").size,
        n_vert_levels=vertical_grid.n_vert_levels,
    )
    reference = DryReferenceState(
        rho_base=rho_base,
        rho_theta_base=rho_theta_base,
        pressure_base=pressure_base,
        exner_base=exner_base,
    )
    reference.validate(rho.shape)
    saved_exner = native_exner
    model_zz = np.asarray(vertical_grid.zz, dtype=rho.dtype)
    if saved_exner is None:
        exner_scale = rho.dtype.type(rgas / reference_pressure)
        saved_exner = rkind_libm.powf_rkind(
            _columns(lambda b: model_zz[:, b] * exner_scale * rho_theta[:, b], rho.shape[1]),
            rho.dtype.type(rgas / (cp - rgas)),
        )
    saved_pressure_perturbation = native_pressure_perturbation
    if saved_pressure_perturbation is None:
        gas = rho.dtype.type(rgas)
        base_rtheta = np.asarray(reference.rho_theta_base)
        base_exner = np.asarray(reference.exner_base)
        saved_pressure_perturbation = _columns(
            lambda b: model_zz[:, b] * gas * (
                saved_exner[:, b] * rho_theta_perturbation[:, b]
                + base_rtheta[:, b] * (saved_exner[:, b] - base_exner[:, b])
            ), rho.shape[1])
    saved_normal_velocity = native_normal_velocity
    if saved_normal_velocity is None:
        saved_normal_velocity = _normal_velocity(state, mesh)
    saved_vertical_velocity = native_vertical_velocity
    if saved_vertical_velocity is None:
        saved_vertical_velocity = _interface_velocity(
            state,
            vertical_grid,
            mesh,
            terrain_metrics,
        )
    saved_diagnostics = DrySavedDiagnostics(
        theta_m=_threaded_copy(theta_m),
        exner=_threaded_copy(saved_exner),
        density_perturbation=_threaded_copy(rho_perturbation),
        rho_theta_perturbation=_threaded_copy(rho_theta_perturbation),
        pressure_perturbation=_threaded_copy(saved_pressure_perturbation),
        normal_velocity=_threaded_copy(saved_normal_velocity),
        vertical_velocity=_threaded_copy(saved_vertical_velocity),
    )
    saved_diagnostics.validate(
        rho.shape,
        rho.dtype,
        _mesh_array(mesh, "dcEdge").size,
    )
    if return_saved_diagnostics:
        return state, reference, saved_diagnostics
    return state, reference


def make_synthetic_x1_case(
    mesh: object,
    *,
    n_vert_levels: int = 4,
    ztop: float = 30_000.0,
    perturbation_amplitude: float = 2.0e-5,
    wind_speed: float = 2.0,
    n_scalars: int = 1,
    time_seconds: float = 0.0,
    rgas: float = 287.0,
    cp: float = 1004.5,
    gravity: float = 9.80616,
    reference_pressure: float = 100_000.0,
    reference_temperature: float = 288.0,
) -> SyntheticDryCase:
    """Construct a deterministic flat, hydrostatic x1/global dry state.

    The perturbation is a zero-area-mean spherical wave in potential
    temperature and normal velocity.  Setting both amplitudes to zero gives an
    exact quiescent fixed point, useful for long-horizon orchestration smoke.
    """

    if n_scalars < 0:
        raise ValueError("n_scalars must be non-negative")
    if perturbation_amplitude < 0.0 or not np.isfinite(perturbation_amplitude):
        raise ValueError("perturbation_amplitude must be finite and non-negative")
    if not np.isfinite(wind_speed):
        raise ValueError("wind_speed must be finite")
    ncells = _mesh_array(mesh, "areaCell").size
    nedges = _mesh_array(mesh, "dcEdge").size
    terrain = np.zeros(ncells, dtype=np.float64)
    vertical = build_vertical_grid(
        mesh,
        terrain,
        n_vert_levels=n_vert_levels,
        ztop=ztop,
        terrain_smoothing_passes=0,
        smooth_surfaces=False,
        xnutr=0.0,
    )
    z_mid = 0.5 * (vertical.zgrid[:-1] + vertical.zgrid[1:])
    pressure_base = reference_pressure * np.exp(
        -gravity * z_mid / (rgas * reference_temperature)
    )
    physical_rho = pressure_base / (rgas * reference_temperature)
    exner_base = (pressure_base / reference_pressure) ** (rgas / cp)
    theta_base = reference_temperature / exner_base
    rho_base = physical_rho / vertical.zz
    rho_theta_base = physical_rho * theta_base / vertical.zz

    lat_cell = _mesh_array(mesh, "latCell").astype(np.float64, copy=False)
    lon_cell = _mesh_array(mesh, "lonCell").astype(np.float64, copy=False)
    pattern = np.cos(lat_cell) ** 2 * np.cos(3.0 * lon_cell)
    area = _mesh_array(mesh, "areaCell").astype(np.float64, copy=False)
    pattern -= np.sum(pattern * area) / np.sum(area)
    pattern /= max(float(np.max(np.abs(pattern))), np.finfo(np.float64).tiny)
    vertical_shape = np.exp(-((z_mid - 6_000.0) / 4_000.0) ** 2)

    rho = rho_base.copy()
    rho_theta = rho_theta_base * (
        1.0 + perturbation_amplitude * vertical_shape * pattern[None, :]
    )
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    lat_edge = _mesh_array(mesh, "latEdge").astype(np.float64, copy=False)
    lon_edge = _mesh_array(mesh, "lonEdge").astype(np.float64, copy=False)
    edge_wave = np.cos(lat_edge) * np.sin(2.0 * lon_edge)
    vertical_wind_shape = np.exp(-z_mid[:, :1] / 18_000.0)
    normal_velocity = wind_speed * vertical_wind_shape * edge_wave[None, :]
    rho_edge = 0.5 * (rho[:, cells[:, 0]] + rho[:, cells[:, 1]])
    rho_u = rho_edge * normal_velocity
    rho_w = np.zeros((n_vert_levels + 1, ncells), dtype=np.float64)
    scalars = np.empty((n_scalars, n_vert_levels, ncells), dtype=np.float64)
    for scalar in range(n_scalars):
        background = 0.2 + 0.1 * scalar
        scalars[scalar] = background + 0.05 * vertical_shape * pattern[None, :]

    state = PrognosticState(
        rho=rho,
        rho_theta=rho_theta,
        rho_u=rho_u,
        rho_w=rho_w,
        scalars=scalars,
        time_seconds=float(time_seconds),
    )
    state.validate(n_cells=ncells, n_edges=nedges, n_vert_levels=n_vert_levels)
    reference = DryReferenceState(
        rho_base=rho_base.copy(),
        rho_theta_base=rho_theta_base.copy(),
        pressure_base=pressure_base.copy(),
        exner_base=exner_base.copy(),
    )
    reference.validate((n_vert_levels, ncells))
    return SyntheticDryCase(state=state, vertical_grid=vertical, reference=reference)


class DryDycoreDriver:
    """Reusable CPU driver with mesh/advection setup cached across steps."""

    def __init__(
        self,
        mesh: object,
        vertical_grid: VerticalGrid,
        reference: DryReferenceState,
        config: DryDycoreConfig | None = None,
        *,
        advection_coefficients: AdvectionCoefficients | None = None,
        terrain_metrics: TerrainMetrics | None = None,
        reference_wind_profiles: V841ReferenceWindProfiles | None = None,
        regional: RegionalRuntime | None = None,
        index_qv: int | None = None,
        rgas: float = 287.0,
        cp: float = 1004.5,
        gravity: float = 9.80616,
        reference_pressure: float = 100_000.0,
    ) -> None:
        self.mesh = mesh
        self.vertical_grid = vertical_grid
        self.reference = reference
        self.config = DryDycoreConfig() if config is None else config
        self.config.validate()
        # Import lazily: config_v841 derives from DryDycoreConfig and therefore
        # cannot be imported while this module is defining the base class.
        from .config_v841 import V841DryDycoreConfig

        selected_release = getattr(self.config, "source_release", V823_SOURCE_RELEASE)
        if selected_release == V841_SOURCE_RELEASE:
            if not isinstance(self.config, V841DryDycoreConfig):
                _refuse(
                    "source_release",
                    selected_release,
                    "receipt relabeling cannot select different numerical source semantics",
                    "V841DryDycoreConfig(source_release='v8.4.1')",
                )
        elif selected_release != V823_SOURCE_RELEASE:
            _refuse(
                "source_release",
                selected_release,
                "only source-pinned v8.2.3 and v8.4.1 implementations exist",
                "source_release='v8.2.3' or V841DryDycoreConfig",
            )
        self.source_release = selected_release
        self.rgas = float(rgas)
        self.cp = float(cp)
        self.gravity = float(gravity)
        self.reference_pressure = float(reference_pressure)
        if self.cp <= self.rgas or min(self.rgas, self.gravity, self.reference_pressure) <= 0.0:
            raise ValueError("dry thermodynamic constants are invalid")
        self.ncells = _mesh_array(mesh, "areaCell").size
        self.nedges = _mesh_array(mesh, "dcEdge").size
        self.nlev = vertical_grid.n_vert_levels
        if self.source_release == V841_SOURCE_RELEASE:
            if reference_wind_profiles is None:
                _refuse(
                    "u_init",
                    None,
                    "v8.4.1 unconditionally subtracts the initialized reference Coriolis flow",
                    "reference_wind_profiles=V841ReferenceWindProfiles(u_init=..., v_init=...)",
                )
            reference_wind_profiles.validate(self.nlev)
            self.reference_wind_profiles = reference_wind_profiles
            self.acoustic_offcentering: AcousticOffcenteringV841 | None = (
                build_v841_acoustic_offcentering(
                    self.vertical_grid.rdzw,
                    minimum=self.config.config_epssm_minimum,
                    maximum=self.config.config_epssm_maximum,
                    transition_bottom_z=(
                        self.config.config_epssm_transition_bottom_z
                    ),
                    transition_top_z=self.config.config_epssm_transition_top_z,
                )
            )
        else:
            if reference_wind_profiles is not None:
                _refuse(
                    "u_init",
                    "provided",
                    "v8.2.3 does not consume the v8.4.1 perturbation-Coriolis profiles",
                    "omit reference_wind_profiles or select V841DryDycoreConfig",
                )
            self.reference_wind_profiles = None
            self.acoustic_offcentering = None
        self.terrain_metrics = terrain_metrics
        self.reference.validate((self.nlev, self.ncells))
        cells = _mesh_array(mesh, "cellsOnEdge")
        self.regional = regional
        self.index_qv = index_qv
        if regional is not None:
            if self.source_release != V841_SOURCE_RELEASE:
                _refuse(
                    "source_release",
                    self.source_release,
                    "the regional limited-area branch is transcribed from "
                    "v8.4.1 and pinned by the CANDIDATE-REGIONAL-DRY record",
                    "V841DryDycoreConfig with config_apply_lbcs=True",
                )
            # mpas_atm_bdy_checks: config_apply_lbcs and boundary cells must
            # agree, and the lbc stream must be usable.  A RegionalRuntime
            # only constructs with a non-empty lbc inventory, which is what
            # a valid input_interval means here.
            regional_bdy_checks(
                regional.masks,
                config_apply_lbcs=bool(self.config.config_apply_lbcs),
                lbc_input_interval_valid=True,
            )
        else:
            if self.config.config_apply_lbcs:
                _refuse(
                    "config_apply_lbcs",
                    True,
                    "no lateral-boundary driving state is loaded and admitted",
                    "regional=RegionalRuntime(mesh, lbc_paths=..., ...) built "
                    "over the culled mesh's lbc file series",
                )
            if np.any(cells < 0):
                _refuse(
                    "config_apply_lbcs",
                    True,
                    "a boundary edge was found and no exterior driving state exists",
                    "config_apply_lbcs=True with regional=RegionalRuntime(...) "
                    "over the culled mesh's lbc file series",
                )
        if self.config.config_moist_physics:
            if self.source_release != V841_SOURCE_RELEASE:
                _refuse(
                    "config_moist_physics",
                    True,
                    "the moist-coefficient path is transcribed from v8.4.1",
                    "V841DryDycoreConfig with config_moist_physics=True",
                )
            if index_qv is None:
                _refuse(
                    "index_qv",
                    None,
                    "moist coefficients need the water-vapor scalar slot",
                    "index_qv=<position of qv in state.scalars>",
                )
        elif index_qv is not None:
            _refuse(
                "index_qv",
                index_qv,
                "a declared qv slot must not be silently ignored",
                "config_moist_physics=True when index_qv is supplied",
            )
        if self.config.config_moist_physics and regional is None:
            _refuse(
                "config_moist_physics",
                True,
                "the moist-coefficient path's only packaged oracle is the "
                "CANDIDATE-REGIONAL-DRY record, which is regional",
                "config_moist_physics=True together with regional=RegionalRuntime(...)",
            )
        if regional is not None and not self.config.config_split_dynamics_transport:
            _refuse(
                "config_split_dynamics_transport",
                self.config.config_split_dynamics_transport,
                "the packaged regional oracle pins split transport; the "
                "coupled-transport regional branch has no oracle to match",
                "config_split_dynamics_transport=True",
            )
        has_terrain = np.max(np.abs(vertical_grid.zxu), initial=0.0) > 2.0e-14
        if has_terrain and self.config.config_terrain_following is False:
            _refuse(
                "config_terrain_following",
                False,
                "the initialized zxu field is non-flat",
                "config_terrain_following=True or None with native terrain_metrics",
            )
        if has_terrain and terrain_metrics is None:
            _refuse(
                "zb",
                None,
                "non-flat zxu requires frozen zb/zb3 omega metrics",
                "terrain_metrics=load_mpas_vertical_grid(...).terrain_metrics",
            )
        if self.config.config_terrain_following is True and terrain_metrics is None:
            _refuse(
                "config_terrain_following",
                True,
                "terrain following was requested without zb/zb3 metrics",
                "terrain_metrics=<native TerrainMetrics>",
            )
        if terrain_metrics is not None:
            terrain_metrics.validate(
                nlev=self.nlev,
                ncells=self.ncells,
                max_edges=_mesh_array(mesh, "edgesOnCell").shape[1],
            )
        self._regional_coupling: TerrainCoupling | None = None
        if regional is not None:
            if terrain_metrics is None:
                _refuse(
                    "zb",
                    None,
                    "a regional real-terrain run needs the native zb/zb3 "
                    "omega metrics",
                    "terrain_metrics=load_mpas_vertical_grid(...).terrain_metrics",
                )
            # The padded recovery coupling: connectivity remapped so ring-7
            # stored-0 slots address the explicit garbage elements, exactly
            # the native memory model.  Only recover_velocities consumes it;
            # its specified-zone w and u outputs are overwritten/zeroed by
            # the regional stages before any consumer reads them.
            pm = regional.padded_mesh
            dtype = np.asarray(vertical_grid.zz).dtype
            signs = edge_signs_on_cells(mesh).astype(dtype, copy=False)
            signs_padded = np.concatenate(
                [signs, np.zeros((1, signs.shape[1]), dtype=dtype)], axis=0
            )
            zb_pad = np.zeros(
                (self.nlev + 1, 1, signs.shape[1]),
                dtype=np.asarray(terrain_metrics.zb_cell).dtype,
            )
            self._regional_coupling = TerrainCoupling(
                n_edges_on_cell=pm.arrays["nEdgesOnCell"],
                edges_on_cell=pm.arrays["edgesOnCell"],
                cells_on_edge=pm.arrays["cellsOnEdge"],
                edges_on_cell_sign=signs_padded,
                zb_cell=np.concatenate(
                    [np.asarray(terrain_metrics.zb_cell), zb_pad], axis=1
                ),
                zb3_cell=np.concatenate(
                    [np.asarray(terrain_metrics.zb3_cell), zb_pad], axis=1
                ),
                config_coef_3rd_order=float(
                    self.config.config_coef_3rd_order
                ),
            )
        if advection_coefficients is None:
            advection_coefficients = build_advection_coefficients(
                mesh,
                config_scalar_adv_order=self.config.config_scalar_adv_order,
                n_vert_levels=self.nlev,
                source_order_v841=(
                    self.source_release == V841_SOURCE_RELEASE
                ),
                allow_regional_sentinels=self.regional is not None,
            )
        self.advection_coefficients = advection_coefficients
        self.mixing_config = (
            _mixing_config(replace(self.config,config_mpas_cam_coef=0.0)
                if self.source_release == V841_SOURCE_RELEASE else self.config)
            if self.config.config_horiz_mixing == "2d_smagorinsky"
            else None
        )
        self._deformation_weights = None
        self._v841_mixing_config = None
        if (
            self.mixing_config is not None
            and getattr(self.config, "source_release", V823_SOURCE_RELEASE)
            == V841_SOURCE_RELEASE
        ):
            # Native computes the deformation weights at model start
            # (atm_initialize_deformation_weights, mpas_atm_core.F:1620) --
            # they are never read from file -- and the v8.4.1 Smagorinsky
            # is the deformation form, not the v8.2.3 defc-table form.
            from .mixing_v841 import (
                V841MixingConfig,
                initialize_deformation_weights_v841,
            )

            self._v841_mixing_config = V841MixingConfig(
                config_horiz_mixing=self.config.config_horiz_mixing,
                config_len_disp=self.config.config_len_disp,
                config_visc4_2dsmag=self.config.config_visc4_2dsmag,
                config_smagorinsky_coef=self.config.config_smagorinsky_coef,
                config_del4u_div_factor=self.config.config_del4u_div_factor,
                config_h_ScaleWithMesh=self.config.config_h_ScaleWithMesh,
                config_mpas_cam_coef=self.config.config_mpas_cam_coef,
            )
            self._deformation_weights = initialize_deformation_weights_v841(
                mesh,
                dtype=np.asarray(vertical_grid.zz).dtype,
            )
        if self.source_release == V841_SOURCE_RELEASE:
            self.evidence = V841_IMPLEMENTATION_EVIDENCE
            self.frozen_source = V841_SOURCE
        elif self.config.config_dynamics_split_steps == 3:
            self.evidence = NATIVE_SPLIT3_IMPLEMENTATION_EVIDENCE
            self.frozen_source = NATIVE_SPLIT3_SOURCE
        else:
            self.evidence = (
                ORIGINAL_JW_BRANCH_EVIDENCE
                if (
                    self.mixing_config is not None
                    or self.config.config_divergence_damping
                    or self.config.config_xnutr > 0.0
                )
                else WHOLE_STEP_EVIDENCE
            )
            self.frozen_source = FROZEN_SOURCE
        if self.source_release == V841_SOURCE_RELEASE:
            self.damping_coefficients = build_v841_vertical_velocity_damping(
                self.vertical_grid.zgrid,
                xnutr=self.config.config_xnutr,
                damping_start_z=self.config.config_zd,
            )
        else:
            self.damping_coefficients = _frozen_vertical_damping(
                self.mesh,
                self.vertical_grid,
                self.config,
            )

    def _validate_state(self, state: PrognosticState) -> None:
        state.validate(n_cells=self.ncells, n_edges=self.nedges, n_vert_levels=self.nlev)
        if state.rho.dtype != state.rho_theta.dtype or state.rho.dtype != state.rho_u.dtype:
            raise TypeError("rho, rho_theta, and rho_u must share a precision")
        if np.any(state.rho_theta <= 0.0):
            raise ValueError("rho_theta must remain strictly positive in the dry path")
        if self.reference_wind_profiles is not None:
            self.reference_wind_profiles.validate(self.nlev, state.rho.dtype)

    def metrics(self, state: PrognosticState) -> StateMetrics:
        self._validate_state(state)
        return _state_metrics(self.mesh, state, self.vertical_grid)

    def _diagnostics(
        self,
        state: PrognosticState,
        *,
        outer_dt: float,
        cached_v: FloatArray | None,
        rk_step: int | None,
        normal_velocity: FloatArray | None = None,
    ) -> SolveDiagnostics:
        velocity = (
            _normal_velocity(state, self.mesh)
            if normal_velocity is None
            else np.asarray(normal_velocity, dtype=state.rho.dtype)
        )
        if velocity.shape != (self.nlev, self.nedges):
            raise ValueError("normal_velocity override has the wrong shape")
        inverse_kwargs: dict[str, FloatArray] = {}
        if self.source_release == V841_SOURCE_RELEASE:
            inverse_kwargs = {
                "inv_area_cell": precomputed_mesh_inverse_v841(
                    self.mesh, "areaCell", state.rho.dtype
                ),
                "inv_area_triangle": precomputed_mesh_inverse_v841(
                    self.mesh, "areaTriangle", state.rho.dtype
                ),
                "inv_dc_edge": precomputed_mesh_inverse_v841(
                    self.mesh, "dcEdge", state.rho.dtype
                ),
                "inv_dv_edge": precomputed_mesh_inverse_v841(
                    self.mesh, "dvEdge", state.rho.dtype
                ),
            }
        if self.regional is not None:
            # Regional solve diagnostics run in the native padded memory
            # model: connectivity remapped to explicit garbage elements,
            # field garbage columns holding the native allocation values
            # (u/v 0, precomputed inverses 0, rho_zz 1 after recover), so
            # every ring-6/7 diagnostic the history stream carries is
            # computed through the same garbage-element arithmetic as the
            # reference executable.
            pm = self.regional.padded_mesh
            padded_kwargs = {
                name: pad_cells_column(value, 0.0)
                for name, value in inverse_kwargs.items()
            }
            padded_cached = (
                None if cached_v is None else pad_cells_column(cached_v, 0.0)
            )
            padded = compute_solve_diagnostics(
                pm,
                pad_cells_column(velocity, 0.0),
                pad_cells_column(state.rho, 1.0),
                dt=outer_dt,
                apvm_upwinding=self.config.config_apvm_upwinding,
                rk_step=rk_step,
                cached_tangential_velocity=padded_cached,
                **padded_kwargs,
            )
            return SolveDiagnostics(
                h_edge=padded.h_edge[:, : self.nedges],
                tangential_velocity=padded.tangential_velocity[:, : self.nedges],
                vorticity=padded.vorticity[:, : pm.n_vertices],
                divergence=padded.divergence[:, : self.ncells],
                kinetic_energy=padded.kinetic_energy[:, : self.ncells],
                pv_edge=padded.pv_edge[:, : self.nedges],
                pv_vertex=padded.pv_vertex[:, : pm.n_vertices],
                pv_cell=padded.pv_cell[:, : self.ncells],
                grad_pv_normal=padded.grad_pv_normal[:, : self.nedges],
                grad_pv_tangential=padded.grad_pv_tangential[:, : self.nedges],
            )
        return compute_solve_diagnostics(
            self.mesh,
            velocity,
            state.rho,
            dt=outer_dt,
            apvm_upwinding=self.config.config_apvm_upwinding,
            rk_step=rk_step,
            cached_tangential_velocity=cached_v,
            **inverse_kwargs,
        )

    def _vertical_coefficients(
        self,
        state: PrognosticState,
        dts: float,
        *,
        exner: FloatArray | None = None,
        theta_m: FloatArray | None = None,
        rho_theta_perturbation: FloatArray | None = None,
        cqw: FloatArray | None = None,
        qtot: FloatArray | None = None,
    ):
        _, computed_exner, theta = _pressure_exner(
            state,
            self.vertical_grid,
            rgas=self.rgas,
            cp=self.cp,
            reference_pressure=self.reference_pressure,
            native_gnu_powf=self.source_release == V841_SOURCE_RELEASE,
        )
        coefficient_exner = computed_exner if exner is None else np.asarray(
            exner, dtype=state.rho.dtype
        )
        coefficient_theta = theta if theta_m is None else np.asarray(
            theta_m, dtype=state.rho.dtype
        )
        coefficient_rtheta_perturbation = (
            state.rho_theta - self.reference.rho_theta_base
            if rho_theta_perturbation is None
            else np.asarray(rho_theta_perturbation, dtype=state.rho.dtype)
        )
        if coefficient_exner.shape != state.rho.shape:
            raise ValueError(
                f"coefficient exner shape {coefficient_exner.shape} != {state.rho.shape}"
            )
        if coefficient_theta.shape != state.rho.shape:
            raise ValueError(
                f"coefficient theta_m shape {coefficient_theta.shape} != {state.rho.shape}"
            )
        if coefficient_rtheta_perturbation.shape != state.rho.shape:
            raise ValueError(
                "coefficient rho_theta_perturbation shape "
                f"{coefficient_rtheta_perturbation.shape} != {state.rho.shape}"
            )
        if self.source_release == V841_SOURCE_RELEASE:
            if self.acoustic_offcentering is None:
                raise RuntimeError("v8.4.1 acoustic off-centering was not initialized")
            return compute_vertical_implicit_coefficients_v841(
                dts=dts,
                offcentering=self.acoustic_offcentering,
                zz=self.vertical_grid.zz,
                cqw=np.ones_like(state.rho) if cqw is None else cqw,
                exner=coefficient_exner,
                theta=coefficient_theta,
                rho_base=self.reference.rho_base,
                rho_theta_base=self.reference.rho_theta_base,
                exner_base=self.reference.exner_base,
                rho_theta_perturbation=coefficient_rtheta_perturbation,
                qtot=np.zeros_like(state.rho) if qtot is None else qtot,
                rdzw=self.vertical_grid.rdzw,
                fzm=self.vertical_grid.fzm,
                fzp=self.vertical_grid.fzp,
                rdzu=self.vertical_grid.rdzu,
                gravity=self.gravity,
                rgas=self.rgas,
                cp=self.cp,
            )
        if cqw is not None or qtot is not None:
            _refuse(
                "config_moist_physics",
                True,
                "the v8.2.3 acoustic coefficients are pinned to dry values",
                "V841DryDycoreConfig for the moist-coefficient path",
            )
        return compute_vertical_implicit_coefficients(
            dts=dts,
            epssm=self.config.config_epssm,
            zz=self.vertical_grid.zz,
            cqw=np.ones_like(state.rho),
            # Despite the historical argument names in acoustic.py, frozen
            # F:1785-1788 binds p/pb to exner/exner_base.  The coefficient
            # equations at F:1889-1904 therefore consume Exner, not pressure.
            pressure=coefficient_exner,
            theta=coefficient_theta,
            rho_base=self.reference.rho_base,
            rho_theta_base=self.reference.rho_theta_base,
            pressure_base=self.reference.exner_base,
            rho_theta_perturbation=coefficient_rtheta_perturbation,
            qtot=np.zeros_like(state.rho),
            rdzw=self.vertical_grid.rdzw,
            fzm=self.vertical_grid.fzm,
            fzp=self.vertical_grid.fzp,
            rdzu=self.vertical_grid.rdzu,
            gravity=self.gravity,
            rgas=self.rgas,
            cp=self.cp,
        )

    def _recover_normal_velocity(self, state: PrognosticState) -> FloatArray:
        if self.regional is None:
            return _normal_velocity(state, self.mesh)
        return regional_normal_velocity(
            state.rho, state.rho_u, self.regional.cells_on_edge_remapped
        )

    def _recover_vertical_velocity(self, state: PrognosticState) -> FloatArray:
        if self.regional is None:
            return _interface_velocity(
                state,
                self.vertical_grid,
                self.mesh,
                self.terrain_metrics,
                third_order_coefficient=self.config.config_coef_3rd_order,
            )
        assert self._regional_coupling is not None
        recovered = recover_velocities(
            rho_zz=pad_cells_column(state.rho, 1.0),
            ru=pad_cells_column(state.rho_u, 0.0),
            rw=pad_cells_column(state.rho_w, 0.0),
            zz=pad_cells_column(self.vertical_grid.zz, 1.0),
            fzm=self.vertical_grid.fzm,
            fzp=self.vertical_grid.fzp,
            cf1=self.vertical_grid.cf1,
            cf2=self.vertical_grid.cf2,
            cf3=self.vertical_grid.cf3,
            coupling=self._regional_coupling,
        ).w[:, : self.ncells]
        # atm_zero_gradient_w_bdy (F:7868-7902) plus the masked recover
        # second pass (F:4492): the specified-zone w column is identically
        # zero after every RK stage.
        zero_speczone_w(masks=self.regional.masks, w=recovered)
        return recovered

    def _rebuild_saved_diagnostics(
        self, state: PrognosticState
    ) -> DrySavedDiagnostics:
        """Lossy fallback for callers without a native/carried sidecar."""

        if self.regional is not None:
            _refuse(
                "saved_diagnostics",
                None,
                "the regional branch carries native-exact coupled "
                "diagnostics; rebuilding them lossily would silently break "
                "the byte pin",
                "saved_diagnostics from load_mpas_initial_state("
                "return_saved_diagnostics=True) or the previous step",
            )

        _, exner, theta = _pressure_exner(
            state,
            self.vertical_grid,
            rgas=self.rgas,
            cp=self.cp,
            reference_pressure=self.reference_pressure,
            native_gnu_powf=self.source_release == V841_SOURCE_RELEASE,
        )
        density_perturbation = state.rho - self.reference.rho_base
        rtheta_perturbation = state.rho_theta - self.reference.rho_theta_base
        pressure_perturbation, _, _ = _pressure_perturbation(
            state,
            self.reference,
            self.vertical_grid,
            rgas=self.rgas,
            cp=self.cp,
            reference_pressure=self.reference_pressure,
            native_gnu_powf=self.source_release == V841_SOURCE_RELEASE,
            exner=exner,
            rho_theta_perturbation=rtheta_perturbation,
        )
        result = DrySavedDiagnostics(
            theta_m=theta,
            exner=exner,
            density_perturbation=density_perturbation,
            rho_theta_perturbation=rtheta_perturbation,
            pressure_perturbation=pressure_perturbation,
            normal_velocity=_normal_velocity(state, self.mesh),
            vertical_velocity=_interface_velocity(
                state,
                self.vertical_grid,
                self.mesh,
                self.terrain_metrics,
                third_order_coefficient=self.config.config_coef_3rd_order,
            ),
        )
        result.validate(state.rho.shape, state.rho.dtype, self.nedges)
        return result

    def _advance_dynamics_subcycle(
        self,
        state: PrognosticState,
        *,
        time_level_one: DrySavedDiagnostics,
        initial_diagnostics: SolveDiagnostics,
        schedule: RKSchedule,
        outer_dt: float,
        cells: NDArray[np.int64],
        zb_cell: FloatArray,
        zb3_cell: FloatArray,
        dynamics_substep: int = 1,
        moist_cqu: FloatArray | None = None,
        moist_cqw: FloatArray | None = None,
        moist_qtot: FloatArray | None = None,
    ) -> _DynamicsSubcycleResult:
        """Advance one chained dry-dynamics RK3 subcycle without model time."""

        self._validate_state(state)
        saved = state.copy()
        time_level_one.validate(saved.rho.shape, saved.rho.dtype, self.nedges)
        if schedule.full_timestep != outer_dt:
            raise ValueError("the dynamics schedule must retain the outer timestep")

        pressure_perturbation_saved = time_level_one.pressure_perturbation
        exner_saved = time_level_one.exner
        theta_saved = time_level_one.theta_m
        density_perturbation_saved = time_level_one.density_perturbation
        rho_theta_perturbation_saved = time_level_one.rho_theta_perturbation

        # Frozen lines 755-783 compute the first vertical solve coefficients
        # before RK and recompute them only at RK stage two.  Exner is a
        # diagnostic updated only after RK stage three (F:2819-2836), so both
        # coefficient evaluations use the saved value; stage two uses the
        # current work-level theta_m as required by F:1785-1803.
        coefficients = self._vertical_coefficients(
            saved,
            schedule.stages[0].acoustic_timestep,
            exner=exner_saved,
            theta_m=theta_saved,
            rho_theta_perturbation=rho_theta_perturbation_saved,
            cqw=moist_cqw,
            qtot=moist_qtot,
        )
        current = saved.copy()
        current_theta_m = theta_saved.copy()
        current_density_perturbation = density_perturbation_saved.copy()
        current_rtheta_perturbation = rho_theta_perturbation_saved.copy()
        current_exner = exner_saved.copy()
        current_pressure_perturbation = pressure_perturbation_saved.copy()
        current_normal_velocity = time_level_one.normal_velocity.copy()
        current_vertical_velocity = time_level_one.vertical_velocity.copy()
        diagnostics = initial_diagnostics
        cached_v = diagnostics.tangential_velocity
        final_flux_u: FloatArray | None = None
        final_flux_w: FloatArray | None = None

        v841_inv_area_cell = None
        v841_inv_dc_edge = None
        if self.source_release == V841_SOURCE_RELEASE:
            v841_inv_area_cell = precomputed_mesh_inverse_v841(
                self.mesh, "areaCell", saved.rho.dtype
            )
            v841_inv_dc_edge = precomputed_mesh_inverse_v841(
                self.mesh, "dcEdge", saved.rho.dtype
            )

        if moist_qtot is None:
            dpdz_saved = -self.gravity * density_perturbation_saved
        else:
            # atm_compute_dyn_tend_work F:6467:
            # dpdz = -gravity*(rb*qtot + rr_save*(1.+qtot))
            one = saved.rho.dtype.type(1.0)
            dpdz_saved = -saved.rho.dtype.type(self.gravity) * (
                self.reference.rho_base * moist_qtot
                + density_perturbation_saved * (one + moist_qtot)
            )
        cqu = np.ones_like(saved.rho_u) if moist_cqu is None else moist_cqu
        euler_ru = pressure_gradient_euler_tendency(
            self.mesh,
            pressure_perturbation_saved,
            dpdz_saved,
            cqu,
            self.vertical_grid.zz,
            self.vertical_grid.zxu,
            inv_dc_edge=v841_inv_dc_edge,
        )
        tend_rho_saved = density_tendency(
            self.mesh,
            saved.rho_u,
            saved.rho_w,
            self.vertical_grid.rdzw,
            inv_area_cell=v841_inv_area_cell,
        )
        euler_rw = np.zeros_like(saved.rho_w)
        for level in range(1, self.nlev):
            # F:6784: tend_w_euler -= cqw*(rdzu*(pp(k)-pp(k-1)) -
            # (fzm*dpdz(k)+fzp*dpdz(k-1))); cqw is one on the dry path and
            # multiplying by exact 1.0f changes no bit.
            inner = (
                self.vertical_grid.rdzu[level]
                * (pressure_perturbation_saved[level] - pressure_perturbation_saved[level - 1])
                - (
                    self.vertical_grid.fzm[level] * dpdz_saved[level]
                    + self.vertical_grid.fzp[level] * dpdz_saved[level - 1]
                )
            )
            if moist_cqw is None:
                euler_rw[level] = -inner
            else:
                euler_rw[level] = -(moist_cqw[level] * inner)

        # Frozen F:4647-4701,4822-4923,5071-5133,5250-5304 computes
        # horizontal filter increments only during RK1 and stores the complete
        # Euler pools for unchanged reuse during RK2/RK3.  Keep the no-mixing
        # branch on its existing arithmetic path rather than adding zero arrays.
        saved_mixing_euler = None
        if self.mixing_config is not None:
            if self.source_release == V841_SOURCE_RELEASE:
                from .mixing_v841 import compute_dry_mixing_tendencies_v841

                assert self._deformation_weights is not None
                saved_mixing = compute_dry_mixing_tendencies_v841(
                    self.mesh,
                    self._deformation_weights,
                    normal_velocity=current_normal_velocity,
                    tangential_velocity=diagnostics.tangential_velocity,
                    vertical_velocity=current_vertical_velocity,
                    theta_m=theta_saved,
                    rho_edge=diagnostics.h_edge,
                    divergence=diagnostics.divergence,
                    vorticity=diagnostics.vorticity,
                    # Frozen F:810 passes the outer transport timestep into
                    # every dynamics subcycle; the Smagorinsky cap must never
                    # see the shorter dynamics timestep.
                    dt=outer_dt,
                    config=self._v841_mixing_config,
                    cam_damping_levels=getattr(self.config,"config_number_cam_damping_levels",4),
                    initial_u_tendency=euler_ru,
                )
            else:
                saved_mixing = compute_dry_mixing_tendencies(
                    self.mesh,
                    normal_velocity=current_normal_velocity,
                    tangential_velocity=diagnostics.tangential_velocity,
                    vertical_velocity=current_vertical_velocity,
                    theta_m=theta_saved,
                    rho_edge=diagnostics.h_edge,
                    divergence=diagnostics.divergence,
                    vorticity=diagnostics.vorticity,
                    # Frozen F:810 passes the outer transport timestep into
                    # every dynamics subcycle; the Smagorinsky cap at
                    # F:4641/4670 must therefore never see the shorter
                    # dynamics timestep.
                    dt=outer_dt,
                    config=self.mixing_config,
                )
            if self.source_release == V841_SOURCE_RELEASE:
                saved_mixing_euler = AppliedMixingTendencies(
                    u=saved_mixing.tend_u_euler,
                    w=euler_rw + saved_mixing.tend_w_euler,
                    theta=saved_mixing.tend_theta_euler,
                )
            else:
                saved_mixing_euler = apply_saved_euler_mixing(
                    euler_ru,
                    euler_rw,
                    np.zeros_like(saved.rho_theta),
                    saved_mixing,
                )

        # atm_compute_dyn_tend_work F:6455-6466 writes tend_rho ONLY under
        # rk_step == 1, so the pool array carries the stage-1 values (and any
        # regional adjustment made to them) into stages 2 and 3, where the
        # regional stages adjust the ALREADY-ADJUSTED array again.  tend_u
        # (F:6517, "first use of tend_u") and tend_theta (F:6818, "= 0.0")
        # reinitialize every stage, so only the density tendency accumulates.
        # Replicated, not fixed: measured on the x1-cull ladder, resetting it
        # per stage instead leaves a relaxation-zone divergence that diffuses
        # inward within one step.
        stage_tend_rho = tend_rho_saved
        if self.regional is not None:
            stage_tend_rho = tend_rho_saved.copy()
        for stage in schedule.stages:
            if stage.stage == 2 and self.config.config_time_integration_order == 3:
                coefficients = self._vertical_coefficients(
                    current,
                    stage.acoustic_timestep,
                    exner=exner_saved,
                    theta_m=current_theta_m,
                    rho_theta_perturbation=current_rtheta_perturbation,
                    cqw=moist_cqw,
                    qtot=moist_qtot,
                )
            rho_edge = 0.5 * (current.rho[:, cells[:, 0]] + current.rho[:, cells[:, 1]])
            normal_velocity = current_normal_velocity
            horizontal_mass_divergence = mass_flux_divergence(
                self.mesh,
                current.rho_u,
                inv_area_cell=v841_inv_area_cell,
            )
            tend_ru = vertical_transport_u(
                self.mesh,
                normal_velocity,
                current.rho_w,
                fzm=self.vertical_grid.fzm,
                fzp=self.vertical_grid.fzp,
                rdzw=self.vertical_grid.rdzw,
            )
            if self.source_release == V841_SOURCE_RELEASE:
                if self.reference_wind_profiles is None:
                    raise RuntimeError("v8.4.1 reference winds were not initialized")
                tend_ru = vector_invariant_momentum_tendency_v841(
                    self.mesh,
                    normal_velocity=normal_velocity,
                    rho_edge=rho_edge,
                    pv_edge=diagnostics.pv_edge,
                    kinetic_energy=diagnostics.kinetic_energy,
                    horizontal_divergence=horizontal_mass_divergence,
                    reference_wind=self.reference_wind_profiles,
                    initial_tendency=tend_ru,
                )
            else:
                tend_ru += vector_invariant_momentum_tendency(
                    self.mesh,
                    normal_velocity=normal_velocity,
                    rho_edge=rho_edge,
                    pv_edge=diagnostics.pv_edge,
                    kinetic_energy=diagnostics.kinetic_energy,
                    horizontal_divergence=horizontal_mass_divergence,
                )
            if self.source_release == V841_SOURCE_RELEASE and self.config.config_rayleigh_damp_u:
                from .upper_damping_v841 import apply_rayleigh_u
                apply_rayleigh_u(tend_ru,rho_edge,normal_velocity,
                    levels=getattr(self.config,"config_number_rayleigh_damp_u_levels",6),
                    days=getattr(self.config,"config_rayleigh_damp_u_timescale_days",5.0))
            if saved_mixing_euler is None:
                tend_ru += euler_ru
            else:
                tend_ru += saved_mixing_euler.u
            tend_rt = _rho_theta_tendency(
                self.mesh,
                current,
                saved,
                self.vertical_grid,
                self.advection_coefficients,
                rk_step=stage.stage,
                third_order_coefficient=self.config.config_coef_3rd_order,
                theta_m=current_theta_m,
                theta_m_saved=theta_saved,
                inv_area_cell=v841_inv_area_cell,
                skip_cells=(
                    None
                    if self.regional is None
                    else self.regional.masks.spec_zone_mask_cell != 0.0
                ),
            )
            if saved_mixing_euler is not None:
                tend_rt += saved_mixing_euler.theta
            tend_rw = _vertical_momentum_transport(
                self.mesh,
                current,
                self.vertical_grid,
                self.advection_coefficients,
                self.terrain_metrics,
                third_order_coefficient=self.config.config_coef_3rd_order,
                vertical_velocity=current_vertical_velocity,
                inv_area_cell=v841_inv_area_cell,
                skip_cells=(
                    None
                    if self.regional is None
                    else self.regional.masks.spec_zone_mask_cell != 0.0
                ),
            )
            if saved_mixing_euler is None:
                tend_rw += euler_rw
            else:
                tend_rw += saved_mixing_euler.w
            tend_omega = convert_w_tendency_to_omega(
                self.mesh,
                tend_rw,
                tend_ru,
                fzm=self.vertical_grid.fzm,
                fzp=self.vertical_grid.fzp,
                zz=self.vertical_grid.zz,
                zb_cell=zb_cell,
                zb3_cell=zb3_cell,
            )
            if self.regional is not None:
                runtime = self.regional
                masks = runtime.masks
                # atm_srk3:2300-2351, against the shared tend_rho pool.
                adjust_dynamics_speczone_tend(
                    masks=masks,
                    tend_ru=tend_ru,
                    tend_rho=stage_tend_rho,
                    tend_rt=tend_rt,
                    tend_rw=tend_omega,
                    ru_driving_tend=runtime.driving.tendency("ru"),
                    rt_driving_tend=runtime.driving.tendency("rtheta_m"),
                    rho_driving_tend=runtime.driving.tendency("rho_zz"),
                )
                time_dyn = dynamics_time_offset(
                    outer_dt=outer_dt,
                    dynamics_split=self.config.config_dynamics_split_steps,
                    dynamics_substep=dynamics_substep,
                    rk_timestep=rk_timestep_f32(
                        outer_dt=outer_dt,
                        dynamics_split=self.config.config_dynamics_split_steps,
                        rk_step=stage.stage,
                        order=self.config.config_time_integration_order,
                    ),
                )
                adjust_dynamics_relaxzone_tend(
                    self.mesh,
                    masks=masks,
                    mesh_scaling_regional_cell=runtime.mesh_scaling_cell,
                    mesh_scaling_regional_edge=runtime.mesh_scaling_edge,
                    config_relax_zone_divdamp_coef=(
                        runtime.config_relax_zone_divdamp_coef
                    ),
                    dt=outer_dt,
                    tend_ru=tend_ru,
                    tend_rho=stage_tend_rho,
                    tend_rt=tend_rt,
                    ru=current.rho_u,
                    theta_m=current_theta_m,
                    rho_zz=current.rho,
                    ru_driving_values=runtime.driving.state_at(
                        "ru", runtime.clock, time_dyn
                    ),
                    rt_driving_values=runtime.driving.state_at(
                        "rtheta_m", runtime.clock, time_dyn
                    ),
                    rho_driving_values=runtime.driving.state_at(
                        "rho_zz", runtime.clock, time_dyn
                    ),
                )
            acoustic = AcousticStepState(
                ru_p=np.zeros_like(current.rho_u),
                rw_p=np.zeros_like(current.rho_w),
                rtheta_pp=np.zeros_like(current.rho_theta),
                rtheta_pp_old=np.zeros_like(current.rho_theta),
                rho_pp=np.zeros_like(current.rho),
                ru_avg=np.zeros_like(current.rho_u),
                ww_avg=np.zeros_like(current.rho_w),
            )
            forcing = AcousticStepForcing(
                rho_zz=current.rho,
                # Acoustic source binds theta_m at state time level 1 and the
                # diagnostic Exner that is not refreshed until RK stage 3
                # (F:2180-2192, F:2819-2836).
                theta_m=theta_saved,
                zz=self.vertical_grid.zz,
                exner=exner_saved,
                cqu=cqu,
                zxu=self.vertical_grid.zxu,
                dss=self.damping_coefficients,
                tend_ru=tend_ru,
                tend_rho=stage_tend_rho,
                tend_rt=tend_rt,
                tend_rw=tend_omega,
                w=current_vertical_velocity,
                rw=current.rho_w,
                rw_save=saved.rho_w,
            )
            for small_step in range(1, stage.acoustic_steps + 1):
                if self.config.config_divergence_damping:
                    acoustic.rtheta_pp_old = capture_rtheta_pp_old(
                        acoustic.rtheta_pp,
                        small_step=small_step,
                    )
                if self.source_release == V841_SOURCE_RELEASE:
                    if self.acoustic_offcentering is None:
                        raise RuntimeError(
                            "v8.4.1 acoustic off-centering was not initialized"
                        )
                    acoustic = advance_acoustic_step_v841(
                        self.mesh,
                        acoustic,
                        forcing,
                        coefficients,
                        dts=stage.acoustic_timestep,
                        small_step=small_step,
                        offcentering=self.acoustic_offcentering,
                        fzm=self.vertical_grid.fzm,
                        fzp=self.vertical_grid.fzp,
                        rdzw=self.vertical_grid.rdzw,
                        gravity=self.gravity,
                        rgas=self.rgas,
                        cp=self.cp,
                        specified_zone_edge=(
                            None
                            if self.regional is None
                            else self.regional.masks.spec_zone_mask_edge
                        ),
                        specified_zone_cell=(
                            None
                            if self.regional is None
                            else self.regional.masks.spec_zone_mask_cell
                        ),
                    )
                else:
                    acoustic = advance_acoustic_step(
                        self.mesh,
                        acoustic,
                        forcing,
                        coefficients,
                        dts=stage.acoustic_timestep,
                        small_step=small_step,
                        epssm=self.config.config_epssm,
                        fzm=self.vertical_grid.fzm,
                        fzp=self.vertical_grid.fzp,
                        rdzw=self.vertical_grid.rdzw,
                        gravity=self.gravity,
                        rgas=self.rgas,
                        cp=self.cp,
                    )
                if self.config.config_divergence_damping:
                    if self.regional is None:
                        acoustic.ru_p = divergence_damping_3d(
                            self.mesh,
                            acoustic.ru_p,
                            theta_saved,
                            acoustic.rtheta_pp,
                            acoustic.rtheta_pp_old,
                            native_stored_inverse=(self.source_release == V841_SOURCE_RELEASE),
                            dts=stage.acoustic_timestep,
                            config_smdiv=self.config.config_smdiv,
                            config_len_disp=self.config.config_len_disp,
                        )
                    else:
                        # atm_divergence_damping_3d F:4183: the update is
                        # scaled by (1 - specZoneMaskEdge); gathers on
                        # one-cell ring-7 edges read the zeroed garbage
                        # column exactly as native reads theta_m/rtheta_pp
                        # at the garbage cell.
                        shim = _RegionalEdgeShim(
                            self.regional.cells_on_edge_remapped,
                            self.mesh,
                        )
                        acoustic.ru_p = divergence_damping_3d(
                            shim,
                            acoustic.ru_p,
                            pad_cells_column(theta_saved, 0.0),
                            pad_cells_column(acoustic.rtheta_pp, 0.0),
                            pad_cells_column(acoustic.rtheta_pp_old, 0.0),
                            native_stored_inverse=(self.source_release == V841_SOURCE_RELEASE),
                            dts=stage.acoustic_timestep,
                            config_smdiv=self.config.config_smdiv,
                            config_len_disp=self.config.config_len_disp,
                            spec_zone_mask_edge=(
                                self.regional.masks.spec_zone_mask_edge
                            ),
                            n_cells_solve=self.ncells,
                        )
            candidate_density_perturbation = (
                density_perturbation_saved + acoustic.rho_pp
            )
            candidate_rtheta_perturbation = (
                rho_theta_perturbation_saved + acoustic.rtheta_pp
            )
            candidate = PrognosticState(
                rho=(candidate_density_perturbation + self.reference.rho_base),
                rho_theta=(
                    candidate_rtheta_perturbation
                    + self.reference.rho_theta_base
                ),
                rho_u=saved.rho_u + acoustic.ru_p,
                rho_w=saved.rho_w + acoustic.rw_p,
                scalars=current.scalars.copy(),
                time_seconds=saved.time_seconds,
            )
            if self.source_release == V841_SOURCE_RELEASE:
                enforce_recovered_rw_endpoints_v841(candidate.rho_w)
            else:
                candidate.rho_w[0] = 0.0
                candidate.rho_w[-1] = 0.0
            regional_u_overwrite = None
            if self.regional is not None:
                # atm_srk3:2442-2485: recover "will not have set outermost
                # edge velocities correctly"; u and ru in the specified zone
                # take the interpolated driving states at this stage's
                # dynamics time.
                regional_u_overwrite = (
                    self.regional.driving.state_at(
                        "u", self.regional.clock, time_dyn
                    ),
                    self.regional.driving.state_at(
                        "ru", self.regional.clock, time_dyn
                    ),
                )
                candidate.rho_u[
                    :, self.regional.masks.spec_edges
                ] = regional_u_overwrite[1][:, self.regional.masks.spec_edges]
            if np.any(candidate.rho <= 0.0):
                raise FloatingPointError(
                    f"rho became non-positive in RK stage {stage.stage}; reduce config_dt"
                )
            if np.any(candidate.rho_theta <= 0.0):
                raise FloatingPointError(
                    f"rho_theta became non-positive in RK stage {stage.stage}; reduce config_dt"
                )
            flux_dtype = candidate.rho.dtype
            inverse_steps = flux_dtype.type(1.0) / flux_dtype.type(stage.acoustic_steps)
            flux_u = saved.rho_u + acoustic.ru_avg * inverse_steps
            flux_w = saved.rho_w + acoustic.ww_avg * inverse_steps
            if stage.stage == 3:
                final_flux_u = flux_u.copy()
                final_flux_w = flux_w.copy()
            if (
                self.config.config_scalar_advection
                and not self.config.config_split_dynamics_transport
                and candidate.scalars.shape[0] > 0
            ):
                transported = advance_scalar_transport(
                    self.mesh,
                    saved.scalars,
                    current.scalars,
                    saved.rho,
                    candidate.rho,
                    flux_u,
                    flux_w,
                    stage.large_timestep,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=False,
                    config_time_integration_order=self.config.config_time_integration_order,
                    coefficients=self.advection_coefficients,
                    fzm=self.vertical_grid.fzm,
                    fzp=self.vertical_grid.fzp,
                    rdzw=self.vertical_grid.rdzw,
                    config_scalar_adv_order=self.config.config_scalar_adv_order,
                    config_scalar_vadv_order=self.config.config_scalar_vadv_order,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    config_apply_lbcs=False,
                    inv_area_cell=v841_inv_area_cell,
                )
                candidate.scalars = np.asarray(transported.scalars)
            elif not self.config.config_scalar_advection:
                candidate.scalars = saved.scalars.copy()
            elif self.config.config_split_dynamics_transport:
                candidate.scalars = saved.scalars.copy()
            self._validate_state(candidate)
            current = candidate
            current_density_perturbation = candidate_density_perturbation
            current_rtheta_perturbation = candidate_rtheta_perturbation
            current_theta_m = current.rho_theta / current.rho
            if stage.stage == 3:
                (
                    current_pressure_perturbation,
                    current_exner,
                    current_theta_m,
                ) = _pressure_perturbation(
                    current,
                    self.reference,
                    self.vertical_grid,
                    rgas=self.rgas,
                    cp=self.cp,
                    reference_pressure=self.reference_pressure,
                    native_gnu_powf=self.source_release == V841_SOURCE_RELEASE,
                    rho_theta_perturbation=current_rtheta_perturbation,
                )
            current_normal_velocity = self._recover_normal_velocity(current)
            if regional_u_overwrite is not None:
                current_normal_velocity[
                    :, self.regional.masks.spec_edges
                ] = regional_u_overwrite[0][:, self.regional.masks.spec_edges]
            current_vertical_velocity = self._recover_vertical_velocity(current)
            # Frozen lines 1088-1105 retain tangential velocity on stages one
            # and two and reconstruct it after stage three.
            diagnostics = self._diagnostics(
                current,
                # APVM at frozen F:5786 multiplies by the outer timestep even
                # though this state belongs to a shorter dynamics subcycle.
                outer_dt=outer_dt,
                cached_v=cached_v,
                rk_step=stage.stage,
                normal_velocity=current_normal_velocity,
            )
            cached_v = diagnostics.tangential_velocity

        next_diagnostics = DrySavedDiagnostics(
            theta_m=current_theta_m.copy(),
            exner=current_exner.copy(),
            density_perturbation=current_density_perturbation.copy(),
            rho_theta_perturbation=current_rtheta_perturbation.copy(),
            pressure_perturbation=current_pressure_perturbation.copy(),
            normal_velocity=current_normal_velocity.copy(),
            vertical_velocity=current_vertical_velocity.copy(),
        )
        next_diagnostics.validate(current.rho.shape, current.rho.dtype, self.nedges)
        if final_flux_u is None or final_flux_w is None:
            raise RuntimeError("the dynamics RK subcycle did not reach stage three")
        return _DynamicsSubcycleResult(
            state=current,
            saved_diagnostics=next_diagnostics,
            diagnostics=diagnostics,
            mass_flux_u=final_flux_u,
            mass_flux_w=final_flux_w,
        )

    def step(
        self,
        state: PrognosticState,
        *,
        saved_diagnostics: DrySavedDiagnostics | None = None,
    ) -> DryStepResult:
        """Advance one complete outer step with native dynamics subcycling."""

        self._validate_state(state)
        outer_saved = state.copy()
        if saved_diagnostics is None:
            time_level_one = self._rebuild_saved_diagnostics(outer_saved)
        else:
            saved_diagnostics.validate(
                outer_saved.rho.shape,
                outer_saved.rho.dtype,
                self.nedges,
            )
            time_level_one = DrySavedDiagnostics(
                theta_m=np.asarray(saved_diagnostics.theta_m).copy(),
                exner=np.asarray(saved_diagnostics.exner).copy(),
                density_perturbation=np.asarray(
                    saved_diagnostics.density_perturbation
                ).copy(),
                rho_theta_perturbation=np.asarray(
                    saved_diagnostics.rho_theta_perturbation
                ).copy(),
                pressure_perturbation=np.asarray(
                    saved_diagnostics.pressure_perturbation
                ).copy(),
                normal_velocity=np.asarray(
                    saved_diagnostics.normal_velocity
                ).copy(),
                vertical_velocity=np.asarray(
                    saved_diagnostics.vertical_velocity
                ).copy(),
            )

        moist_cqu: FloatArray | None = None
        moist_cqw: FloatArray | None = None
        moist_qtot: FloatArray | None = None
        if self.regional is not None:
            # mpas_atm_core.F:735-781: the lbc_in stream is read (and the
            # boundary tendencies re-formed) at the start of any step whose
            # clock reached an interval end, before any dynamics.
            self.regional.ensure_interval()
        if self.config.config_moist_physics:
            assert self.index_qv is not None
            if not 0 <= self.index_qv < outer_saved.scalars.shape[0]:
                _refuse(
                    "index_qv",
                    self.index_qv,
                    "the declared qv slot is outside the scalar array",
                    f"0 <= index_qv < {outer_saved.scalars.shape[0]}",
                )
            assert self.regional is not None
            moist_qtot, moist_cqw, moist_cqu = compute_moist_coefficients(
                outer_saved.scalars,
                moist_indices=(self.index_qv,),
                cells_on_edge_remapped=self.regional.cells_on_edge_remapped,
                nlev=self.nlev,
            )
        before = self.metrics(outer_saved)
        outer_dt = self.config.config_dt
        dynamics_splits = self.config.config_dynamics_split_steps
        dynamics_schedule = RKSchedule.from_mpas(
            outer_dt,
            order=self.config.config_time_integration_order,
            acoustic_substeps=self.config.config_number_of_sub_steps,
            dynamics_splits=dynamics_splits,
        )
        scalar_schedule = RKSchedule.from_mpas(
            outer_dt,
            order=self.config.config_time_integration_order,
            acoustic_substeps=self.config.config_number_of_sub_steps,
            dynamics_splits=1,
        )
        cells = _mesh_array(self.mesh, "cellsOnEdge").astype(
            np.int64, copy=False
        )
        max_edges = _mesh_array(self.mesh, "edgesOnCell").shape[1]
        zero_zb = np.zeros(
            (self.nlev + 1, self.ncells, max_edges), dtype=state.rho.dtype
        )
        zb_cell = (
            zero_zb
            if self.terrain_metrics is None
            else np.asarray(self.terrain_metrics.zb_cell, dtype=state.rho.dtype)
        )
        zb3_cell = (
            zero_zb
            if self.terrain_metrics is None
            else np.asarray(self.terrain_metrics.zb3_cell, dtype=state.rho.dtype)
        )

        diagnostics = self._diagnostics(
            outer_saved,
            outer_dt=outer_dt,
            cached_v=None,
            rk_step=3,
            normal_velocity=time_level_one.normal_velocity,
        )
        current = outer_saved
        carried_diagnostics = time_level_one
        split_flux_u_sum: FloatArray | None = None
        split_flux_w_sum: FloatArray | None = None
        for _dynamics_substep in range(1, dynamics_splits + 1):
            subcycle = self._advance_dynamics_subcycle(
                current,
                time_level_one=carried_diagnostics,
                initial_diagnostics=diagnostics,
                schedule=dynamics_schedule,
                outer_dt=outer_dt,
                cells=cells,
                zb_cell=zb_cell,
                zb3_cell=zb3_cell,
                dynamics_substep=_dynamics_substep,
                moist_cqu=moist_cqu,
                moist_cqw=moist_cqw,
                moist_qtot=moist_qtot,
            )
            current = subcycle.state
            carried_diagnostics = subcycle.saved_diagnostics
            diagnostics = subcycle.diagnostics
            split_flux_u_sum = accumulate_split_flux(
                subcycle.mass_flux_u, split_flux_u_sum
            )
            split_flux_w_sum = accumulate_split_flux(
                subcycle.mass_flux_w, split_flux_w_sum
            )

        if split_flux_u_sum is None or split_flux_w_sum is None:
            raise RuntimeError("the dynamics subcycle loop did not execute")
        split_flux_u = finish_split_flux(split_flux_u_sum, dynamics_splits)
        split_flux_w = finish_split_flux(split_flux_w_sum, dynamics_splits)

        v841_inv_area_cell = None
        if self.source_release == V841_SOURCE_RELEASE:
            v841_inv_area_cell = precomputed_mesh_inverse_v841(
                self.mesh, "areaCell", outer_saved.rho.dtype
            )

        scalar_stage_timesteps: tuple[float, float, float] | None = None
        if (
            self.config.config_scalar_advection
            and self.config.config_split_dynamics_transport
            and current.scalars.shape[0] > 0
        ):
            # Frozen F:1193-1268 performs one outer scalar RK after all dry
            # subcycles.  Its t0 density is deliberately the outer state, not
            # the final subcycle's time-level-one density.
            scalar_stage = outer_saved.scalars.copy()
            for stage in scalar_schedule.stages:
                transported = advance_scalar_transport(
                    self.mesh,
                    outer_saved.scalars,
                    scalar_stage,
                    outer_saved.rho,
                    current.rho,
                    split_flux_u,
                    split_flux_w,
                    stage.large_timestep,
                    rk_step=stage.stage,
                    config_scalar_advection=True,
                    config_monotonic=self.config.config_monotonic,
                    config_positive_definite=self.config.config_positive_definite,
                    config_split_dynamics_transport=True,
                    config_time_integration_order=self.config.config_time_integration_order,
                    coefficients=self.advection_coefficients,
                    fzm=self.vertical_grid.fzm,
                    fzp=self.vertical_grid.fzp,
                    rdzw=self.vertical_grid.rdzw,
                    config_scalar_adv_order=self.config.config_scalar_adv_order,
                    config_scalar_vadv_order=self.config.config_scalar_vadv_order,
                    config_coef_3rd_order=self.config.config_coef_3rd_order,
                    config_apply_lbcs=self.regional is not None,
                    bdy_mask_cell=(
                        None
                        if self.regional is None
                        else self.regional.masks.bdy_mask_cell
                    ),
                    bdy_mask_edge=(
                        None
                        if self.regional is None
                        else self.regional.masks.bdy_mask_edge
                    ),
                    inv_area_cell=v841_inv_area_cell,
                )
                scalar_stage = np.asarray(transported.scalars)
                if self.regional is not None:
                    # atm_srk3:2688-2717: after every split-transport RK
                    # stage the relaxation zone is filtered toward -- and
                    # the specified zone set to -- the driving scalars at
                    # this stage's transport time.
                    dt_rk = transport_rk_timestep_f32(
                        outer_dt=outer_dt, rk_step=stage.stage,
                        order=self.config.config_time_integration_order,
                    )
                    bdy_adjust_scalars(
                        self.mesh,
                        masks=self.regional.masks,
                        mesh_scaling_regional_cell=(
                            self.regional.mesh_scaling_cell
                        ),
                        scalars_new=scalar_stage,
                        scalars_driving=self.regional.driving.state_at(
                            "scalars", self.regional.clock, dt_rk
                        ),
                        dt=outer_dt,
                        dt_rk=dt_rk,
                    )
            current.scalars = scalar_stage
            scalar_stage_timesteps = tuple(
                stage.large_timestep for stage in scalar_schedule.stages
            )  # type: ignore[assignment]

        if self.config.config_moist_physics:
            # atm_srk3:2798-2800: DO_PHYSICS builds clamp negative scalar
            # mixing ratios at the end of every step regardless of the
            # physics suite; the pinned dry reference arms ran such a build.
            clamp_negative_scalars(current.scalars)
        if self.regional is not None:
            runtime = self.regional
            masks = runtime.masks
            dt_f32 = np.float32(outer_dt)
            # atm_srk3:2828-2849: reset specified-zone theta_m/rtheta_p to
            # the driving state at the end of the full timestep.
            rt_values = runtime.driving.state_at(
                "rtheta_m", runtime.clock, dt_f32
            )
            rho_values = runtime.driving.state_at(
                "rho_zz", runtime.clock, dt_f32
            )
            reset_speczone_values(
                masks=masks,
                theta_m=carried_diagnostics.theta_m,
                rho_theta=current.rho_theta,
                rt_driving_values=rt_values,
                rho_driving_values=rho_values,
            )
            # F:8238: rtheta_p := rt_driving - rtheta_base.  The carried
            # perturbation seeds the next step's rtheta_pp accumulation
            # (rtheta_p_save), so it takes the native subtraction, not a
            # re-derivation from the state sum.
            spec = masks.spec_cells
            carried_diagnostics.rho_theta_perturbation[:, spec] = (
                rt_values[:, spec]
                - self.reference.rho_theta_base[:, spec]
            )
            # atm_srk3:2852-2878: specified-zone scalars take the driving
            # values at the end of the full timestep.
            bdy_set_scalars(
                masks=masks,
                scalars_new=current.scalars,
                scalars_driving=runtime.driving.state_at(
                    "scalars", runtime.clock, dt_f32
                ),
            )
            runtime.advance_clock(outer_dt)
        current.time_seconds = outer_saved.time_seconds + outer_dt
        after = self.metrics(current)
        dynamics_stage_timesteps = tuple(
            stage.large_timestep for stage in dynamics_schedule.stages
        )
        receipt = StepReceipt(
            evidence=self.evidence,
            frozen_source=self.frozen_source,
            source_release=self.source_release,
            start_time_seconds=outer_saved.time_seconds,
            end_time_seconds=current.time_seconds,
            stage_acoustic_steps=tuple(
                stage.acoustic_steps for stage in dynamics_schedule.stages
            ),  # type: ignore[arg-type]
            dynamics_split_steps=dynamics_splits,
            dynamics_timestep_seconds=outer_dt / dynamics_splits,
            dynamics_stage_timesteps=dynamics_stage_timesteps,  # type: ignore[arg-type]
            scalar_transport_stage_timesteps=scalar_stage_timesteps,
            split_flux_reduction=SPLIT_FLUX_REDUCTION,
            before=before,
            after=after,
            mass_relative_drift=_relative_change(after.mass, before.mass),
            energy_relative_drift=_relative_change(
                after.energy_proxy, before.energy_proxy
            ),
        )
        return DryStepResult(
            state=current,
            receipt=receipt,
            saved_diagnostics=carried_diagnostics,
        )

    def run(
        self,
        state: PrognosticState,
        steps: int,
        *,
        bounds: StabilityBounds | None = None,
        saved_diagnostics: DrySavedDiagnostics | None = None,
    ) -> DryRunResult:
        """Run N complete steps, failing immediately on a named stability bound."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        policy = StabilityBounds() if bounds is None else bounds
        policy.validate()
        initial = self.metrics(state)
        current = state.copy()
        max_mass_drift = 0.0
        max_energy_drift = 0.0
        max_velocity = initial.max_abs_velocity
        current_diagnostics = saved_diagnostics
        if current_diagnostics is not None:
            current_diagnostics.validate(
                current.rho.shape,
                current.rho.dtype,
                self.nedges,
            )
        for index in range(steps):
            result = self.step(current, saved_diagnostics=current_diagnostics)
            current = result.state
            current_diagnostics = result.saved_diagnostics
            metrics = result.receipt.after
            mass_drift = _relative_change(metrics.mass, initial.mass)
            energy_drift = _relative_change(metrics.energy_proxy, initial.energy_proxy)
            max_mass_drift = max(max_mass_drift, mass_drift)
            max_energy_drift = max(max_energy_drift, energy_drift)
            max_velocity = max(max_velocity, metrics.max_abs_velocity)
            if not metrics.all_finite:
                raise FloatingPointError(f"non-finite state after dry step {index + 1}")
            if metrics.min_density < policy.min_density:
                raise FloatingPointError(
                    f"min_density bound failed after step {index + 1}: "
                    f"{metrics.min_density} < {policy.min_density}"
                )
            if mass_drift > policy.max_mass_relative_drift:
                raise FloatingPointError(
                    f"max_mass_relative_drift failed after step {index + 1}: "
                    f"{mass_drift} > {policy.max_mass_relative_drift}"
                )
            if energy_drift > policy.max_energy_relative_drift:
                raise FloatingPointError(
                    f"max_energy_relative_drift failed after step {index + 1}: "
                    f"{energy_drift} > {policy.max_energy_relative_drift}"
                )
            if metrics.max_abs_velocity > policy.max_abs_velocity:
                raise FloatingPointError(
                    f"max_abs_velocity failed after step {index + 1}: "
                    f"{metrics.max_abs_velocity} > {policy.max_abs_velocity}"
                )
        final = self.metrics(current)
        if current_diagnostics is None:
            current_diagnostics = self._rebuild_saved_diagnostics(current)
        return DryRunResult(
            state=current,
            saved_diagnostics=current_diagnostics,
            receipt=RunReceipt(
                evidence=self.evidence,
                frozen_source=self.frozen_source,
                source_release=self.source_release,
                steps=steps,
                start_time_seconds=state.time_seconds,
                end_time_seconds=current.time_seconds,
                initial=initial,
                final=final,
                max_mass_relative_drift=max_mass_drift,
                max_energy_relative_drift=max_energy_drift,
                max_abs_velocity=max_velocity,
                bounds=policy,
            ),
        )


def advance_rk3_step(
    mesh: object,
    state: PrognosticState,
    vertical_grid: VerticalGrid,
    reference: DryReferenceState,
    config: DryDycoreConfig | None = None,
    *,
    advection_coefficients: AdvectionCoefficients | None = None,
    terrain_metrics: TerrainMetrics | None = None,
    saved_diagnostics: DrySavedDiagnostics | None = None,
    reference_wind_profiles: V841ReferenceWindProfiles | None = None,
) -> DryStepResult:
    """Stateless convenience API for one full RK3 step."""

    return DryDycoreDriver(
        mesh,
        vertical_grid,
        reference,
        config,
        advection_coefficients=advection_coefficients,
        terrain_metrics=terrain_metrics,
        reference_wind_profiles=reference_wind_profiles,
    ).step(state, saved_diagnostics=saved_diagnostics)


def run_dry_dycore(
    mesh: object,
    state: PrognosticState,
    vertical_grid: VerticalGrid,
    reference: DryReferenceState,
    config: DryDycoreConfig,
    steps: int,
    *,
    bounds: StabilityBounds | None = None,
    advection_coefficients: AdvectionCoefficients | None = None,
    terrain_metrics: TerrainMetrics | None = None,
    saved_diagnostics: DrySavedDiagnostics | None = None,
    reference_wind_profiles: V841ReferenceWindProfiles | None = None,
) -> DryRunResult:
    """Stateless N-step convenience API."""

    return DryDycoreDriver(
        mesh,
        vertical_grid,
        reference,
        config,
        advection_coefficients=advection_coefficients,
        terrain_metrics=terrain_metrics,
        reference_wind_profiles=reference_wind_profiles,
    ).run(
        state,
        steps,
        bounds=bounds,
        saved_diagnostics=saved_diagnostics,
    )


__all__ = [
    "WHOLE_STEP_EVIDENCE",
    "ORIGINAL_JW_BRANCH_EVIDENCE",
    "NATIVE_SPLIT3_IMPLEMENTATION_EVIDENCE",
    "FROZEN_SOURCE",
    "NATIVE_SPLIT3_SOURCE",
    "SPLIT_FLUX_REDUCTION",
    "DryDycoreConfig",
    "DryReferenceState",
    "DrySavedDiagnostics",
    "SyntheticDryCase",
    "TerrainMetrics",
    "NativeVerticalData",
    "StateMetrics",
    "StabilityBounds",
    "StepReceipt",
    "DryStepResult",
    "RunReceipt",
    "DryRunResult",
    "DryDycoreDriver",
    "load_mpas_vertical_grid",
    "load_mpas_initial_state",
    "make_synthetic_x1_case",
    "advance_rk3_step",
    "run_dry_dycore",
]

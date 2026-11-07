"""Runtime-free ensemble physics routing without changing a selection.

The ordinary registry remains the authority for runnable options. An option
without a qualified member launch uses its original operation per member.
These rows describe execution coverage, rather than new physics admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping

from woof.config import radiation_scheme_ids
from woof.physics_compat import RRTMG_VARIANT_LEGACY, rrtmg_variant

PHYSICS_EXECUTION_CONTRACT = "gpuwm-ensemble-physics-execution-v1"
PHYSICS_COMPONENTS = (
    "microphysics", "radiation", "surface_layer", "land_surface", "pbl", "cumulus",
)

# These are existing, qualified leaves. A leaf row does not assert that its
# surrounding driver, option combinations or dynamically changing clock have
# a qualified packed binding. The caller must supply the admitted binding.
_NATIVE_LEAVES = {
    ("microphysics", "thompson-mp8"): "woof.ensemble.batch_physics.prepare_thompson_column_batch",
    ("microphysics", "ishmael-mp55"): "woof.ensemble.batch_physics.prepare_registered_column_batch",
    ("surface_layer", "revised-mm5"): "woof.ensemble.batch_physics.prepare_sfclay_column_batch",
    ("surface_layer", "classic-mm5"): "woof.ensemble.batch_physics.prepare_sfclay_column_batch",
    ("land_surface", "noah"): "woof.ensemble.batch_physics.prepare_noah_column_batch",
    ("pbl", "ysu"): "woof.ensemble.batch_physics.prepare_ysu_column_batch",
    ("cumulus", "kain-fritsch-cu99"): "woof.core.kf.launch_kf_old",
}


def _value(settings, key, default=None):
    return settings.get(key, default) if isinstance(settings, Mapping) else getattr(settings, key, default)


@dataclass(frozen=True)
class ComponentCapability:
    component: str
    option_id: str | None
    selectors: tuple[tuple[str, object], ...]
    implemented: bool
    native_leaf: str | None = None
    parameters: tuple[tuple[str, object], ...] = ()

    def receipt(self):
        result = asdict(self)
        result["selectors"] = dict(self.selectors)
        result["parameters"] = dict(self.parameters)
        result["fallback"] = "original_member_operation"
        return result


@dataclass(frozen=True)
class SuiteExecutionPlan:
    members: int
    mode: str
    components: tuple[ComponentCapability, ...]
    native_fallback_reasons: tuple[str, ...]

    @property
    def uses_native_batch(self):
        return self.mode == "native_batch"

    def receipt(self):
        return {
            "contract": PHYSICS_EXECUTION_CONTRACT,
            "members": self.members,
            "mode": self.mode,
            "components": [row.receipt() for row in self.components],
            "native_fallback_reasons": list(self.native_fallback_reasons),
            "selection_policy": "unchanged",
            "clock_policy": "independent_original_member_clocks",
        }


def capability_matrix(*, registry=None):
    """Every implemented registry physics option, with original fallback.

    Off rows and newly registered schemes are included automatically. The
    turbulence and urban rows are also included because they affect whether
    the surrounding driver can use the packed native binding.
    """
    if registry is None:
        from woof.physics_registry import physics_registry
        registry = physics_registry()
    rows = []
    for component, declaration in sorted(registry["components"].items()):
        for option_id, option in sorted(declaration["options"].items()):
            if not option.get("implemented", False):
                continue
            rows.append(ComponentCapability(
                component, option_id, tuple(sorted((option.get("selectors") or {}).items())),
                True, _NATIVE_LEAVES.get((component, option_id))))
    return tuple(rows)


def selected_capabilities(cfg, *, registry=None):
    """Resolve metadata for the requested suite, without mutating it."""
    if registry is None:
        from woof.physics_registry import physics_registry
        registry = physics_registry()
    from types import SimpleNamespace
    lw, sw = radiation_scheme_ids(SimpleNamespace(
        ra_physics=_value(cfg, "ra_physics", 0),
        ra_lw_physics=_value(cfg, "ra_lw_physics", -1),
        ra_sw_physics=_value(cfg, "ra_sw_physics", -1)))
    effective = {"ra_lw_physics": lw, "ra_sw_physics": sw}
    rows = capability_matrix(registry=registry)
    result = []
    for component, declaration in sorted(registry["components"].items()):
        values = tuple(sorted((key, effective.get(key, _value(cfg, key, 0)))
                              for key in declaration.get("selector_keys", ())))
        row = next((row for row in rows if row.component == component and row.selectors == values), None)
        # The ordinary config door diagnoses an unknown selector. Ensemble
        # routing must neither reinterpret it nor substitute a known scheme.
        row = row if row is not None else ComponentCapability(component, None, values, False)
        if component == "radiation" and (lw, sw) == (4, 4):
            variant = rrtmg_variant(cfg)
            row = replace(row, parameters=(("ra_rrtmg_variant", variant),),
                          native_leaf="woof.ensemble.batch_physics_init.prepare_legacy_member_radiation"
                          if variant == RRTMG_VARIANT_LEGACY else None)
        result.append(row)
    return tuple(result)


def plan_suite(cfg, *, members=1, registry=None):
    """Pick the qualified packed driver or the unchanged original driver.

    This is a conservative binding decision. Falling back does not refuse an
    ensemble, disable a scheme, pin an adaptive timestep or remove a feature.
    """
    if isinstance(members, bool) or not isinstance(members, int) or members < 1:
        raise ValueError("members must be a positive integer")
    components = selected_capabilities(cfg, registry=registry)
    reasons = []
    if _value(cfg, "obs_nudge_opt", 0):
        reasons.append("observation FDDA retains independent member report QC, time windows and captured tendency graphs on the original member driver")
    if _value(cfg, "chem_sets", ""):
        reasons.append("active chemistry keeps the original member species arrays, source-hour caches and mass ledger; the native packed graph lacks chemistry producers and typed state ownership")
    for row in components:
        if not row.implemented:
            reasons.append(f"{row.component} selection has no qualified packed registry row")
    selections = {
        "mp_physics": 8, "sf_surface_physics": 2, "bl_pbl_physics": 1,
        "sf_urban_physics": 0, "sf_surface_mosaic": 0,
        "topo_wind": 0, "gwd_opt": 0, "slope_rad": 0, "topo_shading": 0,
        # WRF's swint_opt = 1 rewrites the surface shortwave from a carrier
        # the packed driver does not bank per member, and aer_opt = 3 needs
        # the aerosol-aware Thompson scheme the packed binding does not
        # run; both run on the original member driver.
        "swint_opt": 0, "aer_opt": 0,
        # These options mutate surface classes or carry a member-local
        # clock/history bank. The original CUDA driver owns those arrays;
        # the packed Noah shared-field bank cannot alias them across members.
        "sst_update": 0, "sst_skin": 0, "tmn_update": 0,
    }
    for key, required in selections.items():
        value = _value(cfg, key, 0)
        if key == "mp_physics" and value == 55:
            continue
        if value != required:
            if key == "sf_surface_physics" and value == 3:
                reasons.append("sf_surface_physics=3 lacks a qualified packed RUC state and fused land driver graph")
            elif key == "bl_pbl_physics" and value == 5:
                reasons.append("bl_pbl_physics=5 lacks a qualified packed MYNN turbulence, diffusion and TKE transport driver graph")
            else:
                reasons.append(f"{key}={value!r} has no qualified packed producer and driver composition")
    if _value(cfg, "cu_physics", 0) not in (0, 99):
        reasons.append(f"cu_physics={_value(cfg, 'cu_physics', 0)!r} has no qualified packed producer and driver composition")
    if _value(cfg, "sf_sfclay_physics", 0) not in (1, 91):
        surface = _value(cfg, "sf_sfclay_physics", 0)
        reasons.append(f"sf_sfclay_physics={surface!r} has no qualified packed surface flux and carrier driver binding")
    lw, sw = (dict(next(row for row in components if row.component == "radiation").selectors)[key]
              for key in ("ra_lw_physics", "ra_sw_physics"))
    if (lw, sw) != (4, 4) or rrtmg_variant(cfg) != RRTMG_VARIANT_LEGACY:
        reasons.append(f"radiation ({lw}, {sw}, {rrtmg_variant(cfg)!r}) has no qualified packed column, optics, flux and carrier driver binding")
    if not _value(cfg, "moist", False):
        reasons.append("the packed column initializer requires moist state")
    if _value(cfg, "use_adaptive_time_step", False):
        reasons.append("adaptive members lack a qualified packed scheduler with independent CFL reductions and integer clocks")
    if _value(cfg, "bldt", 0.0) != 0.0:
        reasons.append("positive PBL cadence retains its original held composition target")
    mode = "ordinary_single" if members == 1 else "member_local" if reasons else "native_batch"
    return SuiteExecutionPlan(members, mode, components, tuple(reasons))


def preset_capabilities(*, members=2):
    """Route all front-door fixed-template presets from their authority."""
    from woof.physics_compat import SINGLE_DOMAIN_PHYSICS_PROFILES, single_domain_runtime_switches
    return {name: plan_suite(single_domain_runtime_switches(name), members=members)
            for name in SINGLE_DOMAIN_PHYSICS_PROFILES}


__all__ = ["PHYSICS_EXECUTION_CONTRACT", "PHYSICS_COMPONENTS", "ComponentCapability",
           "SuiteExecutionPlan", "capability_matrix", "selected_capabilities", "plan_suite", "preset_capabilities"]

"""Physics route completeness, original call ordering and lossless handoff."""
from dataclasses import replace
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.ensemble.physics_execution import (
    AdmittedTendencyTarget, MemberPhysicsBinding, MemberPhysicsExecutor,
    TENDENCY_FIELDS, compatible_piece_groups,
)
from woof.ensemble.suite_capabilities import (
    capability_matrix, plan_suite, preset_capabilities, selected_capabilities,
)
from woof.physics_compat import (
    SINGLE_DOMAIN_PHYSICS_PROFILES, single_domain_runtime_switches,
)
from woof.physics_registry import physics_registry


def _cfg(**changes):
    return RunConfig(nx=8, ny=8, nz=4, dx=3000.0, dy=3000.0,
                     dt=12.0, ztop=12000.0, run_seconds=120.0,
                     moist=True, mp_physics=8, sf_surface_physics=2,
                     sf_sfclay_physics=91, bl_pbl_physics=1,
                     ra_lw_physics=4, ra_sw_physics=4,
                     ra_rrtmg_variant="rrtmg_legacy", **changes)


def test_matrix_covers_every_implemented_registry_row_and_dispatch_selector():
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    from woof.config import MP_PHYSICS_ACCEPTED
    registry = deepcopy(physics_registry())
    matrix = capability_matrix(registry=registry)
    expected = {(component, option_id) for component, item in registry["components"].items()
                for option_id, option in item["options"].items() if option.get("implemented")}
    assert {(row.component, row.option_id) for row in matrix} == expected
    for selector, dispatch in PHYSICS_SLOT_DISPATCH.items():
        if selector == "sf_ocean_physics":
            # NAMED GAP (2.9.1): the ocean mixed layer is routed by the
            # dispatch table but has no physics-registry component yet, so
            # the capability matrix has no ocean rows.  Owner: the OML port's
            # registry row (tools/build_registry.py); remove this skip with it.
            continue
        values = {dict(row.selectors)[selector] for row in matrix if selector in dict(row.selectors)}
        assert values == set(dispatch)
    assert {dict(row.selectors)["mp_physics"] for row in matrix if row.component == "microphysics"} == {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 24, 26, 27, 28, 29, 30, 38, 40, 50, 51, 52, 53,
        55, 56, 95, 96, 97, 106, 900}
    assert {dict(row.selectors)["cu_physics"] for row in matrix if row.component == "cumulus"} == {0, 1, 2, 3, 5, 6, 7, 10, 11, 14, 16, 93, 94, 95, 96, 99}
    assert all(row.receipt()["fallback"] == "original_member_operation" for row in matrix)


def test_all_front_door_presets_route_without_changing_their_switches():
    plans = preset_capabilities(members=20)
    assert set(plans) == set(SINGLE_DOMAIN_PHYSICS_PROFILES)
    for preset, plan in plans.items():
        switches = single_domain_runtime_switches(preset)
        for row in plan.components:
            if row.component != "radiation":
                for key, value in row.selectors:
                    assert value == switches.get(key, 0)
        assert plan.mode in ("native_batch", "member_local")
        assert plan.receipt()["selection_policy"] == "unchanged"
        assert plan.receipt()["clock_policy"] == "independent_original_member_clocks"
    native = {name for name, plan in plans.items() if plan.uses_native_batch}
    assert native == {"thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"}


@pytest.mark.parametrize("changes", [
    {"mp_physics": 50}, {"mp_physics": 28}, {"mp_physics": 1},
    {"sf_surface_physics": 3}, {"sf_surface_physics": 4},
    {"bl_pbl_physics": 9}, {"sf_sfclay_physics": 5},
    {"cu_physics": 1}, {"cu_physics": 2}, {"cu_physics": 3}, {"cu_physics": 16}, {"cu_physics": 95},
    {"cu_physics": 93},
    {"ra_rrtmg_variant": "rte-rrtmgp"}, {"ra_lw_physics": 1, "ra_sw_physics": 1},
    {"ra_lw_physics": 90, "ra_sw_physics": 90},
    {"use_adaptive_time_step": True}, {"bldt": 5.0},
    {"sf_surface_mosaic": 1}, {"sf_urban_physics": 1},
    {"topo_wind": 1}, {"gwd_opt": 3}, {"slope_rad": 1},
    {"swint_opt": 1}, {"aer_opt": 3},
])
def test_unqualified_piece_uses_original_route_and_preserves_config(changes):
    cfg = replace(_cfg(), **changes)
    before = vars(cfg).copy()
    plan = plan_suite(cfg, members=10)
    assert plan.mode == "member_local"
    assert plan.native_fallback_reasons
    assert vars(cfg) == before


def test_n1_keeps_original_and_unknown_registered_options_get_original_fallback():
    assert plan_suite(_cfg(), members=1).mode == "ordinary_single"
    assert plan_suite(_cfg(), members=10).mode == "native_batch"
    registry = deepcopy(physics_registry())
    registry["components"]["microphysics"]["options"]["new-mp"] = {
        "selectors": {"mp_physics": 1001}, "implemented": True}
    cfg = replace(_cfg(), mp_physics=1001)
    plan = plan_suite(cfg, members=10, registry=registry)
    mp = next(row for row in plan.components if row.component == "microphysics")
    assert mp.option_id == "new-mp"
    assert mp.native_leaf is None and plan.mode == "member_local"


def test_resolved_radiation_spelling_has_the_same_plan():
    cfg = _cfg()
    aggregate = replace(cfg, ra_lw_physics=-1, ra_sw_physics=-1, ra_physics=4)
    assert selected_capabilities(cfg) == selected_capabilities(aggregate)
    assert plan_suite(aggregate, members=10).uses_native_batch
    radiation = next(row for row in selected_capabilities(cfg) if row.component == "radiation")
    assert radiation.receipt()["parameters"] == {"ra_rrtmg_variant": "rrtmg_legacy"}
    assert radiation.native_leaf.endswith("prepare_legacy_member_radiation")


class _Driver:
    def __init__(self, state, member, calls):
        self.state, self.member, self.calls = state, member, calls
        self.result = object()

    def compute(self, state, cfg):
        assert state is self.state
        self.calls.append(("compute", self.member, cfg.dt, state.elapsed_seconds))
        return self.result

    def accept_microphysics(self, result, *, dt):
        self.calls.append(("accept", self.member, result, dt))

    def finish_step(self):
        self.calls.append(("finish", self.member))


def _executor(count=3, *, cfg=None, microphysics_apply=None):
    calls = []
    bindings = []
    for member in range(count):
        state = SimpleNamespace(elapsed_seconds=Fraction(12, 1), domain_start_offset=0.0)
        driver = _Driver(state, member, calls)
        bindings.append(MemberPhysicsBinding(state, replace(cfg or _cfg()), driver, member))
    return MemberPhysicsExecutor(bindings, microphysics_apply=microphysics_apply), calls


def test_original_compute_objects_and_live_independent_configs_are_preserved():
    executor, calls = _executor()
    assert executor.compute() == tuple(binding.driver.result for binding in executor.bindings)
    executor.bindings[1].cfg = replace(executor.bindings[1].cfg, dt=Fraction(9, 2))
    executor.bindings[1].state.elapsed_seconds = Fraction(57, 2)
    executor.bindings[2].config_provider = lambda: replace(_cfg(), dt=Fraction(7, 2))
    executor.compute()
    assert calls[-3:] == [("compute", 0, 12.0, Fraction(12)),
                          ("compute", 1, Fraction(9, 2), Fraction(57, 2)),
                          ("compute", 2, Fraction(7, 2), Fraction(12))]
    assert executor.receipt()["clock_policy"] == "independent_original_member_clocks"
    assert executor.receipt()["call_counts"]["compute"] == [2, 2, 2]


def test_n1_packed_delivery_is_the_original_object_without_copier():
    executor, calls = _executor(1)
    assert executor.compute_packed() is executor.bindings[0].driver.result
    assert len(calls) == 1


def test_microphysics_call_and_accept_use_each_actual_step_and_due_flag():
    seen = []
    def original(state, cfg, dt, *, refl_10cm_due):
        seen.append((state, cfg.mp_physics, dt, refl_10cm_due))
        return cfg.mp_physics
    executor, calls = _executor(microphysics_apply=original)
    for binding, mp, dt in zip(executor.bindings, (8, 18, 50), (12.0, 6.0, 3.0)):
        binding.cfg = replace(binding.cfg, mp_physics=mp, dt=dt)
    assert executor.apply_microphysics(refl_10cm_due=(True, False, True)) == (8, 18, 50)
    assert [(mp, dt, due) for state, mp, dt, due in seen] == [(8, 12.0, True), (18, 6.0, False), (50, 3.0, True)]
    assert calls == [("accept", 0, 8, 12.0), ("accept", 1, 18, 6.0), ("accept", 2, 50, 3.0)]
    executor.finish_step()
    assert calls[-3:] == [("finish", 0), ("finish", 1), ("finish", 2)]


def test_disabled_driver_and_microphysics_are_original_noops():
    cfg = replace(_cfg(), mp_physics=0, sf_surface_physics=0, sf_sfclay_physics=0,
                  bl_pbl_physics=0, ra_lw_physics=0, ra_sw_physics=0)
    executor, calls = _executor(cfg=cfg)
    assert executor.compute() == (None, None, None)
    assert executor.apply_microphysics() == (None, None, None)
    assert calls == []


def test_ozone_only_nested_consumer_keeps_original_cadence_without_held_forcing():
    cfg = replace(_cfg(), mp_physics=0, sf_surface_physics=0, sf_sfclay_physics=0,
                  bl_pbl_physics=0, ra_lw_physics=0, ra_sw_physics=0)
    executor, calls = _executor(1, cfg=cfg)
    executor.bindings[0].driver.cam_ozone = object()
    assert executor.compute() == (None,)
    assert calls == [("compute", 0, 12.0, Fraction(12))]


def test_shared_state_or_driver_is_rejected_before_physics_mutation():
    executor, _calls = _executor(2)
    a, b = executor.bindings
    with pytest.raises(ValueError, match="member state"):
        MemberPhysicsExecutor((a, replace(b, driver=a.driver)))
    with pytest.raises(ValueError, match="cannot be shared"):
        MemberPhysicsExecutor((a, replace(a, member_id=1)))


def test_piece_groups_preserve_independent_clock_and_original_fallback_order():
    executor, _calls = _executor(4)
    executor.bindings[1].state.elapsed_seconds = 18.0
    executor.bindings[3].cfg = replace(executor.bindings[3].cfg, mp_physics=18)
    assert compatible_piece_groups(executor.bindings, "microphysics") == ((0, 2), (1,), (3,))
    seen = []
    assert executor.run_piece("microphysics", lambda binding: seen.append(binding.member_id) or binding.member_id) == (0, 1, 2, 3)
    assert seen == [0, 1, 2, 3]
    seen.clear()
    def native(indices, bindings):
        seen.append(("batch", indices))
        return tuple(binding.member_id + 10 for binding in bindings)
    result = executor.run_piece("microphysics", lambda binding: seen.append(binding.member_id) or binding.member_id,
                                admitted_batch=native)
    assert result == (10, 1, 12, 3)
    assert seen == [("batch", (0, 2)), 1, 3]
    seen.clear()
    result = executor.run_piece("microphysics", lambda binding: seen.append(binding.member_id) or binding.member_id,
                                admitted_batch=native, batch_admission=lambda indices, bindings: False)
    assert result == (0, 1, 2, 3) and seen == [0, 1, 2, 3]


def test_a_mutating_native_failure_is_not_retried_as_original_physics():
    executor, _calls = _executor(2)
    seen = []
    def native(indices, bindings):
        seen.append("native")
        raise RuntimeError("failure after mutation")
    with pytest.raises(RuntimeError, match="after mutation"):
        executor.run_piece("microphysics", lambda binding: seen.append("original"), admitted_batch=native)
    assert seen == ["native"]


def _held(member):
    fields = {}
    words = np.array([0, 0x80000000, 0x7FC00123, 0x7F800000, 0x3F800000, 0xBF800000], dtype=np.uint32)
    for name in TENDENCY_FIELDS:
        levels, ny, nx = (3 if name == "rw" else 2), (4 if name == "rv" else 3), (5 if name == "ru" else 4)
        raw = np.resize(words, levels * ny * nx).reshape(levels, ny, nx).copy()
        raw[-1, -1, -1] = np.uint32(0x3F800000 + member)
        fields[name] = raw.view(np.float32)
    fields["extra_scalars"] = {"ni": fields["rqv"].copy()}
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("layout", ["outermost", "column"])
def test_admitted_copy_preserves_every_float_word_and_scalar_member(layout):
    sources = tuple(_held(member) for member in range(4))
    def allocate(source):
        nz, ny, nx = source.shape
        return np.zeros((4, nz, ny, nx) if layout == "outermost" else (nz, 4 * ny, nx), np.float32)
    arrays = {name: allocate(getattr(sources[0], name)) for name in TENDENCY_FIELDS}
    extras = {"ni": allocate(sources[0].extra_scalars["ni"])}
    target = AdmittedTendencyTarget(arrays, members=4, layout=layout, extra_scalars=extras, array_module=np)
    result = target(sources)
    for member, held in enumerate(sources):
        for name in TENDENCY_FIELDS:
            assert np.array_equal(target._view(getattr(result, name), member).view(np.uint32), getattr(held, name).view(np.uint32))
        assert np.array_equal(target._view(result.extra_scalars["ni"], member).view(np.uint32), held.extra_scalars["ni"].view(np.uint32))


def test_target_rejects_a_missing_category_before_any_copy():
    sources = (_held(0), _held(1))
    arrays = {name: np.full((2, *getattr(sources[0], name).shape), 7.0, np.float32) for name in TENDENCY_FIELDS}
    extras = {"ni": np.full((2, *sources[0].rqv.shape), 7.0, np.float32)}
    target = AdmittedTendencyTarget(arrays, members=2, extra_scalars=extras, array_module=np)
    sources[1].rw = None
    with pytest.raises(ValueError, match="admitted categories"):
        target(sources)
    assert all(np.all(array == 7.0) for array in arrays.values())
    assert np.all(extras["ni"] == 7.0)

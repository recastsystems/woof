"""Automatic initialization handoff, exact packing and fresh wave lifetimes."""
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, replace
from types import SimpleNamespace
from contextlib import nullcontext
import hashlib
import json

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core.clock import DomainClock, DomainTicks
from woof.core.physics import PhysicsDriver
from woof.core.state import DomainState
from woof.ensemble.admission import MemoryComponent, AllocatorMargin, EnsembleMemoryModel
from woof.ensemble.batch_products import default_product_requests, DEFAULT_THRESHOLDS
from woof.ensemble.batch_product_output import headline_diagnostic_memory_plan
from woof.ensemble.packing import CardBudget
from woof.ensemble.prepared_execution import (
    InitializedCardEvidence, InitializedCardBootstrap, WarmInitializedRoot,
    native_output_memory_components, plan_initialized_member_execution,
    execute_initialized_member_execution, make_automatic_prepared_executor,
    ordinary_member_execution_inputs, member_device_ids_for_request,
)


def source():
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000.0, dy=3000.0, ztop=12000.0,
        dt=12.0, run_seconds=120.0, time_step_sound=4, moist=True, mp_physics=8,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rrtmg_legacy")
    domain = SimpleNamespace(run=cfg, grid_id=1, parent_id=0, tiles=None)
    exp = SimpleNamespace(root=domain, domains=(domain,), tiles=SimpleNamespace(mode="off"),
        devices=SimpleNamespace(count=1), relocation=None, run_seconds=120.0,
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    inputs = SimpleNamespace(experiment=exp, domains=(object(),), stream_head=None,
        source="fixture", execution_plan="fixture")
    state = DomainState(cfg, array_module=np)
    state.p_top = np.float32(5000)
    state.cf1, state.cf2, state.cf3 = (np.float32(.5), np.float32(.3), np.float32(.2))
    radiation_type = type("RRTMGLegacyRadiation", (), {"__module__": "woof.core.rrtmg_legacy"})
    radiation = radiation_type()
    radiation._ozone_provider = None
    driver = PhysicsDriver.__new__(PhysicsDriver)
    driver.state, driver.radiation_callable = state, radiation
    driver.fields = {"soil": np.arange(4 * 64, dtype=np.float32).reshape(4, 8, 8)}
    state.physics = driver
    spec = DomainTicks(1, 0, 1, 12, np.float32(12), 24, None, None, None,
        None, None, None, None, lbc_interval_ticks=24)
    node = SimpleNamespace(cfg=domain, state=state, clock=DomainClock(spec, 1, 120),
        grid=object(), parent=None, children=[], coupler=None)
    return inputs, node


def collector(members):
    requests = default_product_requests(DEFAULT_THRESHOLDS)[0]
    return SimpleNamespace(members=members, requests=requests, tile_rows=2,
        keep_member_files=False, submit=lambda **kwargs: None,
        memory_plan=lambda shape, **kwargs: headline_diagnostic_memory_plan(shape, **kwargs))


def evidence(device=0, available=1300):
    return InitializedCardEvidence(CardBudget(device, available, 10_000, "fixture"),
        MemoryComponent("runtime", "runtime", fixed_bytes=10, evidence="fixture runtime"),
        AllocatorMargin(minimum_bytes=1, evidence="fixture margin"), 100,
        evidence="fixture after-bootstrap sample")


def model(*args, **kwargs):
    return EnsembleMemoryModel((MemoryComponent("fixture", "state", fixed_bytes=100, per_member_bytes=300),),
                               inventory_id="fixture"), {"fixture": "after_bootstrap"}


def xp():
    return SimpleNamespace(cuda=SimpleNamespace(Device=lambda device: nullcontext(),
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))


def test_packing_prices_whole_roster_before_allocating_any_native_member():
    inputs, node = source()
    plan = plan_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(), model_factory=model)
    assert [batch.member_indices for batch in plan.packing.batches] == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    assert plan.packing.capacities == (4,)
    assert all(batch.required_bytes <= batch.available_bytes for batch in plan.packing.batches)
    assert node.clock.step_count == 0 and node.state.physics.state is node.state
    assert plan.receipt()["admissions"][0]["free_sample_timing"] == "after_bootstrap"


def test_unqualified_initialized_owner_inventory_declines_before_member_allocation():
    from woof.ensemble.batch_state import BatchStateUnsupported
    inputs, node = source()
    def unsupported(*args, **kwargs):
        raise BatchStateUnsupported("fixture future scratch")
    plan = plan_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(),
        model_factory=unsupported)
    assert not plan.native and "fixture future scratch" in plan.fallback_reasons[0]
    assert node.clock.ticks == 0 and node.state.physics.state is node.state


def test_warm_source_shell_detachment_does_not_change_source_words_or_original_clock():
    inputs, node = source()
    lease = WarmInitializedRoot(inputs, node)
    soil = node.state.physics.fields["soil"].tobytes()
    waves = [lease.wave_node(), lease.wave_node()]
    for wave in waves:
        assert wave.state is not node.state and wave.state.physics is not node.state.physics
        assert wave.state.p is node.state.p
        assert wave.state.physics.fields is node.state.physics.fields
        wave.state.physics.state = None
        wave.state.physics = None
        wave.clock.prepare_step()
        wave.clock.advance()
        lease.require_unchanged()
    assert node.clock.ticks == 0 and node.clock.step_count == 0
    assert node.state.physics.fields["soil"].tobytes() == soil
    assert waves[0].clock is not waves[1].clock


def test_wave_executor_has_fresh_exact_source_each_wave_and_no_retry():
    inputs, node = source()
    starts = []
    def native(inputs, wave, **kwargs):
        starts.append((kwargs["member_ids"], wave.clock.ticks, wave.clock.dtbc_fp32.tobytes(), wave.state.p.tobytes()))
        wave.state.physics.state = None
        wave.state.physics = None
        while not wave.clock.at_stop_time:
            wave.clock.prepare_step()
            wave.clock.advance()
        return {"status": "PASS", "completed_seconds": wave.clock.elapsed_seconds}
    result = execute_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(),
        array_module=xp(), model_factory=model, native_runner=native)
    assert result["status"] == "PASS" and result["members_completed"] == list(range(10))
    assert [item[0] for item in starts] == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    assert all(item[1:] == starts[0][1:] for item in starts)
    assert node.clock.ticks == 0
    calls = []
    def failing(*args, **kwargs):
        calls.append(kwargs["member_ids"])
        raise RuntimeError("fixture launch failure")
    with pytest.raises(RuntimeError, match="fixture launch failure"):
        execute_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(),
            array_module=xp(), model_factory=model, native_runner=failing)
    assert calls == [(0, 1, 2, 3)]


def test_two_card_execution_keeps_one_original_bootstrap_per_card():
    inputs, node = source()
    other_inputs, other_node = source()
    nodes, calls = {0: node, 1: other_node}, []
    def native(inputs, wave, **kwargs):
        calls.append((kwargs["member_ids"], wave.state.p is nodes[0].state.p, wave.state.p is nodes[1].state.p))
        return {"status": "PASS"}
    result = execute_initialized_member_execution(inputs, node,
        {"members": 8, "member_device_ids": (0, 1)}, collector(8), evidence=evidence(0),
        card_bootstraps=(InitializedCardBootstrap(other_inputs, other_node, evidence(1)),),
        array_module=xp(), model_factory=model, native_runner=native)
    assert len(calls) == 2 and result["members_completed"] == list(range(8))
    assert sorted((row[1], row[2]) for row in calls) == [(False, True), (True, False)]
    assert node.clock.ticks == other_node.clock.ticks == 0


def test_singleton_tail_goes_to_original_executor_and_missing_seam_declines_before_native():
    inputs, node = source()
    calls = []
    native = lambda *args, **kwargs: calls.append(kwargs["member_ids"]) or {"status": "PASS"}
    assert execute_initialized_member_execution(inputs, node, 9, collector(9), evidence=evidence(),
        array_module=xp(), model_factory=model, native_runner=native) is None
    assert not calls
    ordinary = []
    result = execute_initialized_member_execution(inputs, node, 9, collector(9), evidence=evidence(),
        array_module=xp(), model_factory=model, native_runner=native,
        ordinary_executor=lambda **kwargs: ordinary.append(kwargs) or {"status": "PASS"})
    assert result["status"] == "PASS" and calls == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert ordinary[0]["member_id"] == 8 and ordinary[0]["inputs"] is inputs


def test_active_stochastic_and_streamed_configuration_decline_without_array_operations():
    inputs, node = source()
    for change in ("stochastic", "streaming"):
        if change == "stochastic":
            node.state._ensemble_stochastic = SimpleNamespace(enabled=True)
        else:
            node.state._ensemble_stochastic = None
            inputs.experiment.tiles.mode = "on"
        plan = plan_initialized_member_execution(inputs, node, 10, collector(10),
            model_factory=lambda *args, **kwargs: pytest.fail("native allocation planner called"))
        assert not plan.native and plan.fallback_reasons


def test_actual_output_plans_price_full_roster_replay_health_and_separate_host_counters():
    inputs, node = source()
    components, host = native_output_memory_components(node, collector(20))
    rows = {component.name: component.inventory(4) for component in components}
    assert rows["native_roster_replay"]["required_bytes"] > 0
    assert rows["native_full_state_health"]["required_bytes"] > 0
    assert next(component for component in components if component.name == "native_full_state_health").inventory(10)["required_bytes"] > rows["native_full_state_health"]["required_bytes"]
    assert host["replay_members"] == 20 and host["counter_endpoint_storage"].startswith("host")
    assert host["counter_endpoint_bank_bytes"] == host["maximum_retained_endpoints_per_member"] * 20 * 8 * 8 * 4


def test_automatic_factory_handoff_runs_whole_native_roster_without_original_first_step(tmp_path):
    inputs, node = source()
    original_steps, initials = [], []
    def runner(inputs, **options):
        initials.append(options)
        result = options["ensemble_bootstrap"](inputs=inputs, model=object(), node=node,
            output_directory=options["output_directory"], observer=None, step_observer=None)
        if result is not None:
            return result
        original_steps.append(0)
        return {"status": "PASS"}
    run = make_automatic_prepared_executor(runner, request=10, collector=collector(10), output_directory=tmp_path,
        evidence_provider=lambda **kwargs: evidence(), array_module=xp(), model_factory=model,
        native_runner=lambda *args, **kwargs: {"status": "PASS"})
    result = run(inputs)
    assert result.native_complete and result.report["members_completed"] == list(range(10))
    assert len(initials) == 1 and not original_steps


def test_automatic_n1_keeps_original_first_receipt_without_device_or_new_clock(tmp_path):
    inputs, node = source()
    seen = []
    receipt = {"status": "PASS", "original_words": "fixture"}
    def runner(inputs, **options):
        assert options["ensemble_bootstrap"](inputs=inputs, model=object(), node=node,
            output_directory=tmp_path, observer=None, step_observer=None) is None
        seen.append(node.clock)
        return receipt
    run = make_automatic_prepared_executor(runner, request=1, collector=collector(1), output_directory=tmp_path,
        evidence_provider=lambda **kwargs: pytest.fail("device sampled"))
    result = run(inputs)
    assert not result.native_complete and result.report is receipt
    assert seen == [node.clock] and result.first_member_id == 0


def test_first_member_runs_in_the_directory_the_session_claimed(tmp_path):
    # The ensemble session claims every member directory before execution and leaves member-provenance.json in it;
    # member-roster ensembles then stopped with FileExistsError on member-0000 (rescued fix efb7cbed66a).
    inputs, node = source()
    claimed = tmp_path / "members" / "member-0000"
    claimed.mkdir(parents=True)
    (claimed / "member-provenance.json").write_text("{}")
    seen = []
    def runner(inputs, **options):
        assert options["ensemble_bootstrap"](inputs=inputs, model=object(), node=node,
            output_directory=options["output_directory"], observer=None, step_observer=None) is None
        seen.append(options["output_directory"])
        return {"status": "PASS"}
    make = lambda: make_automatic_prepared_executor(runner, request=1, collector=collector(1),
        output_directory=tmp_path, evidence_provider=lambda **kwargs: pytest.fail("device sampled"))
    result = make()(inputs)
    assert result.first_member_id == 0 and seen == [claimed] and result.first_output_directory == claimed
    # Anything else in that directory is another run's output and is still refused.
    (claimed / "wrfout_d01_2024-01-01_00:00:00").write_text("")
    with pytest.raises(FileExistsError):
        make()(inputs)
    assert len(seen) == 1


def test_automatic_capture_exposes_initialization_callback_before_restart_and_health(tmp_path):
    from woof.ensemble.runtime_context import current_capture
    inputs, node = source()
    events = []
    def callback_factory(**identity):
        assert identity["member_id"] == 0 and identity["prepared_member"] is None
        return lambda **kwargs: events.append(("initialized", kwargs["model"]))
    model = object()
    def runner(inputs, **options):
        current_capture().initialize_callback(model=model)
        events.append(("validate-restart-and-health", model))
        assert options["ensemble_bootstrap"](inputs=inputs, model=model, node=node,
            output_directory=tmp_path, observer=None, step_observer=None) is None
        return {"status": "PASS"}
    run = make_automatic_prepared_executor(runner, request=1, collector=collector(1), output_directory=tmp_path,
        initialize_callback_factory=callback_factory,
        evidence_provider=lambda **kwargs: pytest.fail("device sampled"))
    assert not run(inputs).native_complete
    assert events == [("initialized", model), ("validate-restart-and-health", model)]


def test_sparse_member_replay_uses_original_source_id_and_seed_before_initializer(tmp_path):
    from woof.ensemble.recipes import build_recipe
    from woof.ensemble.member_preparation import SourceManifestBinding, PreparedMemberInput, PreparedMemberRoster
    start = datetime(2024, 5, 25, 18, tzinfo=timezone.utc)
    whole = build_recipe(source="gfs", cycle=start, start=start, end=start + timedelta(hours=12),
        count=20, base_seed=42, kind="input-ensemble")
    recipe = whole.select_members((19,))
    member = recipe.members[0]
    doc = dict(source=member.trajectory.source, cycle=member.trajectory.cycle.isoformat(), member=member.trajectory.member)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = SourceManifestBinding(member.trajectory, path, digest, {**doc, "manifest_sha256": digest})
    real_inputs, node = source()
    selected = PreparedMemberInput(19, member.seed, real_inputs, object(), object(), member.trajectory,
        (binding,), (recipe.start, recipe.end), "b" * 64, recipe.sha256)
    roster = PreparedMemberRoster(recipe, (selected,), shared_geometry_sha256="b" * 64)
    calls = []
    def runner(inputs, **options):
        assert inputs is real_inputs
        assert options["ensemble_bootstrap"](inputs=inputs, model=object(), node=node,
            output_directory=tmp_path, observer=None, step_observer=None) is None
        return {"status": "PASS"}
    run = make_automatic_prepared_executor(runner, request=1, collector=collector(1), output_directory=tmp_path,
        member_roster=roster, initialized_member_binding=lambda **kwargs: calls.append(kwargs),
        evidence_provider=lambda **kwargs: pytest.fail("device sampled"))
    result = run(object())
    assert result.first_member_id == 19 and result.first_seed == whole.members[19].seed
    assert calls[0]["member_id"] == 19 and calls[0]["prepared_member"] is selected


@dataclass(frozen=True)
class _TypedInputs:
    experiment: object
    stream_head: object
    initial: object
    boundaries: object


@pytest.mark.parametrize("mode", ["off", "auto", "on"])
def test_actual_stream_budget_overlay_preserves_inputs_physics_clock_and_restart_identity(mode):
    from woof.experiment import experiment_from_run_config
    from woof.core.streaming import StreamingOptions
    from woof.core.model import restart_identity_payload
    from woof.ensemble.packing import MemberBatch
    inputs, node = source()
    exp = experiment_from_run_config(replace(node.cfg.run, output_interval_s=24), inputs.experiment.start_time)
    options = StreamingOptions(mode=mode, vram_budget_bytes=900,
        tile_nx=8 if mode == "on" else None, tile_ny=8 if mode == "on" else None)
    exp = replace(exp, tiles=options)
    original = _TypedInputs(exp, object(), np.ones((8, 8), np.float32), (object(), object()))
    batch = MemberBatch(0, 0, (0,), "ordinary_streamed_member", None, 1024)
    components = tuple(MemoryComponent(name, name, fixed_bytes=size, evidence="fixture exact plan")
        for name, size in (("products", 64), ("health", 32), ("stochastic", 96)))
    overlay, receipt = ordinary_member_execution_inputs(original, batch=batch, external_components=components,
        allocator_margin=AllocatorMargin(minimum_bytes=128, evidence="fixture margin"))
    assert overlay.experiment.tiles.mode == ("auto" if mode == "off" else mode)
    assert overlay.experiment.tiles.vram_budget_bytes == 704
    assert overlay.experiment.root.run is original.experiment.root.run
    assert overlay.stream_head is original.stream_head and overlay.initial is original.initial
    assert overlay.boundaries is original.boundaries
    assert restart_identity_payload(overlay.experiment) == restart_identity_payload(exp)
    assert receipt["ensemble_withheld_bytes"] == 320
    if mode == "on":
        assert overlay.experiment.tiles.tile_nx == overlay.experiment.tiles.tile_ny == 8


def test_device_selection_uses_all_visible_cards_without_changing_spatial_group():
    from woof.core.devices import DeviceOptions
    inputs, node = source()
    assert member_device_ids_for_request(inputs, 20, visible_count=2) == (0, 1)
    assert member_device_ids_for_request(inputs, {"members": 20, "member_device_ids": (1,)}, visible_count=2) == (1,)
    inputs.experiment.devices = DeviceOptions(count=2, ids=(0, 1))
    assert member_device_ids_for_request(inputs, 20, visible_count=2) == (0,)
    with pytest.raises(ValueError, match="overlap"):
        member_device_ids_for_request(inputs, {"members": 20, "member_device_ids": (0, 1)}, visible_count=2)


def _distinct_source(words):
    """A second ordinary root whose member words differ from the fixture's."""
    inputs, node = source()
    node.state.u[...] = np.float32(words)
    node.state.physics.fields["soil"][...] = np.float32(10 * words)
    return inputs, node


def _member_sources(root_id, others):
    from woof.ensemble.prepared_batch import NativeMemberSources, snapshot_member_source
    return NativeMemberSources(root_id, tuple(snapshot_member_source(node, member_id=member)
                                              for member, node in others.items()))


def test_member_specific_inputs_without_bound_sources_decline_by_name():
    inputs, node = source()
    plan = plan_initialized_member_execution(inputs, node, 4, collector(4), evidence=evidence(),
        model_factory=model, input_provider=lambda **kwargs: inputs)
    assert not plan.native
    assert any("not all bootstrapped as member sources" in reason for reason in plan.fallback_reasons)
    assert node.clock.ticks == 0 and node.state.physics.state is node.state


def test_bound_compatible_member_sources_make_the_roster_native():
    from woof.ensemble.prepared_batch import bind_member_sources, member_sources_of
    inputs, node = source()
    others = {member: _distinct_source(member)[1] for member in (1, 2, 3)}
    sources = _member_sources(0, others)
    assert sources.compatibility_reasons(node) == ()
    bind_member_sources(node, sources)
    assert member_sources_of(node).member_ids == (0, 1, 2, 3)
    plan = plan_initialized_member_execution(inputs, node, 4, collector(4), evidence=evidence(),
        model_factory=model, input_provider=lambda **kwargs: inputs)
    assert plan.native and plan.fallback_reasons == ()
    assert [batch.member_indices for batch in plan.packing.batches] == [(0, 1, 2, 3)]
    # Every requested member needs its own bootstrap; a fifth one has none.
    plan = plan_initialized_member_execution(inputs, node, 5, collector(5), evidence=evidence(),
        model_factory=model, input_provider=lambda **kwargs: inputs)
    assert not plan.native and "not all bootstrapped as member sources" in plan.fallback_reasons[0]


@pytest.mark.parametrize("change", ["clock", "scalar", "shared_state", "shared_land", "owner"])
def test_incompatible_member_sources_are_refused_by_name_before_allocation(change):
    from woof.ensemble.prepared_batch import bind_member_sources
    inputs, node = source()
    _other_inputs, other = _distinct_source(7)
    for root in (node, other):
        root.state.physics.fields["xland"] = np.ones((8, 8), np.float32)
    expected = {"clock": "integer clock", "scalar": "scalar state metadata",
                "shared_state": "shared base-state or metric fields",
                "shared_land": "shared land surface fields", "owner": "shared physics owners"}[change]
    if change == "clock":
        other.clock = DomainClock(other.clock.spec, 1, 240)
    elif change == "scalar":
        other.state.p_top = np.float32(4000)
    elif change == "shared_state":
        other.state.mub2d[...] = np.float32(5)
    elif change == "shared_land":
        other.state.physics.fields["xland"][...] = np.float32(2)
    elif change == "owner":
        other.state.physics.noah_params = {"table": np.arange(3, dtype=np.float32)}
    sources = _member_sources(0, {1: other})
    reasons = sources.compatibility_reasons(node)
    assert reasons and all("member 1" in reason for reason in reasons)
    assert any(expected in reason for reason in reasons), reasons
    bind_member_sources(node, sources)
    plan = plan_initialized_member_execution(inputs, node, 2, collector(2), evidence=evidence(),
        model_factory=lambda *args, **kwargs: pytest.fail("native allocation planner called"),
        input_provider=lambda **kwargs: inputs)
    assert not plan.native and any(expected in reason for reason in plan.fallback_reasons)


def _bootstrapping_runner(inputs, node, members, runs):
    """An ordinary runner that initializes each member's own root and hands it to the seam."""
    nodes = {id(inputs): node}
    nodes.update({id(pair[0]): pair[1] for pair in members.values()})
    def runner(member_inputs, **options):
        runs.append((member_inputs, options["output_directory"]))
        result = options["ensemble_bootstrap"](inputs=member_inputs, model=object(), node=nodes[id(member_inputs)],
            output_directory=options["output_directory"], observer=None, step_observer=None)
        return result if result is not None else {"status": "PASS"}
    return runner


def _pooled_xp():
    module = xp()
    module.get_default_memory_pool = lambda: SimpleNamespace(used_bytes=lambda: 0)
    return module


def test_automatic_handoff_bootstraps_each_member_and_packs_them_as_their_own_sources(tmp_path):
    from woof.ensemble.prepared_batch import member_sources_of
    inputs, node = source()
    members = {member: _distinct_source(member) for member in (1, 2, 3)}
    member_inputs = {0: inputs, **{member: pair[0] for member, pair in members.items()}}
    runs, packed = [], []
    def native(inputs, wave, **kwargs):
        sources = member_sources_of(wave)
        packed.append((kwargs["member_ids"], None if sources is None else sources.member_ids))
        return {"status": "PASS"}
    run = make_automatic_prepared_executor(_bootstrapping_runner(inputs, node, members, runs),
        request=4, collector=collector(4), output_directory=tmp_path,
        input_provider=lambda *, member_id, **kwargs: member_inputs[member_id],
        evidence_provider=lambda **kwargs: evidence(), array_module=_pooled_xp(), model_factory=model,
        native_runner=native)
    result = run(inputs)
    assert result.native_complete and result.report["members_completed"] == [0, 1, 2, 3]
    # The first member's runner runs once; every other member is bootstrapped
    # once through its own inputs, under the initialization folder, and the
    # whole roster is packed in one wave with its sources bound to the root.
    assert [member for member, _ in runs] == [inputs, member_inputs[1], member_inputs[2], member_inputs[3]]
    assert [out.relative_to(tmp_path).as_posix() for _, out in runs][1:] == [
        f".ensemble-initialization/member-{member:04d}" for member in (1, 2, 3)]
    assert packed == [((0, 1, 2, 3), (0, 1, 2, 3))]
    assert [row["member_id"] for row in result.admission["member_bootstraps"]] == [1, 2, 3]
    assert all(row["forecast_steps"] == 0 and row["host_snapshot_bytes"] > 0
               for row in result.admission["member_bootstraps"])
    assert result.admission["native_admitted"] and not result.admission["fallback_reasons"]


def test_surface_callback_uses_roster_seed_and_changes_driver_before_member_snapshot(tmp_path, monkeypatch):
    from woof.ensemble.prepared_execution import _bootstrap_member_sources
    from woof.ensemble.prepared_batch import native_prepared_eligibility
    from woof.ensemble.production import PreparedEnsembleSession
    from woof.ensemble.runtime_context import current_capture
    from woof.ensemble import surface_recipe
    inputs, root = source()
    other_inputs, other = source()
    for node in (root, other):
        node.state.physics.fields["xland"] = np.ones((8, 8), dtype=np.float32)
    assert native_prepared_eligibility(inputs, root, members=2).eligible
    selected = {3: SimpleNamespace(member_id=3, seed=731), 7: SimpleNamespace(member_id=7, seed=907)}
    roster = SimpleNamespace(members=tuple(selected.values()), select=lambda ids: tuple(selected[index] for index in ids))
    request = {"members": 2, "base_seed": 42, "recipe": "surface-state",
               "perturbation": {"kind": "surface-state", "soil_moisture_scale": [0.8, 1.2]}}
    session = PreparedEnsembleSession(request, output_directory=tmp_path,
                                     member_roster=roster, array_module=_pooled_xp())
    events = []
    def realize(value, *, seed, **unused):
        events.append(("realized", seed))
        return object(), {"realized_fp32_hex": "device-drawn-words"}
    def apply(state, *, member_id, seed, domain_id, **unused):
        events.append(("surface-applied", member_id, seed, domain_id))
        state.physics.fields["soil"][...] = np.float32(907)
        return {"member_id": member_id, "seed": seed, "domain_id": domain_id,
                "realized_fp32_hex": "device-drawn-words"}
    monkeypatch.setattr(surface_recipe, "realize_surface_recipe", realize)
    monkeypatch.setattr(surface_recipe, "apply_surface_recipe", apply)
    original_static = other.state.physics.fields["xland"].tobytes()
    def runner(member_inputs, **options):
        assert member_inputs is other_inputs
        capture = current_capture()
        assert capture.member_id == 7
        model = SimpleNamespace(walk_parent_first=lambda: iter((other,)))
        capture.initialize_callback(model=model)
        events.append(("snapshot-handoff", 7))
        return options["ensemble_bootstrap"](inputs=member_inputs, node=other, model=model)
    sources, receipts = _bootstrap_member_sources(runner, node=root, ids=(3, 7), first_id=3,
        member_inputs_for=lambda member: other_inputs, out=tmp_path, options={},
        array_module=_pooled_xp(), initialize_callback_factory=session._initialization_callback)
    assert events == [("realized", 907), ("surface-applied", 7, 907, 1), ("snapshot-handoff", 7)]
    np.testing.assert_array_equal(sources.snapshots[7].physics.member_array("driver/fields/soil"),
                                  np.full((4, 8, 8), 907, dtype=np.float32))
    assert sources.snapshots[7].physics.member_array("driver/fields/xland").tobytes() == original_static
    assert session._surface_receipts[7]["seed"] == 907
    assert receipts[0]["member_id"] == 7 and receipts[0]["forecast_steps"] == 0


def test_surface_receipt_metadata_does_not_substitute_another_members_numerical_scalars():
    from woof.ensemble.prepared_batch import bind_member_sources
    inputs, node = source()
    _other_inputs, other = source()
    for member, owner in ((0, node), (1, other)):
        owner.state._ensemble_surface_state = {
            "identity": {"member_id": member, "seed": 900 + member},
            "receipt": {"realized_fp32_hex": str(member)},
        }
    sources = _member_sources(0, {1: other})
    assert sources.compatibility_reasons(node) == ()
    assert "_ensemble_surface_state" not in sources.snapshots[1].prepared.scalars
    bind_member_sources(node, sources)
    plan = plan_initialized_member_execution(inputs, node,
        {"members": 2, "recipe": "surface-state", "perturbation": {
            "kind": "surface-state", "soil_moisture_scale": [0.8, 1.2]}},
        collector(2), evidence=evidence(), model_factory=model,
        input_provider=lambda **unused: inputs)
    assert plan.native and not plan.fallback_reasons


def test_automatic_handoff_declines_an_incompatible_member_after_bootstrapping_it(tmp_path):
    inputs, node = source()
    members = {member: _distinct_source(member) for member in (1, 2)}
    members[2][1].state.p_top = np.float32(4000)
    member_inputs = {0: inputs, **{member: pair[0] for member, pair in members.items()}}
    runs = []
    run = make_automatic_prepared_executor(_bootstrapping_runner(inputs, node, members, runs),
        request=3, collector=collector(3), output_directory=tmp_path,
        input_provider=lambda *, member_id, **kwargs: member_inputs[member_id],
        evidence_provider=lambda **kwargs: pytest.fail("device sampled after a named refusal"),
        array_module=_pooled_xp(), model_factory=model,
        native_runner=lambda *args, **kwargs: pytest.fail("native pack built from an incompatible roster"))
    result = run(inputs)
    assert not result.native_complete and result.mode == "ordinary_first"
    assert not result.admission["native_admitted"]
    assert any("member 2" in reason and "scalar state metadata" in reason
               for reason in result.admission["fallback_reasons"])
    assert [row["member_id"] for row in result.admission["member_bootstraps"]] == [1, 2]
    assert [member for member, _ in runs] == [inputs, member_inputs[1], member_inputs[2]]


def test_native_launch_refusal_before_any_pack_begins_is_a_named_decline(capsys):
    """A source-audit drift declines to the ordinary runner; it does not fail the run."""
    from woof.ensemble.native_forecast import NativeLaunchRefused
    inputs, node = source()
    soil = node.state.physics.fields["soil"].tobytes()
    calls, declined = [], []
    def refusing(inputs, wave, **kwargs):
        calls.append(kwargs["member_ids"])
        # The wave's own shell is touched, as a partial preparation would.
        wave.state.physics.state = None
        raise NativeLaunchRefused("native launch preparation refused before any member advanced: "
                                  "ValueError: Thompson base-load indexing changed; shared terrain "
                                  "adapter needs a new source audit")
    result = execute_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(),
        array_module=xp(), model_factory=model, native_runner=refusing, on_decline=declined.append)
    assert result is None
    assert calls == [(0, 1, 2, 3)], "the first pack refused and no later wave was started"
    (reasons,) = declined
    assert len(reasons) == 1 and "needs a new source audit" in reasons[0]
    assert node.clock.ticks == 0 and node.clock.step_count == 0
    assert node.state.physics.state is node.state
    assert node.state.physics.fields["soil"].tobytes() == soil
    said = capsys.readouterr().err
    assert "the native member pack declined and every member runs through the ordinary runner" in said
    assert "needs a new source audit" in said


def test_native_launch_refusal_after_a_pack_began_still_fails_the_run():
    from woof.ensemble.native_forecast import NativeLaunchRefused
    inputs, node = source()
    calls, declined = [], []
    def native(inputs, wave, **kwargs):
        calls.append(kwargs["member_ids"])
        if len(calls) == 1:
            kwargs["launch_prepared"]()
            return {"status": "PASS"}
        raise NativeLaunchRefused("fixture audit drift in a later wave")
    with pytest.raises(RuntimeError, match="after another pack had begun") as caught:
        execute_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(),
            array_module=xp(), model_factory=model, native_runner=native, on_decline=declined.append)
    assert calls == [(0, 1, 2, 3), (4, 5, 6, 7)] and not declined
    assert caught.value.ensemble_member_ids == (4, 5, 6, 7)
    assert "fixture audit drift in a later wave" in str(caught.value)


def test_two_cards_decline_together_when_one_refuses_before_either_began():
    from contextlib import contextmanager
    from woof.ensemble.native_forecast import NativeLaunchRefused
    import threading
    inputs, node = source()
    other_inputs, other_node = source()
    refusal_recorded = threading.Event()
    advanced = []
    @contextmanager
    def scope(device):
        # The first card leaves its device scope only after its refusal was recorded.
        try:
            yield
        finally:
            if device == 0:
                refusal_recorded.set()
    def native(inputs, wave, **kwargs):
        if kwargs["member_ids"][0] == 0:
            raise NativeLaunchRefused("fixture audit drift")
        assert refusal_recorded.wait(10)
        kwargs["launch_prepared"]()          # declines here: the other pack refused first
        advanced.append(kwargs["member_ids"])
        return {"status": "PASS"}
    declined = []
    result = execute_initialized_member_execution(inputs, node,
        {"members": 8, "member_device_ids": (0, 1)}, collector(8), evidence=evidence(0),
        card_bootstraps=(InitializedCardBootstrap(other_inputs, other_node, evidence(1)),),
        array_module=xp(), model_factory=model, native_runner=native, on_decline=declined.append,
        device_scope=scope)
    assert result is None and not advanced
    assert declined == [("fixture audit drift",)]


def test_plan_level_declines_carry_their_reasons_to_the_admission_receipt(tmp_path):
    from woof.ensemble.batch_state import BatchStateUnsupported
    inputs, node = source()
    def unsupported(*args, **kwargs):
        raise BatchStateUnsupported("fixture future scratch")
    def runner(inputs, **options):
        assert options["ensemble_bootstrap"](inputs=inputs, model=object(), node=node,
            output_directory=options["output_directory"], observer=None, step_observer=None) is None
        return {"status": "PASS"}
    run = make_automatic_prepared_executor(runner, request=10, collector=collector(10), output_directory=tmp_path,
        evidence_provider=lambda **kwargs: evidence(), array_module=xp(), model_factory=unsupported,
        native_runner=lambda *args, **kwargs: pytest.fail("native pack started"))
    result = run(inputs)
    assert not result.native_complete
    assert result.admission["native_admitted"] is False
    assert result.admission["fallback"] == "ordinary_member_runner"
    assert "fixture future scratch" in result.admission["fallback_reasons"][0]


def test_step_log_follows_the_pack_that_holds_the_first_member():
    inputs, node = source()
    observed = []
    def native(inputs, wave, **kwargs):
        observed.append((kwargs["member_ids"], "step_observer" in kwargs))
        return {"status": "PASS"}
    log = lambda **step: None
    execute_initialized_member_execution(inputs, node, 10, collector(10), evidence=evidence(),
        array_module=xp(), model_factory=model, native_runner=native, step_observer=log)
    assert observed == [((0, 1, 2, 3), True), ((4, 5, 6, 7), False), ((8, 9), False)]


def test_a_stop_requested_during_a_native_pack_ends_it_at_the_next_step_boundary():
    """The pack observes the run's control at every step it reports.

    The production adapter checks the bound control on each member event;
    this native runner reports steps through the same seam and asks the
    run to stop through the control the scope binds. The pack must end at
    the next reported step, as MemberStopRequested, with the retained root
    untouched; a scope that bound nothing would let it run to completion.
    """
    from woof.ensemble.execution import (MemberRunControl, MemberStopRequested,
                                          current_run_control, member_run_scope)
    inputs, node = source()
    control = MemberRunControl()
    reported = []
    def progress(**event):
        current_run_control().check()
        reported.append(event["outer_step"])
    def native(inputs, wave, *, progress_callback, member_ids, **kwargs):
        assert current_run_control() is control
        for step in range(1, 11):
            if step == 3:
                control.request_stop("the test asked the run to stop")
            progress_callback(status="RUNNING", outer_step=step, member_ids=member_ids)
        return {"status": "PASS"}
    with member_run_scope(control):
        with pytest.raises(MemberStopRequested, match="asked the run to stop"):
            execute_initialized_member_execution(inputs, node, 4, collector(4), evidence=evidence(),
                array_module=xp(), model_factory=model, native_runner=native, progress_callback=progress)
    assert reported == [1, 2]
    assert control.stop_requested and node.clock.ticks == 0 and node.state.physics.state is node.state
    # Without a stop the same pack reports every step and completes.
    reported.clear()
    quiet = MemberRunControl()
    def running(inputs, wave, *, progress_callback, member_ids, **kwargs):
        for step in range(1, 4):
            progress_callback(status="RUNNING", outer_step=step, member_ids=member_ids)
        return {"status": "PASS"}
    with member_run_scope(quiet):
        result = execute_initialized_member_execution(inputs, node, 4, collector(4), evidence=evidence(),
            array_module=xp(), model_factory=model, native_runner=running, progress_callback=progress)
    assert result["status"] == "PASS" and reported == [1, 2, 3]

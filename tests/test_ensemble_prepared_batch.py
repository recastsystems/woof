"""Prepared-native eligibility and exact ordinary bootstrap snapshots."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.config import RunConfig
from woof.core.state import DomainState
from woof.ensemble.batch_state import BatchedDomainState, BatchStateUnsupported, SHARED_STATE_CANDIDATES
from woof.ensemble.batch_storage import BatchArraySpec
from woof.ensemble.prepared_batch import (
    native_prepared_eligibility, prepared_host_from_node, prepare_native_member_batch, native_memory_model_from_node,
)


def _source():
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000.0, dy=3000.0, ztop=12000.0, dt=12.0,
        run_seconds=120.0, time_step_sound=4, moist=True, mp_physics=8,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rrtmg_legacy")
    domain = SimpleNamespace(run=cfg, grid_id=1, parent_id=0, tiles=None)
    exp = SimpleNamespace(root=domain, domains=(domain,), tiles=SimpleNamespace(mode="off"),
                          devices=SimpleNamespace(count=1), relocation=None)
    inputs = SimpleNamespace(experiment=exp, domains=(object(),), stream_head=None)
    state = DomainState(cfg, array_module=np)
    state.p_top = np.float32(5000)
    state.cf1, state.cf2, state.cf3 = (np.float32(0.5), np.float32(0.3), np.float32(0.2))
    radiation_type = type("RRTMGLegacyRadiation", (), {"__module__": "woof.core.rrtmg_legacy"})
    radiation = radiation_type()
    radiation._ozone_provider = None
    state.physics = SimpleNamespace(state=state, radiation_callable=radiation)
    clock = SimpleNamespace(ticks=0, step_ticks=12, tick_den=1, run_ticks=120, step_count=0,
        dt_fp32=np.float32(12), dtbc_fp32=np.float32(0), adaptive_state=None,
        spec=SimpleNamespace(start_ticks=0))
    node = SimpleNamespace(cfg=domain, state=state, clock=clock, parent=None, children=[], coupler=None)
    return inputs, node


def test_native_route_keeps_the_original_initialized_clock_and_suite():
    inputs, node = _source()
    decision = native_prepared_eligibility(inputs, node, members=10)
    assert decision.eligible and not decision.reasons
    before = vars(node.cfg.run).copy()
    for field in ("physics_selection", "streaming_selection", "clock_selection"):
        assert decision.receipt()[field] == "unchanged"
    assert vars(node.cfg.run) == before
    assert not native_prepared_eligibility(inputs, node, members=1).eligible


@pytest.mark.parametrize("chem_sets", ["smoke", "gocart_primary,smoke", "cams_aq"])
def test_active_chemistry_keeps_original_members_before_native_packing(chem_sets, monkeypatch):
    from woof.ensemble.suite_capabilities import plan_suite

    inputs, node = _source()
    node.cfg.run = replace(node.cfg.run, chem_sets=chem_sets)
    before = vars(node.cfg.run).copy()
    original_driver = node.state.physics
    plan = plan_suite(node.cfg.run, members=10)
    assert plan.mode == "member_local" and not plan.uses_native_batch
    assert any("species arrays, source-hour caches and mass ledger" in reason
               for reason in plan.native_fallback_reasons)
    decision = native_prepared_eligibility(inputs, node, members=10)
    assert not decision.eligible and decision.receipt()["fallback"] == "ordinary_member_runner"
    assert any("active chemistry" in reason for reason in decision.reasons)
    monkeypatch.setattr(BatchedDomainState, "from_prepared",
                        lambda *a, **k: pytest.fail("active chemistry reached native state packing"))
    assert prepare_native_member_batch(inputs, node, members=10, available_bytes=0, array_module=np) is None
    assert node.state.physics is original_driver and vars(node.cfg.run) == before


def test_inactive_chemistry_controls_leave_the_default_native_route_unchanged():
    inputs, node = _source()
    original = native_prepared_eligibility(inputs, node, members=10).receipt()
    node.cfg.run = replace(node.cfg.run, chem_sets="", chem_adv_opt=2, dust_opt=3,
                           aerosol_mp_coupling="diagnose", chem_sources="cams-global")
    assert native_prepared_eligibility(inputs, node, members=10).receipt() == original


def test_native_snapshot_rebuilds_only_registered_boundary_mirrors_without_changing_original_tables():
    inputs, node = _source()
    state = node.state
    source = np.arange(19, dtype=np.float32)
    evaluated = np.arange(11, dtype=np.float32)
    ordinary_owner = SimpleNamespace(scratch_slots={"lbc_forcing_tables", "lbc_evaluated_tables"})
    boundary_authority = object()
    state._lateral_boundary_device = ordinary_owner
    state.lateral_boundaries = boundary_authority
    state._scratch.update(lbc_forcing_tables=source, lbc_evaluated_tables=evaluated,
                          lbc_weights_member=np.ones((8, 8), np.float32))
    before = {name: value.tobytes() for name, value in state._scratch.items()}
    snapshot = prepared_host_from_node(node)
    # Exactly the two registered boundary mirrors leave the snapshot; every
    # other scratch slot rides it.  Since development branch (97e9a3ca061) a
    # convection-allowing node also carries the HAILCAST diagnostic slots.
    assert set(snapshot.scratch) == set(before) - {"lbc_forcing_tables", "lbc_evaluated_tables"}
    assert "lbc_weights_member" in snapshot.scratch
    assert {name: value.tobytes() for name, value in state._scratch.items()} == before
    assert state.lateral_boundaries is boundary_authority and state._lateral_boundary_device is ordinary_owner
    ordinary_owner.scratch_slots.remove("lbc_evaluated_tables")
    with pytest.raises(BatchStateUnsupported, match="registered original forcing owner"):
        prepared_host_from_node(node)


@pytest.mark.parametrize("change", [
    {"use_adaptive_time_step": True}, {"mp_physics": 18}, {"sf_sfclay_physics": 5},
    {"bl_pbl_physics": 9}, {"sf_surface_physics": 4}, {"cu_physics": 3},
    {"ra_rrtmg_variant": "rte-rrtmgp"}, {"open_x": True}, {"nested": True},
    {"bldt": 5.0}, {"zadvect_implicit": True}, {"nwp_diagnostics": 1},
    {"tke_budget": True}, {"time_step_sound": 3}, {"km_opt": 3}, {"khdif": 10.0},
    {"hmix_k_diag": True},
])
def test_unqualified_options_have_an_original_fallback_and_are_not_modified(change):
    inputs, node = _source()
    node.cfg.run = replace(node.cfg.run, **change)
    before = vars(node.cfg.run).copy()
    decision = native_prepared_eligibility(inputs, node, members=10)
    assert not decision.eligible and decision.reasons
    assert prepare_native_member_batch(inputs, node, members=10, available_bytes=0, array_module=np) is None
    assert vars(node.cfg.run) == before


def test_streaming_auto_waits_for_the_ordinary_decision_and_does_not_turn_it_off():
    inputs, node = _source()
    inputs.experiment.tiles.mode = "auto"
    assert not native_prepared_eligibility(inputs, members=10).eligible
    assert native_prepared_eligibility(inputs, node, members=10).eligible
    node.state._streamed_domain = object()
    assert not native_prepared_eligibility(inputs, node, members=10).eligible
    assert inputs.experiment.tiles.mode == "auto"
    inputs.experiment.tiles.mode = "on"
    del node.state._streamed_domain
    assert not native_prepared_eligibility(inputs, node, members=10).eligible
    assert inputs.experiment.tiles.mode == "on"


@pytest.mark.parametrize("kind", ["head", "nest", "adaptive_controller", "stepped", "clock_dt", "ozone_parent", "devices"])
def test_runtime_features_remain_original_before_any_native_allocation(kind):
    inputs, node = _source()
    if kind == "head":
        inputs.stream_head = {"source": "as-posted"}
    elif kind == "nest":
        node.children = [object()]
    elif kind == "adaptive_controller":
        node.clock.adaptive_state = object()
    elif kind == "stepped":
        node.clock.step_count, node.clock.ticks = 1, 12
    elif kind == "clock_dt":
        node.clock.dt_fp32 = np.float32(6)
    elif kind == "ozone_parent":
        node.state.physics.radiation_callable._ozone_provider = object()
    elif kind == "devices":
        inputs.experiment.devices.count = 2
    decision = native_prepared_eligibility(inputs, node, members=10)
    assert not decision.eligible and decision.reasons
    original_driver = node.state.physics
    assert prepare_native_member_batch(inputs, node, members=10, available_bytes=0, array_module=np) is None
    assert node.state.physics is original_driver


def test_restart_and_requested_member_files_use_the_original_writer():
    inputs, node = _source()
    assert not native_prepared_eligibility(inputs, node, restart="checkpoint", members=10).eligible
    assert not native_prepared_eligibility(inputs, node, keep_member_files=True, members=10).eligible


def test_mixing_diagnostic_keeps_its_original_scratch_handoff_and_published_names():
    inputs, node = _source()
    node.cfg.run = replace(node.cfg.run, hmix_k_diag=True)
    before = vars(node.cfg.run).copy()
    original_driver = node.state.physics
    decision = native_prepared_eligibility(inputs, node, members=10)
    assert not decision.eligible
    assert any("XKMH/XKHH" in reason for reason in decision.reasons)
    assert prepare_native_member_batch(inputs, node, members=10, available_bytes=0, array_module=np) is None
    assert node.state.physics is original_driver and vars(node.cfg.run) == before
    node.cfg.run = replace(node.cfg.run, hmix_k_diag=False)
    node.state.physics.hmix_k_diag = {"XKMH": np.zeros((8, 8, 8), np.float32)}
    decision = native_prepared_eligibility(inputs, node, members=10)
    assert not decision.eligible and any("initialized mixing diagnostic" in reason for reason in decision.reasons)


def test_snapshot_excludes_reconstructed_owners_and_preserves_all_word_patterns():
    inputs, node = _source()
    words = np.resize(np.array([0, 0x80000000, 0x7FC00017, 0x7F800000], np.uint32), node.state.u.size)
    node.state.u[...] = words.reshape(node.state.u.shape).view(np.float32)
    node.state._lateral_boundary_device = object()
    node.state._scratch_arena = object()
    node.state._dycore_state_workspace = object()
    node.state._scratch["acoustic_cqu"] = np.zeros((8, 8, 9), np.float32)
    extra = BatchArraySpec("batch_trial", (8, 8, 8), "member")
    prepared = prepared_host_from_node(node, extra_specs=(extra,))
    assert prepared.clock["dt_fp32"].tobytes() == node.clock.dt_fp32.tobytes()
    assert prepared.clock["dtbc_fp32"].tobytes() == node.clock.dtbc_fp32.tobytes()
    assert not {"physics", "_lateral_boundary_device", "_scratch_arena", "_dycore_state_workspace"} & prepared.scalars.keys()
    assert prepared.arrays["u"].tobytes() == node.state.u.tobytes()
    assert not np.shares_memory(prepared.arrays["u"], node.state.u)
    shared = tuple(SHARED_STATE_CANDIDATES & prepared.arrays.keys())
    batch = BatchedDomainState.from_prepared((prepared,) * 4, array_module=np, available_bytes=1 << 24,
                                            shared_fields=shared, extra_specs=(extra,))
    for member in range(4):
        assert batch.member_view("u", member).tobytes() == node.state.u.tobytes()
    assert batch.existing_scratch("acoustic_cqu").shape == (4, 8, 8, 9)


def test_a_future_mutable_owner_is_not_silently_lost_during_native_snapshot():
    _inputs, node = _source()
    node.state.new_mutable_owner = object()
    with pytest.raises(BatchStateUnsupported, match="unclassified state metadata"):
        prepared_host_from_node(node)


@pytest.mark.parametrize("timing", ["before_bootstrap", "after_bootstrap"])
def test_native_memory_model_prices_actual_declarations_and_explicit_sampling_basis(monkeypatch, timing):
    import sys
    import types
    import woof.core as core
    from woof.ensemble.admission import MemoryComponent, AllocatorMargin
    from woof.ensemble import batch_moist_dycore
    inputs, node = _source()
    node.state.physics.fields = {"xland": np.ones((8, 8), np.float32), "t2": np.zeros((8, 8), np.float32)}
    legacy = node.state.physics.radiation_callable
    legacy._C, legacy.column_chunk, legacy.o3input, legacy._o33d_grid = {}, 64, 2, None
    radiation = types.ModuleType("woof.core.rrtmg_legacy")
    radiation._lw = SimpleNamespace(NGPTLW=140, LW_BATCH_COLUMN_CHUNK_CEILING=64,
                                    batch_column_chunk=lambda *args, **kwargs: 64)
    radiation._sw = SimpleNamespace(sw_batch_column_chunk=lambda *args, **kwargs: 64)
    seen = []
    def call_peak(**kwargs):
        seen.append(kwargs)
        return 4096 + kwargs["ncol"] * 512 + kwargs["column_chunk"] * 512
    radiation.legacy_radiation_vram_bytes = call_peak
    monkeypatch.setitem(sys.modules, "woof.core.rrtmg_legacy", radiation)
    monkeypatch.setattr(core, "rrtmg_legacy", radiation, raising=False)
    monkeypatch.setattr(batch_moist_dycore, "workspace_specs", lambda cfg: (
        BatchArraySpec("batch_extra", (8, 8, 8), "member"),))
    monkeypatch.setattr(batch_moist_dycore, "required_scratch_slots", lambda cfg: {})
    runtime = MemoryComponent("runtime_context", "runtime", fixed_bytes=4096,
                              basis="measured", evidence="fixture device calibration")
    margin = AllocatorMargin(minimum_bytes=8192, evidence="fixture allocator calibration")
    products = MemoryComponent("products_and_spool", "products", fixed_bytes=2048,
                               per_member_bytes=1024, evidence="fixture product counter and replay plan")
    model, sampling = native_memory_model_from_node(inputs, node, runtime_reservation=runtime,
        allocator_margin=margin, external_components=(products,), bootstrap_live_bytes=16384,
        free_sample_timing=timing, sampled_free_bytes=10**8)
    receipt = model.inventory(10)
    components = {row["name"]: row for row in receipt["components"]}
    assert sampling == {"free_sample_timing": timing, "sampled_free_bytes": 10**8,
                        "bootstrap_live_bytes": 16384, "runtime_reservation_evidence": "fixture device calibration"}
    assert components["ordinary_bootstrap_live"]["required_bytes"] == (16384 if timing == "before_bootstrap" else 0)
    assert components["runtime_context"]["required_bytes"] == 4096
    assert components["products_and_spool"]["required_bytes"] == 12288
    assert receipt["allocator_margin_bytes"] == 8192
    assert components["native_state"]["required_bytes"] > components["native_physics_bank"]["required_bytes"]
    assert components["native_radiation_shared_ozone"]["array_bytes"] == 2048
    assert components["native_radiation_call_peak"]["basis"] == "envelope"
    assert all(row["o3input"] == 0 and row["column_chunk"] == 60 for row in seen)
    boundary = receipt["required_bytes"]
    assert model.largest_that_fits(boundary, max_members=20) == 10
    assert model.largest_that_fits(boundary - 1, max_members=20) == 9


def _forcing(relax_zone=4):
    side = SimpleNamespace(time_law=None)
    field = SimpleNamespace(west=side, east=side, south=side, north=side)
    interval = SimpleNamespace(start_seconds=0.0, end_seconds=3600.0, fields={"u": field, "v": field})
    return SimpleNamespace(intervals=(interval,), spec_bdy_width=5, spec_zone=1, relax_zone=relax_zone, seam_sides=())


def test_member_source_snapshot_carries_state_forcing_and_physics_words_by_path():
    from woof.ensemble.prepared_batch import (NativeMemberSources, snapshot_member_source,
                                               bind_member_sources, member_sources_of)
    inputs, node = _source()
    soil = np.arange(4 * 64, dtype=np.float32).reshape(4, 8, 8)
    node.state.physics.fields = {"soil": soil, "soil_alias": soil, "xland": np.ones((8, 8), np.float32)}
    node.state.lateral_boundaries = _forcing()
    snapshot = snapshot_member_source(node, member_id=3, receipt={"source": "fixture"})
    assert snapshot.member_id == 3 and snapshot.boundaries is node.state.lateral_boundaries
    assert snapshot.prepared.arrays["u"].tobytes() == node.state.u.tobytes()
    assert [(row[0], row[3]) for row in snapshot.physics.structure] == [
        ("driver/fields/soil", "member"), ("driver/fields/xland", "shared")]
    # An alias resolves to the words it shares; nothing aliases the live array.
    assert snapshot.physics.member_array("driver/fields/soil_alias") is snapshot.physics.member_array("driver/fields/soil")
    assert snapshot.physics.member_array("driver/fields/soil") is not soil
    assert snapshot.physics.member_array("driver/fields/soil").tobytes() == soil.tobytes()
    assert snapshot.nbytes > soil.nbytes
    sources = NativeMemberSources(0, (snapshot,))
    assert sources.member_ids == (0, 3) and sources.covers((0, 3)) and not sources.covers((0, 1))
    assert sources.describe(0)["source"].startswith("live ordinary bootstrap")
    assert sources.describe(3)["source"].startswith("own ordinary bootstrap")
    assert sources.describe(3)["host_bytes"] == snapshot.nbytes
    assert sources.describe(3)["bootstrap"] == {"source": "fixture"}
    with pytest.raises(ValueError, match="none for the root"):
        NativeMemberSources(3, (snapshot,))
    assert member_sources_of(node) is None
    bind_member_sources(node, sources)
    assert member_sources_of(node) is sources


def test_member_source_compatibility_names_the_member_and_the_difference():
    from woof.ensemble.prepared_batch import NativeMemberSources, snapshot_member_source
    inputs, node = _source()
    _inputs, other = _source()
    for root in (node, other):
        root.state.physics.fields = {"xland": np.ones((8, 8), np.float32)}
        root.state.lateral_boundaries = _forcing()
    other.state.u[...] = np.float32(3)
    sources = NativeMemberSources(0, (snapshot_member_source(other, member_id=1),))
    assert sources.compatibility_reasons(node) == ()
    other.state.lateral_boundaries = _forcing(relax_zone=3)
    other.state.physics.fields["xland"][...] = np.float32(0)
    sources = NativeMemberSources(0, (snapshot_member_source(other, member_id=1),))
    reasons = sources.compatibility_reasons(node)
    assert len(reasons) == 2 and all(reason.startswith("member 1 ") for reason in reasons)
    assert any("lateral forcing zones" in reason for reason in reasons)
    assert any("shared land surface fields differ from the root and cannot be member-owned: ['driver/fields/xland']" in reason
               for reason in reasons)
    assert sources.member_owned_surface_fields(node) == ()
    # Deep soil temperature from another preparation chain is packed per member instead.
    other.state.lateral_boundaries = _forcing()
    other.state.physics.fields["xland"][...] = np.float32(1)
    for root in (node, other):
        root.state.physics.fields["tmn"] = np.full((8, 8), 285.0, np.float32)
    other.state.physics.fields["tmn"][...] = np.float32(286.5)
    sources = NativeMemberSources(0, (snapshot_member_source(other, member_id=1),))
    assert sources.compatibility_reasons(node) == ()
    assert sources.member_owned_surface_fields(node) == ("tmn",)


def test_single_domain_inputs_count_as_one_domain_for_native_eligibility():
    inputs, node = _source()
    single = SimpleNamespace(experiment=inputs.experiment, stream_head=None)
    assert native_prepared_eligibility(single, node, members=4).eligible

"""Configuration-only component sizing and pre-thread refusal controls."""
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from woof.config import RunConfig
from woof.experiment import DomainConfig
from woof.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
from woof.ensemble.prepared_component_reservation import (
    PreparedDomainAllocationMetadata, plan_prepared_component_reservation)
from woof.ensemble import prepared_component_reservation as inventory


@pytest.fixture
def fixture(monkeypatch):
    from woof.core import preflight
    coefficients = tuple(BatchArraySpec(name + ":metadata_fixture", (3, 4), "shared") for name in
                         ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables"))
    monkeypatch.setattr(inventory, "packaged_coefficient_allocation_specs", lambda: coefficients)
    monkeypatch.setattr(preflight, "_gas_table_meta", lambda: {
        "ngpt_lw": 16, "ngpt_sw": 12, "ngas_lw": 3, "ngas_sw": 4, "nband_lw": 2, "nband_sw": 3})
    cfg = RunConfig(nx=9, ny=9, nz=12, dx=2250.0, dy=2250.0, ztop=10000.0, run_seconds=36.0,
                    moist=True, mp_physics=8, dt=9.0, time_step_sound=3,
                    sf_surface_physics=3, num_soil_layers=6, sf_sfclay_physics=5, bl_pbl_physics=5,
                    ra_physics=4, ra_lw_physics=4, ra_sw_physics=4)
    root = DomainConfig(1, 0, 1, 1, 1, 1, 12.0, cfg)
    child = DomainConfig(2, 1, 2, 2, 3, 3, 12.0, replace(cfg, nx=7, ny=7, dt=3.0))
    exp = SimpleNamespace(domains=(root, child), start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                          feedback=1, smooth_option=2, devices=None, relocation=None)
    inputs = {10: SimpleNamespace(experiment=exp), 21: SimpleNamespace(experiment=exp)}
    metadata = {(member, gid): PreparedDomainAllocationMetadata(5000.0, 32, True, 6, None,
        coefficients, "declared-current-table-identity", True, f"member {member} source header")
        for member in inputs for gid in (1, 2)}
    args = dict(domain_metadata=metadata, ordinary_forecast_bytes={10: 4096, 21: 4096},
        fixed_bytes={"collector": 256, "stochastic": 0, "cuda_owners": 128, "allocator_margin": 256},
        evidence={"ordinary": "complete fixture original forecast envelopes", "collector": "bounded output owner",
            "stochastic": "inactive", "cuda_owners": "explicit queue/event owner", "allocator_margin": "explicit caller reserve"},
        available_bytes=1 << 32)
    return inputs, args


def test_declares_all_native_owners_and_complete_reserves_before_threads(fixture):
    inputs, args = fixture
    decision = plan_prepared_component_reservation(inputs, **args)
    assert decision.eligible, decision.ordinary_reason
    reservation = decision.reservation
    assert set(reservation.native_plans) == {"bank:1", "bank:2", "edge:2", "physics:1", "physics:2"}
    assert reservation.required_bytes == (8192 + 640 + sum(plan.required_bytes(2)
        for plan in reservation.native_plans.values()))
    assert reservation.admit((10, 21)) is None
    reservation.validate_live_plans(reservation.native_plans)
    for gid in (1, 2):
        names = {spec.name for spec in reservation.native_plans[f"physics:{gid}"].arrays}
        assert all("radiation:('radiation', 0):new_table:" + owner + ":metadata_fixture" in names
                   for owner in ("lw_tables", "sw_tables", "lw_cloud_tables", "sw_cloud_tables"))


def test_one_byte_short_component_wave_keeps_admitted_original_wave(fixture):
    inputs, args = fixture
    first = plan_prepared_component_reservation(inputs, **args)
    assert first.eligible, first.ordinary_reason
    result = plan_prepared_component_reservation(inputs, **{**args,
        "available_bytes": first.reservation.required_bytes - 1})
    assert not result.eligible
    assert "original wave remains admitted" in result.ordinary_reason


@pytest.mark.parametrize("native_workspace", [False, True])
def test_radiation_policy_prices_workspace_and_transient_upper_bound(fixture, native_workspace):
    inputs, args = fixture
    metadata = {key: replace(value, native_workspace=native_workspace) for key, value in args["domain_metadata"].items()}
    result = plan_prepared_component_reservation(inputs, **{**args, "domain_metadata": metadata})
    assert result.eligible, result.ordinary_reason
    plan = result.reservation.native_plans["physics:1"]
    assert plan.reserved_bytes > 0
    assert any(spec.name.endswith(":workspace") for spec in plan.arrays) is native_workspace


@pytest.mark.parametrize("selectors", list(product(("wrf_461", "wrf_45"), ("wrf", "air"),
                                                     ("flux", "log_profile"), ("wrf_461", "wrf_45"))))
def test_every_ruc_selector_prices_private_scalar_banks_and_legacy_categories(fixture, selectors):
    inputs, args = fixture
    exp = inputs[10].experiment
    domains = tuple(replace(domain, run=replace(domain.run, ruc_irrigation=selectors[0],
        ruc_qvg_cold_start=selectors[1], ruc_2m_diagnostic=selectors[2], ruc_snow=selectors[3])) for domain in exp.domains)
    for value in inputs.values():
        value.experiment = SimpleNamespace(**{**vars(exp), "domains": domains})
    metadata = {key: replace(value, landuse_categories=4) for key, value in args["domain_metadata"].items()}
    result = plan_prepared_component_reservation(inputs, **{**args, "domain_metadata": metadata})
    assert result.eligible, result.ordinary_reason
    names = {spec.name for spec in result.reservation.native_plans["physics:1"].arrays}
    for name in ("member_qvg_air", "member_irrigation", "member_log_profile"):
        assert "land:('land', 0):workspace:" + name in names
    assert ("land:('land', 0):fields:landusef" in names) is (selectors[0] == "wrf_45")


@pytest.mark.parametrize("field,value,reason", [("sf_surface_physics", 2, "selected land"),
    ("bl_mynn_version", "gsd_41", "GSD cloud"), ("swint_opt", 1, "solar/aerosol"),
    ("aer_opt", 3, "solar/aerosol"), ("alb_sol", 1, "solar/aerosol")])
def test_unbound_scheme_and_active_radiation_policy_return_original_reason(fixture, field, value, reason):
    inputs, args = fixture
    exp = inputs[10].experiment
    domains = tuple(replace(domain, run=replace(domain.run, **{field: value})) for domain in exp.domains)
    for source in inputs.values():
        source.experiment = SimpleNamespace(**{**vars(exp), "domains": domains})
    result = plan_prepared_component_reservation(inputs, **args)
    assert not result.eligible and reason in result.ordinary_reason


def test_unknown_metadata_is_not_interpreted_as_absent_categories(fixture):
    inputs, args = fixture
    result = plan_prepared_component_reservation(inputs, **{**args, "domain_metadata": {}})
    assert not result.eligible and len(result.missing_metadata) == 4
    assert "before threads" in result.ordinary_reason


def test_changed_live_inventory_is_refused_before_component_allocation(fixture):
    from woof.ensemble.batch_state import BatchStateUnsupported
    inputs, args = fixture
    result = plan_prepared_component_reservation(inputs, **args)
    assert result.eligible, result.ordinary_reason
    live = dict(result.reservation.native_plans)
    plan = live["bank:1"]
    first = plan.arrays[0]
    live["bank:1"] = BatchMemoryPlan((replace(first, shape=(first.shape[0] + 1, *first.shape[1:])), *plan.arrays[1:]), 0)
    with pytest.raises(BatchStateUnsupported, match="pre-thread reservation"):
        result.reservation.validate_live_plans(live)


def test_configuration_inventory_does_not_construct_bank_backings(fixture, monkeypatch):
    from woof.ensemble.batch_storage import BatchStorage
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation metadata queried or allocated CUDA state")
    monkeypatch.setattr(BatchStorage, "__init__", forbidden)
    inputs, args = fixture
    result = plan_prepared_component_reservation(inputs, **args)
    assert result.eligible, result.ordinary_reason


@pytest.mark.parametrize("edrates", [0, 1])
def test_cu99_declares_every_gather_return_and_native_call_before_threads(fixture, monkeypatch, edrates):
    inputs, args = fixture
    exp = inputs[10].experiment
    domains = tuple(replace(domain, run=replace(domain.run, cu_physics=99,
                    kf_edrates=edrates)) for domain in exp.domains)
    for value in inputs.values():
        value.experiment = SimpleNamespace(**{**vars(exp), "domains": domains})
    import cupy as cp
    def forbidden(*unused, **kwargs):
        raise AssertionError("configuration-only CU99 sizing initialized CUDA")
    monkeypatch.setattr(cp.cuda, "Device", forbidden)
    result = plan_prepared_component_reservation(inputs, **args)
    assert result.eligible, result.ordinary_reason
    from woof.ensemble.batch_kf import kf_old_column_layout, kf_old_call_reservation
    for domain in domains:
        cfg = domain.run
        plan = result.reservation.native_plans[f"physics:{domain.grid_id}"]
        names = {spec.name for spec in plan.arrays}
        gathered, returned = kf_old_column_layout(cfg.nz, cfg.ny, cfg.nx, edrates=edrates)
        assert {f"cumulus:('cumulus', 0):inputs:{name}" for name in gathered} <= names
        assert {f"cumulus:('cumulus', 0):returned:{name}" for name in returned} <= names
        assert plan.reserved_bytes >= kf_old_call_reservation(cfg.nz, cfg.ny, cfg.nx,
            members=2, edrates=edrates)


@pytest.mark.parametrize("chem_sets", ["smoke", "gocart_primary,smoke", "cams_aq"])
@pytest.mark.parametrize("active_grid", [1, 2])
def test_active_chemistry_keeps_original_owners_before_component_inventory(
        fixture, monkeypatch, chem_sets, active_grid):
    inputs, args = fixture
    original = inputs[10].experiment
    domains = tuple(replace(domain, run=replace(domain.run, chem_sets=chem_sets))
                    if domain.grid_id == active_grid else domain for domain in original.domains)
    for member in inputs.values():
        member.experiment = SimpleNamespace(**{**vars(original), "domains": domains})
    configs = [vars(domain.run).copy() for domain in domains]
    def forbidden(*unused, **kwargs):
        pytest.fail("active chemistry queried component metadata or native banks")
    monkeypatch.setattr(inventory, "prepared_component_allocation_metadata", forbidden)
    monkeypatch.setattr(inventory, "_plans", forbidden)
    args = {key: value for key, value in args.items() if key != "domain_metadata"}
    decision = plan_prepared_component_reservation(inputs, **args)
    assert not decision.eligible and decision.reservation is None
    assert f"domain {active_grid}: active chemistry" in decision.ordinary_reason
    assert "species arrays, source-hour caches and mass ledger" in decision.ordinary_reason
    assert [vars(domain.run) for domain in domains] == configs


def test_missing_coefficient_backing_refuses_incomplete_cold_plan(fixture):
    inputs, args = fixture
    metadata = {key: replace(value, coefficient_uploads=value.coefficient_uploads[:-1])
                for key, value in args["domain_metadata"].items()}
    result = plan_prepared_component_reservation(inputs, **{**args, "domain_metadata": metadata})
    assert not result.eligible and "catalogue differs" in result.ordinary_reason


@pytest.fixture
def prepared_inputs(fixture, monkeypatch):
    from woof.prepared_domain_tree_forecast import PreparedTreeInputs
    inputs, args = fixture
    exp = SimpleNamespace(**vars(inputs[10].experiment), vertical=SimpleNamespace(p_top=1234.5),
                          column_chunk=37, tiles=None)
    monkeypatch.setattr(inventory, "packaged_coefficient_identity", lambda: "current-host-table-content")
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata derivation read a numerical prepared array")
    bundles = []
    for dc in exp.domains:
        reader = SimpleNamespace(metadata={"base_scalars": {"p_top": exp.vertical.p_top}},
            arrays={"surface/" + name: {"shape": [6, dc.run.ny, dc.run.nx]}
                    for name in ("TSLB", "SMOIS", "SH2O")}, read_array=forbidden)
        bundles.append(SimpleNamespace(grid_id=dc.grid_id, parent_id=dc.parent_id,
            cache_reader=reader, static_fields={"LANDUSEF": SimpleNamespace(shape=(21, dc.run.ny, dc.run.nx))},
            authority_sha256={"cache": "fixture-manifest-sha", "static": "fixture-static-sha"}))
    tree = PreparedTreeInputs(prepared_root=Path("prepared"), hierarchy_root=Path("prepared/hierarchy"),
        preparation_receipt_path=Path("prepared/receipt.json"), artifact_receipt_path=Path("prepared/artifacts.json"),
        artifact_manifest_path=Path("prepared/manifest.json"), experiment_config=Path("experiment.toml"),
        experiment=exp, grids=(), domains=tuple(bundles), forcing_hours=(0, 1), boundary_interval_seconds=3600,
        source_identity={}, execution_plan={}, authority_sha256={}, source="mapped")
    return {member: tree for member in inputs}, {key: value for key, value in args.items() if key != "domain_metadata"}


def test_auto_metadata_binds_original_chunk_top_layout_and_no_array_reads(prepared_inputs):
    inputs, args = prepared_inputs
    metadata = inventory.prepared_component_allocation_metadata(inputs)
    assert set(metadata) == {(member, gid) for member in inputs for gid in (1, 2)}
    for row in metadata.values():
        assert (row.p_top, row.radiation_column_chunk, row.soil_layers, row.landuse_categories) == (1234.5, 37, 6, 21)
        assert row.native_workspace and row.resident
        assert row.radiation_table_identity == "current-host-table-content"
        assert "fixture-manifest-sha" in row.evidence
    auto = plan_prepared_component_reservation(inputs, **args)
    explicit = plan_prepared_component_reservation(inputs, **args, domain_metadata=metadata)
    assert auto.eligible, auto.ordinary_reason
    assert auto.reservation.native_plans == explicit.reservation.native_plans
    assert auto.reservation.required_bytes == explicit.reservation.required_bytes


@pytest.mark.parametrize("name", ["TSLB", "SMOIS", "SH2O"])
def test_auto_metadata_refuses_source_soil_shape_before_any_native_plan(prepared_inputs, name):
    inputs, args = prepared_inputs
    inputs[10].domains[0].cache_reader.arrays["surface/" + name]["shape"][0] = 4
    result = plan_prepared_component_reservation(inputs, **args)
    assert not result.eligible and result.reservation is None
    assert "prepared " + name + " shape" in result.ordinary_reason


def test_auto_metadata_refuses_prepared_top_drift(prepared_inputs):
    inputs, args = prepared_inputs
    inputs[10].domains[0].cache_reader.metadata["base_scalars"]["p_top"] = 5000.0
    result = plan_prepared_component_reservation(inputs, **args)
    assert not result.eligible and "prepared scalar top differs" in result.ordinary_reason


def test_configured_tile_road_uses_its_original_decision(prepared_inputs):
    from woof.core.streaming import StreamingOptions, StreamingDecision
    inputs, args = prepared_inputs
    inputs[10].experiment.tiles = StreamingOptions(mode="auto")
    result = plan_prepared_component_reservation(inputs, **args)
    assert not result.eligible and "original cold streaming decision" in result.ordinary_reason
    decisions = {(member, gid): StreamingDecision(False, "original resident admission")
                 for member in inputs for gid in (1, 2)}
    result = plan_prepared_component_reservation(inputs, **args, resident_decisions=decisions)
    assert result.eligible, result.ordinary_reason
    decisions[(10, 1)] = StreamingDecision(True, "original tiled admission")
    result = plan_prepared_component_reservation(inputs, **args, resident_decisions=decisions)
    assert not result.eligible and "streamed members retain original" in result.ordinary_reason


def test_stock_coefficient_identity_covers_same_shape_content_changes(monkeypatch):
    tables = {"lw_tables": SimpleNamespace(coefficients=np.arange(6, dtype=np.float64).reshape(2, 3),
                                          _device={99: object()})}
    monkeypatch.setattr(inventory, "_packaged_coefficient_tables", lambda: tuple(tables.items()))
    first = inventory.packaged_coefficient_identity()
    tables["lw_tables"]._device.clear()
    assert inventory.packaged_coefficient_identity() == first
    tables["lw_tables"].coefficients[0, 0] = 1.0
    assert inventory.packaged_coefficient_identity() != first


def test_prepared_metadata_retains_original_resolved_mynn_width_after_pricing_scope(prepared_inputs, monkeypatch):
    from woof.core import mynn_pbl_scratch
    inputs, args = prepared_inputs
    monkeypatch.setattr(mynn_pbl_scratch, "resolve_mynn_column_chunk", lambda nz: 32)
    metadata = inventory.prepared_component_allocation_metadata(inputs)
    assert all(row.mynn_column_chunk == 32 for row in metadata.values())
    first = plan_prepared_component_reservation(inputs, **args, domain_metadata=metadata)
    monkeypatch.setattr(mynn_pbl_scratch, "resolve_mynn_column_chunk", lambda nz: 100)
    second = plan_prepared_component_reservation(inputs, **args, domain_metadata=metadata)
    assert first.eligible and second.eligible
    assert first.reservation.native_plans == second.reservation.native_plans


def test_original_scratch_reuse_keeps_complete_fallback_envelope_and_only_drops_borrowed_backings(fixture):
    inputs, args = fixture
    args = {**args, "domain_metadata": {key: replace(value, mynn_column_chunk=32)
            for key, value in args["domain_metadata"].items()}}
    own = plan_prepared_component_reservation(inputs, **args)
    reused = plan_prepared_component_reservation(inputs, **args, reuse_original_workspaces=True)
    assert own.eligible and reused.eligible
    assert reused.reservation.ordinary_forecast_bytes == own.reservation.ordinary_forecast_bytes
    assert reused.reservation.fixed_bytes == own.reservation.fixed_bytes
    removed = 0
    for gid in (1, 2):
        first = own.reservation.native_plans[f"physics:{gid}"]
        second = reused.reservation.native_plans[f"physics:{gid}"]
        before = {row["name"]: row for row in first.inventory(2)}
        after = {row["name"]: row for row in second.inventory(2)}
        deleted = set(before) - set(after)
        assert deleted
        assert all(name == "radiation:('radiation', 0):workspace" or
            (name.startswith("pbl:('pbl', 0):workspace:mynn_pbl:") and "out_" not in name)
            for name in deleted)
        assert all(before[name] == after[name] for name in after)
        removed += sum(before[name]["allocated_bytes"] for name in deleted)
        assert first.reserved_bytes == second.reserved_bytes
    assert own.reservation.required_bytes - reused.reservation.required_bytes == removed


def test_edge_only_banks_drop_unreferenced_fields_and_keep_all_original_and_physics_reserves(fixture):
    from woof.ensemble.batch_nesting import prepared_tree_edge_state_fields
    inputs, args = fixture
    full = plan_prepared_component_reservation(inputs, **args)
    subset = plan_prepared_component_reservation(inputs, **args, edge_state_only=True)
    assert full.eligible and subset.eligible
    assert full.reservation.ordinary_forecast_bytes == subset.reservation.ordinary_forecast_bytes
    assert full.reservation.fixed_bytes == subset.reservation.fixed_bytes
    dependencies = prepared_tree_edge_state_fields(inputs[10].experiment.domains)
    removed = 0
    for gid in (1, 2):
        before = {row["name"]: row for row in full.reservation.native_plans[f"bank:{gid}"].inventory(2)}
        after = {row["name"]: row for row in subset.reservation.native_plans[f"bank:{gid}"].inventory(2)}
        assert set(after) == dependencies[gid]
        assert all(before[name] == after[name] for name in after)
        removed += sum(row["allocated_bytes"] for name, row in before.items() if name not in after)
        assert full.reservation.native_plans[f"physics:{gid}"] == subset.reservation.native_plans[f"physics:{gid}"]
    assert full.reservation.native_plans["edge:2"] == subset.reservation.native_plans["edge:2"]
    assert full.reservation.required_bytes - subset.reservation.required_bytes == removed

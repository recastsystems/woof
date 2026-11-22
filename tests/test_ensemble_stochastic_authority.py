"""Forecast-only stochastic overlays retain complete prepared authority."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import SimpleNamespace, MappingProxyType
import json

import pytest

from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from woof.ensemble.stochastic_authority import configure_member_inputs
from woof.ensemble.stochastic_model import StochasticModelProvider


def prepared(tmp_path):
    from woof.prepared_domain_tree_forecast import PreparedTreeInputs, PreparedDomainBundle
    cfg = RunConfig(nx=16, ny=14, nz=8, dx=3000., dy=3000., ztop=20000.,
        dt=12., run_seconds=24., cu_physics=3, cu_gf_version="wrf_461",
        bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=3,
        num_soil_layers=9)  # GF SPP consumes through GFDRV (GFL has no SPP pattern)
    exp = experiment_from_run_config(cfg, datetime(2024, 1, 1, tzinfo=timezone.utc))
    source = MappingProxyType({"cache_header": "a" * 64, "cache_content": "b" * 64})
    bundle = PreparedDomainBundle(1, 0, tmp_path, tmp_path, tmp_path / "static",
        tmp_path / "geometry", tmp_path / "domain", SimpleNamespace(header={"original": object()}),
        MappingProxyType({"original": object()}), MappingProxyType({"immutable_fields": object()}), source)
    result = PreparedTreeInputs(tmp_path, tmp_path, tmp_path / "prepared", tmp_path / "artifact",
        tmp_path / "manifest", tmp_path / "experiment", exp, (object(),), (bundle,), (0, 3), 10800,
        MappingProxyType({"original_source": object()}), MappingProxyType({"physics": "original"}),
        MappingProxyType({"experiment_config": "c" * 64}), "fixture",
        prepared_head_sha256="d" * 64, stream_head=MappingProxyType({"original_head": object()}))
    return result


def test_selected_spp_forecast_flags_reuse_original_prepared_owners_and_record_exact_configs(tmp_path):
    original = prepared(tmp_path)
    controls = {"spp": {"conv": 1, "pbl": 1, "lsm": 1}, "spp_configs": {
        "conv": {}, "pbl": {"stddev": .075}, "lsm": {"stddev": .15}}}
    bound, receipt = configure_member_inputs(StochasticModelProvider.from_mapping(controls), original)
    assert bound is not original
    assert replace(bound.experiment.root.run, spp_conv=0, spp_pbl=0, spp_lsm=0) == original.experiment.root.run
    assert bound.domains is original.domains and bound.grids is original.grids
    assert bound.source_identity is original.source_identity and bound.stream_head is original.stream_head
    assert bound.authority_sha256 is original.authority_sha256
    assert bound.domains[0].cache_identity is original.domains[0].cache_identity
    assert receipt["spp_selectors"] == [{"grid_id": 1,
        "prepared_flags": {"spp_conv": 0, "spp_pbl": 0, "spp_lsm": 0},
        "selected_flags": {"spp_conv": 1, "spp_pbl": 1, "spp_lsm": 1}}]
    assert receipt["prepared_head_sha256"] == "d" * 64
    assert receipt["process_configurations"]["spp"]["pbl"]["stddev"] == .075
    assert receipt["domains"][0]["authority_sha256"] == dict(original.domains[0].authority_sha256)
    assert "header edits" in receipt["prepared_authority_policy"]


def test_all_off_returns_original_inputs_and_creates_no_authority_owner(tmp_path):
    original = prepared(tmp_path)
    provider = StochasticModelProvider.from_mapping({"sppt": False, "skebs": False, "spp": False})
    bound, receipt = configure_member_inputs(provider, original)
    assert bound is original and receipt is None


@pytest.mark.parametrize("change", ["physics", "clock", "float_words", "geometry", "domain_roster"])
def test_overlay_cannot_weaken_full_typed_configuration_guards(tmp_path, change):
    original = prepared(tmp_path)
    selected = original.experiment
    if change == "domain_roster":
        selected = replace(selected, domains=())
    else:
        cfg = selected.root.run
        values = {"physics": {"mp_physics": 8}, "clock": {"dt": 7.5},
                  "float_words": {"clock_dt": -0.0}, "geometry": {"nx": cfg.nx+1}}[change]
        selected = replace(selected, domains=(replace(selected.root, run=replace(cfg, **values, spp_pbl=1)),))
    provider = SimpleNamespace(configure_experiment=lambda exp: selected, enabled_for_experiment=lambda exp: True)
    with pytest.raises(ValueError, match="runtime stochastic overlay"):
        configure_member_inputs(provider, original)
    assert original.experiment.root.run.clock_dt == 0.0 and original.experiment.root.run.spp_pbl == 0


@pytest.mark.parametrize("outcome", [None, RuntimeError("stochastic member failure"), SystemExit(143)])
def test_session_records_runtime_authority_before_any_member_initializer(tmp_path, outcome):
    from contextlib import nullcontext
    from woof.ensemble.production import PreparedEnsembleSession
    from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from woof.ensemble.packing import CardBudget
    from woof.ensemble.runtime_context import current_capture
    from woof.ensemble.lineage import build_lineage
    original = prepared(tmp_path)
    collector = SimpleNamespace(submit=lambda **kwargs: None,
        finish_run=lambda: {"frames": 0}, require_complete=lambda: {})
    provider = StochasticModelProvider.from_mapping({"spp": {"pbl": 1, "lsm": 1}})
    request = {"members": 2, "provenance": [build_lineage({"member": member,
        "source": "gefs", "source_member": f"p{member+1:02d}",
        "cycle": "2024-01-01T00", "seed": 42+member, "rung": "source-member"})
        for member in range(2)]}
    session = PreparedEnsembleSession(request, output_directory=tmp_path, collector=collector,
        cards=(CardBudget(0, 1000),), device_scope=lambda device: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("fixture", "forecast", fixed_bytes=100),)),
        stochastic_provider=provider)
    def runner(bound, **kwargs):
        live = json.loads((tmp_path / "ensemble-run.json").read_text())
        assert live["status"] == "running"
        assert [row["member_id"] for row in live["runtime_stochastic_authorities"]] == [0, 1]
        assert bound.experiment.root.run.spp_pbl == bound.experiment.root.run.spp_lsm == 1
        assert bound.domains is original.domains and bound.source_identity is original.source_identity
        assert current_capture().initialize_callback is not None
        for row in session.request.provenance:
            retained = tmp_path / "members" / f"member-{row['member']:04d}" / "member-provenance.json"
            assert json.loads(retained.read_text()) == row
            stochastic = row["perturbation_recipe"]["runtime_stochastic"]
            assert stochastic["spp_selectors"][0]["selected_flags"] == {
                "spp_conv": 0, "spp_pbl": 1, "spp_lsm": 1}
        if outcome is not None:
            raise outcome
        return {"status": "PASS"}
    if outcome is None:
        receipt = session.run_prepared(runner, original)
        assert receipt["status"] == "PASS"
    else:
        with pytest.raises(type(outcome)) as caught:
            session.run_prepared(runner, original)
        assert caught.value is outcome
        receipt = json.loads((tmp_path / "ensemble-run.json").read_text())
        assert receipt["status"] == ("interrupted" if isinstance(outcome, SystemExit) else "failed")
    assert len(receipt["runtime_stochastic_authorities"]) == 2
    assert receipt["request"]["provenance"] == list(session.request.provenance)
    for row in session.request.provenance:
        retained = tmp_path / "members" / f"member-{row['member']:04d}" / "member-provenance.json"
        assert json.loads(retained.read_text()) == row
    retained_original, configured = session._stochastic_input_bindings[id(original)]
    assert retained_original is original and configured.domains is original.domains


def test_automatic_spp_detects_member_variant_and_preserves_the_off_primary(tmp_path):
    from contextlib import nullcontext
    from woof.ensemble.production import PreparedEnsembleSession
    from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from woof.ensemble.packing import CardBudget
    from woof.ensemble.runtime_context import current_capture
    original = prepared(tmp_path)
    domain = original.experiment.root
    active_exp = replace(original.experiment, domains=(replace(domain,
        run=replace(domain.run, spp_pbl=1)),))
    active = replace(original, experiment=active_exp)
    variants = (original, active)
    collector = SimpleNamespace(submit=lambda **kw: None, finish_run=lambda: {}, require_complete=lambda: {})
    run = PreparedEnsembleSession(2, output_directory=tmp_path / "run", collector=collector,
        input_provider=lambda **kw: variants[kw["member_id"]],
        cards=(CardBudget(0, 1000),), device_scope=lambda device: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("fixture", "forecast", fixed_bytes=100),)))
    def runner(inputs, **kw):
        member = current_capture().member_id
        assert inputs is variants[member]
        assert inputs.experiment.root.run.spp_pbl == member
        assert run.stochastic_provider.enabled_for_experiment(inputs.experiment) is bool(member)
        live = json.loads((tmp_path / "run/ensemble-run.json").read_text())
        assert [row["member_id"] for row in live["runtime_stochastic_authorities"]] == [1]
        assert current_capture().initialize_callback is not None
        return {"status": "PASS"}
    receipt = run.run_prepared(runner, original)
    assert run.stochastic_provider.spp is False
    assert receipt["status"] == "PASS" and original.experiment.root.run.spp_pbl == 0
    assert active.experiment.root.run.spp_pbl == 1


def test_automatic_off_n1_retains_original_inputs_and_no_stochastic_echo(tmp_path):
    from contextlib import nullcontext
    from woof.ensemble.production import PreparedEnsembleSession
    from woof.ensemble.admission import EnsembleMemoryModel, MemoryComponent
    from woof.ensemble.packing import CardBudget
    original = prepared(tmp_path)
    run = PreparedEnsembleSession(1, output_directory=tmp_path / "run",
        collector=SimpleNamespace(submit=lambda **kw: None, finish_run=lambda: {}, require_complete=lambda: {}),
        cards=(CardBudget(0, 1000),), device_scope=lambda device: nullcontext(),
        memory_model=EnsembleMemoryModel((MemoryComponent("fixture", "forecast", fixed_bytes=100),)))
    def runner(inputs, **kw):
        assert inputs is original
        return {"status": "PASS"}
    receipt = run.run_prepared(runner, original)
    assert run.stochastic_provider is None and receipt["runtime_stochastic_authorities"] == []

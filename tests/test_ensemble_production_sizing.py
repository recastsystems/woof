"""Cold card packing, real output reservations and original bootstrap lifetimes."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from woof.config import RunConfig
from woof.experiment import experiment_from_run_config
from woof.core.model import restart_identity_payload
from woof.ensemble.admission import MemoryComponent, AllocatorMargin
from woof.ensemble.batch_product_output import headline_diagnostic_memory_plan, replay_memory_plan_for_shape
from woof.ensemble.batch_products import FieldProducts
from woof.ensemble.packing import CardBudget, MemberBatch
from woof.ensemble.prepared_execution import AutomaticPreparedResult, InitializedCardEvidence
from woof.ensemble.production import PreparedEnsembleSession
from woof.ensemble.production_memory import ordinary_output_memory_components, ordinary_ensemble_memory_model
from woof.ensemble.runtime_context import current_capture
from woof.ensemble.stochastic_model import StochasticModelProvider


@dataclass(frozen=True)
class Inputs:
    experiment: object
    initial: object
    boundaries: object
    stream_head: object = None
    boundary_interval_seconds: float = 3600.


#: These sessions run copies of one prepared input on purpose: they test the
#: session's packing, clocks and receipts, not a forecast ensemble.  A
#: session with no member source refuses N > 1 unless its caller says so.
COPIES = "engine mechanics test: every member runs the one prepared input on purpose"


def prepared():
    cfg = RunConfig(nx=16, ny=14, nz=8, dx=3000., dy=3000., ztop=20000.,
                    dt=12., run_seconds=24.)
    exp = experiment_from_run_config(cfg, datetime(2024, 1, 1, tzinfo=timezone.utc))
    return Inputs(exp, np.arange(16 * 14, dtype=np.float32), (object(), object()))


class Collector:
    def __init__(self, members):
        self.members, self.tile_rows = members, 2
        self.requests = (FieldProducts("wind10", "m s-1", (10.,)),)

    def memory_plan(self, shape, **kwargs):
        return headline_diagnostic_memory_plan(shape, **kwargs)

    def submit(self, **kwargs):
        pass

    def finish_run(self):
        return {"domain_manifests": []}

    def require_complete(self):
        return {}


class DeviceModule:
    """No CUDA import, allocation or field operation in this fixture."""
    def __init__(self, cards=2, free=1_000_000, reusable=0):
        self.local = threading.local()
        self.free, self.reusable = free, reusable
        self.cuda = SimpleNamespace(Device=self.device, runtime=SimpleNamespace(
            getDevice=lambda: getattr(self.local, "device", 0), getDeviceCount=lambda: cards,
            memGetInfo=lambda: (self.free, 100_000_000),
            getDeviceProperties=lambda device: {"name": b"fixture"}))

    @contextmanager
    def device(self, device):
        before = getattr(self.local, "device", 0)
        self.local.device = device
        try:
            yield
        finally:
            self.local.device = before

    def get_default_memory_pool(self):
        return SimpleNamespace(free_bytes=lambda: self.reusable)


def estimate(size):
    values = {"state": size, "scratch": 1024, "physics": 1024}
    domain = SimpleNamespace(category_bytes=lambda category: values.get(category, 0),
        items=tuple(SimpleNamespace(category=category) for category in values))
    subtotal = sum(values.values())
    return SimpleNamespace(domains=(domain,), dycore_state_saved_bytes=0,
        scratch_arena_saved_bytes=0, k_tables_bytes=0, workspace_bytes=0,
        transient_peak_bytes=0, subtotal_bytes=subtotal,
        peak_envelope_bytes=subtotal + 1024, envelope_basis="fixture original forecast and health envelope")


def preflight(monkeypatch, size=10_000):
    import woof.core.preflight as original
    seen = []
    def price(exp, **kwargs):
        seen.append((exp, kwargs))
        return estimate(size)
    monkeypatch.setattr(original, "estimate_experiment", price)
    monkeypatch.setattr(original, "local_memory_profile_from_device", lambda *args, **kwargs:
        SimpleNamespace(name="fixture", resident_thread_capacity=128))
    return seen


def test_output_inventory_prices_distinct_workspaces_max_upload_and_complete_roster():
    inputs, collector = prepared(), Collector(20)
    exp = inputs.experiment
    bigger = replace(exp, domains=(replace(exp.root, run=replace(exp.root.run, nx=20, nz=12)),))
    components = ordinary_output_memory_components(exp, collector, other_experiments=(bigger, exp))
    assert [row.name for row in components] == ["ensemble_diagnostic_qpf_buffers",
        "ensemble_host_upload_peak", "ensemble_full_roster_replay"]
    diagnostic = sum(collector.memory_plan(shape).required_bytes(1) for shape in ((14, 16), (14, 20)))
    assert components[0].inventory(1)["required_bytes"] == diagnostic
    assert components[2].fixed_bytes == replay_memory_plan_for_shape(collector.requests, (14, 20),
        members=20, tile_rows=2).required_bytes(20)
    assert all(row.evidence for row in components)


def test_original_forecast_and_health_peak_is_added_once_with_external_allocator_margin():
    inputs, collector = prepared(), Collector(10)
    model, external, margin = ordinary_ensemble_memory_model(estimate(10_000), inputs.experiment,
        collector, inventory_id="fixture")
    rows = [row.inventory(1) for row in external]
    assert model.required_bytes(1) == 10_000 + 3072 + sum(row["required_bytes"] for row in rows) + margin.required_bytes(rows)
    assert "health" in margin.evidence and "EXTERNAL_MARGIN_BYTES" in margin.evidence


def test_completed_member_cyclone_swaths_still_count_in_next_wave_admission():
    inputs, collector = prepared(), Collector(31)
    collector.cyclone_request = dict(grid_id=1, products=['max-wind', 'rain'],
        common_grid=dict(south=18, north=30, west=-98, east=-80, ny=401, nx=601))
    components = ordinary_output_memory_components(inputs.experiment, collector)
    rows = {row.name: row for row in components}
    retained = rows['cyclone_full_roster_swaths'].inventory(1)['required_bytes']
    # Even one active forecast must reserve maps for all 31 completed/live
    # captures. Pricing only its ordinary history workspace admitted a card
    # too small to hold the final roster's common-grid planes.
    assert retained >= 31 * 401 * 601 * 95
    assert rows['cyclone_swath_call_peak'].fixed_bytes > 401 * 601 * 8
    collector.cyclone_request = None
    assert all(not row.name.startswith('cyclone_')
               for row in ordinary_output_memory_components(inputs.experiment, collector))


def test_default_session_uses_every_visible_card_and_original_member_clocks(tmp_path, monkeypatch):
    seen = preflight(monkeypatch)
    inputs, xp = prepared(), DeviceModule(2, reusable=128)
    observed = []
    def runner(bound, **kwargs):
        assert bound is inputs
        observed.append((current_capture().member_id, xp.cuda.runtime.getDevice(), bound.experiment.root.run.dt))
        return {"status": "PASS"}
    session = PreparedEnsembleSession(4, output_directory=tmp_path, collector=Collector(4), array_module=xp,
                                      identical_members=COPIES)
    result = session.run_prepared(runner, inputs)
    assert sorted(observed) == [(0, 0, 12.), (1, 1, 12.), (2, 0, 12.), (3, 1, 12.)]
    assert len(seen) == 2 and len(result["ordinary_memory_inventory"]) == 2
    assert result["packing"]["waves"] == 2
    assert all(row["own_pool_reusable_bytes"] == 128 for row in result["ordinary_memory_sampling"])
    assert all(any(component["name"] == "ensemble_full_roster_replay"
                   for component in row["components"]) for row in result["ordinary_memory_inventory"])


def test_zero_fit_changes_actual_tile_door_and_keeps_posted_source_physics_clock(tmp_path, monkeypatch):
    preflight(monkeypatch, size=1_000_000)
    inputs = prepared()
    inputs = replace(inputs, stream_head=object())
    before = restart_identity_payload(inputs.experiment)
    observed = []
    def runner(bound, **kwargs):
        assert bound.experiment.tiles.mode == "auto"
        assert bound.experiment.tiles.vram_budget_bytes < 500_000
        assert bound.initial is inputs.initial and bound.boundaries is inputs.boundaries
        assert bound.stream_head is inputs.stream_head
        assert restart_identity_payload(bound.experiment) == before
        assert bound.experiment.root.run is inputs.experiment.root.run
        observed.append(current_capture().member_id)
        return {"status": "PASS"}
    session = PreparedEnsembleSession(2, output_directory=tmp_path, collector=Collector(2),
        array_module=DeviceModule(1, free=500_000), identical_members=COPIES)
    result = session.run_prepared(runner, inputs)
    assert observed == [0, 1] and inputs.experiment.tiles.mode == "off"
    assert all(row["execution_mode"] == "ordinary_streamed_member" for row in result["packing"]["batches"])
    assert all(row["result"]["ensemble_execution_overlay"]["ensemble_withheld_bytes"] > 0
               for row in result["member_results"])


def test_provider_bound_once_and_every_physics_configuration_priced_before_any_initializer(tmp_path, monkeypatch):
    prices = preflight(monkeypatch)
    inputs, order = prepared(), []
    variants = {}
    def provider(*, member_id, **kwargs):
        order.append(("provider", member_id))
        cfg = replace(inputs.experiment.root.run, mp_physics=8 if member_id == 0 else 16)
        exp = replace(inputs.experiment, domains=(replace(inputs.experiment.root, run=cfg),))
        variants[member_id] = replace(inputs, experiment=exp)
        return variants[member_id]
    def runner(bound, **kwargs):
        member = current_capture().member_id
        assert bound is variants[member]
        assert len(prices) == 3
        order.append(("initialized", member))
        return {"status": "PASS"}
    session = PreparedEnsembleSession(3, output_directory=tmp_path, collector=Collector(3),
        array_module=DeviceModule(1), input_provider=provider)
    result = session.run_prepared(runner, inputs)
    assert order[:3] == [("provider", 0), ("provider", 1), ("provider", 2)]
    assert len([row for row in order if row[0] == "provider"]) == 3
    assert result["ordinary_memory_inventory"][0]["components"][0]["name"] == "ordinary_variant_forecast_peak"


def test_default_other_card_bootstrap_uses_original_initializer_and_stops_before_forecast(tmp_path, monkeypatch):
    preflight(monkeypatch)
    base = prepared()
    @dataclass(frozen=True)
    class NativeInputs(Inputs):
        execution_plan: str = "fixture"
    inputs = NativeInputs(base.experiment, base.initial, base.boundaries)
    xp, clocks, original_calls = DeviceModule(3), [], []
    import woof.ensemble.prepared_execution as prepared_execution
    def sample(inputs, array_module=None):
        device = xp.cuda.runtime.getDevice()
        return InitializedCardEvidence(CardBudget(device, 100_000, 1_000_000, "fixture"),
            MemoryComponent("runtime", "runtime", fixed_bytes=100, evidence="fixture runtime"),
            AllocatorMargin(minimum_bytes=10, evidence="fixture margin"), 1000,
            evidence="fixture after original bootstrap")
    monkeypatch.setattr(prepared_execution, "sample_initialized_card_evidence", sample)
    def runner(bound, *, ensemble_bootstrap, output_directory, **kwargs):
        assert bound is inputs and output_directory.is_dir()
        assert current_capture() is None
        clock = SimpleNamespace(step_count=0, elapsed_seconds=0.)
        clocks.append(clock)
        node = SimpleNamespace(clock=clock, state=object())
        original_calls.append(xp.cuda.runtime.getDevice())
        result = ensemble_bootstrap(inputs=bound, model=SimpleNamespace(root=node), node=node,
            output_directory=output_directory, observer=None, step_observer=None)
        assert result is not None, "bootstrap must bypass the original step loop"
        return result
    def factory(original, **kwargs):
        assert kwargs["request"].member_device_ids == (0, 1, 2)
        assert kwargs["stochastic_enabled"] is False
        def run(shared, **options):
            for device in (1, 2):
                result = kwargs["card_bootstrap_factory"](device_id=device, shared_inputs=shared,
                                                          request=kwargs["request"])
                assert result.node.clock.step_count == 0
            report = {"status": "PASS", "admission": {"packing": {"fixture": "three original cards"}},
                      "member_results": []}
            return AutomaticPreparedResult("native_complete", report, 0, 0, tmp_path, {})
        return run
    monkeypatch.setattr(prepared_execution, "make_automatic_prepared_executor", factory)
    session = PreparedEnsembleSession({"members": 6, "stochastic": {"sppt": False, "skebs": False, "spp": False}},
        output_directory=tmp_path, collector=Collector(6), array_module=xp, identical_members=COPIES)
    result = session.run_prepared(runner, inputs)
    assert original_calls == [1, 2] and clocks[0] is not clocks[1]
    assert [row["device_id"] for row in result["additional_card_bootstraps"]] == [1, 2]
    assert all(row["forecast_steps"] == 0 for row in result["additional_card_bootstraps"])
    assert not list(tmp_path.rglob("wrfout_*"))


def test_spp_true_without_declared_consumers_fails_before_any_initializer(tmp_path):
    session = PreparedEnsembleSession({"members": 2, "stochastic": {"spp": True}},
        output_directory=tmp_path, collector=Collector(2), array_module=object())
    with pytest.raises(ValueError, match="requires selected spp_conv"):
        session.run_prepared(lambda **kw: pytest.fail("initializer was entered"), prepared())


def test_stochastic_memory_declares_original_words_2d_spp_and_measured_fft_workspace():
    inputs = prepared()
    # GF SPP needs the previous pin's GFDRV (WRF v4.8.0's GFL carries no SPP
    # pattern, so spp_conv with the GFL default is refused by name).
    cfg = replace(inputs.experiment.root.run, cu_physics=3, cu_gf_version="wrf_461",
        bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=3,
        spp_conv=1, spp_pbl=1, spp_lsm=1)
    exp = replace(inputs.experiment, domains=(replace(inputs.experiment.root, run=cfg),))
    provider = StochasticModelProvider.from_mapping({"sppt": True, "skebs": True, "spp": True})
    assert provider.configure_experiment(exp) is exp
    rows = [component.inventory(1) for component in provider.memory_components(exp, fft_workspace_bytes=12345)]
    assert rows[-1]["basis"] == "measured" and rows[-1]["required_bytes"] == 12345
    source = rows[0]["arrays"]
    assert len([row for row in source if row["dtype"] == np.dtype(np.complex64).str]) >= 6
    assert all(len(row["shape"]) == 2 for row in source if row["name"].startswith("spp:"))
    assert all(row["basis"] == "envelope" for row in rows[:-1])
    off = StochasticModelProvider.from_mapping({"sppt": False, "skebs": False, "spp": False})
    assert not off.enabled_for_experiment(inputs.experiment)
    assert off.memory_components(inputs.experiment, fft_workspace_bytes=0) == ()
    assert off.sample_fft_workspace_bytes(inputs.experiment, array_module=object()) == 0


def test_streamed_stochastic_admission_prices_local_windows_not_resident_rate_volumes(monkeypatch):
    from woof.core import streaming
    from woof.ensemble.production_memory import ordinary_stochastic_streamed_inputs
    inputs = prepared()
    cfg = replace(inputs.experiment.root.run, nx=64, ny=64, nz=50)
    inputs = replace(inputs, experiment=replace(inputs.experiment,
        domains=(replace(inputs.experiment.root, run=cfg),)))
    provider = StochasticModelProvider.from_mapping({"sppt": True})
    output = ordinary_output_memory_components(inputs.experiment, Collector(4))
    full = output + provider.memory_components(inputs.experiment, fft_workspace_bytes=1024)
    assert sum(row.inventory(1)["required_bytes"] for row in full) > 4_000_000
    original = restart_identity_payload(inputs.experiment)
    budgets = []
    monkeypatch.setattr(streaming, "planner_machine", lambda **kwargs: kwargs)
    def decide(cfg, options, **kwargs):
        assert kwargs["allow_resident"] is False
        budgets.append(options.vram_budget_bytes)
        return streaming.StreamingDecision(True, "fixture original planner", tile_nx=16,
            tile_ny=16, halo=4, nbuffers=2)
    monkeypatch.setattr(streaming, "decide", decide)
    monkeypatch.setattr(streaming, "tile_specs", lambda *args: [SimpleNamespace(cny=24, cnx=24),
                                                              SimpleNamespace(cny=20, cnx=24)])
    batch = MemberBatch(0, 0, (0,), "ordinary_streamed_member", None, 4_000_000)
    admitted, receipt = ordinary_stochastic_streamed_inputs(inputs, batch=batch,
        external_components=full, allocator_margin=AllocatorMargin(evidence="fixture margin"),
        stochastic_provider=provider, fft_workspace_bytes=1024)
    assert len(budgets) >= 2 and budgets == sorted(budgets, reverse=True)
    assert restart_identity_payload(admitted.experiment) == original
    assert admitted.initial is inputs.initial and admitted.boundaries is inputs.boundaries
    assert receipt["stochastic_window_admission"]["decisions"][0]["window_shapes"] == ((20, 24), (24, 24))
    rates = next(row for row in receipt["external_components"] if row["name"] == "stochastic_rate_peak_d01")
    assert all(array["shape"][-2] <= 25 and array["shape"][-1] <= 25 for array in rates["arrays"])
    assert any("buffer1:shape1:" in array["name"] for array in rates["arrays"])


def test_session_progress_combines_members_and_retains_checkpoint_owners(tmp_path, monkeypatch):
    preflight(monkeypatch)
    events, inputs = [], prepared()
    def runner(bound, **options):
        member = current_capture().member_id
        callback = options["progress_callback"]
        callback(model_elapsed_seconds=12., outer_step=1, last_checkpoint=f"member-{member}.nc")
        callback(model_elapsed_seconds=24., outer_step=2)
        callback.complete(24.)
        return {"status": "PASS", "completed_seconds": 24.}
    session = PreparedEnsembleSession(4, output_directory=tmp_path, collector=Collector(4),
        array_module=DeviceModule(1), identical_members=COPIES)
    result = session.run_prepared(runner, inputs, progress_callback=lambda **event: events.append(event))
    assert [event["model_elapsed_seconds"] for event in events] == sorted(event["model_elapsed_seconds"] for event in events)
    assert all(event["last_checkpoint"] is None for event in events)
    rows = result["ensemble_progress"]["members"]
    assert [row["elapsed_seconds"] for row in rows] == [24.] * 4
    assert [row["last_checkpoint"] for row in rows] == [f"member-{member}.nc" for member in range(4)]
    assert events[-1]["outer_step"] == 8


def test_prepared_tree_uses_observer_api_without_adding_runtime_callback_keyword(tmp_path, monkeypatch):
    preflight(monkeypatch)
    inputs, events = prepared(), []
    def runner(bound, *, output_directory, first_products, observer=None):
        observer(model_elapsed_seconds=24., outer_step=2)
        return {"status": "PASS"}
    session = PreparedEnsembleSession(2, output_directory=tmp_path, collector=Collector(2),
        array_module=DeviceModule(1), identical_members=COPIES)
    report = session.run_prepared(runner, inputs, observer=lambda **event: events.append(event))
    assert [event["model_elapsed_seconds"] for event in events] == [12., 24.]
    assert report["ensemble_progress"]["members"][1]["elapsed_seconds"] == 24.


def test_config_runtime_summary_binds_the_final_manifest_bytes(tmp_path):
    from woof.ensemble.admission import EnsembleMemoryModel
    import hashlib
    inputs = prepared()
    model = EnsembleMemoryModel((MemoryComponent("fixture", "forecast", fixed_bytes=10),))
    session = PreparedEnsembleSession(1, output_directory=tmp_path, collector=Collector(1),
        cards=(CardBudget(0, 1_000_000),), memory_model=model, device_scope=lambda _: nullcontext())
    summary = session.run_experiment(lambda *args, **kwargs: {"status": "PASS"},
        inputs.experiment, SimpleNamespace(), tmp_path)
    assert summary.ensemble_manifest == tmp_path / "ensemble-run.json"
    assert summary.ensemble_manifest_sha256 == hashlib.sha256(summary.ensemble_manifest.read_bytes()).hexdigest()


def test_bootstrap_array_ownership_guard_uses_only_metadata():
    from woof.ensemble.production_memory import require_bootstrap_device
    class Array:
        def __init__(self, device):
            self.device = SimpleNamespace(id=device)
        def get(self):
            pytest.fail("weather array transfer")
    State = type("State", (), {"__module__": "woof.core.state"})
    state = State()
    state.fields = {"state": Array(1)}
    node = SimpleNamespace(state=state)
    require_bootstrap_device(node, array_module=SimpleNamespace(ndarray=Array), device_id=1)
    with pytest.raises(ValueError, match="another physical card"):
        require_bootstrap_device(node, array_module=SimpleNamespace(ndarray=Array), device_id=0)


@pytest.fixture
def hydro_members(tmp_path, monkeypatch):
    """Real metadata plans and session orchestration; no forecast/device execution."""
    import sys
    from types import ModuleType
    from woof.hydro.memory import ArrayMemory, StageMemory
    from woof.hydro import runtime
    from woof.ensemble import nextgen_capture
    from woof import numerics
    # Metadata resolution fixes a process mode too. Restore it along with the
    # environment so these session checks cannot select later tests' numerics.
    monkeypatch.setattr(numerics, "_mode", None)
    monkeypatch.setenv(numerics.ENV, "fast")
    monkeypatch.delenv("GPUWM_WRF_EXACT", raising=False)
    monkeypatch.delenv("WOOF_NEXTGEN_ENABLED", raising=False)
    monkeypatch.delenv("WOOF_NEXTGEN_SETTINGS", raising=False)
    module = ModuleType("woof.hydro._ensemble_memory_test")
    module.prepare = lambda **kw: pytest.fail("hydro construction entered during metadata admission")
    module.prepare_memory = lambda **kw: StageMemory((
        ArrayMemory("storage", (kw["options"]["cells"],), "float64"),), "fixture metadata", 0)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    class MetadataCollector:
        def __init__(self, original, member_ids, **unused):
            self.original = original
            self.nextgen_points = SimpleNamespace(options={"device_budget_bytes": 4096}, receipt=lambda: {})
            self.resident_inventory = SimpleNamespace(receipt=lambda: {})
        def __getattr__(self, name):
            return getattr(self.original, name)
    monkeypatch.setattr(nextgen_capture, "NextGenCollector", MetadataCollector)
    @dataclass(frozen=True)
    class HydroInputs(Inputs):
        experiment_config: object = None
    base = prepared()
    configs = []
    for index, cells in enumerate((64, 4096)):
        path = tmp_path / f"member-config-{index}.toml"
        path.write_text('[hydro]\nenabled = true\n[[hydro.stages]]\nname = "runoff"\n'
                        f'factory = "{module.__name__}:prepare"\n[hydro.stages.options]\ncells = {cells}\n')
        configs.append(path)
    members = tuple(HydroInputs(base.experiment, base.initial, base.boundaries,
                               experiment_config=configs[index]) for index in (0, 1, 0))
    assert all(member.experiment is base.experiment for member in members)
    return base, members, runtime


@pytest.mark.parametrize("ambient", [None, {"enabled": True, "stages": []}])
def test_every_member_hydro_plan_is_priced_before_initialization(hydro_members, tmp_path, monkeypatch, ambient):
    from woof.hydro.memory import forecast_plan
    from woof.core import preflight as original
    base, members, runtime = hydro_members
    preflight(monkeypatch)
    expected = []
    for member in members[:2]:
        with runtime.settings_scope(runtime.read_settings(member.experiment_config)):
            expected.append(forecast_plan(base.experiment).inventory()["required_bytes"])
    priced = []
    def price(exp, **kwargs):
        value = forecast_plan(exp)
        size = 0 if value is None else value.inventory()["required_bytes"]
        priced.append(size)
        return estimate(10_000 + size)
    monkeypatch.setattr(original, "estimate_experiment", price)
    initialized = []
    def runner(bound, **kwargs):
        assert priced == expected  # including distinct hydro on the SAME Experiment
        initialized.append(current_capture().member_id)
        return {"status": "PASS"}
    session = PreparedEnsembleSession(3, output_directory=tmp_path / "run", collector=Collector(3),
        array_module=DeviceModule(1), input_provider=lambda member_id, **kw: members[member_id])
    with runtime.settings_scope(ambient):
        result = session.run_prepared(runner, base)
        assert runtime.current_settings() is ambient
    assert initialized == [0, 1, 2]
    assert priced == expected and expected[0] < expected[1]
    peak = result["ordinary_memory_inventory"][0]["components"][0]
    assert peak["name"] == "ordinary_variant_forecast_peak"
    assert peak["required_bytes"] >= expected[1]


def test_member_hydro_scope_reaches_ordinary_tile_admission(hydro_members, tmp_path, monkeypatch):
    from woof.ensemble import prepared_execution
    base, members, runtime = hydro_members
    preflight(monkeypatch)
    original = prepared_execution.ordinary_member_execution_inputs
    seen = []
    def tile_inputs(inputs, **kwargs):
        settings = runtime.current_settings()
        seen.append(None if settings is None else settings.get("settings_path"))
        return original(inputs, **kwargs)
    monkeypatch.setattr(prepared_execution, "ordinary_member_execution_inputs", tile_inputs)
    session = PreparedEnsembleSession(3, output_directory=tmp_path / "run", collector=Collector(3),
        array_module=DeviceModule(1), input_provider=lambda member_id, **kw: members[member_id])
    with runtime.settings_scope(None):
        session.run_prepared(lambda *args, **kwargs: {"status": "PASS"}, base)
        assert runtime.current_settings() is None
    assert seen == [str(member.experiment_config) for member in members]

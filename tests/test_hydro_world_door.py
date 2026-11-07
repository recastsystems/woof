"""Default-on hydro door decision table, receipts, cost budget and products (CPU)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.ensemble.optional_routing import activation_receipt, member_settings
from woof.hydro import cost, door, forecast_products, runtime
from woof.hydro.config import configure
from woof.static import rust_bridge

ROOT = Path(__file__).resolve().parents[1]
CONUS = ROOT / "configs" / "conus3km.toml"

#: The door reads a real domain's parent grid through the native static-fields
#: reader. Breakage (2.9.4 cut, science stage, native-free host tree): four door
#: tests on configs/conus3km.toml failed "parent grid unreadable" instead of
#: skipping where that library is not built; Stage 1 runs them with the natives.
_STATIC_BRIDGE_UNAVAILABLE = rust_bridge.unavailable_reason()
needs_static_bridge = pytest.mark.skipif(
    _STATIC_BRIDGE_UNAVAILABLE is not None,
    reason=("the native static-fields bridge is not built, so the door cannot read "
            f"the domain's parent grid here: {_STATIC_BRIDGE_UNAVAILABLE}"))


def test_provider_signal_death_reports_the_exit_instead_of_an_empty_failure(tmp_path, monkeypatch):
    from woof.hydro import provider
    recipe = tmp_path / 'recipe.json'
    recipe.write_text('{}')
    monkeypatch.setattr(provider, 'recipe_path', lambda row: recipe)
    monkeypatch.setattr(provider, '_tool', lambda spec: '/unused/tool')
    with pytest.raises(RuntimeError, match=r'return code -9 \(SIGKILL\): no tool diagnostic'):
        provider.prepare(row={'preparation': {'parameters': {}}},
                         footprint={'grid_key': 'grid', 'west': 0, 'south': 0, 'east': 1, 'north': 1},
                         out=tmp_path, config={'geopackage': '/unused/fabric.gpkg'},
                         runner=lambda *args, **kwargs: SimpleNamespace(returncode=-9, stdout='', stderr=''))
    stage = json.loads((tmp_path / 'provider-stages.json').read_text())[-1]
    assert stage['stage'] == 'parameters' and stage['returncode'] == -9


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (runtime.ENV_ENABLED, runtime.ENV_SETTINGS, door.ENV_DECISION,
                 door.ENV_TABLE, door.ENV_CACHE, "WOOF_RW_HYDRO_WORLD"):
        monkeypatch.delenv(name, raising=False)


def box_geometry(south=30.0, north=31.0, west=-98.0, east=-97.0, lsm=4, center=None):
    points = [(south + (north - south) * j / 4, west + (east - west) * i / 4)
              for j in range(5) for i in range(5)]
    return door.geometry_from_points([p[0] for p in points], [p[1] for p in points],
                                     center_lon=(west + east) / 2 if center is None else center,
                                     parent_lsm=lsm, grid_key="k" * 64)


def prepared_table(tmp_path, *, coverage=((-99.0, 29.0, -96.0, 32.0),), licence="verified",
                   realization="lower-colorado-hyfeatures"):
    manifest = tmp_path / "prepared" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema": "arwen.nextgen.hydrofabric.v1"}))
    shipped = json.loads(door.TABLE.read_text())
    shipped["providers"] = [{
        "id": "test-prepared", "owner": "test", "priority": 1, "kind": "prepared",
        "coverage": [list(box) for box in coverage], "min_covered_fraction": 0.0,
        "realization": realization, "manifest_binding": "fabric_manifest",
        "manifest": str(manifest),
        "data_licence": {"status": licence, "name": "test terms", "reason": "test row",
                         "attribution": "test"}}] + shipped["providers"]
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(shipped))
    return path, manifest


def conus_copy(tmp_path, lsm):
    soil = chr(10) + "num_soil_layers = 9" if lsm == 3 else ""  # RUC has no default soil geometry
    text = CONUS.read_text().replace("sf_surface_physics = 2", f"sf_surface_physics = {lsm}{soil}")
    path = tmp_path / f"conus-lsm{lsm}.toml"
    path.write_text(text)
    return path


# -- decision table -----------------------------------------------------------

def test_fabric_present_turns_hydro_on_with_bound_realization(tmp_path, monkeypatch):
    monkeypatch.delenv(door.ENV_CACHE, raising=False)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    table, manifest = prepared_table(tmp_path)
    decision = door.decide(settings={}, requested="", table=table, geometry=box_geometry())
    assert decision["enabled"] is True and decision["reason"] == "fabric_covered"
    assert decision["provider"] == "test-prepared"
    assert decision["candidates"][0]["covered_fraction"] == 1.0
    composed, path, build = door.materialize(decision, config_path=tmp_path / "x.toml")
    settings = runtime.read_settings(path)
    assert settings["enabled"] is True and settings["activation_reason"] == "fabric_covered"
    expanded = configure(settings)
    assert expanded["realization_identity"]["id"] == "lower-colorado-hyfeatures"
    assert expanded["stages"][0]["options"]["manifest"] == str(manifest)
    assert "door" not in settings  # never inside the sealed realization identity


def test_start_up_state_lands_under_the_data_root_never_beside_the_config(tmp_path, monkeypatch):
    """2.9.4 defect: the gfs_12km quickstart wrote ~1.5 GB of hydro start-up
    state into a checkout's configs/.gpuwm-hydro. The fabric, its warm water
    state and hydro-door.toml go under the user data root and are reused there."""
    monkeypatch.delenv(door.ENV_CACHE, raising=False)
    data_root = tmp_path / "case-data"
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(data_root))
    configs = tmp_path / "checkout" / "configs"
    configs.mkdir(parents=True)
    config = configs / "gfs_12km.toml"
    config.write_text("# quickstart\n")
    before = sorted(p.relative_to(configs) for p in configs.rglob("*"))
    table, _ = prepared_table(tmp_path)
    decision = door.decide(settings={}, requested="", table=table, geometry=box_geometry())
    _, path, _ = door.materialize(decision, config_path=config)
    assert sorted(p.relative_to(configs) for p in configs.rglob("*")) == before
    assert not (configs / ".gpuwm-hydro").exists()
    assert door.cache_root(config) == data_root / door.CACHE_DIRNAME
    assert path.is_file() and path.is_relative_to(data_root / door.CACHE_DIRNAME)
    other = tmp_path / "elsewhere" / "copy.toml"
    other.parent.mkdir()
    other.write_text("# quickstart\n")
    _, again, _ = door.materialize(decision, config_path=other)
    assert again == path  # keyed by the request, so any configuration reuses it


def _tools_present(row):
    return []


def _tools_missing(row):
    return ["builder executable rw_hydro_world not found (set WOOF_RW_HYDRO_WORLD or build tools/hydro_world)"]


@needs_static_bridge
@pytest.mark.parametrize("lsm,realization", [(2, "parent-noah-routed"), (3, "parent-ruc-routed"),
                                             (4, "parent-noahmp-routed")])
def test_conus_default_suites_attach_through_native_provider(tmp_path, lsm, realization):
    """Native recipe admission must select the same proven parent-LSM realization."""
    decision = door.decide(settings={}, requested="", config_path=conus_copy(tmp_path, lsm),
                           probe_row=_tools_present)
    assert decision["enabled"] is True and decision["reason"] == "fabric_covered"
    assert decision["provider"] == "conus-nhdplus-v21"
    assert decision["realization"] == realization
    assert decision["candidates"][0]["verdict"] == "available"
    assert "CONUS land" in decision["coverage_scope"]


def _hrrr_v4_geometry(lsm=3):
    """The HRRR v4 parent grid (the 1799 x 1059 native mass grid, which reaches
    into Canada and Mexico), sampled on the door's own lattice."""
    import numpy as np
    from woof.ingest.hrrr import hrrr_source_grid
    from woof.ingest.hrrr_target import HRRR_SOURCE_NX, HRRR_SOURCE_NY
    grid = hrrr_source_grid()
    x, y = np.meshgrid(np.linspace(1.0, HRRR_SOURCE_NX, door.LATTICE),
                       np.linspace(1.0, HRRR_SOURCE_NY, door.LATTICE))
    lat, lon = grid.ij_to_latlon(x.ravel(), y.ravel())
    return door.geometry_from_points(np.asarray(lat, float).tolist(), np.asarray(lon, float).tolist(),
                                     center_lon=float(grid.cen_lon), parent_lsm=lsm, grid_key="h" * 64)


def test_hrrr_v4_grid_gets_a_covering_fabric_by_default():
    """A 0.809-covered parent takes native CONUS land, with scope explicit."""
    decision = door.decide(settings={}, requested="", geometry=_hrrr_v4_geometry(),
                           probe_row=_tools_present)
    rows = {c["id"]: c for c in decision["candidates"]}
    assert 0.7 < rows["conus-nhdplus-v21"]["covered_fraction"] < 1.0
    assert rows["hydro-world-global"]["covered_fraction"] == 1.0
    assert decision["enabled"] is True and decision["reason"] == "fabric_covered"
    assert decision["provider"] == "conus-nhdplus-v21"
    assert decision["realization"] == "parent-ruc-routed"
    shipped = {row["id"]: row for row in json.loads(door.TABLE.read_text())["providers"]}
    licence = shipped["conus-nhdplus-v21"]["data_licence"]
    assert licence["status"] == "verified"
    assert "NHDPlusV2_metadata.htm" in licence["reason"] and "Use_Constraints: None" in licence["reason"]
    line = door.launch_line(decision)
    assert line.startswith("go: hydro -- on: fabric conus-nhdplus-v21, covering 0.")
    assert "CONUS land catchments" in line


def test_launch_line_states_the_reason_when_off():
    decision = door.decide(settings={}, requested="", geometry=_hrrr_v4_geometry(),
                           probe_row=_tools_missing)
    line = door.launch_line(decision)
    assert line.startswith("go: hydro -- off (no_covering_fabric): ")
    assert "[hydro-world-global] builder executable rw_hydro_world not found" in line
    assert door.launch_line(door.decide(settings={}, requested="off")) == "go: hydro -- off (cli_off)"


def test_fabric_absent_noahmp_conus_names_every_provider_gap(tmp_path):
    decision = door.decide(settings={}, requested="", config_path=conus_copy(tmp_path, 4),
                           probe_row=_tools_missing)
    assert decision["enabled"] is False and decision["reason"] == "no_covering_fabric"
    rows = {c["id"]: c for c in decision["candidates"]}
    assert rows["conus-nhdplus-v21"]["covered_fraction"] == 1.0
    assert rows["hydro-world-global"]["covered_fraction"] == 1.0
    text = json.dumps(decision["gaps"])
    assert "data licence pending" not in text and "no admitted coupled realization" not in text
    assert "rw_hydro_world" in text
    assert {gap["owner"] for gap in decision["gaps"]} == {"hydro-data", "hydro-world-fabric"}


def test_global_tile_root_is_acquired_not_a_gap(monkeypatch, tmp_path):
    """A fresh install has no source tiles; the global row fetches them at preparation
    (woof.hydro.acquire), so a missing default root is not a rejection."""
    shipped = json.loads(door.TABLE.read_text())
    row = next(r for r in shipped["providers"] if r["id"] == "hydro-world-global")
    need = row["builder"]["paths"][0]
    assert need["default"] and need["acquire"] == "woof.hydro.acquire:acquire"
    monkeypatch.setenv(need["env"], str(tmp_path / "not-yet-fetched"))
    monkeypatch.setattr(door, "_executable", lambda spec: tmp_path / spec["filename"])
    monkeypatch.setattr(door.subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=0, stdout="usage: build | admit", stderr=""))
    reasons = door.probe(row)
    assert not any("not available" in reason for reason in reasons), reasons


def test_partial_cover_falls_through_to_next_row(tmp_path):
    table, _ = prepared_table(tmp_path, coverage=((-97.5, 29.0, -96.0, 32.0),))
    rows = json.loads(table.read_text())
    rows["providers"][0]["min_covered_fraction"] = 1.0
    table.write_text(json.dumps(rows))
    decision = door.decide(settings={}, requested="", table=table, geometry=box_geometry())
    assert decision["enabled"] is False
    assert decision["candidates"][0]["covered_fraction"] == 0.6
    assert "covers 0.600" in decision["candidates"][0]["reasons"][0]


def test_unverified_licence_or_missing_realization_never_turns_on(tmp_path):
    for kwargs, needle in (({"licence": "pending"}, "data licence pending"),
                           ({"licence": "research-only"}, "data licence research-only (test terms)"),
                           ({"realization": None}, "no admitted coupled realization"),
                           ({"realization": "no-such-row"}, "unknown realization")):
        table, _ = prepared_table(tmp_path, **kwargs)
        decision = door.decide(settings={}, requested="", table=table, geometry=box_geometry())
        assert decision["enabled"] is False
        assert needle in json.dumps(decision["candidates"][0]["reasons"])


def test_dateline_domain_is_covered_by_dateline_row(tmp_path):
    table, _ = prepared_table(tmp_path, coverage=((170.0, -20.0, -170.0, -10.0),))
    geometry = box_geometry(south=-18, north=-12, west=175.0, east=185.0, center=180.0)
    geometry["points"] = [(lat, door._wrap180(lon)) for lat, lon in geometry["points"]]
    assert geometry["crosses_dateline"] is True
    decision = door.decide(settings={}, requested="", table=table, geometry=geometry)
    assert decision["enabled"] is True


def test_opt_out_never_reads_the_table_or_the_domain(tmp_path):
    decision = door.decide(settings={"enabled": False, "activation_reason": "cli_off"},
                           requested="off", table=tmp_path / "absent.json",
                           geometry_reader=lambda path: pytest.fail("geometry read"))
    assert decision == {"schema": door.SCHEMA, "requested": "off", "enabled": False,
                        "reason": "cli_off", "gaps": []}


def test_explicit_settings_pass_through_unchanged(tmp_path):
    decision = door.decide(settings={"enabled": True, "realization": "x",
                                     "activation_reason": "cli_enabled"},
                           requested="s.toml", table=tmp_path / "absent.json")
    assert decision["enabled"] is True and decision["fabric"] == "explicit settings"


def test_bare_hydro_flag_without_fabric_refuses_with_gaps(tmp_path):
    with pytest.raises(door.HydroDoorRefusal, match="no hydrofabric covers") as error:
        door.resolve_launch({"enabled": True, "activation_reason": "cli_enabled"},
                            requested="", config_path=conus_copy(tmp_path, 4))
    assert "hydro-data" in str(error.value)


def test_unreadable_domain_is_a_named_gap(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("")
    decision = door.decide(settings={}, requested="", config_path=path)
    assert decision["reason"] == "no_domain_geometry"
    assert str(path) in decision["gaps"][0]["gap"]


# -- launcher, runner and ensemble members ------------------------------------

def _launch_args(path, hydro=None):
    return SimpleNamespace(config=path, experiment_config=path, hydro=hydro)


def test_go_scope_receipt_carries_door_decision_and_member_state(tmp_path, monkeypatch):
    table, manifest = prepared_table(tmp_path)
    monkeypatch.setenv(door.ENV_TABLE, str(table))
    monkeypatch.setattr(door, "TABLE", Path(tmp_path / "only-extra.json"))
    shipped = json.loads(table.read_text())
    (tmp_path / "only-extra.json").write_text(json.dumps({**shipped, "providers": []}))
    monkeypatch.setattr(door, "domain_geometry", lambda path: box_geometry())
    monkeypatch.setenv(door.ENV_CACHE, str(tmp_path / "cache"))
    experiment = tmp_path / "experiment.toml"
    experiment.write_text("")
    args = _launch_args(experiment, hydro="on")

    @runtime.go_scope
    def launch(args):
        runtime.rebind_launch(args, admitted=True)
        settings = runtime.current_settings()
        members = {member: member_settings(SimpleNamespace(experiment_config=None))
                   for member in (0, 1)}
        members[0]["settings_bindings"]["fabric_manifest"] = "mutated"
        receipts = {m: activation_receipt(s) for m, s in members.items()}
        return settings, members, receipts, os.environ[runtime.ENV_SETTINGS]
    settings, members, receipts, env_path = launch(args)
    assert settings["enabled"] is True and settings["realization"] == "lower-colorado-hyfeatures"
    assert Path(env_path).name == "hydro-door.toml"
    # One independent settings object per member: no shared mutable state.
    assert members[1]["settings_bindings"]["fabric_manifest"] == str(manifest)
    for receipt in receipts.values():
        assert receipt["enabled"] is True
        assert receipt["door"]["provider"] == "test-prepared"
        assert receipt["door"]["footprint"]["parent_lsm"] == 4
    assert door.ENV_DECISION not in os.environ and runtime.ENV_SETTINGS not in os.environ


def test_go_scope_off_is_off_for_every_member(tmp_path, monkeypatch):
    table, _ = prepared_table(tmp_path)
    monkeypatch.setenv(door.ENV_TABLE, str(table))
    monkeypatch.setattr(door, "domain_geometry", lambda path: pytest.fail("door ran"))
    experiment = tmp_path / "experiment.toml"
    experiment.write_text("")

    @runtime.go_scope
    def launch(args):
        return [activation_receipt(member_settings(SimpleNamespace(experiment_config=None)))
                for _ in range(3)]
    receipts = launch(_launch_args(experiment, hydro="off"))
    assert all(r["enabled"] is False and r["reason"] == "cli_off" for r in receipts)
    assert all(r["door"]["reason"] == "cli_off" for r in receipts)


def test_runner_without_launcher_decides_once_and_receipts_the_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(door, "probe", _tools_missing)
    inputs = _launch_args(conus_copy(tmp_path, 4))

    @runtime.forecast_scope
    def forecast(inputs):
        return activation_receipt(runtime.current_settings())
    receipt = forecast(inputs)
    # Hydro is opt-in (2.9.4.1): a runner with no launcher and no request decides once, off, without probing.
    assert receipt["enabled"] is False
    assert receipt["door"]["reason"] == "default_off"
    assert door.ENV_DECISION not in os.environ


def test_builder_row_builds_once_then_reuses(tmp_path, monkeypatch):
    shipped = json.loads(door.TABLE.read_text())
    row = dict(shipped["providers"][1], data_licence={
        "status": "verified", "name": "CC BY 4.0", "reason": "", "attribution": "x"},
        realization="lower-colorado-hyfeatures")
    row.pop("preparation", None)  # Exercise the generic builder, without physical provider stages.
    # Without the acquisition hook and default root: the source root is the env variable alone.
    row["builder"] = dict(row["builder"], paths=[
        {k: v for k, v in need.items() if k != "acquire"} | {"default": None}
        for need in row["builder"]["paths"]])
    fake = tmp_path / "rw_hydro_world"
    fake.write_text("#!/bin/sh\necho 'usage: rw_hydro_world build | admit'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("WOOF_RW_HYDRO_WORLD", str(fake))
    monkeypatch.setenv("GPUWM_HYDRO_WORLD_TILES", str(tmp_path))
    monkeypatch.setenv(door.ENV_CACHE, str(tmp_path / "cache"))
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        Path(argv[argv.index("--out") + 1]).write_text("gpkg")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def prepare(config):
        out = Path(json.loads(Path(config).read_text())["output"])
        out = out / "content-key"
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text("{}")
        return {"output": str(out)}  # Native rw_hydrofabric prepare's response.
    decision = {"_row": row, "footprint": door._summary(box_geometry()),
                "manifest_binding": "fabric_manifest"}
    _, first, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                       runner=runner, prepare=prepare)
    assert build["reused"] is False and calls[0][1] == "build"
    assert Path(build["manifest"]).parent.name == "content-key"
    assert "--bbox" in calls[0] and calls[0][calls[0].index("--bbox") + 1] == "-98.0,30.0,-97.0,31.0"
    _, second, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                        runner=runner, prepare=prepare)
    assert build["reused"] is True and len(calls) == 1 and first == second
    first_bytes = first.read_bytes()
    # A different source root must not reuse a fabric from the previous root.
    other_sources = tmp_path / "other-sources"
    other_sources.mkdir()
    monkeypatch.setenv("GPUWM_HYDRO_WORLD_TILES", str(other_sources))
    _, changed, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                   runner=runner, prepare=prepare)
    assert build["reused"] is False and len(calls) == 2
    assert changed != first and first.read_bytes() == first_bytes
    _, repeated, build = door.materialize(decision, config_path=tmp_path / "e.toml",
                                          runner=runner, prepare=prepare)
    assert repeated == changed and build["reused"] is True and len(calls) == 2
    monkeypatch.delenv("GPUWM_HYDRO_WORLD_TILES")
    with pytest.raises(RuntimeError, match="source tiles.*not available"):
        door.materialize(decision, config_path=tmp_path / "e.toml", runner=runner,
                         prepare=prepare)
    assert len(calls) == 2


# -- cost budget ----------------------------------------------------------------

def test_retry_after_initial_water_failure_preserves_partial_and_reuses_completion(tmp_path):
    root = tmp_path / "fabric-key"
    root.mkdir()
    (root / "parameterized.gpkg").write_bytes(b"previous successful stage")
    request = {"initial_water": "90 days ERA5 then live NWM soil"}
    fresh, manifest = door._build_location(root, request)
    assert fresh == root.with_name("fabric-key-retry-0001") and manifest is None
    fresh.mkdir()
    completed = fresh / "routing" / "manifest.json"
    completed.parent.mkdir()
    completed.write_text("{}")
    (fresh / "door-request.json").write_text(json.dumps(request))
    (fresh / "door-prepared.json").write_text(json.dumps({"manifest": str(completed)}))
    assert door._build_location(root, request) == (fresh, completed)
    assert (root / "parameterized.gpkg").read_bytes() == b"previous successful stage"
    completed.unlink()  # A stale completion cannot masquerade as a warm cache.
    assert door._build_location(root, request) == (root.with_name("fabric-key-retry-0002"), None)

class FakeEvent:
    def __init__(self, clock):
        self.clock, self.t, self.done = clock, None, True

    def record(self, stream):
        self.t = self.clock.pop(0)

    def synchronize(self):
        pass


def metered(per_call_ms, calls):
    times = []
    for index in range(calls):
        base = index * 100.0
        times += [base, base + per_call_ms / 2, base + 10, base + 10 + per_call_ms / 2]
    meter = cost.HydroCostMeter(ring=4, events=lambda: FakeEvent(times),
                                elapsed=lambda a, b: b.t - a.t)
    for _ in range(calls):
        meter.before_lsm_start(stream=0)
        meter.before_lsm_end(stream=0)
        meter.advance_start(stream=0)
        meter.advance_end(stream=0)
    stages = [SimpleNamespace(name="routing", parallelism="6723 MC reaches level-ordered")]
    return SimpleNamespace(hydro=SimpleNamespace(interval_seconds=300.0, prepared_stages=stages,
                                                steps=calls, elapsed_seconds=calls * 300.),
                           hydro_cost=meter)


def test_cost_receipt_names_parallelism_and_passes_budget():
    physics = metered(1.25, 12)
    block = cost.receipt(physics, forecast_seconds=240 * 0.2 + 0.015, atmosphere_steps=240)
    assert block["status"] == "PASS" and block["sampled_calls"] == 12
    assert block["hydro_ms_per_lsm_call_mean"] == pytest.approx(1.25)
    assert block["fraction_of_atmosphere_step"] == pytest.approx(15.0 / 48000.0)
    assert block["parallelism"][0]["parallelism"].startswith("6723 MC")
    cost.refuse_if_over_budget(block)


def test_cost_over_budget_is_a_named_defect_without_failing_forecast(capsys):
    physics = metered(50.0, 12)
    block = cost.receipt(physics, forecast_seconds=12 * 0.01 + 0.6, atmosphere_steps=12)
    assert block["status"] == "DEFECT" and block["defect"]["name"] == "hydro-cost-over-budget"
    cost.refuse_if_over_budget(block)
    assert "DEFECT recorded; forecast continues" in capsys.readouterr().err


def test_cost_receipt_without_hydro_says_not_run():
    assert cost.receipt(SimpleNamespace(), forecast_seconds=1, atmosphere_steps=1)["status"] == "not-run"


# -- products ---------------------------------------------------------------------

def test_matched_gauges_use_exact_roster_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    (tmp_path / "gauges.json").write_text(json.dumps({"rows": [
        {"feature_id": "wb-2420816", "station_id": "08117995"},
        {"feature_id": "wb-999", "station_id": "x"}]}))
    gauges, why = forecast_products.matched_gauges(manifest, ["2420816", "5"])
    assert gauges == ["2420816"] and why is None
    assert forecast_products.matched_gauges(None, [])[1].startswith("no prepared fabric")


def test_products_without_history_stage_name_the_gap(tmp_path):
    physics = SimpleNamespace(hydro=SimpleNamespace(components={"routing": object()}))
    block = forecast_products.publish(physics, {}, outdir=tmp_path, dx_m=3000.0)
    assert block["status"] == "gap" and "prepare_history" in block["gap"]
    assert forecast_products.render_domain(3000.0) == "d01-3km"
    assert forecast_products.render_domain(750.0) == "d01-750m"


def test_successful_renderer_does_not_hide_missing_gauge_binding(tmp_path, monkeypatch, capsys):
    from woof.hydro import output

    history_dir = tmp_path / "written" / "history"
    history_dir.mkdir(parents=True)
    request = history_dir / "renderer-request.json"
    request.write_text(json.dumps({"node_ids": ["5"]}))
    (tmp_path / "renderer-receipt.json").write_text(json.dumps({"outputs": []}))
    writer = SimpleNamespace(finish=lambda *args, **kwargs:
                             {"output_directory": str(history_dir.parent)})
    monkeypatch.setattr(output, "prepare", lambda **kwargs: writer)
    physics = SimpleNamespace(hydro=SimpleNamespace(
        components={"history": object.__new__(output.History)}))
    block = forecast_products.publish(
        physics, {"settings_bindings": {"fabric_manifest": str(tmp_path / "manifest.json")}},
        outdir=tmp_path / "forecast", dx_m=5000.0,
        renderer_runner=lambda path: tmp_path / "renderer-receipt.json")
    # Gauges are optional: every owed product was rendered, so the run published;
    # the missing gauge binding stays named in the receipt and on stderr.
    assert block["status"] == "rendered"
    assert block["products"][0]["status"] == "rendered"
    assert "no gauges.json" in block["products"][0]["gauge_gap"]
    assert "no gauge matched" in block["products"][0]["note"]
    assert "no gauges.json" in capsys.readouterr().err


def test_shipped_global_provider_uses_dem_resolution_auto():
    """The default door delegates resolution choice to the accepted DEM row."""
    from pathlib import Path
    table = json.loads((Path(__file__).resolve().parents[1] /
                        "woof/data/hydro/fabric-providers.v1.json").read_text())
    provider = next(row for row in table["providers"] if row["id"] == "hydro-world-global")
    assert provider["builder"]["source_table"]["row"] == "auto"
    assert provider["realization"] == "parent-noahmp-routed"
    assert provider["realization_by_parent_lsm"] == {
        "2": "parent-noah-routed", "3": "parent-ruc-routed", "4": "parent-noahmp-routed"}
    assert provider["parent_lsm"] == [2, 3, 4]
    assert provider["preparation"]["initial_source"] == "era5-arco"


def test_default_run_keeps_the_forecast_when_fabric_preparation_fails(tmp_path, monkeypatch):
    """Hydro is opt-in (2.9.4.1): a run without --hydro never prepares a fabric, so a failing
    preparation cannot touch it; --hydro on refuses with the failure named."""
    table, _ = prepared_table(tmp_path)
    monkeypatch.setattr(door, "domain_geometry", lambda path: box_geometry())
    monkeypatch.setattr(door, "materialize", lambda decision, **kw: (_ for _ in ()).throw(
        RuntimeError("hydro provider parameters: boom")))
    settings, path, decision = door.resolve_launch({}, requested=None, config_path=tmp_path / "e.toml",
                                                   table=table)
    assert path is None and settings["enabled"] is False
    assert decision["reason"] == "default_off"
    with pytest.raises(door.HydroDoorRefusal, match="boom"):
        door.resolve_launch({}, requested="", config_path=tmp_path / "e.toml", table=table)


@needs_static_bridge
def test_adaptive_parent_clock_picks_an_interval_free_realization(tmp_path):
    """Both accumulated Noah and instantaneous Noah-MP follow the adaptive clock."""
    for lsm, enabled in ((2, True), (4, True)):
        path = conus_copy(tmp_path, lsm)
        path.write_text(path.read_text().replace(
            f"sf_surface_physics = {lsm}", f"sf_surface_physics = {lsm}" + chr(10)
            + "use_adaptive_time_step = true", 1))
        decision = door.decide(settings={}, requested="", config_path=path, probe_row=_tools_present)
        assert decision["enabled"] is enabled, decision.get("gaps")


def test_driver_follows_a_variable_lsm_interval_on_stage_clocks():
    from woof.hydro.driver import ResidentHydroDriver, Stage

    class Fake:
        def __init__(self):
            self.arrays, self.restart_identity, self.calls = {}, {}, []

        def launch(self, dt, *, stream=None):
            self.calls.append(dt)

    every, routed = Fake(), Fake()
    driver = ResidentHydroDriver(device_id=0, interval_seconds=15.0, identity={}, after=(
        Stage("routing", routed, routed.launch, cadence_steps=20, interval_seconds=300.0),))
    driver.allow_variable_interval()
    for dt in [15.0] * 10 + [18.0] * 10 + [24.0] * 30:  # 150 + 180 + 720 = 1050 s
        driver.before_lsm()
        driver.advance(dt)
    assert routed.calls == [300.0] * 3 and driver.elapsed_seconds == 1050.0
    fixed = ResidentHydroDriver(device_id=0, interval_seconds=15.0, identity={}, after=(
        Stage("exchange", every, every.launch),))
    with pytest.raises(ValueError, match="need a fixed one: exchange"):
        fixed.allow_variable_interval()


@pytest.mark.parametrize("command,members,recipe", [("go", 1, None), ("ensemble", 4, "wofs")])
def test_a_launch_builds_its_fabric_only_after_admission(tmp_path, monkeypatch, command, members, recipe):
    """go_scope binds admission_pending for an ensemble launch; the fabric is built by
    rebind_launch(args, admitted=True) once the members' disk and card admission passed."""
    from woof.hydro import runtime
    config = tmp_path / "e.toml"
    config.write_text("[experiment]\nname='e'\n")
    built = []
    monkeypatch.setattr(runtime, "_launch_settings", lambda requested, cfg, **kw: (
        built.append(kw) or ({"enabled": True}, None,
                             {"schema": door.SCHEMA, "enabled": True, "reason": "fabric_covered", "gaps": []})))
    seen = {}

    go = command == "go"  # woof go's door is go_scope(beside=True) and binds beside after admission

    @runtime.go_scope(beside=go)
    def launch(args):
        seen["entry"] = door.inherited_decision()["reason"]
        seen["built_before"] = list(built)
        runtime.rebind_launch(args, admitted=True, beside=go)
        seen["after"] = door.inherited_decision()["reason"]
        return 0

    # Hydro is opt-in (2.9.4.1): the launch asks for it, so its admission is pending.
    args = SimpleNamespace(config=config, hydro="on", command=command, members=members,
                           recipe=recipe, dry_run=False, readiness=False)
    assert launch(args) == 0
    assert seen == {"entry": runtime.ADMISSION_PENDING, "built_before": [], "after": "fabric_covered"}
    assert door.launch_line(runtime._admission_decision(None)) == (
        "go: hydro -- waiting for disk and card admission; preparation has not started")
    assert built and not built[0].get("plan_only")


@pytest.mark.parametrize("map_present", [False, True])
def test_required_network_map_is_independent_of_optional_gauges(tmp_path, monkeypatch, map_present):
    from woof.hydro import output
    history_dir = tmp_path / "written" / "history"
    history_dir.mkdir(parents=True)
    (history_dir / "renderer-request.json").write_text(json.dumps({
        "node_ids": ["5"], "frames": [{"valid_time": "2026-10-21T13:00:00Z"}]}))
    map_path = tmp_path / "d01-12km" / "streamflow_map" / "2026-10-21" / "map.png"
    map_path.parent.mkdir(parents=True)
    if map_present:
        map_path.write_bytes(b"rendered map fixture")
    renderer_receipt = tmp_path / "renderer.json"
    renderer_receipt.write_text(json.dumps({"outputs": [str(map_path)]}))
    monkeypatch.setattr(output, "prepare", lambda **kwargs: SimpleNamespace(
        finish=lambda *args, **kw: {"output_directory": str(history_dir.parent)}))
    physics = SimpleNamespace(hydro=SimpleNamespace(
        components={"history": object.__new__(output.History)}))
    block = forecast_products.publish(physics, {}, outdir=tmp_path / "forecast", dx_m=12000.,
        renderer_runner=lambda path: renderer_receipt)
    # Gauges are optional; a missing owed map is what makes the run partial.
    assert block["status"] == ("rendered" if map_present else "partial")
    assert block["products"][0]["gauge_gap"]  # Gauge absence stays visible.
    assert block["required_streamflow_complete"] is map_present


@pytest.mark.parametrize("prepared", [False, True])
def test_native_preparation_failure_falls_through_worldwide(tmp_path, monkeypatch, prepared):
    """An empty native footprint or failed VPU cannot switch requested hydro off."""
    monkeypatch.setattr(door, "domain_geometry", lambda _: box_geometry())
    monkeypatch.setattr(door, "probe", _tools_present)
    monkeypatch.setenv(door.ENV_CACHE, str(tmp_path / "cache"))
    calls = []
    def materialize(decision, **kwargs):
        calls.append(decision["provider"])
        if decision["provider"] == "conus-nhdplus-v21":
            raise RuntimeError("no native catchment with an outlet in the domain")
        return {"enabled": True}, tmp_path / "world.toml", {"reused": True}
    monkeypatch.setattr(door, "materialize", materialize)
    if prepared:
        d = door.decide(settings={}, requested="", config_path=tmp_path / "x.toml")
        memo = door._failure_memo(d, tmp_path / "x.toml")
        memo.parent.mkdir(parents=True)
        memo.write_text(json.dumps({"gap": "no native catchment", "at": "test"}))
    settings, path, decision = door.resolve_launch({}, requested="",
        config_path=tmp_path / "x.toml", prepared_launch=prepared)
    assert settings["enabled"] and decision["provider"] == "hydro-world-global"
    # An explicit --hydro never takes a prepared root's failure memo (2.9.4.1):
    # it tries the native row again and falls through to the worldwide one.
    assert calls == ["conus-nhdplus-v21", "hydro-world-global"]
    assert "no native catchment" in decision["preparation_fallbacks"][0]["gap"]


def test_non_conus_domain_still_selects_worldwide():
    decision = door.decide(settings={}, requested="",
        geometry=box_geometry(west=4, east=12, south=46, north=52), probe_row=_tools_present)
    assert decision["provider"] == "hydro-world-global"


def test_native_sources_clip_padding_without_changing_parent_contract():
    row = {"id": "conus-nhdplus-v21", "coverage_policy": "source-land",
           "coverage": [[-125, 24, -66.5, 49.5]]}
    footprint = {"west": -136, "south": 17, "east": -55, "north": 53,
                 "parent_lsm": 2, "start_time": "2026-10-21T12:00:00Z",
                 "parent_grid_contract": {"grid": {"nx": 620, "ny": 360}}}
    clipped = door.source_footprint(row, footprint)
    assert [clipped[k] for k in ("west", "south", "east", "north")] == [-125, 24, -66.5, 49.5]
    assert clipped["parent_grid_contract"] is footprint["parent_grid_contract"]
    assert clipped["parent_lsm"] == 2 and clipped["start_time"] == footprint["start_time"]
    assert footprint["west"] == -136
    assert door.source_footprint({"coverage_policy": "whole-domain"}, footprint) is footprint


def test_output_library_uses_packaged_bridge_ladder(tmp_path, monkeypatch):
    from woof import bridges, bridge_assets
    from woof.hydro import output
    library = tmp_path / "librw_nextgen_output.so"
    library.write_bytes(b"library fixture")
    calls = []
    monkeypatch.setattr(bridges, "artifact_candidates", lambda env, name, **kw:
        calls.append((env, name, kw)) or [library])
    assert output._library_path() == library
    assert calls[0][0] == "WOOF_NEXTGEN_OUTPUT_LIBRARY"
    artifact = next(a for a in bridge_assets.BUNDLED_ARTIFACTS if a.name == "rw_nextgen_output")
    assert artifact.crate == "tools/nextgen_output" and artifact.on_demand
    assert bridge_assets.LIBRARY_ABI[artifact.name] == ("rw_nextgen_output_abi_version", 1)


def test_output_library_first_use_fetches_the_roster_artifact(tmp_path, monkeypatch):
    from woof import bridges
    from woof.hydro import output
    monkeypatch.delenv("WOOF_NEXTGEN_OUTPUT_LIBRARY", raising=False)
    library = tmp_path / "librw_nextgen_output.so"
    monkeypatch.setattr(bridges, "artifact_candidates", lambda *args, **kw: [library])
    fetched = []
    def stage(artifact):
        # The public fetch API takes a roster name, not a platform filename.
        assert bridges.fetched_on_demand(artifact)
        fetched.append(artifact)
        library.write_bytes(b"downloaded library fixture")
        return True
    monkeypatch.setattr(bridges, "fetch_on_demand", stage)
    assert output._library_path() == library
    assert fetched == ["rw_nextgen_output"]


def _beside_door(tmp_path, monkeypatch, release):
    """A covered domain whose fabric build waits on ``release`` (a slow first-run build)."""
    import threading
    table, manifest = prepared_table(tmp_path)
    monkeypatch.setenv(door.ENV_TABLE, str(table))
    monkeypatch.setattr(door, "TABLE", Path(tmp_path / "only-extra.json"))
    shipped = json.loads(table.read_text())
    (tmp_path / "only-extra.json").write_text(json.dumps({**shipped, "providers": []}))
    monkeypatch.setattr(door, "domain_geometry", lambda path: box_geometry())
    monkeypatch.setenv(door.ENV_CACHE, str(tmp_path / "cache"))
    real, started = door.materialize, threading.Event()

    def slow(decision, **kw):
        started.set()
        assert release.wait(30), "the launch never released the build"
        return real(decision, **kw)
    monkeypatch.setattr(door, "materialize", slow)
    experiment = tmp_path / "experiment.toml"
    experiment.write_text("")
    return _launch_args(experiment, hydro="on"), manifest, started


def test_go_builds_the_covering_fabric_beside_the_fetch_not_ahead_of_it(tmp_path, monkeypatch):
    """R15 D9 (first run on a bare install): the fabric build (273 s) ran ahead of the fetch, so the
    chained root stage came ready 366 s after launch against its 300 s bound. The launch now
    runs while the fabric builds; the go memory gate does not wait for it; the first reader
    of the answer (the forecast) waits and gets exactly the synchronous door's settings."""
    import threading
    from woof.hydro.memory import forecast_plan
    release = threading.Event()
    args, manifest, started = _beside_door(tmp_path, monkeypatch, release)
    seen = {}

    @runtime.go_scope(beside=True)
    def launch(args):
        # go_main admits disk first, then binds the door beside its fetch.
        assert door.inherited_decision()["reason"] == runtime.ADMISSION_PENDING
        runtime.rebind_launch(args, admitted=True, beside=True)
        assert started.wait(30)
        # The fetch and the preparation run here while the fabric builds.
        seen["pending"] = runtime.beside_pending()
        seen["decision"] = door.inherited_decision()
        seen["line"] = door.launch_line(door.inherited_decision())
        seen["env_settings"] = os.environ.get(runtime.ENV_SETTINGS)
        seen["gate"] = forecast_plan(object())  # the gate prices nothing yet, and does not wait
        seen["still_pending"] = runtime.beside_pending()
        release.set()
        # The forecast: the first reader joins and binds the final answer.
        settings = runtime.current_settings()
        return (settings, os.environ[runtime.ENV_SETTINGS], door.inherited_decision(),
                activation_receipt(member_settings(SimpleNamespace(experiment_config=None))))
    settings, env_path, decision, receipt = launch(args)
    assert seen["pending"] and seen["still_pending"] and seen["gate"] is None
    assert seen["decision"]["reason"] == "fabric_covered" and seen["decision"]["build"]["pending"]
    assert "preparing beside the fetch" in seen["line"]
    assert seen["env_settings"] is None  # no runner can start on a half-made answer
    assert settings["enabled"] is True and settings["realization"] == "lower-colorado-hyfeatures"
    assert Path(env_path).name == "hydro-door.toml"
    assert decision["build"]["beside"] is True and decision["build"]["seconds"] >= 0.0
    assert decision["build"]["waited_seconds"] >= 0.0
    assert receipt["enabled"] is True and receipt["door"]["build"]["beside"] is True
    assert not runtime.beside_pending()
    assert door.ENV_DECISION not in os.environ and runtime.ENV_SETTINGS not in os.environ


def test_prepare_only_launch_still_completes_the_fabric_and_a_failed_launch_does_not_wait(
        tmp_path, monkeypatch):
    """A launch whose forecast never reads the answer (--prepare-only) leaves the domain's
    fabric complete before it returns; a launch that raises returns at once."""
    import threading
    release = threading.Event()
    args, _, started = _beside_door(tmp_path, monkeypatch, release)

    @runtime.go_scope(beside=True)
    def prepare_only(args):
        runtime.rebind_launch(args, admitted=True, beside=True)
        assert started.wait(30)
        release.set()
        return 0
    assert prepare_only(args) == 0
    assert not runtime.beside_pending()
    assert list((tmp_path / "cache").rglob("hydro-door.toml"))

    stuck = threading.Event()
    (tmp_path / "second").mkdir()
    args, _, started = _beside_door(tmp_path / "second", monkeypatch, stuck)

    @runtime.go_scope(beside=True)
    def fails(args):
        runtime.rebind_launch(args, admitted=True, beside=True)
        assert started.wait(30)
        raise RuntimeError("fetch failed")
    with pytest.raises(RuntimeError, match="fetch failed"):
        fails(args)
    assert not runtime.beside_pending()
    stuck.set()


def test_plain_go_scope_and_explicit_hydro_still_decide_before_the_launch(tmp_path, monkeypatch):
    """Only the default-on `woof go` door builds beside: bare --hydro keeps refusing before
    the fetch, and every other launcher (standby, run) binds the final answer on entry."""
    import threading
    release = threading.Event()
    release.set()
    args, _, _ = _beside_door(tmp_path, monkeypatch, release)

    @runtime.go_scope
    def plain(args):
        return runtime.beside_pending(), runtime.current_settings()["enabled"]
    assert plain(args) == (False, True)

    @runtime.go_scope(beside=True)
    def explicit(args):
        return runtime.beside_pending()
    assert explicit(_launch_args(args.config, hydro="off")) is False


@pytest.mark.parametrize("map_complete", [False, True])
def test_forecast_requires_network_map_but_retains_optional_gauge_gap(tmp_path, monkeypatch, map_complete):
    monkeypatch.setattr(cost, "receipt", lambda *a, **kw: {"status": "PASS"})
    monkeypatch.setattr(runtime, "current_settings", lambda: {})
    monkeypatch.setattr(forecast_products, "publish", lambda *a, **kw: {
        "status": "partial", "gauge_gap": "no prepared gauges",
        "required_streamflow_complete": map_complete})
    physics = SimpleNamespace(hydro=SimpleNamespace(check_health=lambda: {}))
    if map_complete:
        result = forecast_products.finalize_forecast(physics, outdir=tmp_path, dx_m=12000.,
            forecast_seconds=60., atmosphere_steps=60)
        assert result["hydro_products"]["gauge_gap"] == "no prepared gauges"
    else:
        with pytest.raises(RuntimeError, match="products incomplete"):
            forecast_products.finalize_forecast(physics, outdir=tmp_path, dx_m=12000.,
                forecast_seconds=60., atmosphere_steps=60)
    assert (tmp_path / "hydro-receipt.json").is_file()


def test_disk_estimate_only_discounts_the_matching_complete_cache(tmp_path, monkeypatch):
    from woof.hydro import runtime
    config = tmp_path / "run.toml"
    config.write_text("[experiment]\nname='estimate'\n")
    footprint = dict(west=0, east=10, south=0, north=6, time_origin="2026-10-21")
    row = dict(kind="builder")
    request = dict(provider="provider", footprint=footprint)
    monkeypatch.setattr(door, "cache_root", lambda *a: tmp_path / "cache")
    monkeypatch.setattr(door, "decide", lambda **kw: dict(
        reason="fabric_covered", provider="provider", footprint=footprint, _row=row))
    monkeypatch.setattr(door, "preparation_request", lambda *a: (request, []))
    provider = tmp_path / "cache/provider"
    unrelated = provider / "static-unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "static-ready.json").touch()
    full, _ = runtime.hydro_disk_estimate(config)
    assert full == int(60 * (runtime.HYDRO_FULL_BYTES_PER_DEG2 + runtime.HYDRO_TILE_BYTES_PER_DEG2))
    static_request = dict(request, footprint={k: v for k, v in footprint.items() if k not in door._TIME_KEYS})
    static = provider / ("static-" + door._digest(static_request)[:16])
    static.mkdir()
    (static / "static-ready.json").touch()
    warm, _ = runtime.hydro_disk_estimate(config)
    assert warm == int(60 * runtime.HYDRO_START_BYTES_PER_DEG2)
    start = provider / door._digest(request)[:16]
    start.mkdir()
    (start / "hydro-door.toml").touch()
    (start / "door-request.json").write_text(json.dumps(request))
    manifest = start / "manifest.json"
    (start / "door-prepared.json").write_text(json.dumps(dict(manifest=str(manifest))))
    assert runtime.hydro_disk_estimate(config)[0] == warm
    manifest.touch()
    assert runtime.hydro_disk_estimate(config) == (0, None)

"""CPU gates for the [devices] front door; rank APIs are supplied by the core lane."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from conftest import requires_cupy
from woof.config import RunConfig, load_device_options
from woof.core.devices import (DEVICES_OFF, DeviceOptions, DevicesRefused,
    override_device_count, refuse_unrouted_devices, validate_device_count)
from woof.experiment import build_experiment, experiment_from_run_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def exp():
    return experiment_from_run_config(RunConfig(nx=120, ny=90, nz=12, dx=3000, dy=3000, ztop=12000, dt=15, run_seconds=120), datetime(2026, 1, 1))


@pytest.fixture
def rank_api(monkeypatch):
    from woof.core import streaming
    from tilestream.harness import halo_radius
    from tilestream.multigpu import plan_split
    from woof.core.adaptive_clock import acoustic_step_ceiling
    def halo(cfg, *, max_map_factor=1.0):
        return halo_radius(replace(cfg, time_step_sound=acoustic_step_ceiling(cfg, max_map_factor))) + (
            max(cfg.spec_zone, cfg.relax_zone) if cfg.specified else 0)
    def specs(cfg, options, *, halo):
        gy, gx = options.resolved_grid(cfg.nx, cfg.ny)
        px, py = streaming._periodic_axes(cfg)
        return plan_split(cfg.nx, cfg.ny, halo, gx=gx, gy=gy,
                          periodic_x=px, periodic_y=py)
    monkeypatch.setattr(streaming, "ranked_halo", halo, raising=False)
    monkeypatch.setattr(streaming, "ranked_specs", specs, raising=False)
    return halo, specs


@pytest.fixture
def declared_two_card_probe(monkeypatch):
    """Gate arithmetic uses declared cards, independent of local CUDA hardware."""
    from woof.core import devices
    from woof.core.devices_memory import GIB
    monkeypatch.setattr(devices, "probe_devices", lambda: {
        "visible_count": 2,
        "cards": {dev: {"free_bytes": 96 * GIB, "total_bytes": 96 * GIB,
                        "profile": None} for dev in (0, 1)},
    })


@pytest.mark.parametrize("table", [
    {"bogus": 2}, {"count": True}, {"count": 0},
    {"count": 2.5}, {"count": 1, "ids": [0]},
    {"count": 1, "grid": "1x1"}, {"count": 1, "transport": "host"},
    {"count": 2, "grid": "2x2"}, {"count": 2, "grid": "1x0"},
    {"count": 2, "grid": "xyz"}, {"count": 2, "grid": "1x2x3"},
    {"count": 2, "grid": 2}, {"count": 2, "ids": [0]},
    {"count": 2, "ids": [-1, 0]}, {"count": 2, "ids": [0.5, 1]},
    {"count": 2, "transport": "bogus"}, {"count": 2, "ids": [False, 0]},
])
def test_table_refusals(table):
    with pytest.raises(ValueError):
        DeviceOptions.from_mapping(table)


def test_table_roundtrip_and_grid(tmp_path):
    text = '[devices]\ncount = 2\nids = [0, 0]\ntransport = "host"\n'
    path = tmp_path / "options.toml"
    path.write_text(text)
    options = load_device_options(path)
    assert options.resolved_grid(1797, 1057) == (1, 2)
    assert options.device_ids() == (0, 0)
    assert DeviceOptions.from_mapping(options.to_mapping()) == options
    assert DeviceOptions.from_mapping(None) is DEVICES_OFF
    for bad in ([], 3, "devices"):
        with pytest.raises(ValueError):
            DeviceOptions.from_mapping(bad)


def test_rank_count_uses_geometry_and_visible_card_guards():
    options = DeviceOptions.from_mapping({"count": 9, "grid": "3x3", "ids": [0] * 9})
    assert options.resolved_grid(144, 120) == (3, 3)
    assert DeviceOptions.from_mapping(options.to_mapping()) == options
    validate_device_count(options, 1)
    with pytest.raises(DevicesRefused, match="visible card count = 8"):
        validate_device_count(DeviceOptions(count=9), 8)


def test_experiment_table_and_restart_identity(exp):
    from woof.core.model import restart_identity_payload, experiment_fingerprint
    split = replace(exp, devices=DeviceOptions(count=2))
    assert restart_identity_payload(exp) == restart_identity_payload(split)
    catalog = SimpleNamespace(run_provenance={})
    assert experiment_fingerprint(exp, catalog) == experiment_fingerprint(split, catalog)
    from woof.core.streaming import StreamingOptions
    with pytest.raises(DevicesRefused, match="two stores and two planners"):
        replace(split, tiles=StreamingOptions(mode="auto"))


@pytest.mark.parametrize("name", ["aifs_single_demo.toml", "aigfs_hybrid_demo.toml"])
def test_authority_preserves_devices_and_default_bytes(name):
    from woof.prepared_single_domain_forecast import _render_materialized_experiment, _experiment_tables
    from woof.core.model import restart_identity_payload
    text = (ROOT / "configs" / name).read_text()
    default, exp, receipt = _render_materialized_experiment(text, source="gfs", profile=None)
    assert default == text
    declared = text + '\n[devices]\ncount = 2\nids = [0, 0] # repeated card proof\n'
    rendered, split, _ = _render_materialized_experiment(declared, source="gfs", profile=None)
    assert rendered == declared
    readback = build_experiment(_experiment_tables(tomllib.loads(rendered)), source="readback")
    assert readback.devices.device_ids() == (0, 0)
    assert restart_identity_payload(exp) == restart_identity_payload(split)


def test_per_domain_devices_refused():
    text = (ROOT / "configs" / "aifs_single_demo.toml").read_text()
    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    raw["domain"][0]["devices"] = {"count": 2}
    with pytest.raises(ValueError, match="a tree has one split"):
        build_experiment(raw, source="test")


@pytest.mark.parametrize("route", ["woof run", "woof go nested tree", "offline child", "downscale", "run-plan experiment"])
def test_unrouted_refusal(exp, route):
    refuse_unrouted_devices(exp, route)
    with pytest.raises(DevicesRefused, match="silent one-card run"):
        refuse_unrouted_devices(replace(exp, devices=DeviceOptions(count=2)), route)


def test_card_id_and_flag_refusals(capsys):
    options = DeviceOptions(count=2)
    validate_device_count(options, 2)
    with pytest.raises(DevicesRefused, match=r"ids = \[0, 1\].*visible card count = 1"):
        validate_device_count(options, 1)
    assert override_device_count(options, 1) == DEVICES_OFF
    assert "replaces [devices] count = 2" in capsys.readouterr().out
    for options in (DeviceOptions(count=2, grid="1x2"), DeviceOptions(count=2, ids=(0, 0))):
        with pytest.raises(DevicesRefused, match="--devices 1 contradicts"):
            override_device_count(options, 1)


def test_forecast_store_and_stepper_wiring(exp, monkeypatch):
    from woof import prepared_single_domain_forecast as runner
    from woof.core import streaming
    split = replace(exp, devices=DeviceOptions(count=2, ids=(0, 0)))
    node = SimpleNamespace(cfg=exp.root, clock=object(), state=SimpleNamespace())
    bundle = SimpleNamespace(geography={"setup/msfu": np.array([1.7])})
    decision = SimpleNamespace(road="ranks", halo=22)
    stepper = SimpleNamespace(tiled_run=SimpleNamespace(transport_report={"0:0": "device"}))
    def decide(cfg, options, *, max_map_factor=1.0):
        assert cfg is exp.root.run and options is split.devices
        assert max_map_factor == 1.7
        return decision
    def builder(given, *, clock, options, seam="zeros", check_geography=True,
                step_mode="threads", snapshot_limits=None, mynn_column_chunks=None):
        assert given is bundle and clock is node.clock and options is split.devices
        assert snapshot_limits is None
        assert mynn_column_chunks is None
        return lambda state, cfg, selected: stepper
    def make(state, cfg, *, decision, build):
        # The template the store road holds, as steppers_for_tree passes
        # it: make_stepper publishes the store on this object, and None
        # stopped every ranked forecast before its first step.
        assert state is node.state
        return build(state, cfg, decision)
    monkeypatch.setattr(streaming, "ranked_decision", decide, raising=False)
    monkeypatch.setattr(streaming, "ranked_domain_builder", builder, raising=False)
    monkeypatch.setattr(streaming, "make_stepper", make)
    assert runner._devices_init_road(split)[0] == "store"
    assert runner._devices_init_road(exp) is None
    decisions = {}
    assert runner._devices_stepper(bundle, node, split, decisions) == {1: stepper}
    assert decisions[1] is decision
    assert node.state._streamed_domain is stepper


def test_rank_interior_refusal(exp, rank_api):
    from woof.core.preflight import estimate_devices
    tiny = replace(exp, domains=(replace(exp.root, run=replace(exp.root.run, nx=16, ny=16)),),
                   devices=DeviceOptions(count=2))
    with pytest.raises(ValueError):
        estimate_devices(tiny)


def test_per_card_admission(rank_api):
    from woof.core.preflight import _load_experiment_any, estimate_experiment, estimate_devices
    from woof.core.devices_memory import devices_gate, GIB
    exp = _load_experiment_any(ROOT / "tests" / "fixtures" / "devices_hrrr_grid_admission.toml")
    assert (exp.root.run.nx, exp.root.run.ny, exp.root.run.nz) == (1797, 1057, 64)
    assert len(exp.vertical.eta_levels) == exp.root.run.nz + 1
    from woof.boundary_fields import source_boundary_species
    one = estimate_experiment(exp, vram_gib=96, forcing_intervals=18,
                              boundary_species=source_boundary_species("hrrr"))
    capacity = 96 * GIB
    assert one.peak_envelope_bytes >= 1.10 * capacity, (
        f"one-card premise lost: envelope {one.peak_envelope_bytes:,} B "
        f"({one.peak_envelope_bytes/GIB:.2f} GiB), capacity {capacity:,} B "
        f"(96 GiB); require at least {1.10*capacity:,.0f} B (105.6 GiB)")
    split = replace(exp, devices=DeviceOptions(count=2))
    price = estimate_devices(split, vram_gib=96, forcing_intervals=18, source="hrrr")
    gate = devices_gate(price, budgets={0: 96 * GIB, 1: 96 * GIB})
    print(f"one card: REFUSED: {one.peak_envelope_bytes/GIB:.2f} GiB; budget 96.00 GiB")
    print(gate["verdict"])
    assert not gate["refuse"]
    assert {row["card"] for row in price["cards"]} == {0, 1}
    for row in price["cards"]:
        assert row["total_bytes"] <= 0.90 * capacity, (
            f"two-card premise lost: card {row['card']} envelope "
            f"{row['total_bytes']:,} B ({row['total_bytes']/GIB:.2f} GiB), "
            f"capacity {capacity:,} B; require at least 10% headroom")
    repeated = estimate_devices(replace(split, devices=DeviceOptions(count=2, ids=(0, 0))),
                                vram_gib=96, forcing_intervals=18, source="hrrr")
    assert devices_gate(repeated, budgets={0: 96 * GIB})["refuse"]
    assert repeated["cards"][0]["resident_bytes"] == sum(row["resident_bytes"] for row in price["cards"])


def test_rank_admission_uses_tile_mynn_workspace(exp, rank_api, monkeypatch):
    """The priced chunk must be the one the rank factory makes it walk."""
    from woof.core import preflight as pf
    from woof.core import mynn_pbl_scratch as mynn
    from woof.core.devices_memory import estimate_devices
    from tilestream.harness import tile_config

    monkeypatch.delenv(mynn.MYNN_PBL_COLUMN_CHUNK_ENV, raising=False)
    monkeypatch.setattr(mynn, "_PINNED", None)
    monkeypatch.setattr(mynn, "_RESOLVED", {})
    monkeypatch.setattr(mynn, "resolve_mynn_column_chunk", lambda nz: 98304)
    run = replace(exp.root.run, nx=800, ny=600, nz=50,
                  bl_pbl_physics=5, sf_sfclay_physics=5)
    split = replace(exp, domains=(replace(exp.root, run=run),),
                    devices=DeviceOptions(count=2))
    price = estimate_devices(split, vram_gib=96)
    for rank in price["rank_shapes"]:
        ny, nx = rank["compute_shape"]
        local = replace(split, devices=DEVICES_OFF,
                        domains=(replace(split.root, run=tile_config(run, nx, ny)),))
        tile = pf.estimate_experiment(local, vram_gib=96, tile_buffer=True,
                                      forcing_intervals=1)
        resident = pf.estimate_experiment(local, vram_gib=96)
        assert rank["resident_bytes"] == tile.peak_envelope_bytes
        assert resident.peak_envelope_bytes > tile.peak_envelope_bytes
        assert pf.mynn_pbl_column_chunk(local.root.run, tile_buffer=True) == 8192
        assert pf.mynn_pbl_column_chunk(local.root.run) == 98304


def test_rank_forcing_and_loader_prices_do_not_retain_the_host_series(exp, rank_api):
    """Charging host forcing to every rank/template refused fitting runs."""
    from woof.core import preflight as pf
    from woof.core.devices_memory import estimate_devices
    from woof.ingest.prepared_store import default_slab_rows
    from tilestream.harness import tile_config

    cfg = replace(exp.root.run, nx=300, ny=220, specified=True,
                  spec_zone=1, relax_zone=4, spec_bdy_width=5)
    split = replace(exp, domains=(replace(exp.root, run=cfg),),
                    devices=DeviceOptions(count=2))
    one = estimate_devices(split, forcing_intervals=1)
    six = estimate_devices(split, forcing_intervals=6)
    assert one["cards"] == six["cards"]
    assert six["host_boundary_bytes"] == 6 * one["host_boundary_bytes"]
    rows = default_slab_rows(cfg.nx, cfg.ny)
    local = replace(split, devices=DEVICES_OFF, domains=(replace(
        split.root, run=tile_config(cfg, cfg.nx, cfg.ny % rows or rows)),))
    priced = pf.estimate_experiment(local, forcing_intervals=6).domains[0]
    assert six["cards"][0]["template_bytes"] == (
        priced.resident_bytes - priced.category_bytes("lbc"))
    assert priced.category_bytes("lbc") > 0


def test_route_admission_calls(exp, tmp_path):
    from woof.runtime import run_experiment
    from woof.offline_child import resolve_child_run_config
    from woof.runplan import streaming_decision, PlanError
    split = replace(exp, devices=DeviceOptions(count=2))
    with pytest.raises(DevicesRefused, match="woof run"):
        run_experiment(split, None, tmp_path)
    with pytest.raises(PlanError, match="silent one-card run"):
        streaming_decision(split, chain="experiment")
    path = tmp_path / "child.toml"
    path.write_text('[devices]\ncount = 2\n')
    with pytest.raises(DevicesRefused, match="offline child and downscale"):
        resolve_child_run_config(path)


def test_forecast_flag_reaches_forecast_only():
    from woof import go_cli
    plan = {"source": "gfs", "runner": go_cli.RUNNER_MODULE,
            "config": Path("config.toml"), "wps_namelist": Path("namelist.wps"),
            "authority": Path("authority"), "prepared": Path("prepared"),
            "run": Path("run"), "physics_profile": None,
            "devices_count_override": 2}
    assert "--devices" not in go_cli.authority_command(plan)
    command = go_cli.forecast_command(plan, {"proof": "p", "source_manifest": "m", "prepared_content": "c"})
    assert command[command.index("--devices") + 1] == "2"


def test_clock_history_and_pool_samples(exp, monkeypatch):
    import sys
    from woof import prepared_single_domain_forecast as runner
    class Device:
        def __init__(self, dev): self.dev = dev
        def __enter__(self): return self
        def __exit__(self, *args): pass
    pool = SimpleNamespace(used_bytes=lambda: 12, total_bytes=lambda: 20)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(Device=Device), get_default_memory_pool=lambda: pool))
    stepper = SimpleNamespace(report={"dt": 3.0, "time_step_sound": 8})
    history, peaks, observed = [], {}, []
    callback = runner._devices_step_observer(
        SimpleNamespace(cfg=exp.root), stepper, DeviceOptions(count=2), history, peaks,
        lambda **event: observed.append(event))
    callback(step_count=1, dt=3, model_seconds=3)
    stepper.report = {"dt": 4.0, "time_step_sound": 10}
    callback(step_count=2, dt=4, model_seconds=7)
    assert [(row["dt"], row["time_step_sound"]) for row in history] == [(3, 8), (4, 10)]
    assert peaks == {"0": {"pool_used_bytes": 12, "pool_held_bytes": 20},
                     "1": {"pool_used_bytes": 12, "pool_held_bytes": 20}}
    assert len(observed) == 2


def test_memory_gate_and_check_declared_cards(rank_api, declared_two_card_probe, tmp_path, monkeypatch, capsys):
    from woof import cli, go_cli
    from woof.core import preflight as pf
    from woof.core.devices_memory import GIB
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 512 * GIB)
    path = tmp_path / "split.toml"
    path.write_text((ROOT / "tests" / "fixtures" / "devices_hrrr_grid_admission.toml").read_text()
                    + "\n[devices]\ncount = 2\n")
    gate = go_cli.memory_gate({"config": path}, vram_gib=96)
    print(gate["verdict"])
    assert not gate["refuse"]
    assert len(gate["devices"]["cards"]) == 2
    assert "preparation" in gate
    parser = cli.build_parser()
    args = parser.parse_args(["check", str(path), "--budget-gib", "96", "--vram-gib", "96"])
    assert pf.check_main(args) == 0
    output = capsys.readouterr().out
    assert "go: [devices] 2 slabs" in output
    assert "card 0: ADMITTED" in output and "card 1: ADMITTED" in output


def test_devices_table_relay_preserves_ids():
    from woof.stage_cli import devices_flags, StageRefusal
    import json
    options = DeviceOptions(count=2, ids=(0, 0), transport="host")
    flags = devices_flags("single", options=options)
    assert flags[0] == "--devices-table"
    assert DeviceOptions.from_mapping(json.loads(flags[1])) == options
    assert devices_flags("single", options=DEVICES_OFF) == []
    # The tree runner takes the same table (and reads its domains key).
    tree = DeviceOptions(count=2, ids=(0, 0), domains=(1,))
    flags = devices_flags("tree", options=tree)
    assert DeviceOptions.from_mapping(json.loads(flags[1])) == tree


def test_sim_devices_table_travels_beside_the_bound_config():
    """``woof sim --devices-table JSON``: the [tiles] precedent for a split.

    A prepared bundle binds its experiment TOML byte for byte, so a table
    added to that file is refused by the forecast's receipt check; the JSON
    rides beside it, is validated by the TOML parser's own rules, and is the
    only way to name ``ids`` (the one-card proof needs ``[0, 0]``).
    """
    import argparse
    import json
    from woof import stage_cli
    parser = argparse.ArgumentParser()
    stage_cli.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["sim", "prepared", "--experiment-config", "e.toml",
                              "--outdir", "out",
                              "--devices-table", '{"count": 2, "ids": [0, 0]}'])
    options = stage_cli._devices_table_argument(args.devices_table)
    assert options == DeviceOptions(count=2, ids=(0, 0))
    assert stage_cli.devices_flags("single", options=options) == [
        "--devices-table", json.dumps({"count": 2, "ids": [0, 0]}, sort_keys=True)]
    assert stage_cli._devices_table_argument(None) is None
    with pytest.raises(stage_cli.StageRefusal, match="--devices-table refused"):
        stage_cli._devices_table_argument('{"count": 2, "ids": [0]}')
    with pytest.raises(stage_cli.StageRefusal, match="--devices-table refused"):
        stage_cli._devices_table_argument("not json")


@pytest.mark.parametrize("name,authority,fingerprint", [
    ("gfs_12km_10gib.toml", "6347eda248eefe9c26800434c08330ed8f1e82845fd8dbeeb94d5769334e6693",
     "ae44a252a8ba33904df37aa13b2eb2f72b8d8b0e9293e9955c14edd126079634"),
    ("gfs_12km_quickstart.toml", "c255bc9f49862d543104985b531969b2c8c2f540f9b20fa64743583d75f23127",
     "c94880c5f76091cab9850bf6ae554997f27097929430ccb4b1140bba1f871515"),
])
def test_default_authority_and_fingerprint_base_golden(name, authority, fingerprint, tmp_path, monkeypatch):
    import hashlib
    from datetime import timezone
    from woof import go_cli
    from woof.prepared_single_domain_forecast import materialize_named_source_authorities
    from woof.core.model import experiment_fingerprint
    from woof.experiment import load_experiment
    monkeypatch.setattr(go_cli.run_stamp_module, "utcnow",
                        lambda: datetime(2026, 9, 30, tzinfo=timezone.utc))
    path = ROOT / "configs" / name
    plan = go_cli.plan_from_config(path, outdir=tmp_path / "run", run_stamp=False)
    assert "devices" not in plan and "devices_sentence" not in plan
    destination = tmp_path / "authority"
    materialize_named_source_authorities(
        source="gfs", base_experiment_config=path,
        base_wps_namelist=path.with_suffix(".namelist.wps"),
        physics_profile=plan["profile"], output_directory=destination)
    payload = (destination / "experiment.toml").read_bytes()
    # the emitter's comment lines name WOOF where the golden's named the engine (tests/woof_engine_spelling.py)
    from woof_engine_spelling import engine_spelling
    assert hashlib.sha256(engine_spelling(payload.decode("utf-8")).encode("utf-8")).hexdigest() == authority
    assert (destination / "namelist.wps").read_bytes() == path.with_suffix(".namelist.wps").read_bytes()
    exp = load_experiment(destination / "experiment.toml")
    # The retained golden predates 48847c9eb's local_face default. Inverse
    # only that declared value move, and prove the live identity binds it.
    assert all(domain.run.terrain_clock == "local_face" for domain in exp.domains)
    # ... and 2.9.0's WRF 4.7/4.8 upstream-fix defaults (golden/wrf-fixes):
    # the golden was recorded under WRF 4.6.1's equation of time, YSU
    # entrainment, Noah-MP daylight-only snow aging and SLUCM rain floors.
    # Each declared move binds the live identity on its own.
    assert all(domain.run.solar_generation == "wrf_480" for domain in exp.domains)
    old_sun = replace(exp, domains=tuple(replace(
        domain, run=replace(domain.run, solar_generation="wrf_461",
                            ysu_version="wrf_461", noahmp_snow_age="wrf_461",
                            urban_water_balance="wrf_471",
                            # 2.9.7: correct-mode HN1 (F22), read under the
                            # default Thompson; the golden predates it.
                            hydrometeor_handover_numbers=False))
        for domain in exp.domains))
    historical = replace(old_sun, domains=tuple(replace(
        domain, run=replace(domain.run, terrain_clock="measured"))
        for domain in old_sun.domains))
    catalog = SimpleNamespace(run_provenance={})
    # The 2.9.4 capabilities campaign's RunConfig keys, each at its default,
    # are the only later identity move (named per lane in test_water_overlay).
    from test_water_overlay import without_the_capability_campaign_keys
    without_the_capability_campaign_keys(monkeypatch, exp)
    assert experiment_fingerprint(exp, catalog) != experiment_fingerprint(old_sun, catalog)
    assert experiment_fingerprint(old_sun, catalog) != experiment_fingerprint(historical, catalog)
    assert experiment_fingerprint(historical, catalog) == fingerprint


def test_host_staging_and_prepared_store_products(exp, rank_api):
    from woof.core.preflight import estimate_devices
    split = replace(exp, devices=DeviceOptions(count=2, transport="host"))
    census = estimate_devices(split)
    assert census["host_staging_bytes"] > 0
    arrays = {"state/thp": np.zeros((12, 90, 120), dtype=np.float32)}
    geography = {"setup/msfu": np.zeros((90, 121), dtype=np.float64)}
    actual = estimate_devices(split, inventory=arrays, geography=geography)
    assert actual["host_store_bytes"] == arrays["state/thp"].nbytes + geography["setup/msfu"].nbytes
    repeated = estimate_devices(replace(split, devices=DeviceOptions(count=2, ids=(0, 0), transport="host")))
    assert repeated["host_staging_bytes"] == 0


def test_frame_snapshots_price_halos_and_repeated_cards(exp, rank_api):
    from math import prod
    from woof.core.devices_memory import estimate_devices, devices_gate
    from woof.core import streaming
    cfg = exp.root.run
    options = DeviceOptions(count=2, ids=(0, 1))
    split = replace(exp, devices=options)
    inventory = {"state/u": np.zeros((cfg.nz, cfg.ny, cfg.nx + 1), np.float32),
                 "state/thp": np.zeros((cfg.nz, cfg.ny, cfg.nx), np.float32),
                 "scratch/refl_10cm": np.zeros((cfg.nz, cfg.ny, cfg.nx), np.float32)}
    price = estimate_devices(split, inventory=inventory)
    specs = streaming.ranked_specs(cfg, options, halo=price["halo"])
    expected = [sum(prod(tuple(a.shape[:-2]) +
                        (spec.cny + a.shape[-2] - cfg.ny,
                         spec.cnx + a.shape[-1] - cfg.nx)) * a.dtype.itemsize
                    for a in inventory.values()) for spec in specs]
    assert [r["frame_snapshot_bytes"] for r in price["rank_shapes"]] == expected
    assert [r["frame_snapshot_bytes"] for r in price["cards"]] == expected
    for row in price["cards"]:
        assert row["total_bytes"] >= sum(row[key] for key in (
            "resident_bytes", "seam_bytes", "template_bytes", "frame_snapshot_bytes"))
    repeated = estimate_devices(
        replace(split, devices=replace(options, ids=(0, 0))), inventory=inventory)
    assert repeated["cards"][0]["frame_snapshot_bytes"] == sum(expected)
    gate = devices_gate(price, budgets={r["card"]: r["total_bytes"]
                                       for r in price["cards"]})
    assert not gate["refuse"] and "frame snapshots" in gate["verdict"]
    assert devices_gate(price, budgets={r["card"]: r["total_bytes"] - 1
                                       for r in price["cards"]})["refuse"]


def test_ineligible_ranks_never_price_partial_snapshots(exp, rank_api, monkeypatch):
    from woof.core import devices_memory as memory
    monkeypatch.setattr(memory, "FRAME_SNAPSHOT_LIMIT_BYTES", 1000)
    split = replace(exp, devices=DeviceOptions(count=2, ids=(0, 0)))
    price = memory.estimate_devices(split)
    assert [rank["frame_snapshot_bytes"] for rank in price["rank_shapes"]] == [0, 0]
    assert [rank["frame_snapshot_budget_bytes"] for rank in price["rank_shapes"]] == [0, 0]
    assert price["cards"][0]["frame_snapshot_bytes"] == 0


@pytest.mark.parametrize("ids", [(0, 1), (0, 0)])
def test_snapshot_allowance_uses_only_room_after_required_storage(exp, rank_api, ids):
    """Optional frame overlap must not refuse a fitting resident state."""
    from woof.core.devices_memory import estimate_devices, devices_gate

    split = replace(exp, devices=DeviceOptions(count=2, ids=ids))
    full = estimate_devices(split)
    required = {row["card"]: row["total_bytes"] - row["frame_snapshot_bytes"]
                for row in full["cards"]}
    budgets = {dev: size + 1001 for dev, size in required.items()}
    tight = estimate_devices(split, budgets=budgets)
    assert not devices_gate(tight, budgets=budgets)["refuse"]
    for row in tight["cards"]:
        assert row["frame_snapshot_bytes"] == 1001
        assert row["total_bytes"] == budgets[row["card"]]
        assert sum(tight["rank_shapes"][rank]["frame_snapshot_bytes"]
                   for rank in row["ranks"]) == 1001
    exact = estimate_devices(split, budgets=required)
    assert all(row["frame_snapshot_bytes"] == 0 for row in exact["rank_shapes"])
    assert not devices_gate(exact, budgets=required)["refuse"]
    short = {dev: size - 1 for dev, size in required.items()}
    refused = estimate_devices(split, budgets=short)
    assert devices_gate(refused, budgets=short)["refuse"]
    assert all(row["frame_snapshot_bytes"] == 0 for row in refused["rank_shapes"])


def test_admitted_snapshot_limits_reach_the_rank_builder(exp, monkeypatch):
    """Reverting this handoff would allocate the old 4 GiB after admission."""
    from woof import prepared_single_domain_forecast as forecast
    from woof.core import streaming

    split = replace(exp, devices=DeviceOptions(count=2))
    built = []
    def builder(bundle, **kwargs):
        built.append(kwargs)
        return object()
    monkeypatch.setattr(streaming, "ranked_domain_builder", builder)
    monkeypatch.setattr(streaming, "ranked_decision", lambda *a, **kw:
                        SimpleNamespace(halo=1))
    monkeypatch.setattr(streaming, "make_stepper", lambda *a, **kw:
                        SimpleNamespace(tiled_run=SimpleNamespace(
                            transport_report={"path": "host"})))
    node = SimpleNamespace(cfg=split.root, state=SimpleNamespace(), clock=object())
    forecast._devices_stepper(
        SimpleNamespace(geography={}), node, split, {},
        admission={"rank_shapes": [{"frame_snapshot_bytes": 123},
                                    {"frame_snapshot_bytes": 0}]})
    assert built[0]["snapshot_limits"] == (123, 0)


def test_template_memory_credit_counts_only_priced_unique_device_spans(exp):
    """Alias or unrelated live bytes must not turn a real overrun into a fit."""
    from woof.core.devices_memory import priced_template_live_bytes

    def array(pointer, size, device=0, contiguous=True):
        return SimpleNamespace(data=SimpleNamespace(ptr=pointer), nbytes=size,
                               device=SimpleNamespace(id=device),
                               flags=SimpleNamespace(c_contiguous=contiguous))
    template = SimpleNamespace(
        u=array(100, 100), v=array(150, 100),
        thp=array(300, 100, device=1), phi=array(500, 100, contiguous=False),
        unknown=array(700, 10000), _scratch={"unknown": array(20000, 10000)})
    assert priced_template_live_bytes(template, exp.root.run, device=0) == 150
    assert priced_template_live_bytes(template, exp.root.run, device=1) == 100


def test_host_bundle_credit_deduplicates_aliases_and_does_not_load_or_credit_files(tmp_path):
    """Credit only allocated bundle bytes, never reclaimable mappings or future input."""
    from woof.core.devices_memory import priced_host_live_bytes

    values = np.arange(40, dtype=np.uint8)
    tendency = np.ones(8, dtype=np.uint8)
    mapped = np.memmap(tmp_path / "mapped.bin", mode="w+", dtype=np.uint8, shape=(64,))
    side = SimpleNamespace(array_items=lambda: (("value", values[24:]),
                                                ("tendency", tendency)))
    boundary = SimpleNamespace(**{name: side for name in ("west", "east", "south", "north")})
    interval = SimpleNamespace(fields={"u": boundary}, unpriced=np.ones(100))
    store = {"a": values[:16], "alias": values[8:24]}
    geography = {"b": values[20:32], "strided": values[::2], "file": mapped[:32]}
    result = priced_host_live_bytes(
        store, geography, SimpleNamespace(intervals=(interval, interval)))
    assert result == {"host_store_bytes": 32, "host_boundary_bytes": 16}
    class Lazy:
        def __iter__(self):
            raise AssertionError("admission must not read future forcing")
    assert priced_host_live_bytes(store, geography, SimpleNamespace(intervals=Lazy())) == {
        "host_store_bytes": 32, "host_boundary_bytes": 0}


def test_post_loader_admission_credits_only_the_priced_template(exp, monkeypatch):
    """Retained template and host bundle are paid once; unrelated allocations stay paid."""
    import sys
    from woof import prepared_single_domain_forecast as forecast
    from woof.core import devices_memory, preflight as pf, resident_admission

    class Card:
        def __init__(self, device):
            self.device = device
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    runtime = SimpleNamespace(is_hip=False,
        getDeviceCount=lambda: 2,
        getDeviceProperties=lambda _: {"name": b"test", "multiProcessorCount": 1,
                                      "maxThreadsPerMultiProcessor": 32},
        deviceGetLimit=lambda _: 1024)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(Device=Card, runtime=runtime),
        get_default_memory_pool=lambda: SimpleNamespace(free_bytes=lambda: 100)))
    monkeypatch.setattr(pf, "device_free_and_total_bytes", lambda _: (700, 1000))
    monkeypatch.setattr(pf, "read_compile_platform", lambda: None)
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 60)
    monkeypatch.setattr(resident_admission, "device_free_bytes", lambda: 800)
    monkeypatch.setattr(devices_memory, "priced_template_live_bytes", lambda *a, **k: 200)
    sampled = []
    def estimate(*args, budgets, **kwargs):
        sampled.append(dict(budgets))
        return {"cards": [{"card": dev, "resident_bytes": 650, "seam_bytes": 0,
                           "template_bytes": 150 if dev == 0 else 0,
                           "frame_snapshot_bytes": 0,
                           "total_bytes": 800 if dev == 0 else 650}
                          for dev in (0, 1)],
                "host_store_bytes": 200, "host_staging_bytes": 40,
                "host_boundary_bytes": 50, "host_bytes": 290}
    monkeypatch.setattr(pf, "estimate_devices", estimate)
    split = replace(exp, devices=DeviceOptions(count=2))
    forcing = np.ones(10, dtype=np.float64)
    side = SimpleNamespace(array_items=lambda: (("value", forcing),))
    boundary = SimpleNamespace(**{name: side for name in ("west", "east", "south", "north")})
    result = forecast._admit_devices_forecast(
        split, exp.root.run, geography={"g": np.ones(20, np.float32)},
        inventory={"a": np.ones(32, np.float32)}, forcing_intervals=1, template=object(),
        boundaries=SimpleNamespace(intervals=(SimpleNamespace(fields={"u": boundary}),)))
    assert sampled == [{0: 800, 1: 800}, {0: 950, 1: 800}]
    assert result["budget_terms"][0] == {
        "device_free_bytes": 700, "reusable_pool_bytes": 100,
        "template_live_credit_bytes": 150, "budget_bytes": 950}
    assert result["budget_terms"][1]["template_live_credit_bytes"] == 0
    assert result["host_budget_terms"] == {
        "available_bytes": 60, "store_live_credit_bytes": 200,
        "boundary_live_credit_bytes": 50, "budget_bytes": 310}


def test_boundary_frame_and_host_budget_refusals(exp, rank_api):
    from woof.core.preflight import estimate_devices
    from woof.core.devices_memory import devices_gate
    cfg = replace(exp.root.run, nx=300, ny=400, specified=True,
                  spec_zone=1, relax_zone=100)
    split = replace(exp, domains=(replace(exp.root, run=cfg),), devices=DeviceOptions(count=2))
    with pytest.raises(DevicesRefused, match="twice boundary frame 100"):
        estimate_devices(split)
    safe = replace(exp, devices=DeviceOptions(count=2))
    estimate = estimate_devices(safe)
    gate = devices_gate(estimate, host_budget=estimate["host_bytes"] - 1)
    assert gate["refuse"] and "host: REFUSED" in gate["verdict"]


def test_downscale_refuses_before_output_reservation(tmp_path):
    from woof.downscale import _downscale_main
    path = tmp_path / "child.toml"
    path.write_text('[devices]\ncount = 2\n')
    with pytest.raises(DevicesRefused, match="woof downscale.*silent one-card run"):
        _downscale_main(SimpleNamespace(child_config=path), None, [])


def test_default_public_snapshot_omits_new_control(exp):
    from woof.experiment import experiment_config_document
    from woof.runplan import _config_snapshot, _schema_default_resolutions
    assert "devices" not in experiment_config_document(exp)
    assert "devices" not in _config_snapshot(exp, None)["experiment"]
    assert all(row["key"] != "devices" for row in _schema_default_resolutions({}))
    split = replace(exp, devices=DeviceOptions(count=2))
    assert experiment_config_document(split)["devices"] == {"count": 2}


@pytest.mark.parametrize("table", [
    {"count": 2, "domains": []}, {"count": 2, "domains": [0]},
    {"count": 2, "domains": [1, 1]}, {"count": 2, "domains": ["1"]},
    {"count": 2, "domains": [True]}, {"count": 2, "domains": 1},
    {"count": 1, "domains": [1]},
])
def test_domains_key_refusals(table):
    """``[devices] domains``: a typo must not split a grid the list does not name."""
    with pytest.raises(ValueError):
        DeviceOptions.from_mapping(table)


def test_split_grid_ids_names_the_tree_grids():
    both = DeviceOptions(count=2, ids=(0, 0))
    assert both.split_grid_ids([1, 2]) == (1, 2)
    nest = DeviceOptions(count=2, ids=(0, 0), domains=(2,))
    assert nest.split_grid_ids([1, 2]) == (2,)
    assert nest.to_mapping()["domains"] == [2]
    assert nest.to_json()["domains"] == [2]
    assert DeviceOptions.from_mapping(nest.to_mapping()) == nest
    assert "domains" not in both.to_json()
    assert DEVICES_OFF.split_grid_ids([1, 2]) == ()
    with pytest.raises(ValueError, match=r"grid\(s\) \[3\]"):
        DeviceOptions(count=2, domains=(3,)).split_grid_ids([1, 2])


def _tree():
    from woof.experiment import load_experiment
    return load_experiment(ROOT / "configs" / "gfs_wrf_hierarchy_proof.toml")


def test_tree_split_refusals_name_their_breakage():
    from woof.core.devices import validate_tree_devices
    tree = _tree()
    assert len(tree.domains) == 2
    assert validate_tree_devices(tree) == ()
    split = replace(tree, devices=DeviceOptions(count=2, ids=(0, 0)))
    assert validate_tree_devices(split) == (1, 2)
    nest_only = replace(tree, devices=DeviceOptions(count=2, ids=(0, 0), domains=(2,)))
    assert validate_tree_devices(nest_only) == (2,)
    parent_only = replace(tree, devices=DeviceOptions(count=2, ids=(0, 0), domains=(1,)))
    with pytest.raises(DevicesRefused, match="resident grid"):
        validate_tree_devices(parent_only)
    with pytest.raises(DevicesRefused, match="names grid"):
        replace(tree, devices=DeviceOptions(count=2, domains=(7,)))
    moving = SimpleNamespace(devices=split.devices, domains=split.domains,
                             relocation=SimpleNamespace(enabled=True),
                             start_time=split.start_time,
                             domain_start_time=lambda gid: split.start_time)
    with pytest.raises(DevicesRefused, match="moving nest"):
        validate_tree_devices(moving)
    late = SimpleNamespace(devices=split.devices, domains=split.domains,
                           relocation=split.relocation, start_time=split.start_time,
                           domain_start_time=lambda gid: (split.start_time if gid == 1
                                                          else datetime(2030, 1, 1)))
    with pytest.raises(DevicesRefused, match="start after the run does"):
        validate_tree_devices(late)


def test_tree_runner_reads_the_split_beside_the_bound_config():
    """The tree runner's own door: --devices-table and --devices parse."""
    import json
    from woof.prepared_domain_tree_forecast import build_parser
    args = build_parser().parse_args([
        "--prepared-root", "p", "--experiment-config", "e.toml",
        "--experiment-config-sha256", "0" * 64, "--outdir", "o",
        "--devices-table", json.dumps({"count": 2, "ids": [0, 0], "domains": [1]}),
        "--devices", "2"])
    assert DeviceOptions.from_mapping(json.loads(args.devices_table)) == \
        DeviceOptions(count=2, ids=(0, 0), domains=(1,))
    assert args.devices == 2


def test_check_devices_flag_prices_every_card(rank_api, tmp_path, monkeypatch, capsys):
    """``woof check CONFIG --devices N`` reports the per-card envelope.

    The named breakage: a domain only a split can hold, checked alone, was
    priced on one card and refused (158.66 GiB against 93.93) while
    ``woof go --devices 2`` admitted it at 86.1 and 85.2 GiB per card.
    """
    from woof import cli
    from woof.core import preflight as pf
    from woof.core.devices_memory import GIB
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 512 * GIB)
    path = tmp_path / "one.toml"
    path.write_text((ROOT / "tests" / "fixtures" / "devices_hrrr_grid_admission.toml").read_text())
    parser = cli.build_parser()
    args = parser.parse_args(["check", str(path), "--budget-gib", "96",
                              "--vram-gib", "96", "--devices", "2"])
    assert pf.check_main(args) == 0
    captured = capsys.readouterr()
    assert "--devices 2 replaces [devices] count = 1" in captured.err
    assert "go: [devices] 2 slabs on cards [0, 1]" in captured.out
    assert "card 0: ADMITTED" in captured.out and "card 1: ADMITTED" in captured.out


def test_resident_nest_under_a_split_parent_sizes_its_arena_from_the_tree():
    """The arena of a resident nest under a split (or streamed) parent.

    ``shared_scratch_arena_shapes`` takes the whole tree for exactly this
    road, and the alias planner beside it did not: it asked for the shapes
    of the resident domains alone, and a nest whose parent is not among
    them raised KeyError ("domain 2 names parent 1 ...") before the first
    restore.  MEASURED: an SF 2.25 km / 750 m tree with only the parent
    split stopped there.
    """
    from woof.core import preflight as pf
    from woof.core.state import build_shared_scratch_arena
    tree = _tree()
    parent, nest = tree.domains
    aliases = pf.shared_scratch_arena_aliases((nest,), tree.domains)
    assert isinstance(aliases, dict)
    total = pf.shared_scratch_arena_bytes((nest,), tree.domains)
    assert total > 0
    with pytest.raises(KeyError, match="names parent"):
        pf.shared_scratch_arena_aliases((nest,))
    assert callable(build_shared_scratch_arena)


def test_a_split_tree_is_priced_per_card_by_one_function(rank_api):
    """``estimate_devices_tree``: the check, the go gate and the tree runner.

    The named breakage: ``woof check --devices`` and the ``woof go`` gate
    refused every tree ("rank memory admission for a nested tree"), so a
    nested template that fits only split could not be priced, or run, from
    the front door.  Each split grid is priced exactly as a split single
    domain is, each resident grid on the first card, and every card row
    says what each grid puts on it.
    """
    from woof.core.devices_memory import (GIB, devices_gate, estimate_devices,
                                           estimate_devices_tree, tree_grid_as_root)
    tree = _tree()
    both = replace(tree, devices=DeviceOptions(count=2, ids=(0, 1)))
    estimate = estimate_devices_tree(both)
    assert estimate["split_grid_ids"] == [1, 2]
    rows = {row["card"]: row for row in estimate["cards"]}
    host = 0
    for dc in tree.domains:
        alone = estimate_devices(replace(tree, domains=(tree_grid_as_root(dc),),
                                         devices=DeviceOptions(count=2, ids=(0, 1))))
        host += alone["host_bytes"]
        for row in alone["cards"]:
            assert rows[row["card"]]["grids"][int(dc.grid_id)] == row["total_bytes"]
    assert estimate["host_bytes"] == host
    assert all(row["total_bytes"] == sum(row["grids"].values()) for row in estimate["cards"])
    nest = estimate_devices_tree(
        replace(tree, devices=DeviceOptions(count=2, ids=(0, 1), domains=(2,))))
    rows = {row["card"]: row for row in nest["cards"]}
    assert set(rows[0]["grids"]) == {1, 2} and set(rows[1]["grids"]) == {2}
    assert [g["road"] for g in nest["grids"]] == ["resident", "split"]
    gate = devices_gate(nest, budgets={0: 96 * GIB, 1: 96 * GIB}, host_budget=None)
    assert not gate["refuse"]
    assert "card 0: ADMITTED: d01 " in gate["verdict"] and "+ d02 " in gate["verdict"]
    tight = devices_gate(nest, budgets={0: 1, 1: 96 * GIB}, host_budget=None)
    assert tight["refuse"] and "card 0: REFUSED" in tight["verdict"]
    with pytest.raises(DevicesRefused, match="resident grid"):
        estimate_devices_tree(
            replace(tree, devices=DeviceOptions(count=2, ids=(0, 1), domains=(1,))))


def _go_tree(tmp_path, table):
    source = ROOT / "configs" / "moving_nest_20110427_static_2km.toml"
    path = tmp_path / source.name
    # The [fetch] route on a source with a native chain: a [case_data]
    # table would take the config-driven route instead.
    text = source.read_text().split("[case_data]")[0]
    text = text.replace('source = "era5"', 'source = "gfs"').replace(
        'cycle = "2011-04-26T12"', 'cycle = "2011-04-27T06"')
    path.write_text(text + "\n" + table)
    path.with_suffix(".namelist.wps").write_text(
        source.with_suffix(".namelist.wps").read_text())
    return path


def test_go_carries_a_split_tree_to_the_tree_runner(rank_api, declared_two_card_probe, tmp_path, monkeypatch):
    """``woof go TREE`` with ``[devices]``: planned, priced and relayed.

    It used to stop at "woof go nested tree or moving nests does not read
    [devices]" while the tree runner beneath it splits a tree.  What the
    runner cannot split is still refused here by name, before the download.
    """
    from woof import go_cli
    from woof.core import preflight as pf
    from woof.core.devices_memory import GIB
    monkeypatch.setattr(pf, "host_available_bytes", lambda: 512 * GIB)
    path = _go_tree(tmp_path, '[devices]\ncount = 2\nids = [0, 1]\n')
    plan = go_cli.plan_from_config(path, outdir=tmp_path / "run", run_stamp=False)
    assert plan["domains"] == 2 and plan["runner"] == go_cli.TREE_RUNNER_MODULE
    assert plan["devices"]["count"] == 2
    assert "d01 split 1x2" in plan["devices_sentence"]
    assert "d02 split" in plan["devices_sentence"]
    gate = go_cli.memory_gate(plan, vram_gib=96)
    assert not gate["refuse"]
    assert "card 0: ADMITTED: d01 " in gate["verdict"]
    assert "card 1: ADMITTED: d01 " in gate["verdict"]
    command = go_cli.tree_forecast_command(
        {**plan, "devices_count_override": 2},
        digests={"preparation_receipt": "r", "experiment_config": "c"})
    assert command[command.index("--devices") + 1] == "2"
    assert "--devices" not in go_cli.tree_forecast_command(
        plan, digests={"preparation_receipt": "r", "experiment_config": "c"})
    (tmp_path / "nest").mkdir()
    nest_only = _go_tree(tmp_path / "nest", '[devices]\ncount = 2\ndomains = [2]\n')
    plan = go_cli.plan_from_config(nest_only, outdir=tmp_path / "run2", run_stamp=False)
    assert "d01 resident on card 0" in plan["devices_sentence"]
    (tmp_path / "parent").mkdir()
    parent_only = _go_tree(tmp_path / "parent", '[devices]\ncount = 2\ndomains = [1]\n')
    with pytest.raises(DevicesRefused, match="resident grid"):
        go_cli.plan_from_config(parent_only, outdir=tmp_path / "run3", run_stamp=False)


def test_run_plan_prepared_chains_admit_a_split_tree(tmp_path):
    """run-plan's prepared chains hand a tree to the tree runner, which splits it."""
    from woof.runplan import PlanError, streaming_decision
    tree = _tree()
    split = replace(tree, devices=DeviceOptions(count=2, ids=(0, 0)))
    for chain in ("prepared:go", "prepared:hrrr", "prepared:staged", "prepared:existing"):
        assert streaming_decision(split, chain=chain) is None
    with pytest.raises(PlanError, match="silent one-card run"):
        streaming_decision(split, chain="experiment")
    parent_only = replace(tree, devices=DeviceOptions(count=2, ids=(0, 0), domains=(1,)))
    with pytest.raises(PlanError, match="resident grid"):
        streaming_decision(parent_only, chain="prepared:go")


@pytest.mark.parametrize("same_card,rc,verdict,hits,accepted", [
    (True, 1, "NOT-FIRED control wrong_card did not fire", {"calls": 2}, True),
    (True, 0, "FIRED-DIFFER", {"calls": 2}, False),
    (False, 0, "FIRED-REFUSED CUDARuntimeError: illegal address", {"calls": 2, "foreign": 1}, True),
    (False, 1, "FIRED-REFUSED CUDARuntimeError: illegal address", {"calls": 2, "foreign": 1}, False),
    (False, 0, "FIRED-DIFFER", {"calls": 2}, False),
    (False, 0, "FIRED-DIFFER", {"foreign": 1}, False),
    (False, 1, "ERROR FileNotFoundError: missing table", {"calls": 2, "foreign": 1}, False),
])
def test_wrong_card_verdict_requires_reached_fault_and_child_status(
        monkeypatch, same_card, rc, verdict, hits, accepted):
    import json
    import subprocess
    from tilestream.ranks_gate import run_wrong_card_control
    stdout = "WRONG_CARD_SEAM " + json.dumps(hits) + "\nWRONG_CARD " + verdict
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=stdout, stderr="", returncode=rc))
    cards = [0, 0] if same_card else [0, 1]
    if accepted:
        result = run_wrong_card_control(cards)
        assert ("SKIPPED" if same_card else "fired") in result
    else:
        with pytest.raises(AssertionError):
            run_wrong_card_control(cards)


def test_wrong_card_child_rejects_unrelated_failures(monkeypatch, capsys):
    from tilestream import ranks_gate
    def missing(*args, **kwargs):
        raise FileNotFoundError("missing physics table")
    monkeypatch.setattr(ranks_gate, "integrate", missing)
    assert ranks_gate.wrong_card_child([0, 1]) == 1
    assert "WRONG_CARD ERROR FileNotFoundError" in capsys.readouterr().out


@pytest.mark.parametrize("status,fired", [(700, True), (719, True), (2, False)])
def test_wrong_card_child_judges_the_wrapped_cuda_cause(monkeypatch, capsys, status, fired):
    from tilestream import ranks_gate
    cuda_error = type("CUDARuntimeError", (Exception,),
                      {"__module__": "cupy_backends.cuda.api.runtime"})("CUDA memory fault")
    cuda_error.status = status
    def wrapped(*args, **kwargs):
        raise RuntimeError("rank step failed") from cuda_error
    monkeypatch.setattr(ranks_gate, "integrate", wrapped)
    assert ranks_gate.wrong_card_child([0, 1]) == (0 if fired else 1)
    assert ("WRONG_CARD FIRED-REFUSED" if fired else "WRONG_CARD ERROR") in capsys.readouterr().out


def test_bench_cli_forwards_the_requested_card_list(monkeypatch, capsys):
    from tilestream import ranks_gate
    cards = []
    monkeypatch.setattr(ranks_gate, "benchmark", lambda devices: cards.extend(devices))
    assert ranks_gate.main(["bench", "--devices", "1", "--devices", "0"]) == 0
    assert cards == [1, 0]
    assert "RANKS bench: PASS" in capsys.readouterr().out


@pytest.mark.parametrize("card", [0, 1])
def test_single_rank_gate_selects_requested_card_without_public_split_keys(card):
    from tilestream.ranks_gate import _SingleCardGateOptions
    options = _SingleCardGateOptions(card)
    assert options.device_ids() == (card,)
    assert options.resolved_grid(128, 104) == (1, 1)
    assert not options.enabled and options.to_mapping() == {"count": 1}
    with pytest.raises(ValueError, match="one-card run has no split"):
        DeviceOptions(count=1, ids=(card,))


@pytest.mark.parametrize("count", [1, 9, 16])
def test_run_plan_accepts_slab_counts_for_geometry_admission(tmp_path, count):
    """The plan leaves slab geometry and card availability to admission."""
    from woof.runplan import load_plan
    from test_runplan import _staged_plan
    plan = load_plan(_staged_plan(tmp_path, tmp_path / "run",
                                 run_options={"devices": count}))
    assert plan.run_options["devices"] == count


@pytest.mark.parametrize("count", [0, -1, True, 2.5, "9"])
def test_run_plan_refuses_nonpositive_or_noninteger_slab_counts(tmp_path, count):
    from woof.runplan import PlanError, load_plan
    from test_runplan import _staged_plan
    with pytest.raises(PlanError, match="positive integer slab count"):
        load_plan(_staged_plan(tmp_path, tmp_path / "run",
                               run_options={"devices": count}))


# The staged chain this drives starts with run-plan's capability preflight,
# which refuses a card-integrating plan at acceptance (CapabilityMissing:
# "needs cupy") on an install without cupy, before the fetch stage runs.
# Without cupy the chain exits 1 there and this test would measure that
# refusal instead of the split table reaching sim; it needs what every
# other execute_plan chain test in test_staged_chain_as_posted.py needs.
@requires_cupy
@pytest.mark.parametrize("posted", [False, True])
def test_posted_chain_keeps_split_table_and_count_override(tmp_path, monkeypatch, posted):
    """Posted fetch lifecycle and split composition must both reach sim."""
    import threading
    from woof import runplan
    from test_staged_chain_as_posted import _handoff, _run, _schedule

    options = DeviceOptions(count=2, ids=(0, 0), transport="host")
    resolve = runplan.resolve_plan
    load = runplan.load_plan
    prepared = threading.Event()
    overlapped = []

    def resolve_split(*args, **kwargs):
        resolution, exp, data = resolve(*args, **kwargs)
        return resolution, replace(exp, devices=options), data

    def load_split(*args, **kwargs):
        plan = load(*args, **kwargs)
        return replace(plan, run_options={**plan.run_options, "devices": 2})

    # The helper imports load_plan into its own module, while the executor
    # resolves its experiment through runplan.resolve_plan.
    import test_staged_chain_as_posted as chain_test
    monkeypatch.setattr(runplan, "resolve_plan", resolve_split)
    monkeypatch.setattr(chain_test, "load_plan", load_split)

    def fetch(arguments, run_dir, **kwargs):
        output = Path(arguments[arguments.index("--out") + 1])
        if posted:
            _schedule(output)
        _handoff(output, ["--source", "icon-eu", "--input-list", "inputs.txt",
                          "--author-input-manifest", "inputs.json"], posted=posted)
        if posted:
            overlapped.append(prepared.wait(20))
        return {"source": "icon-eu"}

    def prep(arguments):
        prepared.set()
        output = Path(arguments[arguments.index("--output-root") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "proof.json").write_text("{}", encoding="utf-8")

    code, staged, events, _plan = _run(tmp_path, monkeypatch, fetch=fetch, prep=prep)
    assert code == 0
    command = next(value for label, value in staged if label == "sim_command")
    assert command["devices"] == 2
    assert command["devices_options"] is options
    assert command["progress_format"] == "jsonl"
    if posted:
        assert overlapped == [True], "preparation did not overlap the posted fetch"
        fetched = next(event for event in events if event["event"] == "stage_finished"
                       and event["stage"] == "fetch")
        assert fetched["fetch"]["as_posted"] is True


@pytest.mark.parametrize("diff,mix,mosaic,canopy", [
    (2, True, 0, "dominant"), (1, False, 1, "every_tile"),
])
def test_checkpoint_echo_keeps_mosaic_and_diffusion_semantics(exp, diff, mix, mosaic, canopy):
    """Resident and streamed headers share default omission and active options."""
    from dataclasses import asdict
    from woof.io.restart import _require_config_match, configuration_echo
    cfg = replace(exp.root.run, diff_opt=diff, mix_full_fields=mix,
                  sf_surface_mosaic=mosaic, mosaic_cat=3,
                  mosaic_urban_canopy=canopy)
    echoed = configuration_echo(cfg)
    _require_config_match(echoed, cfg, "checkpoint")
    if mosaic == 0:
        assert not {"sf_surface_mosaic", "mosaic_cat", "mosaic_urban_canopy"} & echoed.keys()
        assert not {"diff_opt", "mix_full_fields"} & echoed.keys()
    else:
        assert {key: echoed[key] for key in ("sf_surface_mosaic", "mosaic_cat",
                                             "mosaic_urban_canopy", "diff_opt",
                                             "mix_full_fields")} == {
            "sf_surface_mosaic": 1, "mosaic_cat": 3,
            "mosaic_urban_canopy": "every_tile", "diff_opt": 1,
            "mix_full_fields": False}
        with pytest.raises(ValueError, match="diff_opt"):
            _require_config_match(echoed, replace(cfg, diff_opt=2), "checkpoint")
        with pytest.raises(ValueError, match="mosaic_urban_canopy"):
            _require_config_match(echoed, replace(cfg, mosaic_urban_canopy="dominant"),
                                  "checkpoint")
    assert asdict(cfg)["mosaic_cat"] == 3


@pytest.mark.parametrize("selector,reason", [
    ("sf_surface_mosaic", "LANDUSEF"), ("slope_rad", "shadow"),
])
@pytest.mark.parametrize("door", ["source", "tree", "price", "ranked"])
def test_split_refuses_physics_without_its_domain_setup(exp, rank_api, selector, reason, door):
    """Every split door must refuse before building a partial physics state."""
    from woof.core import streaming
    from woof.core.devices import validate_device_road, validate_tree_devices
    from woof.core.devices_memory import estimate_devices

    cfg = replace(exp.root.run, **{selector: 1})
    domain = replace(exp.root, run=cfg)
    options = DeviceOptions(count=2, ids=(0, 0))
    # Prepared admission may receive an already resolved experiment, so its
    # entry points must repeat the guard rather than rely on construction.
    requested = SimpleNamespace(domains=(domain,), root=domain, devices=options)
    with pytest.raises(DevicesRefused, match=selector) as refused:
        if door == "source":
            replace(exp, domains=(domain,), devices=options)
        elif door == "tree":
            validate_tree_devices(requested)
        elif door == "price":
            estimate_devices(requested)
        else:
            streaming.ranked_decision(cfg, requested.devices)
    assert reason.lower() in str(refused.value).lower()


@pytest.mark.parametrize("selector", ["sf_surface_mosaic", "slope_rad"])
def test_nonlocal_physics_remains_available_on_resident_grids(exp, selector):
    """The split refusal must not remove the resident physics capability."""
    from woof.core.devices import validate_device_road, validate_tree_devices

    cfg = replace(exp.root.run, **{selector: 1})
    resident = replace(exp, domains=(replace(exp.root, run=cfg),))
    validate_device_road(resident.devices, domains=resident.domains)
    assert validate_tree_devices(resident) == ()
    tree = _tree()
    parent = replace(tree.root, run=replace(tree.root.run, **{selector: 1}))
    nest_only = replace(tree, domains=(parent, tree.domains[1]),
                        devices=DeviceOptions(count=2, ids=(0, 0), domains=(2,)))
    validate_device_road(nest_only.devices, domains=nest_only.domains)
    assert validate_tree_devices(nest_only) == (2,)


@pytest.mark.parametrize("partial_probe", [False, True])
def test_split_check_without_a_card_budget_fails_closed(rank_api, monkeypatch, capsys, partial_probe):
    """An unmeasured estimate cannot establish that either card fits."""
    import json
    from woof import cli, doctor
    from woof.core import devices, preflight
    from woof.core.devices_memory import GIB

    monkeypatch.setattr(doctor, "_cuda_headers_check", lambda **_kw: SimpleNamespace(
        status="verified", detail="CUDA compilation is available", action=None))
    reading = ({"visible_count": 2, "cards": {0: {"free_bytes": 95 * GIB}}}
               if partial_probe else None)
    monkeypatch.setattr(devices, "probe_devices", lambda: reading)
    monkeypatch.setattr(preflight, "profile_from_device_probe", lambda _: None)
    monkeypatch.setattr(preflight, "live_device_local_memory_profile", lambda: None)
    monkeypatch.setattr(preflight, "declares_the_local_card", lambda *_: False)
    monkeypatch.setattr(preflight, "host_available_bytes", lambda: 512 * GIB)
    path = ROOT / "tests" / "fixtures" / "devices_hrrr_grid_admission.toml"
    args = cli.build_parser().parse_args([
        "check", str(path), "--devices", "2", "--vram-gib", "96", "--json"])
    assert preflight.check_main(args) == 2
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert len(document["devices"]["cards"]) == 2
    assert all(card["total_bytes"] > 0 for card in document["devices"]["cards"])
    assert document["refuse"] is True
    assert document["evaluable"] is False
    assert document["check_exit_code"] == 2
    assert document["unmeasured_cards"] == ([1] if partial_probe else [0, 1])
    assert "UNMEASURED" in document["verdict"]
    assert "budget" in captured.err.lower()
    assert "--free-gib" in captured.err or "--budget-gib" in captured.err


@pytest.mark.parametrize("selector", ["sf_surface_mosaic", "slope_rad"])
def test_ranked_constructor_refuses_unbuilt_physics_before_cuda_import(exp, monkeypatch, selector):
    import sys
    from tilestream.ranks import RankedRun

    # A CUDA import would fail. The named physics refusal must happen first,
    # without looking at dummy store or geography inputs or allocating.
    monkeypatch.setitem(sys.modules, "cupy", None)
    cfg = replace(exp.root.run, **{selector: 1})
    with pytest.raises(DevicesRefused, match=selector):
        RankedRun(None, cfg, options=DeviceOptions(count=2, ids=(0, 0)),
                  scalars=None, geography=None, template=None)


def test_a_split_tree_that_does_not_fit_is_refused_by_name(rank_api, monkeypatch, capsys):
    """The tree runner's per-card admission raises ``DevicesRefused``.

    The named breakage: ``_admit_devices_tree`` raised a name it never
    imported, so a nested ``[devices]`` tree too big for its cards died on
    ``NameError: name 'DevicesRefused' is not defined`` instead of the
    refusal that says which card is over (shipped in 2.8.7; found by
    tests/test_no_undefined_names.py).  Runs the real pricing with a
    stand-in CUDA runtime whose cards hold 1 GiB each.
    """
    import sys
    from woof.core import device_probe, preflight
    from woof.core.devices_memory import GIB
    from woof.core.resident_admission import MEMORY_GATE_OVERRIDE_ENV
    from woof.prepared_domain_tree_forecast import _admit_devices_tree

    class Device:
        def __init__(self, dev): self.dev = dev
        def __enter__(self): return self
        def __exit__(self, *args): return False
    runtime = SimpleNamespace(is_hip=False, getDeviceCount=lambda: 2,
                              memGetInfo=lambda: (1 * GIB, 2 * GIB))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(Device=Device, runtime=runtime)))
    monkeypatch.setattr(device_probe, "cuda_device_identity", lambda dev: None)
    monkeypatch.setattr(preflight, "host_available_bytes", lambda: 512 * GIB)
    monkeypatch.delenv(MEMORY_GATE_OVERRIDE_ENV, raising=False)
    tree = replace(_tree(), devices=DeviceOptions(count=2, ids=(0, 1)))
    with pytest.raises(DevicesRefused, match=r"card 0: REFUSED"):
        _admit_devices_tree(tree, (1, 2), forcing_intervals=None,
                            forcing_interval_seconds=3600.0, source=None)
    assert "split on cards [0, 1]" in capsys.readouterr().out

"""The one-command global run, its machine surface and the export door.

CPU only: no card, no network.  The mesh stage is driven against a local
archive through a ``file://`` URL; the export door against a stand-in converter
that writes what the real one writes (a frame and its JSON receipt)."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile

import pytest

from woof.hex import global_door, run_status
from woof.hex.cli import build_parser, main


# ---------------------------------------------------------------------------
# run_status
# ---------------------------------------------------------------------------
def test_status_and_events_follow_the_stages(tmp_path):
    status = run_status.StatusWriter(tmp_path, "global", stages=("mesh", "static"), row="global-120km")
    status.skip("mesh", "cached")
    status.stage("static", step_count=3)
    status.step(2, note="halfway")
    doc = run_status.read_status(tmp_path)
    assert doc["schema"] == run_status.SCHEMA and doc["state"] == "running"
    assert doc["stage"] == "static" and doc["stage_index"] == 1 and doc["stage_count"] == 2
    assert doc["step"] == 2 and doc["step_count"] == 3 and doc["row"] == "global-120km"
    status.done("finished")
    doc = run_status.read_status(tmp_path)
    assert doc["state"] == "done"
    kinds = [event["event"] for event in run_status.read_events(tmp_path)]
    assert kinds == ["start", "stage-skip", "stage-start", "step", "stage-end", "done"]
    assert not (tmp_path / "status.json.tmp").exists()
    assert status.log_path.read_text(encoding="utf-8").count("\n") == len(kinds)


def test_a_refusal_is_not_a_failure(tmp_path):
    status = run_status.StatusWriter(tmp_path, "global", stages=("mesh",))
    status.stage("mesh")
    status.refused("no pin")
    doc = run_status.read_status(tmp_path)
    assert doc["state"] == "refused" and doc["reason"] == "no pin"
    assert run_status.read_events(tmp_path)[-1]["event"] == "refused"


def test_a_foreign_status_document_is_refused(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"schema": "gpuwm-global-status-v1"}))
    with pytest.raises(ValueError, match="gpuwm-hex-status-v1"):
        run_status.read_status(tmp_path)


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def test_the_shipped_table_has_the_four_resolutions_and_the_morrison_arm():
    table = global_door.load_table()
    rows = {row["id"]: row for row in table["rows"]}
    for km, mesh in ((120, "x1.40962"), (60, "x1.163842"), (30, "x1.655362"), (15, "x1.2621442")):
        assert rows[f"global-{km}km"]["mesh"] == mesh
        # MPAS physics is the hex default (owner ruling 2026-10-25): the four default rows take the mpas arm.
        assert rows[f"global-{km}km"]["physics"] == "mpas"
        assert rows[f"global-{km}km-morrison"]["physics"] == "morrison"
    assert table["physics"]["mpas"] == {**table["physics"]["mpas"], "route": "mpas", "overrides": {}}
    assert table["physics"]["arwen"]["overrides"] == {} and table["physics"]["arwen"]["route"] == "arwen"
    assert table["physics"]["morrison"]["overrides"]["mp_physics"] == "10"
    for mesh_id, mesh in table["meshes"].items():
        assert mesh["url"].endswith(f"/{mesh_id}.tar.gz")
        assert mesh["grid_member"] == f"{mesh_id}.grid.nc"
        for key in ("archive_sha256", "grid_sha256"):
            assert len(mesh[key]) == 64 and int(mesh[key], 16) >= 0, (mesh_id, key)
    assert (global_door.DATA / table["vertical_spec"]).is_file()
    assert table["static"]["argv"] == ["--native-init", "--native-gwd-static"]


def test_the_runtime_mesh_name_never_shadows_a_registry_row():
    from woof.hex.drivers.mpas_mesh_binding import MESH_BINDINGS

    for mesh_id in global_door.load_table()["meshes"]:
        assert global_door.runtime_mesh_name(mesh_id) not in MESH_BINDINGS


def _args(tmp_path, *extra):
    parser = build_parser()
    return parser.parse_args(["global", "--row", "global-60km-morrison", "--cycle", "2026-10-11T12",
                              "--work-dir", str(tmp_path / "run"), "--cache-dir", str(tmp_path / "cache"),
                              "--stage", "plan", *extra])


def test_global_turbines_are_a_forecast_option(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    commands = global_door.plan(_args(tmp_path, '--wind-farms', 'uswtdb'))['commands']
    argv = commands['forecast']
    assert argv[argv.index('--wind-farms') + 1] == 'uswtdb'
    assert '--wind-farms' not in commands['init']


def test_the_plan_names_every_stage_and_composes_the_existing_doors(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    value = global_door.plan(_args(tmp_path))
    assert value["schema"] == global_door.PLAN_SCHEMA
    assert value["stages"] == list(global_door.STAGES)
    commands = value["commands"]
    assert commands["init"][:5] == ["init-source", "--source", "gfs", "--cycle", "2026-10-11T12"]
    forecast = commands["forecast"]
    assert forecast[forecast.index("--mesh") + 1] == "hex-x1.163842-native"
    assert forecast[forecast.index("--start-time") + 1] == "2026-10-11_12:00:00"
    physics = [forecast[i + 1] for i, token in enumerate(forecast) if token == "--physics"]
    assert "mp_physics=10" in physics and "config_radt_lw_scheme=rrtmgp_lw" in physics
    assert "--samehost-devices" not in forecast  # the forecast door picks the card count
    assert commands["products"][0] == "products" and commands["export"][0] == "export"
    assert commands["score"][0] == "score"
    assert value["paths"]["static"].endswith("x1.163842.native.static.nc")
    assert value["static_argv"][-2] == "--receipt"
    assert value["cards"]["selected_cards"] is None
    # every argv is a real door's argv: the parser accepts each one
    parser = build_parser()
    for name, argv in commands.items():
        parser.parse_args(argv)


def test_devices_and_extra_forecast_arguments_reach_the_forecast_door(tmp_path):
    value = global_door.plan(_args(tmp_path, "--devices", "0,1", "--forecast-arg=--checkpoint-hours=6"))
    forecast = value["commands"]["forecast"]
    assert forecast[forecast.index("--samehost-devices") + 1] == "0,1"
    assert "--checkpoint-hours=6" in forecast
    assert {k: v for k, v in value["cards"].items() if k != "lease"} == {"selected_cards": 2, "basis": "--devices names the cards"}
    assert value["cards"]["lease"]["vram_mib_per_card"] is None  # no grid in the test cache


def test_the_model_label_and_brand_reach_the_product_stage(tmp_path, monkeypatch):
    # WOOF sounding labels (2026-10-23): the door forwards what the host calls the model to its soundings.
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    brand = "WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z"
    value = global_door.plan(_args(tmp_path, "--model-label", "WOOF Hex", "--brand", brand))
    products = value["commands"]["products"]
    assert products[products.index("--model-label") + 1] == "WOOF Hex"
    assert products[products.index("--brand") + 1] == brand
    build_parser().parse_args(products)
    # unset, nothing is forwarded and the products door keeps its own default
    bare = global_door.plan(_args(tmp_path))["commands"]["products"]
    assert "--model-label" not in bare and "--brand" not in bare


def test_the_first_look_and_the_full_pass_file_into_one_case(tmp_path, monkeypatch):
    # Parity run group 4: on the MPAS-physics route the products door named the case from the grid file
    # (x1.40962-<cycle>) while the first look used the runtime mesh name, so every live hour was drawn twice.
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    value = global_door.plan(_args(tmp_path))
    products = value["commands"]["products"]
    assert value["case"] == f"{global_door.runtime_mesh_name(value['mesh']['id'])}-2026101112"
    assert products[products.index("--case") + 1] == value["case"]
    build_parser().parse_args(products)


def test_a_wall_estimate_needs_the_same_card(monkeypatch):
    table = global_door.load_table()
    on_5070 = global_door.wall_estimate(table, "x1.40962", 1, 24.0, card_names=["NVIDIA GeForce RTX 5070 Ti"])
    assert on_5070 is not None and on_5070["seconds"] == pytest.approx(5.5 * 24)
    # a rate measured on a 96 GB workstation card is not an estimate for a 16 GB one
    assert global_door.wall_estimate(table, "x1.163842", 1, 24.0, card_names=["NVIDIA GeForce RTX 5070 Ti"]) is None
    assert global_door.wall_estimate(table, "x1.163842", 1, 24.0,
                                     card_names=["NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]) is not None
    assert global_door.wall_estimate(table, "x1.40962", 1, 24.0, card_names=[]) is None


def test_run_plan_prints_json_and_runs_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    rc = main(["run-plan", "--row", "global-120km", "--cycle", "2026-10-11T12",
               "--work-dir", str(tmp_path / "run"), "--cache-dir", str(tmp_path / "cache")])
    assert rc == 0
    value = json.loads(capsys.readouterr().out)
    assert value["row"]["id"] == "global-120km" and value["mesh"]["cells"] == 40962
    assert not (tmp_path / "run").exists()


def test_list_and_sources_speak_json(capsys):
    assert main(["global", "--list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {row["id"] for row in listed["rows"]} >= {"global-15km", "global-120km-morrison"}
    assert main(["sources", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema"] == "gpuwm-hex.sources/v1"
    assert "gfs" in {row["id"] for row in value["init_sources"]}
    assert "global-30km" in {row["id"] for row in value["global_rows"]}


def test_a_bad_cycle_and_an_unknown_row_are_refused_by_name(tmp_path):
    with pytest.raises(global_door.GlobalDoorError, match="YYYY-MM-DDTHH"):
        global_door.plan(build_parser().parse_args(
            ["global", "--row", "global-120km", "--cycle", "2026-10-11", "--work-dir", str(tmp_path)]))
    with pytest.raises(global_door.GlobalDoorError, match="no global row"):
        global_door.plan(build_parser().parse_args(
            ["global", "--row", "global-7km", "--cycle", "2026-10-11T12", "--work-dir", str(tmp_path)]))


# ---------------------------------------------------------------------------
# the mesh stage, against a local archive
# ---------------------------------------------------------------------------
def _local_table(tmp_path, grid_bytes: bytes, *, archive_pin: str | None = None,
                 grid_pin: str | None = None) -> Path:
    archive = tmp_path / "x1.42.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("x1.42.grid.nc")
        info.size = len(grid_bytes)
        tar.addfile(info, io.BytesIO(grid_bytes))
    table = json.loads(global_door.TABLE.read_text(encoding="utf-8"))
    table["meshes"] = {"x1.42": {
        "url": archive.as_uri(), "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_pin or hashlib.sha256(archive.read_bytes()).hexdigest(),
        "grid_member": "x1.42.grid.nc", "grid_sha256": grid_pin or hashlib.sha256(grid_bytes).hexdigest(),
        "grid_bytes": len(grid_bytes), "cells": 42, "nominal_dx_m": 120000.0}}
    table["rows"] = [{"id": "tiny", "mesh": "x1.42", "physics": "arwen", "hours": 1}]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def _mesh_plan(tmp_path, table):
    args = build_parser().parse_args(["global", "--row", "tiny", "--cycle", "2026-10-11T12",
                                      "--work-dir", str(tmp_path / "run"), "--cache-dir",
                                      str(tmp_path / "cache"), "--table", str(table)])
    return global_door.plan(args)


def test_the_mesh_stage_fetches_verifies_and_then_skips_out_loud(tmp_path):
    table = _local_table(tmp_path, b"not really a grid")
    value = _mesh_plan(tmp_path, table)
    status = run_status.StatusWriter(tmp_path / "run", "global", stages=("mesh",))
    first = global_door.stage_mesh(value, status)
    grid = Path(value["paths"]["grid"])
    assert first == {"grid": str(grid), "cached": False}
    assert grid.read_bytes() == b"not really a grid"
    receipt = json.loads((grid.parent / "fetch-receipt.json").read_text())
    assert receipt["grid_sha256"] == hashlib.sha256(b"not really a grid").hexdigest()
    assert not any(p.name.endswith((".part", ".tar.gz")) for p in grid.parent.iterdir())
    second = global_door.stage_mesh(value, status)
    assert second["cached"] is True
    skips = [e for e in run_status.read_events(tmp_path / "run") if e["event"] == "stage-skip"]
    assert skips and "pinned sha256" in skips[0]["reason"]


def test_an_archive_that_is_not_the_pinned_one_is_removed_and_refused(tmp_path):
    table = _local_table(tmp_path, b"grid", archive_pin="0" * 64)
    value = _mesh_plan(tmp_path, table)
    status = run_status.StatusWriter(tmp_path / "run", "global", stages=("mesh",))
    with pytest.raises(global_door.GlobalDoorError, match="not the published mesh"):
        global_door.stage_mesh(value, status)
    assert not list((tmp_path / "cache").rglob("*.tar.gz"))


def test_a_cut_off_download_is_resumed_not_refused_as_a_wrong_archive(tmp_path, monkeypatch):
    # The measured failure: the link dropped part way (Content-Length set, body short), http.client returned an
    # empty block, and the short file was hashed and refused as "not the published mesh".  Here every
    # answer is cut after 4,000 bytes; the door must resume with Range requests and stage the grid.
    import http.server
    import socket
    import threading

    grid_bytes = os.urandom(20_000)  # incompressible, so the archive spans several cuts
    table = _local_table(tmp_path, grid_bytes)
    body = (tmp_path / "x1.42.tar.gz").read_bytes()
    ranges = []

    class Cut(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            first = int((self.headers.get("Range") or "bytes=0-")[6:].split("-")[0])
            ranges.append(first)
            self.send_response(206 if first else 200)
            if first:
                self.send_header("Content-Range", f"bytes {first}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(body) - first))
            self.end_headers()
            self.wfile.write(body[first:first + 4000])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Cut)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(global_door.time, "sleep", lambda seconds: None)
    try:
        value = _mesh_plan(tmp_path, table)
        value["mesh"]["url"] = f"http://127.0.0.1:{server.server_port}/x1.42.tar.gz"
        status = run_status.StatusWriter(tmp_path / "run", "global", stages=("mesh",))
        result = global_door.stage_mesh(value, status)
    finally:
        server.shutdown()
    assert Path(result["grid"]).read_bytes() == grid_bytes
    assert len(ranges) > 1 and ranges == sorted(ranges) and ranges[1] == 4000


def test_a_grid_that_is_not_the_pinned_one_is_removed_and_refused(tmp_path):
    table = _local_table(tmp_path, b"grid", grid_pin="1" * 64)
    value = _mesh_plan(tmp_path, table)
    status = run_status.StatusWriter(tmp_path / "run", "global", stages=("mesh",))
    with pytest.raises(global_door.GlobalDoorError, match="grid was removed"):
        global_door.stage_mesh(value, status)
    assert not Path(value["paths"]["grid"]).exists()


def test_an_unpinned_mesh_is_never_fetched(tmp_path):
    table = _local_table(tmp_path, b"grid", archive_pin="PENDING")
    value = _mesh_plan(tmp_path, table)
    status = run_status.StatusWriter(tmp_path / "run", "global", stages=("mesh",))
    with pytest.raises(global_door.GlobalDoorError, match="no pinned archive_sha256"):
        global_door.stage_mesh(value, status)


def test_a_static_builder_without_the_native_path_is_refused_by_name(tmp_path, monkeypatch):
    table = _local_table(tmp_path, b"grid")
    value = _mesh_plan(tmp_path, table)
    old = tmp_path / "rw_mpas_static"
    old.write_text("#!/bin/sh\necho 'usage: rw_mpas_static --grid G --out S'\n")
    old.chmod(old.stat().st_mode | stat.S_IEXEC)
    from woof.hex import engines

    monkeypatch.setattr(engines, "resolve", lambda spec, *a, **k: old)
    status = run_status.StatusWriter(tmp_path / "run", "global", stages=("static",))
    if os.name == "nt":
        pytest.skip("shell stand-in")
    with pytest.raises(global_door.GlobalDoorError, match="--native-init"):
        global_door.stage_static(value, status)


def test_the_static_refusal_reaches_the_reader_whole(tmp_path, monkeypatch):
    """Hex global run, 2026-10-16: the door kept only the refusal's last line
    ("--geog DIR pointing at a complete archive") and dropped the datasets and
    the fetch command above it."""
    table = _local_table(tmp_path, b"grid")
    value = _mesh_plan(tmp_path, table)
    Path(value["paths"]["static"]).parent.mkdir(parents=True, exist_ok=True)
    engine = tmp_path / "rw_mpas_static"
    engine.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --help ]; then echo '--native-init --native-gwd-static'; exit 0; fi\n"
        "echo 'the geography archive is missing 2 dataset(s) a static needs: "
        "/g/modis_landuse_20class_30s, /g/landuse_30s. Terrain ...' >&2\n"
        "echo 'what does work:' >&2\n"
        "echo '  woof fetch-geog --datasets mesh, which stages every dataset this builder opens' >&2\n"
        "echo '  --geog DIR pointing at a complete archive' >&2\n"
        "exit 1\n")
    engine.chmod(engine.stat().st_mode | stat.S_IEXEC)
    from woof.hex import engines

    monkeypatch.setattr(engines, "resolve", lambda spec, *a, **k: engine)
    status = run_status.StatusWriter(tmp_path / "run", "global", stages=("static",))
    if os.name == "nt":
        pytest.skip("shell stand-in")
    with pytest.raises(global_door.GlobalDoorError) as refused:
        global_door.stage_static(value, status)
    said = str(refused.value)
    for needed in ("modis_landuse_20class_30s", "landuse_30s",
                   "woof fetch-geog --datasets mesh", "--geog DIR"):
        assert needed in said, said


def test_a_refused_stage_writes_refused_status_and_a_receipt(tmp_path, monkeypatch):
    table = _local_table(tmp_path, b"grid", archive_pin="PENDING")
    args = build_parser().parse_args(
        ["global", "--row", "tiny", "--cycle", "2026-10-11T12", "--work-dir", str(tmp_path / "run"),
         "--cache-dir", str(tmp_path / "cache"), "--table", str(table), "--stage", "mesh"])
    with pytest.raises(global_door.GlobalDoorError):
        global_door.run_global(args)
    # the CLI turns the same refusal into exit code 2 and the same status
    assert main(["global", "--row", "tiny", "--cycle", "2026-10-11T12", "--work-dir", str(tmp_path / "run"),
                 "--cache-dir", str(tmp_path / "cache"), "--table", str(table), "--stage", "mesh"]) == 2
    doc = run_status.read_status(tmp_path / "run")
    assert doc["state"] == "refused" and "pinned" in doc["reason"]
    receipt = json.loads((tmp_path / "run" / "global-receipt.json").read_text())
    assert "refused" in receipt
    assert json.loads((tmp_path / "run" / "global-plan.json").read_text())["row"]["id"] == "tiny"


# ---------------------------------------------------------------------------
# the export door
# ---------------------------------------------------------------------------
STAND_IN = r'''
import json, sys
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--out-dir") + 1]); out.mkdir(parents=True, exist_ok=True)
report = Path(argv[argv.index("--json") + 1])
frames = []
for i, history in enumerate(argv[argv.index("--history") + 1: argv.index("--mesh")]):
    name = f"wrfout_d01_2026-10-11_{12 + i:02d}_00_00"
    (out / name).write_bytes(b"CDF")
    frames.append({"output": str(out / name), "output_sha256": "ab" * 32, "valid_time": f"2026-10-11T{12 + i:02d}:00:00"})
report.write_text(json.dumps({"frames": frames, "argv": argv}))
'''


def test_export_files_lat_lon_frames_by_day_with_the_converter_receipt(tmp_path):
    if os.name == "nt":
        pytest.skip("shell stand-in")
    run = tmp_path / "forecast"
    run.mkdir()
    for hour in (12, 13):
        (run / f"cuda-history.2026-10-11_{hour}.00.00.nc").write_bytes(b"h")
    grid = tmp_path / "x1.grid.nc"
    grid.write_bytes(b"g")
    init = tmp_path / "init.nc"
    init.write_bytes(b"i")
    script = tmp_path / "convert.py"
    script.write_text(STAND_IN)
    exe = tmp_path / "rw_mpas_convert"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    rc = main(["export", str(run), "--mesh", str(grid), "--init", str(init), "--convert-exe", str(exe),
               "--out", str(tmp_path / "out"), "--simulation-start", "2026-10-11_12:00:00"])
    assert rc == 0
    frames = sorted((tmp_path / "out" / "global" / "2026-10-11").iterdir())
    assert [f.name for f in frames] == ["wrfout_d01_2026-10-11_12_00_00", "wrfout_d01_2026-10-11_13_00_00"]
    receipt = json.loads((tmp_path / "out" / "export-receipt.json").read_text())
    assert receipt["schema"] == "gpuwm-hex.export/v1" and receipt["window"] == "global-mesh"
    assert receipt["remap"] == "product" and len(receipt["frames"]) == 2
    argv = receipt["converter"]["report"]["argv"]
    assert argv[argv.index("--init") + 1] == str(init) and argv[argv.index("--remap") + 1] == "product"
    assert not (tmp_path / "out.export-scratch").exists()


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------
def _plan_for(tmp_path, row, *extra):
    parser = build_parser()
    return global_door.plan(parser.parse_args(
        ["global", "--row", row, "--cycle", "2026-10-11T12", "--work-dir", str(tmp_path / row),
         "--cache-dir", str(tmp_path / "cache"), "--stage", "plan", *extra]))


def _arwen(row):
    """The WOOF arm of a default row: MPAS physics is the hex default since 0.4.6, so the YSU cap tests name it."""
    return [] if row.endswith("-morrison") else ["--physics-row", "arwen"]


def _dt_args(forecast):
    return [forecast[i + 1] for i, token in enumerate(forecast) if token == "--dt"]


#: woof 2.9.0's YSU and New Tiedtke, which lack the donor bounds, and the engine identities that carry them.
UNBOUNDED_YSU = "ysu-wrf-v4.8.0-topdown-cap-noftz-v1"
BOUNDED_YSU = "ysu-wrf-v4.8.0-entrainment-courant-v2"
UNBOUNDED_NT = "new-tiedtke-wrf461-cumastrn-v1"
BOUNDED_NT = "new-tiedtke-wrf461-cumastrn-donor-bound-v2"
ENGINE_290 = frozenset({UNBOUNDED_YSU, UNBOUNDED_NT, "grell-freitas-wrf461-gfdrv-corrected-k22-v1"})
ENGINE_BOUNDED = frozenset({BOUNDED_YSU, BOUNDED_NT, "grell-freitas-wrf461-gfdrv-corrected-k22-v1"})


def test_the_ysu_arms_cap_the_step_where_long_steps_make_hot_layers(tmp_path, monkeypatch):
    # 600 s at 120 km and 360 s at 60 km made single-layer hot spots (worst +24 and +23 K); 300 s did not.
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    monkeypatch.setattr(global_door, "engine_identities", lambda: ENGINE_290)
    table = global_door.load_table()
    for arm in ("arwen", "morrison"):
        assert table["physics"][arm]["max_dt_seconds"] == 300 and "hot" in table["physics"][arm]["max_dt_why"]
    for row, mesh_dt in (("global-120km", 600.0), ("global-60km", 360.0), ("global-60km-morrison", 360.0)):
        value = _plan_for(tmp_path, row, *_arwen(row))
        assert value["step"] == {**value["step"], "dt_seconds": 300.0, "capped": True, "mesh_dt_seconds": mesh_dt}
        assert _dt_args(value["commands"]["forecast"]) == ["300"]
        assert build_parser().parse_args(value["commands"]["forecast"]).dt == "300"
    for row, mesh_dt in (("global-30km", 180.0), ("global-15km", 90.0)):
        value = _plan_for(tmp_path, row, *_arwen(row))
        assert value["step"]["capped"] is False and value["step"]["dt_seconds"] == mesh_dt
        assert _dt_args(value["commands"]["forecast"]) == []  # the forecast door's own default


def test_a_callers_own_step_still_wins_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    monkeypatch.setattr(global_door, "engine_identities", lambda: ENGINE_290)
    forecast = _plan_for(tmp_path, "global-120km", *_arwen("global-120km"), "--forecast-arg=--dt=600")["commands"]["forecast"]
    assert _dt_args(forecast) == ["300"] and forecast[-1] == "--dt=600"
    assert build_parser().parse_args(forecast).dt == "600"


def test_an_engine_with_both_donor_bounds_retires_the_cap(tmp_path, monkeypatch):
    # The cap's defect is two explicit tendencies overshooting their donor layers: YSU's PBL-top
    # entrainment and New Tiedtke's rates.  An engine carrying both bounds ran the 120 km globe at
    # 600 s for 24 h with no hot layer below level 30, so the rows run the mesh's own step there,
    # and the plan says which cap was retired and by what.
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    monkeypatch.setattr(global_door, "engine_identities", lambda: ENGINE_BOUNDED)
    table = global_door.load_table()
    assert table["physics"]["arwen"]["max_dt_retired_by_engine"] == [BOUNDED_YSU, BOUNDED_NT]
    for row, mesh_dt in (("global-120km", 600.0), ("global-60km", 360.0)):
        value = _plan_for(tmp_path, row, *_arwen(row))
        assert value["step"]["capped"] is False and value["step"]["dt_seconds"] == mesh_dt
        assert value["step"]["cap_retired"]["cap_seconds"] == 300.0
        assert set(value["step"]["cap_retired"]["engine_identities"]) <= ENGINE_BOUNDED
        assert _dt_args(value["commands"]["forecast"]) == []  # the forecast door's own default


def test_one_bound_alone_keeps_the_cap(tmp_path, monkeypatch):
    # With the YSU bound alone the same run kept one +7.9 K hot layer (New Tiedtke, Tibet, +18 h).
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    monkeypatch.setattr(global_door, "engine_identities", lambda: frozenset({BOUNDED_YSU, UNBOUNDED_NT}))
    value = _plan_for(tmp_path, "global-120km", *_arwen("global-120km"))
    assert value["step"]["capped"] is True and value["step"]["dt_seconds"] == 300.0
    assert value["step"]["engine_lacks"] == [BOUNDED_NT]


def test_the_real_engine_identities_are_read():
    # Published woof 2.9.0 carries woof.ysu_generations; an older engine tree (the landing pool's
    # pinned engine checkout) does not, and the door then keeps the cap, which
    # test_an_engine_that_names_nothing_keeps_the_cap covers.
    pytest.importorskip("woof.ysu_generations", reason="engine predates the YSU identity table")
    found = global_door.engine_identities()
    assert any(value.startswith("ysu-") for value in found)
    assert any(value.startswith("new-tiedtke-") for value in found)


def test_an_engine_that_names_nothing_keeps_the_cap():
    mesh = {"nominal_dx_m": 120000.0}
    arm = {"max_dt_seconds": 300, "max_dt_retired_by_engine": [BOUNDED_YSU, BOUNDED_NT]}
    assert global_door.row_step(mesh, arm)["capped"] is True
    assert global_door.row_step(mesh, arm, engine=ENGINE_290)["dt_seconds"] == 300.0
    assert global_door.row_step(mesh, arm, engine=ENGINE_BOUNDED)["dt_seconds"] == 600.0


def test_a_cap_lands_on_a_step_that_divides_the_radiation_interval():
    mesh = {"nominal_dx_m": 120000.0}
    assert global_door.row_step(mesh, {"max_dt_seconds": 280})["dt_seconds"] == 225.0
    assert global_door.row_step(mesh, {})["capped"] is False


def test_the_score_config_carries_its_own_provenance(tmp_path, monkeypatch):
    # A forecast stopped before its receipt still leaves frames; the scorer must not need the receipt.
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    value = _plan_for(tmp_path, "global-120km")
    config = value["score_config"]
    assert config["init"] == value["paths"]["init"] and config["grid"] == value["paths"]["grid"]
    assert config["mesh"] == "hex-x1.40962-native" and config["start"] == "2026-10-11_12:00:00"
    assert config["region"]["west"] == -180.0


def test_the_forecast_doors_default_step_takes_the_ysu_cap_until_the_engine_carries_both_bounds():
    # A bare woof hex forecast on the 120 km globe took the mesh's own 600 s step with YSU and showed
    # the single hot layers the global rows cap away (hex 0.4.0 recut, 2026-10-16).
    from woof.hex.global_door import default_step_cap, load_table

    table = load_table()
    arwen = table["physics"]["arwen"]
    mesh = {"dt_seconds": 600.0, "source": "mesh"}
    ysu = {"config_pbl_scheme": "bl_ysu", "config_convection_scheme": "auto"}
    capped = default_step_cap(mesh, ysu, radiation_seconds=1800.0, engine=frozenset(), table=table)
    assert capped["dt_seconds"] == 300.0 and capped["source"] == "physics cap"
    assert capped["mesh_dt_seconds"] == 600.0 and capped["arm"] == "arwen"
    assert capped["engine_lacks"] == sorted(arwen["max_dt_retired_by_engine"])
    fixed = frozenset(arwen["max_dt_retired_by_engine"])
    retired = default_step_cap(mesh, ysu, radiation_seconds=1800.0, engine=fixed, table=table)
    assert retired["dt_seconds"] == 600.0 and retired["cap_retired"]["arm"] == "arwen"
    # Grell-Freitas with YSU is the morrison arm, whose cap no engine bound retires.
    gf = {"config_pbl_scheme": "bl_ysu", "config_convection_scheme": "cu_grell_freitas"}
    assert default_step_cap(mesh, gf, radiation_seconds=1800.0, engine=fixed, table=table)["dt_seconds"] == 300.0
    # A step at or under the cap, another boundary layer, or a step the caller chose is left alone.
    assert default_step_cap({"dt_seconds": 180.0, "source": "mesh"}, ysu, radiation_seconds=1800.0,
                            engine=frozenset(), table=table)["dt_seconds"] == 180.0
    assert default_step_cap(mesh, {"config_pbl_scheme": "bl_mynn"}, radiation_seconds=1800.0,
                            engine=frozenset(), table=table)["dt_seconds"] == 600.0
    assert default_step_cap({"dt_seconds": 600.0, "source": "caller"}, ysu, radiation_seconds=1800.0,
                            engine=frozenset(), table=table)["dt_seconds"] == 600.0


def test_a_bare_forecast_with_no_suite_reads_the_engine_seams_ysu():
    # The first version of the forecast door's cap read physics.selectors, and a bare forecast names no
    # suite (physics is None), so the cap never bound: the x1.40962 smoke took 600 s (hex 0.4.0 recut).
    from argparse import Namespace
    from woof.hex.forecast_door import default_step_selectors
    from woof.hex.global_door import default_step_cap, load_table

    bare = default_step_selectors(None, Namespace(convection="auto"))
    assert bare["config_pbl_scheme"] == "bl_ysu"
    capped = default_step_cap({"dt_seconds": 600.0, "source": "mesh"}, bare, radiation_seconds=1800.0,
                              engine=frozenset(), table=load_table())
    assert capped["dt_seconds"] == 300.0 and capped["arm"] == "arwen"
    assert default_step_selectors(None, Namespace(convection="gf"))["config_convection_scheme"] == "cu_grell_freitas"


def test_a_short_run_takes_a_cadence_that_divides_it(tmp_path, monkeypatch):
    # Unified-image check 2026-10-16: --hours 1 on global-60km kept the row's 180 min and was refused rc 2.
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    row = ("--row", "global-15km-morrison")  # a 3-hourly row: the cheap rows now store hourly
    value = global_door.plan(_args(tmp_path, *row, "--hours", "1"))
    assert value["history_every_minutes"] == 60
    assert value["history_cadence"]["row_default"] == 180
    assert "divides the 1 h run" in value["history_cadence"]["source"]
    forecast = value["commands"]["forecast"]
    assert forecast[forecast.index("--history-every-minutes") + 1] == "60"
    assert global_door.plan(_args(tmp_path, *row))["history_every_minutes"] == 180  # 24 h keeps the row default
    # The default product set's hourly windows (qpf_1h, run-max wind, daily temperature extremes) need
    # hourly frames, which the 120/60/30 km rows store by default.
    assert global_door.plan(_args(tmp_path))["history_every_minutes"] == 60
    assert global_door.history_cadence(0.5, None, 180, {300.0, 360.0}, "r")["minutes"] == 30
    assert global_door.history_cadence(1.75, None, 180, {300.0}, "r")["minutes"] == 15


def test_an_explicit_cadence_that_does_not_divide_the_run_is_refused_before_any_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(global_door, "_visible_cards", lambda: [])
    with pytest.raises(global_door.GlobalDoorError, match="does not divide the 1 h run"):
        global_door.plan(_args(tmp_path, "--hours", "1", "--history-every-minutes", "180"))
    value = global_door.plan(_args(tmp_path, "--hours", "1", "--history-every-minutes", "30"))
    assert value["history_cadence"] == {"minutes": 30, "source": "--history-every-minutes", "row_default": 60}


@pytest.mark.parametrize('accepted', [True, False])
def test_global_score_completion_preserves_partial_coverage_and_real_failures(tmp_path, monkeypatch, accepted):
    from woof.hex import observation_scorecard
    monkeypatch.setattr(global_door, '_visible_cards', lambda: [])
    monkeypatch.setattr(observation_scorecard, 'join_automatic_scorer', lambda *a: None)
    called = []
    def run(argv, env):
        called.append(argv)
        return 3
    monkeypatch.setattr(global_door, '_run_door', run)
    def completed(*args):
        if not called or not accepted:
            return None
        return {'returncode': 0, 'status': 'PARTIALLY_MEASURED', 'measured_metrics': 299,
                'note': 'partial observation coverage'}
    monkeypatch.setattr(observation_scorecard, 'completed_scorecard', completed)
    args = _args(tmp_path, '--stage', 'score')
    assert global_door.run_global(args) == (0 if accepted else 3)
    receipt = json.loads((Path(args.work_dir) / 'global-receipt.json').read_text())
    result = receipt['stages']['score']
    assert result['returncode'] == (0 if accepted else 3)
    if accepted:
        assert result['status'] == 'PARTIALLY_MEASURED' and result['scorer_returncode'] == 3

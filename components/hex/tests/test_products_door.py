"""The product door over stand-in engines: views, layout, standard-set
expansion, the remap it asks for, and the points stage."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from threading import Event

import pytest

from woof.hex import products_door, render_door
from woof.hex.errors import MpasPortError


class TestRemapResolution:
    def test_auto_is_product_for_one_run_and_nearest_for_a_composite(self):
        assert render_door.resolve_remap("auto", composite=False) == "product"
        assert render_door.resolve_remap("auto", composite=True) == "nearest"
        assert render_door.resolve_remap("nearest", composite=False) == "nearest"

    def test_product_with_a_composite_is_refused_by_name(self):
        with pytest.raises(MpasPortError, match="no single mesh"):
            render_door.resolve_remap("product", composite=True)

    def test_unknown_remap_is_refused(self):
        with pytest.raises(MpasPortError):
            render_door.resolve_remap("bilinear", composite=False)

    @pytest.mark.parametrize("remap,expect", [("product", True), ("nearest", False)])
    def test_the_converter_is_asked_for_the_remap(self, monkeypatch, tmp_path, remap, expect):
        seen = {}

        class Result:
            returncode = 0
            stdout = "REMAP\tproduct\tabc\nBUDGET\tt\tRAINC+RAINNC\trel=1e-5\n"
            stderr = ""

        def fake_run(command, *, log_prefix, env=None):
            seen["command"] = [str(c) for c in command]
            Path(command[command.index("--json") + 1]).write_text("{}", encoding="utf-8")
            return Result()

        monkeypatch.setattr(render_door, "_run", fake_run)
        render_door.convert_frames(
            Path("rw_mpas_convert"), histories=[tmp_path / "h.nc"], mesh=tmp_path / "m.nc",
            scratch=tmp_path / "s", window="global-mesh", field_set="full",
            simulation_start=None, nc_format="cdf2", remap=remap)
        command = seen["command"]
        assert ("--remap" in command) is expect
        if expect:
            assert command[command.index("--remap") + 1] == "product"


class TestHelpers:
    def test_case_token_comes_from_the_data(self):
        token = products_door.case_token(None, Path("/m/x1.655362.ordered.grid.nc"), "2026-10-03_12:00:00")
        assert token == "x1.655362-2026100312"
        assert products_door.case_token("my run", Path("m.nc"), "2026-10-03_12:00:00") == "my-run"
        # a grid stored under a role name takes the forecast's mesh name, else its folder's
        nameless = Path("/meshes/global30/ordered.grid.nc")
        assert products_door.case_token(None, nameless, "2026-10-03_12:00:00") == "global30-2026100312"
        assert products_door.case_token(None, nameless, "2026-10-03_12:00:00",
                                        "x1.655362-ordered") == "x1.655362-ordered-2026100312"

    def test_window_centre_reads_both_window_kinds(self):
        lam = {"convert": {"window_spec": {"kind": "lambert", "centre_lat": 37.5, "centre_lon": -97.5,
                                           "dx_metres": 3000.0, "nx": 300, "ny": 200}}}
        assert products_door.window_centre(lam) == (37.5, -97.5, 300.0)
        ll = {"convert": {"window_spec": json.dumps({"kind": "latlon", "south": -10.0, "north": 10.0,
                                                    "west": 20.0, "east": 40.0, "spacing_degrees": 0.1})}}
        lat, lon, half = products_door.window_centre(ll)
        assert (lat, lon) == (0.0, 30.0) and half == pytest.approx(10 * 111.195)
        assert products_door.window_centre({}) is None

    def test_histories_are_discovered_in_time_order_or_refused(self, tmp_path):
        with pytest.raises(MpasPortError, match="no cuda-history"):
            products_door.discover_histories(tmp_path)
        for stamp in ("2026-10-03_18.00.00", "2026-10-03_12.00.00"):
            (tmp_path / f"cuda-history.{stamp}.nc").write_bytes(b"h")
        found = products_door.discover_histories(tmp_path)
        assert [p.name for p in found] == ["cuda-history.2026-10-03_12.00.00.nc",
                                           "cuda-history.2026-10-03_18.00.00.nc"]
        assert products_door.simulation_start(None, None, found) == "2026-10-03_12:00:00"

    def test_the_forecast_door_hands_its_own_files_to_the_product_door(self, tmp_path):
        from woof.hex import forecast_door

        request = argparse.Namespace(out=tmp_path / "fc", grid=tmp_path / "g.nc",
                                     init=tmp_path / "i.nc", start_time="2026-10-03_12:00:00")
        command = forecast_door.products_command(request, [tmp_path / "fc" / "h.nc"])
        assert command[:2] == ["woof hex", "products"]
        assert command[command.index("--forecast") + 1] == str(tmp_path / "fc")
        assert command[command.index("--init") + 1] == str(tmp_path / "i.nc")
        assert command[command.index("--out") + 1] == str(tmp_path / "fc" / "products")
        parser = argparse.ArgumentParser()
        products_door.add_products_arguments(parser)
        parsed = parser.parse_args(command[2:])
        assert parsed.remap == "auto" and not parsed.no_points

    def test_the_forecast_door_offers_the_opt_out(self):
        from woof.hex import forecast_door

        parser = argparse.ArgumentParser()
        forecast_door.add_forecast_arguments(parser)
        assert "--no-products" in parser.format_help()


def _stand_in_engines(monkeypatch, tmp_path, *, uniform: bool, seen: list):
    """Converter, catalog and renderer stand-ins through render_door._run."""

    basemap = tmp_path / "basemap"
    (basemap / "natural_earth_110m").mkdir(parents=True, exist_ok=True)
    (basemap / "natural_earth_110m" / "ne_110m_coastline.shp").write_bytes(b"s")
    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(basemap))
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda explicit=None: Path("rw_mpas_convert"))
    monkeypatch.setattr(render_door, "resolve_renderer_exe", lambda explicit=None: Path("rw_wrfbatch"))
    monkeypatch.setattr(render_door, "sha256_file", lambda path: "0" * 64)
    # the stand-in engines are current builds: they take the frame-parallel flags
    monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)

    class Result:
        def __init__(self):
            self.returncode, self.stdout, self.stderr = 0, "", ""

    def fake_run(command, *, log_prefix, env=None):
        command = [str(c) for c in command]
        seen.append((log_prefix, command))
        result = Result()
        if log_prefix == "CONVERT":
            window = command[command.index("--window") + 1]
            if window == "mesh" and uniform:
                result.returncode = 1
                result.stderr = ("rw_mpas_convert: this mesh is uniform to within the refinement "
                                 "factor -- so there is no refined region to derive a window from")
                return result
            frames = Path(command[command.index("--out-dir") + 1])
            outs = []
            histories = command[command.index("--history") + 1:command.index("--mesh")]
            for k, history in enumerate(histories):
                # The frame is named by its history's own valid time, as the converter names it, so a frame
                # converted in another call (a live draw's, reused from the run's frame cache) is the same frame.
                stamp = re.search(r"_(\d{2})\.00\.00", Path(history).name)
                hour = int(stamp.group(1)) if stamp else 12 + 6 * k
                wrfout = frames / f"wrfout_d01_2026-10-03_{hour:02d}_00_00"
                wrfout.write_bytes(b"w")
                outs.append({"output": str(wrfout), "valid_time": f"2026-10-03T{hour:02d}:00:00",
                             "lead_seconds": 3600 * (hour - 12), "absent_wrf_fields": [],
                             "precipitation_budget": {"relative_difference": 1e-5}})
            spec = ({"kind": "lambert", "centre_lat": 37.5, "centre_lon": -97.5, "dx_metres": 3000.0,
                     "nx": 300, "ny": 300} if window == "mesh" else {"kind": "latlon"})
            Path(command[command.index("--json") + 1]).write_text(json.dumps({
                "grid_label": "hex 48-3 km", "window_spec": spec,
                "product_remap": {"sha256": "r"}, "frames": outs}), encoding="utf-8")
        elif log_prefix == "CATALOG":
            result.stdout = ("PRODUCT\t2m_temperature_10m_winds\tdirect\trenderable\tready\t\n"
                             "PRODUCT\ttotal_qpf\tdirect\trenderable\tready\t\n"
                             "PRODUCT\tprecipitation_type\tdirect\tmissing-fields\tnot stored: categorical_rain_surface\t\n"
                             "PRODUCT\tqpf_6h\twindowed\trenderable\t6 h APCP differenced (F006 minus F000)\trenderable\n"
                             "PRODUCT\tqpf_24h\twindowed\tblocked\t24-h QPF requires forecast hour >= 24\twindowed-blocked\n")
        elif log_prefix == "RENDER":
            raw = Path(command[command.index("--out-dir") + 1])
            drawn = command[command.index("--products") + 1].split(",")
            lines = []
            # Every frame of the series arrives in one start, in one store.
            for wrfout in [c for c in command if Path(c).name.startswith("wrfout_d01_")]:
                lead = int(Path(wrfout).name.split("_")[3]) - 12
                for slug in drawn:
                    if slug == "qpf_6h" and lead < 6:
                        lines.append(f"SKIPPED {slug} {wrfout}: F{lead:03d}: 6-h QPF requires forecast hour >= 6")
                        continue
                    png = raw / f"rustwx_wrf_20261003_12z_f{lead:03d}_hex48-3km_{slug}.png"
                    png.write_bytes(b"png")
                    lines.append(f"RENDERED {slug} {png}")
            result.stdout = "\n".join(lines) + "\n"
        return result

    monkeypatch.setattr(render_door, "_run", fake_run)


def _points_stand_in(monkeypatch, seen: list):
    # Stand-in history bytes do not carry real NetCDF headers.
    monkeypatch.setattr(products_door, "histories_without_surface_pressure", lambda histories: [])
    class Done:
        returncode, stderr = 0, ""
        stdout = "POINT\tx\nFINISHED\tpoints=1\n"

    def fake_subprocess_run(command, **kwargs):
        command = [str(c) for c in command]
        seen.append(("POINTS", command))
        out = Path(command[command.index("--out-dir") + 1])
        names = [item.split("@")[0] for item in command[command.index("--points") + 1].split(";")]
        slug = names[0].lower().replace(" ", "-")
        (out / slug).mkdir(parents=True, exist_ok=True)
        (out / slug / f"sounding_{slug}_20261003_1800_f006.png").write_bytes(b"s")
        (out / slug / f"meteogram_{slug}_20261003_12.png").write_bytes(b"m")
        Path(command[command.index("--json") + 1]).write_text('{"soundings": []}', encoding="utf-8")
        return Done()

    monkeypatch.setattr(products_door.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(products_door, "resolve_points_exe", lambda explicit, convert: Path("rw_hex_points"))
    # the stand-in rw_hex_points is a current build: it draws --brand
    monkeypatch.setattr(products_door, "points_take_brand", lambda exe: True)


def _arguments(tmp_path, histories, **over):
    parser = argparse.ArgumentParser()
    products_door.add_products_arguments(parser)
    init = tmp_path / "init.nc"
    init.write_bytes(b"i")
    argv = ["--history", *[str(h) for h in histories], "--mesh", str(tmp_path / "patch.grid.nc"),
            "--init", str(init), "--out", str(tmp_path / "out")]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return parser.parse_args(argv)


def _histories(tmp_path):
    (tmp_path / "patch.grid.nc").write_bytes(b"m")
    out = []
    for stamp in ("2026-10-03_12.00.00", "2026-10-03_18.00.00"):
        path = tmp_path / f"cuda-history.{stamp}.nc"
        path.write_bytes(b"h")
        out.append(path)
    return out


@pytest.mark.parametrize("context_before_global", [False, True])
def test_a_refined_mesh_gets_global_core_context_and_points_in_the_ruled_layout(
        monkeypatch, tmp_path, context_before_global):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    if context_before_global:
        # The core can finish and launch its context while global is delayed.
        run = render_door._run
        context_started = Event()

        def delayed_global(command, *, log_prefix, env=None):
            if log_prefix == "CONVERT":
                window = command[command.index("--window") + 1]
                if window == "global-mesh":
                    assert context_started.wait(timeout=10), "context conversion never started"
                elif window.startswith("region:"):
                    result = run(command, log_prefix=log_prefix, env=env)
                    context_started.set()
                    return result
            return run(command, log_prefix=log_prefix, env=env)

        monkeypatch.setattr(render_door, "_run", delayed_global)
    histories = _histories(tmp_path)
    args = _arguments(tmp_path, histories, case="patch-case")
    assert products_door.run_products(args) == 0

    case = tmp_path / "out" / "patch-case"
    # case -> domain -> product -> valid-day
    assert (case / "global" / "2m_temperature_10m_winds" / "2026-10-03"
            / "hex48-3km_global_2m_temperature_10m_winds_f006.png").is_file()
    assert (case / "refined-core" / "total_qpf" / "2026-10-03"
            / "hex48-3km_refined-core_total_qpf_f000.png").is_file()
    assert (case / "refined-context" / "total_qpf" / "2026-10-03").is_dir()
    assert (case / "points" / "sounding-oklahoma-city" / "2026-10-03"
            / "sounding_oklahoma-city_20261003_1800_f006.png").is_file()
    assert (case / "points" / "meteogram-oklahoma-city" / "2026-10-03").is_dir()
    assert not list(case.rglob("rustwx_*"))

    converts = [c for stage, c in seen if stage == "CONVERT"]
    assert all(c[c.index("--remap") + 1] == "product" for c in converts)
    # All three views run concurrently; converter arrival order is unspecified.
    windows = [c[c.index("--window") + 1] for c in converts]
    assert len(windows) == 3
    assert windows.count("global-mesh") == windows.count("mesh") == 1
    context = [c for c in converts if c[c.index("--window") + 1].startswith("region:")]
    assert len(context) == 1
    context_window = context[0][context[0].index("--window") + 1]
    assert context_window.startswith("region:37.500,-97.500,")
    if context_before_global:
        assert windows.index(context_window) < windows.index("global-mesh")
    renders = [c for stage, c in seen if stage == "RENDER"]
    assert all("precipitation_type" not in c[c.index("--products") + 1] for c in renders)
    # one renderer start per view, every frame in it, one shared store
    assert len(renders) == 3
    assert all(sum(Path(c).name.startswith("wrfout_d01_") for c in r) == 2 for r in renders)
    assert all(Path(r[r.index("--store-root") + 1]).name == "store-series" for r in renders)
    # the interval product is filed at the frame that closes its window ...
    assert (case / "refined-core" / "qpf_6h" / "2026-10-03"
            / "hex48-3km_refined-core_qpf_6h_f006.png").is_file()
    assert not list(case.rglob("*qpf_6h_f000.png"))
    render_manifest = json.loads((case / "refined-core" / "render-manifest.json").read_text())
    assert render_manifest["series_store"] is True
    first, second = render_manifest["frames"]
    # ... the frame that cannot close it carries the renderer's own skip ...
    assert [item["product"] for item in first["skipped"]] == ["qpf_6h"]
    assert second["skipped"] == [] and "qpf_6h" in second["rendered"]
    # ... and a window the run is too short for is named, never dropped
    index = (case / "index.html").read_text(encoding="utf-8")
    assert "refined-core/qpf_6h/2026-10-03/hex48-3km_refined-core_qpf_6h_f006.png" in index
    assert "points/sounding-oklahoma-city" in index

    manifest = json.loads((case / "products-manifest.json").read_text())
    core = next(v for v in manifest["views"] if v["view"] == "refined-core")
    assert "precipitation_type" in core["products_unavailable"]
    # a window the run is too short for is not requested, and named with the reason in the manifest
    assert "qpf_24h" not in core["products_unavailable"]
    assert "needs 24 h" in manifest["windows_not_requested"]["qpf_24h"]
    assert "categorical_rain_surface" in core["products_unavailable"]["precipitation_type"]
    # the door's one history-header read also runs through subprocess.run, which this stand-in records
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    named = points[points.index("--points") + 1]
    assert "Oklahoma City@" in named and "Tokyo" not in named  # only points inside the core
    assert manifest["failed_views"] == []


def test_a_uniform_mesh_draws_global_and_the_default_regions(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=True, seen=seen)
    _points_stand_in(monkeypatch, seen)
    histories = _histories(tmp_path)
    args = _arguments(tmp_path, histories)
    assert products_door.run_products(args) == 0
    manifest = json.loads(next((tmp_path / "out").rglob("products-manifest.json")).read_text())
    views = {v["view"]: v for v in manifest["views"]}
    assert set(views) == {"global", "refined-core", "north-america", "europe"}
    assert views["refined-core"]["note"].startswith("uniform mesh")
    # the door's one history-header read also runs through subprocess.run, which this stand-in records
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert "Tokyo@" in points[points.index("--points") + 1]


@pytest.mark.parametrize("takes_brand", [True, False])
def test_the_model_label_and_brand_reach_the_point_pictures(monkeypatch, tmp_path, capsys, takes_brand):
    # WOOF sounding labels (2026-10-23): a host names the model top left and the brand top right.
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=True, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "points_take_brand", lambda exe: takes_brand)
    histories = _histories(tmp_path)
    brand = "WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z"
    args = _arguments(tmp_path, histories, model_label="WOOF Hex", brand=brand)
    assert products_door.run_products(args) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert points[points.index("--model-label") + 1] == "WOOF Hex"
    if takes_brand:
        assert points[points.index("--brand") + 1] == brand
    else:
        # an older rw_hex_points refuses unknown flags, so it draws without the brand and says so
        assert "--brand" not in points
        assert "POINTS brand not drawn" in capsys.readouterr().out
    assert points.index("--history") > points.index("--json")


def test_a_missing_points_engine_is_refused_by_name(monkeypatch, tmp_path):
    from woof.hex import engines

    monkeypatch.delenv(products_door.POINTS_ENV, raising=False)
    monkeypatch.delenv(engines.POINTS.gpuwm_env, raising=False)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)
    with pytest.raises(MpasPortError, match="rw_hex_points not found") as refusal:
        products_door.resolve_points_exe(None, tmp_path / "rw_mpas_convert")
    # the shared ladder's refusal, with the build line and the way round it
    assert "--bin rw_hex_points" in str(refusal.value)
    assert "--no-points" in str(refusal.value)
    # the build beside the converter in use is the one a run takes
    beside = tmp_path / "rw_hex_points"
    beside.write_bytes(b"x")
    assert products_door.resolve_points_exe(None, tmp_path / "rw_mpas_convert") == beside
    # doctor's engine table names it
    assert engines.POINTS in engines.ENGINES


def test_a_series_shares_a_store_only_when_every_frame_has_its_own_lead():
    assert render_door.series_leads([{"lead_seconds": 0}, {"lead_seconds": 10800}]) == [0, 10800]
    assert render_door.series_leads([{"lead_seconds": 0}]) is None            # nothing to window
    assert render_door.series_leads([{"lead_seconds": 0}, {"lead_seconds": 0}]) is None
    assert render_door.series_leads([{"lead_seconds": 0}, {}]) is None        # an older converter
    name = "rustwx_wrf_20261003_12z_f006_hex48-3km_qpf_6h.png"
    assert render_door.engine_lead_seconds(name) == 21600
    sub = "rustwx_wrf_20261003_12z_f000_hex48-3km_total_qpf_lead_000h30m00s.png"
    assert render_door.engine_lead_seconds(sub) == 1800
    assert render_door.engine_lead_seconds("something-else.png") is None


def test_without_distinct_leads_the_render_keeps_a_store_per_frame(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(render_door, "series_leads", lambda frames: None)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case")
    args.no_points = True
    assert products_door.run_products(args) == 0
    renders = [c for stage, c in seen if stage == "RENDER"]
    assert len(renders) == 6  # three views, two frames, a store each
    assert {Path(r[r.index("--store-root") + 1]).name for r in renders} == {"store-000", "store-001"}


def test_the_default_product_stage_draws_maps_when_the_point_tool_is_absent(monkeypatch, tmp_path, capsys):
    # woof 2.9.0 builds rw_hex_points standalone, so a pip install can lack
    # it. The forecast's automatic stage must still draw the maps (they never
    # need it) and name what it skipped; before, the whole stage refused.
    from woof.hex import forecast_door, products_door, render_door

    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda explicit=None: tmp_path / "rw_mpas_convert")

    def absent(explicit, convert_exe):
        raise products_door.ProductDoorError("rw_hex_points not found\n  # build line")

    monkeypatch.setattr(products_door, "resolve_points_exe", absent)
    seen = {}

    def fake_run(args):
        seen["no_points"] = args.no_points
        return 0

    monkeypatch.setattr(products_door, "run_products", fake_run)
    monkeypatch.setattr(forecast_door, "release_device_memory", lambda: None)
    out = tmp_path / "fc"
    out.mkdir()
    request = argparse.Namespace(out=out, grid=tmp_path / "g.nc", init=tmp_path / "i.nc",
                                 start_time="2026-10-03_12:00:00")
    # the in-command stage (--no-products-detached); the detached default
    # builds the same command and hands it to its own process
    inline = argparse.Namespace(products_detached=False)
    assert forecast_door.run_default_products(request, [out / "h.nc"], inline) == 0
    assert seen["no_points"] is True
    printed = capsys.readouterr().out
    assert "soundings and meteograms skipped: rw_hex_points not found" in printed
    assert "the maps are drawn" in printed

    # With the point tool present the stage draws everything, as before.
    monkeypatch.setattr(products_door, "resolve_points_exe", lambda explicit, convert_exe: tmp_path / "rw_hex_points")
    assert forecast_door.run_default_products(request, [out / "h.nc"], inline) == 0
    assert seen["no_points"] is False


def test_the_forecast_door_can_hand_the_product_stage_to_its_own_process(monkeypatch, tmp_path):
    from woof.hex import forecast_door

    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    assert "--products-detached" in parser.format_help()

    started = {}

    class Child:
        pid = 4242

    def fake_popen(command, **kwargs):
        started["command"], started["kwargs"] = [str(c) for c in command], kwargs
        return Child()

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = tmp_path / "fc"
    out.mkdir()
    request = argparse.Namespace(out=out, grid=tmp_path / "g.nc", init=tmp_path / "i.nc",
                                 start_time="2026-10-03_12:00:00")
    arguments = argparse.Namespace(products_detached=True)
    assert forecast_door.run_default_products(request, [out / "h.nc"], arguments) == 0
    command = started["command"]
    assert command[1:4] == ["-m", "woof.hex.cli", "products"]
    assert command[command.index("--forecast") + 1] == str(out)
    # no card for the child, and it outlives the forecast process
    assert started["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert started["kwargs"]["start_new_session"] is True
    assert (out / "products.log").is_file()


def test_the_product_stage_is_detached_by_default_so_the_cards_are_released(monkeypatch, tmp_path):
    """A bare forecast must not hold its card grant while CPU-only products draw
    (rented box 2026-10-13: four cards held ~66 GB each for 30 min after the forecast)."""
    from woof.hex import forecast_door

    parser = argparse.ArgumentParser()
    forecast_door.add_forecast_arguments(parser)
    assert parser.parse_args([]).products_detached is True
    assert parser.parse_args(["--no-products-detached"]).products_detached is False

    started = {}

    class Child:
        pid = 4343

    def fake_popen(command, **kwargs):
        started["command"] = [str(c) for c in command]
        return Child()

    import subprocess

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = tmp_path / "fc"
    out.mkdir()
    request = argparse.Namespace(out=out, grid=tmp_path / "g.nc", init=tmp_path / "i.nc",
                                 start_time="2026-10-03_12:00:00")
    # a caller that never set the flag gets the default, the detached stage
    assert forecast_door.run_default_products(request, [out / "h.nc"], argparse.Namespace()) == 0
    assert started["command"][1:4] == ["-m", "woof.hex.cli", "products"]


def test_releasing_device_memory_is_a_no_op_without_a_device_stack(monkeypatch):
    import sys

    from woof.hex import forecast_door

    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy -> ImportError
    assert forecast_door.release_device_memory() is None


def test_what_the_point_pictures_need_is_asked_for_before_any_view_is_drawn(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)

    def absent(explicit, convert):
        raise products_door.ProductDoorError("rw_hex_points is not installed")

    monkeypatch.setattr(products_door, "resolve_points_exe", absent)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case")
    with pytest.raises(MpasPortError, match="rw_hex_points is not installed"):
        products_door.run_products(args)
    assert seen == []  # no converter, catalog or renderer start was spent


def test_frames_in_flight_are_sized_from_the_host(tmp_path):
    histories = []
    for k in range(9):
        path = tmp_path / f"h{k}.nc"
        path.write_bytes(b"x" * 1000)
        histories.append(path)
    gib = 1 << 30
    # memory: half of 192 kB available / (8 x 1 kB x 3 views) = 4 frames
    assert products_door.auto_frame_jobs(histories, 3, available=192_000, cpus=256) == 4
    # cores: 12 cores / (2 x 3 views) = 2
    assert products_door.auto_frame_jobs(histories, 3, available=gib, cpus=12) == 2
    # never more than the frames there are, nor than the ceiling
    assert products_door.auto_frame_jobs(histories[:2], 1, available=gib, cpus=256) == 2
    assert products_door.auto_frame_jobs(histories, 1, available=gib, cpus=256) == len(histories)
    many = histories * 10   # 90 frames on a big host: the converter's ceiling, no lower fixed cap
    assert products_door.auto_frame_jobs(many, 1, available=gib, cpus=256) == products_door.MAX_FRAME_JOBS == 64
    # An Arizona box (192 vCPU, 2 TB) with three views of 1.5 GB frames: 28 a view, not 8
    big = 1536 * (1 << 20)
    assert min(int(0.5 * 2048 * gib // (products_door.FRAME_MEMORY_FACTOR * big * 3)), 192 // 6) == 28
    # a host that cannot hold two frames, or cannot be read, runs one at a time
    assert products_door.auto_frame_jobs(histories, 3, available=10_000, cpus=256) == 1
    assert products_door.auto_frame_jobs([], 3, available=gib, cpus=256) == 1


def test_frame_jobs_reach_both_engines_and_only_when_they_take_them(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case", frame_jobs="2")
    args.no_points = True
    assert products_door.run_products(args) == 0
    converts = [c for stage, c in seen if stage == "CONVERT"]
    assert converts and all(c[c.index("--jobs") + 1] == "2" for c in converts)
    imports = [c for stage, c in seen if stage in ("CATALOG", "RENDER")]
    assert imports and all(c[c.index("--import-jobs") + 1] == "2" for c in imports)
    manifest = json.loads((tmp_path / "out" / "patch-case" / "products-manifest.json").read_text())
    assert manifest["frame_jobs"] == 2

    # an engine built before the flags is never handed them
    seen.clear()
    monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: False)
    args = _arguments(tmp_path, _histories(tmp_path), case="older-engines", frame_jobs="2")
    args.no_points = True
    assert products_door.run_products(args) == 0
    assert not any("--jobs" in c or "--import-jobs" in c for _stage, c in seen)

    with pytest.raises(MpasPortError, match="--frame-jobs"):
        render_door.checked_frame_jobs("many")


def test_after_a_forecast_the_run_folder_is_the_whole_command(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    run = tmp_path / "fc"
    run.mkdir()
    grid = tmp_path / "meshes" / "patch.grid.nc"
    grid.parent.mkdir()
    grid.write_bytes(b"m")
    init = tmp_path / "meshes" / "patch.init.nc"
    init.write_bytes(b"i")
    for stamp in ("2026-10-03_18.00.00", "2026-10-04_00.00.00"):
        (run / f"cuda-history.{stamp}.nc").write_bytes(b"h")
    # the forecast door's own receipt: grid, init and the run's start (six hours
    # before the first frame kept here, which a guess from the frames would miss)
    (run / "forecast-receipt.json").write_text(json.dumps({
        "mesh": {"grid": str(grid)}, "init": {"path": str(init)},
        "schedule": {"start_time": "2026-10-03_12:00:00"}}), encoding="utf-8")
    parser = argparse.ArgumentParser()
    products_door.add_products_arguments(parser)
    assert products_door.run_products(parser.parse_args([str(run)])) == 0

    case = run / "products" / "patch-2026100312"
    manifest = json.loads((case / "products-manifest.json").read_text())
    assert manifest["mesh"] == str(grid) and manifest["init"] == str(init)
    assert manifest["simulation_start"] == "2026-10-03_12:00:00"
    converts = [c for stage, c in seen if stage == "CONVERT"]
    assert all(c[c.index("--mesh") + 1] == str(grid) and c[c.index("--init") + 1] == str(init)
               for c in converts)
    assert not list(run.glob("products/*/rustwx_*"))

    # a receipt that names a grid no longer on disk is refused as the receipt's
    grid.unlink()
    with pytest.raises(MpasPortError, match=r"forecast-receipt.json \(mesh.grid\) names a missing file"):
        products_door.run_products(parser.parse_args([str(run)]))
    # and with no receipt and no flag the door says what it lacks
    (run / "forecast-receipt.json").unlink()
    with pytest.raises(MpasPortError, match="no mesh: --mesh was not given"):
        products_door.run_products(parser.parse_args([str(run)]))


def test_completed_chemistry_run_adds_species_to_normal_products(monkeypatch, tmp_path):
    """No extra product flags are required after a completed aerosol forecast."""
    histories = _histories(tmp_path)
    args = _arguments(tmp_path, histories, case="aerosols")
    args.forecast_dir = str(tmp_path)
    args.no_points = args.no_mesh_view = True
    species = [dict(name=name, dry_deposition_m_s=0., settling_m_s=0.,
                    mass_extinction_m2_kg=1.) for name in ("dust_bin_7", "salt_fine")]
    (tmp_path / "chemistry-receipt.json").write_text(json.dumps({
        "schema": "gpuwm-hex.chemistry/v1", "species": species}))
    seen = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=True, seen=seen)
    renderer = render_door._run
    # Real Rust raw-plane import uses wrf_ store names, never NetCDF names.
    stored = {
        "var:wrf_dust_bin_7_column_mass", "var:wrf_dust_bin_7_surface_concentration",
        "var:wrf_dust_bin_7_aod550", "var:wrf_dust_bin_7_dry_deposition",
        "var:wrf_dust_bin_7_settled_mass", "var:wrf_salt_fine_column_mass",
        "var:wrf_salt_fine_surface_concentration", "var:wrf_salt_fine_aod550",
        "var:wrf_salt_fine_dry_deposition", "var:wrf_salt_fine_settled_mass",
        "var:wrf_aod550",
    }
    def with_chemistry_catalog(command, *, log_prefix, env=None):
        result = renderer(command, log_prefix=log_prefix, env=env)
        if log_prefix == "CATALOG":
            result.stdout += "".join(f"PRODUCT\t{slug}\tdirect\trenderable\tready\t\n"
                for slug in sorted(stored))
        return result
    monkeypatch.setattr(render_door, "_run", with_chemistry_catalog)
    assert products_door.run_products(args) == 0
    commands = [command for label, command in seen if label == "RENDER"]
    assert commands
    drawn = commands[0][commands[0].index("--products")+1].split(",")
    assert stored <= set(drawn)
    assert "var:dust_bin_7_column_mass" not in drawn
    manifest = json.loads((tmp_path / "out/aerosols/products-manifest.json").read_text())
    assert "var:wrf_dust_bin_7_dry_deposition" in manifest["chemistry_products"]


def test_limited_area_products_only_convert_the_domain(monkeypatch, tmp_path):
    seen = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "forecast_receipt", lambda path: {
        "admission": {"configuration": "limited-area"}})
    args = _arguments(tmp_path, _histories(tmp_path))
    args.no_points = True
    assert products_door.run_products(args) == 0
    converts = [command for stage, command in seen if stage == "CONVERT"]
    assert [command[command.index("--window") + 1] for command in converts] == ["mesh"]
    manifest = json.loads(next((tmp_path / "out").rglob("products-manifest.json")).read_text())
    assert [view["view"] for view in manifest["views"]] == ["refined-core"]
    assert manifest["maps_rendered"] > 0 and manifest["failed_views"] == []


def test_window_products_follow_the_stored_frames(tmp_path):
    """Rolling 2.9.4 run: a 3-hourly global history was asked for hourly windows it cannot close."""
    def frames(hours, step):
        return [tmp_path / f"cuda-history.2024-10-{8 + (12 + h) // 24:02d}_{(12 + h) % 24:02d}.00.00.nc"
                for h in range(0, hours + 1, step)]
    start = "2024-10-08_12:00:00"
    hourly, withheld = products_door.windowed_for_history(frames(24, 1), start)
    assert hourly == products_door.WINDOWED_PRODUCTS and withheld == {}
    three, withheld = products_door.windowed_for_history(frames(24, 3), start)
    assert three == ("qpf_6h", "qpf_24h")
    assert set(withheld) == {"qpf_1h", "10m_wind_run_max", "2m_temp_0_24h_max", "2m_temp_0_24h_min"}
    assert "every 1 h" in withheld["qpf_1h"] and "F001" in withheld["qpf_1h"]
    short, withheld = products_door.windowed_for_history(frames(6, 1), start)
    assert short == ("qpf_1h", "qpf_6h", "10m_wind_run_max")
    assert "needs 24 h" in withheld["qpf_24h"]


# -- the sounding brand from the run's own records ----------------------------------------------

@pytest.mark.parametrize("shape, km, expect", [
    ("global", 120.0, "WOOF Hex · Global 120 km · Init GFS 23 Oct 2026 18Z"),
    ("variable-resolution", 3.0, "WOOF Hex · Variable-resolution globe to 3 km · Init GFS 23 Oct 2026 18Z"),
    ("region", 3.75, "WOOF Hex · Region 3.75 km · Init GFS 23 Oct 2026 18Z"),
    ("storm-following", 3.0, "WOOF Hex · Storm-following 3 km · Init GFS 23 Oct 2026 18Z"),
])
def test_the_default_brand_names_each_shape(shape, km, expect):
    assert products_door.default_brand(shape=shape, spacing_km=km, source="GFS",
                                       start="2026-10-23_18:00:00") == expect


def test_the_default_brand_drops_what_the_run_does_not_record():
    assert products_door.default_brand(shape="global", spacing_km=120.0,
                                       start="2026-10-23_18:00:00") == "WOOF Hex · Global 120 km · Init 23 Oct 2026 18Z"
    assert products_door.default_brand(shape="region", spacing_km=3.0, source="HRRR") == "WOOF Hex · Region 3 km · Init HRRR"
    assert products_door.default_brand(shape="variable-resolution") == "WOOF Hex · Variable-resolution globe"
    assert products_door.default_brand() == "WOOF Hex"
    assert products_door.default_brand(model=None) == "WOOF"
    # a spacing measured from the grid's edges is written to the kilometre, or the quarter below 10 km
    assert products_door.default_brand(shape="global", spacing_km=120.4, spacing_measured=True) == "WOOF Hex · Global 120 km"
    assert products_door.default_brand(shape="variable-resolution", spacing_km=2.93,
                                       spacing_measured=True) == "WOOF Hex · Variable-resolution globe to 3 km"


def test_the_init_source_is_read_through_the_source_table():
    # what each door writes as --init-source: a row id or alias first, or the row's display name
    assert products_door.brand_source("gfs 2026-10-23T18; global row global-120km") == "GFS"
    assert products_door.brand_source("gfs analysis; moving run member m0") == "GFS"
    assert products_door.brand_source("era5 2024-07-14T15; spec abc") == "ERA5"
    assert products_door.brand_source("NCEP HRRR 3 km CONUS (native hybrid wrfnat + soil) 2024-07-14T15; cull") == "HRRR"
    assert products_door.brand_source("ifs-ens 2026-10-23T00") == "IFS ENS"
    # text that names no row gives no source, never a guess
    assert products_door.brand_source("staged analysis; moving run member m0") is None
    assert products_door.brand_source(None) is None


def test_a_storm_following_run_brands_its_soundings_from_its_own_records(monkeypatch, tmp_path):
    # A storm-following run (2026-10-23): `moving products` and the segment product stage
    # passed no --brand, and the soundings had nothing top right.
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    run = tmp_path / "run"
    fc = run / "segment-000" / "fc"
    fc.mkdir(parents=True)
    grid = tmp_path / "m0.grid.nc"
    grid.write_bytes(b"m")
    init = tmp_path / "m0.init.nc"
    init.write_bytes(b"i")
    for stamp in ("2026-10-23_18.00.00", "2026-10-24_00.00.00"):
        (fc / f"cuda-history.{stamp}.nc").write_bytes(b"h")
    (fc / "forecast-receipt.json").write_text(json.dumps({
        "mesh": {"name": "m0", "grid": str(grid)}, "init": {"path": str(init),
                 "source": "gfs analysis; moving run member m0"},
        "schedule": {"start_time": "2026-10-23_18:00:00"}}), encoding="utf-8")
    rows = tmp_path / "m0-rows.json"
    (run / "controller.json").write_text(json.dumps({
        "segments": [{"index": 0, "path": str(fc.parent)}],
        "members": {"m0": {"name": "m0", "mesh_rows": str(rows)}}}), encoding="utf-8")
    asked = []

    def member_row(name, mesh, extra_rows=()):
        asked.append((name, [str(p) for p in extra_rows]))
        return (3.0, 100.0) if str(rows) in [str(p) for p in extra_rows] and name == "m0" else (None, None)

    monkeypatch.setattr(products_door, "_mesh_row_spacing", member_row)
    parser = argparse.ArgumentParser()
    products_door.add_products_arguments(parser)
    assert products_door.run_products(parser.parse_args([str(fc)])) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert points[points.index("--brand") + 1] == "WOOF Hex · Storm-following 3 km · Init GFS 23 Oct 2026 18Z"
    manifest = json.loads(next((fc / "products").rglob("products-manifest.json")).read_text(encoding="utf-8"))
    assert manifest["points"]["brand_from"] == "run metadata"


def test_a_global_row_without_brand_gets_one(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=True, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "_mesh_row_spacing", lambda name, mesh, extra_rows=(): (120.0, 120.0))
    monkeypatch.setattr(products_door, "forecast_receipt", lambda path: {
        "init": {"source": "gfs 2026-10-22T00; global row global-120km"},
        "schedule": {"start_time": "2026-10-22_00:00:00"}})
    args = _arguments(tmp_path, _histories(tmp_path))
    assert products_door.run_products(args) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert points[points.index("--brand") + 1] == "WOOF Hex · Global 120 km · Init GFS 22 Oct 2026 00Z"


def test_a_first_look_without_a_forecast_receipt_reads_the_init_source_receipt(monkeypatch, tmp_path):
    # the first look draws f000 with --history while the forecast runs, before its receipt exists
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "_mesh_row_spacing", lambda name, mesh, extra_rows=(): (3.0, 60.0))
    (tmp_path / "init-source-receipt.json").write_text(json.dumps({
        "source": "era5", "cycle": "2024-07-14_15:00:00"}), encoding="utf-8")
    args = _arguments(tmp_path, _histories(tmp_path), simulation_start="2024-07-14_15:00:00")
    assert products_door.run_products(args) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert points[points.index("--brand") + 1] == "WOOF Hex · Variable-resolution globe to 3 km · Init ERA5 14 Jul 2024 15Z"


def test_frames_drawn_live_keep_their_maps_and_the_full_pass_draws_the_rest(monkeypatch, tmp_path):
    """2026-10-24 (hex global 60 km 24 h): every hour was drawn while the forecast ran, then the full pass rendered
    all 25 frames again (81 s). With the first look's receipt the pass draws the windowed maps over every frame, every
    map of the frames no live draw took, and the points; a frame rewritten after its draw is drawn again."""
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    histories = _histories(tmp_path)
    assert products_door.run_products(_arguments(tmp_path, histories, case="patch-case")) == 0   # f000 and f006 on disk
    case = tmp_path / "out" / "patch-case"
    f000 = histories[0]
    receipt = tmp_path / "live-frames.json"

    def write_receipt():
        receipt.write_text(json.dumps({
            "schema": "gpuwm-hex.live-frames/v1", "out": str(tmp_path / "out"), "case": "patch-case",
            "drawn": [f000.name], "stat": {f000.name: [f000.stat().st_size, f000.stat().st_mtime_ns]},
            "products": ["2m_temperature_10m_winds", "total_qpf"]}), encoding="utf-8")

    write_receipt()
    seen.clear()
    assert products_door.run_products(_arguments(tmp_path, histories, case="patch-case", live_frames=str(receipt))) == 0
    renders = [c for stage, c in seen if stage == "RENDER"]
    frames_of = lambda c: sorted(Path(x).name for x in c if Path(x).name.startswith("wrfout_d01_"))
    series = [c for c in renders if c[c.index("--products") + 1] == "qpf_6h"]
    rest = [c for c in renders if "qpf_6h" not in c[c.index("--products") + 1]]
    assert len(series) == len(rest) == 3                           # each view: the windowed maps, and the rest
    assert all(len(frames_of(c)) == 2 for c in series)             # the windowed maps over the whole series
    assert all(len(frames_of(c)) == 1 for c in rest)               # every map of the frame not drawn live (f006)
    manifest = json.loads((case / "products-manifest.json").read_text())
    assert manifest["live_frames"]["drawn_live"] == [f000.name]
    assert manifest["live_frames"]["drawn_here"] == [histories[1].name]
    assert manifest["first_maps"]["maps"][0] == "2m_temperature_10m_winds"
    assert (case / "global" / "2m_temperature_10m_winds" / "2026-10-03"
            / "hex48-3km_global_2m_temperature_10m_winds_f000.png").is_file()
    assert any(stage == "POINTS" for stage, _ in seen)

    # The forecast rewrote f000 after its draw: the receipt no longer names that frame, so every frame is drawn.
    os.utime(f000, ns=(f000.stat().st_atime_ns, f000.stat().st_mtime_ns + 10**9))
    seen.clear()
    assert products_door.run_products(_arguments(tmp_path, histories, case="patch-case", live_frames=str(receipt))) == 0
    renders = [c for stage, c in seen if stage == "RENDER"]
    assert len(renders) == 3 and all(len(frames_of(c)) == 2 for c in renders)
    assert json.loads((case / "products-manifest.json").read_text())["live_frames"] is None


def _write_patch_mesh(path: Path, *, lat0=35.0, lon0=-97.0, half_deg=3.0, step_deg=0.25, limited=True):
    """A tiny MPAS-shaped grid file: a lat-lon patch of cell centres (radians),
    unit-sphere areas, and a boundary ring when ``limited``."""
    np = pytest.importorskip("numpy")
    netcdf = pytest.importorskip("netCDF4")
    lats = np.arange(lat0 - half_deg, lat0 + half_deg + 1e-9, step_deg)
    lons = np.arange(lon0 - half_deg, lon0 + half_deg + 1e-9, step_deg)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    n = glat.size
    side = np.radians(step_deg)
    with netcdf.Dataset(path, "w") as grid:
        grid.setncattr("sphere_radius", 1.0)
        grid.createDimension("nCells", n)
        grid.createVariable("latCell", "f8", ("nCells",))[:] = np.radians(glat.ravel())
        grid.createVariable("lonCell", "f8", ("nCells",))[:] = np.radians(glon.ravel())
        grid.createVariable("areaCell", "f8", ("nCells",))[:] = (
            side * side * np.cos(np.radians(glat.ravel())))
        ring = np.zeros(glat.shape, dtype="i4")
        if limited:
            ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = 1
        grid.createVariable("bdyMaskCell", "i4", ("nCells",))[:] = ring.ravel()


def test_mesh_point_check_splits_on_and_off_mesh_points(tmp_path):
    # graded-plateau 1 km region, box J, 2026-10-24: London was 7,025 km off a
    # 300 km Oklahoma region and its refusal took every sounding with it.
    mesh = tmp_path / "patch.grid.nc"
    _write_patch_mesh(mesh)
    check = products_door.mesh_point_check(mesh, [
        ("Oklahoma City", 35.47, -97.52), ("London", 51.51, -0.13), ("Wichita", 37.69, -97.34)])
    on_mesh, off_mesh, (clat, clon) = check
    assert [p[0] for p in on_mesh] == ["Oklahoma City", "Wichita"]
    assert [p[0] for p in off_mesh] == ["London"]
    assert abs(clat - 35.0) < 0.1 and abs(clon + 97.0) < 0.1
    # a point just outside the patch is off by the cell-spacing rule, with its distance
    near = products_door.mesh_point_check(mesh, [("Edge", 38.6, -97.0)])[1][0]
    assert near[3] is not None and near[3] > near[4]


def test_a_limited_area_run_drops_off_mesh_points_and_draws_the_rest(monkeypatch, tmp_path, capsys):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "forecast_receipt", lambda path: {
        "admission": {"configuration": "limited-area"}})
    histories = _histories(tmp_path)
    _write_patch_mesh(tmp_path / "patch.grid.nc")
    args = _arguments(tmp_path, histories,
                      points="Oklahoma City@35.47,-97.52;London@51.51,-0.13;Wichita@37.69,-97.34")
    assert products_door.run_products(args) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    named = points[points.index("--points") + 1]
    assert named == "Oklahoma City@35.47,-97.52;Wichita@37.69,-97.34"
    assert "POINTS dropped: London (51.51, -0.13) is off this limited-area mesh" in capsys.readouterr().out
    manifest = json.loads(next((tmp_path / "out").rglob("products-manifest.json")).read_text())
    assert [d["name"] for d in manifest["points"]["dropped"]] == ["London"]
    assert manifest["points"]["rc"] == 0


def test_a_limited_area_run_defaults_to_the_cities_on_its_mesh(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "forecast_receipt", lambda path: {
        "admission": {"configuration": "limited-area"}})
    histories = _histories(tmp_path)
    _write_patch_mesh(tmp_path / "patch.grid.nc")
    assert products_door.run_products(_arguments(tmp_path, histories)) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    names = [item.split("@")[0] for item in points[points.index("--points") + 1].split(";")]
    assert names[0] == "Oklahoma City"  # nearest the domain centre first
    assert {"Wichita", "Tulsa"} <= set(names)
    assert not {"London", "Tokyo", "Sydney", "Denver", "Chicago"} & set(names)
    assert len(names) <= products_door.DEFAULT_REGION_POINT_COUNT


def test_a_limited_area_run_with_no_city_takes_its_centre(monkeypatch, tmp_path):
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    monkeypatch.setattr(products_door, "forecast_receipt", lambda path: {
        "admission": {"configuration": "limited-area"}})
    histories = _histories(tmp_path)
    _write_patch_mesh(tmp_path / "patch.grid.nc", lat0=-60.0, lon0=-140.0, half_deg=1.0)
    assert products_door.run_products(_arguments(tmp_path, histories)) == 0
    points = [c for stage, c in seen if stage == "POINTS" and Path(str(c[0])).name == "rw_hex_points"][0]
    assert points[points.index("--points") + 1].startswith("Domain centre@-60.0")


# ---------------------------------------------------------------------------
# each view's map headline (2026-10-30)
# ---------------------------------------------------------------------------
def test_each_view_title_names_the_view_as_the_site_menu_does():
    """THE BREAKAGE (2026-10-30): the site menu read 'Core zoom · 8 → 38 km' over a picture
    headlined '(hex 133-6.9 km) ... Δx 8.063 km' (run_713c9e1913164e02): every view carried the
    whole mesh's range.  Each view now asks the converter for its own name and the spacing its
    picture samples."""
    assert products_door.view_title("refined-core") == "Core zoom {spacing}"
    assert products_door.view_title("refined-core", moving=True) == "Moving core {spacing}"
    assert products_door.view_title("refined-context") == "Around the core {spacing}"
    assert products_door.view_title("global") == "Whole globe {spacing}"
    assert products_door.view_title("region") == "Region {spacing}"
    assert products_door.view_title("event") == "Around the point {spacing}"
    assert products_door.view_title("north-america") == "North America {spacing}"
    # a long view name keeps its whole words that leave the spacing room in the 32-character headline
    long = products_door.view_title("southern-great-plains-and-ozarks")
    assert long == "Southern Great {spacing}"
    assert len(long.replace("{spacing}", "12.5-128 km")) <= render_door.GRID_LABEL_MAX_CHARS


def test_a_caller_view_label_is_drawn_in_plain_text_or_refused_by_name():
    labels = products_door.view_labels(["refined-core=Core zoom · 8 → 38 km",
                                        "refined-context=Around the core · 8 → 128 km"])
    assert labels == {"refined-core": "Core zoom 8-38 km", "refined-context": "Around the core 8-128 km"}
    for bad, why in (("refined-core", "VIEW=TEXT"), ("refined-core=a|b", "'|'"),
                     ("refined-core=" + "x" * 33, "32 characters"), ("Core Zoom=x", "--view")):
        with pytest.raises(MpasPortError, match=__import__("re").escape(why)):
            products_door.view_labels([bad])
    # the doors forward it to their product stage with the other labels
    parser = argparse.ArgumentParser()
    products_door.add_label_arguments(parser)
    forwarded = products_door.label_argv(parser.parse_args(["--view-label", "global=Whole globe 8-128 km"]))
    assert forwarded == ["--view-label", "global=Whole globe 8-128 km"]


@pytest.mark.parametrize("caller", [False, True])
def test_each_view_is_headlined_by_its_own_picture_and_files_keep_the_mesh_token(monkeypatch, tmp_path, caller):
    """The converter draws the view's name with the spacing of the cells its picture samples
    (rw_mpas_convert --grid-label), and rw_wrfbatch then names its raw pictures after that label;
    the delivered names keep the mesh's token, so the WOOF worker files them as before."""
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    run = render_door._run
    sampled = {"mesh": "3-19 km", "global-mesh": "3-48 km"}
    by_frames: dict[str, str] = {}

    def slug(label: str) -> str:
        return "".join(ch for ch in label.lower() if ch.isalnum() or ch in "-.")

    def as_the_new_engines(command, *, log_prefix, env=None):
        result = run(command, log_prefix=log_prefix, env=env)
        command = [str(c) for c in command]
        if log_prefix == "CONVERT" and "--grid-label" in command:
            window = command[command.index("--window") + 1]
            drawn = command[command.index("--grid-label") + 1].replace("{spacing}", sampled.get(window, "3-30 km"))
            receipt = Path(command[command.index("--json") + 1])
            doc = json.loads(receipt.read_text(encoding="utf-8"))
            doc.update(grid_label=drawn, mesh_grid_label=doc["grid_label"])
            receipt.write_text(json.dumps(doc), encoding="utf-8")
            by_frames[command[command.index("--out-dir") + 1]] = drawn
        elif log_prefix == "RENDER":
            frames = next(str(Path(c).parent) for c in command if Path(c).name.startswith("wrfout_d01_"))
            lines = []
            for line in result.stdout.splitlines():
                if line.startswith("RENDERED ") and frames in by_frames:
                    product, _, path = line[len("RENDERED "):].partition(" ")
                    moved = Path(path).with_name(Path(path).name.replace("_hex48-3km_", f"_{slug(by_frames[frames])}_"))
                    Path(path).rename(moved)
                    line = f"RENDERED {product} {moved}"
                lines.append(line)
            result.stdout = "\n".join(lines) + "\n"
        return result

    monkeypatch.setattr(render_door, "_run", as_the_new_engines)
    histories = _histories(tmp_path)
    over = {"case": "patch-case"}
    if caller:
        over["view_label"] = "refined-core=Core zoom · 3 → 19 km"
    assert products_door.run_products(_arguments(tmp_path, histories, **over)) == 0

    converts = {c[c.index("--window") + 1].split(":")[0]: c for stage, c in seen if stage == "CONVERT"}
    asked = {window: c[c.index("--grid-label") + 1] for window, c in converts.items()}
    assert asked == {"mesh": "Core zoom 3-19 km" if caller else "Core zoom {spacing}",
                     "region": "Around the core {spacing}", "global-mesh": "Whole globe {spacing}"}
    case = tmp_path / "out" / "patch-case"
    core = json.loads((case / "refined-core" / "render-manifest.json").read_text())
    assert core["grid_label"] == "Core zoom 3-19 km" and core["mesh_grid_label"] == "hex 48-3 km"
    world = json.loads((case / "global" / "render-manifest.json").read_text())
    assert world["grid_label"] == "Whole globe 3-48 km"
    # delivered names and folders exactly as before the headline changed
    assert (case / "refined-core" / "total_qpf" / "2026-10-03"
            / "hex48-3km_refined-core_total_qpf_f000.png").is_file()
    assert (case / "global" / "2m_temperature_10m_winds" / "2026-10-03"
            / "hex48-3km_global_2m_temperature_10m_winds_f006.png").is_file()
    delivered = [p.name for p in case.rglob("*.png")]
    assert delivered and not [n for n in delivered if "zoom" in n or "globe" in n or "around" in n]


def _wide_context(monkeypatch):
    """The stand-in converter lays the context window down as an Arizona zoom globe did: a 2000 x 1401 Lambert
    grid at the core's 3.4 km (woof hex 0.4.10, 2026-10-30)."""
    fake = render_door._run

    def run(command, *, log_prefix, env=None):
        result = fake(command, log_prefix=log_prefix, env=env)
        if log_prefix == "CONVERT":
            command = [str(c) for c in command]
            window = command[command.index("--window") + 1]
            if window.startswith(("region:", "lambert:", "mesh")):
                report = Path(command[command.index("--json") + 1])
                doc = json.loads(report.read_text(encoding="utf-8"))
                doc["window_spec"] = ({"kind": "lambert", "centre_lat": 34.047, "centre_lon": -111.11,
                                       "dx_metres": 3427.0, "nx": 2000, "ny": 1401}
                                      if not window.startswith("mesh") else dict(doc["window_spec"]))
                doc["window_spec"].update(truelat1=29.047, truelat2=39.047, stand_lon=-111.11)
                report.write_text(json.dumps(doc), encoding="utf-8")
        return result

    monkeypatch.setattr(render_door, "_run", run)


def test_the_context_view_is_one_picture_not_tiles(monkeypatch, tmp_path):
    """An Arizona zoom globe on 8 cards (woof hex 0.4.10, 2026-10-30): --tiles auto cut the 6,850 km context
    view into six more passes (2,996 maps where 0.4.9 rendered 477), drawn after the forecast for 31 min."""
    seen: list = []
    _stand_in_engines(monkeypatch, tmp_path, uniform=False, seen=seen)
    _points_stand_in(monkeypatch, seen)
    _wide_context(monkeypatch)
    args = _arguments(tmp_path, _histories(tmp_path), case="patch-case")
    args.no_points = True
    assert products_door.run_products(args) == 0
    converts = [c[c.index("--window") + 1] for stage, c in seen if stage == "CONVERT"]
    assert not any(w.startswith("lambert:") for w in converts)          # no cut was converted
    context = tmp_path / "out" / "patch-case" / "refined-context" / "total_qpf"
    assert (context / "2026-10-03").is_dir() and not (context / "full").exists()

    # asked for by name, the cuts are still drawn
    seen.clear()
    args = _arguments(tmp_path, _histories(tmp_path), case="tiled-case", tiles="regional")
    args.no_points = True
    assert products_door.run_products(args) == 0
    converts = [c[c.index("--window") + 1] for stage, c in seen if stage == "CONVERT"]
    assert sum(w.startswith("lambert:") for w in converts) >= 6


def test_a_variable_resolution_globe_is_drawn_at_its_pictures_pixels(monkeypatch, tmp_path):
    """An Arizona zoom globe (woof hex 0.4.10, 2026-10-30): the globe was converted at the mesh's 9.4 km median,
    a 4240 x 2121 grid (11 GB, 117 s a frame), for an 1800 x 1000 picture."""
    zoom = [3.0] * 600 + [6.0] * 100 + [12.0] * 100 + [24.0] * 100 + [48.0] * 100 + [96.0] * 100
    monkeypatch.setattr(products_door, "mesh_cell_spacing_km", lambda mesh: zoom)
    window, why = products_door.global_window("mesh.nc", "1800x1000")
    assert window == "latlon:-90,90,-180,179.8,0.2" and "variable-resolution" in why
    # a quasi-uniform mesh keeps the sphere at its own spacing
    monkeypatch.setattr(products_door, "mesh_cell_spacing_km", lambda mesh: [15.0 + 0.01 * k for k in range(100)])
    assert products_door.global_window("mesh.nc", "1800x1000") == ("global", None)
    # a refined mesh whose median is coarser than a pixel is never drawn finer than it holds
    monkeypatch.setattr(products_door, "mesh_cell_spacing_km", lambda mesh: [3.0] * 10 + [60.0] * 90)
    assert products_door.global_window("mesh.nc", "1800x1000") == ("global", None)
    # a grid that cannot be read keeps the old window
    monkeypatch.setattr(products_door, "mesh_cell_spacing_km", lambda mesh: None)
    assert products_door.global_window("mesh.nc", "1800x1000") == ("global", None)


def test_the_context_box_is_converted_at_its_pictures_pixels():
    """An Arizona zoom globe (woof hex 0.4.10, 2026-10-30): the 6,850 km context box was a 2000 x 1401 grid at
    3.4 km for a 1400 x 1000 picture (4.9 km a pixel)."""
    window, why = products_door.context_window(34.047, -111.11, 6850.0, "1400x1000", 3000.0)
    assert window.startswith("lambert:34.047,-111.11,") and ",1400,981," in window and why
    # the converter's own region window: near the equator (lat-lon there), and where a pixel is finer than the core
    assert products_door.context_window(5.0, -60.0, 6850.0, "1400x1000", 3000.0) == ("region:5.000,-60.000,6850,4795", None)
    assert products_door.context_window(34.0, -97.0, 1500.0, "1400x1000", 3000.0)[1] is None

"""Focused tests for the render door's pure logic and named refusals.

The engine legs (rw_mpas_convert, rw_wrfbatch) are proven against the real
binaries on the render node; what lives here is everything the door decides
WITHOUT them: executable resolution refusals, the scratch-placement gate,
the ruled layout, engine-filename splitting and the product-coverage
refusal grammar.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from woof.hex.render_door import (  # noqa: E402
    RenderDoorError,
    default_scratch,
    place_path,
    refuse_scratch_inside_out,
    refuse_uncovered_products,
    resolve_convert_exe,
    resolve_renderer_exe,
    split_engine_name,
    valid_day,
)


class TestValidDay:
    def test_iso_instant(self):
        assert valid_day("2025-03-15T00:00:00") == "2025-03-15"

    def test_wrf_stamp(self):
        assert valid_day("2025-03-15_00:00:00") == "2025-03-15"

    def test_unreadable_is_undated_not_guessed(self):
        assert valid_day("hour twelve") == "undated"
        assert valid_day(None) == "undated"
        assert valid_day("2025-02-31_00:00:00") == "undated"


class TestSplitEngineName:
    def test_engine_name_splits_on_known_slug(self):
        domain, product = split_engine_name(
            "rustwx_arwen_20250314_12z_f012_d01-22km_composite_reflectivity.png",
            "composite_reflectivity")
        assert domain == "d01-22km"
        assert product == "composite_reflectivity"

    def test_underscore_domain_survives(self):
        domain, product = split_engine_name(
            "rustwx_arwen_20250314_12z_f000_native_grid_2m_temperature.png",
            "2m_temperature")
        assert domain == "native_grid"
        assert product == "2m_temperature"

    def test_unparseable_name_files_under_native_grid(self):
        domain, product = split_engine_name("whatever.png", "slp")
        assert domain == "native_grid"
        assert product == "slp"

    def test_generic_var_slug_colon_and_hash_suffix(self):
        # The event says var:wrf_psfc; the filename spells it var_wrf_psfc
        # plus a content-hash suffix, and a ':' directory cannot exist on
        # Windows -- the product dir must use the safe spelling.
        domain, product = split_engine_name(
            "rustwx_wrf_20250314_12z_f012_d01-22km_var_wrf_psfc_647248f9d12a5e11.png",
            "var:wrf_psfc")
        assert domain == "d01-22km"
        assert product == "var_wrf_psfc"


class TestLayout:
    def test_ruled_layout_domain_product_day(self, tmp_path):
        path = place_path(tmp_path, "d01-22km", "slp", "2025-03-15", "x.png")
        assert path == tmp_path / "d01-22km" / "slp" / "2025-03-15" / "x.png"

    def test_missing_facts_stay_nameable(self, tmp_path):
        path = place_path(tmp_path, "", "", "", "x.png")
        assert path == (tmp_path / "native_grid" / "unclassified"
                        / "undated" / "x.png")


class TestScratchPlacement:
    def test_default_scratch_is_a_sibling(self, tmp_path):
        out = tmp_path / "png"
        scratch = default_scratch(out)
        assert scratch.parent == out.parent
        assert scratch != out
        assert out not in scratch.parents

    def test_scratch_inside_out_is_refused_by_name(self, tmp_path):
        out = tmp_path / "png"
        with pytest.raises(RenderDoorError, match="inside the delivered"):
            refuse_scratch_inside_out(out / "scratch", out)
        with pytest.raises(RenderDoorError, match="inside the delivered"):
            refuse_scratch_inside_out(out, out)

    def test_sibling_scratch_is_accepted(self, tmp_path):
        refuse_scratch_inside_out(tmp_path / "png.render-scratch",
                                  tmp_path / "png")


class TestExecutableResolution:
    """The refusals, on a machine that has none of the binaries.

    Emptying ``PATH`` and the environment variables is no longer enough to
    simulate that machine: the ladder also reads woof's bridge directories,
    which is the whole point of the change that added them -- a developer box
    that has ever run ``woof fetch-bridges`` HAS these binaries, and these
    tests used to pass there only because the door could not see them.  So
    the woof rung is emptied explicitly, and there is a test below proving
    it is a rung at all.
    """

    @pytest.fixture(autouse=True)
    def _a_machine_with_no_engines(self, monkeypatch):
        from woof.hex import engines

        for name in (
            "WOOF_HEX_RW_WRFBATCH",
            "MPAS_PORT_RW_WRFBATCH",
            "WOOF_RW_WRFBATCH",
            "WOOF_HEX_RW_MPAS_CONVERT",
            "MPAS_PORT_RW_MPAS_CONVERT",
            "WOOF_RW_MPAS_CONVERT",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PATH", "")
        monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: ())

    def test_missing_renderer_names_breakage_and_remedy(self):
        with pytest.raises(RenderDoorError) as caught:
            resolve_renderer_exe(None)
        message = str(caught.value)
        assert "rw_wrfbatch not found" in message
        assert "cargo build" in message
        assert "MPAS_PORT_RW_WRFBATCH" in message
        # The staged-bundle remedy comes FIRST, because it is one command
        # and needs no toolchain.  A refusal that only offers a cargo build
        # sends a user who has already installed woof the long way round.
        assert "woof fetch-bridges" in message

    def test_missing_converter_names_breakage_and_remedy(self):
        with pytest.raises(RenderDoorError) as caught:
            resolve_convert_exe(None)
        message = str(caught.value)
        assert "rw_mpas_convert not found" in message
        assert "cargo build" in message
        assert "woof fetch-bridges" in message

    def test_env_pointing_at_missing_file_is_a_hard_error(self, monkeypatch):
        monkeypatch.setenv("MPAS_PORT_RW_WRFBATCH", "/no/such/rw_wrfbatch")
        with pytest.raises(RenderDoorError, match="which is not a file"):
            resolve_renderer_exe(None)

    def test_the_gpuwm_bridge_directory_is_a_real_rung(self, monkeypatch, tmp_path):
        """A staged binary resolves with NO environment variable set.

        This is the defect the shared ladder closed, stated as a test: a
        user who ran the one command that installs the engine still met a
        refusal, because no door looked where that command writes.
        """

        from woof.hex import engines

        staged = tmp_path / engines.executable_name("rw_wrfbatch")
        staged.write_bytes(b"")
        staged.chmod(0o755)
        monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (staged,))

        assert resolve_renderer_exe(None) == staged.resolve()


class TestProductCoverage:
    ROWS = {
        "slp": ("derived", "renderable", "sea-level pressure"),
        "helicity_0_3km": ("derived", "missing-fields",
                           "needs stored U, V on model levels"),
    }

    def test_group_keywords_never_refuse(self):
        refuse_uncovered_products("all", self.ROWS, Path("h.nc"), [])
        refuse_uncovered_products("derived", self.ROWS, Path("h.nc"), [])

    def test_renderable_slug_passes(self):
        refuse_uncovered_products("slp", self.ROWS, Path("h.nc"), [])

    def test_uncovered_product_names_product_and_fields(self):
        with pytest.raises(RenderDoorError) as caught:
            refuse_uncovered_products(
                "slp,helicity_0_3km", self.ROWS, Path("h.nc"), ["W", "REFL"])
        message = str(caught.value)
        assert "helicity_0_3km" in message
        assert "needs stored U, V" in message
        assert "W, REFL" in message

    def test_windowed_refusal_says_what_a_window_needs_first(self):
        # RRFS hrrrv5 qpf_1h (2026-10-28): the absent-field list ("RAINC")
        # read as the cause while one frame per store was.
        rows = {"qpf_1h": ("windowed", "excluded", "windowed accumulations "
                           "need more than one stored whole-hour frame")}
        with pytest.raises(RenderDoorError) as caught:
            refuse_uncovered_products("qpf_1h", rows, Path("h.nc"), ["RAINC"])
        message = str(caught.value)
        assert "two or more frames" in message
        assert message.index("two or more frames") < message.index("RAINC")

    def test_unknown_slug_lists_renderable_catalog(self):
        with pytest.raises(RenderDoorError) as caught:
            refuse_uncovered_products("no_such", self.ROWS, Path("h.nc"), [])
        message = str(caught.value)
        assert "no_such" in message
        assert "slp" in message


class TestCatalogListing:
    """The door reads the renderer's --list-products rows.

    woof 2.8.0's rw_wrfbatch prints six tab-separated fields per row (a
    machine reason code after the detail). A parser that demanded exactly
    five dropped every row and refused every render, so both widths are
    read here, and a listing with no rows at all still refuses.
    """

    @staticmethod
    def _listing(monkeypatch, tmp_path, stdout):
        from woof.hex import render_door

        def fake_run(command, *, log_prefix):
            class Result:
                returncode = 0
                stderr = ""

            Result.stdout = stdout
            return Result()

        monkeypatch.setattr(render_door, "_run", fake_run)
        return render_door.list_catalog(
            Path("rw_wrfbatch"), Path("wrfout_d01"), scratch=tmp_path,
            heavy=False)

    def test_six_field_rows_of_the_current_renderer_are_read(
            self, monkeypatch, tmp_path):
        stdout = (
            "THEME default\n"
            "PRODUCT\tslp\tderived\trenderable\tsea-level pressure\t\n"
            "PRODUCT\t2m_temp_0_24h_max\twindowed\texcluded\twindowed "
            "accumulations need more than one stored whole-hour frame\t"
            "windowed-needs-whole-hour-frames\n"
            "CATALOG total=2 excluded=1 missing-fields=0 renderable=1\n"
        )
        rows, summary, _ = self._listing(monkeypatch, tmp_path, stdout)
        assert rows["slp"] == ("derived", "renderable", "sea-level pressure")
        assert rows["2m_temp_0_24h_max"][:2] == ("windowed", "excluded")
        assert summary.endswith("renderable=1")

    def test_five_field_rows_are_still_read(self, monkeypatch, tmp_path):
        rows, _, _ = self._listing(
            monkeypatch, tmp_path,
            "PRODUCT\tslp\tderived\trenderable\tsea-level pressure\n")
        assert rows == {"slp": ("derived", "renderable",
                                "sea-level pressure")}

    def test_a_listing_with_no_rows_refuses(self, monkeypatch, tmp_path):
        with pytest.raises(RenderDoorError, match="no catalog rows"):
            self._listing(monkeypatch, tmp_path, "THEME default\n")


class TestComposeArguments:
    """The compose door's refusals and the command it builds.

    A composite is several model runs going into one frame.  Everything
    here is about the ways that can be asked for incoherently, because a
    composite built from a half-stated source list would render cleanly
    and show the wrong weather.
    """

    @staticmethod
    def _parser():
        from woof.hex.cli import build_parser

        return build_parser()

    def test_compose_makes_history_and_mesh_optional(self):
        parsed = self._parser().parse_args(
            ["render", "--compose", "sources.json", "--out", "png",
             "--window", "composite"])
        assert parsed.compose == "sources.json"
        assert parsed.history is None and parsed.mesh is None

    def test_a_single_run_render_still_requires_both(self):
        parsed = self._parser().parse_args(
            ["render", "--history", "h.nc", "--mesh", "m.nc", "--out", "png"])
        assert parsed.compose is None
        assert parsed.window == "mesh"

    def test_composite_window_is_offered(self):
        parsed = self._parser().parse_args(
            ["render", "--compose", "s.json", "--out", "png",
             "--window", "composite"])
        assert parsed.window == "composite"

    def test_compose_command_carries_no_second_source_list(self, tmp_path):
        """--compose owns the meshes; --history/--mesh must not also appear.

        Two source lists in one command can disagree about which run is
        the base, and the composite would then draw the wrong run
        everywhere the overlays do not cover.
        """

        from woof.hex import render_door

        seen: dict[str, list[str]] = {}

        def fake_run(command, *, log_prefix):
            seen["command"] = [str(part) for part in command]
            report = tmp_path / "scratch" / "convert-report.json"
            report.write_text('{"frames": []}', encoding="utf-8")

            class Result:
                returncode = 0
                stdout = "COMPOSITE\t0\tcoarse\tpoints=4\tshare=1.0000\n"
                stderr = ""

            return Result()

        original = render_door._run
        render_door._run = fake_run
        try:
            report, command = render_door.convert_frames(
                Path("rw_mpas_convert"), histories=None, mesh=None,
                scratch=tmp_path / "scratch", window="composite",
                field_set="full", simulation_start=None, nc_format="cdf2",
                compose=tmp_path / "sources.json")
        finally:
            render_door._run = original

        assert "--compose" in command
        assert "--history" not in command
        assert "--mesh" not in command
        assert report == {"frames": []}

    def test_base_only_rides_with_compose_and_not_alone(self, tmp_path):
        from woof.hex import render_door

        def fake_run(command, *, log_prefix):
            (tmp_path / "scratch").mkdir(parents=True, exist_ok=True)
            (tmp_path / "scratch" / "convert-report.json").write_text(
                '{"frames": []}', encoding="utf-8")

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

        original = render_door._run
        render_door._run = fake_run
        try:
            _, command = render_door.convert_frames(
                Path("rw_mpas_convert"), histories=None, mesh=None,
                scratch=tmp_path / "scratch", window="composite",
                field_set="full", simulation_start=None, nc_format="cdf2",
                compose=tmp_path / "sources.json", compose_base_only=True)
        finally:
            render_door._run = original
        assert "--compose-base-only" in command

    def test_incoherent_commands_are_refused_before_a_binary_is_needed(
            self, tmp_path):
        """Named refusals, reachable without an engine installed.

        Each names the concrete breakage: two source lists that can
        disagree, a subtraction with nothing to subtract from, and a
        window sized to regions that were never given.
        """

        import argparse

        from woof.hex.render_door import run_render

        cases = [
            (argparse.Namespace(
                compose="s.json", compose_base_only=False, history=["h.nc"],
                mesh=None, out=str(tmp_path / "png"), scratch=None,
                window="composite"),
             "two source lists"),
            (argparse.Namespace(
                compose=None, compose_base_only=True, history=["h.nc"],
                mesh="m.nc", out=str(tmp_path / "png"), scratch=None,
                window="mesh"),
             "no composite without --compose"),
            (argparse.Namespace(
                compose=None, compose_base_only=False, history=None,
                mesh=None, out=str(tmp_path / "png"), scratch=None,
                window="composite"),
             "no fine region to frame"),
            (argparse.Namespace(
                compose=None, compose_base_only=False, history=None,
                mesh=None, out=str(tmp_path / "png"), scratch=None,
                window="mesh"),
             "required unless --compose"),
        ]
        for namespace, expected in cases:
            with pytest.raises(RenderDoorError) as caught:
                run_render(namespace)
            assert expected in str(caught.value), (
                f"refusal for {namespace.window}/"
                f"compose={namespace.compose} did not name the breakage: "
                f"{caught.value}")


class TestNativeResolutionAndNames:
    """The live hex gallery's two defects, as the door now answers them.

    MEASURED 2026-10-10: a 15 km mesh drawn on 20 km and 28 km grids,
    headlined ``(d01 20 km)``, filed as ``rustwx_wrf_..._d01-20km_<p>.png``.
    """

    def test_global_means_the_mesh_spacing_and_the_old_box_keeps_its_name(self):
        from woof.hex.render_door import converter_window

        assert converter_window("global") == "global-mesh"
        assert converter_window("global-0.25") == "global"
        assert converter_window("mesh") == "mesh"
        spec = "region:40,-105,9000,7000,30,60"
        assert converter_window(spec) == spec
        with pytest.raises(RenderDoorError, match="not a render window"):
            converter_window("atlantic")

    def test_window_argument_takes_names_and_specs_only(self):
        import argparse

        from woof.hex.render_door import window_argument

        assert window_argument("global") == "global"
        assert window_argument("region:85,0,8400,8400") == "region:85,0,8400,8400"
        with pytest.raises(argparse.ArgumentTypeError, match="region:"):
            window_argument("lambert-ish")

    def test_names_carry_model_mesh_view_product_and_lead(self):
        from woof.hex.render_door import (
            default_view, grid_token, lead_token, product_file_name)

        assert grid_token("hex 15 km", "d01-20km") == "hex15km"
        assert grid_token("hex 60-3 km", "d01-3km") == "hex60-3km"
        assert grid_token(None, "d01-20km") == "d01-20km"
        assert lead_token(21600, "x.png") == "f006"
        assert lead_token(1800, "x.png") == "f000h30m"
        # Sub-minute histories must not overwrite f000 or each other.
        assert lead_token(30, "x.png") == "f000h00m30s"
        assert lead_token(90, "x.png") == "f000h01m30s"
        assert lead_token(None, "rustwx_wrf_20261010_0z_f012_d01_t2.png") == "f012"
        assert default_view("global") == "global"
        assert default_view("global-0.25") == "global"
        assert default_view("region:40,-105,9000,7000") == "region"
        assert product_file_name("hex15km", "global", "500mb_height_winds",
                                 "f006") == "hex15km_global_500mb_height_winds_f006.png"

    def test_a_view_that_would_split_a_path_is_refused(self):
        from woof.hex.render_door import checked_view

        assert checked_view("north-america") == "north-america"
        for bad in ("North America", "a/b", "", "-x"):
            with pytest.raises(RenderDoorError, match="--view"):
                checked_view(bad)

    def test_variables_keyword_is_a_group_not_a_missing_product(self):
        refuse_uncovered_products("all,variables", {}, Path("h.nc"), [])

    @pytest.mark.parametrize("source_label", [None, "Hex LES | WOOF physics"])
    def test_a_render_files_native_names_and_carries_georef(
            self, monkeypatch, tmp_path, source_label):
        """The door end to end over stand-in engines: what it asks the
        converter for, and where and under what name each picture lands."""

        import argparse
        import json

        from woof.hex import render_door

        history = tmp_path / "cuda-history.2026-10-10_06.00.00.nc"
        mesh = tmp_path / "mesh.nc"
        history.write_bytes(b"h")
        mesh.write_bytes(b"m")
        monkeypatch.setattr(render_door, "resolve_convert_exe",
                            lambda explicit=None: Path("rw_mpas_convert"))
        monkeypatch.setattr(render_door, "resolve_renderer_exe",
                            lambda explicit=None: Path("rw_wrfbatch"))
        monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)
        monkeypatch.setattr(render_door, "sha256_file", lambda path: "0" * 64)
        basemap = tmp_path / "basemap"
        (basemap / "natural_earth_110m").mkdir(parents=True)
        (basemap / "natural_earth_110m" / "ne_110m_coastline.shp").write_bytes(b"s")
        monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(basemap))
        seen: dict[str, list[str]] = {}
        environments: dict[str, dict | None] = {}

        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        def fake_run(command, *, log_prefix, env=None):
            command = [str(part) for part in command]
            seen[log_prefix] = command
            environments[log_prefix] = env
            result = Result()
            if log_prefix == "CONVERT":
                frames = Path(command[command.index("--out-dir") + 1])
                wrfout = frames / "wrfout_d01_2026-10-10_06_00_00"
                wrfout.write_bytes(b"w")
                Path(command[command.index("--json") + 1]).write_text(json.dumps({
                    "grid_label": "hex 15 km",
                    "frames": [{"output": str(wrfout),
                                "valid_time": "2026-10-10T06:00:00",
                                "lead_seconds": 21600,
                                "absent_wrf_fields": []}]}), encoding="utf-8")
            elif log_prefix == "CATALOG":
                result.stdout = ("PRODUCT\t500mb_height_winds\tdirect\t"
                                 "renderable\tready\t\n")
            elif log_prefix == "RENDER":
                raw = Path(command[command.index("--out-dir") + 1])
                png = raw / ("rustwx_wrf_20261010_0z_f006_hex15km_"
                             "500mb_height_winds.png")
                png.write_bytes(b"png")
                (raw / "render-georef.json").write_text(json.dumps({
                    "schema": "rustwx.render-georef/v1",
                    "panels": {png.name: {"projection": "latlon"}},
                    "without_georeference": []}), encoding="utf-8")
                result.stdout = f"RENDERED 500mb_height_winds {png}\n"
            return result

        monkeypatch.setattr(render_door, "_run", fake_run)
        out = tmp_path / "png"
        args = argparse.Namespace(
            history=[str(history)], mesh=str(mesh), init=None, compose=None,
            compose_base_only=False, out=str(out), scratch=None,
            products="500mb_height_winds", frames="all", window="global",
            field_set="full", simulation_start="2026-10-10_00:00:00",
            size="1600x1000", heavy=False, nc_format="cdf2",
            keep_scratch=False, convert_exe=None, renderer_exe=None,
            view=None, model_label="hex", source_label=source_label)
        assert render_door.run_render(args) == 0

        expected_source = source_label if source_label is not None else "Hex forecast"
        render = seen["RENDER"]
        assert render[render.index("--source-label") + 1] == expected_source

        convert = seen["CONVERT"]
        assert convert[convert.index("--window") + 1] == "global-mesh"
        assert convert[convert.index("--model-label") + 1] == "hex"
        filed = (out / "global" / "500mb_height_winds" / "2026-10-10"
                 / "hex15km_global_500mb_height_winds_f006.png")
        assert filed.read_bytes() == b"png"
        assert not list(out.rglob("rustwx_*"))
        georef = json.loads((out / "global" / "render-georef.json").read_text())
        assert list(georef["panels"]) == [
            "500mb_height_winds/2026-10-10/"
            "hex15km_global_500mb_height_winds_f006.png"]
        manifest = json.loads(
            (out / "global" / "render-manifest.json").read_text())
        assert manifest["frames"][0]["placed"] == [
            "global/500mb_height_winds/2026-10-10/"
            "hex15km_global_500mb_height_winds_f006.png"]
        assert manifest["view"] == "global"
        assert manifest["grid_label"] == "hex 15 km"
        assert manifest["source_label"] == expected_source
        # Both renderer calls are handed the basemap root by name, and the
        # receipt says which one was drawn from.
        for stage in ("CATALOG", "RENDER"):
            assert environments[stage]["RUSTWX_BASEMAP_DIR"] == str(basemap)
        assert manifest["engine"]["basemap_dir"] == str(basemap)

    def test_a_render_with_no_basemap_geometry_is_refused_before_converting(
            self, monkeypatch, tmp_path):
        """A relocated renderer drew 59 maps with no coastline, border or
        state line and reported every one RENDERED (2026-10-11).  With no
        Natural Earth set anywhere on the ladder the door refuses by name,
        and it does so before the converter is launched."""

        import argparse

        from woof.hex import engines, render_door

        history = tmp_path / "cuda-history.2026-10-10_06.00.00.nc"
        mesh = tmp_path / "mesh.nc"
        history.write_bytes(b"h")
        mesh.write_bytes(b"m")
        renderer = tmp_path / "bin" / "rw_wrfbatch"
        renderer.parent.mkdir()
        renderer.write_bytes(b"x")
        monkeypatch.setattr(render_door, "resolve_convert_exe",
                            lambda explicit=None: Path("rw_mpas_convert"))
        monkeypatch.setattr(render_door, "resolve_renderer_exe",
                            lambda explicit=None: renderer)
        monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)
        monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
        monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
        monkeypatch.setattr(engines, "USER_BRIDGE_DIR", tmp_path / "bridges")
        monkeypatch.setattr(render_door, "is_basemap_root", lambda path: False)
        launched: list[str] = []
        monkeypatch.setattr(
            render_door, "_run",
            lambda command, **kwargs: launched.append(kwargs["log_prefix"]))
        args = argparse.Namespace(
            history=[str(history)], mesh=str(mesh), init=None, compose=None,
            compose_base_only=False, out=str(tmp_path / "png"), scratch=None,
            products="500mb_height_winds", frames="all", window="global",
            field_set="full", simulation_start=None, size="1600x1000",
            heavy=False, nc_format="cdf2", keep_scratch=False,
            convert_exe=None, renderer_exe=None, view=None, model_label="hex")
        with pytest.raises(RenderDoorError, match="no basemap geometry"):
            render_door.run_render(args)
        assert launched == []

    def test_the_basemap_root_beside_a_built_renderer_is_found(
            self, monkeypatch, tmp_path):
        from woof.hex import engines, render_door

        monkeypatch.delenv("RUSTWX_BASEMAP_DIR", raising=False)
        monkeypatch.delenv("RUSTWX_ASSETS_DIR", raising=False)
        monkeypatch.setattr(engines, "USER_BRIDGE_DIR", tmp_path / "bridges")
        renderer = tmp_path / "rustwx" / "target" / "release" / "rw_wrfbatch"
        renderer.parent.mkdir(parents=True)
        renderer.write_bytes(b"x")
        root = tmp_path / "rustwx" / "assets" / "basemap"
        (root / "natural_earth_10m").mkdir(parents=True)
        (root / "natural_earth_10m" / "ne_10m_coastline.shp").write_bytes(b"s")
        assert render_door.resolve_basemap_dir(renderer) == root
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(RenderDoorError, match="--basemap-dir"):
            render_door.resolve_basemap_dir(renderer, str(elsewhere))


def test_region_latlon_is_a_window_the_export_door_passes_to_the_converter():
    """The WOOF parent route exports on region-latlon:, the regular lat-lon box the engine's mapped
    NetCDF route reads; the door must pass it through, not refuse it as an unknown window."""
    from woof.hex import render_door

    spec = "region-latlon:36,-91,2400,1800"
    assert render_door.window_argument(spec) == spec
    assert render_door.converter_window(spec) == spec
    assert render_door.default_view(spec) == "region"


def test_the_arwen_backend_publishes_sea_ice_and_snow_when_the_land_surface_holds_them():
    from woof.hex import cuda_arwen_physics_v841 as backend

    assert {"xice", "snow", "snowh"} <= set(backend._OPTIONAL_ARWEN_EXPORT_FIELDS)


def _fake_converter(rd, monkeypatch, tmp_path, *, accepts_flag, diag_named=None):
    """A converter stand-in: records its command and writes the receipt a real one would."""
    import json as _json
    from types import SimpleNamespace

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        history = Path(command[command.index('--history') + 1])
        named = diag_named(history) if diag_named else history.with_name(
            history.name.replace('history', 'diag'))
        frame = {"output": "frame.nc", "split_streams": {"diag": str(named), "diag_sha256": None,
                                                         "diag_fields": ["REFC", "T2"]}}
        scratch = Path(command[command.index('--json') + 1]).parent
        (scratch / 'convert-report.json').write_text(_json.dumps({"frames": [frame]}))
        return SimpleNamespace(stdout='', stderr='', returncode=0)

    monkeypatch.setattr(rd, '_run', fake_run)
    monkeypatch.setattr(rd, 'engine_accepts', lambda exe, flag: accepts_flag and flag.startswith('--diag'))
    return seen


def test_render_diagnostics_reach_the_shipped_converter_by_pairing_not_by_flag(tmp_path, monkeypatch):
    """THE BREAKAGE (RRFS bigmesh maps, 2026-10-28): `woof hex render --diagnostics FILE` exited 2,
    "rw_mpas_convert: unknown flag --diagnostics": the converter woof ships reads a history's diag
    stream file from beside it (history.T.nc -> diag.T.nc) and has no such flag.  The door now pairs
    the named file under that name and checks the converter's receipt names it back."""
    from woof.hex import render_door as rd

    run = tmp_path / 'run'
    run.mkdir()
    history = run / 'history.2023-03-31_18.00.00.nc'
    history.write_bytes(b'history-state')
    elsewhere = tmp_path / 'native-diagnostics.nc'
    elsewhere.write_bytes(b'valid-time-diagnostics')
    seen = _fake_converter(rd, monkeypatch, tmp_path, accepts_flag=False)
    report, cmd = rd.convert_frames(Path('convert'), histories=[history], mesh=tmp_path / 'grid.nc',
                                    scratch=tmp_path / 'scratch', window='mesh', field_set='full',
                                    simulation_start=None, nc_format='cdf2', init=None,
                                    diagnostics=elsewhere)
    assert '--diagnostics' not in cmd and '--diagnostics-sha256' not in cmd
    converted = Path(seen['command'][seen['command'].index('--history') + 1])
    assert converted.parent == tmp_path / 'scratch' / 'diagnostics-pair'
    assert converted.name == history.name
    paired = converted.with_name('diag.2023-03-31_18.00.00.nc')
    assert paired.read_bytes() == b'valid-time-diagnostics'
    assert converted.read_bytes() == b'history-state'
    record = report['diagnostics']
    assert record['read'] is True and record['fields'] == ['REFC', 'T2']
    assert record['sha256'] == rd.sha256_file(elsewhere)


def test_render_diagnostics_beside_the_history_convert_in_place(tmp_path, monkeypatch):
    from woof.hex import render_door as rd

    history = tmp_path / 'history.2023-03-31_18.00.00.nc'
    history.write_bytes(b'h')
    beside = tmp_path / 'diag.2023-03-31_18.00.00.nc'
    beside.write_bytes(b'd')
    seen = _fake_converter(rd, monkeypatch, tmp_path, accepts_flag=False)
    report, _ = rd.convert_frames(Path('convert'), histories=[history], mesh=tmp_path / 'grid.nc',
                                  scratch=tmp_path / 'scratch', window='mesh', field_set='full',
                                  simulation_start=None, nc_format='cdf2', diagnostics=beside)
    assert Path(seen['command'][seen['command'].index('--history') + 1]) == history
    assert report['diagnostics']['how'] == 'beside'
    assert not (tmp_path / 'scratch' / 'diagnostics-pair').exists()


def test_render_diagnostics_refusals_name_their_breakage(tmp_path, monkeypatch):
    from woof.hex import render_door as rd

    history = tmp_path / 'history.2023-03-31_18.00.00.nc'
    history.write_bytes(b'h')
    other_time = tmp_path / 'diag.2023-03-31_17.00.00.nc'
    other_time.write_bytes(b'd')
    _fake_converter(rd, monkeypatch, tmp_path, accepts_flag=False)
    kwargs = dict(histories=[history], mesh=tmp_path / 'grid.nc', scratch=tmp_path / 'scratch',
                  window='mesh', field_set='full', simulation_start=None, nc_format='cdf2',
                  diagnostics=other_time)
    with pytest.raises(rd.RenderDoorError, match='two\\s+valid times as one'):
        rd.convert_frames(Path('convert'), **kwargs)
    for bad in ({'compose': tmp_path / 'compose.json'}, {'histories': []},
                {'histories': [Path('a'), Path('b')]}):
        with pytest.raises(rd.RenderDoorError, match='requires one history'):
            rd.convert_frames(Path('convert'), **{**kwargs, **bad})
    unnamed = tmp_path / 'out.nc'
    unnamed.write_bytes(b'h')
    with pytest.raises(rd.RenderDoorError, match="no 'history' token"):
        rd.convert_frames(Path('convert'), **{**kwargs, 'histories': [unnamed],
                                              'diagnostics': tmp_path / 'diag.2023-03-31_17.00.00.nc'})


def test_a_frame_completed_from_another_diag_file_is_refused(tmp_path, monkeypatch):
    from woof.hex import render_door as rd

    history = tmp_path / 'history.2023-03-31_18.00.00.nc'
    history.write_bytes(b'h')
    given = tmp_path / 'given-diagnostics.nc'
    given.write_bytes(b'd')
    stray = tmp_path / 'stray-diag.nc'
    stray.write_bytes(b'x')
    _fake_converter(rd, monkeypatch, tmp_path, accepts_flag=False, diag_named=lambda h: stray)
    with pytest.raises(rd.RenderDoorError, match='not drawn from fields of another file'):
        rd.convert_frames(Path('convert'), histories=[history], mesh=tmp_path / 'grid.nc',
                          scratch=tmp_path / 'scratch', window='mesh', field_set='full',
                          simulation_start=None, nc_format='cdf2', diagnostics=given)


def test_a_converter_that_takes_the_diagnostics_flag_still_gets_it_pinned(tmp_path, monkeypatch):
    from woof.hex import render_door as rd
    import hashlib

    init = tmp_path / 'init.nc'
    diag = tmp_path / 'diagnostics.nc'
    init.write_bytes(b'init-coordinates')
    diag.write_bytes(b'valid-time-diagnostics')
    history = tmp_path / 'history.nc'
    history.write_bytes(b'h')
    seen = _fake_converter(rd, monkeypatch, tmp_path, accepts_flag=True)
    kwargs = dict(histories=[history], mesh=tmp_path / 'grid.nc', scratch=tmp_path / 'scratch',
                  window='mesh', field_set='full', simulation_start=None, nc_format='cdf2',
                  init=init, diagnostics=diag)
    _, cmd = rd.convert_frames(Path('convert'), **kwargs)
    for flag, path in (('--init-sha256', init), ('--diagnostics-sha256', diag)):
        assert cmd[cmd.index(flag) + 1] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(rd.RenderDoorError, match='requires one history, --init'):
        rd.convert_frames(Path('convert'), **{**kwargs, 'init': None})


RRFS_V2_GRID = {"kind": "lambert", "centre_lat": 38.5, "centre_lon": -97.5, "dx_metres": 3000.0,
                "nx": 1800, "ny": 1060, "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5,
                "earth_radius_m": 6370000.0}


def _spec(window: str) -> dict:
    values = window.split(':', 1)[1].split(',')
    return {"kind": "lambert", "centre_lat": float(values[0]), "centre_lon": float(values[1]),
            "dx_metres": float(values[2]), "nx": int(values[3]), "ny": int(values[4]),
            "truelat1": float(values[5]), "truelat2": float(values[6]), "stand_lon": float(values[7])}


def test_regional_tiles_are_exact_sub_grids_of_a_conus_window():
    """RRFS v2's 3 km Lambert grid (1800 x 1060) is wider than its picture, so the door draws it
    whole plus six regional cuts; every cut lands on the full window's own grid points."""
    from woof.hex import render_door as rd

    assert rd.lambert_ijll(RRFS_V2_GRID, 900.5, 530.5) == pytest.approx((38.5, -97.5), abs=1e-12)
    plan, why = rd.plan_tiles(RRFS_V2_GRID, 'auto', 1400, 1000)
    assert [cut['name'] for cut in plan] == ['nw', 'nc', 'ne', 'sw', 'sc', 'se']
    assert 'wider than' in why
    for cut in plan:
        tile = _spec(cut['window'])
        assert (tile['nx'], tile['ny']) == (cut['i'][1] - cut['i'][0] + 1, cut['j'][1] - cut['j'][0] + 1)
        for i, j in ((1, 1), (tile['nx'], tile['ny']), (1, tile['ny']), (tile['nx'], 1)):
            assert rd.lambert_ijll(tile, i, j) == pytest.approx(
                rd.lambert_ijll(RRFS_V2_GRID, cut['i'][0] + i - 1, cut['j'][0] + j - 1), abs=1e-9)
    columns = {cut['name']: cut['i'] for cut in plan}
    rows = {cut['name']: cut['j'] for cut in plan}
    assert columns['nw'][0] == 1 and columns['ne'][1] == 1800 and rows['sw'][0] == 1 and rows['nw'][1] == 1060
    # neighbours overlap, so a feature on a seam is whole in one of them
    assert columns['nw'][1] > columns['nc'][0] and columns['nc'][1] > columns['ne'][0]
    assert rows['sw'][1] > rows['nw'][0]


def test_tiles_auto_draws_a_window_that_fits_its_picture_whole():
    from woof.hex import render_door as rd

    small = dict(RRFS_V2_GRID, nx=478, ny=502)
    assert rd.plan_tiles(small, 'auto', 1200, 900)[0] == []
    assert len(rd.plan_tiles(small, 'regional', 1200, 900)[0]) == 6
    assert rd.plan_tiles(small, 'none', 1200, 900)[0] == []
    globe = {"kind": "latlon", "south": -90.0, "north": 90.0, "west": -180.0, "east": 180.0,
             "spacing_degrees": 0.25}
    assert rd.plan_tiles(globe, 'auto', 1800, 1000)[0] == []
    cuts = {cut['name']: cut['window'] for cut in rd.plan_tiles(globe, 'regional', 1800, 1000)[0]}
    assert cuts['ne'] == 'latlon:-9.0,90.0,36.0,180.0,0.25'
    with pytest.raises(rd.RenderDoorError, match='window grid'):
        rd.plan_tiles(None, 'regional', 1200, 900)
    assert rd.plan_tiles(None, 'auto', 1200, 900)[0] == []


def test_tiles_file_as_product_subfolders(tmp_path):
    from woof.hex.render_door import place_path

    assert place_path(tmp_path, "mesh", "mslp_10m_winds", "2023-03-31", "x.png", tile="ne") == \
        tmp_path / "mesh" / "mslp_10m_winds" / "ne" / "2023-03-31" / "x.png"
    assert place_path(tmp_path, "mesh", "mslp_10m_winds", "2023-03-31", "x.png") == \
        tmp_path / "mesh" / "mslp_10m_winds" / "2023-03-31" / "x.png"


def test_the_case_index_lists_each_tile_on_its_own(tmp_path):
    from woof.hex.products_door import index_tree

    for tile in ("full", "ne"):
        folder = tmp_path / "refined-core" / "mslp_10m_winds" / tile / "2023-03-31"
        folder.mkdir(parents=True)
        for lead in ("f003", "f006"):
            token = "refined-core" if tile == "full" else f"refined-core-{tile}"
            (folder / f"hex3km_{token}_mslp_10m_winds_{lead}.png").write_bytes(b"png")
    tree = index_tree(tmp_path)["refined-core"]
    assert sorted(tree) == ["mslp_10m_winds", "mslp_10m_winds/ne"]
    assert [row[0] for row in tree["mslp_10m_winds/ne"]] == ["f003", "f006"]
    assert all("/ne/" in row[1] for row in tree["mslp_10m_winds/ne"])


def test_an_open_ocean_window_gets_a_labelled_graticule():
    """The AL10 storm-following box (GFS 2026-10-22 18Z) sat over open ocean
    and its maps carried no geography at all; every hex map now carries
    latitude/longitude lines and labels inside the frame."""

    import numpy as np
    from woof.hex.render_door import graticule_overlays

    lat, lon = np.meshgrid(np.linspace(38.0, 42.5, 160), np.linspace(-69.0, -63.0, 160), indexing="ij")
    overlays = graticule_overlays(lat, lon)
    texts = [label["text"] for label in overlays["labels"]]
    assert overlays["step_deg"] == 1.0
    assert "40\u00b0N" in texts and "66\u00b0W" in texts
    for label in overlays["labels"]:
        assert 38.0 <= label["lat"] <= 42.5 and -69.0 <= label["lon"] <= -63.0
    # Across the antimeridian the lines stay on one side of the seam.
    lat, lon = np.meshgrid(np.linspace(40.0, 45.0, 80), np.linspace(178.0, 184.0, 80), indexing="ij")
    overlays = graticule_overlays(lat, ((lon + 180.0) % 360.0) - 180.0)
    assert {"180\u00b0", "179\u00b0E", "179\u00b0W"} <= {label["text"] for label in overlays["labels"]}
    # The whole globe is drawn every 30 degrees.
    lat, lon = np.meshgrid(np.linspace(-89.5, 89.5, 90), np.linspace(-179.5, 179.5, 180), indexing="ij")
    assert graticule_overlays(lat, lon)["step_deg"] == 30.0


#: The model label the hex 0.4.10 gate's MPAS-route real run passes (n4-realrun-0410.sh).
GATE_MPAS_LABEL = "woof hex 0.4.10 MPAS physics (default)"


def test_a_long_model_label_is_fitted_at_a_word_and_said_so(tmp_path, monkeypatch, capsys):
    """THE BREAKAGE (hex 0.4.10 gate maps, 2026-10-29): the door sliced the model label at 32
    characters with no word said, so every MPAS-route map read 'woof hex 0.4.10 MPAS physics (d'
    and every V8XB conus3km map '... bec7ce68: RRF'.  A longer label now ends on its last whole
    word that fits with '...' after it (the converter refuses more than 32), and the door says so."""
    from types import SimpleNamespace

    from woof.hex import render_door as rd

    assert rd.fitted_model_label("hex") == "hex"
    assert rd.fitted_model_label("  WOOF   Hex ") == "WOOF Hex"
    assert rd.fitted_model_label("x" * 32) == "x" * 32
    assert rd.fitted_model_label(GATE_MPAS_LABEL) == "woof hex 0.4.10 MPAS physics..."
    assert rd.fitted_model_label("hex 0.4.10 RC HB18 bec7ce68: RRFS hrrrv5") == "hex 0.4.10 RC HB18 bec7ce68..."
    assert rd.fitted_model_label("y" * 40) == "y" * 29 + "..."
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = [str(part) for part in command]
        Path(command[command.index("--json") + 1]).write_text('{"frames": []}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rd, "_run", fake_run)
    rd.convert_frames(Path("rw_mpas_convert"), histories=[tmp_path / "history.nc"], mesh=tmp_path / "grid.nc",
                      scratch=tmp_path / "scratch", window="mesh", field_set="full", simulation_start=None,
                      nc_format="cdf2", model_label=GATE_MPAS_LABEL)
    drawn = seen["command"][seen["command"].index("--model-label") + 1]
    assert drawn == "woof hex 0.4.10 MPAS physics..."
    assert len(drawn) <= rd.MODEL_LABEL_MAX_CHARS
    assert f"{GATE_MPAS_LABEL!r} -> {drawn!r}" in capsys.readouterr().out


def test_the_filename_label_does_not_rename_the_model_on_the_map(tmp_path, monkeypatch):
    """--label leads the delivered filenames (hexmoving_<view>_<product>_f001.png); it used to also
    replace --model-label on every map, so a caller naming its files 'hexmoving' got maps that named
    the model 'hexmoving'."""
    import argparse

    from woof.hex import render_door as rd

    history = tmp_path / "history.2026-10-24_13.00.00.nc"
    mesh = tmp_path / "mesh.nc"
    history.write_bytes(b"h")
    mesh.write_bytes(b"m")
    basemap = tmp_path / "basemap"
    (basemap / "natural_earth_110m").mkdir(parents=True)
    (basemap / "natural_earth_110m" / "ne_110m_coastline.shp").write_bytes(b"s")
    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(basemap))
    monkeypatch.setattr(rd, "resolve_convert_exe", lambda explicit=None: Path("rw_mpas_convert"))
    monkeypatch.setattr(rd, "resolve_renderer_exe", lambda explicit=None: Path("rw_wrfbatch"))
    monkeypatch.setattr(rd, "engine_accepts", lambda exe, flag: True)
    asked = {}

    class Stop(Exception):
        pass

    def fake_convert(*args, **kwargs):
        asked.update(kwargs)
        raise Stop

    monkeypatch.setattr(rd, "convert_frames", fake_convert)
    args = argparse.Namespace(
        history=[str(history)], mesh=str(mesh), init=None, compose=None, compose_base_only=False,
        out=str(tmp_path / "png"), scratch=None, products="composite_reflectivity", frames="all",
        window="global", field_set="full", simulation_start=None, size="1400x1000", heavy=False,
        nc_format="cdf2", keep_scratch=False, convert_exe=None, renderer_exe=None, view=None,
        label="hexmoving", model_label="WOOF Hex", source_label=None)
    with pytest.raises(Stop):
        rd.run_render(args)
    assert asked["model_label"] == "WOOF Hex"


def test_a_view_label_reaches_only_a_converter_that_takes_it(tmp_path, monkeypatch, capsys):
    """THE BREAKAGE (2026-10-30): the site menu read 'Core zoom · 8 → 38 km' over a picture
    headlined '(hex 133-6.9 km) ... Δx 8.063 km' (run_713c9e1913164e02).  The door passes the
    view's headline as rw_mpas_convert --grid-label; a converter built before the flag refuses
    unknown flags, so it keeps the mesh's headline, said once, and the view still draws."""
    from types import SimpleNamespace

    from woof.hex import render_door as rd

    monkeypatch.setattr(rd, "_GRID_LABEL_UNSUPPORTED", set())
    monkeypatch.setattr(rd, "engine_accepts", lambda exe, flag: True)
    assert rd.drawn_grid_label(Path("new/rw_mpas_convert"), "Core zoom {spacing}") == "Core zoom {spacing}"
    assert rd.drawn_grid_label(Path("new/rw_mpas_convert"), None) is None
    monkeypatch.setattr(rd, "engine_accepts", lambda exe, flag: flag != "--grid-label")
    for _ in range(2):
        assert rd.drawn_grid_label(Path("old/rw_mpas_convert"), "Core zoom {spacing}") is None
    assert capsys.readouterr().out.count("predates --grid-label") == 1

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = [str(part) for part in command]
        Path(command[command.index("--json") + 1]).write_text('{"frames": []}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rd, "_run", fake_run)
    common = dict(histories=[tmp_path / "history.nc"], mesh=tmp_path / "grid.nc", scratch=tmp_path / "scratch",
                  window="mesh", field_set="full", simulation_start=None, nc_format="cdf2", model_label="hex")
    rd.convert_frames(Path("rw_mpas_convert"), grid_label="Core zoom {spacing}", **common)
    assert seen["command"][seen["command"].index("--grid-label") + 1] == "Core zoom {spacing}"
    rd.convert_frames(Path("rw_mpas_convert"), **common)
    assert "--grid-label" not in seen["command"]


def test_a_composite_with_a_view_label_is_refused_by_name(tmp_path, monkeypatch):
    """A --compose frame switches meshes point by point and the converter writes it no grid label,
    so a label asked for there would be dropped without a word."""
    import argparse

    from woof.hex import render_door as rd

    monkeypatch.setattr(rd, "resolve_convert_exe", lambda explicit=None: Path("rw_mpas_convert"))
    monkeypatch.setattr(rd, "resolve_renderer_exe", lambda explicit=None: Path("rw_wrfbatch"))
    monkeypatch.setattr(rd, "engine_accepts", lambda exe, flag: True)
    args = argparse.Namespace(
        history=None, mesh=None, init=None, compose=str(tmp_path / "sources.json"), compose_base_only=False,
        out=str(tmp_path / "png"), scratch=None, products="all", frames="all", window="composite",
        field_set="full", simulation_start=None, size="1400x1000", heavy=False, nc_format="cdf2",
        keep_scratch=False, convert_exe=None, renderer_exe=None, view=None, model_label="hex",
        source_label=None, grid_label="Composite {spacing}")
    with pytest.raises(RenderDoorError, match="--grid-label"):
        rd.run_render(args)


def test_a_later_pass_reuses_the_frames_the_live_draws_converted(monkeypatch, tmp_path):
    """An Arizona zoom globe (woof hex 0.4.10, 2026-10-30) and its RTX 5090 replay: after the forecast every view
    converted every frame again for its whole-run maps, the live-drawn ones included (70 of the pass's 110 s)."""
    import shutil

    from woof.hex import render_door

    histories = []
    for hour in range(3):
        path = tmp_path / f"history.2026-10-30_{hour:02d}.00.00.nc"
        path.write_bytes(b"h" * (hour + 1))
        histories.append(path)
    converted: list[list[str]] = []

    def fake_convert(exe, *, histories, scratch, **kwargs):
        converted.append([Path(h).name for h in histories])
        frames = Path(scratch) / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        out = []
        for h in histories:
            wrfout = frames / ("wrfout_d01_" + Path(h).name[8:21])
            wrfout.write_bytes(b"w")
            out.append({"output": str(wrfout), "valid_time": Path(h).name[8:21]})
        return {"window_spec": {"kind": "latlon"}, "grid_label": "g", "frames": out}, ["convert"]

    monkeypatch.setattr(render_door, "convert_frames", fake_convert)
    monkeypatch.setattr(render_door, "sha256_file", lambda path: "0" * 64)
    # A roomy cache disk: the cache keeps 20 GB and 10 % of its disk free (FRAME_CACHE_FREE_*), which a test host's
    # temporary disk may not have (a 16 GB tmpfs, or a big drive near 90 % full), and then nothing is cached.
    from types import SimpleNamespace
    monkeypatch.setattr(render_door.shutil, "disk_usage",
                        lambda path: SimpleNamespace(total=1000 * 2**30, used=100 * 2**30, free=900 * 2**30))
    cache = tmp_path / "cache" / "global"
    common = dict(mesh=tmp_path / "m.nc", init=None, window="global-mesh", field_set="full", simulation_start="s",
                  nc_format="cdf5", model_label="hex", remap="product", jobs=1, explicit_jobs=False)
    # two live draws, one frame each, in their own scratch (removed after each draw)
    for k in range(2):
        report, _ = render_door.convert_frames_cached(cache, Path("conv"), histories=[histories[k]],
                                                      scratch=tmp_path / f"live-{k}", **common)
        assert len(report["frames"]) == 1
        shutil.rmtree(tmp_path / f"live-{k}")
    # the full pass: only the frame no live draw took is converted; the series is whole and in order
    report, _ = render_door.convert_frames_cached(cache, Path("conv"), histories=histories,
                                                  scratch=tmp_path / "full", **common)
    assert converted == [[histories[0].name], [histories[1].name], [histories[2].name]]
    assert [f["valid_time"] for f in report["frames"]] == [h.name[8:21] for h in histories]
    assert all(Path(f["output"]).is_file() for f in report["frames"])
    assert report["window_spec"] == {"kind": "latlon"}
    # a frame the forecast rewrote is converted again, and another window is its own cache
    histories[0].write_bytes(b"rewritten")
    render_door.convert_frames_cached(cache, Path("conv"), histories=histories[:1], scratch=tmp_path / "again", **common)
    assert converted[-1] == [histories[0].name]
    render_door.convert_frames_cached(cache, Path("conv"), histories=histories[1:2], scratch=tmp_path / "other",
                                      **{**common, "window": "latlon:-90,90,-180,179.8,0.2"})
    assert converted[-1] == [histories[1].name]


def test_the_global_window_has_no_more_columns_than_its_picture(monkeypatch):
    """global_picture_window: owner canary run_b8d705f1f8db9c92 (48 h 96 -> 3 km zoom globe, 2026-10-30) rendered its
    global view from the mesh's 3 km median spacing, 4240 x 2121 = 9.0 M points for an 1800-pixel picture, and the
    renderer was killed for host memory at 85-96 GB. The picture's grid is one column per pixel; a mesh coarser than
    that keeps its own spacing."""
    from woof.hex import render_door

    spacing = {"m": 3_015.0}
    monkeypatch.setattr(render_door, "mesh_median_spacing_m", lambda mesh: spacing["m"])
    window, why = render_door.global_picture_window(Path("zoom.grid.nc"), 1800)
    assert window == "latlon:-90,90,-180,179.8,0.2"
    assert "1800 x 901" in why
    # One column per pixel, never past the converter's explicit-window ceiling.
    assert render_door.global_picture_window(Path("zoom.grid.nc"), 4000)[0] == "latlon:-90,90,-180,179.82,0.18"
    spacing["m"] = 15_000.0  # a uniform 15 km globe: 2,668 columns asked of 1,800 pixels
    assert render_door.global_picture_window(Path("g15.grid.nc"), 1800)[0] == "latlon:-90,90,-180,179.8,0.2"
    spacing["m"] = 30_000.0  # 1,334 columns: the mesh's own spacing, as before
    assert render_door.global_picture_window(Path("g30.grid.nc"), 1800) is None
    spacing["m"] = None  # a mesh the decoder cannot read: the converter decides
    assert render_door.global_picture_window(Path("x.grid.nc"), 1800) is None

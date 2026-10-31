"""The start data's credit on every hex map, composed per map by the render door (render_door.SOURCE_CREDIT_ENV).

THE BREAKAGE THESE PREVENT (WOOF, 2026-10-29): the host printed ECMWF's CC BY 4.0 credit through one render theme for
the whole run, whose source line replaced every map's own: each hex ensemble member's maps lost their member number,
and a line longer than the renderer's source slot (at most half the subtitle row) was cut with "..." at 1200 x 900.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from woof.hex import render_door
from woof.hex.render_door import credited_source_line, source_credit_theme, source_line_budget

ECMWF = "© 2026 ECMWF, CC BY 4.0, modified"
ECCC = "Data Source: Environment and Climate Change Canada"
MEMBER = "Hex forecast, member mem003 of 36"
SIZES = [(1800, 1000), (1400, 1000), (1200, 900), (1000, 750), (700, 525)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("credit", [ECMWF, ECCC])
@pytest.mark.parametrize("label", ["Hex forecast", MEMBER, None])
def test_the_credit_is_whole_and_the_line_fits_its_canvas(size, credit, label):
    line = credited_source_line(label, credit, *size)
    assert line.endswith(credit)
    # Only the credit alone may pass the budget: a licence's words are never shortened.
    assert len(line) <= source_line_budget(*size) or line == credit


def test_a_member_map_keeps_its_member_number_on_every_canvas_the_doors_draw():
    # The products door's views: the globe at 1800 x 1000, every regional view at 1400 x 1000.
    assert credited_source_line(MEMBER, ECMWF, 1800, 1000) == f"source: Hex mem003 of 36 · {ECMWF}"
    assert credited_source_line(MEMBER, ECMWF, 1400, 1000) == f"Hex mem003 of 36 · {ECMWF}"
    # `woof hex render`'s own 1200 x 900 default keeps 40 characters with both subtitle halves whole: the member
    # line alone (41) already cuts the left half there, so the credit stands alone.
    assert credited_source_line(MEMBER, ECMWF, 1200, 900) == ECMWF


def test_a_plain_map_keeps_its_whole_line_where_it_fits():
    assert credited_source_line("Hex forecast", ECMWF, 1400, 1000) == f"source: Hex forecast · {ECMWF}"
    assert credited_source_line("Hex forecast", ECMWF, 1200, 900) == f"Hex · {ECMWF}"


def test_no_credit_writes_no_theme_and_a_host_theme_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.delenv(render_door.SOURCE_CREDIT_ENV, raising=False)
    monkeypatch.delenv("RUSTWX_THEME", raising=False)
    assert source_credit_theme(tmp_path, MEMBER, 1400, 1000) is None
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv(render_door.SOURCE_CREDIT_ENV, f"  {ECMWF}\n")
    path = Path(source_credit_theme(tmp_path, MEMBER, 1400, 1000))
    # No `extends`: the renderer keeps its plain look and only the source line changes.
    assert json.loads(path.read_text(encoding="utf-8")) == {"text": {"source_label": f"Hex mem003 of 36 · {ECMWF}"}}
    monkeypatch.setenv("RUSTWX_THEME", str(tmp_path / "host-theme.json"))
    assert source_credit_theme(tmp_path / "other", MEMBER, 1400, 1000) is None


@pytest.mark.parametrize("credit", [None, ECMWF])
def test_the_render_door_hands_the_renderer_its_own_maps_credited_line(tmp_path, monkeypatch, credit):
    """Over stand-in engines: with a credit named, the renderer's environment names a theme carrying this map's own
    line (the member number) and the credit; with none, the environment carries no theme, as before."""

    history = tmp_path / "cuda-history.2026-10-10_06.00.00.nc"
    mesh = tmp_path / "mesh.nc"
    history.write_bytes(b"h")
    mesh.write_bytes(b"m")
    monkeypatch.setattr(render_door, "resolve_convert_exe", lambda explicit=None: Path("rw_mpas_convert"))
    monkeypatch.setattr(render_door, "resolve_renderer_exe", lambda explicit=None: Path("rw_wrfbatch"))
    monkeypatch.setattr(render_door, "engine_accepts", lambda exe, flag: True)
    monkeypatch.setattr(render_door, "sha256_file", lambda path: "0" * 64)
    basemap = tmp_path / "basemap"
    (basemap / "natural_earth_110m").mkdir(parents=True)
    (basemap / "natural_earth_110m" / "ne_110m_coastline.shp").write_bytes(b"s")
    monkeypatch.setenv("RUSTWX_BASEMAP_DIR", str(basemap))
    monkeypatch.delenv("RUSTWX_THEME", raising=False)
    if credit:
        monkeypatch.setenv(render_door.SOURCE_CREDIT_ENV, credit)
    else:
        monkeypatch.delenv(render_door.SOURCE_CREDIT_ENV, raising=False)
    environments: dict[str, dict | None] = {}
    themes: dict[str, dict] = {}

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(command, *, log_prefix, env=None):
        command = [str(part) for part in command]
        environments[log_prefix] = env
        if env and env.get("RUSTWX_THEME"):
            themes[log_prefix] = json.loads(Path(env["RUSTWX_THEME"]).read_text(encoding="utf-8"))
        result = Result()
        if log_prefix == "CONVERT":
            frames = Path(command[command.index("--out-dir") + 1])
            wrfout = frames / "wrfout_d01_2026-10-10_06_00_00"
            wrfout.write_bytes(b"w")
            Path(command[command.index("--json") + 1]).write_text(json.dumps({
                "grid_label": "hex 121 km",
                "frames": [{"output": str(wrfout), "valid_time": "2026-10-10T06:00:00", "lead_seconds": 21600,
                            "absent_wrf_fields": []}]}), encoding="utf-8")
        elif log_prefix == "CATALOG":
            result.stdout = "PRODUCT\t2m_temperature\tdirect\trenderable\tready\t\n"
        elif log_prefix == "RENDER":
            raw = Path(command[command.index("--out-dir") + 1])
            png = raw / "rustwx_wrf_20261010_0z_f006_hex121km_2m_temperature.png"
            png.write_bytes(b"png")
            (raw / "render-georef.json").write_text(json.dumps({
                "schema": "rustwx.render-georef/v1", "panels": {png.name: {"projection": "latlon"}},
                "without_georeference": []}), encoding="utf-8")
            result.stdout = f"RENDERED 2m_temperature {png}\n"
        return result

    monkeypatch.setattr(render_door, "_run", fake_run)
    args = argparse.Namespace(
        history=[str(history)], mesh=str(mesh), init=None, compose=None, compose_base_only=False,
        out=str(tmp_path / "png"), scratch=None, products="2m_temperature", frames="all", window="global",
        field_set="full", simulation_start="2026-10-10_00:00:00", size="1400x1000", heavy=False, nc_format="cdf2",
        keep_scratch=False, convert_exe=None, renderer_exe=None, view=None, model_label="hex", source_label=MEMBER)
    assert render_door.run_render(args) == 0
    if credit:
        assert themes["RENDER"] == {"text": {"source_label": f"Hex mem003 of 36 · {ECMWF}"}}
    else:
        assert "RUSTWX_THEME" not in environments["RENDER"]

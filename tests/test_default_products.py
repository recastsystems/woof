"""A local run's default product set draws what it asks for, and says what it did not.

Each test names the concrete breakage it prevents, measured on real
WOOF-ArWen runs of the 2.8 line:

* the default ``general`` preset asked every run for 10m_wind_gusts,
  precipitation_type and cloud_cover; no wrfout carries their fields, so
  every run drew 20 of 24 pictures, and the catalog a local run was
  offered listed every product of every model;
* the TUI's ``all`` drew 137 of an 18 h run's 204 folders as raw
  variables named with a hash, and asked it for 41 windows that close at
  F024 or F048 (805 skipped attempts) with advice about HRRR cycles;
* every skip line of a 19-frame series named the LAST file;
* the end-of-run note named ``qpf_1h`` (24 pictures) and never the three
  products that drew none.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

import woof.cli as cli
from woof import go_cli, render_receipts, runplan, rustwx, tui_products
from woof.render_layout import NESTED

#: The three products the default preset used to ask for when no local run
#: could draw them (a7b32a707 took them out of it): no wrfout import wrote a
#: gust, a precipitation category or a total cloud fraction.  The preset and
#: closing-note cells below still use them as that run's undrawn three.
PRESET_DROPPED = ("10m_wind_gusts", "precipitation_type", "cloud_cover")

#: What the real renderer's wrfout lane says today.  90ae5edc9 taught it to
#: derive two of the three itself (rw-wrfbatch wrf_process.rs: the UPP
#: CALGUST 10 m gust and the categorical precipitation type), so a local
#: run draws them from hour 0; a total cloud fraction is still never on a
#: wrfout.
NEVER_ON_WRFOUT = ("cloud_cover",)
DERIVED_ON_WRFOUT = ("10m_wind_gusts", "precipitation_type")


# ---------------------------------------------------------------------------
# The presets
# ---------------------------------------------------------------------------


def test_the_default_preset_asks_for_nothing_a_local_run_cannot_draw():
    document = tui_products.presets()
    general = next(row for row in document["presets"]
                   if row["id"] == document["default"])
    for slug in PRESET_DROPPED:
        assert slug not in general["products"], slug
    # The cloud panel a wrfout DOES carry: low, middle and high layers.
    assert "cloud_cover_levels" in general["products"]
    count = len(general["products"])
    assert general["description"].startswith(f"{count} "), general


def test_no_preset_names_a_product_the_lane_record_says_it_cannot_draw():
    """A product that stays in a preset must draw.

    The lane record's ``unavailable`` map is this lane's statement of what
    it cannot draw and why; a preset naming one of those promised a
    picture no run would ever contain.
    """

    unavailable = tui_products.lane_capabilities()["unavailable"]
    for preset in tui_products.presets()["presets"]:
        named = [slug for slug in preset["products"] if slug in unavailable]
        assert named == [], (preset["id"], named)
    block = tui_products.preset_availability()
    assert all(rows == {} for rows in block.values()), block


# ---------------------------------------------------------------------------
# The catalog a local run is offered
# ---------------------------------------------------------------------------


_ROWS = [
    ("2m_temperature", "direct", "drawable", "0", "2m AGL Temperature"),
    ("cloud_cover_levels", "direct", "drawable", "0", "Cloud Cover Levels"),
    ("10m_wind_gusts", "direct", "missing", "",
     "a wrfout import writes no wind_gust_10m_agl"),
    ("cloud_cover", "direct", "missing", "",
     "a wrfout import writes no total_cloud_cover_entire_atmosphere"),
    ("qpf_12h", "windowed", "drawable", "12", "its window first closes at F012"),
    ("qpf_24h", "windowed", "drawable", "24", "its window first closes at F024"),
]


def test_the_local_run_block_keeps_what_a_wrfout_can_draw():
    block = runplan.local_run_catalog(_ROWS)
    names = [row["name"] for row in block["products"]]
    assert names == ["2m_temperature", "cloud_cover_levels", "qpf_12h", "qpf_24h"]
    assert "wind_gust_10m_agl" in block["unavailable"]["10m_wind_gusts"]
    assert block["run_hours"] is None
    # No rows is "not asked", never "nothing drawable".
    assert runplan.local_run_catalog([]) is None


def test_a_run_length_takes_out_the_windows_that_close_after_the_run():
    block = runplan.local_run_catalog(_ROWS, run_hours=18.0)
    names = [row["name"] for row in block["products"]]
    assert "qpf_12h" in names and "qpf_24h" not in names
    reason = block["unavailable"]["qpf_24h"]
    assert "forecast hour 24" in reason and "18 h long" in reason
    assert "HRRR" not in reason


def _plan(tmp_path, run_hours: float) -> Path:
    from test_case_data import make_case_toml
    from test_runplan import _write_plan

    config = make_case_toml(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace(
        "run_seconds = 3600.0", f"run_seconds = {run_hours * 3600.0}"),
        encoding="utf-8")
    return _write_plan(tmp_path, config, tmp_path / "run")


def test_a_plan_narrows_the_catalog_to_its_own_length(tmp_path, monkeypatch):
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": "rust", "products": [{"name": row[0]} for row in _ROWS],
        "local_run": runplan.local_run_catalog(_ROWS)})
    document = runplan.plan_render_catalog(
        runplan.load_plan(_plan(tmp_path, 18.0)))
    local = document["local_run"]
    assert local["run_hours"] == 18.0
    names = {row["name"] for row in local["products"]}
    assert names == {"2m_temperature", "cloud_cover_levels", "qpf_12h"}
    assert set(local["unavailable"]) == {"10m_wind_gusts", "cloud_cover",
                                         "qpf_24h"}
    # The vocabulary itself is untouched: a door checking a spelling still
    # reads every product the engine knows.
    assert len(document["products"]) == len(_ROWS)


def test_the_picker_offers_the_local_run_list_and_keeps_the_vocabulary(
        monkeypatch):
    monkeypatch.setattr(runplan, "render_catalog", lambda: {
        "engine": "rust", "products": [{"name": row[0]} for row in _ROWS],
        "local_run": runplan.local_run_catalog(_ROWS)})
    document = tui_products.catalog_document()
    offered = {row["name"] for row in document["products"]}
    assert offered.isdisjoint(PRESET_DROPPED)
    assert {row["name"] for row in document["vocabulary"]} >= set(
        PRESET_DROPPED) - {"precipitation_type"}


# ---------------------------------------------------------------------------
# The closing note
# ---------------------------------------------------------------------------


def _png(root: Path, product: str, lead: int) -> Path:
    path = (root / "d01-3km" / product / "2026-09-26"
            / f"arwen_wrf_20260926_6z_f{lead:03d}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + product.encode() + bytes([lead]))
    return path


def _live_and_finalize_receipts(root: Path) -> None:
    """What a run of the old default preset left: every frame drawn live
    (skipping the three and, on its lone frame, qpf_1h), then a windowed
    pass that drew qpf_1h and skipped nothing."""

    spec = "2m_temperature,qpf_1h," + ",".join(PRESET_DROPPED)
    for lead in range(3):
        frame = f"/run/wrfout/wrfout_d01_2026-09-26_{6 + lead:02d}_00_00"
        skipped = [(slug, f"{frame}: not stored: a field")
                   for slug in PRESET_DROPPED]
        skipped.append(("qpf_1h", f"{frame}: windowed accumulations need "
                                  "more than one stored whole-hour frame"))
        render_receipts.publish_invocation(
            root=root, engine="rust", requested_spec=spec,
            written=[_png(root, "2m_temperature", lead)], failures=[],
            skipped=skipped, layout=NESTED)
    render_receipts.publish_invocation(
        root=root, engine="rust", requested_spec="qpf_1h",
        written=[_png(root, "qpf_1h", lead) for lead in (1, 2)],
        failures=[], skipped=[], layout=NESTED)


def test_the_summary_counts_every_product_that_drew_nothing_in_any_pass(
        tmp_path):
    root = tmp_path / "png"
    _live_and_finalize_receipts(root)
    summary = render_receipts.read_summary(root)
    undrawn = [row["name"] for row in summary["undrawn_families"]]
    assert sorted(undrawn) == sorted(PRESET_DROPPED)
    assert summary["undrawn_family_count"] == 3
    headline, detail = render_receipts.undrawn_note(summary)
    for slug in PRESET_DROPPED:
        assert slug in headline and slug in detail
    # qpf_1h drew two pictures; its lone-frame skips are not a zero.
    assert "qpf_1h" not in headline


def test_the_render_stage_closes_with_every_product_that_drew_nothing(
        tmp_path, monkeypatch, capsys):
    from woof import render

    monkeypatch.setattr(render, "drawable_engine",
                        lambda: ("rust", "declared by the test"))
    plan = {"run": tmp_path / "run", "render": tmp_path / "png",
            "render_products": "2m_temperature,qpf_1h," +
                               ",".join(PRESET_DROPPED)}
    frame = tmp_path / "run" / "wrfout" / "wrfout_d01_2026-09-26_06_00_00"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"one frame")

    def finalize(label, command, **_kw):
        # The end-of-run pass: in a real run the live renders published
        # their receipts first; here the pass publishes all of them.
        _live_and_finalize_receipts(plan["render"])

    monkeypatch.setattr(go_cli, "_run_stage", finalize)
    assert go_cli._render_stage(plan, explain=True, observer=None)
    printed = capsys.readouterr().out
    closing = [line for line in printed.splitlines()
               if "drew no picture in this run" in line]
    assert len(closing) == 1, printed
    for slug in PRESET_DROPPED:
        assert slug in closing[0]
    assert "qpf_1h" not in closing[0]


def test_the_closing_note_states_each_products_own_recorded_reason(tmp_path):
    """Each product that drew nothing is named with the reason it has.

    The note gave one cause for every product, "the frames do not carry
    their input fields or the time window they need", so a section term
    dropped for want of a line read as a forecast missing a field.  The
    reason the renderer recorded for each product is what the note says.
    """

    root = tmp_path / "png"
    section = "xsec:QCLOUD=0.01,0.1/wa"
    no_line = rustwx.drop_storeless_terms(section)[1][0][1]
    frame = "/run/wrfout/wrfout_d01_2026-09-26_06_00_00"
    render_receipts.publish_invocation(
        root=root, engine="rust",
        requested_spec=f"2m_temperature,cloud_cover,{section}",
        written=[_png(root, "2m_temperature", 0)], failures=[],
        skipped=[(section, no_line),
                 ("cloud_cover", f"{frame}: not stored: CLDFRA")],
        layout=NESTED)
    headline, detail = render_receipts.undrawn_note(
        render_receipts.read_summary(root))
    first, *reasons = headline.splitlines()
    assert "drew no picture in this run" in first
    assert section in first and "cloud_cover" in first
    assert "do not carry their input fields" not in headline
    by_product = dict(line.strip().split(": ", 1) for line in reasons)
    # Since c34b071fbf7 each reason is ranked by how many frames or render
    # attempts it affected, and says so ahead of the recorded cause.
    assert by_product[section] == "1 render attempt(s): " + " ".join(no_line.split())
    assert "--section" in by_product[section]
    assert by_product["cloud_cover"] == f"1 render attempt(s): {frame}: not stored: CLDFRA"
    assert f"{section}: skipped 1 time(s)" in detail


def test_a_pass_note_tells_a_frame_skip_from_a_product_that_drew_nothing():
    """One pass's note: `qpf_1h` skipped at F000 is not a product with no picture.

    Measured on a real 6 h run of the General preset: the end-of-run
    pass named `qpf_1h` beside nothing else, in the plural.
    """

    from woof import explain
    from woof.render import skip_notice

    reason = "wrfout_d01_2026-09-27_00_00_00: F000: 1-h QPF requires forecast hour >= 1"
    one = explain.render(skip_notice([("qpf_1h", reason)], drawn={"qpf_1h"}),
                         explain=False, command="woof render")
    assert "qpf_1h drew pictures and was skipped only at the frames" in one, one
    assert "drew no picture" not in one
    both = explain.render(skip_notice(
        [("qpf_1h", reason), ("total_qpf", reason), ("cloud_cover", "not stored")],
        drawn={"qpf_1h", "total_qpf"}), explain=False, command="woof render")
    assert "cloud_cover drew no picture in this render" in both, both
    assert "qpf_1h, total_qpf drew pictures and were skipped" in both, both


# ---------------------------------------------------------------------------
# Skip lines name their own frame
# ---------------------------------------------------------------------------


def test_each_skip_line_keeps_the_frame_the_engine_named(monkeypatch,
                                                         tmp_path):
    inputs = [tmp_path / f"wrfout_d01_2026-09-26_{hour:02d}_00_00"
              for hour in range(3)]

    class Result:
        returncode = 0
        stdout = "\n".join((
            f"SKIPPED qpf_6h {inputs[0]}: F000: 6-h QPF requires forecast hour >= 6",
            f"SKIPPED qpf_6h {inputs[1]}: F001: 6-h QPF requires forecast hour >= 6",
            "SKIPPED terrain_height static field: rendered once",
            ""))
        stderr = f"FAILED qpf_1h {inputs[1]}: F001: compute failed\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    _written, failures, skipped = rustwx.run_renderer_series(
        tmp_path / "rw_wrfbatch", inputs, store_root=tmp_path / "store",
        out_dir=tmp_path / "png", products="qpf_6h", frames="all",
        width=400, height=300)
    assert skipped[0] == ("qpf_6h", f"{inputs[0]}: F000: 6-h QPF requires "
                                    "forecast hour >= 6")
    assert skipped[1][1].startswith(f"{inputs[1]}: F001: ")
    # A line no frame owns is filed against the last input, as before.
    assert skipped[2][1] == f"{inputs[2]}: static field: rendered once"
    assert failures == [f"{inputs[1]}: qpf_1h F001: compute failed"]


# ---------------------------------------------------------------------------
# Against the real renderer
# ---------------------------------------------------------------------------


def _renderer_gate():
    from test_render_rust import _RENDERER_SKIP_REASON, _RENDERER_USABLE
    return _RENDERER_USABLE, _RENDERER_SKIP_REASON


_USABLE, _WHY = _renderer_gate()
needs_renderer = pytest.mark.skipif(not _USABLE, reason=_WHY)


def _wrfout_rows() -> dict[str, tuple[str, str]]:
    result = subprocess.run([str(rustwx.find_renderer()), "--list-products"],
                            capture_output=True, text=True,
                            env=rustwx.renderer_env(), check=True)
    rows = {}
    for line in result.stdout.splitlines():
        if line.startswith("WRFOUT\t"):
            _tag, slug, _kind, verdict, hour, _detail = line.split("\t", 5)
            rows[slug] = (verdict, hour)
    return rows


@needs_renderer
def test_every_preset_product_is_one_the_wrfout_lane_draws():
    rows = _wrfout_rows()
    for slug in NEVER_ON_WRFOUT:
        assert rows[slug][0] == "missing", (slug, rows[slug])
    for slug in DERIVED_ON_WRFOUT:
        assert rows[slug] == ("drawable", "0"), (slug, rows[slug])
    for preset in tui_products.presets()["presets"]:
        for slug in preset["products"]:
            if slug.startswith("var:"):
                continue
            assert rows[slug][0] == "drawable", (preset["id"], slug, rows[slug])


@needs_renderer
@pytest.mark.parametrize("hours", [24.0, 18.0], ids=["gfs-3km-24h", "18h"])
def test_the_catalog_for_a_plan_offers_nothing_a_wrfout_lacks(
        hours, tmp_path, capsys):
    """(1) of the defect: `run-plan PLAN --catalog`, the real engine."""

    runplan._RENDER_CATALOG_CACHE.clear()
    assert cli.main(["run-plan", "--catalog", str(_plan(tmp_path, hours))]) == 0
    document = json.loads(capsys.readouterr().out)
    local = document["local_run"]
    assert local["run_hours"] == hours
    offered = {row["name"]: row for row in local["products"]}
    for slug in NEVER_ON_WRFOUT:
        assert slug not in offered
        assert "writes no" in local["unavailable"][slug], local["unavailable"][slug]
    for slug in DERIVED_ON_WRFOUT:
        assert slug in offered and slug not in local["unavailable"]
    assert all(row["minimum_hour"] <= hours for row in offered.values())
    assert "2m_temperature" in offered and "cloud_cover_levels" in offered
    # The vocabulary still carries every model's products for the doors
    # that check a spelling.
    assert len(document["products"]) > len(offered)


def _hourly_series(tmp_path, count: int) -> list[Path]:
    from test_render_rust import _write_wrfout

    frames = []
    for hour in range(count):
        day, clock = divmod(18 + hour, 24)
        stamp = f"1974-04-{3 + day:02d}_{clock:02d}:00:00"
        frames.append(_write_wrfout(
            tmp_path / f"wrfout_d02_{stamp.replace(':', '_')}", (stamp,),
            seed_offset=hour))
    return frames


@needs_renderer
def test_all_on_an_18h_series_draws_named_products_inside_the_run(
        tmp_path, capsys):
    """(2) and (3) of the defect, on a 19-frame series through `woof render`."""

    frames = _hourly_series(tmp_path, 19)
    out = tmp_path / "png"
    rc = cli.main(["render", *map(str, frames), "--engine", "rust",
                   "--series", "--size", "400x300", "--out", str(out),
                   "--run-stamp", "off"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    folders = {png.relative_to(out).parts[-3] for png in out.rglob("*.png")}
    assert folders, "nothing was drawn"
    hashed = [name for name in folders
              if re.fullmatch(r"var_.*_[0-9a-f]{16}", name)]
    assert hashed == [] and not any(n.startswith("var_") for n in folders), folders
    rows = _wrfout_rows()
    for folder in folders:
        verdict, hour = rows.get(folder, ("drawable", "0"))
        assert int(hour or 0) <= 18, (folder, hour)
    summary = render_receipts.read_summary(out)
    for row in summary["skipped_families"]:
        assert int(rows.get(row["name"], ("", "0"))[1] or 0) <= 18, row
        assert all("HRRR" not in reason for reason in row["reasons"]), row
    # Every skip line names the file of its own frame.
    by_name = {frame.name: index for index, frame in enumerate(frames)}
    checked = 0
    for receipt in (out / ".render-receipts").glob("*.json"):
        for row in json.loads(receipt.read_text())["skipped"]:
            match = re.match(r"(?P<file>.+?): F(?P<hour>\d{3}): ", row["reason"])
            if match is None:
                continue
            checked += 1
            assert by_name[Path(match["file"]).name] == int(match["hour"]), row
    assert checked >= 6, "no per-frame skip line was checked"

"""The plain render door answers a `mesh:` or `xsec:` term itself.

WHAT BREAKAGE THESE PIN (gate law).  Three families do not come from the
store, and the renderer refuses each of them for the WHOLE invocation,
before it draws a single picture (`rw-wrfbatch/src/main.rs`): a `mesh:`
term with no `--mesh-grid` is refused at argument validation, a `mesh:`
term beside store products is refused at the entry to the batch render,
and an `xsec:` term with no `--section` is refused before the store
render starts.  So one such term forwarded from this door costs every
other requested product its pictures, and `woof render --products
composite_reflectivity,mesh:cell_area` exited 1 with no pictures at all
where it could have drawn every reflectivity frame.

Both doors of this product answer such a term the same way, per PRODUCT
rather than per invocation, because the renderer's own answer is the
per-invocation one and no door can afford to pass it on: the plain
`woof render` door drops the term before it launches the renderer, and
the `woof downscale` door drops it before the child integrates so the
finalize render draws what the request left.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from woof import render, rustwx
from woof.io.wrfout import WrfoutWriter


# ------------------------------------------------------- the grammar

def test_a_request_with_no_storeless_term_is_returned_unchanged():
    spec, dropped = rustwx.drop_storeless_terms(
        "composite_reflectivity,2m_temperature")
    assert spec == "composite_reflectivity,2m_temperature"
    assert dropped == []


def test_a_mesh_term_is_dropped_and_the_rest_of_the_request_survives():
    spec, dropped = rustwx.drop_storeless_terms(
        "composite_reflectivity,mesh:cell_area,2m_temperature")
    assert spec == "composite_reflectivity,2m_temperature"
    assert [term for term, _why in dropped] == ["mesh:cell_area"]
    why = dropped[0][1]
    assert "cell boundaries" in why
    assert "--mesh-grid" in why


def test_a_meshdiff_term_is_dropped_too():
    spec, dropped = rustwx.drop_storeless_terms("meshdiff:qv,2m_temperature")
    assert spec == "2m_temperature"
    assert [term for term, _why in dropped] == ["meshdiff:qv"]


def test_a_section_term_is_dropped_only_when_no_line_was_given():
    spec, dropped = rustwx.drop_storeless_terms("2m_temperature,xsec:wa")
    assert spec == "2m_temperature"
    assert [term for term, _why in dropped] == ["xsec:wa"]
    assert "--section" in dropped[0][1]

    spec, dropped = rustwx.drop_storeless_terms(
        "2m_temperature,xsec:wa", section="40.0,-100.0,41.0,-99.0")
    assert dropped == []
    assert spec == "2m_temperature,xsec:wa"


def test_a_section_terms_level_list_survives_the_split():
    """`xsec:wa=1,2,5@5` is ONE term whose level list holds commas."""

    spec, dropped = rustwx.drop_storeless_terms(
        "xsec:wa=1,2,5@5,2m_temperature", section="40,-100,41,-99")
    assert dropped == []
    assert "xsec:wa=1,2,5@5" in spec
    spec, dropped = rustwx.drop_storeless_terms(
        "xsec:wa=1,2,5@5,2m_temperature")
    assert spec == "2m_temperature"
    assert [term for term, _why in dropped] == ["xsec:wa=1,2,5@5"]


def test_a_group_keyword_is_never_dropped():
    """The engine expands a group itself and leaves out what it cannot draw."""

    assert rustwx.drop_storeless_terms("all") == ("all", [])


# ------------------------------------- the seam every rust render uses

def test_the_availability_seam_drops_a_mesh_term_before_it_asks_anything(
        tmp_path, monkeypatch):
    """A library caller of the rust render path is covered too.

    Nothing may be launched to decide this: the grammar is knowable with
    no renderer and no file, and a mesh-only request must not pay for a
    catalog listing it cannot use.
    """

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("the door launched the renderer"))
    available, skipped = render._available_window_request(
        Path("rw_wrfbatch"), tmp_path / "wrfout_d01_1974-04-03_22_00_00",
        "mesh:cell_area", tmp_path, heavy=False)
    assert available == ""
    assert [slug for slug, _reason in skipped] == ["mesh:cell_area"]


# ------------------------------------------------------- the CLI door

NX, NY, NZ = 8, 6, 4


def _wrfout(path: Path) -> Path:
    frame = {
        "T2": np.zeros((NY, NX), np.float32),
        "XLAT": np.zeros((NY, NX), np.float32),
        "XLONG": np.zeros((NY, NX), np.float32),
    }
    with WrfoutWriter(path, nx=NX, ny=NY, nz=NZ, dx=3000.0, dy=3000.0,
                      title="woof test") as writer:
        writer.write_frame("1974-04-03_22:00:00", frame)
    return Path(path)


def _rust_door(monkeypatch, pictures: int = 1):
    """The front door on the rust engine, with the engine itself stubbed."""

    from woof import provenance_gate

    asked: dict = {}

    def render_rust(paths, *, products, outdir, **kwargs):
        asked["products"] = products
        asked["section"] = kwargs.get("section")
        # Into the directory the DOOR hands down, which is the run
        # folder it claimed: the receipt publisher reads every PNG it is
        # named and refuses one outside that root, so a stub that spelled
        # its own path would fail there rather than where it is aimed.
        drawn = []
        for index in range(pictures):
            png = (Path(outdir) / "d01-12km" / "composite_reflectivity"
                   / "1974-04-03"
                   / f"arwen_wrf_19740403_22z_f{index:03d}.png")
            png.parent.mkdir(parents=True, exist_ok=True)
            png.write_bytes(b"\x89PNG\r\n\x1a\n")
            drawn.append(png)
        return drawn, [], []

    monkeypatch.setattr(render, "_resolve_engine",
                        lambda choice: ("rust", "stub"))
    monkeypatch.setattr(render, "engine_refusal", lambda *a, **k: None)
    monkeypatch.setattr(render, "missing_basemap_notice", lambda *a: None)
    monkeypatch.setattr(rustwx, "find_renderer", lambda: Path("rw_wrfbatch"))
    monkeypatch.setattr(
        provenance_gate, "bridge_tree_match",
        lambda *a, **k: argparse.Namespace(verdict="stub", basis="stub"))
    monkeypatch.setattr(render, "render_wrfouts_rust", render_rust)
    return asked


def _argv(argv):
    parser = argparse.ArgumentParser(prog="woof")
    render.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(["render", *argv])


def test_the_door_draws_the_other_products_and_names_the_mesh_term(
        tmp_path, monkeypatch, capsys):
    """Per product, not per invocation: the reflectivity frames survive."""

    path = _wrfout(tmp_path / "wrfout_d01_1974-04-03_22_00_00")
    out = tmp_path / "png"
    asked = _rust_door(monkeypatch)
    code = render.render_main(_argv([
        "--products", "composite_reflectivity,mesh:cell_area",
        "--out", str(out), str(path)]))
    captured = capsys.readouterr()
    assert asked["products"] == "composite_reflectivity"
    assert "mesh:cell_area" in captured.err
    assert "cell boundaries" in captured.err
    assert "--mesh-grid" in captured.err
    assert "still drawn" in captured.err
    # The line below the note says how the ENGINE was resolved, and a
    # real render on test host 1 printed a dropped product's sentence there
    # instead: the loop that names the drops had taken that name.
    assert "render: engine rust (stub)" in captured.out
    assert "cell boundaries" not in captured.out
    assert code == 0


def test_a_mesh_only_request_is_refused_and_leaves_no_run_directory(
        tmp_path, monkeypatch, capsys):
    """Nothing is left to draw, so the door refuses instead of rendering."""

    path = _wrfout(tmp_path / "wrfout_d01_1974-04-03_22_00_00")
    out = tmp_path / "png"
    _rust_door(monkeypatch, pictures=0)
    code = render.render_main(_argv([
        "--products", "mesh:cell_area", "--out", str(out), str(path)]))
    captured = capsys.readouterr()
    assert code == 2
    assert "mesh:cell_area" in captured.err
    assert "--mesh-grid" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()


def test_a_section_request_with_a_line_reaches_the_renderer(
        tmp_path, monkeypatch, capsys):
    """The door refuses the term it cannot answer, never one it can."""

    path = _wrfout(tmp_path / "wrfout_d01_1974-04-03_22_00_00")
    out = tmp_path / "png"
    asked = _rust_door(monkeypatch)
    code = render.render_main(_argv([
        "--products", "2m_temperature,xsec:wa",
        "--section", "40,-100,41,-99", "--out", str(out), str(path)]))
    assert asked["products"] == "2m_temperature,xsec:wa"
    assert code == 0


def test_a_section_request_with_no_line_is_refused_before_the_renderer(
        tmp_path, monkeypatch, capsys):
    path = _wrfout(tmp_path / "wrfout_d01_1974-04-03_22_00_00")
    out = tmp_path / "png"
    _rust_door(monkeypatch, pictures=0)
    code = render.render_main(_argv([
        "--products", "xsec:wa", "--out", str(out), str(path)]))
    captured = capsys.readouterr()
    assert code == 2
    assert "--section" in captured.err
    assert not out.exists()


def test_the_dropped_term_is_named_in_the_result_receipt(
        tmp_path, monkeypatch, capsys):
    """`render-summary.json` carries what the door dropped, as a skip."""

    path = _wrfout(tmp_path / "wrfout_d01_1974-04-03_22_00_00")
    out = tmp_path / "png"
    _rust_door(monkeypatch)
    render.render_main(_argv([
        "--products", "composite_reflectivity,mesh:cell_area",
        "--out", str(out), str(path)]))
    # The door prints where it filed the receipt; a test that spelled
    # the path itself would be pinning the run-folder layout instead.
    printed = [line for line in capsys.readouterr().out.splitlines()
               if line.startswith("render: result receipt -> ")]
    summary = json.loads(Path(printed[0].split("-> ", 1)[1])
                         .read_text(encoding="utf-8"))
    assert "mesh:cell_area" in [row["name"]
                                for row in summary["skipped_families"]]
    assert "mesh:cell_area" in summary["requested_families"]


# --------------------------------------------- the downscale door

def _admit(spec, dry_run=True):
    """The downscale door's product admission, renderer catalog stubbed.

    That catalog answers "is this a product name at all", which is a
    separate refusal with its own remedy; every case here is about the
    three families whose answer needs no renderer.
    """

    from woof import downscale, go_cli

    original = go_cli.unknown_render_products
    go_cli.unknown_render_products = lambda spec: []
    try:
        return downscale._admit_render_products(spec, dry_run=dry_run)
    finally:
        go_cli.unknown_render_products = original


def test_the_downscale_door_drops_a_mesh_term_and_keeps_the_rest(capsys):
    """The chain's render stage is `woof render`, which draws the rest.

    A door that refused the whole request here would refuse a run the
    stage it feeds completes: measured at this tip, the same spelling
    through `woof render` draws every reflectivity frame and names the
    mesh term.
    """

    assert _admit("composite_reflectivity,mesh:cell_area") == "composite_reflectivity"
    err = capsys.readouterr().err
    assert "mesh:cell_area" in err
    assert "cell boundaries" in err
    assert "--mesh-grid" in err
    assert "still drawn" in err


def test_the_downscale_door_drops_a_section_term_it_composes_no_line_for(
        capsys):
    assert _admit("2m_temperature,xsec:wa") == "2m_temperature"
    err = capsys.readouterr().err
    assert "xsec:wa" in err and "--section" in err


def test_the_downscale_door_refuses_when_the_drop_leaves_nothing(capsys):
    """Nothing left to draw is the one case that still refuses."""

    from woof.downscale import OfflineChildContractError

    with pytest.raises(OfflineChildContractError) as excinfo:
        _admit("meshdiff:qv")
    message = str(excinfo.value)
    assert "meshdiff:qv" in message
    assert "--mesh-grid" in message
    assert "nothing left to draw" in message


def test_both_doors_of_one_product_answer_one_request_the_same_way():
    """One grammar, asked twice: the doors cannot drift apart."""

    for spec in ("composite_reflectivity,mesh:cell_area",
                 "2m_temperature,xsec:wa",
                 "2m_temperature,composite_reflectivity",
                 "all"):
        assert _admit(spec) == rustwx.drop_storeless_terms(spec)[0]


def test_a_request_that_drops_nothing_is_handed_on_byte_for_byte():
    assert _admit("2m_temperature,composite_reflectivity") == (
        "2m_temperature,composite_reflectivity")
    assert _admit("none") == "none"


def test_the_downscale_door_needs_no_renderer_for_this_answer(monkeypatch):
    """The grammar is knowable with nothing opened and nothing launched."""

    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("the door launched something"))
    assert _admit("composite_reflectivity,mesh:cell_area") == "composite_reflectivity"

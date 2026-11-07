"""``woof render --products olr`` asks the engine for what --help says.

The breakage: ``woof render --help`` listed ``olr`` (and described TOA
outgoing longwave "as synthetic IR"), but the rust engine, the default,
was handed ``olr`` as a raw slug and refused it as an unknown product, so
the advertised product drew nothing.  Measured on test host 1 against a WOOF
history from a real GFS run: 2.8.7 and the 2.8.8 candidate both exit 1,
"unknown product 'olr'".  The engine draws the stored OLR plane as
``var:wrf_olr``; ``simulated_ir_satellite`` is a brightness temperature the
history lane does not serve and is NOT what ``olr`` names.
"""
from __future__ import annotations

from woof import render
from woof.io import history_selection


def test_every_shared_name_has_an_engine_spelling():
    assert set(render.PRODUCTS) == set(render.RUST_PRODUCT_ALIASES)
    assert render.RUST_PRODUCT_ALIASES["olr"] == "var:wrf_olr"
    assert render.parse_products_rust("olr") == "var:wrf_olr"
    assert render.parse_products_rust("t2,olr") == "2m_temperature,var:wrf_olr"


def test_olr_reads_the_same_history_field_either_way():
    inputs = history_selection.PRODUCT_HISTORY_INPUTS
    assert inputs["olr"] == inputs["var:wrf_olr"] == ("OLR",)


def test_the_help_names_the_engine_product_and_no_synthetic_ir():
    import argparse

    parser = argparse.ArgumentParser(prog="woof")
    render.register_cli(parser.add_subparsers())
    choices = parser._subparsers._group_actions[0].choices
    text = choices["render"].format_help()
    flat = " ".join(text.split())
    assert "olr=var:wrf_olr" in flat
    assert "synthetic IR" not in flat
    subcommand = parser._subparsers._group_actions[0]._choices_actions
    summary = " ".join(action.help for action in subcommand
                       if action.dest == "render")
    assert "TOA outgoing longwave radiation in W m-2" in " ".join(
        summary.split())


def test_the_catalog_listing_prints_the_shared_names(monkeypatch, capsys):
    """``--list-products`` and ``--help`` give one answer for the names."""
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    from woof import rustwx

    monkeypatch.setattr(render, "_resolve_engine",
                        lambda engine: ("rust", "test"))
    monkeypatch.setattr(rustwx, "find_renderer", lambda: Path("rw_wrfbatch"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="composite_reflectivity\n", stderr=""))
    assert render.catalog_main(SimpleNamespace(engine="rust")) == 0
    out = capsys.readouterr().out
    assert "olr -> var:wrf_olr" in out
    assert "t2 -> 2m_temperature" in out

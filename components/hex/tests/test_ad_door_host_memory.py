# SPDX-License-Identifier: Apache-2.0
"""A differentiable-engine door says what it needs before it runs and what it used after, and a host with less memory
available than the row's window needs is refused before the first-use download, with its own exit code (host-only).
THE BREAKAGE THIS PREVENTS: the 120 km window holds about 99 GiB of host memory; on a smaller host the runner was
ended part way through, after the download, with no sentence a user could act on."""
import argparse
from pathlib import Path

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets, rows


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def card_fits(monkeypatch, tmp_path):
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    fetched = []

    def asset_root(cache=None):
        fetched.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads the row and plan)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    monkeypatch.delenv("WOOF_HEX_AD_ASSETS", raising=False)
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    return fetched


def test_meminfo_reads_mem_available_in_mib(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       131072000 kB\nMemFree:         2048000 kB\nMemAvailable:   65536000 kB\n")
    assert d.host_available_mib(meminfo) == 64000
    meminfo.write_text("MemTotal:       131072000 kB\n")
    assert d.host_available_mib(meminfo) is None
    assert d.host_available_mib(tmp_path / "absent") is None


@pytest.mark.parametrize("door", list(d.DOORS))
def test_a_host_without_the_memory_is_refused_before_any_download(card_fits, tmp_path, monkeypatch, capsys, door):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 60000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    args = _parse([door, "--need-host-mib", "106496", "--out", str(tmp_path / "o")])
    with pytest.raises(d.HostMemoryError, match="60,000 MiB of memory available .* needs about 106,496 MiB"):
        d.Context(args, door)
    assert card_fits == [] and not (tmp_path / "o").exists()
    # through the command's handler: one sentence on stderr and the host's own exit code, not the generic 2
    assert args.handler(args) == d.HOST_TOO_SMALL_EXIT == 3
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Nothing but the row's mesh file was downloaded" in err and "--need-host-mib" in err


def test_the_row_need_is_table_data_and_a_row_without_one_is_not_judged(card_fits, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 1000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    loaded0 = d.load_table

    def without_need(path=None):   # the shipped default row carries a measured need: here a row without one
        t = loaded0(path)
        t["rows"][t["default_row"]].pop("need_host_mib", None)
        return t

    monkeypatch.setattr(d, "load_table", without_need)
    table = d.load_table()
    row = table["rows"][table["default_row"]]
    assert "need_host_mib" not in row
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["tangent", "--out", str(tmp_path / "o")]), "tangent")
    out = capsys.readouterr().out
    assert "host memory not measured for this row (1,000 MiB available)" in out
    loaded = d.load_table

    def with_need(path=None):
        t = loaded(path)
        t["rows"][t["default_row"]]["need_host_mib"] = 106496
        return t

    monkeypatch.setattr(d, "load_table", with_need)
    with pytest.raises(d.HostMemoryError, match="needs about 106,496 MiB"):
        d.Context(_parse(["tangent", "--out", str(tmp_path / "o")]), "tangent")


def _generator():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "ad_doors_from_latest.py"
    spec = importlib.util.spec_from_file_location("ad_doors_from_latest", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def test_a_door_s_own_host_need_wins_over_the_row_s(card_fits, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 100000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    loaded = d.load_table

    def with_needs(path=None):
        t = loaded(path)
        t["rows"][t["default_row"]].update({"need_host_mib": 90000, "need_host_mib_by_door": {"adjoint-test": 110000}})
        return t

    monkeypatch.setattr(d, "load_table", with_needs)
    with pytest.raises(d.HostMemoryError, match="needs about 110,000 MiB"):
        d.Context(_parse(["adjoint-test", "--out", str(tmp_path / "o")]), "adjoint-test")
    with pytest.raises(AssertionError, match="asset root was resolved"):          # 90,000 fits: on to the assets
        d.Context(_parse(["tangent", "--out", str(tmp_path / "o")]), "tangent")


def test_the_120_km_host_need_goes_from_the_generator_to_the_table_and_the_made_row(tmp_path, monkeypatch):
    gen = _generator()
    spec120 = [c for c in gen.CLASSES if c["recipe"] == "120"][0]
    assert spec120["need_host_mib"] == 81920
    table = d.load_table()
    c120 = [c for c in table["suites"]["hrrrv5"]["window_classes"] if c.get("from_recipe") == "120"]
    assert len(c120) == 1                                       # the table is what the generator writes
    assert (c120[0]["need_host_mib"], c120[0]["need_host_source"]) == (81920, spec120["need_host_source"])

    def recipe(bundle):
        return {"env": {"AD_SCHEMES": "{L}/bundle/" + bundle, "AD_TAPE_PACK": "1"},
                "args": [["kernels", "{L}/kset/k/kernels.json"], ["cubins", "{L}/kset/k/cubins"],
                         ["schemes", "{L}/aux/s.json"], ["tempo-cubin", "{L}/aux/t.cubin"], ["fct-frozen", "1"],
                         ["refuse", "a,b"]], "needs": {"cards": 1}}

    built = gen.build({"id": "T", "recipes": {"480": recipe("gen-A"), "480-best": recipe("gen-B9"),
                                              "120": recipe("gen-B9"), "60": None}}, table)["table"]
    small, c120 = (rows.window_class(built["suites"]["hrrrv5"], n) for n in (2562, 40962))
    assert c120["need_host_mib"] == 81920 and "need_host_mib" not in small
    # a row made for the 120 km mesh carries the class's need; the door then judges the host by it
    for name in ("x1.40962.grid.nc", "x1.40962.ugwp_oro_data.nc"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(rows, "mesh_facts", lambda grid, init=None: {"cells": 40962, "edges": 122880, "levels": 59,
                                                                     "nominal_dc_m": 120000.0})
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    mesh = rows.find_mesh("x1.40962", tmp_path / "x1.40962.grid.nc", tmp_path / "x1.40962.ugwp_oro_data.nc")
    name, row = rows.make_row(table, "hrrrv5", mesh)
    assert name == "hrrrv5-x1.40962" and row["need_host_mib"] == 81920


def test_needs_line_before_the_runner_names_card_host_and_reference_minutes(card_fits, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 120000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    loaded = d.load_table

    def with_reference(path=None):
        t = loaded(path)
        row = t["rows"][t["default_row"]]
        row.update({"need_host_mib": 106496, "reference_minutes": {"adjoint-test": 4.2},
                    "reference_card": "one 96 GB card"})
        return t

    monkeypatch.setattr(d, "load_table", with_reference)
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["adjoint-test", "--out", str(tmp_path / "o")]), "adjoint-test")
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("woof hex adjoint-test: row ")
    assert "MiB of card memory (card " in line and ": 97,000 MiB free)" in line   # the tree conftest hides cards
    assert "106,496 MiB of host memory (120,000 MiB available)" in line
    assert line.endswith("about 4.2 minutes on one 96 GB card")


def test_used_line_reads_the_receipt():
    assert d.used_line({"card_peak_mib": 71156, "host_peak_mib": 101280, "wall_s": 251.3}) == \
        "used: card peak 71,156 MiB, host peak 101,280 MiB, wall 251.3 s"
    assert d.used_line({}) == "used: card peak n/a MiB, host peak n/a MiB, wall None s"
    assert d.host_peak_mib() > 0


def test_every_command_of_the_full_row_has_a_host_need_so_none_runs_unjudged():
    table = d.load_table()
    need = table["rows"]["hrrrv5-full-x1.40962"]["need_host_mib_by_door"]
    assert set(need) == {"adjoint-test", "tangent", "sensitivity", "param-grad"} and min(need.values()) >= 104845
    assert table["rows"][table["default_row"]]["need_host_mib"] >= 74858


def test_param_grad_names_the_bundle_it_runs_in_place_of_the_rows():
    table = d.load_table()
    for name in ("hrrrv5-x1.40962", "hrrrv5-full-x1.40962"):
        words = table["rows"][name]["param_bundle"]["instead_of_row"]
        assert "mountain-drag closure" in words and "differentiated" in words and "about an hour" in words


def test_a_reflectivity_measure_is_refused_with_its_reason():
    table = d.load_table()
    assert "does not pass the gradient check" in d.metric_not_offered(table, "reflectivity")
    assert d.metric_not_offered(table, "theta_x") == ""

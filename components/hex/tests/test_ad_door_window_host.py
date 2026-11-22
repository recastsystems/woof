# SPDX-License-Identifier: Apache-2.0
"""The host memory a window that keeps every step boundary needs, stated from its measured peaks. No card.

THE BREAKAGE (the 0.6.0 candidate's 96 GB check, 2026-11-15): ``adjoint-test`` on the 3 km storm example (59 steps)
stated and admitted itself on 110,592 MiB of host memory, a 3-step measure, and held 262,306 MiB on one card run and
254,885 MiB on another. The runner keeps step boundaries in host memory up to 0.4 of what is available when a window
starts, beside its own memory, so a host with less than the two together was admitted and then ended part way through.
"""
import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_session as s
from woof.hex.ad import assets, rows

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
CUT = "hrrrv5-q12.96.2890.n40.60w91.30"
AT3 = ["adjoint-test", "--row", STORM, "--steps", "59", "--at", "40.6,-91.3", "--radius-km", "150"]


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def table():
    return d.load_table()


@pytest.fixture
def box(monkeypatch, tmp_path):
    """One empty 96 GB card of the right family, an install that holds the plan's libraries, a disk that fits; the
    asset module fetches the mesh file alone (recorded) and refuses any other fetch. The host is the test's to set."""
    for name in d.NO_LOCAL_GPU_ENV + s.SCHEDULER_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("WOOF_HEX_AD_ASSETS", s.MODE_ENV, rows.IMAGE_BUILDER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(s.SESSIONS_ENV, str(tmp_path / "sessions"))
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    fetched = []
    root = tmp_path / "root"

    def asset_root(cache=None, want=None, rows=True):
        if rows is False:
            fetched.append("mesh only")
            for rel in want:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"grid")
            return root
        fetched.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads the example)")

    def geometry(grid, cells, edges):
        rng = np.random.Generator(np.random.PCG64(7))
        return {"latCell": 40.6 + rng.random(cells) - 0.5, "lonCell": -91.3 + rng.random(cells) - 0.5,
                "areaCell": np.ones(cells)}

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    monkeypatch.setattr(d, "mesh_geometry", geometry)
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97252)
    monkeypatch.setattr(d, "_free_disk_mib", lambda folder: 9_000_000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)

    def host(mib):
        monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: mib)
        monkeypatch.setattr(d, "host_available_settled", lambda *a, **k: mib)
    return SimpleNamespace(fetched=fetched, host=host)


def test_the_row_figures_are_the_measured_ones(table):
    row = table["rows"][STORM]
    wh, bmib = row["window_host"], int(row["boundary_mib"])
    three, exact, default = wh["measured"]
    # 3 steps: 100,095,048 KiB with all four boundaries in host memory
    assert three["host_peak_mib"] == 100_095_048 // 1024 and three["store_in_memory_mib"] == 4 * bmib
    assert wh["base_mib"] == three["host_peak_mib"] - 4 * bmib == three["beside_store_mib"] and wh["base_steps"] == 3
    for run in (exact, default):   # 59 steps: the peak less the 19 boundaries its fullest window kept in host memory
        assert run["beside_store_mib"] == run["host_peak_mib"] - run["store_in_memory_mib"]
    assert (exact["host_peak_mib"], default["host_peak_mib"]) == (262306, 254885)
    # the runner's own memory at 59 steps: the largest reading and the first window's extra page-locked memory
    own59 = wh["base_mib"] + wh["per_step_mib"] * 56
    assert exact["beside_store_mib"] + 8068 <= own59 <= exact["beside_store_mib"] + 8068 + 100
    assert wh["store_share"] == 0.4 and 1.0 < wh["margin"] <= 1.1   # the runner's default budget; the stated margin


@pytest.mark.parametrize("steps, available, held", [
    (59, 469_045, 262_306),   # the exact route's fullest window: its store budget 187,618 MiB = 0.4 of this
    (59, 480_230, 254_885),   # the default route's fullest window: its store budget 192,092 MiB
    (3, 300_000, 97_749)])
def test_the_need_covers_every_measured_peak_with_its_margin(table, steps, available, held):
    need = d.store_need(table["rows"][STORM], steps, available, line_on=False)
    assert held < need["host_mib"] <= 1.16 * held   # over the peak, and not so far over that a host that fits is refused
    assert need["line"] is False and f"{steps} steps" in need["host_for"]


def test_the_need_follows_the_host(table):
    row = table["rows"][STORM]
    wh = row["window_host"]
    own = wh["base_mib"] + wh["per_step_mib"] * 56
    big, small = (d.store_need(row, 59, a, line_on=False) for a in (460_544, 180_000))
    assert big["ram_mib"] == int(0.4 * 460_544) and small["ram_mib"] == 72_000
    assert small["host_mib"] == -(-wh["margin"] * (own + 72_000) // 1) < big["host_mib"]
    assert small["files_mib"] == 60 * row["boundary_mib"] - 72_000   # what is not in host memory is on the disk
    # the smallest host: where the runner's own memory and its share of what is available fit together
    least = small["host_smallest_mib"]
    assert least == big["host_smallest_mib"]
    assert d.store_need(row, 59, least, line_on=False)["host_mib"] <= least
    assert d.store_need(row, 59, least - 1000, line_on=False)["host_mib"] > least - 1000
    # a short window fits a smaller host: at 3 steps the whole store is four boundaries
    assert d.store_need(row, 3, 90_000, line_on=False)["host_mib"] <= 90_000 < small["host_smallest_mib"]
    # a row without the measure, and a gradient line, are as before
    assert d.store_need({k: v for k, v in row.items() if k != "window_host"}, 59, 180_000, False)["host_mib"] is None
    line = d.store_need(row, 59, 500_000, line_on=True)
    assert line["host_mib"] == 258_057 and line["host_smallest_mib"] is None


def test_a_host_under_the_floor_is_refused_in_one_sentence_before_the_example_is_fetched(box, tmp_path, capsys):
    box.host(100_000)
    args = _parse(AT3 + ["--out", str(tmp_path / "o")])
    with pytest.raises(d.HostMemoryError) as caught:
        d.Context(args, "adjoint-test")
    words = str(caught.value)
    assert "this host has 100,000 MiB of memory available" in words and "needs about 102,948 MiB" in words
    assert "59 steps: the runner's own memory and the 1 of 60 step boundaries" in words
    assert "(--bstore-ram-gib 10, the other 59 in the run folder)" in words
    assert "Run it where at least 102,948 MiB are available" in words and "a shorter window (--steps)" in words
    assert words.endswith("Nothing but the row's mesh file was downloaded")
    assert box.fetched == ["mesh only"] and not (tmp_path / "o").exists()
    box.fetched.clear()
    assert args.handler(args) == d.HOST_TOO_SMALL_EXIT == 3   # through the command: one sentence, the host's own code
    assert capsys.readouterr().err.count("\n") == 1 and box.fetched == ["mesh only"]


@pytest.mark.parametrize("available", [102_948, 120_000, 158_958, 180_000, 250_000, 500_000])
def test_a_host_that_holds_it_goes_on_to_the_downloads(box, tmp_path, available):
    box.host(available)
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(AT3 + ["--out", str(tmp_path / "o")]), "adjoint-test")
    assert box.fetched == ["mesh only", "asset_root"]


def test_the_override_and_the_other_doors_keep_their_rules(box, tmp_path):
    box.host(120_000)
    with pytest.raises(AssertionError, match="asset root was resolved"):   # the user's own figure stands
        d.Context(_parse(AT3 + ["--need-host-mib", "100000", "--out", str(tmp_path / "o")]), "adjoint-test")
    with pytest.raises(d.HostMemoryError, match=r"needs about 228,577 MiB \(the gradient with its answer check"):
        d.Context(_parse(["sensitivity", "--row", STORM, "--steps", "59", "--circle", "40.6,-91.3,150",
                          "--session", "off", "--out", str(tmp_path / "s")]), "sensitivity")
    box.host(60_000)   # a 3-step tangent: four boundaries, at least one of them in host memory
    with pytest.raises(d.HostMemoryError, match=r"needs about 70,902 MiB \(3 steps: the runner's own memory"):
        d.Context(_parse(["tangent", "--row", STORM, "--steps", "3", "--at", "40.6,-91.3", "--radius-km", "150",
                          "--out", str(tmp_path / "t")]), "tangent")


def test_each_window_is_judged_again_when_it_starts(table, monkeypatch):
    row = table["rows"][STORM]
    ctx = SimpleNamespace(row=row, row_name=STORM, steps=59, split=None, args=SimpleNamespace(need_host_mib=None),
                          store_need=d.store_need(row, 59, 304_188, line_on=False))
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 469_045)   # the host freed up meanwhile
    got = d.Context.window_host_admission(ctx, "window-descent")
    assert got == {"window": "window-descent", "available_mib": 469_045, "need_mib": 289_195,
                   "store_in_memory_mib": 187_618}
    assert d.Context.need_host_mib(ctx) == 289_195 > ctx.store_need["host_mib"]   # the receipt's need: the largest
    # the window before has just ended and its memory is still coming back: a short reading is read again, settled
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 90_000)
    waited = []
    monkeypatch.setattr(d, "host_available_settled", lambda wait_s=None: waited.append(wait_s) or 300_000)
    assert d.Context.window_host_admission(ctx, "selfdot")["available_mib"] == 300_000
    assert waited == [d.WINDOW_RELEASE_WAIT_S]
    monkeypatch.setattr(d, "host_available_settled", lambda wait_s=None: 90_000)   # another job took the memory
    with pytest.raises(d.HostMemoryError, match="its selfdot window, 59 steps"):
        d.Context.window_host_admission(ctx, "selfdot")
    # not judged: a gradient line's windows, --need-host-mib, a row without the measure
    ctx.args.need_host_mib = 100_000
    assert d.Context.window_host_admission(ctx, "selfdot") is None
    line = SimpleNamespace(row=row, row_name=STORM, steps=59, split=None, args=SimpleNamespace(need_host_mib=None),
                           store_need=d.store_need(row, 59, 130_000, line_on=True))
    assert d.Context.window_host_admission(line, "selfdot") is None
    cut = SimpleNamespace(row=table["rows"][CUT], row_name=CUT, steps=15, split=None, store_need=None,
                          args=SimpleNamespace(need_host_mib=None))
    assert d.Context.window_host_admission(cut, "window") is None


def test_the_default_route_asks_the_card_for_its_measured_peak(table):
    route = table["rows"][STORM]["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    assert 96_214 < 96_571 < route["need_free_mib"] <= 97_252   # over both readings, under an empty 96 GB card's free
    assert "96,214 MiB" in route["need_free_source"] and "96,571 MiB" in route["need_free_source"]


def test_a_host_too_small_for_the_default_store_runs_with_a_smaller_one(table, monkeypatch):
    """THE BREAKAGE: the 3 km adjoint-test (59 steps) was refused on every host with less than 158,958 MiB available
    (a 128 GB or 192 GB machine) although the runner's own memory is 87,805 MiB: only its default store budget, 0.4 of
    what is available, did not fit. The runner restores the same bytes from memory or from its files."""
    row = table["rows"][STORM]
    bmib = int(row["boundary_mib"])
    small = d.store_need(row, 59, 120_000, line_on=False)
    assert small["bstore_ram_gib"] == 25 and small["ram_mib"] == 25 * 1024
    assert small["host_mib"] == 119_076 <= 120_000
    assert small["files_mib"] == 60 * bmib - 25 * 1024   # the rest is on the disk, and the disk need says so
    assert small["disk_mib"] >= small["files_mib"]
    assert "(--bstore-ram-gib 25, the other 58 in the run folder)" in small["host_for"]
    # a host the default suits keeps the runner's own budget: no word added
    assert d.store_need(row, 59, 158_958, line_on=False)["bstore_ram_gib"] is None
    assert d.store_need(row, 59, 500_000, line_on=False)["bstore_ram_gib"] is None
    assert d.store_need(row, 59, 500_000, line_on=True)["bstore_ram_gib"] is None   # the gradient line: its own
    # each window: the admission carries the budget, and the runner's words take it (replacing one they name)
    ctx = SimpleNamespace(row=row, row_name=STORM, steps=59, split=None, args=SimpleNamespace(need_host_mib=None),
                          store_need=small)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 120_000)
    got = d.Context.window_host_admission(ctx, "window")
    assert got == {"window": "window", "available_mib": 120_000, "need_mib": 119_076,
                   "store_in_memory_mib": 25 * 1024, "bstore_ram_gib": 25}
    assert d.with_bstore_ram_gib(["runner", "window", "--steps", "59"], 25) == [
        "runner", "window", "--steps", "59", "--bstore-ram-gib", "25"]
    assert d.with_bstore_ram_gib(["runner", "--bstore-ram-gib", "150", "--steps", "59"], 25) == [
        "runner", "--bstore-ram-gib", "25", "--steps", "59"]

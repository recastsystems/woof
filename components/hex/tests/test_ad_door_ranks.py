# SPDX-License-Identifier: Apache-2.0
"""The rank route of the adjoint doors: --cards with N > 1 cards on a row with a rank split (table rank_splits)
runs the window as N ranks, one a card (ad-binder ranks), and the door gathers one global answer. A row without a
split, and every one-card door, runs as before. CPU only: the cards are stand-ins (compute capability, total and
free memory), and the runner is a stand-in script that writes what ad-binder ranks writes."""
import argparse
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets
from woof.hex.ad import ranks as rk


@pytest.fixture(autouse=True)
def _stand_in_runner_on_hidden_cards(monkeypatch):
    # The battery hides the cards (conftest CUDA_VISIBLE_DEVICES=-1) and these tests start a stand-in runner that
    # opens no device, so the door's hidden-card refusal (refuse_hidden_card) is let through here only.
    from woof.hex import ad_doors as _d
    monkeypatch.setattr(_d, "refuse_hidden_card", lambda: None)

ROW = "hrrrv5-x1.40962"


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


class Reached(Exception):
    pass


@pytest.fixture
def cards(monkeypatch, tmp_path):
    """Stand-in cards: N cards of 97,887 MiB with 97,000 free, a host with 500,000 MiB, 100,000 MiB of /dev/shm.
    The door stops at its needs line (printed just before the asset root) with the Context it built."""
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("WOOF_HEX_AD_ASSETS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(assets, "asset_root", lambda cache=None: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    state = {"total": {}, "free": {}}
    monkeypatch.setattr(d, "total_mib", lambda card: state["total"].get(card, 97887))
    monkeypatch.setattr(d, "free_mib", lambda card: state["free"].get(card, 97000))
    monkeypatch.setattr(rk, "shm_free_mib", lambda path="/dev/shm": state.get("shm", 100000))
    seen = {}

    def stop(self):
        seen["ctx"] = self
        raise Reached()
    monkeypatch.setattr(d.Context, "needs_line", stop)
    state["seen"] = seen
    return state


def _context(argv, door):
    try:
        d.Context(_parse([door, *argv]), door)
    except Reached:
        pass


@pytest.mark.parametrize("n", [2, 4])
def test_n_cards_on_a_split_row_run_as_n_ranks_with_every_card_admitted(cards, tmp_path, monkeypatch, n):
    admitted = []
    real = d.admit_card
    monkeypatch.setattr(d, "admit_card", lambda card, need, wait, archs=None: admitted.append((card, need))
                        or real(card, need, wait, archs))
    monkeypatch.setattr(d, "refuse_rank_pkg_not_pinned", lambda *a: None)   # the package is pinned (not yet: owed)
    monkeypatch.setattr(d, "refuse_row_not_in_pinned_assets", lambda *a: None)
    given = ",".join(str(i) for i in range(n))
    _context(["--cards", given, "--out", str(tmp_path / "o")], "adjoint-test")
    ctx = cards["seen"]["ctx"]
    split = d._ranks.choose(d.load_table()["rank_splits"][ROW], ROW, n, "adjoint-test")
    assert ctx.split["ranks"] == n and ctx.cards == [str(i) for i in range(n)]
    assert admitted == [(str(i), split["need_free_mib_rank"]) for i in range(n)]
    assert ctx.need_host_mib() == n * split["need_host_mib_rank"]
    assert "AD_TAPE_UNREAD" not in ctx.row["asset_env"] and ctx.row["self_dot_dead"] is None
    # the rank kernel set is the row's own plus the owner-reduced row, built by the door after the asset root
    assert ctx.split["ownerreduce"]["selected"] == "ranks/ownerreduce"
    assert ctx.row["assets"]["kernels"] == d.load_table()["rows"][ROW]["assets"]["kernels"]


def test_eight_cards_on_a_row_with_no_eight_rank_plan_are_refused_by_name(cards, tmp_path):
    with pytest.raises(d.AdDoorError, match=r"no 8-rank plan package for this row \(it has 2, 4\)"):
        _context(["--cards", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "o")], "adjoint-test")
    assert "ctx" not in cards["seen"]


def test_more_cards_than_the_row_splits_over_are_refused(cards, tmp_path, monkeypatch):
    table = d.load_table()
    table["rank_splits"][ROW]["cards_max"] = 4
    monkeypatch.setattr(d, "load_table", lambda path=None: table)
    with pytest.raises(d.AdDoorError, match="8 cards given; this row splits over at most 4"):
        _context(["--cards", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "o")], "sensitivity")


def test_a_card_too_small_for_a_rank_is_refused_before_any_download(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "refuse_rank_pkg_not_pinned", lambda *a: None)
    cards["total"]["1"] = 49140   # a 48 GB card beside a 96 GB one
    with pytest.raises(d.AdDoorError, match="card 1 has 49140 MiB in all; this row's window needs 60745 MiB"):
        _context(["--cards", "0,1", "--out", str(tmp_path / "o")], "adjoint-test")
    cards["total"]["1"] = 97887
    cards["free"]["1"] = 30000    # the second card held by another job
    with pytest.raises(d.AdDoorError, match="card 1 has 30000 MiB free; this row's window needs 60745 MiB"):
        _context(["--cards", "0,1", "--out", str(tmp_path / "o")], "adjoint-test")
    assert not (tmp_path / "o").exists()


def test_a_host_or_dev_shm_too_small_for_the_ranks_is_refused(cards, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "refuse_rank_pkg_not_pinned", lambda *a: None)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 100000)
    with pytest.raises(d.HostMemoryError, match="needs about 107,724 MiB"):
        _context(["--cards", "0,1", "--out", str(tmp_path / "o")], "adjoint-test")
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)
    cards["shm"] = 10000
    with pytest.raises(d.AdDoorError, match="/dev/shm has 10,000 MiB free .* about 22,888 MiB"):
        _context(["--cards", "0,1", "--out", str(tmp_path / "o")], "adjoint-test")


def test_a_rank_package_the_downloads_do_not_hold_is_refused_before_any_download(cards, tmp_path):
    with pytest.raises(d.AdDoorError, match="rank plan package ranks/hrrrv5-x1.40962-r2/pk120-r2 is not in the files"):
        _context(["--cards", "0,1", "--out", str(tmp_path / "o")], "adjoint-test")


def test_tangent_and_param_grad_on_a_split_row_say_they_do_not_run_as_ranks_yet(cards, tmp_path):
    for door in ("tangent", "param-grad"):
        with pytest.raises(d.AdDoorError, match=f"the {door} door does not run as ranks yet"):
            _context(["--cards", "0,1", "--out", str(tmp_path / "o")], door)


def test_without_cards_or_on_a_row_with_no_split_the_door_is_as_before(cards, tmp_path, monkeypatch):
    _context(["--out", str(tmp_path / "o")], "adjoint-test")
    ctx = cards["seen"].pop("ctx")
    assert ctx.split is None and ctx.cards == ["0"] and ctx.need_mib() == 73156
    _context(["--card", "1", "--out", str(tmp_path / "o")], "sensitivity")
    ctx = cards["seen"].pop("ctx")
    assert ctx.split is None and ctx.cards == ["1"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")   # a scheduler's two cards, no --cards: as before
    _context(["--out", str(tmp_path / "o")], "adjoint-test")
    ctx = cards["seen"].pop("ctx")
    assert ctx.split is None and ctx.cards == ["0"]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    _context(["--row", "hrrrv5-full-x1.40962", "--cards", "0,1", "--out", str(tmp_path / "o")], "adjoint-test")
    ctx = cards["seen"].pop("ctx")
    assert ctx.split is None and ctx.cards == ["0"]


def test_the_ranks_line_wraps_the_window_line():
    argv = ["/x/ad-binder", "window", "--config", "plan.json", "--out", "/o/window"]
    assert rk.ranks_argv(argv, 2, ["3", "5"], Path("/a/pk")) == [
        "/x/ad-binder", "ranks", "--ranks", "2", "--cards", "3,5", "--pkg", "/a/pk", "--",
        "window", "--config", "plan.json", "--out", "/o/window"]
    with pytest.raises(rk.RankSplitError, match="2 ranks need 2 cards"):
        rk.ranks_argv(argv, 2, ["0"], Path("/a/pk"))


def test_rank_plans_and_their_maps_are_named_when_missing(tmp_path):
    prefix = tmp_path / "pk"
    with pytest.raises(rk.RankSplitError, match=r"pk-r0/plan.json and 1 more missing"):
        rk.refuse_rank_plans_missing(prefix, 2)
    for k in range(2):
        (tmp_path / f"pk-r{k}").mkdir()
        (tmp_path / f"pk-r{k}" / "plan.json").write_text("{}")
    with pytest.raises(rk.RankSplitError, match=r"pk-r0/rank-gather.json and 1 more missing"):
        rk.refuse_rank_plans_missing(prefix, 2)


# ------------------------------------------------------------------- a 6-cell, 2-level mesh cut in two ranks
# global cells 0..5; rank 0 owns 0,1,2 and holds 3 as halo; rank 1 owns 3,4,5 and holds 2 as halo
L2G = [[0, 1, 2, 3], [3, 4, 5, 2]]


def _maps(tmp_path):
    maps = []
    for k, l2g in enumerate(L2G):
        rdir = tmp_path / f"pk-r{k}"
        rdir.mkdir(exist_ok=True)
        np.asarray(l2g, "<i4").tofile(rdir / "l2g-cell.i32")
        doc = {"schema": rk.GATHER_SCHEMA, "rank": k, "ranks": 2,
               "entities": {"cell": {"l2g": "l2g-cell.i32", "owned": 3, "local": 4}},
               # global theta_m: two copies (alloc 10 and 11); local copies in allocs 100+k and 200+k
               "fields": {"theta_m": {"grid": "cell",
                                      "global": [{"alloc": 10, "offset": 0, "shape": [2, 6]},
                                                 {"alloc": 11, "offset": 8, "shape": [2, 6]}],
                                      "local": [{"alloc": 100 + k, "offset": 0, "shape": [2, 4]},
                                                {"alloc": 200 + k, "offset": 4, "shape": [2, 4]}]}}}
        (rdir / rk.GATHER_FILE).write_text(json.dumps(doc))
        maps.append(rk.load_gather(rdir))
    return maps


def test_the_objective_goes_to_the_owner_and_the_direction_to_every_copy(tmp_path):
    maps = _maps(tmp_path)
    # J = sum over level 1 of copy 0 of every cell, weight 1 (word = level * 6 + cell)
    objective = [[10, 6 + c, 1.0] for c in range(6)]
    cut = [rk.cut_terms(objective, m, owner_only=True) for m in maps]
    assert sorted(t[1] for t in cut[0]) == [4 + 0, 4 + 1, 4 + 2] and all(t[0] == 100 for t in cut[0])
    assert sorted(t[1] for t in cut[1]) == [4 + 0, 4 + 1, 4 + 2] and all(t[0] == 101 for t in cut[1])
    assert sum(len(c) for c in cut) == 6   # every global word once: J is the same sum
    # a direction on cell 2 (owned by rank 0, halo of rank 1) and cell 3 (the other way), copy 1 at byte offset 8
    direction = [[11, 2 + 0 * 6 + 2, 0.5], [11, 2 + 1 * 6 + 3, -1.0]]
    d0, d1 = (rk.cut_terms(direction, m, owner_only=False) for m in maps)
    assert d0 == [[200, 1 + 2, 0.5], [200, 1 + 4 + 3, -1.0]]
    assert d1 == [[201, 1 + 3, 0.5], [201, 1 + 4 + 0, -1.0]]
    with pytest.raises(rk.RankSplitError, match="allocation 12 word 0 is in no field"):
        rk.cut_terms([[12, 0, 1.0]], maps[0], owner_only=True)


def test_the_gradient_is_every_ranks_copies_summed_onto_the_global_mesh(tmp_path):
    maps = _maps(tmp_path)
    rng = np.random.default_rng(3)
    # each rank allocation: copy 0 at word 0 (alloc 100+k), copy 1 at word 1 (alloc 200+k), 8 words each
    local = {(k, a): rng.standard_normal(9).astype(np.float32) for k in range(2) for a in (100 + k, 200 + k)}
    got, rep = rk.gather_allocs(maps, [10, 11], lambda r, a: local[(r, a)], {10: 12, 11: 14})
    for ga, la, off in ((10, 100, 0), (11, 200, 1)):
        want = np.zeros((2, 6))
        for k, l2g in enumerate(L2G):
            words = local[(k, la + k)][off:off + 8].reshape(2, 4)
            for p, g in enumerate(l2g):
                want[:, g] += words[:, p]
        assert np.array_equal(got[ga][off * 2:off * 2 + 12].reshape(2, 6), want.astype(np.float32))
    assert rep[10]["covered_words"] == 12 and rep[10]["halo_words_nonzero"] == 4


# ------------------------------------------------------------------- the ranks' receipts, and a stand-in runner
def _rank_receipts(out, rel=(1e-8, 1e-8), transport=(None, None)):
    out.mkdir(parents=True, exist_ok=True)
    for k in range(2):
        (out / f"r{k}").mkdir(exist_ok=True)
        peer = {"rank": k, "ranks": 2}
        if transport[k]:
            peer["transport"] = transport[k]
        (out / f"r{k}" / "window.json").write_text(json.dumps(
            {"tangent_dir": {"w_Mx": 2.0, "x_MTw": 2.0, "rel": rel[k]}, "peer": peer, "refused": {}}))
        (out / f"r{k}.stderr").write_text(json.dumps({"window_tangent_step": 1, "reference_check": True,
                                                       "boundary_hashes": "equal"}) + "\n")
    (out / "ranks.json").write_text(json.dumps({"schema": "ad.ranks/v1", "ranks": 2, "cards": "0,1",
                                                "transport": "one tmpfs mailbox", "rank_rows": [], "wall_s": 1.0}))


def test_one_receipt_from_the_ranks_and_a_disagreement_is_refused(tmp_path):
    _rank_receipts(tmp_path / "a")
    doc = rk.compose_window_doc(tmp_path / "a", 2, "file mailbox")
    assert doc["tangent_dir"]["rel"] == 1e-8 and doc["ranks"]["exchange_seen"] == ["file mailbox"] * 2
    _rank_receipts(tmp_path / "b", rel=(1e-8, 2e-8))
    with pytest.raises(rk.RankSplitError, match="the ranks disagree on tangent_dir.rel"):
        rk.compose_window_doc(tmp_path / "b", 2)
    # asked for the shared slots, and the runner used the file mailbox (a runner without AD_PEER_SHM)
    with pytest.raises(rk.RankSplitError, match="asked for exchange 'shm fenced .* report \\['file mailbox'"):
        rk.compose_window_doc(tmp_path / "a", 2, "shm fenced (AD_PEER_SHM=1)")
    _rank_receipts(tmp_path / "c", transport=("shm fenced (AD_PEER_SHM=1)",) * 2)
    assert rk.compose_window_doc(tmp_path / "c", 2, "shm fenced (AD_PEER_SHM=1)")["ranks"]["exchange_seen"] == [
        "shm fenced (AD_PEER_SHM=1)"] * 2


def test_run_window_runs_the_launcher_and_reads_every_rank(tmp_path, monkeypatch):
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(d, "process_card_mib", lambda pid: {})
    runner = tmp_path / "ad-binder"
    # a stand-in launcher: writes what ad-binder ranks writes (ranks.json, r<k>/window.json, r<k>.stderr)
    runner.write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
a = sys.argv
assert a[1:4] == ["ranks", "--ranks", "2"] and a[a.index("--") + 1] == "window", a
out = Path(a[a.index("--out", a.index("--")) + 1])
for k in range(2):
    (out / f"r{{k}}").mkdir(parents=True, exist_ok=True)
    (out / f"r{{k}}" / "window.json").write_text(json.dumps({{"tangent_dir": {{"w_Mx": 1.0, "x_MTw": 1.0, "rel": 3e-9}},
                                                             "peer": {{"rank": k}}, "refused": {{}}}}))
    (out / f"r{{k}}.stderr").write_text(json.dumps({{"window_tangent_step": 1, "reference_check": True,
                                                     "boundary_hashes": "equal"}}) + "\\n")
(out / "ranks.json").write_text(json.dumps({{"ranks": 2, "cards": a[5], "rank_rows": []}}))
""")
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    window = [str(runner), "window", "--config", "plan.json", "--out", str(tmp_path / "o" / "window")]
    argv = rk.ranks_argv(window, 2, ["0", "1"], tmp_path / "pk")
    doc = d.run_window(argv, {}, tmp_path, tmp_path / "o" / "window", "0,1", ranks=2, exchange_label="file mailbox")
    assert doc["tangent_dir"]["rel"] == 3e-9 and doc["ranks"]["cards"] == "0,1"
    assert doc["door_validity"]["valid"] is True and doc["door_validity"]["boundaries_checked"] == 2
    assert [r["valid"] for r in doc["door_validity"]["ranks"]] == [True, True]


def test_a_split_says_it_is_experimental_before_it_runs_and_a_one_card_run_does_not(cards, tmp_path, monkeypatch,
                                                                                      capsys):
    monkeypatch.setattr(d, "refuse_rank_pkg_not_pinned", lambda *a: None)
    monkeypatch.setattr(d, "refuse_row_not_in_pinned_assets", lambda *a: None)
    _context(["--cards", "0,1", "--out", str(tmp_path / "o")], "sensitivity")
    out = capsys.readouterr().out
    lines = [x for x in out.splitlines() if "EXPERIMENTAL" in x]
    assert len(lines) == 1 and "not been shown equal to the one-card gradient" in lines[0]
    assert "--card N" in lines[0]
    ctx = cards["seen"]["ctx"]
    ctx.rank_prefix = tmp_path / "r"   # set after the asset root in a real run
    assert ctx.split_block()["experimental"]["line"] in lines[0]   # the receipt block carries the same line
    cards["seen"].clear()
    _context(["--card", "0", "--out", str(tmp_path / "o1")], "sensitivity")
    assert "EXPERIMENTAL" not in capsys.readouterr().out and cards["seen"]["ctx"].split is None


def test_every_split_row_is_experimental_and_the_storm_box_line_names_what_differed():
    table = d.load_table()["rank_splits"]
    rows = {k: v for k, v in table.items() if isinstance(v, dict) and "plans" in v}
    assert rows and all(v["experimental"]["line"] and v["experimental"]["until"] for v in rows.values())
    storm = rows["hrrrv5-q3.96.38278.n40.60w91.30"]["experimental"]["line"]
    assert "12 of 13" in storm and "7.4 per cent" in storm
    for name, v in rows.items():
        for n in v["plans"]:
            assert rk.choose(v, name, int(n), v["doors"][0])["experimental"] == v["experimental"]


def test_a_made_row_with_rank_plans_takes_the_made_split_and_drops_the_one_process_host_budget():
    table = d.load_table()
    row = {"generated": True, "need_free_mib": 90660, "window_args": ["--anchors", "1", "--host-budget-gb", "320",
                                                                      "--vram-budget-gb", "93"],
           "assets": {}, "window_env": {"AD_TAPE_UNREAD": "x", "AD_ANCHORS_AHEAD": "0"}}
    split = rk.made_split(table, row)
    assert split is not None and rk.made_split(table, {**row, "generated": False}) is None
    s = rk.choose(split, "hrrrv5-made", 4, "sensitivity")
    assert s["ranks"] == 4 and s["need_free_mib_rank"] == 90660 and s["exchange_env"] == {"AD_PEER_SHM": "1"}
    assert s["experimental"]["line"]
    out = d.split_row(row, s)
    assert out["window_args"] == ["--anchors", "1", "--vram-budget-gb", "93"]
    assert out["window_env"] == {"AD_ANCHORS_AHEAD": "0"}
    with pytest.raises(rk.RankSplitError):
        rk.choose(split, "hrrrv5-made", 9, "sensitivity")

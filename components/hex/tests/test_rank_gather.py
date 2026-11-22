# SPDX-License-Identifier: Apache-2.0
"""rank-gather.json: the writer (from the chain's aligned views and the ownership files), the one reader the door and
tc3_views.py gather share, and the door's use of it on the 3 km storm box's rank entries. CPU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets
from woof.hex.ad import ownerreduce as own
from woof.hex.ad import rank_gather as rg
from woof.hex.ad import ranks as rk

# ----------------------------------------------------------------------------------------------------------------
# A limited-area mesh of 6 cells (plus the regional padding cell), 2 levels, cut in two ranks. Rank 0 owns 0,1,2 and
# holds 3 as halo; rank 1 owns 3,4,5 and holds 2 as halo. Three fields, laid out the ways the recordings lay them out:
#   A  [level, cell+pad]  in allocation 10 at byte 0        (entity axis last)
#   R  [3] replicated     in allocation 10 at byte 56       (no entity axis)
#   B  [cell+pad, level]  in allocation 11 at byte 8        (entity axis first)
# Each rank: A [2, 4+1] in allocation 100+k at byte 0, R [3] at byte 40, B [4+1, 2] in allocation 200+k at byte 4.
L2G = [[0, 1, 2, 3], [3, 4, 5, 2]]
OWNED = 3


def _view(alloc, offset, shape):
    strides, acc = [], 4
    for s in reversed(shape):
        strides.insert(0, acc)
        acc *= s
    return {"allocation": alloc, "offset": offset, "shape": list(shape), "strides": strides, "dtype": "float32"}


def _views_and_owners():
    ranks, owners = [], []
    for k, l2g in enumerate(L2G):
        b = {"fields.A": {"native": _view(10, 0, [2, 7]), "rank": _view(100 + k, 0, [2, 5]),
                          "entity_axis": [1, "cell", "local", 1]},
             "fields.R": {"native": _view(10, 56, [3]), "rank": _view(100 + k, 40, [3]), "entity_axis": None},
             "fields.B": {"native": _view(11, 8, [7, 2]), "rank": _view(200 + k, 4, [5, 2]),
                          "entity_axis": [0, "cell", "local", 1]}}
        ranks.append({"rank": k, "boundaries": {"0": b, "12": b}})
        owners.append({"cell_l2g": np.asarray(l2g), "n_owned_cells": OWNED,
                       "edge_l2g": np.asarray([0]), "n_owned_edges": 1,
                       "vertex_l2g": np.asarray([0]), "n_owned_vertices": 1})
    return {"ranks": ranks, "steps": ["0", "12"]}, owners


def _maps(tmp_path, global_plan=None):
    views, owners = _views_and_owners()
    docs = rg.from_views(views, owners, ["0"], global_plan)
    for k, doc in enumerate(docs):
        rg.save(doc, tmp_path / f"pk-r{k}")
    return [rg.load(tmp_path / f"pk-r{k}") for k in range(2)]


def _planted():
    rng = np.random.default_rng(7)
    return {(k, a): rng.standard_normal(13).astype(np.float32) for k in range(2) for a in (100 + k, 200 + k)}


def _hand_sum(local):
    """The gradient on the one-card allocations, summed by hand: every rank's mesh copies, owned and halo, into their
    global cell; the padding entry maps to nothing; the replicated words summed over the ranks."""
    g10, g11 = np.zeros(17), np.zeros(16)
    for k, l2g in enumerate(L2G):
        a, b = local[(k, 100 + k)], local[(k, 200 + k)]
        for lev in range(2):
            for p, c in enumerate(l2g):
                g10[lev * 7 + c] += a[lev * 5 + p]
                g11[2 + c * 2 + lev] += b[1 + p * 2 + lev]
        g10[14:17] += a[10:13]
    return {10: g10.astype(np.float32), 11: g11.astype(np.float32)}


def test_the_writer_makes_one_map_per_rank_with_the_ownership_and_every_copy(tmp_path):
    maps = _maps(tmp_path, {"path": "/p", "plan_sha256": "ab"})
    for k, m in enumerate(maps):
        assert (tmp_path / f"pk-r{k}" / "rank-gather.json").is_file()
        assert m["entities"]["cell"]["owned"] == 3 and m["entities"]["cell"]["local"] == 4
        assert m["entities"]["cell"]["_l2g"].tolist() == L2G[k]
        assert sorted(m["fields"]) == ["fields.A", "fields.B", "fields.R"]
        assert m["fields"]["fields.A"]["local"][0]["count"] == 4 and m["fields"]["fields.R"]["grid"] is None
        assert m["global_plan"]["plan_sha256"] == "ab"


def test_a_planted_rank_gradient_gathers_to_the_global_order_word_for_word_halo_summed(tmp_path):
    maps = _maps(tmp_path)
    local = _planted()
    got, rep = rg.gather_allocs(maps, [10, 11], lambda r, a: local[(r, a)], {10: 17, 11: 16})
    want = _hand_sum(local)
    assert np.array_equal(got[10], want[10]) and np.array_equal(got[11], want[11])
    # 2 halo words of A and 2 of B on each rank carry gradient: owned-only would have dropped them
    assert rep[10]["halo_words_nonzero"] == 4 and rep[11]["halo_words_nonzero"] == 4
    # the padding cell is no rank's: it stays 0
    assert got[10][6] == 0 and got[10][13] == 0 and got[11][2 + 12] == 0 and got[11][2 + 13] == 0
    # the door's ranks module reads through the same functions (one reader)
    assert rk.gather_allocs is rg.gather_allocs and rk.cut_terms is rg.cut_terms and rk.RankSplitError is rg.RankGatherError


def test_the_objective_is_cut_to_the_owner_once_and_a_direction_to_every_copy(tmp_path):
    maps = _maps(tmp_path)
    objective = [[10, lev * 7 + c, 1.0 + c] for lev in range(2) for c in range(6)]   # A, every mesh word
    cut = [rg.cut_terms(objective, m, owner_only=True) for m in maps]
    assert sum(len(c) for c in cut) == 12 and sum(t[2] for c in cut for t in c) == sum(t[2] for t in objective)
    # rank 1 owns cells 3,4,5 at local 0,1,2
    assert sorted((t[1], t[2]) for t in cut[1]) == sorted((lev * 5 + p, 1.0 + c) for lev in range(2)
                                                          for p, c in enumerate([3, 4, 5]))
    direction = [[11, 2 + 2 * 2 + 1, 0.5]]   # B at cell 2, level 1: rank 0 owns it, rank 1 holds it as halo
    assert rg.cut_terms(direction, maps[0], owner_only=False) == [[200, 1 + 2 * 2 + 1, 0.5]]
    assert rg.cut_terms(direction, maps[1], owner_only=False) == [[201, 1 + 3 * 2 + 1, 0.5]]
    with pytest.raises(rg.RankGatherError, match="replicated copy: it has no one owner"):
        rg.cut_terms([[10, 15, 1.0]], maps[0], owner_only=True)
    with pytest.raises(rg.RankGatherError, match="allocation 12 word 0 is in no field"):
        rg.cut_terms([[12, 0, 1.0]], maps[0], owner_only=True)


def test_the_rank_maps_must_name_the_plan_the_door_was_given(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "plan.json").write_text("{}")
    sha = rg.plan_sha256(plan)
    assert rg.refuse_other_plan(_maps(tmp_path / "a", {"path": str(plan), "plan_sha256": sha}), plan) == "equal"
    assert rg.refuse_other_plan(_maps(tmp_path / "b"), plan) == "not recorded in the maps"
    with pytest.raises(rg.RankGatherError, match="give the one-card plan the rank package was split from"):
        rg.refuse_other_plan(_maps(tmp_path / "c", {"path": "/x", "plan_sha256": "0" * 64}), plan)


def test_the_cli_writes_and_gathers_through_the_same_reader(tmp_path):
    views, owners = _views_and_owners()
    (tmp_path / "views.json").write_text(json.dumps(views))
    od = tmp_path / "owners"
    od.mkdir()
    for k, z in enumerate(owners):
        np.savez(od / f"rank-{k}.npz", **z)
    assert rg.main(["write", "--views", str(tmp_path / "views.json"), "--owners", str(od),
                    "--rank-prefix", str(tmp_path / "pk"), "--steps", "0"]) == 0
    local = _planted()
    dirs = []
    for k in range(2):
        rd = tmp_path / f"w{k}"
        rd.mkdir()
        for a in (100 + k, 200 + k):
            local[(k, a)].tofile(rd / f"grad-{a}.f32")
        dirs.append(str(rd))
    (tmp_path / "catalog.json").write_text(json.dumps({"allocations": [{"id": 10, "bytes": 68}, {"id": 11, "bytes": 64}]}))
    assert rg.main(["gather", "--rank-prefix", str(tmp_path / "pk"), "--ranks", "2", "--rank-dirs", ",".join(dirs),
                    "--allocs", "10,11", "--native-catalog", str(tmp_path / "catalog.json"),
                    "--out", str(tmp_path / "g")]) == 0
    want = _hand_sum(local)
    for a in (10, 11):
        assert np.array_equal(np.fromfile(tmp_path / "g" / f"grad-{a}.f32", "<f4"), want[a])


def test_the_door_gathers_rank_dumps_into_one_card_allocations_and_reads_them_as_on_one_card(tmp_path):
    maps = _maps(tmp_path)
    local = _planted()
    win = tmp_path / "window"
    for k in range(2):
        (win / f"r{k}").mkdir(parents=True)
        for a in (100 + k, 200 + k):
            local[(k, a)].tofile(win / f"r{k}" / f"grad-{a}.f32")

    class Ctx:
        rank_maps = maps
    f = {"name": "A", "copies": [(10, 0)], "copy_shapes": [[2, 7]], "shape": [2, 6]}
    rep = d.Context.gather_rank_grads(Ctx(), win, [f])
    assert rep["allocs"]["10"]["halo_words_nonzero"] == 4 and (win / "gather.json").is_file()
    assert np.array_equal(d._sum_copies(win, f), _hand_sum(local)[10][:14].reshape(2, 7)[:, :6].astype(np.float64))
    # each rank's dump list: the rank allocations holding a copy of the global ones
    assert rg.local_allocs(maps[1], [10]) == [101] and rg.local_allocs(maps[1], [10, 11]) == [101, 201]


# ---------------------------------------------------------------------------- the owner-reduced kernel set
F = own.FUNCTION


def _kset(root: Path, cubins: Path | None = None) -> Path:
    root.mkdir(parents=True)
    cub = cubins or root / "cubins"
    cub.mkdir(parents=True, exist_ok=True)
    (root / "specs").mkdir()
    (root / "kernels.json").write_text(json.dumps({"kernels": [{"function": "a", "derivative": {}},
                                                               {"function": F, "derivative": {"spec": {}}, "old": 1}]}))
    (cub / "a.fatbin").write_bytes(b"A")
    (cub / f"{F}.fatbin").write_bytes(b"OLD")
    (root / "specs" / "a.json").write_text("{}")
    return root / "kernels.json"


def _selected(root: Path) -> Path:
    (root / "specs").mkdir(parents=True)
    (root / "cubins").mkdir()
    (root / "kernels.json").write_text(json.dumps({"kernels": [{"function": F, "derivative": {"spec": {}}, "new": 1}]}))
    (root / "specs" / f"{F}.json").write_text('{"selected": true}')
    (root / "cubins" / f"{F}.fatbin").write_bytes(b"NEW")
    (root / "cubins" / f"{F}.provenance.json").write_text("{}")
    return root


@pytest.mark.parametrize("apart", [False, True])
def test_the_door_builds_the_owner_reduced_set_from_the_rows_own_set_as_add_selected_row_does(tmp_path, apart):
    cubins = tmp_path / "imgs" if apart else None
    base = _kset(tmp_path / "base", cubins)
    sel = _selected(tmp_path / "sel")
    got = own.ensure(base, sel, tmp_path / "cache", cubins=cubins)
    k = json.loads(got["kernels"].read_text())
    rows = {r["function"]: r for r in k["kernels"]}
    assert rows[F].get("new") == 1 and "old" not in rows[F] and set(rows) == {"a", F}
    assert Path(rows[F]["derivative"]["spec"]["path"]).parent == got["kernels"].parent / "specs"
    assert Path(rows[F]["derivative"]["spec"]["path"]).read_text() == '{"selected": true}'
    assert (got["cubins"] / "a.fatbin").read_bytes() == b"A" and (got["cubins"] / f"{F}.fatbin").read_bytes() == b"NEW"
    assert own.ensure(base, sel, tmp_path / "cache", cubins=cubins) == got   # built once
    with pytest.raises(own.OwnerReduceError, match="is missing"):
        own.ensure(base, tmp_path / "nothing", tmp_path / "cache", cubins=cubins)


# ------------------------------------------------------------------------- the storm box's rank entries
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


class Reached(Exception):
    pass


@pytest.fixture
def storm(monkeypatch, tmp_path):
    """Stand-in cards (97,887 MiB, 97,000 free), a 500,000 MiB host, 100,000 MiB of /dev/shm, and the storm box row
    (a copy of the 120 km row under the storm row's name: the --mesh path that makes it needs the mesh files, and is
    run for real in a dry run). The door stops at its needs line."""
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("WOOF_HEX_AD_ASSETS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(assets, "asset_root", lambda cache=None: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)
    monkeypatch.setattr(d, "refuse_row_not_in_pinned_assets", lambda *a: None)
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    state = {"total": {}, "free": {}, "seen": {}}
    monkeypatch.setattr(d, "total_mib", lambda card: state["total"].get(card, 97887))
    monkeypatch.setattr(d, "free_mib", lambda card: state["free"].get(card, 97000))
    monkeypatch.setattr(rk, "shm_free_mib", lambda path="/dev/shm": 100000)
    table = d.load_table()
    table["rows"][STORM] = dict(table["rows"]["hrrrv5-x1.40962"], mesh="q3.96.38278.n40.60w91.30", cells=38278,
                                window_env={"AD_MYNN_FWD_LANES": "1", "X": "1"})
    monkeypatch.setattr(d, "load_table", lambda path=None: table)

    def stop(self):
        state["seen"]["ctx"] = self
        raise Reached()
    monkeypatch.setattr(d.Context, "needs_line", stop)
    return state


def _ctx(argv, door="sensitivity"):
    try:
        d.Context(_parse([door, "--row", STORM, *argv]), door)
    except Reached:
        pass


@pytest.mark.parametrize("n,exchange", [(2, "file"), (4, "shm-fenced"), (8, "shm-fenced")])
def test_the_storm_box_runs_as_2_4_or_8_ranks_with_its_rank_plans_named(storm, tmp_path, n, exchange):
    _ctx(["--cards", ",".join(str(i) for i in range(n)), "--rank-plans", str(tmp_path / "pk"),
          "--out", str(tmp_path / "o")])
    ctx = storm["seen"]["ctx"]
    assert ctx.split["ranks"] == n and ctx.cards == [str(i) for i in range(n)] and ctx.split["exchange"] == exchange
    assert ctx.need_mib() == 90795 and ctx.need_host_mib() is None
    assert ctx.split["ownerreduce"]["selected"] == "ranks/ownerreduce"
    assert "AD_MYNN_FWD_LANES" not in ctx.row["window_env"] and ctx.row["window_env"]["X"] == "1"


def test_the_storm_box_rank_package_must_be_given_or_pinned(storm, tmp_path):
    with pytest.raises(d.AdDoorError, match=r"ranks/hrrrv5-q3.96.38278.n40.60w91.30-r2/storm-r2 is not in the files.*"
                                            r"--rank-plans PREFIX"):
        _ctx(["--cards", "0,1", "--out", str(tmp_path / "o")])
    with pytest.raises(d.AdDoorError, match="this row splits over at most 8"):
        _ctx(["--cards", ",".join(str(i) for i in range(16)), "--rank-plans", str(tmp_path / "pk"),
              "--out", str(tmp_path / "o")])
    storm["total"]["1"] = 32607   # a 32 GB card does not hold a storm-box rank
    with pytest.raises(d.AdDoorError, match="card 1 has 32607 MiB in all; this row's window needs 90795 MiB"):
        _ctx(["--cards", "0,1", "--rank-plans", str(tmp_path / "pk"), "--out", str(tmp_path / "o")])

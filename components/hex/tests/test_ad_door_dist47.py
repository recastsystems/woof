# SPDX-License-Identifier: Apache-2.0
"""dist47 door facts (processors only): the gradient test on several independent directions, each with a seed of its
own, where a direction the forecast cannot resolve never counts; the fast-against-exact figures printed from the
table's measured ones, rounded up; the "forward not kept" line said once in a command; and a command that starts
while a program told to end by another command is still giving its memory back.

THE BREAKAGES THESE HOLD SHUT: the gradient test ran its finite-difference legs along one direction, a mid-level
temperature bump, and a gradient wrong in its moisture part passed it; the command printed moisture 1.1e-06 and
stored pressure 9.6e-05 where the card run read 1.315e-06 and 9.778e-05; a session command printed one line twice;
and a command that arrived while a stopped program was ending read the card at once and was refused.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_session as s

STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
R120 = "hrrrv5-x1.40962"
_LOAD = d.load_table        # the table as the package holds it (a test below stands its own in the door's place)

# ---------------------------------------------------------------------------------------------- a stand-in window
NLEV, NCOL = 9, 48
NAMES = ("theta_m", "qv", "rho_zz", "rho_p", "rtheta_p", "rho_theta", "exner", "pressure_p")


class _Map:
    """Each field in its own allocation with a second copy that is padded by one column and starts two words in."""
    window_steps = 3

    def resolve(self, table, name, row=None, window_end=False):
        i = NAMES.index(name)
        return {"name": name, "units": "unit", "grid": "cell", "what": name, "alloc": 100 + i, "offset": 0,
                "shape": [NLEV, NCOL], "read_shape": [NLEV, NCOL], "words": NLEV * NCOL, "place": "the stand-in's copy",
                "copies": [(100 + i, 0), (200 + i, 8)], "copy_shapes": [[NLEV, NCOL], [NLEV, NCOL + 1]]}


def _lev():
    return np.arange(NLEV)[:, None] * np.ones((1, NCOL))


class _State:
    def field(self, f):
        k = _lev()
        return {"theta_m": 300.0 + 2.0 * k, "qv": 0.012 * np.exp(-k / 3.0), "rho_zz": 1.1 - 0.05 * k,
                "rho_p": 0.0 * k, "rtheta_p": 3.0 + 0.0 * k, "rho_theta": 320.0 + 0.0 * k, "exner": 0.95 - 0.02 * k,
                "pressure_p": 10.0 + 0.0 * k}[f["name"]]


def _box():
    """A patch of cells around 40.6N 91.3W, as the storm box is."""
    lat, lon = np.meshgrid(np.linspace(39.4, 41.8, 6), np.linspace(-93.0, -89.6, 8), indexing="ij")
    return {"latCell": lat.ravel(), "lonCell": lon.ravel(), "areaCell": np.ones(NCOL)}


class _Forecast:
    """A stand-in for the runner's window: the measure answers a change of the stored start linearly (the true
    gradient ``c``: on theta_m, a mean over the columns weighted to the low levels; on qv, a low-level weight), with
    a little curvature and, per window, a roughness that does not shrink with the step (``rough``) or the real
    change over the true slope given outright per step size (``ratios``). The "adjoint" hands back ``c`` with the
    sign of one field's part flipped when ``flip`` names it (the planted defect); the tangent is right."""

    def __init__(self, flip=None, rough=None, ratios=None):
        self.flip, self.rough, self.ratios, self.seen = flip, dict(rough or {}), dict(ratios or {}), []
        self.dumps = True
        k = _lev()
        self.c = {100: (np.exp(-k / 3.0) / np.exp(-k / 3.0).sum()).ravel(),
                  101: (40.0 * np.exp(-k / 2.0) / NCOL).ravel()}

    def _true(self, terms):
        return float(sum(self.c[a][w] * v for a, w, v in terms if a in self.c))

    def _adjoint(self, terms):
        sign = {100: -1.0 if self.flip == "theta_m" else 1.0, 101: -1.0 if self.flip == "qv" else 1.0}
        return float(sum(sign[a] * self.c[a][w] * v for a, w, v in terms if a in self.c))

    def window(self, ctx, extra, sub="window", door_args_key="window_args", **_):
        word = lambda flag: extra[extra.index(flag) + 1]
        out = ctx.out / sub
        out.mkdir(parents=True, exist_ok=True)
        terms = json.loads(Path(word("--taylor-dir")).read_text())["terms"]
        self.seen.append({"sub": sub, "key": door_args_key, "terms": terms, "extra": list(extra)})
        slope, predicted = self._true(terms), self._adjoint(terms)
        rough = self.rough.get(sub, 0.0)
        rows = []
        for i, h in enumerate(float(x) for x in word("--taylor").split(",")):
            dj = h * slope * (1.0 + 0.02 * h) + rough * (-1.0) ** i * (1.0 + 0.37 * i)
            if sub in self.ratios:
                dj = h * slope * self.ratios[sub][i]
            rows.append({"h": h, "dJ": dj, "fd_quotient": dj / h,
                         **({"ratio": dj / (h * predicted), "err": abs(dj / (h * predicted) - 1.0)} if predicted else {})})
        if "--dump-grad" in extra and self.dumps:   # the adjoint's own answer, every listed allocation whole
            for alloc in (int(x) for x in word("--dump-grad").split(",")):
                base, pad, off = (alloc - 100, 0, 0) if alloc < 200 else (alloc - 200, 1, 2)
                g = np.zeros((NLEV, NCOL + pad))
                if alloc in self.c:
                    g[:, :NCOL] = self.c[alloc].reshape(NLEV, NCOL) * (-1.0 if NAMES[base] == self.flip else 1.0)
                np.concatenate([np.zeros(off), g.ravel()]).astype("<f4").tofile(out / f"grad-{alloc}.f32")
        ctx.windows.append({"window": sub, "valid": True})
        big = max(abs(slope), abs(predicted), 1e-300)
        return {"tangent_dir": {"w_Mx": slope, "x_MTw": predicted, "rel": abs(slope - predicted) / big},
                "taylor": {"rows": rows}, "zero_direction": {"nonzero_words": 0}}


def _run(tmp_path, monkeypatch, forecast, directions=None, table=None, taylor=None):
    """adjoint-test through its own handler on the stand-in window; (exit code, receipt)."""
    table = table or _LOAD()

    class Ctx:
        def __init__(self, args, door):
            self.args, self.door, self.table = args, door, table
            self.row_name, self.row = "stand-in", {"dt_s": 60.0}
            self.fmap, self.geometry, self.plan = _Map(), _box(), tmp_path / "plan"
            self.out = Path(args.out)
            self.out.mkdir(parents=True, exist_ok=True)
            self.steps, self.stages, self.windows, self.split = 3, [], [], None

        def stage(self, name, t0, parallelism):
            self.stages.append(name)

        def rank_list(self, name, items, form):
            return ",".join(f"{a}:{o}:{b}" for a, o, b in items)

        def rank_json(self, path, owner_only):
            return str(path)

        def window(self, extra, **kw):
            return forecast.window(self, extra, **kw)

        def receipt(self, result, passed=None, strict=None):
            doc = {"result": result, "valid": True, "passed": bool(passed), "windows": self.windows, "wall_s": 0.0}
            (self.out / "receipt.json").write_text(json.dumps(doc))
            return doc

    monkeypatch.setattr(d, "Context", Ctx)
    monkeypatch.setattr(d, "PlanState", lambda plan: _State())
    monkeypatch.setattr(d, "gradient_route", lambda args, door: {})
    monkeypatch.setattr(d, "load_table", lambda path=None: table)
    out = tmp_path / f"at{len(list(tmp_path.glob('at*')))}"
    args = argparse.Namespace(out=str(out), at="40.6,-91.3", level=5, amplitude=None, radius_km=300.0,
                              level_width=None, metric="theta_m", levels=None, box=None, circle=None, taylor=taylor,
                              no_self_dot=True, seed=None, directions=directions)
    code = d.run_adjoint_test(args)
    return code, json.loads((out / "receipt.json").read_text())


# ---------------------------------------------------------------------- item 3: several independent directions
def test_adjoint_test_directions(tmp_path, monkeypatch, capsys):
    """One direction in the receipt before, several after, each with its seed; no two the same direction and no two
    seeds equal; a direction the forecast cannot resolve is printed UNRESOLVED and does not count; and a sign flip
    planted in one field's gradient, which the bump alone passes, fails the verdict along a resolved direction."""
    table = _LOAD()
    spec = table["doors"]["adjoint-test"]["directions"]
    rough = {"window-smooth-draw": 1.0}      # the forecast is rough along one direction: the measure jumps about
    forecast = _Forecast(rough=rough)
    code, rec = _run(tmp_path, monkeypatch, forecast)
    said = capsys.readouterr().out
    got = rec["result"]["directions"]
    assert [x["name"] for x in got] == spec["order"] and len(got) >= 3            # one before, several after
    assert [x["window"] for x in got] == ["window", "window-descent", "window-near-ground-moisture",
                                          "window-smooth-draw"]
    # every direction is a file of its own with its sha256, and the receipt lists its seed
    assert len({x["sha256"] for x in got}) == len(got) and all(Path(x["file"]).is_file() for x in got)
    assert [x["seed"] for x in got] == [spec["rows"][x["name"]]["seed"] for x in got]
    seeds = [x["seed"] for x in got if x["seed"] is not None]
    check_seed = int(table["doors"]["adjoint-test"]["self_dot_args"][1])      # the whole-state check's direction
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds) and check_seed not in seeds
    # no two are the same direction: the receipt's cosines, and the direction files' own words read again here
    for a, b in itertools.combinations(got, 2):
        assert abs(b["cosine"][a["name"]]) < 0.99 and b["independent"]
    files = {x["name"]: {(t[0], t[1]): t[2] for t in json.loads(Path(x["file"]).read_text())["terms"]} for x in got}
    for a, b in itertools.combinations(files, 2):
        dot = sum(v * files[b].get(key, 0.0) for key, v in files[a].items())
        norm = (sum(v * v for v in files[a].values()) * sum(v * v for v in files[b].values())) ** 0.5
        assert abs(dot / norm) < 0.99, (a, b)
    # each direction's ratio per step size; the rough one is UNRESOLVED and does not count; the rest are inside
    by = {x["name"]: x for x in got}
    assert all(len(x["rows"]) == 5 and all(r["ratio"] is not None for r in x["rows"]) for x in got)
    assert [by[n]["status"] for n in spec["order"]] == ["inside", "inside", "inside", "unresolved"]
    assert not any(r["counts"] for r in by["smooth-draw"]["rows"])
    assert "less than the forecast's own roughness" in by["smooth-draw"]["why"]
    over = rec["result"]["directions_verdict"]
    assert code == 0 and rec["passed"] and over["verdict"] == "pass"
    assert over["resolved"] == ["bump", "descent", "near-ground-moisture"] and over["unresolved"] == ["smooth-draw"]
    assert "UNRESOLVED" in said and "PASS: 3 directions resolved" in said and "seed 211" in said
    # the further windows are the table's own words (no second zero pass), on the first window's objective and steps
    first, later = forecast.seen[0], forecast.seen[1:]
    assert first["key"] == "window_args" and {w["key"] for w in later} == {"direction_window_args"}
    word = lambda w, flag: w["extra"][w["extra"].index(flag) + 1]
    assert all(word(w, "--objective") == word(first, "--objective") and word(w, "--taylor") == word(first, "--taylor")
               for w in later)
    assert "--dump-grad" in first["extra"] and not any("--dump-grad" in w["extra"] for w in later)
    assert not list((tmp_path / "at0" / "window").glob("grad-*.f32"))     # the dumped gradient does not stay

    # THE PLANTED DEFECT: the gradient's moisture part with the wrong sign. The bump moves no moisture: it passes.
    code, rec = _run(tmp_path, monkeypatch, _Forecast(flip="qv", rough=rough), directions="bump")
    one = rec["result"]
    assert [x["status"] for x in one["directions"]] == ["inside"] and one["checks_passed"]   # all the old test saw
    assert code == 1 and one["directions_verdict"]["verdict"] == "unresolved"       # one direction is never a PASS
    assert "one direction is not a gradient test" in capsys.readouterr().out
    code, rec = _run(tmp_path, monkeypatch, _Forecast(flip="qv", rough=rough))
    by = {x["name"]: x for x in rec["result"]["directions"]}
    assert code == 1 and not rec["passed"] and rec["result"]["checks_passed"]
    assert by["bump"]["status"] == "inside"
    assert by["near-ground-moisture"]["status"] == "outside" and any(r["counts"] for r in
                                                                     by["near-ground-moisture"]["rows"])
    assert "near-ground-moisture" in rec["result"]["directions_verdict"]["outside"]
    assert "FAIL: outside the bar along" in capsys.readouterr().out


def test_a_direction_drawn_twice_counts_once_and_unresolved_directions_never_make_a_pass(tmp_path, monkeypatch, capsys):
    """THE BREAKAGE: a "second" direction drawn with the seed of another draw was that draw, and a test on it read
    as a second, independent pass."""
    table = _LOAD()
    spec = table["doors"]["adjoint-test"]["directions"]
    spec["rows"]["twin"] = dict(spec["rows"]["near-ground-moisture"])      # the same seed: the same direction
    spec["order"] = spec["order"] + ["twin"]
    # the bump's real change jumps between step sizes, with two small ones that happen to agree: the first window's
    # own Taylor check takes those two, and the direction is still not one the forecast lets this window read
    jumpy = {"window": [1.02, 1.5, 1.005, 1.006, 1.4]}
    code, rec = _run(tmp_path, monkeypatch, _Forecast(ratios=jumpy), directions="near-ground-moisture,twin", table=table)
    assert rec["result"]["checks_passed"] and rec["result"]["checks"]["taylor"]["pass"]
    by = {x["name"]: x for x in rec["result"]["directions"]}
    assert set(by) == {"bump", "near-ground-moisture", "twin"}               # the bump always runs
    assert by["twin"]["cosine"]["near-ground-moisture"] == pytest.approx(1.0)
    assert not by["twin"]["independent"] and by["twin"]["repeats"] == "near-ground-moisture"
    assert by["bump"]["status"] == "unresolved" and by["near-ground-moisture"]["status"] == by["twin"]["status"] == "inside"
    over = rec["result"]["directions_verdict"]
    assert code == 1 and over["verdict"] == "unresolved" and over["resolved"] == ["near-ground-moisture"]
    assert "NOT INDEPENDENT of near-ground-moisture" in capsys.readouterr().out
    # every direction rough: nothing resolved, never a pass
    everywhere = {f"window{x}": 1.0 for x in ("", "-descent", "-near-ground-moisture", "-smooth-draw")}
    code, rec = _run(tmp_path, monkeypatch, _Forecast(rough=everywhere), directions="all", table=_LOAD())
    assert code == 1 and rec["result"]["directions_verdict"]["resolved"] == []
    # a first window that dumps no gradient: the descent direction is not run, that is said, and nothing passes
    silent = _Forecast()
    silent.dumps = False
    code, rec = _run(tmp_path, monkeypatch, silent, directions="descent,near-ground-moisture", table=_LOAD())
    by = {x["name"]: x for x in rec["result"]["directions"]}
    assert by["descent"]["status"] == "not run" and "dumped no gradient" in by["descent"]["why"]
    assert by["bump"]["status"] == by["near-ground-moisture"]["status"] == "inside"
    assert code == 1 and rec["result"]["directions_verdict"]["verdict"] == "fail"
    assert "NOT RUN" in capsys.readouterr().out
    # an unknown direction is refused by name before anything runs
    with pytest.raises(d.AdDoorError, match="--directions north: north is not a test direction.*Nothing was downloaded"):
        d.direction_names(_LOAD(), None, "north")


def test_the_rows_default_lists_and_the_parser_word():
    table = d.load_table()
    spec = table["doors"]["adjoint-test"]["directions"]
    assert d.direction_names(table, table["rows"][R120], None) == spec["default"] == spec["order"]
    assert len(spec["default"]) == 4 and [spec["rows"][n]["kind"] for n in spec["order"]] == ["bump", "descent",
                                                                                              "draw", "draw"]
    assert spec["rows"]["near-ground-moisture"]["fields"] == ["qv"] and spec["rows"]["near-ground-moisture"]["levels"]
    # the 3 km storm box: the short list (each further direction is one more window of the box)
    assert d.direction_names(table, table["rows"][STORM], None) == ["bump", "descent"]
    assert d.direction_names(table, table["rows"][STORM], "all") == spec["order"]
    assert d.direction_names(table, None, "smooth-draw") == ["bump", "smooth-draw"]
    assert table["doors"]["adjoint-test"]["direction_window_args"][-2:] == ["--zero", "0"]
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["adjoint-test", "--out", "x"]).directions is None
    assert parser.parse_args(["adjoint-test", "--out", "x", "--directions", "all"]).directions == "all"


def _window(predicted, changes):
    return {"tangent_dir": {"x_MTw": predicted, "w_Mx": predicted, "rel": 0.0},
            "taylor": {"rows": [{"h": h, "dJ": dj, "fd_quotient": dj / h, "ratio": dj / (h * predicted),
                                 "err": abs(dj / (h * predicted) - 1.0)} for h, dj in changes]}}


def test_a_direction_is_read_only_where_the_real_change_is_steady():
    # the 120 km example's own bump, as the card run's window wrote it (quotients per h, and <x, M^T w>)
    card = [(1.0, 0.042531909071388485), (0.3, 0.04290051439940323), (0.1, 0.042986734505885416),
            (0.03, 0.04309224212947216), (0.01, 0.04273082016960639)]
    bump = d.direction_reading(_window(0.043148919767449403, [(h, q * h) for h, q in card]), 0.01, 0.1, 3)
    assert bump["status"] == "inside" and all(r["counts"] for r in bump["rows"])
    assert bump["value"] == pytest.approx(0.0013135355, rel=1e-6)                # the Taylor value of that run
    # a direction that moves the measure by less than the forecast's roughness: the quotient jumps (11, 0.6, 2.2)
    noise = d.direction_reading(_window(20.26, [(0.3, 3.3), (0.1, 0.0628), (0.03, 0.0663)]), 0.01, 0.1, 3)
    assert noise["status"] == "unresolved" and not any(r["counts"] for r in noise["rows"])
    # a steady real change the gradient misses by a quarter: outside, however rough the smallest step is
    off = d.direction_reading(_window(4.0, [(1.0, 4.96), (0.3, 1.482), (0.1, 0.4928), (0.03, 0.31)]), 0.01, 0.1, 3)
    assert off["status"] == "outside" and [r["counts"] for r in off["rows"]] == [True, True, True, False]
    # steady, and within its own spread of the bar on either side: neither a pass nor a fail
    near = d.direction_reading(_window(1.0, [(1.0, 1.05), (0.3, 0.303), (0.1, 0.0985)]), 0.01, 0.1, 3)
    assert near["status"] == "unresolved" and all(r["counts"] for r in near["rows"])
    # two step sizes cannot show a steady run of three; no change at all is not a reading either
    assert d.direction_reading(_window(4.0, [(0.3, 1.2), (0.1, 0.4)]), 0.01, 0.1, 3)["status"] == "unresolved"
    assert d.direction_reading(_window(4.0, [(1.0, 0.0), (0.3, 0.0), (0.1, 0.0)]), 0.01, 0.1, 3)["status"] == "unresolved"
    # the verdict: a failed check, a direction outside or one that did not run fails; fewer than two resolved is no pass
    ok = lambda name, status, independent=True: {"name": name, "status": status, "independent": independent}
    rule = {"min_resolved": 2}
    assert d.directions_verdict([ok("a", "inside"), ok("b", "inside"), ok("c", "unresolved")], True, rule)["verdict"] == "pass"
    assert d.directions_verdict([ok("a", "inside"), ok("b", "unresolved")], True, rule)["verdict"] == "unresolved"
    assert d.directions_verdict([ok("a", "inside"), ok("b", "inside", False)], True, rule)["verdict"] == "unresolved"
    assert d.directions_verdict([ok("a", "inside"), ok("b", "inside"), ok("c", "outside")], True, rule)["verdict"] == "fail"
    assert d.directions_verdict([ok("a", "inside"), ok("b", "inside"), ok("c", "not run")], True, rule)["verdict"] == "fail"
    assert d.directions_verdict([ok("a", "inside"), ok("b", "inside")], False, rule)["verdict"] == "fail"


def test_the_moisture_direction_is_the_transpose_pair_of_the_gradient_and_stays_near_the_ground(tmp_path):
    """<direction, stored gradient> = <dtheta, dJ/dtheta> + <dqv, dJ/dqv>: the write-back the directions go through
    is the one the sensitivity's gradient comes back through."""
    table = d.load_table()
    rng = np.random.default_rng(5)
    grads = {n: rng.standard_normal((NLEV, NCOL)) for n in d.DESCENT_FIELDS}
    g_th, g_qv = d.theta_qv_gradient(table, _Map(), _State(), grads)
    dth, dqv = rng.standard_normal((NLEV, NCOL)), 1e-3 * rng.standard_normal((NLEV, NCOL))
    made = d.tq_direction(table, _Map(), _State(), dth, dqv, "a direction")
    stored = sum(v * grads[NAMES[a - 100]].ravel()[w] for a, w, v in made["terms"] if a < 200)
    assert stored == pytest.approx(float((made["dth"] * g_th).sum() + (made["dqv"] * g_qv).sum()), rel=1e-9)
    # every copy gets the same words: the padded copy two words in, one column wider
    first = {(a, w): v for a, w, v in made["terms"] if a < 200}
    for (a, w), v in first.items():
        lev, col = divmod(w, NCOL)
        assert made["terms"].count([a + 100, 2 + lev * (NCOL + 1) + col, v]) == 1
        break
    ctx = SimpleNamespace(table=table, fmap=_Map(), geometry=_box(),
                          args=SimpleNamespace(at="40.6,-91.3", radius_km=300.0))
    low = d.make_direction(ctx, _State(), "near-ground-moisture",
                           table["doors"]["adjoint-test"]["directions"]["rows"]["near-ground-moisture"], None)
    assert not np.any(low["dth"]) and np.any(low["dqv"][:7]) and not np.any(low["dqv"][7:])
    assert float(np.max(np.abs(low["dqv"]) / _State().field({"name": "qv"}))) == pytest.approx(0.1)   # a tenth of the vapour
    again = d.make_direction(ctx, _State(), "near-ground-moisture",
                             table["doors"]["adjoint-test"]["directions"]["rows"]["near-ground-moisture"], None)
    assert np.array_equal(low["dqv"], again["dqv"])                           # one seed, one direction
    # cut to its largest words, a direction is still a direction of at most that many words a field
    cut = d.tq_direction(table, _Map(), _State(), dth, dqv, "cut", max_words=50)
    assert cut["cut_to_largest_words"] == 50 and cut["words_per_field"] == 50


def test_a_descent_direction_can_follow_part_of_the_gradient():
    """A descent row of the table may name fields and levels: the direction is minus the gradient there and moves
    nothing elsewhere; a row that names neither follows all of it, word for word as before."""
    rng = np.random.default_rng(11)
    g_th, g_qv = rng.standard_normal((NLEV, NCOL)), rng.standard_normal((NLEV, NCOL))
    th, qv, where = d.descent_part(g_th, g_qv, {"kind": "descent"})
    assert np.array_equal(th, g_th) and np.array_equal(qv, g_qv) and where == "theta and qv"
    th, qv, where = d.descent_part(g_th, g_qv, {"kind": "descent", "fields": ["theta"], "levels": "3-5"})
    assert not np.any(qv) and np.array_equal(th[2:5], g_th[2:5]) and not np.any(th[:2]) and not np.any(th[5:])
    assert where == "theta on model levels 3-5"
    th, qv, where = d.descent_part(g_th, g_qv, {"kind": "descent", "fields": ["qv"]})
    assert not np.any(th) and np.array_equal(qv, g_qv) and where == f"qv on model levels 1-{NLEV}"
    with pytest.raises(d.AdDoorError, match="theta or qv"):
        d.descent_part(g_th, g_qv, {"kind": "descent", "fields": ["wind"]})
    with pytest.raises(d.AdDoorError, match="levels run"):
        d.descent_part(g_th, g_qv, {"kind": "descent", "levels": f"1-{NLEV + 1}"})


# ---------------------------------------------------------------- item 6: the printed fast-against-exact figures
#: the card run's own compare of the default route's answer files with the exact route's (max|A-B| over max|B|, the
#: larger of the two targets), read in the card run's fields-default-vs-exact.txt
CARD_RUN = {"temperature": 2.932e-06, "moisture": 1.315e-06, "wind": 6.467e-06, "the stored pressure": 9.778e-05}


def test_every_printed_fast_against_exact_figure_covers_its_measurement():
    route = d.load_table()["rows"][STORM]["gradient_line"]["boundary_layer"]["routes"]["tape-free"]
    # dist48: the default route runs another form; the card run's figures are kept as the form before's and are not
    # printed for this one. The printing rule is still the one this test pins, on those figures.
    assert "error_measured" not in route
    measured = route["error_measured_before"]
    route = dict(route, error_measured=measured)
    figures = d.error_figures(measured)
    named = {f["name"]: f for f in figures if not f.get("all_fields")}
    assert {n: f["measured"] for n, f in named.items()} == CARD_RUN        # the table holds the card run's figures
    plan = f"the row's example plan ({d.load_table()['rows'][STORM]['example_plan']})"
    words, on_this_plan = d.route_error_words(SimpleNamespace(plan_how=plan, steps=59), route)
    assert on_this_plan and "one card run" in words and "on this example over this 59-step window" in words
    for f in figures:                                    # every figure the command prints is at or above its measurement
        assert f["printed"] >= f["measured"] > 0.0
        assert f"{f['name']} {'within ' if f.get('all_fields') else ''}{f['printed']:.1e}" in words
        assert f["printed"] <= 1.1 * f["measured"]                         # rounded up, not blown up
    every = next(f for f in figures if f.get("all_fields"))
    assert every["measured"] == 9.778e-05 and every["name"] == "every one of the 16 fields"
    assert len(measured["figures"]) + len(measured["both_zero"]) == 16
    assert "moisture 1.4e-06" in words and "the stored pressure 9.8e-05" in words
    assert "1.1e-06" not in json.dumps(route) and "9.6e-05" not in json.dumps(route)   # the understated figures
    assert d.round_up(3.0e-06) == 3.0e-06 and d.round_up(1.315e-06) == 1.4e-06 and d.round_up(9.96e-05) == 1.0e-04


# --------------------------------------------------------------------- item 10b: the line is said once a command
def test_the_forward_not_kept_line_is_printed_once_in_a_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 200_000)
    table = d.load_table()
    ns = argparse.Namespace(checked_forward=False, boundary_layer=None, cloud_derivative=None, keep_forward=None)
    for route in table["rows"][STORM]["gradient_line"]["boundary_layer"]["routes"].values():
        for var, item in (route.get("images") or {}).items():       # stand-in images, the table's pins moved to them
            path = tmp_path / "assets" / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(var.encode())
            item["sha256"] = hashlib.sha256(var.encode()).hexdigest()
    out = tmp_path / "out"
    out.mkdir()
    why = "this host has 200,000 MiB available: enough for the gradient and not for the gradient with its forward kept"
    ctx = SimpleNamespace(table=table, row=table["rows"][STORM], row_name=STORM, door="sensitivity",
                          root=tmp_path / "assets", card="0", cards=["0"], arch="sm_120", split=None, out=out,
                          args=ns, steps=59, keep_forward={"on": False, "offered": True, "why": why})
    d.say_forward_not_kept(ctx, why)                          # where the host memory is judged (the context)
    line = d.gradient_line(ctx, ns)
    d.gradient_line_words(ctx, line, ns, out)                 # where the gradient line's words are made
    d.gradient_line_words(ctx, line, ns, out)                 # and again for a later gradient of a Python session
    said = capsys.readouterr().out
    assert said.count("forward not kept between gradients") == 1
    assert f"woof hex sensitivity: forward not kept between gradients: {why}" in said
    fresh = SimpleNamespace(**{k: v for k, v in vars(ctx).items() if k != "forward_not_kept_said"})
    d.gradient_line_words(fresh, line, ns, out)               # another command says it once itself
    assert capsys.readouterr().out.count("forward not kept between gradients") == 1
    import inspect
    assert "say_forward_not_kept(self, why)" in inspect.getsource(d.Context.__init__)


# ----------------------------------------- item 10c: a program that is ending as the command starts is waited for
ENDING = textwrap.dedent("""
    import ctypes, os, sys, threading, time
    def later():
        time.sleep(float(sys.argv[1]))        # the card and the page-locked host memory are still being given back
        os._exit(0)
    threading.Thread(target=later).start()
    print("up", flush=True)
    ctypes.CDLL(None).syscall(60, 0)          # the first thread ends alone: /proc reads the process as ended
""")


@pytest.mark.skipif(not sys.platform.startswith("linux") or os.uname().machine != "x86_64", reason="/proc and syscall 60")
@pytest.mark.parametrize("door", ["tangent", "sensitivity"])
def test_a_command_waits_for_a_program_another_command_told_to_end(tmp_path, monkeypatch, capsys, door):
    """A stand-in program holds the card 3 s after another command told it to end. Before: the new command found no
    session, read the card at once and was refused. Now it waits for the program's last thread, counts it as
    stopped by this command, and its card reading waits for the memory that comes back."""
    table = d.load_table()
    sessions = tmp_path / "sessions"
    folder = s.private_dir(sessions) / "card0"
    folder.mkdir(mode=0o700)
    monkeypatch.setenv(s.SESSIONS_ENV, str(sessions))
    for var in s.SCHEDULER_ENV + (s.MODE_ENV,):
        monkeypatch.delenv(var, raising=False)
    script = tmp_path / "ending.py"
    script.write_text(ENDING)
    proc = subprocess.Popen([sys.executable, str(script), "3", "--resident", str(folder)], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline().strip() == b"up"
        ticks = s._start_ticks(proc.pid)
        (folder / "session.json").write_text(json.dumps({"schema": s.SCHEMA, "uid": os.geteuid(), "pid": proc.pid,
                                                         "start_ticks": ticks, "command": {}}))
        (folder / "stop").write_text("")                              # another command told it to end
        t0 = time.monotonic()
        while s._start_ticks(proc.pid) is not None and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        assert s._start_ticks(proc.pid) is None and s._threads_left(proc.pid)     # ending, and not gone
        assert s.read_live(sessions, "card0") is None                  # it serves nothing: no session is found ...
        assert s.read_ending(sessions, "card0")["pid"] == proc.pid     # ... and the program is still there
        # the card as that program leaves it: 13,210 MiB free while a thread of it runs, the whole card after
        holding = lambda: s._threads_left(proc.pid)
        monkeypatch.setattr(d, "refuse_no_local_gpu", lambda: None)
        monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
        monkeypatch.setattr(d, "total_mib", lambda card: 97_887)
        monkeypatch.setattr(d, "free_mib", lambda card: 13_210 if holding() else 97_252)
        monkeypatch.setattr(d, "process_card_mib", lambda pid: {})
        args = argparse.Namespace(session=None, session_idle_s=30.0, session_name=None, wait_card=0.0,
                                  checked_forward=False, boundary_layer=None)
        ctx = SimpleNamespace(args=args, door=door, cards=["0"], card="0", row_name=STORM, row=table["rows"][STORM],
                              table=table, steps=3, split=None, need_mib=lambda: 96_000)
        # BEFORE (what a command that does not know of the ending program does): refused at once
        with pytest.raises(d.AdDoorError, match="card 0 has 13210 MiB free; this row's window needs 96000 MiB"):
            d.Context.check_cards(SimpleNamespace(**{**vars(ctx), "session": SimpleNamespace(stopped=[])}))
        assert time.monotonic() - t0 < 2.5 and holding()
        ctx.session = s.open_for(ctx)
        waited = time.monotonic() - t0
        assert not holding() and waited >= 2.0                         # the command waited for its last thread
        stopped = ctx.session.stopped
        assert len(stopped) == 1 and stopped[0]["pid"] == proc.pid and stopped[0]["ended"]
        assert not stopped[0]["ended_by_pid"] and "was ending when this command started" in stopped[0]["why"]
        assert d.Context.check_cards(ctx) == "12.0"                    # admitted: the card has come back
        said = capsys.readouterr().out
        assert f"session card0 (pid {proc.pid}): its program was told to end before this command started" in said
        if door == "sensitivity":
            assert "STARTED FRESH: the program that was loaded for this card" in said and ctx.session.mode == "use"
            assert ctx.session.block()["stopped"] == stopped
        ctx.session.unlock()
    finally:
        proc.wait(timeout=20)
    assert s.read_ending(sessions, "card0") is None                    # gone: nothing to wait for
    ctx.session = s.open_for(ctx)
    assert ctx.session.stopped == []
    ctx.session.unlock()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc")
def test_a_program_started_a_moment_ago_is_found_serving_its_folder(tmp_path):
    """THE BREAKAGE: straight after Popen (posix_spawn) the child's /proc command line reads empty for a moment, so
    read_live found no program in a session folder whose program had just started (the flaky test below, 1 in 40)."""
    folder = s.private_dir(tmp_path / "sessions") / "card0"
    folder.mkdir(mode=0o700)
    for _ in range(20):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "--resident", str(folder)])
        try:
            assert s._serves(proc.pid, folder)
            assert not s._serves(proc.pid, tmp_path)
        finally:
            proc.kill()
            proc.wait(timeout=10)
    assert not s._serves(proc.pid, folder)                              # ended and reaped: serves nothing


def test_a_program_that_is_whole_is_not_taken_for_an_ending_one(tmp_path):
    folder = s.private_dir(tmp_path / "sessions") / "card0"
    folder.mkdir(mode=0o700)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "--resident", str(folder)])
    try:
        (folder / "session.json").write_text(json.dumps({"schema": s.SCHEMA, "uid": os.geteuid(), "pid": proc.pid,
                                                         "start_ticks": s._start_ticks(proc.pid)}))
        assert s.read_ending(tmp_path / "sessions", "card0") is None
        assert s.read_live(tmp_path / "sessions", "card0")["pid"] == proc.pid
        (folder / "session.json").write_text(json.dumps({"schema": s.SCHEMA, "uid": os.geteuid() + 1, "pid": proc.pid}))
        assert s.read_ending(tmp_path / "sessions", "card0") is None         # another user's: never waited for
    finally:
        proc.kill()
        proc.wait(timeout=10)


# ------------------------------------------------- the storm row's texts follow the newest card run of this command
def test_the_exact_routes_cost_and_the_kept_forward_are_the_newest_card_runs():
    """THE BREAKAGE: the table printed the exact route as 35 s and 70 s slower (282.3 against 247.3 s, 254.8 against
    184.4 s), a build earlier than the newest card run of this command, which read 289.6 against 239.5 s and 260.8
    against 180.5 s; it said the kept forward was "not yet measured through this command" and that the repaired
    cloud images "have not yet run a whole window on a card" after that run had measured both."""
    gl = _LOAD()["rows"][STORM]["gradient_line"]
    routes = gl["boundary_layer"]["routes"]
    exact, default = (289.6, 260.8), (239.5, 180.5)        # first gradient, further target: that run's windows
    for text in (routes["taped"]["what"], routes["tape-free"]["saves"]):
        for seconds in exact + default:
            assert f"{seconds} s" in text, seconds
        for more, e, f in zip((50, 80), exact, default):   # the stated cost is the difference, to the nearest 10 s
            assert f"{more} s" in text and abs((e - f) - more) < 5.0
        assert "35 s" not in text and "70 s" not in text and "282.3" not in text and "254.8" not in text
    kept = routes["tape-free"]["keep_forward"]
    assert "not yet measured through this command" not in kept["says"] and "180.5 s" in kept["says"]
    assert "not yet at 59 steps" not in kept["prefill_what"]
    assert "this command has not measured it" not in routes["tape-free"]["speed"]
    cloud = gl["cloud_derivative"]
    assert "have not yet run a whole window on a card" not in cloud["routes"]["gated"]["default_when_before"]
    for text in (cloud["routes"]["gated"]["default_when_before"], cloud["routes"]["gated"]["says"],
                 cloud["routes"]["gated"]["measured"]["at_59_steps"]):
        assert "2.78e-08" in text and "1.75e-08" in text
    assert gl["measured"].count("The build before this one") == 1 and "289.6 s" in gl["measured"]


# ---------------------------------------- the reading rule on the real runner's windows (the 12 km storm cut)
def test_a_direction_off_at_its_steadiest_step_sizes_is_outside_whatever_a_noisy_step_reads():
    """THE BREAKAGE: every step size that counted had to read outside the bar, so one step size at the measure's own
    noise floor (steady only within a wide spread) made a direction that read 12 per cent off UNRESOLVED, and an
    unresolved direction fails no verdict. The windows below are the real runner's, as this command's handler read
    them on a 12 km cut of the storm box (15 steps of 60 s, one home card): <x, M^T w> and dJ per step size."""
    hs = (1.0, 0.3, 0.1, 0.03, 0.01)
    real = {
        "bump": (0.03772319360761677, (0.037800430341461355, 0.01132514043398225, 0.0037763264081575043,
                                       0.0011317931729120656, 0.0003760402938624563)),
        "descent": (-0.3196436243691616, (-0.30507371252154847, -0.0916080335293924, -0.030549837577014904,
                                          -0.009165803253055715, -0.0030654236853899786)),
        "near-ground-moisture": (-0.0027166716993385085, (-0.002242796139196146, -0.000692847747341796,
                                                          -0.00023936533218565925, -7.187530829727562e-05,
                                                          -2.624973921572238e-05)),
        "smooth-draw": (-0.008021533799899987, (-0.008296844771847611, -0.002272049513164509,
                                                -0.0007326657132602499, -0.00021633540106608118,
                                                -7.666001535283629e-05)),
    }
    got = {name: d.direction_reading(_window(predicted, list(zip(hs, changes))), 0.01, 0.1, 3)
           for name, (predicted, changes) in real.items()}
    assert got["bump"]["status"] == "inside" and got["bump"]["value"] == pytest.approx(8.603e-05, rel=1e-3)
    assert got["descent"]["status"] == "outside" and got["descent"]["value"] == pytest.approx(0.04416, rel=1e-3)
    moist = got["near-ground-moisture"]
    assert [r["steadiest"] for r in moist["rows"]] == [False, True, True, True, False]
    assert moist["rows"][-1]["counts"] and moist["rows"][-1]["ratio"] == pytest.approx(0.9662, rel=1e-3)   # the noisy one
    assert moist["status"] == "outside" and moist["value"] == pytest.approx(0.1181, rel=1e-3)   # was: unresolved
    # a draw whose steadiest step sizes still drift towards the bar within their own spread stays unresolved
    draw = got["smooth-draw"]
    assert draw["status"] == "unresolved" and not draw["rows"][0]["counts"] and "steadiest" in draw["why"]
    rule = {"min_resolved": 2}
    listed = [{"name": n, "status": r["status"], "independent": True} for n, r in got.items()]
    over = d.directions_verdict(listed, True, rule)
    assert over["verdict"] == "fail" and over["outside"] == ["descent", "near-ground-moisture"]
    assert over["unresolved"] == ["smooth-draw"]

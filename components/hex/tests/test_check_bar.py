"""The window check's bar in single-precision roundings of the dot (woof.hex.dot_bar), and every door judging by it.

Every reading below is a recorded one (card runs of 15 Nov 2026; rel and sign as read in each run's own
window.json). The bar must pass what is rounding and fail what is a real term."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from woof.hex import ad_doors as d
from woof.hex import dot_bar as b

TABLE = d.load_table()
BAR = b.bar_of(TABLE, "roundings")
STRICT = b.bar_of(TABLE, "strict")


def side(rel: float, sign: int, dot: float = 12.0) -> tuple[float, float]:
    """(y_Mx, x_MTy) of a self-form dot of size ``dot`` whose relative miss is ``rel`` with the given sign."""
    return dot, dot * (1.0 + sign * rel)


def r_of(rel: float, sign: int) -> float:
    return b.roundings(*side(rel, sign))


# the released build, 12 km cut, 14 steps, scored on the model's 12 fields, seeds 1..12
FIELDS_12KM = [(3.2041e-08, -1), (3.2847e-08, -1), (1.0278e-08, +1), (2.2857e-08, -1), (3.9798e-08, +1),
               (1.1190e-08, -1), (4.9796e-09, +1), (8.0621e-09, +1), (7.7634e-10, +1), (1.4224e-08, +1),
               (6.2571e-10, +1), (3.4309e-08, -1)]
# 3 km storm box on NOAA's settings, 58 steps, model fields, seeds 1..5
NOAA_3KM = {1: (1.4060e-07, -1), 2: (4.5157e-08, +1), 3: (7.2397e-09, -1), 4: (2.9965e-08, +1), 5: (5.4327e-08, -1)}
# the README command, 12 km example, 15 steps, 134 allocations, seed 1, limiter held and live
README_HELD = (134.00000009295212, 133.99998860295497)    # 8.574624732323083e-08
README_LIVE = (134.00000008192896, 134.0000274972271)     # 2.045917352483222e-07, the install that failed
# a cumulus port's 12 km window with wrong rows, limiter live, 14 steps, seeds 1..4
CUMULUS = [(8.406e-05, +1), (2.211e-05, -1), (1.532e-05, -1), (3.416e-05, +1)]
# 30 steps at 12 km, seed 4, eight direction sizes (x1, x1.3, x3, x1.7, x2.3, x2.7, x5, x7)
W30 = [(1.92e-06, +1), (2.42e-07, +1), (2.25e-06, +1),
       (1.3168e-06, -1), (2.1498e-06, +1), (9.1551e-07, +1), (2.6759e-06, +1), (2.4379e-06, +1)]


def test_one_rounding_is_float32_epsilon_times_the_dot_and_signed():
    assert b.EPS32 == 2.0 ** -23 == np.finfo(np.float32).eps
    y, x = README_LIVE
    r = b.roundings(y, x)
    assert r > 0 and abs(r - 2.045917352483222e-07 / 2.0 ** -23) < 1e-6      # +1.72 roundings
    assert b.roundings(*README_HELD) < 0 and abs(abs(b.roundings(*README_HELD)) - 0.7193) < 1e-3
    assert b.roundings(1.0, float("nan")) is None and b.roundings(None, 1.0) is None and b.roundings(3.0, 3.0) == 0.0
    assert BAR["rel"] == pytest.approx(4 * 2.0 ** -23) and BAR["rel"] == pytest.approx(4.768e-07, rel=1e-3)
    # the strict setting is the earlier bar, 0.84 of one rounding
    assert STRICT["rel"] == 1e-07 and STRICT["each_within_roundings"] == pytest.approx(0.8389, rel=1e-3)


def test_the_rounding_readings_pass():
    for rel, s in FIELDS_12KM:
        assert b.judge_one(*side(rel, s), BAR)["pass"]
    many = b.judge_many([r_of(rel, s) for rel, s in FIELDS_12KM], BAR)
    assert many["pass"] and many["within_each"] and many["no_drift"] is True and many["n"] == 12
    assert abs(many["mean"]) < 2 * many["se"]
    for seed, (rel, s) in NOAA_3KM.items():
        one = b.judge_one(*side(rel, s), BAR)
        assert one["pass"], seed
    assert b.judge_one(*side(*NOAA_3KM[1]), BAR)["roundings"] == pytest.approx(-1.179, abs=1e-3)
    assert b.judge_one(*README_HELD, BAR)["pass"] and b.judge_one(*README_LIVE, BAR)["pass"]


def test_strict_keeps_the_earlier_verdicts():
    assert b.judge_one(*README_HELD, STRICT)["pass"] and not b.judge_one(*README_LIVE, STRICT)["pass"]
    assert not b.judge_one(*side(*NOAA_3KM[1]), STRICT)["pass"]


def test_the_real_terms_fail():
    for rel, s in CUMULUS:          # 128 to 705 roundings: part (a)
        one = b.judge_one(*side(rel, s), BAR)
        assert not one["pass"] and abs(one["roundings"]) > 100
    w = b.judge_many([r_of(rel, s) for rel, s in W30], BAR)
    assert not w["pass"] and not w["within_each"]
    assert w["positive"] == 7 and w["no_drift"] is False                     # part (b) alone fails it too
    assert w["mean"] == pytest.approx(11.8, abs=0.1) and w["se"] == pytest.approx(4.08, abs=0.05)


def test_drift_alone_fails_when_every_direction_is_inside():
    # eight directions each inside 4 roundings, all one sign: a small real term, not rounding
    w = b.judge_many([1.5, 2.0, 0.8, 2.6, 1.1, 3.0, 1.9, 2.2], BAR)
    assert w["within_each"] and w["no_drift"] is False and not w["pass"]
    # fewer than eight: the drift part is not judged, and says so
    few = b.judge_many([1.5, 2.0, 0.8], BAR)
    assert few["pass"] and few["no_drift"] is None and "needs 8" in few["no_drift_why"]


def _stencil(n: int, seed: int) -> np.ndarray:
    """A mesh-like operator: each row a centre and six neighbours, float32 (dense here for clarity)."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a[i, i] = 1.0 + 0.1 * rng.standard_normal()
        for k in (1, 2, 3):
            a[i, (i + k) % n] = 0.05 * rng.standard_normal()
            a[i, (i - k) % n] = 0.05 * rng.standard_normal()
    return a


def _window(steps: list[np.ndarray], x: np.ndarray, bad: tuple | None = None) -> dict:
    """The self dot of a float32 window: <y, M x> against <M^T y, x>, y = M x at unit weight, sums in binary64.
    ``bad`` = (step, row, column, added value) puts one wrong entry in that step's transpose (one wrong row of the
    adjoint). Also returns each step's forward input z_s and the adjoint w_s that reaches it, so the miss a wrong
    entry adds can be predicted: <M^T y, x> = <A_s^T w_s, z_s>, so entry (r, c) off by e adds e * w_s[c] * z_s[r]."""
    v, zs = x.astype(np.float32), []
    for a in steps:
        zs.append(v)
        v = (a @ v).astype(np.float32)
    y = (v / np.float32(np.sqrt(np.sum(v.astype(np.float64) ** 2)))).astype(np.float32)
    y_mx = float(np.dot(y.astype(np.float64), v.astype(np.float64)))
    w, ws = y, {}
    for s, a in reversed(list(enumerate(steps))):
        ws[s] = w
        at = a.T.copy()
        if bad is not None and bad[0] == s:
            at[bad[1], bad[2]] = np.float32(at[bad[1], bad[2]] + bad[3])
        w = (at @ w).astype(np.float32)
    return {"y_Mx": y_mx, "x_MTy": float(np.dot(w.astype(np.float64), x.astype(np.float64))), "z": zs, "w": ws}


def test_a_float32_window_reads_within_the_bar_and_one_wrong_row_fails_it():
    n, steps = 600, [_stencil(600, s) for s in range(10)]
    rs = []
    for seed in range(12):
        x = np.random.default_rng(100 + seed).standard_normal(n).astype(np.float32)
        w = _window(steps, x)
        rs.append(b.roundings(w["y_Mx"], w["x_MTy"]))
    good = b.judge_many(rs, BAR)
    assert good["pass"], good                     # a correct transpose in single precision: inside, no drift
    assert max(abs(r) for r in rs) < 4
    # one wrong entry of known size in one row of one step's transpose, sized to add `target` roundings
    x = np.random.default_rng(100).standard_normal(n).astype(np.float32)
    base = _window(steps, x)
    s, r = 4, 17
    c = int(np.argmax(np.abs(base["w"][s])))
    unit = b.EPS32 * abs(base["y_Mx"])
    for target, fails in ((12.0, True), (-30.0, True), (0.5, False)):
        e = target * unit / (float(base["w"][s][c]) * float(base["z"][s][r]))
        got = _window(steps, x, (s, r, c, e))
        one = b.judge_one(got["y_Mx"], got["x_MTy"], BAR)
        assert abs(one["roundings"] - target) < 2.0, (target, one)      # the injected size, plus rounding
        assert one["pass"] is (not fails), (target, one)


def test_the_door_check_reads_the_verdict_in_roundings():
    win = {"self_dot": {"units": "pool", "allocations": 134},
           "dots": [{"seed": 1, "y_Mx": README_LIVE[0], "x_MTy": README_LIVE[1], "rel_aligned": 2.045917352483222e-07}]}
    c = d.self_dot_check(win, BAR)
    assert c["pass"] and c["bar"] == pytest.approx(4 * 2.0 ** -23) and c["roundings"] == pytest.approx(1.7163, abs=1e-3)
    assert "4 single-precision roundings" in c["bar_words"] and c["directions"] is None
    assert not d.self_dot_check(win, STRICT)["pass"]
    many = {"self_dot": {"units": "fields"},
            "dots": [dict(zip(("y_Mx", "x_MTy"), side(rel, s)), seed=i + 1, rel_aligned=rel)
                     for i, (rel, s) in enumerate(W30)]}
    c = d.self_dot_check(many, BAR)
    assert not c["pass"] and c["directions"]["positive"] == 7 and c["directions"]["no_drift"] is False
    ok = {"self_dot": {"units": "fields"},
          "dots": [dict(zip(("y_Mx", "x_MTy"), side(rel, s)), seed=i + 1, rel_aligned=rel)
                   for i, (rel, s) in enumerate(FIELDS_12KM)]}
    assert d.self_dot_check(ok, BAR)["pass"]


def test_one_definition_the_table_the_runner_and_no_hidden_bar():
    spec = TABLE["thresholds"]["self_dot_bar"]
    assert spec["default"] == "roundings" and spec["each_within_roundings"] == 4
    assert spec["drift"] == {"min_directions": 8, "standard_errors": 2} and spec["strict"]["relative"] == 1e-07
    assert "self_dot_rel" not in TABLE["thresholds"]
    assert b.runner_env(BAR) == {"AD_DOT_BAR": "roundings:4"}
    assert b.runner_env(STRICT) == {"AD_DOT_BAR": "relative:1e-07"}
    src = Path(__file__).resolve().parents[1] / "rust" / "hexad" / "src" / "runner"
    if not (src / "self_dot.rs").is_file():
        pytest.skip("the runner's sources are not in this tree")
    rs = (src / "self_dot.rs").read_text()
    assert f"pub const DEFAULT_BAR_ROUNDINGS: f64 = {spec['each_within_roundings']:.1f};" in rs
    assert "pub const EPS32: f64 = 1.0 / 8_388_608.0;" in rs
    win = (src / "window.rs").read_text()
    assert "PROGRAM_DOT_BAR" not in win and "self_dot::DotBar::from_env()" in win


def test_the_door_options():
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    d.add_ad_parsers(sub)
    a = p.parse_args(["adjoint-test", "--out", "o", "--dot-directions", "8", "--dot-bar", "strict"])
    assert a.dot_directions == 8 and a.dot_bar == "strict"
    assert d.dot_bar_in_force(TABLE, a)["setting"] == "strict"
    assert d.dot_bar_in_force(TABLE, argparse.Namespace())["setting"] == "roundings"
    assert d.self_dot_args_seeded(["--seeds", "1", "--x", "2"], None, 8)[1] == "1,2,3,4,5,6,7,8"
    assert d.self_dot_args_seeded(["--seeds", "1"], 3, 2)[1] == "3,4"
    assert d.self_dot_args_seeded(["--seeds", "1"], 2)[1] == "2"
    for door in ("tangent", "sensitivity", "param-grad"):
        assert p.parse_args([door, "--out", "o", "--dot-bar", "roundings"]).dot_bar == "roundings"

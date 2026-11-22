# SPDX-License-Identifier: Apache-2.0
"""The differentiable doors say which forecast their linear model is of, and refuse the one they have no rows for.

THE BREAKAGES (read in the 0.6.0 line, 2026-11-15):
* ``woof hex forecast`` runs correct mode by default and every plan is recorded in exact mode; no door said so, and
  no door took ``--physics-mode``: asking for the correct-mode gradient ended in argparse's "unrecognized arguments";
* the recorder appended the words after ``--`` to a forecast command that already said ``--physics-mode exact``, so
  ``record ... -- --physics-mode correct`` recorded the correct-mode forecast, wrote ``"physics_mode": "exact"`` in
  its receipt, and the window ran the exact-mode derivative rows on it. The tangent-against-adjoint check passes on
  such a pair whatever the forward was: both sides are the same rows.

No card, no assets, no network.
"""
import argparse
import json

import pytest

from woof.hex import ad_doors as d
from woof.hex import ad_session as s
from woof.hex.ad import assets, rows
from woof.hex.ad.record import chain, cli

CUT = "hrrrv5-q12.96.2890.n40.60w91.30"
README_12 = ["adjoint-test", "--row", CUT, "--steps", "15", "--at", "40.6,-91.3", "--radius-km", "150"]


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def nothing_fetched(monkeypatch, tmp_path):
    """Any card query, runner lookup or download fails the test: the refusal must come before all of them."""
    for name in d.NO_LOCAL_GPU_ENV + s.SCHEDULER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(s.SESSIONS_ENV, str(tmp_path / "sessions"))
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))

    def touched(*a, **k):
        raise AssertionError("the door went past the physics-mode refusal (card, runner or download)")

    for module, name in ((assets, "asset_root"), (assets, "_fetch"), (d, "resolve_runner"), (d, "refuse_no_local_gpu"),
                         (d, "compute_capability"), (d, "total_mib"), (d, "free_mib")):
        monkeypatch.setattr(module, name, touched)


# ---- the table -----------------------------------------------------------------------------------------------------
def test_the_table_names_the_one_mode_the_rows_are_of():
    block = d.load_table()["physics_mode"]
    assert block["linearised"] == ["exact"] and block["default"] == "exact"
    assert "exact-mode forecast" in block["line"] and "correct mode" in block["line"]
    # every mode the forecast door takes is either linearised or has its sentence
    from woof.hex.correct_mode import PHYSICS_MODES
    assert set(PHYSICS_MODES) == set(block["linearised"]) | set(block["refused"])
    assert chain.RECORD_PHYSICS_MODE in block["linearised"]


# ---- the doors -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("door", ["tangent", "adjoint-test", "sensitivity", "param-grad"])
def test_every_door_takes_the_word_and_defaults_to_the_recorded_mode(door, tmp_path):
    table = d.load_table()
    plain = _parse([door, "--row", CUT, "--out", str(tmp_path)])
    assert plain.physics_mode is None and d.refuse_physics_mode(plain, door, table) == "exact"
    said = _parse([door, "--row", CUT, "--out", str(tmp_path), "--physics-mode", "exact"])
    assert d.refuse_physics_mode(said, door, table) == "exact"


@pytest.mark.parametrize("door", ["tangent", "adjoint-test", "sensitivity", "param-grad"])
def test_correct_mode_is_refused_with_a_plain_sentence(door, tmp_path):
    args = _parse([door, "--row", CUT, "--out", str(tmp_path), "--physics-mode", "correct"])
    with pytest.raises(d.AdDoorError, match=rf"{door}: --physics-mode correct: this build has no tangent or adjoint "
                                            r"of the correct-mode forecast\. Every plan is recorded in exact mode.*"
                                            r"would not be the gradient of that forecast.*Nothing was run and "
                                            r"nothing was downloaded\. Leave --physics-mode out"):
        d.refuse_physics_mode(args, door, d.load_table())


def test_the_readme_command_in_correct_mode_stops_before_any_card_or_download(nothing_fetched, tmp_path):
    args = _parse(README_12 + ["--out", str(tmp_path / "o"), "--physics-mode", "correct"])
    with pytest.raises(d.AdDoorError, match="--physics-mode correct: this build has no tangent or adjoint"):
        d.Context(args, "adjoint-test")
    assert not (tmp_path / "o").exists()


def test_a_misspelt_mode_is_argparse_s_refusal(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        _parse(["sensitivity", "--row", CUT, "--out", str(tmp_path), "--physics-mode", "corect"])
    assert stop.value.code == 2 and "invalid choice: 'corect'" in capsys.readouterr().err


def test_a_mode_with_rows_would_run(tmp_path):
    # when a build ships rows of the corrected schemes the table gains the mode and the refusal leaves: data, not code
    table = d.load_table()
    table["physics_mode"] = dict(table["physics_mode"], linearised=["exact", "correct"], refused={})
    args = _parse(["tangent", "--row", CUT, "--out", str(tmp_path), "--physics-mode", "correct"])
    assert d.refuse_physics_mode(args, "tangent", table) == "correct"


# ---- a plan recorded in another mode -------------------------------------------------------------------------------
def _package(tmp_path, name, receipt):
    plan = tmp_path / name
    plan.mkdir()
    if receipt is not None:
        (plan / "record-receipt.json").write_text(json.dumps(receipt))
    return plan


FORECAST = ["forecast", "--mesh", "m", "--physics-suite", "hrrrv5", "--numerics", "strict", "--physics-mode", "exact"]


def test_the_mode_of_a_plan_is_the_one_its_forecast_words_end_on(tmp_path):
    table = d.load_table()
    plain = _package(tmp_path, "plain", {"physics_mode": "exact", "forecast_args": FORECAST})
    assert d.plan_physics_mode(plain) == "exact"
    assert d.refuse_plan_physics_mode(plain, "exact", table) == "exact"
    old = _package(tmp_path, "old", {"physics_mode": "exact"})   # a receipt with no forecast words
    assert d.plan_physics_mode(old) == "exact"
    other = _package(tmp_path, "other", None)   # a package of another recorder: it says nothing, nothing is refused
    assert d.plan_physics_mode(other) is None and d.refuse_plan_physics_mode(other, "exact", table) is None


@pytest.mark.parametrize("extra", [["--physics-mode", "correct"], ["--physics-mode=correct"],
                                   ["--dt", "60", "--physics-mode", "correct", "--no-products"]])
def test_a_plan_recorded_in_correct_mode_is_refused_whatever_its_receipt_label_says(tmp_path, extra):
    # the receipt of the recorder before this change said "exact" for such a recording
    plan = _package(tmp_path, "p", {"physics_mode": "exact", "forecast_args": FORECAST + extra})
    assert d.plan_physics_mode(plan) == "correct"
    with pytest.raises(d.AdDoorError, match=r"was recorded in physics mode 'correct' .*this command's tangent and "
                                            r"adjoint are of exact mode.*not the derivative of that recording"):
        d.refuse_plan_physics_mode(plan, "exact", d.load_table())


def test_the_last_mode_word_wins_as_in_the_forecast_parser():
    assert d.forwarded_physics_mode(None) is None and d.forwarded_physics_mode(["--dt", "60"]) is None
    assert d.forwarded_physics_mode(["--physics-mode", "correct", "--physics-mode", "exact"]) == "exact"
    assert d.forwarded_physics_mode(["--physics-mode=exact", "--physics-mode", "correct"]) == "correct"
    assert d.forwarded_physics_mode(["--physics-mode"]) == ""
    for words in (None, ["--physics-mode", "correct"], ["--physics-mode=exact"], ["--physics-mode"], ["--x", "1"]):
        assert chain.forwarded_physics_mode(words) == d.forwarded_physics_mode(words)   # one rule in both places
    # and it is the forecast door's own rule: argparse keeps the last value of a repeated option
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-mode", choices=("exact", "correct"))
    assert parser.parse_args(["--physics-mode", "exact", "--physics-mode", "correct"]).physics_mode == "correct"


# ---- the recorder --------------------------------------------------------------------------------------------------
def test_the_recorder_says_exact_itself():
    class O:
        mesh, suite, init, grid, hours, dt, numerics, oro, lam, forecast_extra = "m", "hrrrv5", "i", "g", 1, 60, \
            "strict", None, None, ["--no-hourly"]
    words = chain.forecast_args(O)
    assert chain.forwarded_physics_mode(words) == chain.RECORD_PHYSICS_MODE == "exact"


@pytest.mark.parametrize("extra", [["--physics-mode", "correct"], ["--physics-mode=correct"], ["--physics-mode"]])
def test_the_recorder_refuses_another_mode_after_the_two_dashes(extra, capsys):
    with pytest.raises(SystemExit) as stop:
        chain.refuse_other_physics_mode(extra)
    err = capsys.readouterr().err
    assert stop.value.code == 3
    assert "DEFECT: woof.hex.ad.record: record: --physics-mode" in err and "Nothing was recorded" in err
    assert "a recording is of the exact-mode forecast" in err


def test_the_recorder_takes_exact_and_other_words():
    for extra in ([], ["--physics-mode", "exact"], ["--physics-mode=exact"], ["--no-products", "--dt", "60"]):
        chain.refuse_other_physics_mode(extra)


def test_record_with_correct_after_the_dashes_stops_before_any_build_or_card(tmp_path, monkeypatch, capsys):
    def touched(*a, **k):
        raise AssertionError("the recorder went past the physics-mode refusal")

    monkeypatch.setattr(chain, "record", touched)
    with pytest.raises(SystemExit) as stop:
        cli.main(["record", "--mesh", "m", "--init", str(tmp_path / "i.nc"), "--grid", str(tmp_path / "g.nc"),
                  "--dt", "60", "--hours", "1", "--out", str(tmp_path / "pkg"), "--", "--physics-mode", "correct"])
    assert stop.value.code == 3 and "Nothing was recorded" in capsys.readouterr().err
    assert not (tmp_path / "pkg").exists()

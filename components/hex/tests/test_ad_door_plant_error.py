# SPDX-License-Identifier: Apache-2.0
"""adjoint-test --plant-error KERNEL:SCALE: an error planted on purpose in one backward code, so a user can watch the
transpose check catch a wrong derivative; it can never be left on by accident.

THE BREAKAGE: a check that only ever prints PASS shows nobody that it can fail. The runner's planting switch (AD_PLANT)
was reachable only through an environment word that no command named, and that the door neither said nor read: left in
a shell, it broke a sensitivity map with nothing on screen saying so.
"""
import argparse
import json
from types import SimpleNamespace

import pytest

from woof.hex import ad_doors as d

Q12 = "hrrrv5-q12.96.2890.n40.60w91.30"
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
GLOBAL = "hrrrv5-x1.40962"


@pytest.fixture(autouse=True)
def _no_plant_words(monkeypatch):
    for k in [k for k in __import__("os").environ if k.startswith("AD_PLANT")]:
        monkeypatch.delenv(k)


def _parser():
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser


def _args(*argv):
    return _parser().parse_args([*argv, "--out", "o"])


def test_off_by_default_and_named_in_the_help():
    assert _args("adjoint-test", "--row", Q12).plant_error is None
    assert d.plant_request(_args("adjoint-test", "--row", Q12), d.load_table(), "adjoint-test") is None
    a = _args("adjoint-test", "--row", Q12, "--plant-error", "boundary-layer:1.01")
    assert a.plant_error == "boundary-layer:1.01"
    with pytest.raises(SystemExit):   # tangent and param-grad have no such option at all
        _args("tangent", "--plant-error", "acoustic:1.01")


@pytest.mark.parametrize("name,word", [("boundary-layer", "rrfs_mynnedmf_run_l16:auto"),
                                       ("acoustic", "acoustic_ru_regional_v841:rtheta_pp"),
                                       ("advection", "theta_edge_flux_f32:theta")])
def test_each_named_kernel_is_the_runners_word_on_the_12km_row(name, word):
    plant = d.plant_request(_args("adjoint-test", "--row", Q12, "--plant-error", f"{name}:1.001"), d.load_table(),
                            "adjoint-test")
    assert plant["runner_word"] == f"{word}:1.001" and plant["target"] == word
    assert plant["scale"] == 1.001 and plant["error_per_cent"] == pytest.approx(0.1)
    # the runner reads AD_PLANT as <kernel>:<argument>:<factor> (window.rs plant(): splitn(3, ':'))
    k, a, f = plant["runner_word"].split(":", 2)
    assert (k, a, float(f)) == (*word.split(":"), 1.001)


def test_a_global_row_takes_its_own_kernels_and_refuses_what_it_does_not_offer():
    table = d.load_table()
    plant = d.plant_request(_args("adjoint-test", "--row", GLOBAL, "--plant-error", "acoustic:1.01"), table,
                            "adjoint-test")
    assert plant["runner_word"] == "acoustic_ru_v841:rtheta_pp:1.01" and plant["mesh_kind"] == "global"
    with pytest.raises(d.AdDoorError, match="boundary-layer is not offered on a global mesh; there: acoustic, "
                                            "advection. Nothing was downloaded"):
        d.plant_request(_args("adjoint-test", "--row", GLOBAL, "--plant-error", "boundary-layer:1.01"), table,
                        "adjoint-test")
    assert d.plant_request(_args("adjoint-test", "--row", STORM, "--plant-error", "acoustic:1.01"), table,
                           "adjoint-test")["mesh_kind"] == "limited-area"


@pytest.mark.parametrize("asked", ["boundary-layer", "nonsense:1.01", "boundary-layer:1", "boundary-layer:1.0",
                                   "boundary-layer:3", "boundary-layer:0", "boundary-layer:nan", "boundary-layer:x"])
def test_a_bad_request_is_refused_before_any_download(asked):
    with pytest.raises(d.AdDoorError, match=r"give KERNEL:SCALE, KERNEL one of boundary-layer, acoustic, advection "
                                            r"and SCALE a number from 0\.5 to 2 other than 1.*Nothing was downloaded"):
        d.plant_request(_args("adjoint-test", "--row", Q12, "--plant-error", asked), d.load_table(), "adjoint-test")


def test_the_whole_state_dot_cannot_be_skipped_on_a_planted_run():
    with pytest.raises(d.AdDoorError, match="--plant-error with --no-self-dot"):
        d.plant_request(_args("adjoint-test", "--row", Q12, "--plant-error", "acoustic:1.01", "--no-self-dot"),
                        d.load_table(), "adjoint-test")


def test_sensitivity_refuses_the_option_in_one_sentence(capsys):
    args = _args("sensitivity", "--row", Q12, "--plant-error", "boundary-layer:1.01")
    with pytest.raises(d.AdDoorError, match="--plant-error is refused by sensitivity: an error planted on purpose "
                                            "belongs only in adjoint-test.*Nothing was downloaded"):
        d.run_sensitivity(args)


@pytest.mark.parametrize("word", ["AD_PLANT", "AD_PLANT_RECOMPUTE", "AD_PLANT_LEFT_OUT", "AD_PLANT_HOLD"])
@pytest.mark.parametrize("door", ["tangent", "adjoint-test", "sensitivity", "param-grad"])
def test_a_planting_word_left_in_the_environment_is_refused_by_every_door(monkeypatch, word, door):
    monkeypatch.setenv(word, "acoustic_ru_regional_v841:rtheta_pp:1.01")
    with pytest.raises(d.AdDoorError, match=f"{word} is set in the environment: a test-only switch of the runner that "
                                            f"breaks the derivative on purpose, which this {door} run would take"):
        d.plant_request(_args(door, "--row", Q12), d.load_table(), door)
    with pytest.raises(d.AdDoorError, match="adjoint-test --plant-error KERNEL:SCALE$"):   # where a runner starts
        d.refuse_planted_env("window", before_download=False)


def test_sensitivity_stops_on_a_planting_word_before_anything(monkeypatch):
    monkeypatch.setenv("AD_PLANT", "rrfs_mynnedmf_run_l16:auto:1.01")
    with pytest.raises(d.AdDoorError, match="AD_PLANT is set in the environment"):
        d.run_sensitivity(_args("sensitivity", "--row", Q12))


def test_the_planting_word_reaches_every_window_last(tmp_path):
    """Context.window_line (and Context.window) lay the plant over the route's and the window's own words."""
    def window_command(table, row, assets, door, runner, steps, out, extra, env_overrides, key, extra_env, spec, rep):
        return ["runner"], dict(extra_env)

    ctx = SimpleNamespace(table={}, row={}, assets={}, door="adjoint-test", runner="r", steps=15, out=tmp_path,
                          plan=tmp_path, route_env={"AD_P3_DIRECT": "1"},
                          plant_env={"AD_PLANT": "theta_edge_flux_f32:theta:1.01"})
    real = d.window_command
    try:
        d.window_command = window_command
        _, env = d.Context.window_line(ctx, [], sub="selfdot", extra_env={"AD_DOT_SELF": "1", "AD_PLANT": "other"})
        del ctx.plant_env
        _, plain = d.Context.window_line(ctx, [], sub="selfdot", extra_env={"AD_DOT_SELF": "1"})
    finally:
        d.window_command = real
    assert env["AD_PLANT"] == "theta_edge_flux_f32:theta:1.01" and env["AD_DOT_SELF"] == "1"
    assert "AD_PLANT" not in plain


def _window(out, sub, launches, words=10):
    (out / sub).mkdir(parents=True)
    doc = {"valid_derivative": False}
    if launches is not None:
        doc["planted_error"] = {"switch": "AD_PLANT", "kernel": "k", "argument": "a", "factor": 1.01,
                                "launches": launches, "words_changed": words, "valid": False}
    (out / sub / "window.json").write_text(json.dumps(doc))


PLANT = {"kernel": "boundary-layer", "target": "rrfs_mynnedmf_run_l16:auto", "scale": 1.01, "error_per_cent": 1.0}


def test_the_verdict_says_planted_and_caught_with_its_own_exit_status(tmp_path):
    _window(tmp_path, "window", 15)
    _window(tmp_path, "selfdot", 15, 1000)
    reading = d.planted_reading(tmp_path)
    assert reading["selfdot"]["launches"] == 15 and reading["window"]["words_changed"] == 10
    checks = {"dot": {"value": 1e-6, "bar": 2e-5, "pass": True},
              "self_dot": {"value": 2.4e-4, "bar": 4.77e-7, "bar_roundings": 4, "pass": False}}
    line, code, caught = d.planted_verdict(PLANT, checks, reading)
    assert code == d.PLANTED_CAUGHT_EXIT == 4 and caught
    assert line.startswith("PLANTED: an error was planted on purpose (boundary-layer, rrfs_mynnedmf_run_l16:auto, "
                           "x1.01, 1 per cent; 30 backward launches, 1,010 words changed)")
    assert "the transpose check CAUGHT it: whole-state dot 2.400e-04 = 2,013.3 roundings against the bar of 4" in line
    assert "PASS" not in line


def test_the_verdict_says_when_the_check_did_not_catch_it(tmp_path):
    _window(tmp_path, "window", 15)
    _window(tmp_path, "selfdot", 15)
    checks = {"dot": {"value": 1e-6, "bar": 2e-5, "pass": True},
              "self_dot": {"value": 2.0e-7, "bar": 4.77e-7, "bar_roundings": 4, "pass": True}}
    line, code, caught = d.planted_verdict(PLANT, checks, d.planted_reading(tmp_path))
    assert code == d.PLANTED_MISSED_EXIT == 5 and not caught
    assert line.startswith("PLANTED:") and "NOT CAUGHT: the whole-state dot read 2.000e-07 = 1.68 roundings" in line


@pytest.mark.parametrize("selfdot", [0, None])
def test_a_plant_that_never_ran_is_not_read_as_a_test(tmp_path, selfdot):
    _window(tmp_path, "window", 15)
    _window(tmp_path, "selfdot", selfdot)
    checks = {"dot": {"value": 1e-6, "bar": 2e-5, "pass": True},
              "self_dot": {"value": 2.4e-4, "bar": 4.77e-7, "pass": False}}
    line, code, caught = d.planted_verdict(PLANT, checks, d.planted_reading(tmp_path))
    assert code == 5 and not caught and "NOTHING WAS PLANTED in a checked window" in line


def test_a_planted_receipt_never_stands_for_a_real_check(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"schema": d.RECEIPT_SCHEMA, "door": "adjoint-test", "passed": False, "valid": False,
                                "planted_error": {"kernel": "acoustic", "scale": 1.01}}))
    with pytest.raises(d.AdDoorError, match="that run had an error planted on purpose"):
        d.bind_strict(None, path)


def test_the_table_says_what_the_option_does_and_its_exit_statuses():
    spec = d.load_table()["doors"]["adjoint-test"]["plant_error"]
    assert spec["exit"] == {"caught": d.PLANTED_CAUGHT_EXIT, "not_caught": d.PLANTED_MISSED_EXIT}
    assert {d.PLANTED_CAUGHT_EXIT, d.PLANTED_MISSED_EXIT}.isdisjoint({0, 1, 2, d.HOST_TOO_SMALL_EXIT})
    assert "reverse only" in spec["side"] and "tangent are untouched" in spec["side"]
    line = d.plant_line({**PLANT, "what": spec["kernels"]["boundary-layer"]["what"]})
    assert line.startswith("woof hex adjoint-test: AN ERROR IS PLANTED ON PURPOSE (--plant-error boundary-layer:1.01)")
    assert "the transpose check is expected to FAIL" in line

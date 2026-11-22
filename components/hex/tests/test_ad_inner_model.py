# SPDX-License-Identifier: Apache-2.0
"""--inner-model (0.6.2): off by default the runner's line and environment are the ones a window had before
the option existed; simple swaps the boundary layer's derivative for the held-coefficient diffusion and holds the
cloud scheme, radiation, drag, land and sea ice. CPU only."""
import pytest

from woof.hex import ad_doors as D


ROW_ARGV = ["/x/ad-binder", "window", "--config", "plan.json", "--steps", "15", "--tempo", "all", "--tempo-arena-gib",
            "2", "--tempo-cubin", "/a/split2.cubin", "--checkpoints", "sqrt", "--refuse", "ntiedtke_cutypen,cupy_x"]
ROW_ENV = {"AD_P3_DIRECT": "1", "AD_P3_DIRECT_IMAGE": "/a/direct120.cubin", "AD_P3_DIRECT_FWD": "/a/cap120.cubin",
           "AD_P3_HOLD_OUT": "rqcblten", "AD_SCHEMES": "/a/bundle"}


def test_full_is_the_window_unchanged():
    argv, env = D.inner_model_words(list(ROW_ARGV), dict(ROW_ENV))
    assert argv == ROW_ARGV and env == ROW_ENV
    argv, env = D.inner_model_words(list(ROW_ARGV), {**ROW_ENV, D.INNER_MODEL_ENV: "full"})
    assert argv == ROW_ARGV


def test_reduced_default_keeps_the_boundary_layer():
    argv, env = D.inner_model_words(list(ROW_ARGV), {**ROW_ENV, D.INNER_MODEL_ENV: "reduced"})
    assert "AD_IM_BL" not in env and env["AD_P3_DIRECT"] == "1"   # the row's own boundary-layer derivative
    assert D.INNER_MODEL_ENV not in env
    assert argv[argv.index("--hold") + 1] == "tempo"
    refuse = argv[argv.index("--refuse") + 1].split(",")
    assert refuse[:2] == ["ntiedtke_cutypen", "cupy_x"]
    assert set(refuse[2:]) == set(D.INNER_MODEL_HELD)


def test_reduced_any_subset():
    env = {**ROW_ENV, D.INNER_MODEL_ENV: "reduced", D.INNER_MODEL_KEEP_ENV: "microphysics,radiation"}
    argv, env = D.inner_model_words(list(ROW_ARGV), env)
    assert env["AD_IM_BL"] == "held" and not any(k.startswith("AD_P3_DIRECT") for k in env)
    assert "--tempo" in argv and "--hold" not in argv          # microphysics kept: TEMPO's derivative as the row has it
    refuse = argv[argv.index("--refuse") + 1].split(",")
    assert "rlw_taut" not in refuse and "ugwp_oro_run" in refuse and "oracle_land_ruc_land" in refuse
    with pytest.raises(D.AdDoorError):
        D.inner_model_keep("clouds")


def test_simple_words():
    argv, env = D.inner_model_words(list(ROW_ARGV), {**ROW_ENV, D.INNER_MODEL_ENV: "simple"})
    assert env["AD_IM_BL"] == "fixedk"
    assert not any(k.startswith("AD_P3_DIRECT") for k in env)
    assert env["AD_SCHEMES"] == "/a/bundle" and env["AD_P3_HOLD_OUT"] == "rqcblten"
    for gone in ("--tempo", "--tempo-arena-gib", "--tempo-cubin"):
        assert gone not in argv[2:]
    assert argv[argv.index("--hold") + 1] == "tempo"
    refuse = argv[argv.index("--refuse") + 1].split(",")
    assert refuse[:2] == ["ntiedtke_cutypen", "cupy_x"]   # the row's own list first, unchanged
    for k in ("rlw_taumol_batched", "rsw_taumol_b", "ugwp_oro_run", "oracle_land_ruc_land", "oracle_ice_ruc_ice"):
        assert k in refuse
    assert len(refuse) == len(set(refuse))
    assert argv[argv.index("--checkpoints") + 1] == "sqrt"


def test_simple_keeps_other_holds_and_adds_a_refuse_list():
    base = ["/x/ad-binder", "window", "--config", "plan.json", "--hold", "physics"]
    argv, _ = D.inner_model_words(base, {D.INNER_MODEL_ENV: "simple"})
    assert argv[argv.index("--hold") + 1] == "physics,tempo"
    assert argv.count("--hold") == 1
    assert argv[argv.index("--refuse") + 1].split(",") == list(D.INNER_MODEL_HELD)


def test_option_default_off():
    import argparse
    p = argparse.ArgumentParser()
    D._common(p, "steps")
    assert p.parse_args(["--out", "/tmp/x"]).inner_model == "full"
    for m in ("full", "reduced", "simple"):
        assert p.parse_args(["--out", "/tmp/x", "--inner-model", m]).inner_model == m
    with pytest.raises(SystemExit):
        p.parse_args(["--out", "/tmp/x", "--inner-model", "off"])
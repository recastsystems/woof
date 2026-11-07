"""The TEMF surface-layer oracle fixtures of record, on the CPU.

What these pin, without a device: the fixtures are the bytes the oracle
wrote (its own sha256 receipt); the catalogue reaches every branch the
port must reproduce (convective w*, the S < 1e-15 limiter with both signs
of N2, the Ri clamp, the TE floor, MAVAIL = 0, a first level below 10 m);
and the two WRF arms differ ONLY in the log10-words regime, which is what
makes "the C library's log10f" the whole cause of the declared divergence
rather than a story.
"""

from __future__ import annotations

import numpy as np

from temf_sfclay_oracle import (ARMS, FIXTURE_DIR, OUTPUTS, dims, file_sha256,
                                load, recorded_sha256, regime_columns)


def test_fixtures_are_the_oracles_bytes():
    table = recorded_sha256()
    for arm in ARMS:
        for kind in ("inputs", "outputs"):
            name = f"temf-{kind}-{arm}.csv"
            assert file_sha256(name) == table[name], name
    assert file_sha256("log10-probe.txt") == table["log10-probe.txt"]


def test_the_catalogue_is_78_columns_by_8_steps():
    assert dims("wrf") == (8, 78)
    assert dims("wrf-crlog10") == (8, 78)


def test_the_module_was_the_pinned_source():
    table = recorded_sha256()
    assert table["module_sf_temfsfclay.F"] == (
        "70f2ed6bc48aad5b1bf68be4e01753a132f2ecdb4fa6d95c0ecce5d5554fcd94")
    receipt = (FIXTURE_DIR / "compiler.txt").read_text()
    assert "phys/module_sf_temfsfclay.o" in receipt  # WRF's own object


def test_the_catalogue_reaches_every_branch():
    fx = load("wrf")
    u, v, hfx = fx["in:u"], fx["in:v"], fx["in:hfx"]
    calm = regime_columns("calm")
    assert (u[:, calm] == 0).all() and (v[:, calm] == 0).all()
    # w*: positive HFX into the scheme somewhere, and not everywhere
    assert (hfx > 0).any() and (hfx <= 0).any()
    # TE below TEmin arrives and leaves at TEmin
    te_in, te_out = fx["in:te"], fx["out:te"]
    floor = np.float32(1e-3)
    assert (te_in < floor).any()
    assert (te_out[te_in < floor] == floor).all()
    # a first mass level below 10 m
    zm = fx["in:z"] - fx["in:ht"]
    assert (zm < 10).any()
    # WM_TEMF's 1e-2 floor binds somewhere
    assert (fx["out:wm"] == np.float32(1e-2)).any()


def test_wrf_writes_nan_only_where_mavail_is_zero():
    fx = load("wrf")
    dry = regime_columns("dry-mavail0")
    for name in OUTPUTS:
        bad = np.isnan(fx[f"out:{name}"])
        if name in ("exch", "chs"):
            assert bad[:, dry].all()
            bad[:, dry] = False
        assert not bad.any(), name


def test_the_two_wrf_arms_differ_only_in_the_log10_regime():
    a, c = load("wrf"), load("wrf-crlog10")
    cols = set()
    for name in OUTPUTS:
        x = a[f"out:{name}"].view(np.uint32)
        y = c[f"out:{name}"].view(np.uint32)
        cols.update(np.flatnonzero((x != y).any(axis=0)).tolist())
    assert cols, "no column reaches a log10 word the C library misses"
    assert cols <= set(regime_columns("log10-words").tolist())
    # external inputs are shared; only the carried HFX/WM_TEMF follow each arm
    for name in ("u", "v", "th", "qv", "p", "pi", "rho", "z", "ht", "psfc",
                 "znt", "mavail", "tsk", "fcor", "te", "hd"):
        assert np.array_equal(a[f"in:{name}"].view(np.uint32),
                              c[f"in:{name}"].view(np.uint32)), name


def test_the_log10_receipts_say_what_the_port_relies_on():
    probe = (FIXTURE_DIR / "log10-probe.txt").read_text()
    assert "differ from correctly rounded" in probe
    exhaustive = (FIXTURE_DIR / "log10-exhaustive.txt").read_text()
    assert ("positive finite binary32 words: 2139095039 checked, 0 differ"
            in exhaustive)

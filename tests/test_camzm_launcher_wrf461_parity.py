"""Generated stock-WRF diagnostics for the recovered CU7 column launcher.

Set CAMZM_ORACLE_DIR to a regenerated fixture directory. These tests do not
exercise model dispatch, coupled CAM5 state, or a real forecast capture.
Both numeric modes require exact output words in these diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.verify.camzm_oracle import (
    FULL_INPUTS, FULL_OUTPUTS, GROUPS, INT_SURFACE_INPUTS,
    INT_SURFACE_OUTPUTS, MASS_INPUTS, MASS_OUTPUTS, SURFACE_INPUTS,
    SURFACE_OUTPUTS, dims, fixture, mismatches, step_arrays, tile)


@pytest.fixture(autouse=True)
def _generated_streams_required():
    from woof.verify.camzm_oracle import FIXTURE_DIR
    if not (FIXTURE_DIR / "zm-uw12km-stock.bin").is_file():
        pytest.skip("raw CU7 streams were disposed; set CAMZM_ORACLE_DIR")


def _launch(a, settings, *, budget_bytes=None):
    import cupy as cp

    from woof.core import camzm

    def lev(x):
        return cp.asarray(np.ascontiguousarray(x[:, None, :]))

    inputs = {name: lev(a[name]) for name in MASS_INPUTS + FULL_INPUTS}
    for name in SURFACE_INPUTS:
        inputs[name] = cp.asarray(np.ascontiguousarray(a[name][None, :]))
    for name in INT_SURFACE_INPUTS:
        inputs[name] = cp.asarray(np.ascontiguousarray(
            a[name][None, :].astype(np.int32)))
    kwargs = {} if budget_bytes is None else {"budget_bytes": budget_bytes}
    out = camzm.camzm_step(
        inputs, dt=settings["dt"], dx=settings["dx"], cudt=settings["cudt"],
        itimestep=a["itimestep"], bl_pbl_physics=settings["bl_pbl_physics"],
        sf_sfclay_physics=settings["sf_sfclay_physics"], diagnostics=True,
        **kwargs)
    got = {}
    for name in MASS_OUTPUTS + FULL_OUTPUTS:
        got[name] = cp.asnumpy(out[name])[:, 0, :]
    for name in SURFACE_OUTPUTS + INT_SURFACE_OUTPUTS:
        got[name] = cp.asnumpy(out[name])[0]
    return got


def _computed_steps(fx):
    _, _, nsteps = dims(fx)
    return [s for s in range(1, nsteps + 1) if bool(fx[f"s{s}/in/ran"][0])]


@pytest.mark.gpu
@pytest.mark.parametrize("group", GROUPS)
def test_every_computed_step_of_every_column_is_wrfs_words(group):
    import cupy  # noqa: F401  (marks this test for -m "not gpu")

    fx = fixture(group, stock=True)
    settings = tile(fx)
    for step in _computed_steps(fx):
        a = step_arrays(fx, step)
        got = _launch(a, settings)
        bad = mismatches(got, a["out"])
        assert not bad, f"{group} step {step}: differing words per field {bad}"


@pytest.mark.gpu
def test_a_planted_one_bit_change_is_caught():
    """The comparison can fail, and fails on one bit, in both directions.

    (1) One output word of the fixture with its lowest mantissa bit flipped
    is reported as exactly one differing word.  (2) One input word of one
    convecting column with its lowest mantissa bit flipped moves the port's
    answer for that column and for no other column.
    """
    import cupy  # noqa: F401

    fx = fixture("uw12km", stock=True)
    settings = tile(fx)
    a = step_arrays(fx, 1)
    got = _launch(a, settings)
    assert not mismatches(got, a["out"])

    planted = {k: np.array(v, copy=True) for k, v in a["out"].items()}
    word = planted["rthcuten"].view(np.uint32)
    convecting = np.flatnonzero(a["out"]["lengath2d"])
    col = int(convecting[0])
    k = int(np.flatnonzero(planted["rthcuten"][:, col])[0])
    word[k, col] ^= np.uint32(1)
    assert mismatches(got, planted) == {"rthcuten": 1}

    b = dict(a)
    qv = np.array(a["qv"], copy=True)
    qv.view(np.uint32)[0, col] ^= np.uint32(1)
    b["qv"] = qv
    moved = _launch(b, settings)
    bad = mismatches(moved, a["out"])
    assert bad, "a one-bit change of qv in a convecting column moved nothing"
    for name, value in moved.items():
        if np.asarray(value).dtype != np.float32:
            continue
        cols = np.flatnonzero(np.any(
            np.atleast_2d(value).view(np.uint32)
            != np.atleast_2d(got[name]).view(np.uint32), axis=0))
        assert set(cols.tolist()) <= {col}, (name, cols)


@pytest.mark.gpu
def test_a_chunked_launch_equals_a_single_one():
    import cupy  # noqa: F401

    from woof.core.camzm import camzm_workspace_slots

    fx = fixture("myj3km", stock=True)
    settings = tile(fx)
    _, nk, _ = dims(fx)
    a = step_arrays(fx, 1)
    whole = _launch(a, settings)
    five = 5 * 8 * camzm_workspace_slots(nk)      # five columns per launch
    chunked = _launch(a, settings, budget_bytes=five)
    for name, value in whole.items():
        value = np.asarray(value)
        other = np.asarray(chunked[name])
        if value.dtype == np.float32:
            value, other = value.view(np.uint32), other.view(np.uint32)
        assert np.array_equal(value, other), name

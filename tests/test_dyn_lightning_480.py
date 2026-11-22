"""WRF v4.8.0 dynamic lightning (dyn_lightning_option=1).

The GPU oracle test replays real columns against the labelled-patched native
strict referee's stored words (tests/data/oracles/dyn_lightning_480, written by
tools/dyn_lightning_480_replay.py --oracle-dir). The patched referee differs
from the pristine 4.8.0 objects only in calc_lpi_new's undefined kte slots; the
pristine comparison of every defined word is in the port's receipt.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ORACLE = Path(__file__).parent / "data" / "oracles" / "dyn_lightning_480"


def _cfg(**over):
    from woof.core.dyn_lightning import DYN_LIGHTNING_DEFAULTS
    values = dict(DYN_LIGHTNING_DEFAULTS, moist=True, mp_physics=8)
    values.update(over)
    return SimpleNamespace(**values)


def test_off_drops_every_control_so_old_identities_hold():
    from woof.core.dyn_lightning import DYN_LIGHTNING_DEFAULTS, drop_default_dyn_lightning
    values = dict(DYN_LIGHTNING_DEFAULTS, mp_physics=8)
    drop_default_dyn_lightning(values)
    assert values == {"mp_physics": 8}
    on = dict(DYN_LIGHTNING_DEFAULTS, dyn_lightning_option=1)
    drop_default_dyn_lightning(on)
    assert on["dyn_lightning_option"] == 1 and "j_pos" in on


def test_run_config_defaults_are_the_registry_defaults():
    from woof.config import RunConfig
    from woof.core.dyn_lightning import DYN_LIGHTNING_DEFAULTS
    import dataclasses
    fields = {f.name: f.default for f in dataclasses.fields(RunConfig)}
    for key, value in DYN_LIGHTNING_DEFAULTS.items():
        assert fields[key] == value


@pytest.mark.parametrize("over, words", [
    ({"dyn_lightning_option": 2}, "0 or 1"),
    ({"dyn_lightning_option": 1, "mp_physics": 0}, "returns before its lightning block"),
    ({"dyn_lightning_option": 1, "moist": False}, "returns before its lightning block"),
    ({"dyn_lightning_option": 1, "j_neu": 0.0}, "divides the column energy"),
    ({"dyn_lightning_option": 1, "coul_pos": float("nan")}, "must be finite"),
    ({"dyn_lightning_option": 1, "wrf_reference": "4.7.1"}, "WRF v4.8.0 key"),
])
def test_refusals_name_their_breakage(over, words):
    from woof.core.dyn_lightning import validate_dyn_lightning
    with pytest.raises(ValueError, match=words):
        validate_dyn_lightning(_cfg(**over))


def test_tiles_are_disjoint_and_may_leave_out_the_specified_ring():
    from woof.core.dyn_lightning import _tiles
    assert _tiles(4, 3, None) == ((0, 4, 0, 3),)
    assert _tiles(6, 5, ((1, 5, 1, 4),)) == ((1, 5, 1, 4),)
    with pytest.raises(ValueError, match="overlap"):
        _tiles(4, 3, ((0, 2, 0, 3), (1, 3, 0, 3)))
    with pytest.raises(ValueError, match="outside"):
        _tiles(4, 3, ((0, 5, 0, 3),))


def _gpu():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return None


@pytest.mark.skipif(not ORACLE.exists(), reason="dynamic lightning oracle not present")
@pytest.mark.parametrize("name", ["thompson_bound239", "nssl2_column"])
def test_gpu_strict_replay_is_bitwise_to_the_patched_native_referee(name):
    cp = _gpu()
    if cp is None:
        pytest.skip("no CUDA device")
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("strict comparison: run with GPUWM_WRF_EXACT=1")
    from woof.core.dyn_lightning import launch_dyn_lightning
    z = np.load(ORACLE / f"{name}.npz")
    nz, ny, nx = z["th"].shape
    fields = {k: cp.asarray(z[k]) for k in ("th", "pii", "rho", "dz8w", "w", "qc", "qr",
                                            "qi", "qs", "qg", "qh")}
    out = {k: cp.zeros((nz, ny, nx), cp.float32) for k in ("light_pe", "light_ne", "light_neu", "lpi3d")}
    out.update({k: cp.zeros((ny, nx), cp.float32) for k in ("lpos", "lneg", "lneu", "lpi2d")})
    params = {k: float(z[k]) for k in ("dx", "dy", "dt", "coul_pos", "coul_neg", "coul_neu",
                                        "j_pos", "j_neg", "j_neu")}
    saved = [int(s) for s in z["steps"]]
    for step in range(1, saved[-1] + 1):
        launch_dyn_lightning(fields, out, **params)
        if step in saved:
            for key in ("lpi2d", "lpi3d", "light_pe", "light_ne", "light_neu", "lpos", "lneg", "lneu"):
                got = cp.asnumpy(out[key]).view(np.uint32)
                want = z[f"{key}@{step}"].view(np.uint32)
                assert np.array_equal(got, want), (name, step, key, int((got != want).sum()))

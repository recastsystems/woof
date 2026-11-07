"""Noah's SFCDIAGS writes the same words whatever carries the fields.

``PhysicsDriver._refresh_surface_diagnostics`` takes one launch on C-contiguous
float32 CuPy arrays (``woof.core.noah_sfcdiags.refresh``) and runs the same
kernel on contiguous copies for any other carrier (``refresh_any``).  Until
development branch the second path was a set of CuPy array expressions
whose TH2 called CUDA's powf; the kernel now calls WOOF's own float32 powf,
which is WRF's word (tests/test_noah_wrf461_parity.py and
tools/noah_wrf461_oracle/sfcdiags_sweep.py), so the expressions went and both
paths are one arithmetic.  This runs the driver's two paths on the same
inputs and compares Q2, T2 and TH2 as uint32, over columns built to reach
every branch:
exchange coefficients on and next to the 1e-5 activity floor, zero and
negative ones, surface humidity whose flux inversion goes negative (the
Q2 = qv(k=1) guard), signed zeros, infinities and NaN.
"""
# Replaced in 2.9.1 (the WRF exact chain): test_fused_sfcdiags_is_bitwise_the_array_expressions -> test_fused_sfcdiags_is_bitwise_the_other_carrier_path.
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

_INPUTS = ("psfc", "tsk", "cqs2", "chs2", "qsfc", "qfx", "hfx")


def _columns(rng, n):
    f32 = np.float32
    floor = f32(1.0e-5)
    coefficient = np.concatenate([
        np.array([floor, np.nextafter(floor, f32(0)),
                  np.nextafter(floor, f32(1)), 0.0, -0.0, -1.0e-3, np.nan,
                  np.inf, 1.0], dtype=f32),
        rng.uniform(-2.0e-5, 2.0e-2, n - 9).astype(f32)])
    columns = {
        "psfc": rng.uniform(5.0e4, 1.05e5, n).astype(f32),
        "tsk": rng.uniform(200.0, 330.0, n).astype(f32),
        "cqs2": rng.permutation(coefficient),
        "chs2": rng.permutation(coefficient),
        "qsfc": rng.uniform(-5.0e-4, 3.0e-2, n).astype(f32),
        "qfx": rng.uniform(-1.0e-4, 3.0e-4, n).astype(f32),
        "hfx": rng.uniform(-200.0, 600.0, n).astype(f32),
        "qv1": rng.uniform(0.0, 2.5e-2, n).astype(f32),
    }
    special = {
        "psfc": [0.0, -0.0, np.inf, np.nan],
        "tsk": [0.0, -0.0, np.inf, np.nan],
        "qsfc": [0.0, -0.0, np.inf, np.nan],
        "qfx": [0.0, -0.0, -np.inf, np.nan],
        "hfx": [0.0, -0.0, np.inf, np.nan],
        "qv1": [0.0, -0.0, np.inf, np.nan],
    }
    for name, values in special.items():
        at = rng.choice(n, size=len(values), replace=False)
        columns[name][at] = np.asarray(values, dtype=f32)
    return columns


def _driver(cp, columns, shape):
    from woof.core.physics import PhysicsDriver

    driver = PhysicsDriver.__new__(PhysicsDriver)
    driver.fields = {name: cp.asarray(columns[name].reshape(shape))
                     for name in _INPUTS}
    for name in ("q2", "t2", "th2"):
        driver.fields[name] = cp.full(shape, np.float32(7.0),
                                      dtype=cp.float32)
    atmosphere = {"qv": cp.asarray(
        np.stack([columns["qv1"].reshape(shape)] * 3))}
    return driver, atmosphere


@pytest.mark.gpu
@requires_gpu
def test_fused_sfcdiags_is_bitwise_the_other_carrier_path(monkeypatch):
    cp = pytest.importorskip("cupy")
    from woof.core import noah_sfcdiags

    shape = (61, 83)
    columns = _columns(np.random.default_rng(20260930), shape[0] * shape[1])

    fused, atmosphere = _driver(cp, columns, shape)
    launches = []
    real_refresh = noah_sfcdiags.refresh
    monkeypatch.setattr(noah_sfcdiags, "refresh",
                        lambda *a, **k: (launches.append(1),
                                         real_refresh(*a, **k))[1])
    fused._refresh_surface_diagnostics(atmosphere)
    assert launches == [1]

    monkeypatch.setattr(noah_sfcdiags, "device_arrays", lambda *a: False)
    reference, atmosphere = _driver(cp, columns, shape)
    reference._refresh_surface_diagnostics(atmosphere)

    for name in ("q2", "t2", "th2"):
        got = cp.asnumpy(fused.fields[name]).view(np.uint32)
        want = cp.asnumpy(reference.fields[name]).view(np.uint32)
        differing = int(np.count_nonzero(got != want))
        assert differing == 0, f"{name}: {differing} words differ"
    # The inputs reach the humidity guard and both activity branches.
    q2 = cp.asnumpy(reference.fields["q2"]).ravel()
    guarded = q2.view(np.uint32) == columns["qv1"].view(np.uint32)
    assert guarded.any() and not guarded.all()

"""The WRF 3.9 fork's rain fallout (thompson_version = "wrf_39_noaa", audit T21).

The fork counts surface rain above R1*10 (NOAA-EMC/HRRR v4.1.21
module_mp_thompson.F:3556); WRF v4.6.1 counts it above R1*1000 (:3817).
thompson.cu is byte-frozen, so the fork pass is a copy of its rain-presence
pass in thompson_aerosol_sed.cu under THOMPSON_AA_WRF39 with only that test
changed, and with WOOF's own powf word where the frozen pass calls CUDA's.
These tests bind the copy to the frozen pass within that word difference and
show the surface test acts exactly on trace surface rain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

from woof.core import thompson as classic_thompson                # noqa: E402
from woof.core.thompson_aerosol_launch import (                   # noqa: E402
    thompson_version_scope,
)
from woof.core.thompson_aerosol_sed import (                      # noqa: E402
    launch_wrf39_rain_sedimentation,
)

_ROOT = Path(__file__).resolve().parents[1]
NZ, NY, NX = 30, 4, 16
DT = 20.0


def _require_device():
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:  # pragma: no cover
        pytest.skip("no CUDA runtime")


def _columns(seed: int):
    """Mixed columns: heavy rain, light rain, trace surface rain, empty,
    rewritten (negative carried density) and failed-presence levels."""
    rng = np.random.default_rng(seed)
    shape = (NZ, NY, NX)
    z = np.arange(NZ, dtype=np.float32)[:, None, None]
    pressure = (100000.0 * np.exp(-z * 400.0 / 8000.0)
                * np.ones(shape)).astype(np.float32)
    temperature = (295.0 - 6.5e-3 * 400.0 * z
                   + rng.normal(0.0, 0.5, shape)).astype(np.float32)
    qv = (0.012 * np.exp(-z / 10.0) * np.ones(shape)).astype(np.float32)
    rho = 0.622 * pressure / (287.04 * temperature * (qv + 0.622))
    dz = np.full(shape, 400.0, np.float32)
    qr = np.zeros(shape, np.float32)
    kind = np.arange(NY * NX).reshape(NY, NX) % 6
    heavy = rng.uniform(1e-4, 3e-3, shape).astype(np.float32)
    light = rng.uniform(1e-7, 1e-5, shape).astype(np.float32)
    qr = np.where(kind == 0, heavy, qr)
    qr = np.where(kind == 1, light, qr)
    # Trace: a little rain aloft whose surface concentration sits between
    # R1*10 and R1*1000 kg/m3.
    trace = np.zeros(shape, np.float32)
    trace[:6] = rng.uniform(5e-11, 5e-10, (6, NY, NX)).astype(np.float32)
    qr = np.where(kind == 2, trace, qr)
    qr = np.where(kind == 4, light, qr)
    qr = np.where(kind == 5, heavy * (z < 8), qr).astype(np.float32)
    nr = (qr * rng.uniform(1e5, 1e7, shape)).astype(np.float32)
    density = rho.astype(np.float32)
    density = np.where(qr > 1e-12, density, 0.0).astype(np.float32)
    density = np.where((kind == 4)[None] & (z % 3 == 0), -density, density)
    density = density.astype(np.float32)
    return dict(qr=qr, nr=nr, temperature=temperature, pressure=pressure,
                qv=qv, dz=dz, density=density, kind=kind)


def _run(columns, fork: bool):
    d = {k: cp.asarray(v) for k, v in columns.items() if k != "kind"}
    # Zero running totals, so a trace export is not absorbed by rounding.
    rainnc = cp.zeros((NY, NX), cp.float32)
    rainncv = cp.zeros((NY, NX), cp.float32)
    if fork:
        with thompson_version_scope("wrf_39_noaa"):
            launch_wrf39_rain_sedimentation(
                d["qr"], d["nr"], d["temperature"], d["pressure"], d["qv"],
                d["dz"], rainnc, rainncv, DT,
                reference_density=d["density"])
    else:
        classic_thompson.launch_rain_sedimentation(
            d["qr"], d["nr"], d["temperature"], d["pressure"], d["qv"],
            d["dz"], rainnc, rainncv, DT, reference_density=d["density"],
            accumulate_surface=True, density_carries_rain_presence=True)
    cp.cuda.Device().synchronize()
    return {name: cp.asnumpy(value) for name, value in (
        ("qr", d["qr"]), ("nr", d["nr"]),
        ("rainnc", rainnc), ("rainncv", rainncv))}


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _ulps(a, b):
    def line(x):
        v = np.ascontiguousarray(x, np.float32).view(np.int32).astype(np.int64)
        return np.where(v < 0, -(v & 0x7FFFFFFF), v)
    return np.abs(line(a) - line(b))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fork_rain_fallout_is_the_frozen_pass_except_the_surface_test(seed):
    _require_device()
    columns = _columns(seed)
    fork = _run(columns, fork=True)
    v461 = _run(columns, fork=False)
    # The column state is the frozen pass's, operation for operation, with
    # one named difference: every REAL(4) ** in the copy is WOOF's own powf
    # word (thompson_aerosol_libm.cuh, the word the fork's gfortran build
    # calls), where the byte-frozen thompson.cu keeps CUDA's powf.  The two
    # words differ by one ulp on some arguments, and the sub-stepped fallout
    # carries that a few ulps further, so the copy is held to the frozen pass
    # within the spread development branch measured on these
    # three seeds: qr 3 ulps, nr 8, surface rain 2.
    for name, bound in (("qr", 3), ("nr", 8)):
        assert _ulps(fork[name], v461[name]).max() <= bound, name
    kind = columns["kind"]
    # Columns whose surface rain is never in (R1*10, R1*1000] export the
    # same words up to that libm difference; the empty columns export
    # nothing under either test.
    same = kind != 2
    for name in ("rainnc", "rainncv"):
        assert _ulps(fork[name][same], v461[name][same]).max() <= 2, name
    assert np.all(v461["rainncv"][kind == 3] == 0.0)
    assert np.all(v461["rainncv"][kind == 0] > 0.0)
    # Trace surface rain: v4.6.1 drops it, the fork counts it.
    trace = kind == 2
    assert np.all(v461["rainncv"][trace] == 0.0)
    assert np.all(v461["rainnc"][trace] == 0.0)
    assert np.all(fork["rainncv"][trace] > 0.0)
    np.testing.assert_array_equal(_bits(fork["rainnc"][trace]),
                                  _bits(fork["rainncv"][trace]))
    assert np.all(fork["rainncv"][trace] < 1.0e-6)


def test_the_fork_pass_refuses_outside_the_fork_generation():
    _require_device()
    columns = _columns(0)
    d = {k: cp.asarray(v) for k, v in columns.items() if k != "kind"}
    surface = cp.zeros((NY, NX), cp.float32)
    with pytest.raises(RuntimeError, match="fork"):
        launch_wrf39_rain_sedimentation(
            d["qr"], d["nr"], d["temperature"], d["pressure"], d["qv"],
            d["dz"], surface, surface.copy(), DT,
            reference_density=d["density"])


def test_the_adapter_routes_rain_by_generation():
    """The coupled adapter sends the fork generation to the fork pass and
    v4.6.1 to its tendency-form accumulator pass (the mp=28 accumulator
    rework), whose pptrain the surface totals then add in mp_gt_driver's
    order (development branch); no generation reaches the classic
    in-place rain launcher."""
    text = (_ROOT / "woof" / "core" / "microphysics_aerosol.py").read_text(
        encoding="utf-8")
    start = text.index("launch_wrf39_rain_sedimentation(\n")
    block = text[text.rindex("if wrf39:", 0, start):start + 900]
    assert "else:\n" in block
    assert "launch_aa_rain_sedimentation_accumulate(" in block
    assert "export_surface=True" in block
    assert "launch_aa_surface_precipitation(" in block
    assert "launch_rain_sedimentation(" not in text.replace(
        "launch_aa_rain_sedimentation_accumulate(", "")

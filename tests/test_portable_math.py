"""The host preparation's transcendentals are the same bits on every host.

Breakage prevented: NumPy's float64 and float32 exp, log, pow, arcsin, tan
and arctan take their own vector loops on an AVX-512 Linux host and the C
library elsewhere, and C libraries differ between glibc releases and the
MSVC runtime, so the real-init base state (phb, pb, alb and the float32
geopotential split built from them) and the other preparation paths
A130 lists came out with different last bits on different machines.
``woof.core.portable_math`` routes them through the vendored libm crate in
the CPU preprocessing library; the pins below were measured on WSL Ubuntu
24.04 (glibc 2.39, NumPy 2.5.3) with AVX-512 dispatch on and off and on
test host 4 (glibc 2.43, no AVX-512), identical on all three.
"""

from __future__ import annotations

import hashlib
import math
import threading

import numpy as np
import pytest

from woof.core import portable_math as pm


def _library_or_skip():
    if pm.implementation() != pm.IMPLEMENTATION:
        pytest.skip("the CPU preprocessing library with the portable math "
                    "entries is not built here")


def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:24]


def _inputs(dtype):
    """Deterministic arguments spanning each function's working range."""
    # No transcendental builds an input (np.geomspace would), so the
    # arguments are the same bits on every host.
    ramp = np.linspace(-30.0, 30.0, 40_001)
    positive = np.concatenate([np.linspace(1.0e-6, 1.0e-3, 10_000),
                               np.linspace(1.0e-3, 2.0, 15_000),
                               np.linspace(2.0, 2.0e5, 15_001)])
    unit = np.linspace(-1.0, 1.0, 40_001)
    ratio = np.linspace(-0.2, 0.9, 40_001)
    return {name: np.asarray(values, dtype=dtype) for name, values in (
        ("ramp", ramp), ("positive", positive), ("unit", unit),
        ("ratio", ratio))}


_UNARY_CASES = (
    ("exp", "ramp"), ("log", "positive"), ("log1p", "ratio"),
    ("log10", "positive"), ("sin", "ramp"), ("cos", "ramp"),
    ("tan", "unit"), ("arcsin", "unit"), ("arccos", "unit"),
    ("arctan", "ramp"),
)

#: sha256[:24] of each function's words over :func:`_inputs`, then of
#: power (base ``positive``, exponent R/cp and exponent ``unit``) and of
#: arctan2 (``ramp`` over ``unit``).
PORTABLE_WORD_PINS = {
    "float64": "a05bb48c6c8bb56aa6266ad9",
    "float32": "b656b44f2ac55aad808c731d",
}


def _all_words(dtype):
    values = _inputs(dtype)
    outputs = [getattr(pm, name)(values[argument])
               for name, argument in _UNARY_CASES]
    outputs.append(pm.power(values["positive"], 0.2857142857142857))
    outputs.append(pm.power(values["positive"], values["unit"]))
    outputs.append(pm.arctan2(values["ramp"], values["unit"]))
    return outputs


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_portable_words_are_pinned(dtype):
    _library_or_skip()
    outputs = _all_words(np.dtype(dtype))
    for output in outputs:
        assert output.dtype == np.dtype(dtype)
    assert _digest(*outputs) == PORTABLE_WORD_PINS[dtype]


def test_portable_words_are_within_an_ulp_of_the_c_library():
    """A sanity bound, not the contract: the libm crate is not the host's
    libm.  Measured against glibc 2.39 on these arguments it differs on 65
    (log1p) to 4,206 (log10) of 40,001, by one float64 ulp, and by two for
    FDLIBM's log10."""
    _library_or_skip()
    values = _inputs(np.float64)
    for name, argument in _UNARY_CASES:
        got = getattr(pm, name)(values[argument])
        reference = {"arcsin": math.asin, "arccos": math.acos,
                     "arctan": math.atan}.get(name, getattr(math, name, None))
        want = np.array([reference(float(v)) for v in values[argument]])
        finite = np.isfinite(want)
        gap = np.abs(got[finite] - want[finite])
        bound = 2.0 if name == "log10" else 1.0
        assert np.all(gap <= bound * np.spacing(np.abs(want[finite]))), name


def test_worker_count_and_calling_thread_never_change_an_element():
    _library_or_skip()
    values = _inputs(np.float64)["positive"]
    one = pm.log(values, workers=1)
    many = pm.log(values, workers=7)
    result = {}
    thread = threading.Thread(
        target=lambda: result.setdefault("value", pm.log(values)))
    thread.start()
    thread.join()
    assert one.tobytes() == many.tobytes() == result["value"].tobytes()


def test_a_worker_limit_bounds_the_library_threads(monkeypatch):
    """--preprocess-workers reaches the library on the serial setup path.

    Breakage prevented: real.py's column helpers run their serial branch
    on the calling thread, where the library took the automatic count (up
    to eight threads) even when the preparation was given fewer."""
    from woof.ingest import cpu_backend

    monkeypatch.setattr(cpu_backend, "automatic_workers", lambda: 8)
    assert pm._workers(None) == 8
    with pm.worker_limit(3):
        assert pm._workers(None) == 3
        assert pm._workers(5) == 5
        with pm.worker_limit(1):
            assert pm._workers(None) == 1
        assert pm._workers(None) == 3
    assert pm._workers(None) == 8
    for bad in (0, -2):
        with pytest.raises(ValueError):
            with pm.worker_limit(bad):
                pass
    for bad in (2.0, True, None):
        with pytest.raises(TypeError):
            with pm.worker_limit(bad):
                pass


def test_real_init_serial_helpers_pass_their_worker_count(monkeypatch):
    _library_or_skip()
    from woof.ingest import cpu_backend
    from woof.ingest.real import (_make_real_base, _moist_specific_volume,
                                   _potential_temperature_from_temperature,
                                   _saturation_mixing_ratio,
                                   _temperature_from_potential_temperature)

    monkeypatch.setattr(cpu_backend, "automatic_workers", lambda: 8)
    original = pm._workers
    seen = []

    def spy(workers, size=None):
        seen.append(original(workers, size))
        return seen[-1]

    monkeypatch.setattr(pm, "_workers", spy)
    coord, terrain = _real_base_case(nz=6, ny=5, nx=7)
    base = _make_real_base(coord, terrain, 5000.0, 290.0, column_workers=1)
    pb = np.asarray(base.pb)
    temperature = np.full(pb.shape, 280.0)
    theta = _potential_temperature_from_temperature(
        temperature, pb, column_workers=1)
    _temperature_from_potential_temperature(theta, pb, column_workers=1)
    _moist_specific_volume(theta, np.full(pb.shape, 0.01), pb,
                           column_workers=1)
    _saturation_mixing_ratio(temperature, pb, np.full(pb.shape, 50.0),
                             column_workers=1)
    assert seen and set(seen) == {1}
    seen.clear()
    # A one-row slab takes the serial branch with the count it was given.
    _potential_temperature_from_temperature(
        temperature[:1], pb[:1], column_workers=4)
    assert set(seen) == {4}
    seen.clear()
    # An unnamed count is the automatic one, given at least
    # AUTOMATIC_ELEMENTS_PER_WORKER elements per worker.
    pm.exp(np.ones(3))
    assert seen == [1]
    seen.clear()
    pm.exp(np.ones(8 * pm.AUTOMATIC_ELEMENTS_PER_WORKER))
    assert seen == [8]


def test_operands_keep_numpy_shapes_and_precision():
    _library_or_skip()
    x32 = np.linspace(0.5, 2.0, 12, dtype=np.float32).reshape(3, 4)
    assert pm.power(x32, 0.2857).dtype == np.float32
    assert pm.power(x32, np.float64(0.2857)).dtype == np.float64
    assert pm.exp(x32).shape == (3, 4)
    broadcast = pm.power(x32[:, :1], np.arange(4.0))
    assert broadcast.shape == (3, 4)
    want = np.array([[pm.power(float(a), float(b)) for b in range(4)]
                     for a in x32[:, 0]])
    assert broadcast.tobytes() == want.tobytes()
    scalar = pm.exp(1.0)
    assert isinstance(scalar, np.float64)
    assert abs(scalar - math.e) <= np.spacing(math.e)
    out = np.array([1.0, 2.0])
    assert pm.log(out, out=out) is out
    assert out[0] == 0.0


def test_fallback_is_math_element_by_element_and_says_so_once(
        monkeypatch, capsys):
    monkeypatch.setattr(pm, "_resolved", True)
    monkeypatch.setattr(pm, "_library", None)
    monkeypatch.setattr(pm, "_absent_reason", "no library in this test")
    monkeypatch.setattr(pm, "_warned", False)
    assert pm.implementation() == pm.FALLBACK_IMPLEMENTATION
    values = np.array([-1.0, 0.0, 0.5, 700.0, 710.0])
    assert pm.exp(values).tolist() == [
        math.exp(-1.0), 1.0, math.exp(0.5), math.exp(700.0), math.inf]
    logs = pm.log(np.array([-1.0, 0.0, 2.0]))
    assert math.isnan(logs[0]) and logs[1] == -math.inf
    assert logs[2] == math.log(2.0)
    assert pm.power(np.array([0.0, -8.0, 4.0]), -1.0).tolist() == [
        math.inf, -0.125, 0.25]
    assert pm.power(np.float32(2.0), np.float32(0.5)).dtype == np.float32
    error = capsys.readouterr().err
    assert error.count("[portable-math] WORKAROUND") == 1
    assert "no library in this test" in error


def test_an_explicit_bridge_that_is_missing_fails_loudly(
        monkeypatch, tmp_path):
    from woof.ingest.cpu_backend import CPU_BRIDGE_ENV

    monkeypatch.setenv(CPU_BRIDGE_ENV, str(tmp_path / "absent.so"))
    monkeypatch.setattr(pm, "_resolved", False)
    monkeypatch.setattr(pm, "_library", None)
    with pytest.raises(OSError):
        pm.exp(np.ones(3))


def _real_base_case(nz=40, ny=48, nx=64, hybrid_opt=2):
    from woof.core.grid import make_vertical_coord

    coord = make_vertical_coord(nz, hybrid_opt=hybrid_opt, etac=0.2)
    yy, xx = np.meshgrid(np.linspace(0.0, 1.0, ny), np.linspace(0.0, 1.0, nx),
                         indexing="ij")
    # Built without transcendentals: the same bits on every host.
    bump = np.maximum(
        0.0, 1.0 - ((xx - 0.6) ** 2 + (yy - 0.4) ** 2) / 0.05) ** 2
    terrain = 150.0 + 2600.0 * bump + 40.0 * (xx - xx * xx * xx) * (1.0 - 2.0 * yy)
    return coord, terrain


#: sha256[:24] of (mub, pb, alb, thb, phb) of the real-init analytic base
#: state and of the float32 geopotential split built on it, for each
#: hypsometric_opt, over :func:`_real_base_case` on the terrain-following
#: coordinate.
REAL_INIT_BASE_PINS = {
    1: {"base": "55c58f43ea0f8d5abdb34c2d", "split": "bd7e82bc63264ffab906cb25"},
    2: {"base": "98ce8d4e9a5b5fa36e155be8", "split": "8bfcb34c8952f198313042f7"},
}

#: The same on the hybrid coordinate (hybrid_opt = 2, the default).  Its
#: cubic is WRF's closed form in plain float64 (A130); with the solve and
#: NumPy's cube it had three answers across test host 4 and WSL with AVX-512
#: dispatch on and off.  Measured identical on those three.
REAL_INIT_HYBRID_BASE_PINS = {
    1: {"base": "4e95d0fb0c37d9f35782eb04", "split": "13ba888a9c6e091f226d8f61"},
    2: {"base": "47f16f4a0e6e5f9458f2807e", "split": "59b11e65f7fafbb78606e477"},
}


def _real_init_digests(opt, hybrid_opt=0):
    from woof.ingest.real import _fp32_geopotential_split, _make_real_base

    coord, terrain = _real_base_case(hybrid_opt=hybrid_opt)
    base = _make_real_base(coord, terrain, 5000.0, 290.0,
                           hypsometric_opt=opt, column_workers=3)
    ny, nx = terrain.shape
    across = np.linspace(-1.0, 1.0, ny)
    wave = (across * across * across)[:, None] * (
        1.0 - np.linspace(-1.0, 1.0, nx) ** 2)[None, :]
    dry_mass = np.asarray(base.mub) + 150.0 * wave
    alpha = np.asarray(base.alb) * (1.0 + 0.01 * wave[None])
    php = _fp32_geopotential_split(base, coord, dry_mass, alpha,
                                   hypsometric_opt=opt, column_workers=3)
    return {"base": _digest(base.mub, base.pb, base.alb, base.thb, base.phb),
            "split": _digest(php)}


@pytest.mark.parametrize("opt", [1, 2])
def test_real_init_base_state_digest_is_pinned(opt):
    _library_or_skip()
    assert _real_init_digests(opt) == REAL_INIT_BASE_PINS[opt]


@pytest.mark.parametrize("opt", [1, 2])
def test_real_init_hybrid_base_state_digest_is_pinned(opt):
    _library_or_skip()
    assert (_real_init_digests(opt, hybrid_opt=2)
            == REAL_INIT_HYBRID_BASE_PINS[opt])


def test_real_init_thermodynamics_are_the_portable_expressions():
    """The base state and the T/theta/alpha conversions are spelled with
    the portable functions, element for element."""
    _library_or_skip()
    from woof.core import constants as c
    from woof.ingest.real import (_make_real_base, _moist_specific_volume,
                                   _potential_temperature_from_temperature,
                                   _temperature_from_potential_temperature)

    coord, terrain = _real_base_case(nz=12, ny=9, nx=11)
    base = _make_real_base(coord, terrain, 5000.0, 290.0)
    pb = np.asarray(base.pb)
    temperature = np.maximum(200.0, 290.0 + 50.0 * pm.log(pb / c.P0))
    thb = temperature * pm.power(c.P0 / pb, c.RCP)
    assert np.asarray(base.thb).tobytes() == thb.tobytes()
    alb = c.RD * thb * pm.power(pb / c.P0, c.RCP) / pb
    assert np.asarray(base.alb).tobytes() == alb.tobytes()
    theta = _potential_temperature_from_temperature(
        temperature, pb, column_workers=2)
    assert theta.tobytes() == (
        temperature * pm.power(c.P0 / pb, c.RCP)).tobytes()
    back = _temperature_from_potential_temperature(theta, pb)
    assert back.tobytes() == (theta * pm.power(pb / c.P0, c.RCP)).tobytes()
    qv = np.full(pb.shape, 0.01)
    alpha = _moist_specific_volume(theta, qv, pb, column_workers=2)
    theta_m = theta * (1.0 + c.RVOVRD * qv)
    assert alpha.tobytes() == (
        c.RD * theta_m * pm.power(pb / c.P0, c.RCP) / pb).tobytes()


def test_automatic_worker_count_scales_with_the_array(monkeypatch):
    """An automatic count gives each worker at least
    AUTOMATIC_ELEMENTS_PER_WORKER elements; a named count or a worker_limit
    is taken as given.  A thread count never changes an element."""
    from woof.core import portable_math as pm
    from woof.ingest import cpu_backend
    monkeypatch.setattr(cpu_backend, "automatic_workers", lambda: 100)
    per = pm.AUTOMATIC_ELEMENTS_PER_WORKER
    assert pm._workers(None) == 100
    assert pm._workers(None, 1) == 1
    assert pm._workers(None, 3 * per) == 3
    assert pm._workers(None, 1000 * per) == 100
    assert pm._workers(7, 1) == 7
    with pm.worker_limit(5):
        assert pm._workers(None, 1) == 5

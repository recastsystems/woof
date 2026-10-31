"""Noah's fused forcing prologue writes exactly what the statements wrote.

``PhysicsDriver._run_noah`` fills its forcing fields and clears the rain
carriers with one launch each on CuPy arrays (``woof.core.noah_forcing``)
and keeps its CuPy statements otherwise.  This runs the runner's two paths on
the same inputs, with the Noah kernel itself replaced by a probe that records
every field the kernel would read, and compares those fields and the cleared
carriers as uint32, for the scheme's SR and for SR from temperature, over
inputs with temperatures on and next to freezing, signed zeros, infinities
and NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

_WRITTEN = ("sfcprs", "sfctmp", "qv1", "dz8w1", "rib", "sr", "rainbl")


def _salted(rng, shape, low, high):
    f32 = np.float32
    a = rng.uniform(low, high, shape).astype(f32)
    flat = a.reshape(-1)
    special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0e-40],
                       dtype=f32)
    at = rng.choice(flat.size, size=special.size, replace=False)
    flat[at] = special
    return a


def _run(monkeypatch, cp, rng_seed, use_scheme_sr):
    from woof.core import physics

    rng = np.random.default_rng(rng_seed)
    nz, ny, nx = 4, 23, 31
    shape = (ny, nx)
    freezing = np.float32(273.15)
    temperature = _salted(rng, (nz, ny, nx), 250.0, 300.0)
    near = np.array([np.nextafter(freezing, np.float32(0)), freezing,
                     np.nextafter(freezing, np.float32(400))],
                    dtype=np.float32)
    temperature[0].reshape(-1)[:3] = near
    atmosphere = {
        "p_interface": cp.asarray(_salted(rng, (nz + 1, ny, nx), 5e4, 1e5)),
        "temperature": cp.asarray(temperature),
        "qv": cp.asarray(_salted(rng, (nz, ny, nx), 0.0, 0.02)),
        "dz": cp.asarray(_salted(rng, (nz, ny, nx), 20.0, 400.0)),
    }
    driver = physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    fields = {name: cp.full(shape, np.float32(-7.0), dtype=cp.float32)
              for name in _WRITTEN + ("tsk",)}
    fields["br"] = cp.asarray(_salted(rng, shape, -5.0, 5.0))
    fields["rainbl"] = cp.asarray(_salted(rng, shape, 0.0, 3.0))
    driver.fields = fields
    driver._pending_rainbl = cp.asarray(_salted(rng, shape, 0.0, 3.0))
    driver.microphysics = type("M", (), {})()
    driver.microphysics.sr = cp.asarray(_salted(rng, shape, 0.0, 1.0))
    driver.mp_physics = 10
    driver.noah_params = None
    driver.bldt_seconds = 15.0
    driver.noah_usemonalb = False
    driver.noah_rdlai2d = False
    driver.noah_opt_thcnd = 1
    # The MYJ CHKLOWQ arm (paper/conus-suite-exact, 2.9.11) is off with a
    # non-MYJ surface layer, as initialize_physics sets it.
    driver.noah_myj = False
    seen = {}

    def probe(f, *args, **kwargs):
        for name in _WRITTEN:
            seen[name] = cp.asnumpy(f[name]).view(np.uint32).copy()

    monkeypatch.setattr(physics, "launch_noah", probe)
    # seaice_noah follows lsm in _run_noah (2.8.6); it reads none of the
    # forcing statements under test and has its own WRF oracle gate.
    driver._run_noah_seaice = lambda *args, **kwargs: None
    monkeypatch.setattr(physics, "microphysics_scheme_sr_available",
                        lambda mp: use_scheme_sr)
    driver._run_noah(atmosphere, None, 1)
    after = {name: cp.asnumpy(v).view(np.uint32)
             for name, v in (("rainbl", fields["rainbl"]),
                             ("pending", driver._pending_rainbl))}
    return seen, after


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("use_scheme_sr", [True, False])
def test_fused_noah_forcing_is_bitwise_the_statements(monkeypatch,
                                                      use_scheme_sr):
    cp = pytest.importorskip("cupy")
    from woof.core import noah_forcing

    launches = []
    real_forcing, real_clear = noah_forcing.forcing, noah_forcing.rain_clear
    monkeypatch.setattr(noah_forcing, "forcing",
                        lambda *a, **k: (launches.append("forcing"),
                                         real_forcing(*a, **k))[1])
    monkeypatch.setattr(noah_forcing, "rain_clear",
                        lambda *a, **k: (launches.append("clear"),
                                         real_clear(*a, **k))[1])
    fused = _run(monkeypatch, cp, 20260930, use_scheme_sr)
    assert launches == ["forcing", "clear"]
    monkeypatch.setattr(noah_forcing, "device_arrays", lambda *a: False)
    reference = _run(monkeypatch, cp, 20260930, use_scheme_sr)
    for got, want in zip(fused, reference):
        assert sorted(got) == sorted(want)
        for name in want:
            differing = int(np.count_nonzero(got[name] != want[name]))
            assert differing == 0, f"{name}: {differing} words differ"

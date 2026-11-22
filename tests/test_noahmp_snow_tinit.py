"""noahmp_snow_tinit: cold-start snow-layer temperatures (woof.core.noahmp_snow_tinit)."""
import numpy as np
import pytest

from woof.core.noahmp_snow_tinit import (TFRZ, conduction_profile_host,
                                          validate)


@pytest.mark.parametrize("bad", [0, 3, 1.0, "2", None, True])
def test_bad_values_refused(bad):
    with pytest.raises(ValueError, match="noahmp_snow_tinit"):
        validate(bad)


def test_runtime_parameters_carry_and_refuse():
    from woof.core.noahmp_runtime import NoahmpRuntimeParameters
    assert NoahmpRuntimeParameters().snow_tinit == 1
    p = NoahmpRuntimeParameters(snow_tinit=2)
    assert p.snow_tinit == 2
    # kept out of the option mapping: an option-1 run keys exactly as before
    assert "snow_tinit" not in p.land_options
    assert dict(p.land_options) == dict(NoahmpRuntimeParameters().land_options)
    with pytest.raises(ValueError):
        NoahmpRuntimeParameters(snow_tinit=7)


def test_run_config_default_is_conduction_profile():
    import dataclasses
    from woof.config import RunConfig
    f = {x.name: x for x in dataclasses.fields(RunConfig)}["noahmp_snow_tinit"]
    assert f.default == 2


def test_host_profile_by_hand():
    # two layers (ISNOW=-2): 5 cm over 10 cm, skin 248 K, top soil 266 K, H = 0.15 m
    zsnso = np.array([np.nan, -0.05, -0.15, -0.25, -0.55, -1.15, -2.15], np.float32)
    t = conduction_profile_host(isnow=-2, zsnso=zsnso, tg=np.float32(248.0),
                                tslb1=np.float32(266.0), xice=0.0, xice_threshold=0.5)
    assert t[0] is None
    assert t[1] == pytest.approx(248.0 + 18.0 * 0.025 / 0.15, abs=1e-4)   # 251.0
    assert t[2] == pytest.approx(248.0 + 18.0 * 0.10 / 0.15, abs=1e-4)    # 260.0
    # capped at freezing, untouched without snow or on sea ice
    t = conduction_profile_host(isnow=-1, zsnso=np.array([np.nan, np.nan, -0.04, -0.14, 0, 0, 0], np.float32),
                                tg=np.float32(270.0), tslb1=np.float32(285.0), xice=0.0, xice_threshold=0.5)
    assert t[2] == TFRZ
    assert conduction_profile_host(isnow=0, zsnso=zsnso, tg=250.0, tslb1=260.0, xice=0.0, xice_threshold=0.5) is None
    assert conduction_profile_host(isnow=-2, zsnso=zsnso, tg=250.0, tslb1=260.0, xice=0.9, xice_threshold=0.5) is None


def test_device_matches_host_reference():
    cp = pytest.importorskip("cupy")
    from woof.core.noahmp_snow_tinit import apply_conduction_profile
    rng = np.random.default_rng(7)
    ny, nx = 9, 11
    isnow = rng.integers(-3, 1, size=(ny, nx)).astype(np.int32)
    dz = rng.uniform(0.02, 0.3, size=(3, ny, nx)).astype(np.float32)
    zsnso = np.zeros((7, ny, nx), np.float32)
    for j in range(ny):
        for i in range(nx):
            n = -isnow[j, i]; z = np.float32(0.0)
            for k in range(3 - n, 3):
                z = np.float32(z - dz[k, j, i]); zsnso[k, j, i] = z
            for k in range(3):
                zsnso[3 + k, j, i] = np.float32(z - 0.1 * (k + 1))
    tg = rng.uniform(235.0, 275.0, size=(ny, nx)).astype(np.float32)
    tslb = rng.uniform(255.0, 285.0, size=(4, ny, nx)).astype(np.float32)
    xice = np.where(rng.random((ny, nx)) < 0.15, 1.0, 0.0).astype(np.float32)
    tsno0 = np.full((3, ny, nx), 123.0, np.float32)
    fields = {"isnowxy": cp.asarray(isnow), "zsnsoxy": cp.asarray(zsnso), "tgxy": cp.asarray(tg),
              "tslb": cp.asarray(tslb), "xice": cp.asarray(xice), "tsnoxy": cp.asarray(tsno0)}
    apply_conduction_profile(fields, xice_threshold=0.5)
    got = cp.asnumpy(fields["tsnoxy"])
    touched = 0
    for j in range(ny):
        for i in range(nx):
            ref = conduction_profile_host(isnow=isnow[j, i], zsnso=zsnso[:, j, i], tg=tg[j, i],
                                          tslb1=tslb[0, j, i], xice=xice[j, i], xice_threshold=0.5)
            for k in range(3):
                want = tsno0[k, j, i] if ref is None or ref[k] is None else ref[k]
                assert got[k, j, i].tobytes() == np.float32(want).tobytes(), (j, i, k)
                touched += ref is not None and ref[k] is not None
    assert touched > 50


def test_strict_wrf_process_defaults_to_the_wrf_pin():
    """GPUWM_WRF_EXACT=1 is the 0 ULP proof of WRF: its default must be
    WRF SNOW_INIT (1), or a strict Noah-MP cold start over snow stops
    matching native WRF without any key saying so."""
    import json, os, subprocess, sys
    probe = ("import json; from woof.config import RunConfig; "
             "print(json.dumps(RunConfig.__dataclass_fields__['noahmp_snow_tinit'].default))")
    env = dict(os.environ, GPUWM_WRF_EXACT="1", GPUWM_NO_LOCAL_GPU="1")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == 1

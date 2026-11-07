"""Real SAS callable, held driver rates and checkpoint round trip."""
import gzip
from pathlib import Path
import numpy as np
import pytest
cp = pytest.importorskip("cupy")


def _fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    from woof.config import RunConfig
    from woof.core.state import DomainState
    from woof.core.physics import initialize_physics
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "ksas_wrf461_oracle"))
    import gen_cases
    root = Path(__file__).parent / "fixtures" / "ksas_wrf461"
    for name in ("cases.bin", "ksas-oracle.bin"):
        (tmp_path / name).write_bytes(gzip.decompress((root / (name + ".gz")).read_bytes()))
    groups = gen_cases.read(tmp_path / "cases.bin")
    group = groups[0]
    answer = gen_cases.read_outputs(tmp_path / "ksas-oracle.bin", groups)[0]
    nz = group["nz"]
    cfg = RunConfig(nx=8, ny=8, nz=nz, dx=float(group["dx"]), dy=float(group["dx"]),
                    ztop=18000., dt=float(group["dt"]), run_seconds=600.,
                    moist=True, mp_physics=group["mp_physics"], cu_physics=14,
                    cudt_minutes=0.)
    state = DomainState(cfg)
    state.mub2d[...] = cp.float32(1.e5)
    state.c1h[...] = cp.float32(1.)
    state.c2h[...] = cp.float32(0.)
    state.znu[...] = cp.asarray(group["znu"])
    state.w[...] = cp.asarray(group["arrays"]["w"]).reshape(nz + 1, 8, 8)
    driver = initialize_physics(state, cfg)
    a = group["arrays"]
    for name, key in [("pblh", "hpbl"), ("xland", "xland")]:
        driver.fields[name][...] = cp.asarray(a[key]).reshape(8, 8)
    driver.cumulus_callable.hold_pbl_height(cp.asarray(a["hpbl_hold"]).reshape(8, 8))
    mapping = dict(temperature="t3d", qv="qv3d", qc="qc3d", qi="qi3d",
                   pressure="p3d", exner="pi3d", rho="rho3d", dz="dz8w",
                   u="u3d", v="v3d")
    atmos = {k: cp.asarray(a[v]).reshape(nz, 8, 8) for k, v in mapping.items()}
    return state, cfg, driver, atmos, answer


def _bits(a):
    return cp.asnumpy(a).view(np.uint32)


def test_stock_driver_keeps_six_tendencies_and_held_precipitation(tmp_path):
    from woof.core.physics import couple_column_tendencies
    state, cfg, driver, atmos, want = _fixture(tmp_path)
    driver._run_cumulus(atmos, state, cfg)
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten"):
        assert np.array_equal(_bits(driver.cu_rates[name]).reshape(want[name].shape),
                              want[name].view(np.uint32)), name
    def arr(name):
        return cp.asarray(want[name]).reshape(cfg.nz, 8, 8)
    expected = couple_column_tendencies(
        state, cfg, rtheta=arr("rthcuten"), rqv=arr("rqvcuten"),
        rqc=arr("rqccuten"), rqi=arr("rqicuten"),
        ru=arr("rucuten"), rv=arr("rvcuten"))
    for name in ("rtheta", "rqv", "rqc", "rqi", "ru", "rv"):
        assert np.array_equal(_bits(getattr(driver.cumulus_tendencies, name)),
                              _bits(getattr(expected, name))), name
    for name in ("hbot", "htop"):
        assert np.array_equal(_bits(driver.fields[name]).reshape(-1),
                              want[name].view(np.uint32))
    # A moved nest rebuilds its coupled rates from the carried raw fields.
    # Check momentum too: the scalar CU_* rate table does not carry it.
    for name in ("rucuten", "rvcuten"):
        assert np.array_equal(_bits(driver.fields['ksas_' + name]).reshape(want[name].shape),
                              want[name].view(np.uint32))
    driver.recouple_cumulus_tendencies(state, cfg)
    for name in ("rtheta", "rqv", "rqc", "rqi", "ru", "rv"):
        assert np.array_equal(_bits(getattr(driver.cumulus_tendencies, name)),
                              _bits(getattr(expected, name))), name
    total = np.zeros((8, 8), np.float32)
    rate = want["pratec"].reshape(8, 8)
    for step in range(5):
        state.elapsed_seconds = step * cfg.dt
        driver._advance_cumulus_clock(state, cfg)
        total += rate * np.float32(cfg.dt)
        assert np.array_equal(_bits(driver.rainc), total.view(np.uint32))
        assert np.all(cp.asnumpy(driver.cu_nca) == -100.)
    assert bool(cp.any(driver.cumulus_tendencies.ru))
    assert bool(cp.any(driver.cumulus_tendencies.rv))


def test_checkpoint_accepts_sas_scratch_and_keeps_held_state(tmp_path):
    from woof.io.restart import write_restart, restore_restart
    from woof.core.state import DomainState
    from woof.core.physics import initialize_physics
    state, cfg, driver, atmos, want = _fixture(tmp_path)
    driver._run_cumulus(atmos, state, cfg)
    file = write_restart(tmp_path / "sas-rst.npz", state, cfg)
    fresh, resumed_cfg, new, _, _ = _fixture(tmp_path / "resume")
    assert resumed_cfg == cfg
    restore_restart(file, fresh, cfg)
    for name in ("ru", "rv", "rtheta", "rqv", "rqc", "rqi"):
        assert np.array_equal(_bits(getattr(new.cumulus_tendencies, name)),
                              _bits(getattr(driver.cumulus_tendencies, name))), name
    assert np.array_equal(_bits(new.cu_pratec), _bits(driver.cu_pratec))
    assert np.array_equal(_bits(new.cu_raincv), _bits(driver.cu_raincv))
    for name in ('hbot', 'htop', 'ksas_rucuten', 'ksas_rvcuten'):
        assert np.array_equal(_bits(new.fields[name]), _bits(driver.fields[name])), name
    assert new.cumulus_callable._workspace is None


def test_snapshot_is_released_between_steps(tmp_path):
    state, cfg, driver, atmos, want = _fixture(tmp_path)
    adapter = driver.cumulus_callable
    assert adapter._hold_live
    adapter.reset_pbl_height_hold()
    assert not adapter._hold_live
    adapter(atmosphere=atmos, fields=driver.fields, state=state, cfg=cfg)
    assert not adapter._hold_live

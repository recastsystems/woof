"""Real CUDA kernels and resident checkpoint stream, no device mocks."""
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from conftest import requires_gpu
from woof.core.tmn_update import advance, launch
from woof.io import restart
from tools.tmn_wrf461_oracle.validate import (
    FIELDS, initial, oracle_function, oracle_step, validate)

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    existing = os.environ.get("WOOF_TMN_ORACLE")
    if existing:
        assert Path(existing).is_file()
        return existing
    source = os.environ.get("WRF_SOURCE_ROOT")
    if not source:
        pytest.skip("public WRF v4.6.1 source required via WRF_SOURCE_ROOT")
    build = tmp_path_factory.mktemp("tmn-oracle")
    subprocess.run(["bash", "tools/tmn_wrf461_oracle/build.sh", source, str(build)], check=True)
    return str(build / "libtmn.so")


def test_tmn_full_year_oracle(oracle):
    result = validate(oracle)
    assert result["passed"], result


def test_tmn_constant_bit_mutation_is_caught(oracle):
    assert validate(oracle, steps=24, mutate=True)["mutation_caught"]


def test_tmn_calendar_edge_words(oracle):
    func = oracle_function(oracle)
    host = initial()
    gpu = {k: cp.asarray(v) for k, v in host.items()}
    tsk = np.linspace(200., 330., 64, dtype=np.float32).reshape(1, 64)
    cases = []
    for dt in (np.float32(30.), np.float32(90.), np.float32(30.125)):
        for boundary in (np.float32(150.), np.float32(365.), np.float32(366.)):
            fraction = np.float32(np.float32(dt/2.) / np.float32(86400.))
            for signed in (-fraction, fraction):
                center = np.float32(boundary+signed)
                cases.append((np.nextafter(center, np.float32(-np.inf)), dt, 2024))
                cases.extend([(center, dt, 2024),
                              (np.nextafter(center, np.float32(np.inf)), dt, 2024)])
    cases.extend([(np.float32(365.), np.float32(3600.), 2023),
                  (np.float32(366.), np.float32(3600.), 2024),
                  (np.float32(150.5), np.float32(86400.), 2024)])
    for julian, dt, year in cases:
        host = initial()
        gpu = {k: cp.asarray(v) for k, v in host.items()}
        oracle_step(func, host, tsk, julian, dt, year)
        launch(gpu, cp.asarray(tsk), julian=julian, dt=dt, year=year)
        for k in FIELDS:
            assert cp.asnumpy(gpu[k]).tobytes() == host[k].tobytes(), (k, julian, dt, year)


@pytest.mark.parametrize("split", [13, 75*24+13])
def test_tmn_production_checkpoint_midday_and_midring(tmp_path, split):
    from test_physics_driver import _full_state
    start = datetime(2023, 8, 1)
    state, cfg, driver = _full_state(nx=64, ny=1, dt=3600., tmn_update=1,
                                    radiation_start_time=start)
    state2, cfg2, driver2 = _full_state(nx=64, ny=1, dt=3600., tmn_update=1,
                                       radiation_start_time=start)
    total = 151*24+25
    before_id = {k: driver2.fields[k] for k in FIELDS}
    for step in range(total):
        driver.fields["tsk"][...] = cp.float32(270.+(step % 24)*.25)
        advance(driver.fields, start_time=start, elapsed_seconds=step*3600., dt=3600.)
        if step == split:
            state.elapsed_seconds = float((step+1)*3600.)
            path = restart.write_restart(tmp_path/"restart.npz", state, cfg)
            header = restart.read_restart_header(path)
            assert {"fields/"+k for k in FIELDS} <= set(header["array_manifest"])
            restart.restore_restart(path, state2, cfg2)
            for k in FIELDS:
                assert driver2.fields[k] is before_id[k]
                assert cp.asnumpy(driver2.fields[k]).tobytes() == cp.asnumpy(driver.fields[k]).tobytes()
        elif step > split:
            driver2.fields["tsk"][...] = driver.fields["tsk"]
            advance(driver2.fields, start_time=start, elapsed_seconds=step*3600., dt=3600.)
    hashes = {}
    for k in FIELDS:
        a,b = cp.asnumpy(driver.fields[k]), cp.asnumpy(driver2.fields[k])
        assert a.tobytes() == b.tobytes(), k
        hashes[k] = hashlib.sha256(a.tobytes()).hexdigest()
    (tmp_path/"hashes.json").write_text(json.dumps(hashes, indent=2))


def test_tmn_real_driver_runs_before_land_every_step(oracle):
    from test_restart import _physics_state
    from woof.core.dycore import run_steps
    start=datetime(1974, 4, 3, 23, 59, 50)
    state,cfg,driver = _physics_state(cp, nx=8, ny=8, dt=5., tmn_update=1, bldt=1./3.)
    driver.tmn_start_time = start
    host = {k: cp.asnumpy(driver.fields[k]) for k in FIELDS}
    func = oracle_function(oracle)
    for step in range(12):
        valid = start+timedelta(seconds=step*5.)
        julian = np.float32(valid.timetuple().tm_yday+(valid.hour+valid.minute/60.+valid.second/3600.)/24.)
        tsk=cp.asnumpy(driver.fields["tsk"])
        oracle_step(func, host, tsk, julian, 5., valid.year)
        run_steps(state,cfg,1)
        for k in FIELDS:
            assert cp.asnumpy(driver.fields[k]).tobytes() == host[k].tobytes(), (step,k)
    # Surface/PBL cadence held at 20 s, TMN still accumulates each 5 s step.
    assert driver.call_counts["noah"] < 12



def test_tmn_checkpoint_binds_calendar_origin(tmp_path):
    from test_physics_driver import _full_state
    start = datetime(2023, 8, 1)
    state, cfg, driver = _full_state(tmn_update=1, radiation_start_time=start)
    other, cfg2, driver2 = _full_state(tmn_update=1, radiation_start_time=start+timedelta(hours=1))
    path = restart.write_restart(tmp_path/"calendar.npz", state, cfg)
    before = {k: cp.asnumpy(driver2.fields[k]).tobytes() for k in FIELDS}
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path, other, cfg2)
    assert all(cp.asnumpy(driver2.fields[k]).tobytes() == before[k] for k in FIELDS)



def test_tmn_full_physics_restart_trajectory(tmp_path):
    from test_restart import _physics_state, _assert_restart_equal
    from woof.core.dycore import run_steps
    states = [_physics_state(cp, tmn_update=1, nx=8, ny=8, dt=5.) for _ in range(3)]
    start = datetime(1974, 4, 3, 23, 59, 50)
    for state,cfg,driver in states:
        driver.tmn_start_time = start
    state,cfg,driver = states[0]
    run_steps(state,cfg,40)
    uninterrupted = restart.write_restart(tmp_path/"uninterrupted.npz",state,cfg)
    state2,cfg2,driver2 = states[1]
    run_steps(state2,cfg2,13)
    middle = restart.write_restart(tmp_path/"middle.npz",state2,cfg2)
    state3,cfg3,driver3 = states[2]
    restart.restore_restart(middle,state3,cfg3)
    run_steps(state3,cfg3,27)
    resumed = restart.write_restart(tmp_path/"resumed.npz",state3,cfg3)
    _assert_restart_equal(resumed,uninterrupted)
    a,b = restart._load_restart(resumed,with_arrays=True), restart._load_restart(uninterrupted,with_arrays=True)
    # Full archive hashes of every stored array, including the live model state.
    hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in a[1].items()}
    assert hashes == {k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in b[1].items()}
    (tmp_path/"full-field-hashes.json").write_text(json.dumps(hashes,indent=2))


def test_tmn_live_health_checks_annual_counter():
    from test_physics_driver import _full_state
    from woof.core.health import StateHealthValidator
    state, cfg, driver = _full_state(tmn_update=1, radiation_start_time=datetime(2023, 8, 1))
    validator = StateHealthValidator(state)
    assert validator.validate(phase="tmn-initialized").ok
    driver.fields["nyear"][...] = cp.int32(-1)
    report = validator.validate(phase="tmn-planted-negative-counter")
    assert not report.ok and report.first_bad_field == "surface.nyear"

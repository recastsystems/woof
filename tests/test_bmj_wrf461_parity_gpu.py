"""Betts-Miller-Janjic on the device, against the WRF v4.6.1 column oracle.

The device half of tests/test_bmj_wrf461_parity.py (whose docstring
describes the capture): the production launcher, woof/core/column_cumulus.py,
runs the kernel over the committed capture.  Run under GPUWM_WRF_EXACT=1
(strict arithmetic) and without it (default); both are held at 0 ULP on
every field against the correctly rounded gamma arm, because every
arithmetic operation in bmj.cu is a rounded intrinsic.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from test_bmj_wrf461_parity import (  # noqa: E402
    GAMMA_FIELDS, GAMMA_MAX_ULP, _spec, io, oracle)  # noqa: F401


def _device_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    return cp


def _device_report(oracle):
    cp = _device_or_skip()
    from woof.core.column_cumulus import column_tables, run_columns

    spec = _spec()
    ncol, nz = io.read_cases(oracle / "cases.bin")
    tab = column_tables(spec)                 # verifies the pinned hash
    otab = np.fromfile(oracle / "tables.bin", dtype="<f4")[:int(spec.tables["floats"])]
    rep = {"tables": int(io.ulp_distance(cp.asnumpy(tab), otab).max())}
    for arm, (fb, dt, stepcu) in io.ARMS.items():
        for ref in ("crgamma", "stock"):
            steps = io.read_arm(oracle / f"{ref}-arm{arm}.bin", ncol, nz)
            cld = cp.asarray(steps[0]["in_cldefi"])
            worst = {}
            for st in steps:
                out = run_columns(
                    spec,
                    level={k: st["in_" + k] for k in
                           ("t", "qv", "pmid", "pi", "rho", "dz8w")},
                    interface={"pint": st["in_pint"]},
                    column={"xland": st["in_xland"]},
                    carried={"cldefi": cld}, dt=dt, stepcu=stepcu,
                    rad_feedback=fb, tables=tab)
                got = {k: cp.asnumpy(v) for k, v in out.items()}
                got["cutop"], got["cubot"] = got["htop"], got["hbot"]
                got["cldefi"] = cp.asnumpy(cld)
                for k in io.LEVEL_OUTPUTS + io.SURFACE_OUTPUTS:
                    worst[k] = max(worst.get(k, 0),
                                   int(io.ulp_distance(got[k], st[k]).max()))
            rep[f"arm{arm}-{ref}"] = worst
    return rep


def test_device_matches_the_oracle(oracle):
    """Strict arithmetic (GPUWM_WRF_EXACT=1): 0 ULP on every field against
    the correctly rounded gamma arm.  Default arithmetic runs the same
    kernel, whose every operation is a rounded intrinsic, and is held to
    the same bar; the receipt records the measured distance per field."""
    rep = _device_report(oracle)
    import json
    import os
    if os.environ.get("BMJ_RECEIPT_DIR"):
        from pathlib import Path
        Path(os.environ["BMJ_RECEIPT_DIR"], "device-parity.json").write_text(json.dumps(rep, indent=1)+"\n")
    assert rep["tables"] == 0, rep
    for name, worst in rep.items():
        if name == "tables":
            continue
        for field, ulp in worst.items():
            if name.endswith("crgamma") or field not in GAMMA_FIELDS:
                assert ulp == 0, (name, field, ulp)
            else:
                assert ulp <= GAMMA_MAX_ULP, (name, field, ulp)


def test_the_route_refuses_tables_that_are_not_bmjinit():
    cp = _device_or_skip()
    import dataclasses
    from woof.core import column_cumulus as cc

    spec = _spec()
    bad = dataclasses.replace(
        spec, tables=dict(spec.tables, sha256="0" * 64))
    cc._TABLE_CACHE.pop((cp.cuda.Device().id, spec.module), None)
    with pytest.raises(RuntimeError, match="moist adiabat"):
        cc.column_tables(bad)
    cc._TABLE_CACHE.pop((cp.cuda.Device().id, spec.module), None)


def _complete_manifest(state):
    from woof.io import restart
    return (restart.state_manifest(state) |
            restart._scratch_manifest(state) |
            restart._driver_manifest(state.physics))


def _driver_case(oracle, arm):
    from woof.config import RunConfig, validate_run_config
    from woof.core.state import DomainState
    from woof.core.physics import initialize_physics
    fb, dt, stepcu = io.ARMS[arm]
    ncol, nz = io.read_cases(oracle / 'cases.bin')
    cfg = validate_run_config(RunConfig(
        nx=12, ny=8, nz=nz, dx=12000., dy=12000., ztop=14000.,
        run_seconds=0., moist=True, dt=dt, cu_physics=2,
        cudt_minutes=dt*stepcu/60.0))
    # The callable check deliberately exercises both oracle arms without
    # invoking radiation. The full-state check below uses real radiation.
    from dataclasses import replace
    cfg = replace(cfg, bmj_rad_feedback=fb)
    state = DomainState(cfg)
    steps = io.read_arm(oracle / f'crgamma-arm{arm}.bin', ncol, nz)
    driver = initialize_physics(
        state, cfg, xland=steps[0]['in_xland'].reshape(8,12), glw=0.0)
    driver.stepcu = stepcu
    return state, cfg, driver, steps


@pytest.mark.parametrize('arm', [1, 2])
def test_driver_carries_clouds_and_rain_across_checkpoint(oracle, tmp_path, arm):
    from woof.io import restart
    state, cfg, driver, steps = _driver_case(oracle, arm)
    fb, dt, stepcu = io.ARMS[arm]
    mapping = dict(_spec().level_inputs)
    expected_rain = np.zeros((8,12), dtype=np.float32)
    for index, st in enumerate(steps):
        atmosphere = {key: cp.asarray(st['in_'+name]).reshape(cfg.nz,8,12)
                      for name,key in mapping.items()}
        atmosphere['p_interface'] = cp.asarray(st['in_pint']).reshape(cfg.nz+1,8,12)
        driver._run_cumulus(atmosphere, state, cfg)
        for name in ('cldefi','convcld','ccldfra','qcconv','qiconv','pratec','raincv'):
            got = cp.asnumpy(driver.fields[name])
            assert got.tobytes() == st[name].reshape(got.shape).tobytes(), name
        assert cp.asnumpy(driver.cu_rates['rthcuten']).tobytes() == st['rthcuten'].tobytes()
        assert cp.asnumpy(driver.cu_rates['rqvcuten']).tobytes() == st['rqvcuten'].tobytes()
        # Non-due steps must still accumulate the same held rate. This
        # invokes the actual production advance_ppt route, not a mirror.
        for _ in range(stepcu):
            driver._advance_cumulus_clock(state, cfg)
            expected_rain = np.float32(expected_rain + np.float32(st['pratec'].reshape(8,12)*np.float32(dt)))
            assert cp.asnumpy(driver.rainc).tobytes() == expected_rain.tobytes()
            state.elapsed_seconds += dt
        if index == 1:
            before = {k: cp.asnumpy(v).tobytes() for k,v in _complete_manifest(state).items()}
            path = restart.write_restart(tmp_path/'carry.npz', state, cfg)
            fresh, fresh_cfg, fresh_driver, _ = _driver_case(oracle, arm)
            restart.restore_restart(path, fresh, fresh_cfg)
            after = {k: cp.asnumpy(v).tobytes() for k,v in _complete_manifest(fresh).items()}
            assert after == before
            assert "fields/cldefi" in after and "fields/convcld" in after
            assert "fields/ccldfra" in after
            state, cfg, driver = fresh, fresh_cfg, fresh_driver


def test_radiation_merge_preserves_prognostics(oracle):
    from woof.core.column_cumulus import merge_column_cumulus_clouds
    state, cfg, driver, steps = _driver_case(oracle, 2)
    st = steps[0]
    shape=(cfg.nz,8,12)
    for name in ('qcconv','qiconv','ccldfra'):
        driver.fields[name][...] = cp.asarray(st[name]).reshape(shape)
    qc = cp.full(shape, np.float32(2e-5))
    qi = cp.full(shape, np.float32(3e-6))
    fraction = cp.full(shape, np.float32(.3))
    old = [cp.asnumpy(a).tobytes() for a in (qc,qi,fraction)]
    got = merge_column_cumulus_clouds(qc,qi,fraction, fields=driver.fields,
                                     cfg=cfg,columns=lambda a:a)
    a=np.float32(.3);b=st['ccldfra'].reshape(shape)
    expected=(np.float32(np.float32(2e-5)+st['qcconv'].reshape(shape)),
              np.float32(np.float32(3e-6)+st['qiconv'].reshape(shape)),
              np.float32(np.float32(a+b)-np.float32(np.float32(.5)*np.float32(np.minimum(a,b)+np.maximum(np.float32(0),np.float32(np.float32(a+b)-np.float32(1)))))))
    for field, ref in zip(got,expected):
        assert cp.asnumpy(field).tobytes() == ref.tobytes()
    assert [cp.asnumpy(a).tobytes() for a in (qc,qi,fraction)] == old


def _full_state(*, variant="rte-rrtmgp", radiation=4, nz=40):
    from datetime import datetime
    from woof.config import RunConfig, validate_run_config
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced
    from woof.core.physics import initialize_physics
    radiation_settings = ({'ra_physics': 0, 'ra_lw_physics': 5, 'ra_sw_physics': 5}
                          if radiation == 5 else {'ra_physics': radiation})
    cfg = validate_run_config(RunConfig(
        nx=8, ny=6, nz=nz, dx=12000., dy=12000., ztop=14000.,
        dt=30., run_seconds=0., time_step_sound=4, moist=True,
        mp_physics=6, sf_sfclay_physics=2, sf_surface_physics=2,
        bl_pbl_physics=2, radt_minutes=3., bldt=0.,
        cu_physics=2, cudt_minutes=1.5, bmj_rad_feedback=True,
        ra_rrtmg_variant=variant, **radiation_settings))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z:300.+.0025*np.maximum(np.asarray(z,np.float64)-2000.,0.),
                           p_surf=cfg.p_surf,ztop=cfg.ztop)
    state = init_moist_balanced(cfg,coord,base,
                              lambda z:.019*np.exp(-np.asarray(z,np.float64)/5000.))
    state.u[...] = np.float32(4.)
    state.v[...] = np.float32(1.)
    land=np.ones((6,8),np.float64);land[:,-1]=0.
    tsk=np.full((6,8),308.);tsk[:,-1]=303.
    soil_t=np.stack([tsk-.5,tsk-1.,tsk-1.5,tsk-2.])
    soil_m=np.full((4,6,8),.35);soil_m[:,land==0]=1.
    driver=initialize_physics(state,cfg,landmask=land,tsk=tsk,
        soil_temperature=soil_t,soil_moisture=soil_m,liquid_moisture=soil_m,
        ivgtyp=np.where(land,10,17),isltyp=np.where(land,6,14),vegfra=55.,tmn=292.,
        swdown=850.,glw=380.,pblh=1200.,
        radiation_start_time=datetime(2021,6,1,18),
        radiation_latitude=np.full((6,8),39.),radiation_longitude=np.full((6,8),-87.))
    return state,cfg,driver


def test_goddard_consumes_bmj_clouds_without_changing_model_moisture():
    """Exercise the actual table solver and its optical packing on device."""
    from dataclasses import replace
    from woof.core.physics import _prepare_atmosphere
    from woof.core.radiation_composition import ColumnTableRadiation
    # This is an optical-coupling check, independent of BMJ triggering.
    # Twelve layers bound the table solver's private device storage on a
    # shared card; the restart checks retain forty layers.
    state,cfg,driver = _full_state(radiation=5,nz=12)
    # A cloud layer above the boundary layer, independent of BMJ triggering.
    driver.fields['qcconv'][3:6] = np.float32(3e-4)
    driver.fields['qiconv'][6:9] = np.float32(1e-4)
    driver.fields['ccldfra'][3:9] = np.float32(.75)
    before = {name:cp.asnumpy(getattr(state,name)).tobytes()
              for name in ('qc','qi')}
    atmosphere = _prepare_atmosphere(state)
    from datetime import datetime
    adapter = ColumnTableRadiation(datetime(2021,6,1,18),
        np.full((6,8),39.),np.full((6,8),-87.))
    without = adapter(atmosphere=atmosphere,fields=driver.fields,
                      state=state,cfg=replace(cfg,bmj_rad_feedback=False))
    with_cloud = adapter(atmosphere=atmosphere,fields=driver.fields,state=state,cfg=cfg)
    a = cp.asnumpy(without.cldfra)
    b = cp.asnumpy(driver.fields['ccldfra'])
    expected = np.float32(np.float32(a+b)-np.float32(np.float32(.5)*
        np.float32(np.minimum(a,b)+np.maximum(np.float32(0),
        np.float32(np.float32(a+b)-np.float32(1))))))
    assert cp.asnumpy(with_cloud.cldfra).tobytes() == expected.tobytes()
    assert bool(cp.all(cp.isfinite(with_cloud.swdown)))
    assert float(cp.max(cp.abs(with_cloud.swdown-without.swdown))) > 1.
    assert {name:cp.asnumpy(getattr(state,name)).tobytes()
            for name in ('qc','qi')} == before


@pytest.mark.parametrize('variant', ['rte-rrtmgp', 'rrtmg_legacy'])
def test_full_myj_bmj_restart_matches_uninterrupted(tmp_path, variant):
    import hashlib
    import json
    from woof.core.dycore import run_steps
    from woof.io import restart
    control,cfg,driver=_full_state(variant=variant)
    run_steps(control,cfg,8)
    split,cfg_split,_=_full_state(variant=variant)
    run_steps(split,cfg_split,3)
    checkpoint=restart.write_restart(tmp_path/'split.npz',split,cfg_split)
    resumed,cfg_resumed,_=_full_state(variant=variant)
    restart.restore_restart(checkpoint,resumed,cfg_resumed)
    run_steps(resumed,cfg_resumed,5)
    a=_complete_manifest(control);b=_complete_manifest(resumed)
    assert set(a)==set(b)
    assert {"fields/cldefi", "fields/qcconv", "fields/qiconv",
            "fields/ccldfra", "fields/convcld", "scratch/cu_pratec"} <= set(a)
    hashes={}
    for name in a:
        ha=cp.asnumpy(a[name]);hb=cp.asnumpy(b[name])
        assert ha.tobytes()==hb.tobytes(),name
        hashes[name]=hashlib.sha256(ha.tobytes()).hexdigest()
    assert driver.call_counts['cumulus'] == 3
    assert bool(cp.any(driver.fields['cldefi'] != np.float32(.6))), 'BMJ must change its carried efficiency'
    # Receipts retain every serialized field hash, not raw model data.
    import os
    if os.environ.get('BMJ_RECEIPT_DIR'):
        from pathlib import Path
        Path(os.environ['BMJ_RECEIPT_DIR'],f'restart-hashes-{variant}.json').write_text(json.dumps(hashes,indent=1)+'\n')

"""Device smoke for the cu_physics=3 route through the ENGINE's call path.

Not a parity harness: no oracle CSV is read here.  The parity truth lives
in tests/test_gf_gfdrv_cuda.py (GFDRV bitwise at the WRF boundary); this
file proves the other half of admission -- that a RunConfig asking for
Grell-Freitas gets Grell-Freitas through initialize_physics and
PhysicsDriver.compute, on the same seams every scheme crosses: the
resolver binding, the bind_driver hook, the Task-1 CumulusResult contract,
the cu_rates hold, and the RAINC accumulation.
"""

from __future__ import annotations

import numpy as np
import pytest
from _run_config_fake import run_config_fake

from conftest import requires_gpu

cp = pytest.importorskip("cupy")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("ice_active", [False, True])
def test_real_adapter_routes_cold_condensate_from_its_kernel(ice_active, monkeypatch):
    """The actual kernel split reaches the correct stock publication slot."""
    from types import SimpleNamespace as NS
    from woof.core import gf
    from woof.verify.gf_oracle import load_gf_oracle
    from woof.wrf_exact import ENABLED
    fixture=load_gf_oracle()
    col=int(np.flatnonzero(np.any(fixture.levels["rqicuten"] != 0,axis=1))[0])
    lev=fixture.levels; sf=fixture.surface
    grid=lambda name:cp.asarray(lev[name][col,:,None,None].copy())
    volume=grid("p")
    state=NS(p=volume,ht=cp.asarray([[sf["ht"][col]]],cp.float32),
             w=cp.concatenate((grid("w"),cp.zeros((1,1,1),cp.float32))),
             qi=cp.zeros_like(volume) if ice_active else None)
    atmosphere={dst:grid(src) for dst,src in
                (("u","u"),("v","v"),("temperature","t"),("qv","qv"),
                 ("pressure","p"),("exner","pi"),("rho","rho"),("dz","dz8w"))}
    atmosphere["p_interface"]=cp.concatenate((grid("p8w"),cp.zeros((1,1,1),cp.float32)))
    fields={name:cp.asarray([[sf[name][col]]],cp.float32) for name in ("hfx","qfx","xland")}
    fields["kpbl"]=cp.asarray([[sf["kpbl"][col]]],cp.int32)
    adapter=gf.GrellFreitas()
    adapter.bind_driver(NS(rthratenlw=grid("rthraten"),rthratensw=cp.zeros_like(volume),
                          gf_rthdynten=grid("rthften"),gf_rqvdynten=grid("rqvften"),
                          gf_rthblten=grid("rthblten"),gf_rqvblten=grid("rqvblten")))
    module=gf._gf_module(volume.shape[0]); saved=[]
    class Function:
        def __init__(self,fn): self.fn=fn
        def __getattr__(self,name): return getattr(self.fn,name)
        def __call__(self,launch_grid,block,args):
            self.fn(launch_grid,block,args);saved.append(args[3].copy())
    class Module:
        def get_function(self,name): return Function(module.get_function(name))
    monkeypatch.setattr(gf,"_gf_module",lambda *args,**kwargs:Module())
    cfg=run_config_fake(dt=float(sf["dt"][col]),dx=float(sf["dx"][col]),
           ishallow=int(sf["ishallow"][col]),clos_choice=int(sf["ichoice"][col]),lightning_option=0)
    result=adapter(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
    liquid=saved[0][0,gf._OUT_LEV.index("rqccuten"),:,None,None]
    ice=saved[0][0,gf._OUT_LEV.index("rqicuten"),:,None,None]
    assert bool(cp.any(ice != 0).item()), "the selected column must exercise cold routing"
    expected=liquid if ENABLED or ice_active else liquid+ice
    cp.testing.assert_array_equal(result.rqccuten.view(cp.uint32),expected.view(cp.uint32))
    if ice_active:
        cp.testing.assert_array_equal(result.rqicuten.view(cp.uint32),ice.view(cp.uint32))
    else:
        assert result.rqicuten is None


def _state_for(cfg):
    from woof.core.grid import make_base_state, make_vertical_coord
    from woof.core.moist import init_moist_balanced

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return init_moist_balanced(
        cfg, coord, base,
        lambda z: 0.010 * np.exp(-np.asarray(z) / 2200.0))


@pytest.mark.gpu
@requires_gpu
def test_cu_physics_3_routes_to_grell_freitas_and_steps():
    from woof.config import RunConfig, validate_run_config
    from woof.core.gf import GrellFreitas
    from woof.core.physics import initialize_physics

    cfg = validate_run_config(RunConfig(
        nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
        bl_pbl_physics=1, sf_sfclay_physics=91, bldt=0.0,
        cu_physics=3, cudt_minutes=0.0, ishallow=1))
    state = _state_for(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=290.0)
    # cu_physics=3 auto-binds the Grell-Freitas adapter, the same
    # scheme-ID binding law as KF's.
    assert isinstance(driver.cumulus_callable, GrellFreitas)

    tendencies = driver.compute(state, cfg)
    assert driver.call_counts["cumulus"] == 1
    # GF runs on the model step: a second step is a second due call.
    state.elapsed_seconds = cfg.dt
    driver.compute(state, cfg)
    assert driver.call_counts["cumulus"] == 2

    # The bind_driver hook connected the adapter to the held radiation
    # rates (zero here -- radiation off -- but the wiring must exist).
    assert driver.cumulus_callable._driver is driver

    shape = state.p.shape
    for name in ("rthcuten", "rqvcuten", "rqccuten", "rqicuten"):
        rate = driver.cu_rates[name]
        assert rate.shape == shape
        assert bool(cp.isfinite(rate).all()), name
    # Task-1 contract: RAINC accumulated the kernel's RAINCV increment.
    assert driver.rainc.shape == shape[1:]
    assert bool(cp.isfinite(driver.rainc).all())
    assert bool((driver.rainc >= 0).all())
    assert bool(cp.isfinite(tendencies.rtheta).all())
    assert bool(cp.isfinite(tendencies.rqv).all())


@pytest.mark.gpu
@requires_gpu
def test_the_engine_identity_is_the_corrected_k22():
    """The shipped algorithm IS the corrected indexing, stated where a
    restart would bind it; the WRF-faithful flag has no RunConfig spelling
    and is reachable only by the parity suites' direct kernel launches."""
    import inspect

    from woof.core import gf as gf_module
    from woof.io.restart import CUMULUS_ALGORITHM_IDENTITIES

    assert "corrected-k22" in CUMULUS_ALGORITHM_IDENTITIES[3]
    source = inspect.getsource(gf_module)
    assert "SHIPPED default: corrected k22" in source
    from woof.config import RunConfig
    assert "k22_wrf_faithful" not in RunConfig.__dataclass_fields__


def test_grell_freitas_requires_a_pbl_scheme_and_pinned_cudt():
    from woof.config import RunConfig, validate_run_config

    base = dict(nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
                dt=60.0, run_seconds=0.0, moist=True, mp_physics=10)
    with pytest.raises(ValueError, match="requires a PBL scheme"):
        validate_run_config(RunConfig(
            **base, cu_physics=3, cudt_minutes=0.0))
    with pytest.raises(ValueError, match="cudt_minutes=0"):
        validate_run_config(RunConfig(
            **base, bl_pbl_physics=1, sf_sfclay_physics=91,
            cu_physics=3, cudt_minutes=5.0))
    with pytest.raises(ValueError, match="Grell-family keys"):
        validate_run_config(RunConfig(
            **base, cu_physics=1, cudt_minutes=5.0, ishallow=1))


@pytest.mark.gpu
@requires_gpu
def test_the_driver_refuses_grell_freitas_with_the_pbl_slot_off():
    """The in-process door refuses the pair the config door refuses.

    ``initialize_physics`` had no cu/pbl check of any kind, so a
    hand-built RunConfig -- how the offline child, the DA drivers and
    the harnesses build a driver -- reached Grell-Freitas with the PBL
    slot off and FAILED BY READING GARBAGE at step 1 rather than by
    raising: ``fields["kpbl"]`` is allocated as zeros, only a PBL scheme
    writes it, and gf.cu indexes the column ONE-BASED, so ``t[kpbl]``
    (a divisor), ``zo[kpbl]`` and ``rho[kpbl]`` read slot 0 of an
    uninitialised workspace in both the deep and the shallow arm.

    New Tiedtke is the control: cu_physics=16 reads no KPBL at all, so
    the PBL-off configuration it was refused for (a clone of this
    refusal) is admitted here and starts.
    """
    from woof.config import RunConfig
    from woof.core.physics import initialize_physics

    base = dict(nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
                dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
                bl_pbl_physics=0, sf_sfclay_physics=91,
                cudt_minutes=0.0)
    cfg = RunConfig(**base, cu_physics=3)
    state = _state_for(cfg)
    with pytest.raises(ValueError, match="KPBL"):
        initialize_physics(state, cfg, landmask=1.0, tsk=290.0)

    ntiedtke_cfg = RunConfig(**base, cu_physics=16)
    driver = initialize_physics(
        _state_for(ntiedtke_cfg), ntiedtke_cfg, landmask=1.0, tsk=290.0)
    assert driver.cumulus_callable is not None


@pytest.mark.gpu
@requires_gpu
def test_native_gf_validation_keeps_order_ownership_and_deferral(monkeypatch):
    from woof.config import RunConfig
    from woof.core import health_ledger
    from woof.core.gf import GrellFreitas, _NativeGFCumulusResult
    from woof.core.physics import initialize_physics

    cfg = RunConfig(
        nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10, bl_pbl_physics=1,
        sf_sfclay_physics=91, cu_physics=3, cudt_minutes=0.0)
    state = _state_for(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=290.0)
    names = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rainc")
    labels = tuple(f"cumulus {name}" for name in names[:4]) + (
        "cumulus RAINC increment",)
    values = {name: cp.zeros(state.p.shape if name != "rainc"
                             else state.p.shape[1:], cp.float32)
              for name in names}
    values["rainc"][...] = cp.float32(0.125)
    result = _NativeGFCumulusResult(owner=driver.cumulus_callable, **values)
    monkeypatch.setattr(GrellFreitas, "__call__", lambda self, **kwargs: result)
    status = state.scratch((1,), "physics_validation_status").view(cp.uint32)
    status[...] = cp.uint32(0xFFFFFFFF)
    before = {name: value.copy() for name, value in values.items()}
    driver._run_cumulus({}, state, cfg)
    assert int(status[0].item()) == 0
    for name in names:
        cp.testing.assert_array_equal(values[name].view(cp.uint32),
                                      before[name].view(cp.uint32))
        values[name][...] = cp.float32(9.0)
    for name in names[:4]:
        cp.testing.assert_array_equal(driver.cu_rates[name].view(cp.uint32),
                                      before[name].view(cp.uint32))
    cp.testing.assert_array_equal(driver.rainc, before["rainc"])
    for value in values.values():
        value[...] = cp.float32(0.0)
    for name, label in zip(names, labels):
        values[name].flat[-1] = cp.inf
        with pytest.raises(FloatingPointError) as caught:
            driver._run_cumulus({}, state, cfg)
        assert str(caught.value) == label + " contains a non-finite value"
        values[name].flat[-1] = cp.float32(0.0)
    values["rainc"].flat[0] = cp.inf
    values["rthcuten"].flat[0] = cp.nan
    with pytest.raises(FloatingPointError, match="cumulus rthcuten contains"):
        driver._run_cumulus({}, state, cfg)
    ledger = health_ledger.HealthLedger()
    with health_ledger.deferring(ledger):
        driver._run_cumulus({}, state, cfg)
    assert ledger.records == 1
    with pytest.raises(FloatingPointError, match="cumulus rthcuten contains"):
        ledger.drain()


@pytest.mark.gpu
@requires_gpu
def test_native_gf_admission_refuses_custom_receipts_and_layouts():
    from types import SimpleNamespace

    from woof.core.gf import (
        GrellFreitas, _NativeGFCumulusResult, is_native_gf_result,
    )
    from woof.core.physics import CumulusResult

    owner = GrellFreitas()
    state = SimpleNamespace(p=cp.zeros((3, 2, 4), cp.float32), qi=None)
    cfg = SimpleNamespace(cu_physics=3)
    values = {name: cp.zeros(state.p.shape, cp.float32)
              for name in ("rthcuten", "rqvcuten", "rqccuten")}
    values["rainc"] = cp.zeros((2, 4), cp.float32)
    result = _NativeGFCumulusResult(owner=owner, **values)
    assert is_native_gf_result(result, owner, state, cfg)
    assert not is_native_gf_result(CumulusResult(**values), owner, state, cfg)
    assert not is_native_gf_result(result, GrellFreitas(), state, cfg)
    for name, bad in (
            ("rqicuten", cp.zeros(state.p.shape, cp.float32)),
            ("rthcuten", cp.zeros(state.p.shape, cp.float64)),
            ("rqvcuten", cp.zeros((3, 2, 8), cp.float32)[:, :, ::2]),
            ("rainc", cp.zeros((1, 8), cp.float32)),
            ("rqrcuten", cp.zeros(state.p.shape, cp.float32))):
        original = getattr(result, name)
        setattr(result, name, bad)
        assert not is_native_gf_result(result, owner, state, cfg)
        setattr(result, name, original)


@pytest.mark.gpu
@requires_gpu
def test_gf_overwrites_poisoned_output_slabs(monkeypatch):
    from woof.config import RunConfig
    from woof.core import gf
    from woof.core.physics import initialize_physics

    load = gf._gf_module
    calls = []

    class PoisonedFunction:
        def __init__(self, fn):
            self.fn = fn

        def __getattr__(self, name):
            return getattr(self.fn, name)

        def __call__(self, grid, block, args):
            # A reused allocation must not hide a missing output store.
            for array in args[3:5]:
                array.fill(cp.nan)
            args[5].fill(cp.int32(-77))
            self.fn(grid, block, args)
            for array in args[3:5]:
                assert bool(cp.isfinite(array).all())
            assert not bool((args[5] == cp.int32(-77)).any())
            calls.append(1)

    class PoisonedModule:
        def __init__(self, module):
            self.module = module

        def get_function(self, name):
            return PoisonedFunction(self.module.get_function(name))

    monkeypatch.setattr(gf, "_gf_module", lambda nz, **kw: PoisonedModule(load(nz, **kw)))
    cfg = RunConfig(
        nx=4, ny=2, nz=16, dx=12000.0, dy=12000.0, ztop=9000.0,
        dt=60.0, run_seconds=0.0, moist=True, mp_physics=10,
        bl_pbl_physics=1, sf_sfclay_physics=91,
        cu_physics=3, cudt_minutes=0.0, ishallow=1)
    state = _state_for(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=290.0)
    driver.compute(state, cfg)
    assert calls == [1]

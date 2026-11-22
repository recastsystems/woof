"""Device SPP consumers against the native WRF fixtures."""
import os
import numpy as np
import pytest
from test_spp_consumers import oracle, native_directory
import test_spp_consumers as reference

@pytest.fixture
def backend():
    cp = pytest.importorskip("cupy")
    from woof.core import mynn_pbl_gpu
    from woof.core.mynn_sfclay import mynn_surface_layer
    return "gpu", mynn_pbl_gpu, cp.asnumpy, mynn_surface_layer, cp.asarray

def test_mynn_turbulence_spp_native(oracle, backend):
    reference.test_mynn_turbulence_spp_native(oracle,backend)

def test_mynn_condensation_spp_native(oracle, backend):
    reference.test_mynn_condensation_spp_native(oracle,backend)

def test_mynn_mass_flux_spp_native(oracle, backend):
    reference.test_mynn_mass_flux_spp_native(oracle,backend)

@pytest.mark.parametrize("step",[1,2])
def test_mynn_driver_spp_native(oracle, backend, step):
    reference.test_mynn_driver_spp_native(oracle,backend,step)

def test_mynn_surface_spp_native(oracle, backend):
    reference.test_mynn_surface_spp_native(oracle,backend)


@pytest.mark.parametrize("prefix", ["turbulence", "turbulence-high"])
def test_mynn_spp_diffusivity_operator_is_native_exact(oracle, monkeypatch, prefix):
    import cupy as cp
    from woof.core.spp_kernel_sources import load_spp_module
    monkeypatch.setattr(reference.pbl, "TURBULENCE_ORACLE", oracle / (prefix + "-off.csv"))
    _, off = reference.pbl._turbulence_oracle()
    monkeypatch.setattr(reference.pbl, "TURBULENCE_ORACLE", oracle / (prefix + "-spp.csv"))
    _, on = reference.pbl._turbulence_oracle()
    dfm, dfh, pattern, height = [cp.asfortranarray(cp.asarray(x))
                               for x in (off["dfm"],off["dfh"],on["rstoch"],off["zw"])]
    load_spp_module("mynn_pbl").get_function("mynn_spp_diffusivity")((1,), (128,),
        (dfm,dfh,pattern,height,np.int32(dfm.size)))
    for actual,name in ((dfm,"dfm"),(dfh,"dfh")):
        np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),on[name].view(np.uint32))
    if prefix.endswith("high"):
        assert np.max(on["zw"]) > 20000


def test_connected_spp_consumers_step_through_rk3(monkeypatch):
    import cupy as cp
    import test_ruc_runtime as fixture
    from woof.core import physics
    from woof.core.dycore import step, stability_report
    from woof.ensemble.stochastic import StochasticTimestepHook
    original_config = fixture.RunConfig
    monkeypatch.setattr(fixture, "RunConfig", lambda **kw: original_config(
        **kw, cu_physics=3, cu_gf_version="wrf_461", spp_conv=1, spp_pbl=1, spp_lsm=1))
    state,cfg,driver = fixture._build(nx=16,ny=12,nz=40,
                                     sf_sfclay_physics=5,bl_pbl_physics=5)
    hook = StochasticTimestepHook((13,17), dx=cfg.dx,dy=cfg.dy,dt=cfg.dt,
                                 member_seed=29,spp=True,
                                 spp_levels={"conv":4,"pbl":40,"lsm":9})
    consumed = {"surface":0,"pbl":0,"lsm":0,"conv":0}
    def observe(name, original, key):
        def wrapper(*args,**kwargs):
            expected = driver.spp_patterns[key]
            if name == "surface":
                assert kwargs["pattern_spp_pbl"].data.ptr == expected[0].data.ptr
            else:
                assert kwargs["pattern_spp_"+key] is expected
            consumed[name] += 1
            return original(*args,**kwargs)
        return wrapper
    monkeypatch.setattr(physics,"launch_mynn_surface_layer",observe(
        "surface",physics.launch_mynn_surface_layer,"pbl"))
    monkeypatch.setattr(physics,"mynn_pbl_step",observe("pbl",physics.mynn_pbl_step,"pbl"))
    monkeypatch.setattr(physics,"ruc_lsm_step",observe("lsm",physics.ruc_lsm_step,"lsm"))
    gf_class = type(driver.cumulus_callable)
    original_gf = gf_class.__call__
    def gf_call(self,*args,**kwargs):
        assert driver.spp_patterns["conv"].shape == (4,12,16)
        consumed["conv"] += 1
        return original_gf(self,*args,**kwargs)
    monkeypatch.setattr(gf_class,"__call__",gf_call)
    compute = driver.compute
    def compute_and_finish(s,c):
        result = compute(s,c)
        mapped = {name:getattr(result,attr) for name,attr in
                  (("u","ru"),("v","rv"),("theta","rtheta"),("qv","rqv"))}
        out = hook.after_nonmicrophysics(mapped,tendency_scope="nonmicrophysics")
        assert all(out[name] is mapped[name] for name in mapped)
        return result
    driver.compute = compute_and_finish
    for tick in range(3):
        hook.before_timestep(tick)
        driver.bind_spp_patterns(hook.parameter_patterns)
        step(state,cfg)
        assert not stability_report(state,cfg)["nan"]
    assert all(value > 0 for value in consumed.values()), consumed
    assert bool(cp.any(driver.fields["field_sf"] != 0))
    assert bool(cp.any(driver.fields["exch_h"] != 0))
    for name in ("hfx","qfx","tsk","qke","smois","field_sf"):
        assert bool(cp.all(cp.isfinite(driver.fields[name]))), name
    print("CONNECTED_SPP " + str(consumed))


def test_disabled_spp_provider_n1_is_byte_identical_to_bare_forecast(monkeypatch):
    import cupy as cp
    import test_ruc_runtime as fixture
    from woof.core.dycore import step
    from woof.ensemble.stochastic import StochasticTimestepHook
    original_config = fixture.RunConfig
    monkeypatch.setattr(fixture,"RunConfig",lambda **kw: original_config(**kw,cu_physics=3, cu_gf_version="wrf_461"))
    plain = fixture._build(nx=8,ny=6,nz=40,sf_sfclay_physics=5,bl_pbl_physics=5)
    member = fixture._build(nx=8,ny=6,nz=40,sf_sfclay_physics=5,bl_pbl_physics=5)
    state,cfg,driver = member
    hook = StochasticTimestepHook((1,1),dx=cfg.dx,dy=cfg.dy,dt=cfg.dt,
                                 member_seed=1,enabled=False,spp=True)
    assert hook.spp == {} and hook.parameter_patterns == {}
    assert "field_sf" not in driver.fields
    for tick in range(3):
        hook.before_timestep(tick)
        driver.bind_spp_patterns(hook.parameter_patterns)
        step(*plain[:2])
        step(state,cfg)
    compared = 0
    for name in ("u","v","w","thp","mup","php","qv","qc","qr","qi","qs"):
        got,want = getattr(state,name),getattr(plain[0],name)
        if got is None:
            assert want is None
            continue
        cp.testing.assert_array_equal(got.view(cp.uint32),want.view(cp.uint32),err_msg=name)
        compared += got.size
    assert set(driver.fields) == set(plain[2].fields)
    for name,got in driver.fields.items():
        want = plain[2].fields[name]
        if isinstance(got,cp.ndarray):
            # reshape(-1): a strict process also carries 0-d SCM forcing scalars.
            cp.testing.assert_array_equal(got.reshape(-1).view(cp.uint8),want.reshape(-1).view(cp.uint8),err_msg=name)
            compared += got.nbytes // 4
    print("DISABLED_SPP_N1_IDENTICAL_WORDS " + str(compared))


def test_spp_compiled_sources_have_no_constant_division_rewrites():
    from tools.literal_division_census import production_units, census_unit, rewrite_sites
    selected = [unit for unit in production_units(reference.DATA.parents[2])
                if unit.key.startswith("spp:") or unit.key == "kernels:ruc_spp"]
    assert len(selected) == 5
    for unit in selected:
        assert census_unit(unit)["constant_divisor_sites"] == [], unit.key
        assert rewrite_sites(unit)["rewritten_sites"] == [], unit.key

@pytest.mark.gpu
def test_gf_spp_native_and_zero_pattern_identity(oracle, monkeypatch):
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU explicitly disabled")
    import test_gf_gfdrv_cuda as gf
    import cupy as cp
    from woof.core.spp_kernel_sources import load_spp_module
    from woof.verify.gf_oracle import _read_csv, _f32_columns, _fold_levels
    fixture = gf.load_gf_oracle()
    import woof.verify.gf_oracle as native_gf
    monkeypatch.setattr(native_gf, "GF_ORACLE_DIR", oracle)
    baseline = gf._launch(gf.load_module("gf"), fixture, 1)
    module = load_spp_module("gf")
    n, nz = fixture.ncol, gf.NZ
    def launch(pattern):
        lvin = np.stack([fixture.levels[name] for name in gf.DRV_IN_LEV], axis=1)
        scin = np.concatenate([gf.drv_scalar_inputs(fixture, True),
                               np.broadcast_to(np.asarray(pattern, dtype=np.float32), (n, 4))], axis=1)
        iin = np.stack([fixture.surface[name] for name in ("kpbl", "ishallow", "ichoice")], axis=1).astype(np.int32)
        lev = cp.zeros((n, len(gf.DRV_LEV_FIELDS), nz), dtype=cp.float32)
        sca = cp.zeros((n, len(gf.DRV_SCA_FIELDS)), dtype=cp.float32)
        isc = cp.zeros((n, len(gf.DRV_ISCA_FIELDS)), dtype=cp.int32)
        module.get_function("gf_gfdrv_stage")(((n+63)//64,), (64,), (
            cp.asarray(lvin), cp.asarray(scin), cp.asarray(iin), lev, sca, isc,
            cp.empty(gf.gf_workspace_floats(nz,n), dtype=cp.float32),
            np.int32(1), np.int32(n), np.int32(nz)))
        return dict(lev=cp.asnumpy(lev), sca=cp.asnumpy(sca), isc=cp.asnumpy(isc))
    zero = launch([0, 0, 0, 0])
    for key in baseline:
        np.testing.assert_array_equal(zero[key].view(np.uint32), baseline[key].view(np.uint32), err_msg=key)
    perturbed = launch([.2, -.3, .4, -.5])
    levels = _fold_levels(_f32_columns(*_read_csv(oracle / "gf-spp-levels.csv")), fixture.key, nz)
    scalars = _f32_columns(*_read_csv(oracle / "gf-spp-surface.csv"))
    order_by_key = {(int(c),int(i),int(a)): j for j,(c,i,a) in enumerate(zip(scalars["case"],scalars["idx"],scalars["arm"]))}
    order = [order_by_key[tuple(map(int,key))] for key in fixture.key]
    for name in gf._SEAM_LEV:
        gf._assert_bit_exact(gf._lev(perturbed,name), levels[name], name)
    for name in ("pret", "prets"):
        gf._assert_bit_exact(gf._sca(perturbed,name), scalars[name][order], name)
    assert np.any(perturbed["lev"] != baseline["lev"])

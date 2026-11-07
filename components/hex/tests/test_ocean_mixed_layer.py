"""Public coupling policy and restart configuration checks."""
import json
import pytest
from woof.hex.ocean_mixed_layer import configuration
from woof.hex import s2s_door
from woof.hex.cli import build_parser


def args(path, *extra):
    return build_parser().parse_args(["s2s", "--row", "global-120km", "--source", "gfs",
        "--cycle", "2026-10-20T00", "--members", "1", "--days", "30", "--work-dir", str(path), *extra])


def test_default_prognostic_needs_no_future_analyses(tmp_path):
    options = args(tmp_path)
    value = s2s_door.plan(options)
    assert value["surface"]["ocean_mode"] == "prognostic"
    result = s2s_door.surface_stage(value, options, {})
    assert result["future_surface_records"] == 0
    env = s2s_door.surface_environment(value)
    assert env["WOOF_HEX_OCEAN_MODE"] == "prognostic"
    assert json.loads(env["WOOF_HEX_OCEAN_OPTIONS"])["depth_m"] == 50
    assert not tmp_path.joinpath("surface_series.nc").exists()


def test_coupling_parameters_are_member_restart_identity(tmp_path):
    base = s2s_door.plan(args(tmp_path))
    changed = s2s_door.plan(args(tmp_path, "--ocean-options", '{"depth_m":25}'))
    prescribed = s2s_door.plan(args(tmp_path, "--ocean-mode", "prescribed"))
    assert len({p["signature"] for p in (base, changed, prescribed)}) == 3
    assert changed["surface"]["ocean_options"]["depth_m"] == 25


@pytest.mark.parametrize("bad", [{"depth_m":0}, {"depth_m":300}, {"salinity_psu":float("nan")},
    {"depth_m":True}, {"made_up":1}, {"ice_albedo":1.1}, []])
def test_bad_parameters_refuse_before_launch(bad):
    with pytest.raises(ValueError):
        configuration(bad)


@pytest.mark.parametrize("numeric_mode", ["fast", "strict"])
def test_native_global_energy_freezing_melting_and_inertial_stability(monkeypatch, numeric_mode):
    """Exercise the compiled kernel at the real 120 km global cell count."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device required")
    import numpy as np
    from woof.hex.ocean_mixed_layer import _kernel
    from woof.hex import numerics
    monkeypatch.setattr(numerics, "_mode", numeric_mode)
    _kernel.cache_clear()
    n = 40962
    p = configuration({})
    tf = 273.15-.054*p["salinity_psu"]
    a = lambda value: cp.full(n, value, cp.float32)
    d = lambda value: cp.full(n, value, cp.float64)
    # Thirds represent warm ocean, cold ice, melting ice. Last cell is land.
    T=d(tf); T[:n//3]=290
    H=d(50); U=d(0); W=d(0); V=d(.5); V[:n//3]=0
    A=d(.5); A[:n//3]=0
    Ti=d(260); Ti[2*n//3:]=273.15
    mask=a(1); mask[-1]=0
    snow=d(0); E=cp.zeros(n,cp.float64); entr=E.copy(); residual=E.copy()
    melt=a(0); basal=a(0); net=a(0)
    sw=a(0); sw[:n//3]=500; sw[2*n//3:]=1200
    lw=a(150); lw[:n//3]=400; lw[2*n//3:]=500
    initial_E=p["water_density"]*p["water_heat_capacity"]*H.astype(cp.float64)*(T.astype(cp.float64)-tf)-p["ice_density"]*p["latent_heat"]*V
    before_land=(float(T[-1]),float(V[-1]),float(A[-1]))
    keys=("water_density","water_heat_capacity","ice_density","latent_heat","ice_conductivity",
          "snow_conductivity","salinity_psu","stratification_k_m","max_depth_m","new_ice_thickness_m",
          "basal_exchange","water_albedo","ice_albedo","snow_albedo","emissivity","snow_density")
    params=cp.asarray([p[k] for k in keys],cp.float64)
    for _ in range(30*144):
        _kernel(cp.cuda.Device().id)(((n+127)//128,),(128,),
            (np.int32(n),np.float32(600),np.float32(.02),mask,T,H,U,W,d(tf),d(50),V,A,Ti,snow,
             E,entr,residual,melt,basal,net,a(1e-4),sw,lw,a(0),a(0),a(0),a(0),a(0),a(0),a(0),
             a(270),a(100000),a(.01),a(.01),a(0),a(0),params))
    cp.cuda.get_current_stream().synchronize()
    assert bool(cp.isfinite(T).all()) and bool(cp.isfinite(V).all())
    assert float(T[0]) > 290
    assert float(V[n//2]) > .5
    assert float(V[-2]) < .5
    assert before_land == (float(T[-1]),float(V[-1]),float(A[-1]))
    assert bool(((A>=0)&(A<=1)&(V>=0)&(H>=50)&(H<=200)).all())
    final_E=p["water_density"]*p["water_heat_capacity"]*H.astype(cp.float64)*(T.astype(cp.float64)-tf)-p["ice_density"]*p["latent_heat"]*V
    budget=final_E-initial_E-E-entr
    # Month-long native-cell regression: carried state must close the energy budget.
    assert float(cp.max(cp.abs((budget-residual)[:-1]))) < .01
    assert float(cp.max(cp.abs(residual[:-1]))) < .01


def test_restart_construction_preserves_carried_atmospheric_surface_words(monkeypatch):
    import sys
    import numpy as np
    from types import SimpleNamespace
    from woof.hex.ocean_mixed_layer import MixedLayerOcean
    monkeypatch.setitem(sys.modules,"cupy",np)
    monkeypatch.setattr("woof.hex.ocean_mixed_layer._kernel_for_device", lambda: None)
    fields={"tsk":np.array([[250,281,298]],dtype="f4"),
            "xice":np.array([[.8,0,0]],dtype="f4"),"snow":np.array([[15,0,0]],dtype="f4")}
    before={k:v.copy() for k,v in fields.items()}
    service=SimpleNamespace(n=3,noahmp=object(),
        backend=SimpleNamespace(_seam=SimpleNamespace(_driver=SimpleNamespace(fields=fields))),
        owned={"landmask":np.array([0,0,1]),"sst":np.array([271.5,281,298],dtype="f4"),
               "sstsk":np.array([271.5,281,298],dtype="f4")},
        _read_initial=lambda name:np.array([272,285,300] if name=="sst" else [1,0,-1],dtype="f4"))
    MixedLayerOcean(service,restoring=True)
    for key,value in before.items():
        np.testing.assert_array_equal(fields[key].view("u4"),value.view("u4"))
    np.testing.assert_array_equal(service.owned["sst"],np.array([271.5,281,298],dtype="f4"))

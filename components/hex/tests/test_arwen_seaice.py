"""Real GPU checks for the Noah-MP sea-ice energy owner."""
from datetime import datetime
from types import SimpleNamespace
import os
import numpy as np
import pytest

@pytest.fixture
def cp():
    if os.environ.get("S2S_SURFACE_GPU") != "1":
        pytest.skip("requires GPU mutex and S2S_SURFACE_GPU=1")
    import cupy
    return cupy

def make(threshold=.02):
 from woof.core.mpas_column_batch import MpasColumnBatchPhysics
 seam=MpasColumnBatchPhysics(n_levels=3,n_columns=4,dt=60.,radiation_seconds=60.,surface_pbl_seconds=120.,microphysics_scheme='wsm6',start_time=datetime(2026,10,17),latitude_deg=[80,80,0,45],longitude_deg=[0,90,0,0],terrain_height_m=[0]*4,z_interface_nominal_m=np.array([0,100,300,1000],dtype='f4'),p_top_pa=90000,dx_m=120000,landmask=[0,0,0,1],xland=[1,1,2,1],ivgtyp=[15,15,17,10],isltyp=[16,16,14,6],tsk=[255,263,290,280],tmn=[271.4,271.4,290,280],xice=[1,.5,0,0],snow=[10,10,0,0],snow_depth=[.05,.05,0,0],soil_temperature=260.,soil_moisture=.3,xice_threshold=threshold)
 return seam

def atmosphere(cp):
    full=lambda x:cp.full((3,1,4),x,dtype="f4")
    return dict(u=full(5),v=full(2),temperature=full(250),theta=full(250),
        qv=full(.001),qc=full(0),pressure=full(100000),dz=full(100),
        p_interface=cp.full((4,1,4),101000,dtype="f4"))

def advance(seam, a):
    f=seam._driver.fields
    f["swdown"].fill(0);f["glw"].fill(180);f["psfc"].fill(101000)
    seam._driver._run_sfclay(a,seam._cfg)
    seam._driver._run_noahmp(a,seam._cfg,1)


def test_energy_changes_ice_and_preserves_nonice_words(cp):
    from woof.hex.arwen_seaice import install_noahmp_seaice
    base=make();fixed=make();install_noahmp_seaice(SimpleNamespace(),fixed)
    a=atmosphere(cp);advance(base,a);advance(fixed,a)
    mask=cp.asarray([[False,False,True,True]])
    for key,b in base._driver.fields.items():
        if b.ndim<2 or b.shape[-2:]!=(1,4):continue
        # gpuwm 2.9.4's own sea-ice state exists only on the seam without the hex owner (retire_engine_seaice).
        if key not in fixed._driver.fields:continue
        index=cp.broadcast_to(mask,b.shape)
        assert bool(cp.array_equal(b[index].view(cp.uint8),fixed._driver.fields[key][index].view(cp.uint8))),key
    assert float(fixed._driver.fields["tsk"][0,0]) != float(base._driver.fields["tsk"][0,0])
    assert float(fixed._driver.fields["grdflx"][0,0]) != 0


@pytest.mark.parametrize("threshold", [.02,.5])
def test_series_sst_threshold_and_dt_pbl_ownership(cp,monkeypatch,threshold):
    from woof.hex.arwen_seaice import install_noahmp_seaice
    from woof.hex.mpasphys import seaice_v842 as si
    seam=make(threshold);service=SimpleNamespace(owned={"sstsk":cp.full(4,278.,cp.float32)})
    install_noahmp_seaice(SimpleNamespace(_surface_updates=service),seam)
    calls=[];original=si.driver_seaice
    def run(diag,sfc,shared,options,**kw):
        calls.append((options.dt,options.xice_threshold))
        assert bool(cp.all(kw["sea_temperature"]==278))
        return original(diag,sfc,shared,options,**kw)
    monkeypatch.setattr(si,"driver_seaice",run)
    advance(seam,atmosphere(cp))
    assert calls==[(120.,threshold)]


def test_component_state_is_in_actual_restart_manifest(cp):
    from woof.hex.arwen_seaice import install_noahmp_seaice
    from woof.io.restart import _driver_manifest
    seam=make();install_noahmp_seaice(SimpleNamespace(),seam)
    advance(seam,atmosphere(cp));f=seam._driver.fields
    manifest=_driver_manifest(seam._driver)
    for key in ("sst","tice","hfx","qsfc","znt"):
        assert "fields/hex_seaice_"+key in manifest
    snap=seam.export_state();saved=f["hex_seaice_hfx"].copy()
    f["hex_seaice_hfx"].fill(999);seam.restore_state(snap)
    assert bool(cp.array_equal(saved.view(cp.uint8),f["hex_seaice_hfx"].view(cp.uint8)))

    # Restore into a fresh seam, then advance both real drivers identically.
    restored=make();install_noahmp_seaice(SimpleNamespace(),restored)
    restored.restore_state(snap)
    for other in (seam,restored):
        other._driver._run_sfclay(atmosphere(cp),other._cfg)
        other._driver._run_noahmp(atmosphere(cp),other._cfg,2)
    for key,value in seam._driver.fields.items():
        assert bool(cp.array_equal(value.view(cp.uint8),restored._driver.fields[key].view(cp.uint8))),key


def test_engine_prognostic_seaice_retires_and_hex_owns_the_census(cp):
    # Breakage: gpuwm 2.9.4's default-on prognostic sea ice changed XICE after the land call, so the
    # next Noah-MP census disagreed with the hex owner's classification and every s2s run on 0.4.2 stopped
    # at its first composite step.
    from woof.core.noahmp_runtime import classify_noahmp_surface
    from woof.hex.arwen_seaice import install_noahmp_seaice
    seam = make()
    f = seam._driver.fields
    if "ice_volume" not in f:
        pytest.skip("engine without prognostic sea ice (before woof 2.9.4)")
    xice0 = f["xice"].copy()
    install_noahmp_seaice(SimpleNamespace(), seam)
    assert "ice_volume" not in f and "ice_ocean_mask" not in f
    assert seam._cfg.seaice_prognostic is False
    assert bool(cp.array_equal(f["sst"][:, :2], f["tsk"][:, :2]))
    a = atmosphere(cp)
    for _ in range(3):
        advance(seam, a)
    assert bool(cp.array_equal(f["xice"], xice0))
    masks = classify_noahmp_surface(f["xland"], f["xice"], f["ivgtyp"], xice_threshold=.02, isice=15)
    census = getattr(seam, "last_noahmp_census", None) or seam._driver.last_noahmp_census
    assert census["sea_ice"] == int(masks["sea_ice"].sum())
    assert census["glacier"] == int(masks["glacier"].sum())

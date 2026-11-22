"""Physical and lifecycle checks for the independently written snow solver."""
import numpy as np

def test_packaged_optics_provenance_matches_runtime():
    import hashlib,json
    from pathlib import Path
    from woof.core.snow_papers import TABLE_SHA256,TABLE_WORDS,STATE_WORDS,identity
    data=Path(__file__).resolve().parents[1]/'woof/data/snow_papers'
    receipt=json.loads((data/'receipt.json').read_text())
    pin=json.loads((data/'source-pin.json').read_text())
    assert receipt['algorithm']==identity()['algorithm']
    assert receipt['table_words']==TABLE_WORDS and receipt['state_words']==STATE_WORDS
    assert receipt['table_sha256']==pin['generated_sha256']==TABLE_SHA256
    assert hashlib.sha256((data/'optics.bin').read_bytes()).hexdigest()==TABLE_SHA256
import pytest
from conftest import requires_gpu

@pytest.mark.parametrize("surface,albedo", [(0, 2), (0, 3), (4, 2)])
def test_unselected_snow_controls_preserve_checkpoint_identity(surface, albedo):
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000, dy=3000, ztop=12000,
                    dt=10, run_seconds=60, sf_surface_physics=surface,
                    opt_alb=albedo)
    old = configuration_echo(cfg)
    assert not any(key.startswith("snicar_") for key in old)
    # Even an explicit unused control cannot change this trajectory.
    changed = replace(cfg, snicar_bc_ppb=100)
    assert configuration_echo(changed) == old
    _require_config_match(old, changed, "older-checkpoint.npz")


@pytest.mark.parametrize("name,value", [
    ("snicar_snowshape_opt", 1), ("snicar_snowbc_intmix", False),
    ("snicar_snowdust_intmix", True), ("snicar_bc_ppb", 100.0),
    ("snicar_dust_ppm", 10.0), ("snicar_ice_density", 850.0),
    ("snicar_bubble_radius_um", 500.0), ("snicar_fresh_grain_opt", 2),
])
def test_selected_snow_controls_bind_checkpoint_identity(name, value):
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000, dy=3000, ztop=12000,
                    dt=10, run_seconds=60, sf_surface_physics=4, opt_alb=3)
    old = configuration_echo(cfg)
    assert name in old
    with pytest.raises(ValueError, match=name):
        _require_config_match(old, replace(cfg, **{name: value}), "snow.npz")

@requires_gpu
def test_snow_grain_impurity_and_depth_response():
    import cupy as cp
    from woof.core.snow_papers import evaluate
    n=7
    x=cp.zeros((n,56),dtype=cp.float32)
    x[:,24]=60; x[:,25]=.7; x[:,28]=263.15; x[:,30]=.4; x[:,31]=1
    x[:,34]=100; x[:,42]=-55
    x[:,5:9]=.1
    x[:,52:56]=100
    x[1,42]=-1000
    x[3,34]=.1
    x[4,25]=.1
    x[5,25]=-1
    x[6,34]=0
    carbon=cp.zeros(n,dtype=cp.float32);carbon[2]=1e-6
    y=cp.asnumpy(evaluate(x,cp.ones(n,dtype=cp.int32),carbon=carbon))
    assert np.isfinite(y).all()
    assert ((y[:,:4]>=0)&(y[:,:4]<=1)).all()
    assert y[0,0]>.9 and .3<y[0,1]<.9
    assert y[1,1]<y[0,1] # larger grains absorb more NIR
    assert y[2,0]<y[0,0] # carbon darkens visible snow
    assert y[3,0]<y[0,0] # thin snow reveals dark ground
    assert y[4,0]>y[0,0] # grazing beam reflects more
    assert np.array_equal(y[5,:4],np.zeros(4))
    assert y[5,4]<-55 # night grain evolution is carried
    assert y[6,4]==0
    print('snow responses',y.tolist())

@requires_gpu
def test_snow_papers_restart_trajectory(tmp_path):
    import cupy as cp
    from test_noahmp_runtime import _build,_CARRIED
    from woof.core.dycore import step
    from woof.io.restart import write_restart,restore_restart
    options=dict(nx=6,ny=4,snow_mm=40,snow_depth_m=.2,opt_alb=3)
    a,cfg,da=_build(**options)
    for _ in range(3):step(a,cfg)
    assert float(cp.min(da.fields['taussxy']).item())<0
    path=tmp_path/'snow.npz';write_restart(path,a,cfg)
    b,cfgb,db=_build(**options);restore_restart(path,b,cfgb)
    step(a,cfg);step(b,cfgb)
    for key in (*_CARRIED, "snicar_statexy"):
        np.testing.assert_array_equal(cp.asnumpy(da.fields[key]),cp.asnumpy(db.fields[key]),err_msg=key)
    assert da.noahmp_params.restart_identity()['snow_papers']['status']=='written from papers, not bitwise to WRF'


def test_stock_namelist_preserves_papers_option(tmp_path):
    from test_namelist_import import _import_with
    text,report=_import_with(tmp_path,extra_input="&noah_mp opt_alb=3 /\n")
    assert 'opt_alb = 3' in text


def test_snow_memory_prices_are_conditional():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.core.preflight import noahmp_lsm_transient_shapes, noahmp_lsm_cache_shapes
    from woof.core.noahmp_selected_frames import selected_frames
    cfg = RunConfig(nx=73, ny=60, nz=34, dx=30000, dy=30000,
                    ztop=18000, dt=60, run_seconds=10800,
                    sf_surface_physics=4, opt_alb=3)
    assert noahmp_lsm_transient_shapes(cfg)["noahmp_lsm/snow_papers_transients"] == (4380, 1024)
    assert noahmp_lsm_cache_shapes(cfg)["noahmp_lsm/snow_papers_optics"] == (283703, 4)
    assert selected_frames(snow_papers=True)[0]["snow_papers"] > 0
    for option in (1, 2):
        other = replace(cfg, opt_alb=option)
        assert not any("snow_papers" in key for key in noahmp_lsm_transient_shapes(other))
        assert not any("snow_papers" in key for key in noahmp_lsm_cache_shapes(other))


@requires_gpu
def test_rust_rejects_changed_asset_without_touching_output(tmp_path):
    import ctypes
    from pathlib import Path
    from woof.core.snow_papers import _host_table, TABLE_WORDS, TABLE_SHA256
    from woof.static.rust_bridge import load
    assert _host_table().shape == (TABLE_WORDS,)
    reader = load().gpuwm_snow_optics_load_v1
    source = Path(__file__).resolve().parents[1]/"woof/data/snow_papers/optics.bin"
    bad = bytearray(source.read_bytes()); bad[-1] ^= 1
    changed = tmp_path/"changed.bin"; changed.write_bytes(bad)
    output = np.full(TABLE_WORDS, -123, dtype=np.float32)
    assert reader(str(changed).encode(), TABLE_SHA256.encode(),
                  output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output.size) != 0
    np.testing.assert_array_equal(output, np.full_like(output, -123))

@requires_gpu
@pytest.mark.parametrize('feedback',[0,1])
def test_nested_snow_checkpoint(feedback,tmp_path):
    from test_noahmp_nested_lifecycle import test_live_selected_nested_checkpoint_continuation
    test_live_selected_nested_checkpoint_continuation(feedback,dict(opt_alb=3,snow_mm=40,snow_depth_m=.2),tmp_path)

@requires_gpu
def test_selected_glacier_snow_captures(monkeypatch):
    from test_noahmp_glacier_slab import test_selected_glacier_step_captures_without_host_transfers
    test_selected_glacier_step_captures_without_host_transfers(dict(opt_alb=3),monkeypatch)

@requires_gpu
def test_layered_energy_habits_mixing_and_ice():
    import cupy as cp
    from woof.core.snow_papers import evaluate, refreeze
    n=4
    x=cp.zeros((n,56),dtype=cp.float32)
    x[:,24]=60; x[:,25]=.6; x[:,28]=263.15; x[:,30]=.4
    x[:,31]=1; x[:,34]=100; x[:,42]=-100; x[:,5:9]=.15; x[:,52:56]=100
    layers=cp.zeros((n,3,4),dtype=cp.float32)
    layers[:,:,0]=cp.asarray([10,30,60],dtype=cp.float32)
    layers[:,:,2]=cp.asarray([.05,.12,.23],dtype=cp.float32)
    layers[:,:,3]=263.15
    values=[]
    for shape in range(1,5):
        y,s,a=evaluate(x,cp.ones(n,dtype=cp.int32),layers=layers,
            options=dict(snicar_snowshape_opt=shape),return_fluxes=True)
        np.testing.assert_allclose(cp.asnumpy(cp.sum(a,axis=1)+y[:,:4]),1,atol=3e-6)
        assert bool(cp.all(a>=0))
        values.append(cp.asnumpy(y[0,:4]))
    assert len({tuple(v) for v in values})==4
    clean=values[2]
    for species,concentration in [('bc',100.),('dust',10.)]:
        for internal in (False,True):
            options=dict(snicar_snowbc_intmix=False,snicar_snowdust_intmix=False)
            options['snicar_'+species+('_ppb' if species=='bc' else '_ppm')]=concentration
            options['snicar_snow'+species+'_intmix']=internal
            y,s,a=evaluate(x,cp.ones(n,dtype=cp.int32),layers=layers,options=options,return_fluxes=True)
            assert float(y[0,0])<clean[0]
            np.testing.assert_allclose(cp.asnumpy(cp.sum(a,axis=1)+y[:,:4]),1,atol=3e-6)
    ice=cp.empty((n,4,3),dtype=cp.float32)
    ice[:,:,0]=cp.asarray([.1,.3,.6,1],dtype=cp.float32)
    ice[:,:,1]=cp.asarray([700,800,900,916],dtype=cp.float32)[:,None]
    ice[:,:,2]=cp.asarray([100,500,1000,10000],dtype=cp.float32)[:,None]
    x[:,34]=0; layers[:]=0
    y,s,a=evaluate(x,cp.ones(n,dtype=cp.int32),layers=layers,glacier=cp.ones(n),ice_layers=ice,return_fluxes=True)
    np.testing.assert_allclose(cp.asnumpy(cp.sum(a,axis=1)+y[:,:4]),1,atol=3e-6)
    assert bool(cp.all(cp.isfinite(y))) and bool(cp.any(a[:,4:,:]>0))
    state=cp.zeros((n,21),dtype=cp.float32); state.reshape(n,3,7)[:,:,1]=100
    before=cp.full((n,3),10,dtype=cp.float32);after=cp.full((n,3),12,dtype=cp.float32)
    refreeze(state,before,after,cp.zeros_like(before))
    np.testing.assert_allclose(cp.asnumpy(state.reshape(n,3,7)[:,:,1]),250,atol=1e-5)
    print('four-habit albedo',np.asarray(values).tolist(),'ice albedo',cp.asnumpy(y[:,:4]).tolist())

@requires_gpu
def test_ad_conservative_and_zero_depth_limits():
    import cupy as cp
    from woof.core.snow_papers import _module
    x=cp.asarray([[0.,.1,.8,.6],[10.,0.,.8,.6],[10000.,.05,.8,.6],[1.,.5,0.,(1/1.5)**.5]],dtype=cp.float64)
    y=cp.empty((4,10),dtype=cp.float64)
    _module(cp.cuda.Device().id).get_function('snow_ad_probe')((1,),(32,),(x,y,np.int32(4)))
    a=cp.asnumpy(y)
    np.testing.assert_allclose(a[0,:4],[0,1,0,1],atol=1e-12)
    np.testing.assert_allclose(a[1,:2].sum(),1,atol=1e-12)
    np.testing.assert_allclose(a[1,2:4].sum(),1,atol=1e-12)
    assert np.isfinite(a).all() and (a>=0).all() and (a<=1+1e-12).all()
    assert a[2,1]<1e-12


# ---- clean-snicar-aging (2026-11-09): re-implemented from published
# descriptions, not bitwise to upstream. See docs/PROVENANCE-clean-snicar-aging.md.

def _aging_column(cp, n, *, depth, mass, temperature, dt=3600.):
    x = cp.zeros((n, 56), dtype=cp.float32)
    x[:, 24] = dt; x[:, 25] = .6; x[:, 28] = temperature; x[:, 30] = depth
    x[:, 31] = 1; x[:, 34] = mass; x[:, 5:9] = .15; x[:, 52:56] = 100
    return x


@requires_gpu
@pytest.mark.parametrize("option,expected", [
    # Linear ramp between -30 and 0 C (CLM5 tech note, Snow Aging):
    # r = rmin + (rmax-rmin)*clip((T-243.15)/30, 0, 1).
    (1, [54.526, 54.526 + 150 * .5, 54.526 + 150 * 25 / 30, 204.526]),
    # Lin 2024 RAL talk / Lin et al. 2025: optimized 33.0 / 91.0 um.
    (2, [33.0, 33.0 + 58 * .5, 33.0 + 58 * 25 / 30, 91.0]),
])
def test_fresh_grain_ramp_hand_values(option, expected):
    import cupy as cp
    from woof.core.snow_papers import evaluate
    n = 4
    # Shallow snow (< 2.5 cm) keeps the fresh radius without aging.
    x = _aging_column(cp, n, depth=.01, mass=5, temperature=263.15)
    air = cp.asarray([233.15, 258.15, 268.15, 278.15], dtype=cp.float32)
    _, carried = evaluate(x, cp.ones(n, dtype=cp.int32), air_temperature=air,
                          options=dict(snicar_fresh_grain_opt=option), return_state=True)
    radius = cp.asnumpy(carried.reshape(n, 3, 7)[:, 2, 1])
    np.testing.assert_allclose(radius, expected, rtol=2e-6)


@requires_gpu
def test_wet_growth_follows_clm5_volume_rate():
    """CLM5 Eq. 2.3.75: dr/dt = 1e18*C1*f^3/(4*pi*r^2), C1 = 4.22e-13 m3/s.

    Equal total mass, depth and temperature: the only difference is the
    liquid fraction f = 0.1, so r_wet^3 - r_dry^3 = 3/(4*pi)*4.22e5*f^3*dt.
    Hand value for dt = 3600 s: 0.238732*4.22e5*1e-3*3600 = 362682.3 um^3.
    """
    import cupy as cp
    from woof.core.snow_papers import evaluate
    n = 2
    x = _aging_column(cp, n, depth=.2, mass=50, temperature=273.15)
    layers = cp.zeros((n, 3, 4), dtype=cp.float32)
    layers[:, 2, 0] = cp.asarray([50, 45], dtype=cp.float32)
    layers[:, 2, 1] = cp.asarray([0, 5], dtype=cp.float32)
    layers[:, 2, 2] = .2; layers[:, :, 3] = 273.15
    state = cp.zeros((n, 21), dtype=cp.float32)
    state.reshape(n, 3, 7)[:, 2, 0] = 50; state.reshape(n, 3, 7)[:, 2, 1] = 300
    _, carried = evaluate(x, cp.ones(n, dtype=cp.int32), layers=layers, state=state,
                          air_temperature=273.15, return_state=True)
    r = cp.asnumpy(carried.reshape(n, 3, 7)[:, 2, 1]).astype(np.float64)
    hand = 3 / (4 * np.pi) * 4.22e5 * .1 ** 3 * 3600
    assert abs(hand - 362682.3) < .1
    np.testing.assert_allclose(r[1] ** 3 - r[0] ** 3, hand, rtol=2e-3)


def _reference_dry(table, r, T, grad, density, dt, below):
    """NumPy replay of the regenerated Flanner-Zender 2006 SSA-decay fit:
    r^kappa = r0^kappa * (1 + t/tau), advanced by dt (trilinear tau, kappa)."""
    NR, NW = 32, 480
    OP = 1504; FR = OP + (5 * NR + 2) * NW * 3; MX = FR + NW * 4; AG = MX + 32 * NW * 3
    a = np.asarray(table[AG:], dtype=np.float64)
    def bracket(axis, q):
        i = 0
        while i < len(axis) - 2 and q > axis[i + 1]:
            i += 1
        return i, min(1., max(0., (q - axis[i]) / (axis[i + 1] - axis[i])))
    it, ft = bracket(a[0:8], T); ig, fg = bracket(a[8:15], grad); idn, fd = bracket(a[15:23], density)
    tau = kappa = 0.
    for i in (0, 1):
        for j in (0, 1):
            for l in (0, 1):
                ix = ((it + i) * 7 + ig + j) * 8 + idn + l
                w = (ft if i else 1 - ft) * (fg if j else 1 - fg) * (fd if l else 1 - fd)
                tau += w * a[23 + 2 * ix]; kappa += w * a[24 + 2 * ix]
    r0 = 3e6 / (917 * 60)
    x = max(30 / r0, r / r0) if below else max(1., r / r0)
    return r0 * (x ** kappa + dt / (3600 * tau)) ** (1 / kappa)


@requires_gpu
def test_retuned_fresh_grains_age_from_their_own_size():
    import cupy as cp
    from woof.core.snow_papers import evaluate, _host_table
    n = 1
    x = _aging_column(cp, n, depth=.2, mass=30, temperature=253.15, dt=600.)
    layers = cp.zeros((n, 3, 4), dtype=cp.float32)
    layers[:, 2, 0] = 30; layers[:, 2, 2] = .2; layers[:, :, 3] = 253.15
    state = cp.zeros((n, 21), dtype=cp.float32)
    state.reshape(n, 3, 7)[:, 2, 0] = 30; state.reshape(n, 3, 7)[:, 2, 1] = 33
    out = {}
    for option in (1, 2):
        _, carried = evaluate(x, cp.ones(n, dtype=cp.int32), layers=layers, state=state,
                              air_temperature=253.15, return_state=True,
                              options=dict(snicar_fresh_grain_opt=option))
        out[option] = float(carried.reshape(n, 3, 7)[0, 2, 1])
    table = _host_table()
    # density 150 kg/m3, zero gradient (no neighbour layer, ground at T).
    for option, below in ((1, False), (2, True)):
        np.testing.assert_allclose(out[option], _reference_dry(table, 33., 253.15, 0., 150., 600., below), rtol=2e-5)
    assert out[1] >= 3e6 / (917 * 60)       # option 1: fit start clamp
    assert 33 < out[2] < 3e6 / (917 * 60)   # option 2: grows from 33 um
    print("aged radius after 600 s from 33 um:", out)

# --- Ice refractive-index and dust optical sets.
# Re-implemented from published descriptions, not bitwise to upstream.

def test_optical_set_assets_match_their_pins():
    import hashlib,json
    from pathlib import Path
    from woof.core.snow_papers import ICE_TABLES,DUST_ROWS_FILE,DUST_ROWS_SHA256,TABLE_SHA256
    data=Path(__file__).resolve().parents[1]/'woof/data/snow_papers'
    pin=json.loads((data/'source-pin.json').read_text())['optical_sets']
    assert ICE_TABLES[2]==('optics.bin',TABLE_SHA256)
    for name,digest in [*ICE_TABLES.values(),(DUST_ROWS_FILE,DUST_ROWS_SHA256)]:
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==digest
        if name!='optics.bin':
            assert pin['generated_sha256'][name]==digest


def _bridge_or_skip():
    from woof.static.rust_bridge import load
    try:
        lib=load()
        lib.gpuwm_snow_dust_rows_load_v1
    except Exception as exc:  # pragma: no cover - environment without rebuilt bridge
        pytest.skip(f"rebuilt Rust static-fields bridge required: {exc}")


def test_picard_table_hand_value_and_default_dust_row():
    _bridge_or_skip()
    from woof.core.snow_papers import _host_table,_host_dust_rows,DUST_ROW_START
    fr=1504+(5*32+2)*480*3
    # Picard et al. 2016 Supplement section 2: 445 nm, 0.01867 m-1.
    # By hand k = 0.01867 * 445e-9 / (4 pi) = 8.30815e-9 / 12.56637 = 6.61142e-10.
    iw=24  # bin centre 0.205 + 24*0.01 = 0.445 um
    assert abs(float(_host_table(3)[fr+iw*4+1])-6.61142e-10)<2e-15
    # Warren and Brandt 2008 at 445 nm is far less absorbing (Picard's point).
    assert float(_host_table(2)[fr+iw*4+1])<0.2*6.61142e-10
    # The real part keeps the Warren and Brandt value.
    assert float(_host_table(3)[fr+iw*4])==float(_host_table(2)[fr+iw*4])
    # Dust set 0 is bit-for-bit the in-table row of every ice table.
    rows=_host_dust_rows()
    for ice in (1,2,3):
        assert np.array_equal(rows[:1440],_host_table(ice)[DUST_ROW_START:DUST_ROW_START+1440])
    # Mass absorption at 555 nm (bin 35): Colorado < Zhang < Saharan < Greenland,
    # the ordering the 2021 description reports for these compositions.
    mac=[float(rows[s*1440+35*3+1]) for s in range(4)]
    assert mac[2]<mac[0]<mac[1]<mac[3]


@requires_gpu
def test_ice_and_dust_sets_change_only_when_selected():
    import cupy as cp
    from woof.core.snow_papers import evaluate
    n=1
    x=cp.zeros((n,56),dtype=cp.float32)
    x[:,24]=60; x[:,25]=.7; x[:,28]=263.15; x[:,30]=.4; x[:,31]=1
    x[:,34]=100; x[:,42]=-80; x[:,5:9]=.1; x[:,52:56]=100
    ist=cp.ones(n,dtype=cp.int32)
    run=lambda **o: cp.asnumpy(evaluate(x,ist,options=o))[0]
    base=run()
    # Defaults stated explicitly give the same bytes; a dust set without dust
    # changes nothing.
    assert np.array_equal(base,run(snicar_ice_ri_opt=2,snicar_dust_optics_opt=0))
    assert np.array_equal(base,run(snicar_dust_optics_opt=3))
    w84,p16=run(snicar_ice_ri_opt=1),run(snicar_ice_ri_opt=3)
    # More visible ice absorption in both alternative sets lowers visible
    # albedo; the NIR, where the sets agree, moves far less.
    assert p16[0]<base[0] and w84[0]<base[0]
    assert abs(p16[1]-base[1])<abs(p16[0]-base[0])
    dusty=[run(snicar_dust_ppm=100.,snicar_dust_optics_opt=s)[0] for s in range(4)]
    assert dusty[2]>dusty[0]>dusty[1]>dusty[3]
    assert all(d<base[0] for d in dusty)
    with pytest.raises(ValueError):
        run(snicar_ice_ri_opt=4)
    with pytest.raises(ValueError):
        run(snicar_dust_optics_opt=4)
    print('ice sets vis/nir',base[:2].tolist(),w84[:2].tolist(),p16[:2].tolist(),'dust vis',dusty)


def test_optical_set_controls_leave_default_settings_and_identity_alone():
    from types import SimpleNamespace
    from woof.core.snow_papers import settings
    base=dict(snicar_snowshape_opt=3,snicar_snowbc_intmix=True,snicar_snowdust_intmix=False,
              snicar_bc_ppb=0.,snicar_dust_ppm=0.,snicar_ice_density=900.,snicar_bubble_radius_um=1000.)
    assert settings(SimpleNamespace(**base,snicar_ice_ri_opt=2,snicar_dust_optics_opt=0))==base
    picked=settings(SimpleNamespace(**base,snicar_ice_ri_opt=3,snicar_dust_optics_opt=2))
    assert picked=={**base,'snicar_ice_ri_opt':3,'snicar_dust_optics_opt':2}

# --- Incident spectra and band count ---
# Re-implemented from published descriptions, not bitwise to upstream:
# Flanner et al. 2021 (GMD 14, 7673) s2.4 for the six atmospheres and the
# clear/cloudy split, Lin et al. 2025 (JHM) / Lin 2024 RAL talk for the two
# namelist controls, Flanner et al. 2007 s2 for the five bands.

def test_incident_spectra_asset_and_default_identity():
    import hashlib
    from dataclasses import replace
    from pathlib import Path
    from woof.config import RunConfig
    from woof.core.snow_papers import SPECTRA_SHA256, _host_spectra, settings
    data = Path(__file__).resolve().parents[1]/'woof/data/snow_papers'
    assert hashlib.sha256((data/'solar_spectra.bin').read_bytes()).hexdigest() == SPECTRA_SHA256
    mu, spec = _host_spectra()
    assert spec.shape == (6, mu.size, 480, 2) and np.all(np.diff(mu) > 0)
    assert np.isfinite(spec).all() and (spec >= 0).all()
    cfg = RunConfig(nx=10, ny=10, nz=10, dx=12000, dy=12000, ztop=18000, dt=60,
                    run_seconds=600, sf_surface_physics=4, opt_alb=3)
    # Defaults (packaged G173, 480 bands) never reach an identity.
    assert 'snicar_solarspec_opt' not in settings(cfg)
    assert 'snicar_bandnumber_opt' not in settings(cfg)
    chosen = settings(replace(cfg, snicar_solarspec_opt=2, snicar_bandnumber_opt=1))
    assert chosen['snicar_solarspec_opt'] == 2 and chosen['snicar_bandnumber_opt'] == 1


def _deep_snow_column(mu0):
    import cupy as cp
    x = cp.zeros((1, 56), dtype=cp.float32)
    x[:, 24] = 0        # no aging step: the grain stays at its carried radius
    x[:, 25] = mu0
    x[:, 28] = 263.15
    x[:, 30] = 2.0      # 2 m, 2000 kg/m2: optically semi-infinite
    x[:, 31] = 1
    x[:, 34] = 2000
    x[:, 42] = -200     # 200 um carried radius
    x[:, 5:11] = 0      # black ground under the pack
    x[:, 52:56] = 100
    return x


@requires_gpu
def test_incident_spectrum_weights_match_hand_computation():
    """Broadband VIS/NIR direct and diffuse albedo of a semi-infinite pack is
    the incident-weighted mean of its spectral albedo, with the direct beam
    weighted by the clear-sky (direct) spectrum and the diffuse beam by the
    cloudy-sky (diffuse) spectrum (Flanner 2021 s2.4), linear in cos(SZA)
    between the tabulated nodes. Computed here by hand from the asset, the
    optics table and the single-layer solver, independent of the kernel's
    spectral loop and reduction."""
    import cupy as cp
    from woof.core.snow_papers import evaluate, _host_spectra, _host_table, _module
    mu0 = 0.55
    x = _deep_snow_column(mu0)
    tab = np.asarray(_host_table(), np.float64)
    mu, spectra = _host_spectra()
    k = int(np.searchsorted(mu, mu0) - 1)
    f = (mu0 - mu[k])/(mu[k+1] - mu[k])
    centres = 0.205 + 0.01*np.arange(480)
    for option in range(1, 7):
        out, state = evaluate(x, cp.ones(1, dtype=cp.int32),
                              options=dict(snicar_solarspec_opt=option, snicar_snowshape_opt=1),
                              return_state=True)
        out = cp.asnumpy(out)[0]
        radius = float(cp.asnumpy(state)[0, 7*2+1])
        assert abs(radius - 200) < 1e-2
        axis = tab[:32]
        i = 0
        while i < 30 and radius > axis[i+1]:
            i += 1
        g = np.clip(np.log(radius/axis[i])/np.log(axis[i+1]/axis[i]), 0, 1)
        v = tab[1504+i*480*3:1504+(i+1)*480*3].reshape(480, 3)
        u = tab[1504+(i+1)*480*3:1504+(i+2)*480*3].reshape(480, 3)
        o = v + g*(u - v)
        probe = np.stack([o[:, 0]*2000, o[:, 1]/o[:, 0], o[:, 2], np.full(480, mu0)], 1)
        y = cp.empty((480, 10), dtype=cp.float64)
        _module(cp.cuda.Device().id).get_function('snow_ad_probe')(
            (8,), (64,), (cp.asarray(probe), y, np.int32(480)))
        y = cp.asnumpy(y)
        w = spectra[option-1, k].astype(np.float64)*(1-f) + spectra[option-1, k+1].astype(np.float64)*f
        vis, nir = centres < 0.7, centres >= 0.7
        expect = [(w[m, 0]*y[m, 0]).sum()/w[m, 0].sum() for m in (vis, nir)]
        expect += [(w[m, 1]*y[m, 2]).sum()/w[m, 1].sum() for m in (vis, nir)]
        np.testing.assert_allclose(out[:4], expect, rtol=2e-5, atol=2e-6,
                                   err_msg=f'spectrum option {option}')
        print('spectrum', option, 'kernel', out[:4].tolist(), 'hand', expect)


@requires_gpu
def test_incident_spectrum_physics_and_five_bands():
    """Physical response the published spectra imply, plus 5 vs 480 bands.

    A moist atmosphere removes the 1.4/1.9 um water bands where snow is dark,
    so the remaining NIR beam is weighted toward the brighter 0.7-1.2 um snow
    and NIR albedo rises: mid-latitude summer above sub-Arctic winter, and
    the 556 hPa high-mountain column (less vapour above) below mid-latitude
    summer. Five flux-weighted bands stay within 0.03 of 480 bands in every
    broadband albedo of this 200 um pack (measured here: largest gap 0.023,
    direct NIR, from averaging single-scattering optics over 1.5-5 um)."""
    import cupy as cp
    from woof.core.snow_papers import evaluate
    x = _deep_snow_column(0.5)
    ist = cp.ones(1, dtype=cp.int32)
    run = lambda **o: cp.asnumpy(evaluate(x, ist, options=o))[0]
    nir = {s: run(snicar_solarspec_opt=s)[1] for s in range(7)}
    assert nir[2] > nir[1] > nir[3]          # MLS > MLW > SAW
    assert nir[6] < nir[2]                   # high mountain < MLS
    for s in range(7):
        full, five = run(snicar_solarspec_opt=s), run(snicar_solarspec_opt=s, snicar_bandnumber_opt=1)
        assert np.all(np.abs(full[:4] - five[:4]) < 0.03), (s, full, five)
        assert abs(full[5] - five[5]) < 0.03
        print('spectrum', s, '480 bands', full[:4].tolist(), '5 bands', five[:4].tolist())
    print('NIR direct albedo by spectrum', nir)

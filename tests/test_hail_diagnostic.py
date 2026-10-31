"""Hail diameter lifecycle and the physical PSD tail criterion.

These are covering tests. The release qualification is a complete real case,
with observations and Rust-rendered fields, recorded separately.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import numpy as np
import pytest
from conftest import requires_gpu
from woof.config import RunConfig
from woof.core import hail
from woof.io import restart
from test_restart import _cfg, _fill_setup, _shim_state


def test_automatic_gate_and_explicit_opt_out():
    c=_cfg(mp_physics=6,dx=3000,dy=3000)
    assert hail.enabled(c)
    assert not hail.enabled(replace(c,dx=12000,dy=12000))
    assert not hail.enabled(replace(c,hailcast_opt=0))
    assert not hail.enabled(replace(c,mp_physics=0))
    assert hail.enabled(replace(c,dx=12000,dy=12000,hailcast_opt=1))


def test_history_reset_retains_run_max_and_updraft_memory(monkeypatch):
    from woof.core.uh_diag import reset_up_heli_max
    from woof.io.wrfout import _live_state_history_fields, _VAR_META
    state=_shim_state(_cfg(mp_physics=6,moist=True),monkeypatch)
    for slot in hail.PERSISTENT:
        state.existing_scratch(slot)[...]=7
    fields=_live_state_history_fields(state)
    for slot,(name,units,_) in hail.HISTORY.items():
        assert fields[name] is state.existing_scratch(slot)
        assert _VAR_META[name][1]==units
    reset_up_heli_max(state)
    for slot in hail.PERSISTENT:
        assert np.all(state.existing_scratch(slot)==(0 if slot in hail.WINDOW else 7))


def test_restart_carries_embryos_duration_and_run_max(monkeypatch,tmp_path):
    cfg=_cfg(mp_physics=6,moist=True)
    source=_shim_state(cfg,monkeypatch);_fill_setup(source)
    for i,slot in enumerate(hail.PERSISTENT):
        source.existing_scratch(slot)[...]=i+1
    path=restart.write_restart(tmp_path/'hail.npz',source,cfg)
    restored=_shim_state(cfg,monkeypatch);_fill_setup(restored)
    restart.restore_restart(path,restored,cfg)
    for i,slot in enumerate(hail.PERSISTENT):
        assert np.all(restored.existing_scratch(slot)==i+1)
    off_cfg=replace(cfg,hailcast_opt=0)
    off=_shim_state(off_cfg,monkeypatch);_fill_setup(off)
    restart.restore_restart(path,off,off_cfg)
    assert all(off.existing_scratch(s) is None for s in hail.PERSISTENT)


def test_generated_stone_routines_are_reproducible():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('hail_translate',root/'tools/hail_wrf461/translate.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    assert mod.generate()==(root/'woof/core/kernels/hailcast_wrf.cuh').read_text()


def test_memory_planner_prices_every_hail_slot_and_classifies_lifetimes():
    from woof.core.preflight import scratch_slot_registry, scratch_slot_lifetime
    cfg = _cfg(mp_physics=6, moist=True, dx=3000, dy=3000)
    shapes = hail.scratch_shapes(cfg)
    slots = scratch_slot_registry(cfg)
    assert set(hail.PERSISTENT) <= shapes.keys()
    assert all(slots[name] == shape for name, shape in shapes.items())
    assert all(scratch_slot_lifetime(name) is not None for name in shapes)
    assert scratch_slot_lifetime("hail_zero").kind == "carrying"
    assert not hail.scratch_shapes(replace(cfg, hailcast_opt=0))


@requires_gpu
def test_psd_tail_mass_number_volume_and_run_accumulation():
    import cupy as cp
    from woof.core.kernels import get_kernel
    # Two transported moment distributions, exponential graupel and gamma
    # alpha=1 hail. A second level tests the column maximum separately.
    nz,ny,nx=2,8,16;shape=(nz,ny,nx);n=ny*nx
    rho=cp.ones(shape,cp.float32);qg=cp.full(shape,.002,cp.float32)
    ng=cp.full(shape,1.,cp.float32);qh=cp.full(shape,.004,cp.float32)
    nh=cp.full(shape,.02,cp.float32);qh[1]*=2
    vg=qg/500;vh=qh/900
    fields=[cp.zeros((ny,nx),cp.float32) for _ in range(4)]
    kernel=get_kernel('hail','hail_psd')
    args=(rho,qg,ng,vg,qh,nh,vh,*fields,*map(np.int32,(nx,ny,nz,1,1,1,1,0)),*map(np.float32,(500,900,0,1,0,0)))
    kernel((1,),(128,),args)
    def size(mass,number,density,alpha):
        slope=(np.pi/6*density*(alpha+1)*(alpha+2)*(alpha+3)*number/mass)**(1/3)
        lo,hi=0.,100.
        for _ in range(80):
            mid=(lo+hi)/2
            tail=np.exp(-mid)*(1+mid if alpha==1 else 1)
            if tail*number>.0005:lo=mid
            else:hi=mid
        return (lo+hi)/2/slope
    low=max(size(.002,1,500,0),size(.004,.02,900,1))
    high=max(size(.002,1,500,0),size(.008,.02,900,1))
    np.testing.assert_allclose(fields[0].get(),low,rtol=2e-6)
    np.testing.assert_allclose(fields[1].get(),high,rtol=2e-6)
    fields[0].fill(0);fields[1].fill(0);qg.fill(0);qh.fill(0)
    kernel((1,),(128,),args)
    assert cp.all(fields[0]==0).item()
    np.testing.assert_allclose(fields[3].get(),high,rtol=2e-6)


@requires_gpu
def test_duration_uses_previous_neighbor_and_respects_open_edges():
    import cupy as cp
    from woof.core.kernels import get_kernel
    shape=(16,16);mask=cp.zeros(shape,cp.float32);old=cp.zeros_like(mask);dur=cp.zeros_like(mask)
    old[8,8]=1;dur[8,8]=920
    out=cp.zeros_like(mask)
    get_kernel('hail','hail_duration')((2,),(128,),(mask,old,dur,out,np.int32(16),np.int32(16),np.float32(15),np.int32(0),np.int32(0)))
    result=out.get();expected=np.zeros(shape,np.float32);expected[7:10,7:10]=935
    np.testing.assert_array_equal(result,expected)


@requires_gpu
def test_hailcast_negative_fitted_impact_speed_stays_finite():
    """The real-case trajectory crossed the empirical fit's physical boundary.

    Stock WRF returns NaN for embryo five. Bounding the impact speed before
    its fractional power preserves a 23.935 mm stone in the independent
    corrected Fortran integration, rather than silently returning no hail.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel
    path = Path(__file__).parent / 'data/oracles/hailcast/impact-speed-profile.bin'
    data = path.read_bytes()
    ncol, nz = np.frombuffer(data, dtype='<i4', count=2)
    assert ncol == 1 and nz == 49
    values = np.frombuffer(data, dtype='<f4', offset=8)
    profiles = [cp.asarray(row.copy()) for row in values[:11*nz].reshape(11, nz)]
    terrain, duration = [cp.asarray([v], dtype=cp.float32) for v in values[11*nz:]]
    outputs = [cp.zeros(1, dtype=cp.float32) for _ in range(9)]
    get_kernel('hail', 'hailcast_columns')(
        (1,), (1,), (*profiles, terrain, duration, *outputs,
                    np.int32(1), np.int32(1), np.int32(nz)))
    actual = np.array([value.get()[0] for value in outputs])
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:4], np.zeros(4))
    np.testing.assert_allclose(actual[[4, 5, 8]], 23.934831619262695,
                               rtol=.005, atol=.1)


def test_hrrr_generation_is_reproducible_and_selected_by_version():
    """fix/hailcast: the operational HRRR v4.1.21 HAILCAST is a closed
    translation of the pinned branch file, and hailcast_version picks the
    column kernel; an unknown generation is refused by name."""
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('hail_translate_hrrr',root/'tools/hail_wrf461/translate.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    assert mod.generate('wrf_39_noaa')==(root/'woof/core/kernels/hailcast_wrf39.cuh').read_text()
    assert hail.HAILCAST_KERNELS=={'wrf_39_noaa':'hailcast_columns_wrf39','wrf_461':'hailcast_columns'}
    from woof.config import validate_run_config, ConfigurationRefused
    cfg=_cfg(mp_physics=8, moist=True, dx=3000, dy=3000)
    assert cfg.hailcast_version=='wrf_39_noaa'
    with pytest.raises(ConfigurationRefused, match='hailcast_version'):
        validate_run_config(replace(cfg,hailcast_version='wrf_99'))


@requires_gpu
@pytest.mark.parametrize("version", ["wrf_461", "wrf_39_noaa"])
def test_both_generations_match_their_fortran_on_real_storm_columns(version):
    """fix/hailcast: 94 real updraft columns from WOOF HRRR-configuration
    hail days (2023-03-31, 2023-06-11, 2025-03-14) through each forecast
    kernel, against that generation's own gfortran build
    (tools/hail_wrf461/oracle.py --form, -O0, no contraction). The two
    generations differ by design: on these columns WRF 4.6.1 lands a stone
    in fewer than half, operational HRRR v4 in every one."""
    import cupy as cp
    from woof.core.kernels import get_kernel
    fx = np.load(Path(__file__).parent / 'data/oracles/hailcast/real_columns.npz')
    prof = fx['profiles']; n = prof.shape[1]; nz = prof.shape[2]
    profiles = [cp.asarray(np.ascontiguousarray(prof[v].T)) for v in range(11)]
    outputs = [cp.zeros(n, dtype=cp.float32) for _ in range(9)]
    get_kernel('hail', hail.HAILCAST_KERNELS[version])(
        ((n + 31) // 32,), (32,), (*profiles, cp.asarray(fx['ht']), cp.asarray(fx['wdur']),
                                    *outputs, np.int32(n), np.int32(1), np.int32(nz)))
    actual = np.stack([o.get() for o in outputs[:5]], 1)
    want = fx[version]
    np.testing.assert_allclose(actual, want, rtol=.005, atol=.1)
    assert np.array_equal(actual.max(1) > 0, want.max(1) > 0)
    landed = (want.max(1) > 0).sum()
    assert (landed == n) if version == 'wrf_39_noaa' else (0 < landed < n // 2)

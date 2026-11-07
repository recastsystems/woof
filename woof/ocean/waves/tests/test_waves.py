from pathlib import Path
import json
import math
import os
import pytest
from woof.ocean.waves.model import SpectralGrid, WaveModel
from woof.ocean.waves.cli import validate_config


def test_input_contract():
    with pytest.raises(ValueError, match="0.4 Hz"):
        SpectralGrid(frequencies=16)
    with pytest.raises(ValueError, match="research_data_licenses"):
        validate_config({"forcing": "a", "bathymetry": "b", "output": "c", "input_licenses": ["CC-BY-4.0"]})
    for value in (-1, 0, float("nan"), True):
        with pytest.raises(ValueError, match="dt_seconds"):
            validate_config({"forcing": "a", "bathymetry": "b", "output": "c", "dt_seconds": value})


def test_spectral_capture_configuration_rejects_ambiguous_cell_identity():
    config = {"forcing": "a", "bathymetry": "b", "output": "c"}
    for cells in (None, {}, [{"id": "x", "i": True, "j": 1}],
                  [{"id": "x", "i": -1, "j": 1}],
                  [{"id": "x", "i": 1, "j": 1}] * 2):
        with pytest.raises(ValueError, match="capture_cells"):
            validate_config({**config, "capture_cells": cells})
    validate_config({**config, "capture_cells": [{"id": "x", "i": 0, "j": 0}]})


@pytest.mark.parametrize("command", ["verify", "verify-wind"])
def test_verification_door_dispatches_the_requested_native_operation(monkeypatch, command):
    from types import SimpleNamespace
    from woof.ocean.waves import cli
    monkeypatch.setattr(cli, "native_tools", lambda: (Path("rw_waves_io"), Path("libwaves_io.so")))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.main([command,"field.nc","obs.txt","12345","20","-80","pairs.csv"]) == 0
    assert calls == [["rw_waves_io",command,"field.nc","obs.txt","12345","20","-80","pairs.csv"]]


def device():
    if os.environ.get("GPUWM_NO_LOCAL_GPU") == "1":
        pytest.skip("GPU tests are disabled for this run")
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA card")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA driver")
    return cp


def model():
    cp = device()
    m = WaveModel(cp.full((20, 24), 1000, cp.float32),
                  cp.linspace(20, 24.75, 20, dtype=cp.float32),
                  cp.linspace(-96, -90.25, 24, dtype=cp.float32))
    return cp, m


def test_dia_preserves_discrete_action_and_variance():
    cp, m = model()
    # Broad positive spectrum exercises both mirrored quadruplets, reverse
    # exchanges and fractional frequency/direction interpolation.
    f = m.frequency[None,None,:,None]
    angle = cp.arange(m.nd, dtype=cp.float32)[None,None,None,:] * m.f(2*math.pi/m.nd)
    m.spectrum[:] = 0.001 * cp.exp(-((f - .15)/.08)**2) * (1 + .7*cp.cos(angle))
    wind = cp.full(m.depth.shape, 10, cp.float32)
    m._moments(wind, wind*0, wind, wind*0, 0)
    m.rate.fill(0)
    m._launch("dia", m.nc*m.ns, (m.spectrum, m.rate, m.frequency, m.df,
        m.depth, m.moments, m.i(m.nc), m.i(m.nf), m.i(m.nd),
        m.f(math.log(m.spectral.ratio)), m.f(30)))
    energy = m.rate * m.df[None,None,:,None]
    action = energy / m.frequency[None,None,:,None]
    assert float(cp.sum(cp.abs(energy))) > 0
    assert abs(float(cp.sum(energy))) / float(cp.sum(cp.abs(energy))) < 2e-6
    assert abs(float(cp.sum(action))) / float(cp.sum(cp.abs(action))) < 2e-6
    before = m.spectrum.copy()
    m._launch("dia_apply", m.nc, (m.spectrum, m.rate, m.limiter,
        m.i(m.nc), m.i(m.nf), m.i(m.nd), m.f(30)), blocks=True)
    assert float(cp.min(m.spectrum)) >= 0
    assert float(cp.max(cp.abs(m.spectrum - before))) > 0
    assert float(cp.min(m.limiter)) > 0


def test_transport_conserves_area_weighted_variance_and_stays_positive():
    cp, m = model()
    y, x = cp.meshgrid(cp.arange(m.ny), cp.arange(m.nx), indexing="ij")
    pulse = cp.where((x>6)&(x<17)&(y>6)&(y<14), 0.01, 0).astype(cp.float32)
    m.spectrum[:] = pulse[:,:,None,None]
    area = cp.cos(cp.deg2rad(m.latitude))[:,None,None,None]
    before = float(cp.sum(m.spectrum*area*m.df[None,None,:,None]))
    m._launch("propagate", m.nc*m.ns, (m.spectrum, m.work, m.depth,
        m.latitude, m.cg, m.turn, m.i(m.nx), m.i(m.ny), m.i(m.nf), m.i(m.nd),
        m.f(m.dx), m.f(m.dy), m.f(min(30,m.max_dt))))
    after = float(cp.sum(m.work*area*m.df[None,None,:,None]))
    assert abs(after-before)/before < 3e-7
    assert float(cp.min(m.work)) >= 0
    assert float(cp.max(cp.abs(m.work-m.spectrum))) > 0


def test_output_direction_partition_and_height_units():
    cp, m = model()
    # Northward travelling waves have a meteorological FROM direction of 180.
    m.spectrum.fill(0)
    fi, di = 14, m.nd//4
    target_variance = 2.25
    m.spectrum[:,:,fi,di] = target_variance/(m.df[fi]*m.f(2*math.pi/m.nd))
    zero = cp.zeros(m.depth.shape, cp.float32)
    out = m.output(zero, zero)
    assert bool(cp.allclose(out[0],6,atol=2e-6))
    assert bool(cp.allclose(out[1],1/m.frequency[fi],atol=1e-6))
    assert bool(cp.allclose(out[2],180,atol=1e-4))
    assert bool(cp.allclose(out[0]**2,out[3]**2+out[4]**2,atol=1e-5))
    assert bool(cp.allclose(out[4],out[0],atol=1e-6))


def test_entire_source_path_grows_calm_water_and_preserves_dry_cells():
    cp, m = model()
    # A spatially varying real domain is covered by the accompanying full run.
    # This regression isolates calm-start generation and every default source.
    m.depth[0,:] = 0
    wind = cp.full(m.depth.shape, 15, cp.float32)
    zero = cp.zeros_like(wind)
    elapsed = 0
    while elapsed < 1800:
        elapsed += m.step(wind, zero, wind, zero, alpha=0, dt_limit=min(60,1800-elapsed))
    out = m.output(wind,zero)
    assert float(cp.nanmax(out[0])) > .1
    assert bool(cp.all(cp.isfinite(m.spectrum)))
    assert float(cp.min(m.spectrum)) >= 0
    assert float(cp.max(cp.abs(m.spectrum[0]))) == 0
    assert bool(cp.all(cp.isnan(out[:,0,:])))
    assert float(cp.min(m.limiter)) > 0
    assert float(cp.max(cp.abs(m.rate))) > 0
    assert float(cp.max(m.budgets[...,0])) > 0
    assert float(cp.min(m.budgets[...,1])) < 0
    assert float(cp.min(m.budgets[...,2])) < 0


def test_stiff_directional_refraction_preserves_variance_and_positivity():
    cp, m = model()
    angle = cp.arange(m.nd, dtype=cp.float32) * m.f(2*math.pi/m.nd)
    # Includes convergent/divergent rings and periodic flow in both senses.
    # Directional Courant number exceeds 1000; explicit upwind would be negative.
    m.turn[:] = (0.4*cp.sin(angle)[None,:]+cp.where(cp.arange(m.nf)%3==0,0.6,
                  cp.where(cp.arange(m.nf)%3==1,-0.6,0))[:,None])
    m.spectrum[:] = (1+0.8*cp.cos(angle))[None,None,None,:]
    before = cp.sum(m.spectrum,axis=-1)
    m._launch("refract",m.nc*m.nf,(m.spectrum,m.turn,m.depth,m.i(m.nc),
              m.i(m.nf),m.i(m.nd),m.f(3600)))
    assert bool(cp.all(cp.isfinite(m.spectrum)))
    assert float(cp.min(m.spectrum))>=0
    assert bool(cp.allclose(cp.sum(m.spectrum,axis=-1),before,rtol=3e-6))
    assert float(cp.max(cp.abs(m.spectrum-(1+0.8*cp.cos(angle)))))>0.1


def test_shared_solver_abi_advances_device_sources_and_tracks_water_depth(monkeypatch):
    from types import SimpleNamespace
    from woof.ocean.waves import create_solver
    cp=device()
    shape=(20,24)
    latitude,longitude=cp.meshgrid(cp.linspace(20,24.75,20,dtype=cp.float32),
                                  cp.linspace(-96,-90.25,24,dtype=cp.float32),indexing='ij')
    grid=SimpleNamespace(ny=shape[0],nx=shape[1],latitude=latitude,longitude=longitude)
    wet=cp.ones(shape,cp.float32);wet[0]=0
    bath=SimpleNamespace(depth=cp.full(shape,1000,cp.float32),wet=wet)
    s=create_solver(grid,bath,config={})
    wind=cp.full(shape,15,cp.float32);zero=cp.zeros(shape,cp.float32)
    forcing=SimpleNamespace(u10=wind,v10=zero,rho_air=cp.full(shape,1.225,cp.float32))
    pointers=tuple(a.data.ptr for a in (s.outputs.hs,s.outputs.tp,s.outputs.direction))
    # The legacy standalone path synchronizes for a global source bound.
    # Accidentally delegating to it violates the asynchronous shared contract.
    def reject(*args,**kwargs):raise AssertionError('host synchronization/copy in shared ABI')
    monkeypatch.setattr(WaveModel,'step',reject)
    monkeypatch.setattr(cp,'asnumpy',reject)
    monkeypatch.setattr(cp.cuda.runtime,'deviceSynchronize',reject)
    s.set_forcing(forcing)
    s.set_water_level(cp.full(shape,2,cp.float32))
    for _ in range(30):s.step(60)
    assert tuple(a.data.ptr for a in (s.outputs.hs,s.outputs.tp,s.outputs.direction))==pointers
    assert float(cp.max(s.outputs.hs))>.1
    assert bool(cp.all(cp.isfinite(s.outputs.hs)))
    assert bool(cp.all(s.backend.depth==1002))
    assert bool(cp.all(s.outputs.hs[0]==0))
    assert bool(cp.all(s.outputs.tp[0]==0))
    assert bool(cp.allclose(s.outputs.direction[5:-5,5:-5],90,atol=2))
    s.set_water_level(cp.full(shape,-2000,cp.float32));s.step(60)
    assert bool(cp.all(s.outputs.hs==0))
    assert bool(cp.all(s.outputs.tp==0))
    with pytest.raises(ValueError,match='positive'):
        s.step(0)


def test_spectral_capture_gathers_the_requested_cell_and_interleaved_budgets():
    from woof.ocean.waves.cli import capture_layout, capture_spectra
    cp, m = model()
    m.spectrum[3, 4] = cp.arange(m.ns, dtype=cp.float32).reshape(m.nf, m.nd)
    m.budgets[3, 4, ..., 0] = 11
    m.budgets[3, 4, ..., 1] = -7
    m.budgets[3, 4, ..., 2] = -3
    m.budgets[3, 4, ..., 3] = -1
    payload = capture_spectra(m, [{"id": "probe", "i": 4, "j": 3}])
    layout = capture_layout(m.nf, m.nd)
    assert payload.size == 2*m.nf + layout["cell_words"]
    assert bool(cp.all(payload[:m.nf] == m.frequency))
    assert bool(cp.all(payload[2*m.nf:2*m.nf+m.ns] == cp.arange(m.ns)))
    budget = payload[2*m.nf+m.ns:2*m.nf+5*m.ns].reshape(m.nf, m.nd, 4)
    assert bool(cp.all(budget[..., 0] == 11))
    assert bool(cp.all(budget[..., 3] == -1))
    assert bool(cp.all(payload[2*m.nf+5*m.ns:2*m.nf+6*m.ns] == 0))


def test_shared_solver_near_calm_outputs_share_the_diagnostic_energy_threshold(monkeypatch):
    from types import SimpleNamespace
    from woof.ocean.waves import create_solver
    cp = device()
    shape = (20, 24)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        latitude, longitude = cp.meshgrid(
            cp.linspace(20, 24.75, shape[0], dtype=cp.float32),
            cp.linspace(-96, -90.25, shape[1], dtype=cp.float32), indexing="ij")
        grid = SimpleNamespace(ny=shape[0], nx=shape[1], latitude=latitude, longitude=longitude)
        wet = cp.ones(shape, cp.float32)
        wet[0] = 0
        solver = create_solver(grid, SimpleNamespace(depth=cp.full(shape, 1000, cp.float32), wet=wet))
        m = solver
        # Include Hs below 1e-7, the reported invalid band (1e-7, sqrt(16e-15)],
        # and both sides of the diagnostic's float32 variance cutoff.
        variance = cp.asarray([0, 4e-16, 7.5e-16, 1e-15, 1.0001e-15, 2e-15, 1e-6], cp.float32)
        columns = m.backend.wet_to_full % shape[1]
        for column, value in enumerate(variance, start=4):
            m.state[columns == column, 14, 0] = value / (m.df[14] * (2*math.pi/m.nd) * (2*math.pi*m.frequency[14]))
        zero = cp.zeros(shape, cp.float32)
        solver.set_forcing(SimpleNamespace(u10=zero, v10=zero, rho_air=cp.full(shape,1.225,cp.float32)))
        pointers = tuple(a.data.ptr for a in (solver.outputs.hs, solver.outputs.tp, solver.outputs.direction))
        launched_streams = []
        launch = m._launch
        def checked_launch(*args, **kwargs):
            launched_streams.append(cp.cuda.get_current_stream().ptr)
            return launch(*args, **kwargs)
        def reject(*args, **kwargs):
            raise AssertionError("host synchronization/copy in shared ABI")
        with monkeypatch.context() as guards:
            guards.setattr(m, "_launch", checked_launch)
            guards.setattr(cp, "asnumpy", reject)
            guards.setattr(cp.cuda.runtime, "deviceSynchronize", reject)
            solver.refresh_outputs()
        actual_variance = cp.zeros(shape, cp.float32)
        actual_variance.ravel()[m.backend.wet_to_full] = m.moments[:, 0]
        calm = (wet == 0) | (actual_variance <= cp.float32(1e-15))
        moving = ~calm
        # Prove the regression actually exercises the coupled run's failing band.
        assert bool(cp.any((actual_variance > 1e-14/16) & (actual_variance <= cp.float32(1e-15))))
        assert bool(cp.any(moving))
        fields = (solver.outputs.hs, solver.outputs.tp, solver.outputs.direction)
        assert all(bool(cp.all(cp.isfinite(field))) for field in fields)
        for field in fields:
            assert bool(cp.all(field[calm] == 0))
        assert bool(cp.all(solver.outputs.hs[moving] > 0))
        # The current diagnostic interpolates the spectral peak within its bin.
        assert bool(cp.allclose(solver.outputs.tp[moving], 1/m.frequency[14],
                               rtol=math.sqrt(m.spectral.ratio)-1))
        assert bool(cp.allclose(solver.outputs.direction[moving], 90, atol=1e-4))
        assert tuple(a.data.ptr for a in (solver.outputs.hs, solver.outputs.tp, solver.outputs.direction)) == pointers
        assert launched_streams and set(launched_streams) == {stream.ptr}
        # The output-cadence validation barrier must reject invalid wet action.
        m.state.fill(cp.nan)
        solver.refresh_outputs()
        with pytest.raises(RuntimeError, match="nonfinite"):
            solver.collect()

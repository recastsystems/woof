"""Correlated acceleration must respect its long-time energy-rate budget."""
import math
import json
import pytest

from woof.hex import stochastic

def backscatter_rate_scale(*args):
    return stochastic.backscatter_rate_scale(*args)


@pytest.mark.parametrize('dt,tau', [(30.,21600.),(300.,21600.),(600.,21600.),(1200.,21600.),(600.,1.)])
def test_ar1_velocity_covariance_recovers_requested_energy_rate(dt, tau, tmp_path):
    # Independently propagate exact second moments of v += dt*r and
    # r_next = phi*r + innovation. Unit stationary pattern variance.
    phi = math.exp(-dt/tau)
    steps = max(1000, int(100*tau/dt))
    variance, covariance = 0., 0.
    for _ in range(steps):
        variance += dt*dt + 2*dt*covariance
        covariance = phi*(covariance+dt)
    dissipation, fraction = .2, .1
    rate_variance = fraction*dissipation*backscatter_rate_scale(dt,tau)
    kinetic_rate = .5*variance*rate_variance/(steps*dt)
    old_kinetic_rate = .5*variance*(fraction*dissipation*2/dt)/(steps*dt)
    (tmp_path/'energy-audit.json').write_text(json.dumps(dict(
        dt_s=dt, time_s=tau, steps=steps, stationary_pattern_variance=1.,
        dissipation=dissipation, fraction=fraction, expected_energy_rate=fraction*dissipation,
        covariance_normalized_energy_rate=kinetic_rate, old_white_noise_energy_rate=old_kinetic_rate,
        old_to_new_ratio=old_kinetic_rate/kinetic_rate, no_gpu_execution=True,
        scope='exact second-moment recurrence for a scalar AR(1) acceleration; not a forecast'),
        indent=2)+'\n')
    assert kinetic_rate == pytest.approx(fraction*dissipation, rel=.011)


def test_held_updates_use_the_pattern_update_period():
    assert backscatter_rate_scale(600,21600,3600) == backscatter_rate_scale(3600,21600)
    assert backscatter_rate_scale(600,21600)/(2/600) == pytest.approx(.013888,rel=1e-4)


@pytest.mark.parametrize('interval', [0., 3600.])
def test_production_launches_apply_covariance_budget_to_both_forcings(monkeypatch, interval):
    import sys
    from types import SimpleNamespace
    import numpy as np
    from woof.hex.stochastic import StochasticConfig, StochasticPhysics
    # Capture actual production launch arguments using tiny test-only arrays.
    # No CUDA kernel, GPU context or production CPU backend is used here.
    monkeypatch.setitem(sys.modules, 'cupy', np)
    launches = {}
    monkeypatch.setattr(stochastic, '_kernel', lambda name:
        lambda grid, block, args: launches.setdefault(name, args))
    r = SimpleNamespace(cells=2, edges=1, edge_weight=np.ones(1, np.float32),
        coe=np.array([[0, 1]], np.int32), voe=np.array([[0, 1]], np.int32),
        orientation=np.ones(1, np.float32), dv=np.ones(1, np.float32),
        area=np.full(2, .5, np.float32))
    c = StochasticConfig(skeb=True, time_s=21600., interval_s=interval)
    forcing = StochasticPhysics.__new__(StochasticPhysics)
    forcing.config, forcing.resource = c, r
    forcing.timing_events, forcing.calls = [None]*4, 4  # Do not create events.
    forcing.taper = forcing.vertex_taper = np.ones((1, 2), np.float32)
    forcing.skeb_norm = np.ones(1, np.float32)
    forcing.process = SimpleNamespace(advance=lambda seconds: (
        np.zeros(4, np.float32), np.arange(4, dtype=np.float32),
        np.arange(4, dtype=np.float32)))
    held = SimpleNamespace(rho_theta=np.zeros((1, 2), np.float32),
        rho_u=-np.ones((1, 1), np.float32))
    forcing.apply(held, state=SimpleNamespace(rho=np.ones((1, 2), np.float32)),
        rho_edge=np.ones((1, 1), np.float32), normal_velocity=np.ones((1, 1), np.float32),
        time_seconds=0., dt=600., names=())
    # Independently sum the infinite discrete covariance series.
    h = interval or 600.
    phi = math.exp(-h/c.time_s)
    inverse_covariance = 2*(1-phi)/(h*(1+phi))
    # force_cells argument 20 is the thermal scale (the moisture guard follows it).
    assert launches['force_cells'][20] == pytest.approx(
        inverse_covariance*c.thermal_fraction*300/1004, rel=1e-7)
    assert launches['force_edges'][-1] == pytest.approx(
        inverse_covariance*c.skeb_fraction, rel=1e-7)

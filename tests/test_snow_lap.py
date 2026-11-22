"""clean-snicar-lap: light-absorbing particles in snow (snicar_lap_opt).

Re-implemented from published descriptions, not bitwise to upstream Noah-MP.
The hand cases follow Flanner et al. 2007 sect. 2.4 (meltwater carries
scav * (Q/M) * L out of a layer) and the mass-coordinate remap documented in
woof/core/kernels/snow_lap.cu.
"""
import numpy as np
import pytest
from conftest import requires_gpu


def _probe(columns):
    import cupy as cp
    from woof.core.snow_papers import _lap_module
    x = cp.asarray(np.asarray(columns, dtype=np.float32))
    y = cp.zeros((len(columns), 36), dtype=cp.float32)
    _lap_module(cp.cuda.Device().id).get_function("snow_lap_probe")((1,), (len(columns),), (x, y, np.int32(len(columns))))
    return cp.asnumpy(y)


def _column(L, om, nm, Q, scav):
    return [*np.asarray(L, dtype=np.float64).ravel(), *om, *nm, Q, *scav]


SCAV = [0.2, 0.03, 0.2, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01]


@requires_gpu
def test_meltwater_flushing_hand_case():
    # Two live layers of 50 kg/m2 (slots 1, 2), 10 kg/m2 of meltwater left
    # the pack bottom, the pack is now 45 + 45. Top layer holds 1e-6 kg/m2
    # hydrophilic BC (20 ppb) and 5e-3 kg/m2 of fine dust (100 ppm).
    L = np.zeros((3, 9)); L[1, 0] = 1e-6; L[1, 4] = 5e-3
    y = _probe([_column(L, [0, 50, 50], [0, 45, 45], 10., SCAV)])[0]
    out = y[:27].reshape(3, 9)
    # Hand: out of layer 1 = scav * (Q/M) * L = 0.2*0.2*1e-6 = 4e-8 BC and
    # 0.02*0.2*5e-3 = 2e-5 dust, into layer 2. The same water then leaves
    # layer 2 (Q/M = 0.2 again), carrying 0.2*0.2*4e-8 = 1.6e-9 BC and
    # 0.02*0.2*2e-5 = 8e-8 dust to the soil. Remap: A = max(0, 90-(100-10))
    # = 0, scale 0.9, so old layer k lands wholly in new layer k.
    assert out[1, 0] == pytest.approx(9.6e-7, rel=1e-6)
    assert out[2, 0] == pytest.approx(3.84e-8, rel=1e-5)
    assert out[1, 4] == pytest.approx(4.98e-3, rel=1e-6)
    assert out[2, 4] == pytest.approx(1.992e-5, rel=1e-5)
    assert y[27 + 0] == pytest.approx(1.6e-9, rel=1e-4)
    assert y[27 + 4] == pytest.approx(8e-8, rel=1e-4)
    # Melt enriches the surface: 9.6e-7/45 kg/kg = 21.33 ppb > 20 ppb before.
    assert out[1, 0] / 45 * 1e9 == pytest.approx(21.333, rel=1e-4)


@requires_gpu
def test_snowfall_on_top_buries_particles_by_mass():
    # 100 kg/m2 in one layer, 20 kg/m2 of new snow, Noah-MP re-layers to
    # 20/40/60. Hand: old column [0,100) maps to [20,120); 40 % of its dust
    # lands in layer 1 [20,60), 60 % in layer 2 [60,120), none in the fresh top.
    L = np.zeros((3, 9)); L[2, 4] = 1e-2
    out = _probe([_column(L, [0, 0, 100], [20, 40, 60], 0., SCAV)])[0][:27].reshape(3, 9)
    assert out[0, 4] == 0
    assert out[1, 4] == pytest.approx(4e-3, rel=1e-6)
    assert out[2, 4] == pytest.approx(6e-3, rel=1e-6)


@requires_gpu
def test_flushing_out_of_the_bottom_reaches_the_soil():
    L = np.zeros((3, 9)); L[2, 1] = 1e-6
    y = _probe([_column(L, [0, 0, 10], [0, 0, 5], 5., SCAV)])[0]
    # One layer, frac = min(1, 5/10) = 0.5; hydrophobic BC out 0.03*0.5*1e-6.
    assert y[27 + 1] == pytest.approx(1.5e-8, rel=1e-5)
    assert y[2 * 9 + 1] + y[27 + 1] == pytest.approx(1e-6, rel=1e-6)


def _snow_inputs(n, swe=100.):
    import cupy as cp
    x = cp.zeros((n, 56), dtype=cp.float32)
    x[:, 24] = 60; x[:, 25] = .7; x[:, 28] = 263.15; x[:, 30] = .4; x[:, 31] = 1
    x[:, 34] = swe; x[:, 42] = -55; x[:, 5:9] = .1; x[:, 52:56] = 100
    return x


def _layers(n):
    import cupy as cp
    layers = cp.zeros((n, 3, 4), dtype=cp.float32)
    layers[:, :, 0] = cp.asarray([20., 30., 50.], dtype=cp.float32)
    layers[:, :, 2] = cp.asarray([.08, .12, .2], dtype=cp.float32)
    layers[:, :, 3] = 263.15
    return layers


def _state(n, layers):
    import cupy as cp
    s = cp.zeros((n, 21), dtype=cp.float32)
    for j in range(3):
        s[:, 7 * j] = layers[:, j, 0]; s[:, 7 * j + 1] = 60.
    return s


@requires_gpu
def test_zero_particles_reproduce_the_snow_solver_bitwise():
    import cupy as cp
    from woof.core.snow_papers import evaluate, LAP_WORDS
    n = 4
    x, layers = _snow_inputs(n), _layers(n)
    state = _state(n, layers)
    ist = cp.ones(n, dtype=cp.int32)
    ref = evaluate(x, ist, layers=layers, state=state, return_fluxes=True)
    lap = cp.zeros((n, LAP_WORDS), dtype=cp.float32); lap[:, 28] = 1
    dep = cp.zeros((n, 27), dtype=cp.float32)
    new = evaluate(x, ist, layers=layers, state=state, return_fluxes=True, lap=lap, deposition=dep)
    for a, b in zip(ref, new[:3]):
        assert cp.array_equal(a, b)


@requires_gpu
def test_dust_and_bc_darken_visible_snow_more_than_nir():
    import cupy as cp
    from woof.core.snow_papers import evaluate, LAP_WORDS
    n = 3
    x, layers = _snow_inputs(n), _layers(n)
    state = _state(n, layers)
    ist = cp.ones(n, dtype=cp.int32)
    lap = cp.zeros((n, LAP_WORDS), dtype=cp.float32); lap[:, 28] = 1
    lap[1, 4:9] = cp.asarray([0.2, 0.4, 0.4, 0.0, 0.0], dtype=cp.float32) * 20 * 500e-6   # 500 ppm dust, top layer
    lap[2, 0] = 20 * 200e-9                                                             # 200 ppb hydrophilic BC
    dep = cp.zeros((n, 27), dtype=cp.float32)
    alb = cp.asnumpy(evaluate(x, ist, layers=layers, state=state, return_fluxes=True, lap=lap, deposition=dep)[0])
    vis, nir = alb[:, 0], alb[:, 1]
    assert vis[1] < vis[0] - 0.02 and vis[2] < vis[0] - 0.01
    assert (vis[0] - vis[1]) > (nir[0] - nir[1])
    print("clean / dust / BC direct albedo vis", vis.tolist(), "nir", nir.tolist())


@requires_gpu
def test_particle_budget_closes_through_melt_snowfall_and_deposition():
    import cupy as cp
    from woof.core.snow_papers import evaluate, lap_outflow, LAP_WORDS
    n = 1
    x, layers = _snow_inputs(n), _layers(n)
    state = _state(n, layers)
    ist = cp.ones(n, dtype=cp.int32)
    lap = cp.zeros((n, LAP_WORDS), dtype=cp.float32)
    dep = cp.zeros((n, 27), dtype=cp.float32)
    dep[0, :9] = 1e-10; dep[0, 9:18] = 2e-10; dep[0, 18:] = 1e-5   # dry, wet, initial
    masses = [(20, 30, 50), (15, 30, 50), (0, 25, 50), (5, 25, 48), (0, 0, 30), (0, 0, 0), (0, 0, 10)]
    for step, m in enumerate(masses):
        layers[:, :, 0] = cp.asarray(m, dtype=cp.float32); x[:, 34] = float(sum(m))
        alb, carried, _, lap = evaluate(x, ist, layers=layers, state=state, return_fluxes=True, lap=lap, deposition=dep)
        lap_outflow(lap, cp.asarray([0.02], dtype=cp.float32), 60.)
        state = carried
        h = cp.asnumpy(lap)[0]
        held = h[:27].reshape(3, 9).sum(axis=0)
        assert np.allclose(held + h[38:47], h[29:38], rtol=2e-6, atol=1e-12), step
    assert h[38:47].sum() > 0     # the melt-out step sent particles to the soil


def test_inactive_lap_controls_stay_out_of_checkpoint_identity():
    from dataclasses import replace
    from woof.config import RunConfig
    from woof.io.restart import configuration_echo, _require_config_match
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000, dy=3000, ztop=12000, dt=10, run_seconds=60,
                    sf_surface_physics=4, opt_alb=3)
    old = configuration_echo(cfg)
    assert not any(k.startswith("snicar_lap") for k in old)
    on = replace(cfg, snicar_lap_opt=1)
    assert configuration_echo(on)["snicar_lap_opt"] == 1
    with pytest.raises(ValueError, match="snicar_lap_opt"):
        _require_config_match(old, on, "snow.npz")


def test_lap_options_refuse_without_the_snow_solver():
    from woof.config import RunConfig
    from woof.core.snow_papers import lap_settings
    cfg = RunConfig(nx=6, ny=4, nz=5, dx=3000, dy=3000, ztop=12000, dt=10, run_seconds=60,
                    sf_surface_physics=4, opt_alb=2, snicar_lap_opt=1)
    with pytest.raises(ValueError, match="opt_alb=3"):
        lap_settings(cfg)


def test_deposition_channels_conserve_source_dust_mass():
    from woof.snow_lap_deposition import channels, overlap_matrix, source_row
    row = source_row("geos-fp")
    frac = overlap_matrix([[2 * a, 2 * b] for a, b in row["dust"]["radius_um"]],
                          [[0.1, 1], [1, 2.5], [2.5, 5], [5, 10], [10, 100]])
    assert np.allclose([sum(r) for r in frac], 1.0)
    ch = {c["key"]: c for c in channels(row)}
    assert len(ch) == 18
    dry = {}
    for k in range(1, 6):
        for t in ch[f"dust{k}_dry"]["terms"]:
            dry[t["name"]] = dry.get(t["name"], 0) + t["coef"]
    assert all(abs(v - 1) < 1e-12 for v in dry.values()) and len(dry) == 10


@requires_gpu
def test_table_top_layer_initial_ratio_loads_only_the_surface_layer(tmp_path):
    import cupy as cp, json
    from woof.core.snow_papers import evaluate, LAP_WORDS, LAP_SPECIES
    table = {"schema": "gpuwm-snow-lap-table-v1", "units": "kg m-2 s-1", "initial_mode": "top_layer_mass_ratio",
             "species": {k: {"dry": 0.0, "wet": 0.0, "initial_top_layer_kg_kg": 1e-3 if k == "dust3" else 0.0}
                         for k in LAP_SPECIES}}
    path = tmp_path/"table.json"; path.write_text(json.dumps(table))
    n = 1
    x, layers = _snow_inputs(n), _layers(n)
    state = _state(n, layers)
    lap = cp.zeros((n, LAP_WORDS), dtype=cp.float32)
    dep = cp.zeros((n, 27), dtype=cp.float32); dep[0, 18 + 6] = 1e-3
    opts = {"snicar_lap_opt": 1, "snicar_lap_table": str(path)}
    lap_new = evaluate(x, cp.ones(n, dtype=cp.int32), layers=layers, state=state, options=opts,
                       return_fluxes=True, lap=lap, deposition=dep)[3]
    h = cp.asnumpy(lap_new)[0]
    # 1 mg/g in the 20 kg/m2 top layer = 0.02 kg/m2 of 2.5-5 um dust there only.
    assert h[6] == pytest.approx(0.02, rel=1e-6)
    assert h[9 + 6] == 0 and h[18 + 6] == 0
    assert h[29 + 6] == pytest.approx(0.02, rel=1e-6)

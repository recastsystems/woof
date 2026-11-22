"""clean-snicar-layers: Toon 1989 solver and five-band mode for opt_alb=3.

Re-implemented from published descriptions, not bitwise to upstream.
Provenance: docs/PROVENANCE-clean-snicar-layers.md.
"""
import numpy as np
import pytest
from conftest import requires_gpu

SQ3 = 3.0 ** 0.5


def _gammas(w, g, mu, vis):
    """Toon et al. 1989 Table 1, written independently of the kernel."""
    if vis:   # Eddington
        g1 = (7 - w * (4 + 3 * g)) / 4
        g2 = -(1 - w * (4 - 3 * g)) / 4
        g3 = (2 - 3 * g * mu) / 4
    else:     # hemispheric mean; beam source from quadrature gamma3
        g1 = 2 - w * (1 + g)
        g2 = w * (1 - g)
        g3 = (1 - SQ3 * g * mu) / 2
    return g1, g2, g3, 1 - g3


def _ode_reference(layers, ground, mu, vis, beam, steps=4000):
    """Integrate Toon eqs 1-2 (the two-stream ODEs) by RK4 and shooting.

    dFup/dtau = g1 Fup - g2 Fdn - g3 w piFs e^(-tau/mu),
    dFdn/dtau = g2 Fup - g1 Fdn + g4 w piFs e^(-tau/mu),  piFs = 1/mu.
    The solution is linear in the unknown top upward flux, so two sweeps fix
    the bottom condition Fup = Rs (Fdn + direct). Returns the albedo and the
    per-layer plus ground absorption.
    """
    scaled = []
    for tau, a, g in layers:
        w = 1 - a
        s = 1 - w * g * g
        scaled.append((tau * s, min(1 - a / s, 1 - 1e-12), g / (1 + g)))

    def sweep(up0):
        y = np.array([up0, 0.0 if beam else 1.0])
        tc = 0.0
        tops = []
        for ts, w, g in scaled:
            g1, g2, g3, g4 = _gammas(w, g, mu, vis)
            src = (w / mu) if beam else 0.0

            def rhs(t, y):
                e = src * np.exp(-t / mu)
                return np.array([g1 * y[0] - g2 * y[1] - g3 * e,
                                 g2 * y[0] - g1 * y[1] + g4 * e])
            tops.append((tc, y.copy()))
            h = ts / steps
            t = tc
            for _ in range(steps):
                k1 = rhs(t, y)
                k2 = rhs(t + h / 2, y + h / 2 * k1)
                k3 = rhs(t + h / 2, y + h / 2 * k2)
                k4 = rhs(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            tc += ts
        tops.append((tc, y.copy()))
        return tops

    def residual(tops):
        tc, y = tops[-1]
        direct = np.exp(-tc / mu) if beam else 0.0
        return y[0] - ground * (y[1] + direct)
    t0, t1 = sweep(0.0), sweep(1.0)
    r0, r1 = residual(t0), residual(t1)
    x = -r0 / (r1 - r0)
    tops = [(a[0], a[1] + x * (b[1] - a[1])) for a, b in zip(t0, t1)]
    net = []
    for tc, y in tops:
        direct = np.exp(-tc / mu) if beam else 0.0
        net.append(y[1] + direct - y[0])
    absorbed = [net[k] - net[k + 1] for k in range(len(scaled))] + [net[-1]]
    return tops[0][1][0], np.array(absorbed)


def _probe(cases):
    import cupy as cp
    from woof.core.snow_papers import _module
    x = cp.asarray(np.asarray(cases, dtype=np.float64))
    y = cp.empty((len(cases), 10), dtype=cp.float64)
    _module(cp.cuda.Device().id).get_function("snow_column_probe")(
        ((len(cases) + 31) // 32,), (32,), (x, y, np.int32(len(cases))))
    return cp.asnumpy(y)


def _case(layers, ground, mu, vis, solver):
    row = [len(layers)] + [0.0] * 9 + [ground, mu, 1.0 if vis else 0.0, solver]
    for j, (tau, a, g) in enumerate(layers):
        row[1 + 3 * j:4 + 3 * j] = [tau, a, g]
    return row


@requires_gpu
def test_toon_hand_computed_conservative_slab():
    # Hand computation from the two-stream equations (Toon 1989 eqs 1-2) in
    # the conservative limit: w=1 makes the hemispheric-mean coefficients
    # g1=g2=1-g, and diffuse light on one layer over a black surface then has
    # R = g1*tau/(1+g1*tau). With g=0: tau=1 gives R=1/2 (T=1/2, nothing
    # absorbed in the layer), tau=3 gives R=3/4.
    y = _probe([_case([(1.0, 0.0, 0.0)], 0.0, 0.5, False, 2),
                _case([(3.0, 0.0, 0.0)], 0.0, 0.5, False, 2)])
    np.testing.assert_allclose(y[0, 1], 0.5, atol=1e-6)
    np.testing.assert_allclose(y[0, 6], 0.0, atol=1e-6)
    np.testing.assert_allclose(y[0, 7], 0.5, atol=1e-6)
    np.testing.assert_allclose(y[1, 1], 0.75, atol=1e-6)
    print("hand-computed conservative slab", y[:, [1, 6, 7]].tolist())


@requires_gpu
def test_toon_matches_independent_ode_integration():
    # Three layers of snow-like optics over soil, beam and diffuse, VIS
    # (Eddington) and NIR (hemispheric mean), including a thin top layer.
    stacks = [
        [(0.8, 2e-6, 0.89), (4.0, 5e-6, 0.89), (9.0, 1e-5, 0.88)],
        [(0.3, 3e-3, 0.86), (1.5, 8e-3, 0.87), (2.5, 2e-2, 0.88)],
        [(2.0, 0.08, 0.90)],
    ]
    cases, refs = [], []
    for layers in stacks:
        for mu in (0.9, 0.35):
            for vis in (True, False):
                cases.append(_case(layers, 0.15, mu, vis, 2))
                refs.append((layers, mu, vis))
    y = _probe(cases)
    for row, (layers, mu, vis) in zip(y, refs):
        n = len(layers)
        for beam, col, abcol in ((True, 0, 2), (False, 1, 6)):
            r, ab = _ode_reference(layers, 0.15, mu, vis, beam, steps=3000)
            np.testing.assert_allclose(row[col], r, atol=2e-7, err_msg=str((layers, mu, vis, beam)))
            np.testing.assert_allclose(row[abcol:abcol + n + 1], ab, atol=2e-7)
            np.testing.assert_allclose(row[col] + row[abcol:abcol + 4].sum(), 1.0, atol=1e-9)


@requires_gpu
def test_ad_routine_reproduces_inline_slab_and_add():
    # The dispatch routine must be the same adding-doubling the inline 480-bin
    # path uses (slab + add): compare with the existing snow_ad_probe, which
    # returns one tau=5 layer (a) and two tau=2.5 layers added (b).
    import cupy as cp
    from woof.core.snow_papers import _module
    q = cp.asarray([[5.0, 1e-3, 0.88, 0.6]], dtype=cp.float64)
    ref = cp.empty((1, 10), dtype=cp.float64)
    _module(cp.cuda.Device().id).get_function("snow_ad_probe")((1,), (32,), (q, ref, np.int32(1)))
    ref = cp.asnumpy(ref)[0]
    y = _probe([_case([(5.0, 1e-3, 0.88)], 0.0, 0.6, True, 1),
                _case([(2.5, 1e-3, 0.88), (2.5, 1e-3, 0.88)], 0.0, 0.6, True, 1)])
    np.testing.assert_allclose(y[0, :2], [ref[0], ref[2]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(y[1, :2], [ref[4], ref[6]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(y[:, 0] + y[:, 2:6].sum(axis=1), 1.0, atol=1e-12)


def _column(n=1, radius=100.0, swe=100.0, mu=0.7):
    import cupy as cp
    x = cp.zeros((n, 56), dtype=cp.float32)
    x[:, 24] = 60; x[:, 25] = mu; x[:, 28] = 263.15; x[:, 30] = swe / 250.0
    x[:, 31] = 1; x[:, 34] = swe; x[:, 42] = -radius
    x[:, 5:9] = .15; x[:, 52:56] = 100
    layers = cp.zeros((n, 3, 4), dtype=cp.float32)
    layers[:, :, 0] = cp.asarray([.1, .3, .6], dtype=cp.float32) * swe
    layers[:, :, 2] = cp.asarray([.1, .3, .6], dtype=cp.float32) * swe / 250.0
    layers[:, :, 3] = 263.15
    return x, layers


@requires_gpu
@pytest.mark.parametrize("radius", [100.0, 1000.0])
def test_toon_against_ad_as_dang_2019_reports(radius):
    # Dang et al. 2019 abstract and sect 5.2, against 16-stream DISORT where
    # adding-doubling is accurate: the Toon-based two-stream underestimates
    # diffuse VIS albedo and overestimates diffuse NIR (up to about 0.04);
    # direct errors stay within about 0.01 below 50 deg zenith. The VIS gap
    # (about 0.02 in the paper) grows as grains grow and snow thins; on this
    # deep pack it is small, so only its sign is asserted.
    import cupy as cp
    from woof.core.snow_papers import evaluate
    x, layers = _column(radius=radius)
    ist = cp.ones(1, dtype=cp.int32)
    out = {}
    for solver in (1, 2):
        y, s, a = evaluate(x, ist, layers=layers, return_fluxes=True,
                           options=dict(snicar_solver_opt=solver))
        np.testing.assert_allclose(cp.asnumpy(cp.sum(a, axis=1) + y[:, :4]), 1, atol=3e-6)
        out[solver] = cp.asnumpy(y[0, :4]).astype(float)
    d = out[2] - out[1]
    print("radius", radius, "AD", out[1].tolist(), "Toon", out[2].tolist(), "Toon-AD", d.tolist())
    assert abs(d[0]) < 0.01 and abs(d[1]) < 0.015
    assert -0.03 < d[2] < 0.0
    assert 0.0 < d[3] < 0.05


@requires_gpu
def test_five_band_mode_tracks_480_bins_and_conserves_energy():
    import cupy as cp
    from woof.core.snow_papers import evaluate
    ist = cp.ones(3, dtype=cp.int32)
    x, layers = _column(n=3)
    x[1, 42] = -500
    x[2, 34] = 2; x[2, 30] = .008
    layers[2, :, 0] *= .02; layers[2, :, 2] *= .02
    results = {}
    for solver in (1, 2):
        for bands in (2, 1):   # snicar_bandnumber_opt: 2 = 480 bins, 1 = five bands
            y, s, a = evaluate(x, ist, layers=layers, return_fluxes=True,
                               options=dict(snicar_solver_opt=solver, snicar_bandnumber_opt=bands))
            np.testing.assert_allclose(cp.asnumpy(cp.sum(a, axis=1) + y[:, :4]), 1, atol=3e-6)
            assert bool(cp.all(a >= 0))
            results[solver, bands] = cp.asnumpy(y[:, :4])
            heat = cp.asnumpy(a)
            # Layer-resolved NIR absorption: the top layer takes most.
            assert heat[0, 0, 1] > heat[0, 1, 1] > heat[0, 2, 1] >= 0
    for solver in (1, 2):
        diff = results[solver, 1] - results[solver, 2]
        print("solver", solver, "5-band minus 480-bin", diff.tolist())
        assert np.abs(diff[:, [0, 2]]).max() < 0.01
        assert np.abs(diff[:, [1, 3]]).max() < 0.03


@requires_gpu
def test_default_options_reproduce_previous_kernel_bytes():
    # Option-off identity at the kernel: the base source (ccbb7f5a284) and
    # this source give the same bytes on the default AD / 480-bin path.
    import subprocess
    import cupy as cp
    from pathlib import Path
    from woof.core.noahmp_kernel_sources import _compile_loaded_module
    from woof.core.snow_papers import evaluate, _device_table
    root = Path(__file__).resolve().parents[1]
    try:
        old = subprocess.run(["git", "-C", str(root), "show",
                              "ccbb7f5a284:woof/core/kernels/snow_papers.cu"],
                             check=True, capture_output=True, text=True).stdout
    except Exception:
        pytest.skip("base commit not in this checkout")
    module = _compile_loaded_module(old, ("-std=c++17", "--ftz=false"), "snow_papers_base")
    rng = np.random.default_rng(7)
    n = 64
    x, layers = _column(n=n)
    x[:, 25] = cp.asarray(rng.uniform(-.2, 1, n), dtype=cp.float32)
    x[:, 42] = cp.asarray(-rng.uniform(40, 1400, n), dtype=cp.float32)
    layers[:, :, 0] *= cp.asarray(rng.uniform(0, 2, (n, 3)), dtype=cp.float32)
    ist = cp.ones(n, dtype=cp.int32)
    state = cp.zeros((n, 21), dtype=cp.float32)
    y, carried, a = evaluate(x, ist, layers=layers, state=state, return_fluxes=True)
    forcing = cp.zeros((n, 6), dtype=cp.float32)
    forcing[:, 5] = x[:, 28]
    ice = cp.zeros((n, 4, 3), dtype=cp.float32)
    out = cp.empty((n, 6), dtype=cp.float32)
    nxt = cp.empty((n, 21), dtype=cp.float32)
    heat = cp.empty((n, 7, 4), dtype=cp.float32)
    module.get_function("snow_papers_albedo")((n,), (64,), (
        x, ist, layers, state, forcing, _device_table(cp.cuda.Device().id), ice,
        out, nxt, heat, np.int32(n), np.int32(2)))
    assert cp.asnumpy(y).tobytes() == cp.asnumpy(out).tobytes()
    assert cp.asnumpy(carried).tobytes() == cp.asnumpy(nxt).tobytes()
    assert cp.asnumpy(a).tobytes() == cp.asnumpy(heat).tobytes()

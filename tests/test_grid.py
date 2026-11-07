# tests/test_grid.py
import numpy as np
from woof.core import constants as c
from woof.core.grid import make_vertical_coord, make_base_state, rebalance_hydrostatic

def theta_const(z):
    return np.full_like(np.asarray(z, float), 300.0)

def test_coord_conventions():
    vc = make_vertical_coord(64)
    assert vc.znw[0] == 1.0 and vc.znw[-1] == 0.0
    assert np.all(vc.dnw < 0)
    np.testing.assert_allclose(vc.znu, 0.5 * (vc.znw[:-1] + vc.znw[1:]))

def test_base_state_discrete_balance():
    vc = make_vertical_coord(64)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    # the recurrence must hold to round-off by construction
    resid = (b.phb[1:] - b.phb[:-1]) + vc.dnw * b.mub * b.alb
    assert np.max(np.abs(resid)) < 1e-8
    # and heights must be physically sane for isentropic 300K atmosphere
    z_top_model = b.phb[-1] / c.G
    assert abs(z_top_model - 6400.0) < 50.0

def test_rebalance_matches_base_when_unperturbed():
    vc = make_vertical_coord(32)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    th3 = np.broadcast_to(b.thb[:, None, None], (32, 1, 8)).copy()
    ph3 = rebalance_hydrostatic(th3, b.mub, vc, p_surf=1.0e5)
    np.testing.assert_allclose(ph3[:, 0, 0], b.phb, atol=1e-6)


def test_base_theta_over_constant_terrain_is_the_whole_array_evaluation():
    """A constant terrain evaluates the sounding once per column height and
    broadcasts; the words must be the whole-array evaluation's, and the
    sounding must see one column, not ny*nx of them (the tile-buffer cost
    the development branch profile could not attribute)."""
    from woof.core.grid import _base_theta
    seen = []

    def sounding(z):
        seen.append(np.shape(z))
        return 300.0 + 0.003 * np.asarray(z, dtype=np.float64)

    z = np.linspace(50.0, 15000.0, 20)[:, None, None] + np.zeros((20, 3, 4))
    const = np.full((3, 4), 120.0)
    fast = _base_theta(sounding, z, const)
    assert seen == [(20,)]
    whole = sounding(z)
    np.testing.assert_array_equal(fast, whole)
    assert fast.shape == (20, 3, 4) and fast.flags.c_contiguous
    varied = const.copy()
    varied[1, 2] = 121.0
    seen.clear()
    np.testing.assert_array_equal(_base_theta(sounding, z, varied), whole)
    assert seen == [(20, 3, 4)]
    seen.clear()
    _base_theta(sounding, z[:, 0, 0], None)
    assert seen == [(20,)]

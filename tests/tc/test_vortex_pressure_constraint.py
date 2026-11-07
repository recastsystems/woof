"""CUDA algebra checks, not atmospheric or forecast evidence."""
import pytest

cp = pytest.importorskip('cupy')
from woof.tc.vortex import pressure_gain_limit


def fit(moved, axis, quadratic, linear, mask, floor):
    values=[cp.asarray([v],dtype=cp.float64) for v in (moved,axis,quadratic,linear)]
    return float(pressure_gain_limit(*values,cp.asarray([mask],dtype=cp.uint8),
                                    minimum_pressure_pa=floor))


def test_pressure_and_coriolis_bound():
    # P(g)=100000-1000*g^2-100*g reaches 95800 at g=2.
    assert fit([99000.],[-1000.],[1.],[.1],[1],95800.) == pytest.approx(2.)


def test_all_cells_and_ownership_are_respected():
    assert fit([99000.,99000.,0.],[-1000.,-1000.,0.],
               [1.,4.,0.],[0.,0.,0.],[1,1,0],96000.) == pytest.approx(1.)


def test_retained_low_cannot_be_hidden():
    with pytest.raises(ValueError,match='environment/asymmetry'):
        fit([90000.],[-1000.],[1.],[.1],[1],95800.)


def test_positive_pressure_anomaly_does_not_limit_gain():
    assert fit([100000.],[1000.],[1.],[.1],[1],95800.) == float('inf')


def test_mixed_sign_retains_connected_feasible_interval():
    # P-floor = g^2-3g+2: the admitted interval from zero ends at 1.
    assert fit([100003.],[1.],[1.],[-3.],[1],100000.) == pytest.approx(1.)

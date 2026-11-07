"""Carried condensate must move into dry cells without negative mass/moments."""
import pytest
cp = pytest.importorskip('cupy')
from woof.tc.vortex import Patch, relocate_constituents


def test_dry_destination_preserves_positive_paired_moments_and_outer_bits():
    grid = Patch(10., -100., .1, .1, 201, 201)
    y, x = cp.meshgrid(cp.arange(201), cp.arange(201), indexing='ij')
    mass = cp.where((x-90)**2+(y-100)**2 < 12**2, 1.e-3, 0.)
    fields = cp.ascontiguousarray(cp.stack((mass, mass*2.e8)), dtype=cp.float64)
    moved = relocate_constituents(fields, grid, old_centre=(20.,-91.),
        new_centre=(20.,-88.), source_radius_m=300000., target_radius_m=240000.,
        map_coefficients=(.8,0.))
    assert bool(cp.isfinite(moved).all())
    assert float(moved.min()) >= 0.
    assert float(moved[0].max()) <= .001
    assert float(moved[0,100,120]) == pytest.approx(.001)
    assert float(moved[0,100,90]) == 0.
    cp.testing.assert_allclose(moved[1], moved[0]*2.e8, rtol=1.e-14, atol=1.e-12)
    cp.testing.assert_array_equal(moved[:,:,:20], fields[:,:,:20])
    unchanged = relocate_constituents(fields, grid, old_centre=(20.,-91.),
        new_centre=(20.,-91.), source_radius_m=300000., target_radius_m=300000.,
        map_coefficients=(1.,0.))
    cp.testing.assert_array_equal(unchanged, fields)

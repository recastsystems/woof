"""Host geometry admission only; these checks do not qualify a TC forecast."""
import math
import unittest
from woof.tc.vortex import Patch, radial_map, _disk_inside


class VortexGeometry(unittest.TestCase):
    def test_map_hits_both_diagnosed_radii(self):
        a,b=radial_map(source_rmw=60000.,target_rmw=50000.,source_outer=500000.,target_outer=450000.)
        self.assertAlmostEqual(a*60000.+.5*b*60000.**2,50000.)
        self.assertAlmostEqual(a*500000.+.5*b*500000.**2,450000.)
        self.assertGreater(min(a,a+b*500000.),0.)

    def test_identity_map(self):
        self.assertEqual(radial_map(source_rmw=40000.,target_rmw=40000.,source_outer=500000.,target_outer=500000.),(1.,0.))

    def test_folding_map_is_refused(self):
        with self.assertRaisesRegex(ValueError,"folds"):
            radial_map(source_rmw=40000.,target_rmw=300000.,source_outer=500000.,target_outer=500000.)

    def test_dateline_disk_uses_wrapped_longitude(self):
        patch=Patch(-20.,150.,1.,1.,61,61)
        patch.validate()
        _disk_inside(patch,10.,-179.,300000.,halo_deg=11.)

    def test_filter_halo_is_required(self):
        patch=Patch(0.,-100.,1.,1.,31,31)
        _disk_inside(patch,10.,-85.,200000.)
        with self.assertRaisesRegex(ValueError,"halo"):
            _disk_inside(patch,10.,-85.,200000.,halo_deg=11.)


if __name__ == '__main__':
    unittest.main()

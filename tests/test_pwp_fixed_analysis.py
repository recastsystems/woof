"""A dated analysis can initialize an uncoupled comparison without ocean state."""
import sys
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.mark.parametrize('start,continuing', [(True, False), (False, False), (True, True)])
def test_fixed_analysis_surface_only(monkeypatch, start, continuing):
    from woof.core.surface_ocean import initialize_ocean
    from woof.ocean.mixed_layer import analysis
    monkeypatch.setitem(sys.modules, 'cupy', np)
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return dict(om_active=np.array([[1, 0, 0]], np.float32),
                    om_tmp=np.array([[[299, 0, 0]]], np.float32))
    monkeypatch.setattr(analysis, 'load', load)
    fields = dict(tsk=np.full((1, 3), 301, np.float32),
                  sst=np.full((1, 3), 301, np.float32),
                  xland=np.array([[2, 2, 1]], np.float32))
    names = set(fields)
    cfg = SimpleNamespace(sf_ocean_physics=0, ocean_analysis='dated.nc',
                          ocean_analysis_source='hycom', ocean_levels=40)
    initialize_ocean(fields, cfg, start=start, continuing=continuing,
                     latitude=np.zeros((1, 3)), longitude=np.zeros((1, 3)),
                     expected_time='2022-09-26T00:00:00')
    assert set(fields) == names
    expected = [[299, 301, 301]] if start and not continuing else [[301]*3]
    np.testing.assert_array_equal(fields['tsk'], expected)
    np.testing.assert_array_equal(fields['sst'], expected)
    assert len(calls) == int(start and not continuing)

def test_fixed_analysis_retains_checkpoint_identity():
    from woof.surface_ocean import drop_inert_ocean
    value = dict(sf_ocean_physics=0, ocean_analysis='dated.nc',
                 ocean_analysis_source='hycom', ocean_levels=40, omdt=1)
    drop_inert_ocean(value)
    assert value == dict(ocean_analysis='dated.nc', ocean_analysis_source='hycom', ocean_levels=40)
    absent = dict(sf_ocean_physics=0, ocean_analysis='', ocean_levels=30)
    drop_inert_ocean(absent)
    assert absent == {}

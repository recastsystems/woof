"""Missing diagnostic support must not hide bad active-column data."""
import numpy as np
import pytest
from woof.hex.ocean_tc import validate_snapshot_diagnostics


def snapshot():
    return dict(ocean_pwp_active=np.array([0., 1., 1.]),
                ocean_ohc26=np.array([np.nan, 50., 120.]),
                ocean_d26=np.array([np.nan, 80., np.nan]),
                ocean_d26_censored=np.array([np.nan, 0., 1.]),
                ocean_ild=np.array([np.nan, 40., np.nan]),
                ocean_mld=np.array([np.nan, 30., 180.]),
                ocean_barrier_layer=np.array([np.nan, 10., np.nan]))


def test_inactive_and_censored_diagnostics_are_admitted():
    assert len(validate_snapshot_diagnostics(snapshot())) == 6
    assert validate_snapshot_diagnostics({}) == set()


@pytest.mark.parametrize('name,index,value', [
    ('ocean_ohc26',1,np.nan), ('ocean_d26',1,np.nan),
    ('ocean_d26_censored',2,0.), ('ocean_mld',2,np.inf),
    ('ocean_barrier_layer',1,0.), ('ocean_ild',0,0.),
])
def test_invalid_diagnostics_still_fail(name,index,value):
    arrays=snapshot();arrays[name][index]=value
    with pytest.raises(FloatingPointError):
        validate_snapshot_diagnostics(arrays)

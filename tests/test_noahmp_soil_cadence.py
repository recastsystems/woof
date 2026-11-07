"""Native integer cadence and the state needed between soil updates."""
import math
import pytest
from woof.core.noahmp_soil_cadence import cadence, shapes, validate

@pytest.mark.parametrize("interval,dt,expected", [(0.,60.,1),(29.,60.,1),(90.,60.,2),(150.,60.,3),(180.,60.,3)])
def test_native_positive_nint_cadence(interval,dt,expected):
    for step in range(1,expected*3+1):
        n,update,reset=cadence(interval,dt,step)
        assert n==expected
        assert update==(step%expected==0)
        assert reset==(expected==1 or step%expected==1)

@pytest.mark.parametrize("bad", [-1., math.inf, math.nan, True, 24])
def test_invalid_soil_cadence_rejected(bad):
    with pytest.raises(ValueError,match="finite nonnegative float"):
        validate(bad)

def test_inactive_has_no_additional_carriers_active_has_native_twelve_planes():
    assert shapes(0.,(5,7))=={}
    active=shapes(180.,(5,7))
    assert len(active)==9
    assert active["acc_etrani"]==(4,5,7)
    assert sum(math.prod(value) for value in active.values())==12*5*7

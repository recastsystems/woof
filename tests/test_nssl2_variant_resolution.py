"""WRF public and internally resolved three-moment selector spellings."""
import pytest
from woof.core.nssl2_contract import resolve_nssl2_mode, require_ported_nssl2_mode

@pytest.mark.parametrize("hail,three,sixth", [(1,1,2),(1,2,2),(0,1,1),(0,2,2)])
def test_public_and_resolved_selectors_follow_registry_packages(hail,three,sixth):
    mode=resolve_nssl2_mode(nssl_hail_on=hail,nssl_3moment=three)
    assert mode.sixth_moments==sixth
    assert mode.two_moment
    assert ("qzh" in mode.transported_fields)==(sixth==2)
    with pytest.raises(ValueError,match="three-moment.*not.*ported"):
        require_ported_nssl2_mode(mode)

def test_public_three_moment_switch_forces_numbers_after_defaults():
    mode=resolve_nssl2_mode(nssl_2moment_on=0,nssl_hail_on=1,nssl_3moment=1)
    assert mode.two_moment and mode.sixth_moments==2

def test_resolved_three_moment_switch_does_not_force_numbers():
    with pytest.raises(ValueError,match="allocated.*sixth|sixth.*allocated"):
        resolve_nssl2_mode(nssl_2moment_on=0,nssl_hail_on=2,nssl_3moment=2)

def test_wrf_default_order_still_catches_the_inactive_hail_number_read():
    with pytest.raises(ValueError,match="unallocated hail-number"):
        resolve_nssl2_mode(nssl_2moment_on=0,nssl_3moment=1)

def test_default_mode_is_unchanged():
    mode=resolve_nssl2_mode()
    assert (mode.two_moment,mode.hail,mode.predicted_ccn,mode.density_moments,mode.sixth_moments)==(True,True,True,2,0)

"""The combined WDM family retains each scheme's mass and number inventory."""
import pytest
from woof.config import RunConfig, validate_run_config
from woof.core.wdm6_constants import wdm_scheme, WDM_SCHEMES

@pytest.mark.parametrize("selector,graupel,hail", [(14,False,False),(16,True,False),(26,True,True)])
def test_combined_family_preserves_the_selected_species(selector,graupel,hail):
    row=wdm_scheme(selector)
    assert row.graupel is graupel
    assert row.hail is hail
    assert ("qg" in row.mass_species) is graupel
    assert ("qh" in row.mass_species) is hail
    assert row.number_species == ("nn","nc","nr")
    cfg=RunConfig(nx=8,ny=6,nz=40,dx=1000.,dy=1000.,ztop=10000.,dt=1.,
                  run_seconds=10.,moist=True,moist_cq=True,mp_physics=selector)
    assert validate_run_config(cfg).mp_physics == selector
    from woof.core.device_inventory import state_array_shapes
    from woof.core.nest_fields import nest_field_kinds
    from woof.core.mynn_pbl_runtime import mynn_flag_qs
    shapes=state_array_shapes(cfg)
    kinds=nest_field_kinds(cfg)
    for name in row.mass_species[3:]+row.number_species:
        assert shapes[name] == (40,6,8)
        assert shapes[name+"0"] == (40,6,8)
        assert name in kinds
    assert mynn_flag_qs(selector)
    assert ("qh" in kinds) is hail

def test_the_family_has_all_three_native_selectors():
    assert set(WDM_SCHEMES) == {14,16,26}

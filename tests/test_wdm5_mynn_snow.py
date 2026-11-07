"""WRF Registry.EM_COMMON:3030 declares snow for WDM5's package."""
from woof.core.mynn_pbl_runtime import mynn_flag_qs
from woof.config import RunConfig
from woof.core.preflight import state_array_shapes
from woof.physics_registry import physics_registry

def test_wdm5_allocated_snow_reaches_the_mynn_flag():
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10., moist=True, moist_cq=True, mp_physics=14)
    assert 'qs' in state_array_shapes(cfg)
    assert mynn_flag_qs(14) is True

def test_wdm5_snow_classification_is_published():
    species = physics_registry()['components']['pbl']['options']['mynn']['extensions']['supplied_moisture_species']
    assert 14 in species['flag_qs_true_microphysics_selectors']
    assert 14 not in species['flag_qs_false_microphysics_selectors']

def test_existing_snow_flags_keep_their_wrf_packages():
    for selector in (6, 8, 9, 10, 16, 18, 28):
        assert mynn_flag_qs(selector) is True
    for selector in (0, 1, 50):
        assert mynn_flag_qs(selector) is False

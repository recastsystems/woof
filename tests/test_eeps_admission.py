"""EEPS admission at public config doors and transported-state contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from woof.config import EEPS_PBL_SCHEME, RunConfig, validate_run_config

ROOT = Path(__file__).resolve().parents[1]

def cfg(**changes):
    values = dict(nx=16, ny=16, nz=30, dx=3000., dy=3000., dt=15.,
                  ztop=15000., run_seconds=60., moist=True, mp_physics=8,
                  bl_pbl_physics=EEPS_PBL_SCHEME, sf_sfclay_physics=1,
                  sf_surface_physics=2)
    values.update(changes)
    return RunConfig(**values)

@pytest.mark.parametrize('surface', [1, 2, 5, 91])
@pytest.mark.parametrize('microphysics', [1, 6, 8])
def test_supported_surface_and_moisture_packages(surface, microphysics):
    validate_run_config(cfg(sf_sfclay_physics=surface, mp_physics=microphysics))

@pytest.mark.parametrize('changes,reason', [
    ({'moist': False, 'mp_physics': 0}, 'pair would never move'),
    ({'sf_sfclay_physics': 0, 'sf_surface_physics': 0}, 'divides by WSPD'),
    ({'sf_urban_physics': 2}, 'urban momentum and heat exchange'),
    ({'sf_urban_physics': 3}, 'urban momentum and heat exchange'),
])
def test_refusals_name_the_breakage(changes, reason):
    with pytest.raises(ValueError, match=reason):
        validate_run_config(cfg(**changes))

def test_rebuild_persists_selector_and_carried_scalar_metadata():
    from tools.build_registry import build
    registry = json.loads((ROOT/'woof/physics_registry_v2.json').read_text())
    rebuilt = build(registry)
    option = rebuilt['components']['pbl']['options']['eeps']
    assert option['selectors'] == {'bl_pbl_physics': 16}
    assert option['extensions']['advected_scalars']['names'] == ['pek_adv','pep_adv']

def test_carried_scalars_are_not_conditional_on_ice_species():
    from woof.core.moist import extra_moist_species
    state = SimpleNamespace(qi=None, pek_adv=object(), pep_adv=object())
    assert extra_moist_species(state) == ('pek_adv','pep_adv')
    assert extra_moist_species(SimpleNamespace(qi=None)) == ()

def test_scheme_identity_and_import_mapping_are_registered():
    from woof.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    from woof.namelist_import import _BL_MAP
    assert _BL_MAP[16][0] == 16
    assert PBL_ALGORITHM_IDENTITIES[16].endswith('radsum-zero')


def test_preflight_prices_both_owned_and_transported_closure_pairs():
    from woof.core.preflight import physics_array_shapes, state_array_shapes
    config = cfg(nx=16, ny=12, nz=30)
    shape = (30, 12, 16)
    physics = physics_array_shapes(config)
    state = state_array_shapes(config)
    assert {name: physics["fields/" + name] for name in ("pek_pbl", "pep_pbl")} == {
        "pek_pbl": shape, "pep_pbl": shape}
    assert {name: state[name] for name in ("pek_adv", "pep_adv", "pek_adv0", "pep_adv0")} == {
        name: shape for name in ("pek_adv", "pep_adv", "pek_adv0", "pep_adv0")}

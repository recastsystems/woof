"""CCN coefficient reachability, arithmetic setup and restart identity."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest

from woof.config import RunConfig, validate_run_config
from woof.core.nssl2_contract import nssl2_background_ccn_per_kg


def cfg(value=0.5e9, **kw):
    return RunConfig(nx=6, ny=6, nz=8, dx=1000., dy=1000., ztop=8000.,
                     dt=2., run_seconds=4., moist=True, mp_physics=18,
                     nssl_cccn=value, **kw)


@pytest.mark.parametrize('value', [0., -1e7, 1e7, 0.5e9, 5e10])
def test_cccn_values_reach_config_and_restart(value):
    from woof.io.restart import _nssl2_restart_contract_identity
    c=cfg(value)
    validate_run_config(c)
    assert nssl2_background_ccn_per_kg(c.nssl_cccn) == np.abs(np.float32(value)) / np.float32(1.225)
    identity=_nssl2_restart_contract_identity(c)
    assert identity['resolved_wrf_namelist_defaults']['nssl_cccn'] == value


def test_cccn_default_has_established_float_image():
    assert nssl2_background_ccn_per_kg().view(np.uint32) == np.float32(408163264.).view(np.uint32)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 1e100, True])
def test_undefined_coefficient_refuses_before_allocation(value):
    with pytest.raises((TypeError,ValueError),match='nssl_cccn'):
        validate_run_config(cfg(value))


def test_other_scheme_refuses_nondefault_coefficient():
    c=replace(cfg(1e7),mp_physics=6)
    with pytest.raises(ValueError,match='nssl_cccn.*requires.*18'):
        validate_run_config(c)


@pytest.mark.parametrize('value', [0.5e9, -0.5e9])
def test_native_abs_default_does_not_refuse_a_default_seed_transition(value):
    validate_run_config(cfg(value, nested=True, grid_id=2,
                           nest_microphysics_transition='mp8-to-mp18-mass-diagnosed-v1'))


def test_cross_scheme_transition_uses_the_target_ccn_seed():
    from woof.core.microphysics_transition import resolve_microphysics_transition
    child = cfg(1e7, nested=True, grid_id=2,
                nest_microphysics_transition='mp8-to-mp18-mass-diagnosed-v1')
    validate_run_config(child)
    parent = replace(cfg(0.5e9), mp_physics=8)
    contract = resolve_microphysics_transition(parent, child)
    assert contract.target_nssl_cccn == 1e7
    assert contract.receipt()['nssl2_background_ccn_per_kg'] == float(
        nssl2_background_ccn_per_kg(1e7))


def test_fingerprint_binds_only_nssl_coefficient():
    from woof.core.model import SCHEME_SCOPED_RUN_FIELDS
    assert 'nssl_cccn' in SCHEME_SCOPED_RUN_FIELDS[18]


def test_cold_start_seeds_nondefault_coefficient():
    from woof.core.state import DomainState
    c=cfg(1e7)
    state=DomainState(c,array_module=np)
    assert np.all(state.qnn == nssl2_background_ccn_per_kg(c.nssl_cccn))


def test_namelist_coefficient_import_preserves_value(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from woof.namelist_import import import_namelists
    text=INPUT_TEXT.replace('mp_physics = 10, 10,','mp_physics = 18, 18,')
    text=text.replace('&physics\n','&physics\n nssl_cccn = 1.0e7,\n')
    wps,inp=tmp_path/'namelist.wps',tmp_path/'namelist.input'
    wps.write_text(WPS_TEXT)
    inp.write_text(text)
    toml,_=import_namelists(wps,inp)
    assert 'nssl_cccn = 10000000' in toml


def test_nondefault_receipt_names_actual_coefficients_and_registry_defaults():
    from woof.core.nssl2_contract import nssl2_contract_receipt,DEFAULT_MODE
    r=nssl2_contract_receipt(DEFAULT_MODE,nssl_cccn=1e7)
    assert not r['is_default_lane']
    assert r['wrf_namelist_defaults']['nssl_cccn']==0.5e9
    assert r['coefficient_overrides']['nssl_cccn']==1e7

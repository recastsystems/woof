"""Admission failures seen on the six-hour combined-option release run."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from woof.config import (RunConfig, ConfigurationRefused,
                          microphysics_lookup_table_precondition,
                          validate_radiation_driver_options)

def test_missing_ishmael_is_reported_before_preparation(tmp_path, monkeypatch):
    monkeypatch.setenv('WOOF_ISHMAEL_TABLE_ROOT', str(tmp_path))
    sentence = microphysics_lookup_table_precondition(SimpleNamespace(mp_physics=55))
    assert 'ishmael-qi-qr.bin' in sentence and 'fetch-tables --ishmael' in sentence
    assert microphysics_lookup_table_precondition(SimpleNamespace(mp_physics=10)) is None

def test_hail_table_refusal_obeys_native_write_controls(tmp_path, monkeypatch):
    monkeypatch.setenv('WOOF_THOMPSON_HAIL_TABLE_ROOT', str(tmp_path))
    cfg = SimpleNamespace(mp_physics=38, force_read_thompson=False,
                          write_thompson_mp38table=False)
    assert 'write_thompson_mp38table=true' in microphysics_lookup_table_precondition(cfg)
    cfg.write_thompson_mp38table = True
    assert microphysics_lookup_table_precondition(cfg) is None
    cfg.force_read_thompson = True
    assert 'force_read_thompson=true' in microphysics_lookup_table_precondition(cfg)

def test_farms_is_admitted_after_its_native_port():
    cfg = RunConfig(nx=12, ny=12, nz=49, dx=12000, dy=12000, ztop=20000, dt=60, run_seconds=21600)
    validate_radiation_driver_options(replace(cfg, swint_opt=2))
    with pytest.raises(ConfigurationRefused, match='couple_farms requires swint_opt=2'):
        validate_radiation_driver_options(replace(cfg, swint_opt=1, couple_farms=True))

def test_split_goddard_selectors_price_the_local_frame():
    from woof.core.preflight import domain_kernel_modules, KERNEL_MAX_LOCAL_SIZE_BYTES
    cfg = RunConfig(nx=12, ny=12, nz=49, dx=12000, dy=12000, ztop=20000, dt=60, run_seconds=21600)
    for lw, sw in ((5,5), (5,7), (7,5)):
        d = SimpleNamespace(grid_id=1, run=replace(cfg, ra_physics=0, ra_lw_physics=lw, ra_sw_physics=sw))
        assert 'goddard' in domain_kernel_modules(d, prices_refl=False)
    assert KERNEL_MAX_LOCAL_SIZE_BYTES['goddard'] == 43664

def test_mp40_radius_band_uses_its_native_ice_size_limit():
    from woof.core.rrtmgp import effective_radius_bands
    assert effective_radius_bands('morrison_aero')['effi'] == (1.0,1200.0)
    assert effective_radius_bands('wsm6')['effi'] == (1.0,600.0)


def test_p3_two_category_native_rounded_endpoint_keeps_the_unit_guard():
    import numpy as np
    from woof.core.rrtmgp import effective_radius_bands
    single = effective_radius_bands("p3")["effi"][1]
    combined = effective_radius_bands("p3_2cat")["effi"][1]
    q = np.float32(2.3983697e-11)
    r = np.float32(0.021495)
    actual = np.float32(q / np.float32(q / r)) * np.float32(1e6)
    assert single < actual <= combined
    assert combined < single * 1.000001
    for scheme in ("p3", "p3_2cat"):
        bands = effective_radius_bands(scheme)
        for name in ("effi", "effs"):
            low, high = bands[name]
            assert not low <= np.float32(4.99) / 1000 <= high
            assert not low <= np.float32(25) * 1000 <= high

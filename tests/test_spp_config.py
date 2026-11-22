"""SPP selection requires the schemes that consume its member patterns."""
from dataclasses import replace

import pytest

from woof.config import RunConfig, validate_spp_config


def base_config():
    return RunConfig(nx=12, ny=10, nz=8, dx=3000., dy=3000.,
                     ztop=20000., dt=10., run_seconds=20.)


def test_default_spp_consumers_are_disabled():
    cfg = base_config()
    validate_spp_config(cfg)
    assert (cfg.spp_conv, cfg.spp_pbl, cfg.spp_lsm) == (0, 0, 0)


@pytest.mark.parametrize("name", ["spp_conv", "spp_pbl", "spp_lsm"])
@pytest.mark.parametrize("value", [True, False, -1, 2, 1.0, "1"])
def test_spp_flags_have_integer_boolean_contract(name, value):
    with pytest.raises(ValueError, match=f"{name} must be integer"):
        validate_spp_config(replace(base_config(), **{name: value}))


@pytest.mark.parametrize("name,selectors", [
    # GF SPP is GFDRV's; WRF v4.8.0's GFL call carries no pattern.
    ("spp_conv", {"cu_physics": 3, "cu_gf_version": "wrf_461"}),
    ("spp_pbl", {"bl_pbl_physics": 5, "sf_sfclay_physics": 5}),
    ("spp_lsm", {"sf_surface_physics": 3}),
])
def test_enabled_spp_refuses_a_scheme_without_a_consumer(name, selectors):
    cfg = replace(base_config(), **selectors, **{name: 1})
    validate_spp_config(cfg)
    for selector in selectors:
        with pytest.raises(ValueError, match="has no consumer"):
            validate_spp_config(replace(cfg, **{selector: 0}))

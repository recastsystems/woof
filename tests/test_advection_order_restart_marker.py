"""An omitted legacy marker resumes itself while a new order-2 stencil binds."""
from dataclasses import asdict

import pytest

from woof.config import RunConfig
from woof.io.restart import (
    RestartMismatchError, _drop_default_off_run_keys, _require_config_match,
)


def config(**kw):
    return RunConfig(nx=8, ny=8, nz=20, dx=3000., dy=3000., ztop=15000.,
                     dt=12., run_seconds=120., **kw)


@pytest.mark.parametrize("options", [{}, {"h_sca_adv_order": 2}, {"h_sca_adv_order": 5}])
def test_current_order_echo_resumes_its_own_stencil(options):
    cfg = config(**options)
    stored = asdict(cfg)
    _drop_default_off_run_keys(stored)
    _require_config_match(stored, cfg, "current-order")


def test_legacy_default_and_explicit_order_two_remain_different_models():
    stored = asdict(config())
    _drop_default_off_run_keys(stored)
    with pytest.raises(RestartMismatchError, match="_legacy_h_sca"):
        _require_config_match(stored, config(h_sca_adv_order=2), "legacy-order")

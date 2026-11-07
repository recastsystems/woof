"""A header written by this build resumes its own scalar transport identity."""
from dataclasses import replace
import pytest
from woof.io.restart import configuration_echo, _require_config_match, RestartMismatchError
from test_restart import _cfg

@pytest.mark.parametrize("order", [None, 2, 5])
def test_written_advection_identity_matches_live_configuration(order):
    cfg = _cfg(h_sca_adv_order=order)
    echo = configuration_echo(cfg)
    _require_config_match(echo, cfg, "self.npz")
    assert ("_legacy_h_sca" in echo) == (order == 2)

def test_explicit_second_order_cannot_resume_the_legacy_default():
    cfg = _cfg(h_sca_adv_order=None)
    with pytest.raises(RestartMismatchError, match="_legacy_h_sca"):
        _require_config_match(configuration_echo(cfg), replace(cfg, h_sca_adv_order=2, _legacy_h_sca=False), "legacy.npz")

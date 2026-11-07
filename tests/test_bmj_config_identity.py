"""Config reader and writer use one omission rule for the new option."""
import dataclasses
from pathlib import Path
import pytest
from woof.config import RunConfig
from woof.experiment import _public_config_value
from woof.io.restart import _drop_default_off_run_keys, _require_config_match, RestartMismatchError


def test_inactive_defaults_preserve_legacy_public_and_checkpoint_documents():
    cfg = RunConfig(6, 4, 5, 2000.0, 2000.0, 8000.0, 10.0, 0.0)
    stored = dataclasses.asdict(cfg)
    _drop_default_off_run_keys(stored)
    _require_config_match(stored, cfg, Path("legacy.npz"))
    public = _public_config_value(cfg)
    for name in NAMES:
        assert name not in stored
        assert name not in public


def test_consumed_nondefault_setting_stays_bound_on_restart():
    cfg = dataclasses.replace(RunConfig(6, 4, 5, 2000.0, 2000.0, 8000.0, 10.0, 0.0), cu_physics=SELECTOR, **MOVED)
    stored = dataclasses.asdict(cfg)
    _drop_default_off_run_keys(stored)
    for name, value in MOVED.items():
        assert stored[name] == value
    _require_config_match(stored, cfg, Path("active.npz"))
    with pytest.raises(RestartMismatchError):
        _require_config_match(stored, dataclasses.replace(cfg, **OTHER), Path("active.npz"))

NAMES = ('bmj_rad_feedback',)
SELECTOR = 2
MOVED = {'bmj_rad_feedback': True}
OTHER = {'bmj_rad_feedback': False}

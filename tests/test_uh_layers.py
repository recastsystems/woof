"""0-1/0-2/0-3 km updraft-helicity maxima (woof/core/uh_layers.py, uh_layers = 1).

The whole-run proof is a real forecast (receipt in the FOLD-READY line); these
pin the knob's contract: off allocates nothing, on rides nwp_diagnostics = 1,
the maxima go to history, reset on write and survive a checkpoint.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from woof.config import validate_run_config
from woof.core import uh_layers
from woof.core.uh_diag import reset_up_heli_max
from woof.io import restart
from test_restart import _cfg, _fill_setup, _shim_state

SLOTS = ("up_heli_max01", "up_heli_max02", "up_heli_max03")


def test_off_allocates_nothing(monkeypatch):
    state = _shim_state(_cfg(nwp_diagnostics=1), monkeypatch)
    assert all(state.existing_scratch(slot) is None for slot in SLOTS)
    assert uh_layers.owned_slots(state) == ()


def test_on_allocates_resets_and_reaches_history(monkeypatch):
    from woof.io.history_layout import live_state_history_fields as fields_of
    state = _shim_state(_cfg(nwp_diagnostics=1, uh_layers=1), monkeypatch)
    for slot in SLOTS:
        state.existing_scratch(slot)[...] = 50.0
    fields = fields_of(state)
    assert {"UP_HELI_MAX01", "UP_HELI_MAX02", "UP_HELI_MAX03"} <= set(fields)
    reset_up_heli_max(state)
    assert all((state.existing_scratch(slot) == 0.0).all() for slot in SLOTS)


def test_needs_nwp_diagnostics():
    with pytest.raises(ValueError, match="uh_layers = 1 needs nwp_diagnostics = 1"):
        validate_run_config(replace(_cfg(), uh_layers=1, nwp_diagnostics=0))


def test_checkpoint_carries_the_maxima(monkeypatch, tmp_path):
    cfg = _cfg(nwp_diagnostics=1, uh_layers=1)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    for index, slot in enumerate(SLOTS):
        source.existing_scratch(slot)[...] = np.float32(10 * (index + 1))
    path = restart.write_restart(tmp_path / "uh.npz", source, cfg)
    with np.load(path, allow_pickle=False) as archive:
        assert {f"scratch/{slot}" for slot in SLOTS} <= set(archive.files)


def test_low_level_uh_ensemble_fields_and_thresholds():
    """WOOF uh-layers ship (2026-11-02): 0-1 km probabilities at 10/20/40 (25/50/75 left them empty on the
    2024-05-06 outbreak), and enprod (the hex ensemble door) names uh01/uh02/uh03."""
    from woof.da import enprod
    from woof.ensemble.batch_products import DEFAULT_THRESHOLDS

    assert DEFAULT_THRESHOLDS["uh01"] == ("m2 s-2", (10.0, 20.0, 40.0))
    assert DEFAULT_THRESHOLDS["uh03"] == ("m2 s-2", (25.0, 50.0, 75.0))
    for name, variable in (("uh01", "UP_HELI_MAX01"), ("uh02", "UP_HELI_MAX02"), ("uh03", "UP_HELI_MAX03")):
        assert enprod.FIELDS[name].units == "m2 s-2"
        assert enprod.FIELDS[name].extract.__closure__[0].cell_contents == variable

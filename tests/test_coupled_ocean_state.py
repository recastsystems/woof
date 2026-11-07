"""Coupled checkpoints retain prognostic words through the shared adapter."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from woof.ocean.coupling import DeviceCoupler
from woof.ocean.checkpoint import checkpoint_owner, snapshot


def coupler(waves, surge=None):
    ocean = object.__new__(DeviceCoupler)
    ocean.alpha = np.ones(3); ocean.z0 = np.zeros(3)
    ocean.buffers = {"tau_x": np.ones(3)}
    ocean.waves = waves; ocean.surge = surge
    ocean.exchange = ocean.timeline = ocean.forcing = None
    ocean.prepared_identity = {}; ocean.interval = 60.
    ocean.roughness_feedback = False; ocean.grid = NS(shape=(3,))
    ocean.pending_seconds = 0.; ocean.last_time = None
    ocean.steps = ocean.atmosphere_steps = 0
    return ocean


def test_coupled_snapshot_keeps_prognostic_words_through_surface_adapter():
    spectrum = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    water = np.arange(32, dtype=np.float32)
    waves = NS(restart_state=lambda: {"arrays": {"spectrum": spectrum},
                                     "metadata": {"time_seconds": 0.}})
    surge = NS(restart_state=lambda: {"arrays": {"water": water},
                                     "metadata": {"time_seconds": 0.}})
    ocean = coupler(waves, surge)
    adapter = NS(checkpoint_target=ocean)
    saved = snapshot(checkpoint_owner(adapter))
    assert saved["arrays"]["waves/spectrum"] is spectrum
    assert saved["arrays"]["surge/water"] is water
    assert saved["arrays"]["forcing/tau_x"] is ocean.buffers["tau_x"]
    assert set(saved["metadata"]["components"]) == {"waves", "surge"}


def test_solver_without_restart_state_refuses_with_the_real_missing_contract():
    with pytest.raises(ValueError, match="waves solver lacks restart_state"):
        snapshot(coupler(NS()))

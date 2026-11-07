"""Native coupled forcing buffers and external rates have distinct units."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.ntiedtke import NewTiedtke, finish_advective_forcing
from woof.state_serialization_contract import WRF_MUTS_SLOT


@pytest.mark.parametrize("name,slot", [("gf_rthdynten", "rthften"),
                                       ("gf_rqvdynten", "rqvften")])
def test_native_forcing_decouples_completed_mass_before_next_consumer(name, slot):
    lane = np.arange(1, 13, dtype="f4").reshape(2, 2, 3)
    before = lane.copy()
    saved_mass = np.full((2, 3), 7, dtype="f4")
    state = SimpleNamespace(c1h=np.array([1, .5], dtype="f4"),
                            c2h=np.array([2, 3], dtype="f4"),
                            existing_scratch=lambda key: saved_mass if key == WRF_MUTS_SLOT else None,
                            total_mu=lambda: pytest.fail("current mass replaced saved native mass"))
    state.rthften = lane.copy()
    state.rqvften = lane.copy()
    finish_advective_forcing(state, SimpleNamespace(cu_physics=16))
    lane = getattr(state, slot)
    scheme = NewTiedtke()
    scheme.bind_driver(SimpleNamespace(state=state, **{name: lane}))
    result = scheme._lane(name, 2, 2, 3)
    expected = before / (state.c1h[:, None, None] * saved_mass + state.c2h[:, None, None])
    np.testing.assert_array_equal(result, expected)
    assert result is lane
    # An adapter may supply uncoupled rates while also exposing a state.
    # Consumption must not decouple either native or external rates again.
    external = lane.copy()
    scheme.bind_driver(SimpleNamespace(state=state, **{name: external}))
    assert scheme._lane(name, 2, 2, 3) is external


def test_other_schemes_do_not_enter_nt_forcing_transition():
    finish_advective_forcing(SimpleNamespace(), SimpleNamespace(cu_physics=0))

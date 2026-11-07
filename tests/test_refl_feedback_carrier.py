"""Registry hdu radar scratch participates in the feedback transaction."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core.nest_physics_feedback import _carriers, FEEDBACK_OPERATORS


@pytest.mark.parametrize("streamed", [False, True])
def test_feedback_binds_the_same_radar_array_as_history(streamed):
    refl = np.full((4, 9, 8), -35.0, dtype=np.float32)
    if streamed:
        state = SimpleNamespace(_streamed_domain=SimpleNamespace(
            store={"scratch/refl_10cm": refl}))
    else:
        state = SimpleNamespace(_scratch={"refl_10cm": refl},
                                physics=SimpleNamespace(fields={},
                                                        refl_10cm=refl))
    assert FEEDBACK_OPERATORS["refl_10cm"] == "avg"
    bound = _carriers(state)["refl_10cm"]
    assert bound is refl
    bound[0, 4, 4] = 20.0
    assert refl[0, 4, 4] == 20.0

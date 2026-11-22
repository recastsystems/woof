"""The inline vertical refusals must name the route that DOES work.

A per-domain vertical ladder ships on the offline downscale route.  The
inline nest corridor still shares one ladder across a live tree and its
refusals stand -- but a refusal that says only "not implemented" now
describes the product as less capable than it is, and the reader it turns
away has a working door two lines from the one they knocked on.

This is the retire-its-guards half of the gate law: the guards survive
because the breakage they name is real, and their WORDING is what the fix
retires.
"""

import numpy as np
import pytest

DOOR_PHRASES = ("downscale", "--child-levels")


def _names_the_door(message: str) -> bool:
    return all(phrase in message for phrase in DOOR_PHRASES)


def test_the_experiment_config_refusal_names_the_offline_door():
    """A per-domain vertical key on a [[domain]] table is still refused."""
    from woof.experiment import _reject_domain_vertical_keys

    with pytest.raises(ValueError) as excinfo:
        _reject_domain_vertical_keys({"nz": 96}, 2, "child.toml")
    message = str(excinfo.value)
    assert "rejected" in message
    assert _names_the_door(message), message


@pytest.mark.parametrize("key", ["nz", "e_vert", "eta_levels", "p_top",
                                 "ztop", "hybrid_opt", "etac"])
def test_every_per_domain_vertical_key_is_still_refused(key):
    """NEGATIVE CONTROL: the reworded refusal must still refuse all seven."""
    from woof.experiment import _reject_domain_vertical_keys

    with pytest.raises(ValueError):
        _reject_domain_vertical_keys({key: 1}, 2, "child.toml")


def test_a_domain_table_with_no_vertical_key_is_admitted():
    """NEGATIVE CONTROL: a refusal that fires on everything is not a gate."""
    from woof.experiment import _reject_domain_vertical_keys

    _reject_domain_vertical_keys({"grid_id": 2, "nx": 100}, 2, "child.toml")


def test_the_real_child_init_refusal_names_the_offline_door():
    from woof.experiment import VerticalConfig
    from woof.ingest.nest_init import _shared_vertical_coord

    vertical = VerticalConfig(
        eta_levels=tuple(np.linspace(1.0, 0.0, 9)), p_top=5000.0,
        hybrid_opt=2, etac=0.2)
    with pytest.raises(ValueError) as excinfo:
        _shared_vertical_coord(vertical, 16)
    assert _names_the_door(str(excinfo.value)), str(excinfo.value)


def test_the_shared_ladder_is_still_accepted_at_its_own_nz():
    """NEGATIVE CONTROL: the shape gate must admit the matching ladder."""
    from woof.experiment import VerticalConfig
    from woof.ingest.nest_init import _shared_vertical_coord

    vertical = VerticalConfig(
        eta_levels=tuple(np.linspace(1.0, 0.0, 9)), p_top=5000.0,
        hybrid_opt=2, etac=0.2)
    assert _shared_vertical_coord(vertical, 8).znw.shape == (9,)


def test_the_offline_route_capability_is_declared():
    """``vertical_remapping`` was False; a shipped capability must say so."""
    from woof.offline_child_run import _CAPABILITIES

    assert _CAPABILITIES["vertical_remapping"] != False  # noqa: E712
    assert "conservative" in str(_CAPABILITIES["vertical_remapping"])

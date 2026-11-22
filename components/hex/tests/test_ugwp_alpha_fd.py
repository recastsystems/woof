"""UGWP MPAS_20261104 alpha_fd input: default 12 compiles the pin's source, other values are live, bad ones refused."""
import pytest

from woof.hex.mpasphys import ugwp_oro_v842 as oro


def test_default_literal_is_the_pins_token():
    assert oro.ALPHA_FD_DEFAULT == 12.0
    assert oro.alpha_fd_literal() == "12.0f"
    # The pin hard-coded 12.0f in this expression; the macro is its only replacement.
    assert "UGWP_ALPHA_FD*6.325e-3f*gfk_exp(" in oro.CUDA_SOURCE
    assert "12.0f*6.325e-3f" not in oro.CUDA_SOURCE


def test_call_option_sets_alpha_fd():
    class Call:
        options = {"config_ugwp_alpha_fd": 6.0}
    assert oro.call_alpha_fd(Call()) == 6.0
    assert oro.alpha_fd_literal(6.0) == "6.0f"

    class Bare:
        options = {}
    assert oro.call_alpha_fd(Bare()) == 12.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_bad_alpha_fd_is_refused(value):
    with pytest.raises(ValueError, match="config_ugwp_alpha_fd"):
        oro.alpha_fd_literal(value)

"""The MPAS suite route applies the door's dycore schedule flags (2026-10-27: --acoustic-substeps was dropped)."""
import argparse

import pytest

from woof.hex import forecast_door
from woof.hex.mpasphys.forecast_v842 import namelist_option

TEXT = "&nhyd_model\nconfig_dt=24.0\n/\n&physics\nconfig_physics_suite='convection_permitting'\n/\n"


def args(**kw):
    base = dict(acoustic_substeps=None, time_integration_order=None, monotonic=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_substeps_flag_reaches_the_namelist():
    text = forecast_door.bind_native_dynamics(args(acoustic_substeps=6), TEXT)
    assert int(namelist_option(text, "nhyd_model", "config_number_of_sub_steps")) == 6
    assert float(namelist_option(text, "nhyd_model", "config_dt")) == 24.0
    assert namelist_option(text, "physics", "config_physics_suite") == "convection_permitting"


def test_no_flag_leaves_the_text_alone():
    assert forecast_door.bind_native_dynamics(args(), TEXT) == TEXT


def test_order_and_monotonic_reach_the_namelist():
    text = forecast_door.bind_native_dynamics(args(time_integration_order=3, monotonic=False), TEXT)
    assert int(namelist_option(text, "nhyd_model", "config_time_integration_order")) == 3
    assert forecast_door._namelist_value(namelist_option(text, "nhyd_model", "config_monotonic")) is False


def test_namelist_disagreeing_with_a_flag_is_refused():
    named = TEXT.replace("config_dt=24.0", "config_dt=24.0\nconfig_number_of_sub_steps=4")
    with pytest.raises(ValueError, match="two values"):
        forecast_door.bind_native_dynamics(args(acoustic_substeps=6), named)
    assert forecast_door.bind_native_dynamics(args(acoustic_substeps=4), named) == named


# --dynamics KEY=VALUE on the MPAS suite route (2026-11-04: accepted, recorded, never run).

def test_dynamics_reaches_the_namelist_in_its_group():
    a = args(dynamics=["config_coef_3rd_order=1", "config_epssm_minimum=0.5", "config_epssm_maximum=0.5"])
    text = forecast_door.bind_native_dynamics(a, TEXT)
    assert namelist_option(text, "nhyd_model", "config_coef_3rd_order") == 1.0
    assert namelist_option(text, "damping", "config_epssm_minimum") == 0.5
    assert namelist_option(text, "damping", "config_epssm_maximum") == 0.5
    assert float(namelist_option(text, "nhyd_model", "config_dt")) == 24.0
    assert namelist_option(text, "physics", "config_physics_suite") == "convection_permitting"
    # The real option is held as a real (Registry kind), and run_forecast checks the decoded value against this.
    assert a._native_dynamics_overrides == {"config_coef_3rd_order": 1.0, "config_epssm_minimum": 0.5,
                                            "config_epssm_maximum": 0.5}
    assert isinstance(a._native_dynamics_overrides["config_coef_3rd_order"], float)


def test_no_dynamics_leaves_the_text_and_arguments_alone():
    a = args(dynamics=None)
    assert forecast_door.bind_native_dynamics(a, TEXT) == TEXT
    assert not hasattr(a, "_native_dynamics_overrides")


def test_dynamics_with_a_flag_of_its_own_is_refused():
    with pytest.raises(ValueError, match="flag of its own"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_number_of_sub_steps=6"]), TEXT)


def test_dynamics_the_dycore_does_not_carry_is_refused():
    with pytest.raises(ValueError, match="carries no such option"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_no_such_option=1"]), TEXT)
    with pytest.raises(ValueError, match="carries no such option"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_physics_suite=\"hrrrv5\""]), TEXT)


def test_dynamics_disagreeing_with_the_namelist_is_refused():
    named = TEXT.replace("config_dt=24.0", "config_dt=24.0\nconfig_coef_3rd_order=0.25")
    with pytest.raises(ValueError, match="two values"):
        forecast_door.bind_native_dynamics(args(dynamics=["config_coef_3rd_order=1.0"]), named)
    assert forecast_door.bind_native_dynamics(args(dynamics=["config_coef_3rd_order=0.25"]), named) == named

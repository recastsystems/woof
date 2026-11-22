"""The MPAS suite route binds the door's upper-damping flags into &damping (woof.hex.forecast_door.bind_native_damping).

Brought from fork jacobbieker/woof PR 509 (Apache-2.0).
"""
from __future__ import annotations

import argparse

import pytest

from woof.hex.forecast_door import bind_native_damping
from woof.hex.mpasphys.forecast_v842 import namelist_option

TEXT = "&nhyd_model\n config_dt = 600.0\n/\n"


def _args(**given):
    defaults = dict(cam_coefficient=0.0, cam_damping_levels=4, rayleigh_u=False, damping_start_m=22000.0)
    defaults.update(given)
    return argparse.Namespace(**defaults)


def test_flags_off_their_defaults_reach_the_namelist():
    text = bind_native_damping(_args(cam_coefficient=0.6, rayleigh_u=True, damping_start_m=15000.0), TEXT)
    assert float(namelist_option(text, "damping", "config_mpas_cam_coef")) == 0.6
    assert str(namelist_option(text, "damping", "config_rayleigh_damp_u")).strip(".").lower() in ("true", "t")
    assert float(namelist_option(text, "damping", "config_zd")) == 15000.0
    assert namelist_option(text, "damping", "config_number_cam_damping_levels") is None


def test_flags_at_their_defaults_leave_the_namelist_alone():
    assert bind_native_damping(_args(), TEXT) == TEXT


def test_two_values_for_one_option_are_refused():
    text = "&damping\n config_mpas_cam_coef = 0.2\n/\n"
    with pytest.raises(ValueError, match="two values for one run"):
        bind_native_damping(_args(cam_coefficient=0.6), text)


def test_the_route_config_takes_the_damping_level_counts_and_timescale():
    from dataclasses import fields

    from woof.hex.mpasphys.config_v842 import NativeMpasDycoreConfigV842

    names = {f.name for f in fields(NativeMpasDycoreConfigV842)}
    assert {"config_number_cam_damping_levels", "config_number_rayleigh_damp_u_levels",
            "config_rayleigh_damp_u_timescale_days"} <= names

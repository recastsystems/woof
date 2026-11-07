"""Run the mp=28 plain-powf source rule on CPU despite the device module's
GPU marker.

This file used to run the fork's plain-powf receipt guard (b0556bd76,
development branch).  The two fork sites that receipt covered now call
WOOF's own powf word like every other mp=28 site, so the guard is the
all-sites rule in tests/test_thompson_aerosol_device_helpers.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_no_plain_powf_survives_without_a_device():
    path = Path(__file__).with_name("test_thompson_aerosol_device_helpers.py")
    spec = importlib.util.spec_from_file_location("_thompson_powf_host_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_no_plain_powf_survives_in_any_mp28_arm()

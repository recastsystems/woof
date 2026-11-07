"""The wrfinput door names every WRF v4.6.1 microphysics package correctly.

Breakage this prevents: a wrfinput written by a real WRF package that the
door's tables did not list (mp_physics 15, 38, 95, 96, 97, 106) was refused
with "is not a WRF microphysics package this door recognises", which is
false, and mp_physics 55 / 56 were refused under P3 names although WRF
v4.6.1 assigns them to Jensen ISHMAEL and NTU.  The authority is the
byte-unmodified v4.6.1 Registry the lake oracle already vendors
(tools/lake_wrf461_oracle/upstream/Registry.EM_COMMON, sha256 28cf7a2d...).

CPU only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from woof.ingest import wrfinput

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "tools" / "lake_wrf461_oracle" / "upstream" / "Registry.EM_COMMON"


def _registry_mp_packages() -> dict[int, str]:
    rows = {}
    for line in REGISTRY.read_text().splitlines():
        m = re.match(r"^package\s+(\S+)\s+mp_physics==(\d+)\s", line)
        if m:
            rows[int(m.group(2))] = m.group(1)
    return rows


def test_every_registry_package_is_named_by_the_door():
    rows = _registry_mp_packages()
    assert rows[5] == "fer_mp_hires" and rows[95] == "etampnew"
    named = set(wrfinput.SUPPORTED_MICROPHYSICS) | set(
        wrfinput.UNSUPPORTED_MICROPHYSICS)
    assert sorted(set(rows) - named) == []


@pytest.mark.parametrize("mp", sorted(_registry_mp_packages()))
def test_a_refused_registry_package_is_refused_as_a_wrf_package(mp):
    if mp in wrfinput.SUPPORTED_MICROPHYSICS:
        pytest.skip("restored by the door")
    text = wrfinput.unsupported_microphysics_refusal(mp)
    assert "is a WRF package this port does not carry" in text
    assert "not a WRF microphysics package" not in text


def test_55_and_56_carry_wrfs_own_names():
    rows = _registry_mp_packages()
    assert rows[55] == "jensen_ishmael" and rows[56] == "ntu"
    assert "ISHMAEL" in wrfinput.SUPPORTED_MICROPHYSICS[55]
    assert "NTU" in wrfinput.UNSUPPORTED_MICROPHYSICS[56]
    assert "P3" not in wrfinput.SUPPORTED_MICROPHYSICS[55]
    assert "P3" not in wrfinput.UNSUPPORTED_MICROPHYSICS[56]


def test_the_ferrier_family_is_admitted_at_the_door():
    # The old Eta Ferrier scheme (mp=95) was refused here until its 3.0 port.
    # With test_ferrier_hires_and_advect_are_admitted_by_the_wrfinput_door
    # (mp=15) this replaces test_the_ferrier_family_is_refused_by_name_at_the_door.
    for mp in (95,):
        wrfinput.check_supported_schemes(
            {"MP_PHYSICS": mp, "SF_SURFACE_PHYSICS": 2}, source="test")


def test_ferrier_hires_and_advect_are_admitted_by_the_wrfinput_door():
    for mp in (5, 15):
        wrfinput.check_supported_schemes(
            {"MP_PHYSICS": mp, "SF_SURFACE_PHYSICS": 2}, source="test")

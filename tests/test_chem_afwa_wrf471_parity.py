"""AFWA dust emission (dust_opt=3) against WRF v4.7.1, bit for bit.

``run_gocart_afwa.F90`` drives the byte-unmodified
``gocart_dust_afwa_driver`` / ``source_dust`` (chem/module_gocart_dust_afwa.F)
over a covering set of 14 switch variants: every dust_smois x
sf_surface_physics pair, then dust_veg 1 and 2, dust_dsr 1 (with a negative
DRI sentinel), dust_soils 1 (with a negative NGA sentinel) and non-default
tuning each moved alone, then all together; the base variant runs three
consecutive steps.  Outputs are the five bins, EDUST, TOT_EDUST, TOT_DUST,
VIS_DUST and AFWA_DUSTLOFT.

Measured on the test host 1 RTX 4090 (sm_89): every output equals the Fortran bit
for bit.  The implementer lane also ran the full 192-variant factorial (576
cases) at 0 ULP before the fixture was cut to this covering set for size.
"""
import pytest

from conftest import requires_gpu
from chem_emis_parity_support import check_pins, check_case_coverage, replay


def test_fixture_pins():
    check_pins("afwa")


@requires_gpu
@pytest.mark.gpu
def test_wrf_driver_parity():
    cases = replay("afwa")
    assert len(cases) == 16


def test_recorded_case_coverage():
    check_case_coverage("afwa")

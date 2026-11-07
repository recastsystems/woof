"""GOCART dust emission (dust_opt=1) against WRF v4.7.1, bit for bit.

``run_gocart_dust.F90`` drives the byte-unmodified ``gocart_dust_driver`` /
``source_du`` (chem/module_gocart_dust.F) on a 38 x 2 patch covering every
soil category 1-19 (water included), wet, dry and zero soil moisture, winds
under and over threshold, a lowest layer under 12 m (which switches WRF to
u_phy/v_phy) and class-by-class zero erodibility, four variants of three
consecutive steps.  The port is the table-driven ``chem_dust_gocart`` kernel
handed the five bins' WRF constants as row parameters.

Measured on the test host 1 RTX 4090 (sm_89): every dust mixing ratio and EDUST
accumulator equals the Fortran bit for bit.  Made to fire before commit:
taking source_du's ``0.13*1.0D-2`` with a double 0.13 instead of the float32
literal gfortran widens fails this test and the AFWA one.
"""
import pytest

from conftest import requires_gpu
from chem_emis_parity_support import check_pins, check_case_coverage, replay


def test_fixture_pins():
    check_pins("dust")


@requires_gpu
@pytest.mark.gpu
def test_wrf_driver_parity():
    cases = replay("dust")
    assert len(cases) == 12


def test_recorded_case_coverage():
    check_case_coverage("dust")

"""GOCART sea-salt emission (seas_opt=1) against WRF v4.7.1, bit for bit.

``run_gocart_seasalt.F90`` drives the byte-unmodified
``gocart_seasalt_driver`` / ``source_ss`` (chem/module_gocart_seasalt.F) over
open ocean at 10 m winds 0, 3, 10 and 25 m s-1, an elevated lake (no
emission, since WRF requires z_at_w < 1 mm), land, thin lowest layers and dt
30 and 60 s, three consecutive steps.  The port computes source_ss's per
sub-bin mass factor once per dt in a one-thread table kernel with the same
operation sequence and sums it per column in WRF's sub-bin order.

Measured on the test host 1 RTX 4090 (sm_89): every sea-salt mixing ratio and
EMIS_SEAS equals the Fortran bit for bit.  Made to fire before commit:
module_data_gocart_seas.F's ``pi = 3.141592653559`` is a float32 literal in a
REAL*8 PARAMETER; using the double pi instead fails the test.
"""
import pytest

from conftest import requires_gpu
from chem_emis_parity_support import check_pins, check_case_coverage, replay


def test_fixture_pins():
    check_pins("seasalt")


@requires_gpu
@pytest.mark.gpu
def test_wrf_driver_parity():
    cases = replay("seasalt")
    assert len(cases) == 6


def test_recorded_case_coverage():
    check_case_coverage("seasalt")

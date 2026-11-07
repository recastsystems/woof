"""MYNN was runnable and unreachable at the same time.

The reopening battery found the shipped MYNN suite integrating a full
simulated hour from both GFS and native HRRR -- and every MYNN tuple other
than MYNN+WSM6 unreachable from any front door, because
``woof import-namelist`` had no mapping for ``bl_pbl_physics = 5``
(``implemented: [0, 1, 11]``).  The B-M1 snow fix exists precisely for
MP 6/8/10/18; only MP 6 could be reached.

``woof/namelist_import.py`` says admitting a scheme means widening three
things together -- the importer map, its ``physics_compat`` readiness row,
and its dispatch row.  For MYNN two of the three were already done.  This
file pins all three to each other so the lagging-leg failure cannot recur
silently, and shows the readiness authority following WRF's mixed-pair law.
"""

from __future__ import annotations

import pytest

from woof.namelist_import import _BL_MAP, _MP_MAP, _SFCLAY_ALLOWED
from woof.physics_compat import (
    MYNN_PROFILE_ID,
    pending_wrf_physics_components,
    single_domain_runtime_switches,
)


def _blockers(**selection):
    request = {
        "mp_physics": 6, "sf_sfclay_physics": 91, "bl_pbl_physics": 1,
        "sf_surface_physics": 2, "num_soil_layers": 4,
    }
    request.update(selection)
    return pending_wrf_physics_components(**request)


def test_the_readiness_authority_admits_every_wrf_legal_mynn_pairing():
    assert _blockers(sf_sfclay_physics=5, bl_pbl_physics=5) == ()
    assert _blockers(sf_sfclay_physics=1, bl_pbl_physics=5) == ()
    assert _blockers(sf_sfclay_physics=91, bl_pbl_physics=5) == ()
    assert _blockers(sf_sfclay_physics=5, bl_pbl_physics=0) == ()

    blockers = _blockers(sf_sfclay_physics=5, bl_pbl_physics=1)
    assert [blocker.component for blocker in blockers] == [
        "WRF v4.6.1 PBL/surface-layer compatibility"]


def test_the_importer_can_express_every_suite_the_authority_admits():
    """The leg that lagged: MYNN's two selectors, now mapped."""

    assert 5 in _BL_MAP, "bl_pbl_physics=5 (MYNN) must be importable"
    assert 5 in _SFCLAY_ALLOWED, \
        "sf_sfclay_physics=5 (MYNN) must be importable"
    # And the microphysics the B-M1 snow path exists for are all mapped, so
    # MYNN is no longer pinned to WSM6 by the front door.
    for mp_physics in (6, 8, 10, 18):
        assert mp_physics in _MP_MAP
        assert _blockers(mp_physics=mp_physics, sf_sfclay_physics=5,
                         bl_pbl_physics=5) == ()

    # Both QNSE and BouLac import with their actual paired surface layer.
    assert 4 in _BL_MAP and 4 in _SFCLAY_ALLOWED
    assert _blockers(sf_sfclay_physics=4, bl_pbl_physics=4) == ()
    assert 8 in _BL_MAP, "BouLac imports natively"
    assert 7 in _SFCLAY_ALLOWED, "Pleim-Xiu surface layer is importable"



def test_the_runtime_leg_that_was_already_there():
    """The shipped fixed profile forecasts the pair the importer now writes."""

    switches = single_domain_runtime_switches(MYNN_PROFILE_ID)
    resolved = switches.get("resolved", switches)
    assert int(resolved["sf_sfclay_physics"]) == 5
    assert int(resolved["bl_pbl_physics"]) == 5


@pytest.mark.parametrize(
    "land_surface", (3, 4))
def test_mynn_lsm_pairings_are_admitted_after_the_ownership_port(
        land_surface: int):
    """This test's previous name was
    test_mynn_pairings_nobody_measured_are_still_refused: RUC and Noah-MP
    each brought their own 2-m diagnostic and the pairs stood refused.
    The surface-driver ownership port (product/v1.2-physics-mynn-pairing)
    measured them -- WRF write-back sequencing transcribed, ownership
    tables tested, three-step GPU integrations receipted -- and retired
    both blockers, so the guard now pins the ADMITTED state.  The
    WRF-fatal control lives in the pairing assertion above."""

    blockers = _blockers(
        sf_sfclay_physics=5, bl_pbl_physics=5,
        sf_surface_physics=land_surface,
        num_soil_layers=9 if land_surface == 3 else 4)
    assert blockers == (), (
        f"MYNN/MYNN/lsm{land_surface} was measured and admitted by the "
        f"ownership port; unexpected blockers: {blockers}")

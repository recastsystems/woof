"""The default surface networks come from the domain, never from a region.

Breakage this file prevents: with no networks named, the recent-case author
and the DA observation step both fell back to one fixed Southeast list. An
Iowa DA case then fetched zero stations ("rw_asos: no station survived") and
failed, while Ohio-valley and Southeast cases scored on partial station sets
that silently lacked KY/IN/OH or NC/VA/WV/PA. The default is now the frozen
network table screened against the domain box, through one function every
engine path calls.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from woof.obs.surface_networks import (DOMAIN_MARGIN_DEG, TABLE_SCHEMA,
                                        SurfaceNetwork, SurfaceNetworkError,
                                        SurfaceNetworkTable,
                                        networks_for_domain)

REPO = Path(__file__).resolve().parents[1]

# (W, S, E, N) boxes in the shape the authoring step writes into a case.
IOWA = (-96.6, 40.4, -90.1, 43.5)
OHIO_VALLEY = (-89.0, 36.5, -80.5, 41.9)
SOUTHEAST = (-94.0, 29.0, -81.0, 36.5)
SOUTH_ATLANTIC = (-30.0, -50.0, -25.0, -45.0)

#: The list the defect hard-coded; named here only to show it is gone.
FORMER_REGIONAL_DEFAULT = ("LA_ASOS", "MS_ASOS", "AL_ASOS", "FL_ASOS",
                           "GA_ASOS", "TN_ASOS", "AR_ASOS")


def _table(rows):
    return SurfaceNetworkTable(
        schema=TABLE_SCHEMA, source_url="", source_sha256="", frozen_at="",
        networks=tuple(SurfaceNetwork(**row) for row in rows))


def test_iowa_gets_iowa_and_its_neighbours_and_none_of_the_southeast():
    resolved = networks_for_domain(*IOWA)
    assert {"IA_ASOS", "IL_ASOS", "MN_ASOS", "MO_ASOS", "NE_ASOS", "SD_ASOS",
            "WI_ASOS"} <= set(resolved)
    assert not set(FORMER_REGIONAL_DEFAULT) & set(resolved)


def test_the_ohio_valley_gets_the_states_the_fixed_list_lacked():
    resolved = set(networks_for_domain(*OHIO_VALLEY))
    assert {"OH_ASOS", "KY_ASOS", "IN_ASOS", "WV_ASOS", "PA_ASOS", "IL_ASOS",
            "TN_ASOS", "VA_ASOS"} <= resolved
    assert not {"LA_ASOS", "MS_ASOS", "FL_ASOS"} & resolved


def test_the_southeast_gets_the_old_states_and_the_ones_it_dropped():
    resolved = set(networks_for_domain(*SOUTHEAST))
    assert set(FORMER_REGIONAL_DEFAULT) <= resolved
    assert {"NC_ASOS", "SC_ASOS"} <= resolved
    assert "IA_ASOS" not in resolved


def test_the_answer_is_sorted_so_receipts_do_not_reorder():
    resolved = networks_for_domain(*OHIO_VALLEY)
    assert list(resolved) == sorted(resolved)


def test_open_ocean_is_refused_by_name():
    with pytest.raises(SurfaceNetworkError,
                       match="no ASOS network intersects the domain"):
        networks_for_domain(*SOUTH_ATLANTIC)


def test_the_margin_offers_a_network_just_past_the_domain_edge():
    """A box ending short of a network extent still offers that network.

    Within the margin it is offered (the station-level --bbox then drops
    stations outside the domain); past the margin it is not.
    """

    table = _table([dict(id="X_ASOS", name="X", west=-90.0, south=30.0,
                         east=-85.0, north=35.0)])
    short = DOMAIN_MARGIN_DEG / 2
    assert networks_for_domain(-95.0, 30.0, -90.0 - short, 35.0,
                               table=table) == ("X_ASOS",)
    with pytest.raises(SurfaceNetworkError):
        networks_for_domain(-95.0, 30.0, -90.0 - 2 * DOMAIN_MARGIN_DEG, 35.0,
                            table=table)
    with pytest.raises(SurfaceNetworkError):
        networks_for_domain(-95.0, 30.0, -90.0 - short, 35.0, margin_deg=0.0,
                            table=table)


def test_the_margin_carries_across_the_antimeridian():
    """A box at -179.9 grown westward reaches a network just west of 180."""

    table = _table([dict(id="W__ASOS", name="W", west=178.0, south=-20.0,
                         east=179.8, north=-15.0)])
    assert networks_for_domain(-179.9, -20.0, -175.0, -15.0,
                               table=table) == ("W__ASOS",)


def test_a_box_grown_past_the_whole_band_reads_as_the_whole_band():
    assert networks_for_domain(-179.8, 40.0, 179.8, 42.0)


@pytest.mark.parametrize("margin", [-1.0, float("nan"), 90.0])
def test_a_margin_that_is_not_a_width_is_refused(margin):
    with pytest.raises(SurfaceNetworkError, match="margin"):
        networks_for_domain(*IOWA, margin_deg=margin)


def _plan(bbox):
    return {"model_start_utc": "2026-10-03T21:00:00Z",
            "analysis_times_utc": ["2026-10-03T22:00:00Z"],
            "smoke_only": True, "bbox": list(bbox)}


def test_no_engine_code_carries_a_hard_coded_state_network_list():
    """Two or more literal state network ids on one line is a regional list.

    The only place network ids may live is the frozen table under
    woof/obs/data; usage examples in help text name one or two networks
    with "e.g." and are not defaults.
    """

    pattern = re.compile(r"[\"']([A-Z]{2})_ASOS[\"']")
    offenders = []
    for root in ("woof", "tools"):
        for path in (REPO / root).rglob("*.py"):
            if "target" in path.parts or "rustwx" in path.parts:
                continue
            for number, line in enumerate(
                    path.read_text(encoding="utf-8",
                                   errors="replace").splitlines(), 1):
                if len(pattern.findall(line)) >= 2 or "{state}_ASOS" in line:
                    offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert not offenders, offenders


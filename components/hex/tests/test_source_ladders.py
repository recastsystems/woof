"""Init sources at any posted lead: the publisher's ladders decide which leads exist.

The breakage: one interval per row either refused real leads (IFS 3-hourly to 144 h, AIFS 6-hourly, GFS hourly to
120 h) or admitted leads never posted (IFS 147 h, a 06/18 UTC IFS run past 144 h), which failed later as a missing
object. A later lead of a source whose statics live in step 0 only (IFS z; AIFS z and lsm) also lost its terrain.
"""

import pytest

from woof.hex.init_sources import Cycle, InitSourceRefusal, load_table, plan_products, resolve_source


def _plan(source, cycle, lead, member=None):
    return plan_products(resolve_source(load_table(), source), Cycle.parse(cycle), member, lead)


@pytest.mark.parametrize("source,cycle,lead,member", [
    ("ifs", "2026-10-23T12", 354, None), ("ifs", "2026-10-23T12", 141, None), ("ifs", "2026-10-23T18", 144, None),
    ("ifs-0p1", "2026-10-23T12", 360, None), ("aifs", "2026-10-23T12", 360, None),
    ("ifs-ens", "2026-10-23T12", 354, 7), ("aifs-ens", "2026-10-23T12", 354, 0), ("aifs-ens", "2026-10-23T12", 354, 50),
    ("gefs", "2026-10-23T06", 237, 5), ("gfs", "2026-10-23T18", 119, None), ("gfs", "2026-10-23T18", 348, None),
])
def test_posted_leads_plan(source, cycle, lead, member):
    assert _plan(source, cycle, lead, member)


@pytest.mark.parametrize("source,cycle,lead,member", [
    ("ifs", "2026-10-23T12", 147, None), ("ifs", "2026-10-23T18", 150, None), ("ifs", "2026-10-23T12", 366, None),
    ("aifs", "2026-10-23T12", 3, None), ("gefs", "2026-10-23T06", 243, 5), ("gfs", "2026-10-23T18", 121, None),
])
def test_unposted_leads_are_refused_before_transfer(source, cycle, lead, member):
    with pytest.raises(InitSourceRefusal, match="outside the row's forecast cadence/range"):
        _plan(source, cycle, lead, member)


@pytest.mark.parametrize("source,member,fields", [
    ("ifs", None, {"SOILGEO"}), ("ifs-0p1", None, {"SOILGEO"}), ("ifs-ens", 3, {"SOILGEO"}),
    ("aifs", None, {"SOILGEO", "LANDSEA"}), ("aifs-ens", 4, {"SOILGEO", "LANDSEA"}),
])
def test_a_later_lead_takes_its_step_zero_statics(source, member, fields):
    plans = _plan(source, "2026-10-23T12", 354, member)
    donors = [plan for plan in plans if plan.fields]
    assert {field for plan in donors for field in plan.fields} == fields
    assert all("-0h-" in plan.url and plan.index_select["levtype"] == "sfc" for plan in donors)
    assert all("-354h-" in plan.url for plan in plans if not plan.fields and plan.role != "surface-input")


def test_step_zero_needs_no_donor():
    for source, member in (("ifs", None), ("ifs-0p1", None), ("aifs", None), ("aifs-ens", 2)):
        assert not [plan for plan in _plan(source, "2026-10-23T12", 0, member) if plan.fields]


def test_aifs_ens_control_and_members_read_their_own_files():
    control = _plan("aifs-ens", "2026-10-23T12", 6, 0)[0]
    member = _plan("aifs-ens", "2026-10-23T12", 6, 12)[0]
    assert control.url.endswith("-6h-enfo-cf.grib2") and control.index_select == {"type": "cf"}
    assert member.url.endswith("-6h-enfo-pf.grib2") and member.index_select == {"type": "pf", "number": "12"}

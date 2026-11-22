"""A declared donor that is itself a table route is fetched by its own row.

``woof fetch --source hrrr-native`` refused with "declares a hrrr-prs
donor and this WOOF-ArWen has no route for it" (box S, 2026-10-06): the donor
fetch knew only the two GFS-container sources.  hrrr-prs is a row of the
route table, so the donor is planned and moved the way its own fetch
line would be, into the donor folder, and the handoff binds the primary.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof import fetch, fetch_routes


def _args(tmp_path, **extra):
    return SimpleNamespace(out=tmp_path, force_refetch=False, fetch_workers=None, **extra)


def test_hrrr_native_declares_a_table_route_donor():
    plan = fetch_routes.resolve_request("hrrr-native", cycle=datetime(2024, 5, 21, 18), hours=0)
    assert [(d.source, d.role, tuple(d.leads)) for d in plan.donors] == [
        ("hrrr-prs", "soil_surface_data", (0,))]
    assert fetch._donor_is_table_route(plan.donors[0]) is True
    assert "hrrr-prs" in fetch_routes.table_route_sources()
    assert "gdas" not in fetch_routes.table_route_sources() or \
        not fetch._donor_is_table_route(SimpleNamespace(source="gdas"))


def test_the_hrrr_prs_donor_is_fetched_through_its_own_route(tmp_path, monkeypatch):
    cycle = datetime(2024, 5, 21, 18)
    plan = fetch_routes.resolve_request("hrrr-native", cycle=cycle, hours=0)
    moved = []

    def run_plan(donor_plan, *, out, force, file_workers, **_):
        moved.append((donor_plan, Path(out), force, file_workers))
        return {}

    monkeypatch.setattr(fetch_routes, "run_plan", run_plan)
    monkeypatch.setattr(fetch, "fetch_gfs_fullfile",
                        lambda **kw: pytest.fail("a table-route donor must not ride the GFS ladder"))
    files = fetch._fetch_route_donors(plan, _args(tmp_path))

    assert len(moved) == 1
    donor_plan, out, force, workers = moved[0]
    assert donor_plan.source_id == "hrrr-prs"
    assert donor_plan.cycle == cycle
    assert tuple(donor_plan.leads) == (0,)
    assert out == tmp_path / "donor-hrrr-prs"
    assert (force, workers) == (False, None)
    # The role binds the donor's primary object: the wrfprs analysis.
    assert set(files) == {"soil_surface_data"}
    bound = files["soil_surface_data"]
    assert bound == out / donor_plan.primary_files[0]
    assert "wrfprs" in bound.name and "f00" in bound.name


def test_the_handoff_binds_the_fetched_donor_as_the_supplement(tmp_path, monkeypatch):
    cycle = datetime(2024, 5, 21, 18)
    plan = fetch_routes.resolve_request("hrrr-native", cycle=cycle, hours=0, out=tmp_path)
    monkeypatch.setattr(fetch_routes, "run_plan", lambda donor_plan, **kw: {})
    files = fetch._fetch_route_donors(plan, _args(tmp_path))
    fetch_routes.write_handoff(plan, tmp_path, donor_files=files)
    lines = "\n".join(fetch_routes.handoff_lines(plan, tmp_path))
    assert "soil_surface_data=" in lines
    assert files["soil_surface_data"].name in lines
    assert "STILL NEEDED" not in lines


def test_a_donor_with_neither_a_row_nor_the_gfs_ladder_is_refused_by_name(tmp_path):
    donor = SimpleNamespace(source="no-such-model", role="x", cycle=datetime(2024, 5, 21, 18),
                            leads=(0,), why="test")
    with pytest.raises(ValueError, match="neither a row of .* nor one of the GFS-container"):
        fetch._donor_route_plan(donor, _args(tmp_path))


def test_a_cached_table_route_donor_is_asked_of_its_own_receipt(tmp_path, monkeypatch):
    plan = fetch_routes.resolve_request("hrrr-native", cycle=datetime(2024, 5, 21, 18), hours=0)
    asked = []
    monkeypatch.setattr(fetch_routes, "request_cached",
                        lambda donor_plan, out: asked.append((donor_plan.source_id, Path(out))) or True)
    monkeypatch.setattr(fetch, "cached_request_complete",
                        lambda *a, **k: pytest.fail("a table-route donor has no GFS receipt"))
    assert fetch._donor_cached(plan.donors[0], _args(tmp_path)) is True
    assert asked == [("hrrr-prs", tmp_path / "donor-hrrr-prs")]


def test_the_donor_route_takes_the_primary_transport_pin(tmp_path, monkeypatch):
    """``--transport`` pins every byte of the fetch, the donor's included."""
    plan = fetch_routes.resolve_request("hrrr-native", cycle=datetime(2024, 5, 21, 18), hours=0)
    seen = []
    real = fetch_routes.resolve_request

    def resolve_request(source, **kw):
        seen.append((source, kw.get("host")))
        return real(source, **kw)

    monkeypatch.setattr(fetch_routes, "resolve_request", resolve_request)
    fetch._donor_route_plan(plan.donors[0], _args(tmp_path, transport="aws"))
    assert seen == [("hrrr-prs", "aws")]
    seen.clear()
    fetch._donor_route_plan(plan.donors[0], _args(tmp_path))
    assert seen == [("hrrr-prs", None)]


def test_a_time_lagged_hrrr_request_takes_its_soil_donor_at_its_own_start(tmp_path, monkeypatch):
    """The soil donor counts its leads from the request's start.

    Breakage it prevents: a time-lagged DA member (HRRR 07Z run started
    at f03 for a 10Z start) took its soil column from the 07Z analysis
    (wrfprsf00), three hours stale, and at seventeen hours' lag an
    afternoon top-node soil temperature started a morning forecast.
    A forecast from its own cycle still takes the analysis (f00).
    """
    cycle = datetime(2023, 7, 10, 7)
    lagged = fetch_routes.resolve_request("hrrr-native", cycle=cycle, hours=1, start_hour=3)
    assert tuple(lagged.leads)[0] == 3
    assert [(d.source, tuple(d.leads)) for d in lagged.donors] == [("hrrr-prs", (3,))]
    own = fetch_routes.resolve_request("hrrr-native", cycle=cycle, hours=1)
    assert [(d.source, tuple(d.leads)) for d in own.donors] == [("hrrr-prs", (0,))]
    monkeypatch.setattr(fetch_routes, "run_plan", lambda donor_plan, **kw: {})
    files = fetch._fetch_route_donors(lagged, _args(tmp_path))
    assert "wrfprsf03" in files["soil_surface_data"].name


def test_an_unknown_donor_lead_origin_is_a_table_error():
    with pytest.raises(ValueError, match="lead_origin"):
        fetch_routes._donor_lead_origin({"lead_origin": "valid_time"})

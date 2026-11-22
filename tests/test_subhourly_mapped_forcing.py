"""A mapped source forced every 15 minutes, end to end through the pieces a hex-forced nest uses.

THE BREAKAGE THESE PREVENT (Hex with Regional, 2026-10-29): a WOOF-ArWen nest under a
hex parent could only be forced hourly.  The mapped preparation already sealed a
sub-hourly series (``forcing_offsets_seconds``), but the forecast reader knew only
``forcing_hours`` and refused it ("mapped hex-global proof top-level inventory
differs"), ``woof domain`` took whole hours only and the local posting numbered
its leads in hours.  Forced hourly, a storm moving along the 1 km nest's south edge
was faded in at its 13:00 position from 12:00 and the nest grew a 40-55 dBZ band in
its edge rows where the 3 km parent had one round cell.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import argparse

import pytest

from woof import domain_wizard, local_posting
from woof import prepared_single_domain_forecast as runner
from tests.test_prepared_single_domain_forecast import _cache_metadata_case


# --------------------------------------------------------------------------- the forecast reader


def test_the_reader_takes_a_mapped_proofs_sub_hourly_axis():
    axis = runner.forcing_axis({"forcing_offsets_seconds": [0, 900, 1800, 2700, 3600]},
                               source="hex-global")
    assert not axis.hourly and axis.hours is None
    assert axis.seconds == (0, 900, 1800, 2700, 3600)
    assert axis.identity() == {"forcing_offsets_seconds": [0, 900, 1800, 2700, 3600]}
    hourly = runner.forcing_axis({"forcing_hours": [0, 1, 2]}, source="gfs")
    assert hourly.hours == (0, 1, 2) and hourly.seconds == (0, 3600, 7200)
    assert hourly.describe() == "2 h (7200 s)"


def test_a_native_route_and_a_doubly_spelled_axis_are_refused():
    with pytest.raises(ValueError, match="mapped preparation's"):
        runner.forcing_axis({"forcing_offsets_seconds": [0, 900]}, source="gfs")
    with pytest.raises(ValueError, match="exactly one of"):
        runner.forcing_axis({"forcing_hours": [0, 1], "forcing_offsets_seconds": [0, 3600]},
                            source="hex-global")
    with pytest.raises(ValueError, match="malformed"):
        runner.forcing_axis({"forcing_offsets_seconds": [900, 1800]}, source="hex-global")


def test_the_proof_inventory_takes_the_axis_in_the_spelling_the_proof_uses():
    required = {"schema", "forcing_hours", "boundary_interval_seconds"}
    assert runner.forcing_axis_proof_keys(required, {"forcing_hours": [0, 1]}) == required
    assert runner.forcing_axis_proof_keys(required, {"forcing_offsets_seconds": [0, 900]}) == {
        "schema", "forcing_offsets_seconds", "boundary_interval_seconds"}


def _sub_hourly_case():
    case = _cache_metadata_case()
    user = case["reader"].header["metadata"]["user"]
    start = case["exp"].start_time
    del user["forcing_hours"]
    user.update(forcing_offsets_seconds=[0, 900], boundary_interval_seconds=900,
                last_valid_time=(start + timedelta(minutes=15)).isoformat())
    case["reader"].header["metadata"]["lbc"]["intervals"][0]["end_seconds"] = 900.0
    del case["forcing_hours"]
    case.update(boundary_interval_seconds=900,
                forcing=runner.ForcingAxis("forcing_offsets_seconds", (0, 900), (0, 900)))
    return case


def test_a_15_minute_cache_validates_against_its_15_minute_axis():
    runner._validate_cache_metadata(**_sub_hourly_case())


def test_a_15_minute_cache_is_refused_against_an_hourly_reading():
    case = _sub_hourly_case()
    case["forcing"] = runner.hourly_forcing_axis([0, 1])
    with pytest.raises(ValueError):
        runner._validate_cache_metadata(**case)


# --------------------------------------------------------------------------- woof domain


def _wizard_args(**values):
    return argparse.Namespace(**{"cadence": None, "cadence_minutes": None, **values})


def test_cadence_minutes_names_a_sub_hourly_interval_and_folds_whole_hours():
    assert domain_wizard._cadence_minutes_interval(_wizard_args(cadence_minutes=15)) == 900
    hourly = _wizard_args(cadence_minutes=120)
    assert domain_wizard._cadence_minutes_interval(hourly) is None and hourly.cadence == 2
    assert domain_wizard._cadence_minutes_interval(_wizard_args()) is None
    with pytest.raises(ValueError, match="not both"):
        domain_wizard._cadence_minutes_interval(_wizard_args(cadence=1, cadence_minutes=15))


# --------------------------------------------------------------------------- the local posting


def _posting(tmp_path, **kwargs):
    return local_posting.LocalPosting(
        source="hex-global", cycle=datetime(2023, 3, 31, 12), leads=[0, 1, 2, 3, 4],
        out=tmp_path / "fetch", ready_list=tmp_path / "ready.jsonl",
        progress=lambda line: None, **kwargs)


def test_a_15_minute_posting_names_its_leads_15_minutes_apart(tmp_path):
    posting = _posting(tmp_path, lead_minutes=15)
    assert [posting.rows[lead]["valid_time"] for lead in posting.leads] == [
        f"2023-03-31T12:{minute:02d}:00Z" if minute < 60 else "2023-03-31T13:00:00Z"
        for minute in (0, 15, 30, 45, 60)]
    assert posting.name(3) == "wrfout_d01_2023-03-31_12_45_00"
    assert posting.declaration["lead_minutes"] == 15
    assert posting.schedule()["lead_minutes"] == 15


def test_an_hourly_posting_is_unchanged(tmp_path):
    posting = _posting(tmp_path)
    assert "lead_minutes" not in posting.declaration and "lead_minutes" not in posting.schedule()
    assert posting.name(1) == "wrfout_d01_2023-03-31_13_00_00"
    declaration = {
        "schema": local_posting.LOCAL_POSTING_SCHEMA, "source": "hex-global", "cycle": "2023-03-31T12",
        "leads": [0, 1, 2, 3, 4], "name_template": local_posting.DEFAULT_NAME_TEMPLATE,
        "supplement_roles": [], "ready": "ready_list"}
    assert posting.declaration == declaration


def test_the_command_line_takes_lead_minutes(tmp_path, capsys):
    assert local_posting.main([
        "--source", "hex-global", "--cycle", "2023-03-31T12", "--leads", "0-4", "--lead-minutes", "15",
        "--out", str(tmp_path / "fetch"), "--ready-list", str(tmp_path / "ready.jsonl"),
        "--print-prep-arguments"]) == 0
    flags = capsys.readouterr().out.split()
    assert [flag.replace("\\", "/").rsplit("/", 1)[-1] for flag in flags[1::2]] == [
        f"wrfout_d01_2023-03-31_{stamp}" for stamp in
        ("12_00_00", "12_15_00", "12_30_00", "12_45_00", "13_00_00")]

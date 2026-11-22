"""The first things a new user hits, from the 3.0.3 candidate check (2026-11-15).

THE BREAKAGES THESE PREVENT, each reproduced on the sealed candidate wheel:

* ``woof domain --point -90,37.5`` (meant 37.5 N, 90 W) exited 0 and wrote a
  polar-stereographic domain centred on the South Pole, with a 360-degree
  forcing box, and said nothing about the point being the pole.
* ``woof domain --cycle 2026-11-20T00`` typed on the 15th wrote a config, and
  ``woof go`` then waited "for gfs f000, not posted yet (scheduled from about
  03:30Z)" with no date: four and a half days.

``woof verify-visuals RUN --domain 1`` has its tests beside the run-folder
flags (``test_verify_visuals_run_folder_flags.py``); the rain map's colour
scale has its tests in ``rustwx-products`` (``qpf.rs``).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pytest

from woof import domain_wizard, fetch, fetch_as_posted, go_cli, source_posting
from woof import source_readiness
from woof.cli import main as cli_main

#: The check's own clock: 15:12Z on the 15th, the cycle typed five days on.
NOW = datetime(2026, 11, 15, 15, 12, 11)
FUTURE = datetime(2026, 11, 20, 0)


# --------------------------------------------------------------------------
# Latitude and longitude swapped
# --------------------------------------------------------------------------

def test_the_checks_swapped_pair_is_refused_before_anything_is_written(tmp_path, capsys):
    out = tmp_path / "m1.toml"
    code = cli_main(["domain", "--point", "-90,37.5", "--source", "gfs",
                     "--card", "16gb", "--cycle", "2026-07-29T18", "--hours", "3",
                     "--out", str(out)])
    err = capsys.readouterr().err
    assert code == 2
    assert "South Pole" in err
    assert "did you mean --point 37.5,-90?" in err
    assert not out.exists()


def test_a_domain_centred_on_the_pole_itself_is_written_lat_0():
    assert domain_wizard.swapped_point_refusal(-90.0, 0.0, source="gfs") is None
    assert domain_wizard.swapped_point_refusal(90.0, 0.0, source="gfs") is None
    said = domain_wizard.swapped_point_refusal(89.8, 45.0, source="gfs")
    assert "North Pole" in said and "--point 90,0" in said


def test_a_point_outside_a_regional_grid_that_is_inside_it_swapped_names_the_swap():
    # -80,40 typed for Pittsburgh (40 N, 80 W): Antarctica, off HRRR's grid.
    said = domain_wizard.swapped_point_refusal(-80.0, 40.0, source="hrrr")
    assert said is not None and "did you mean --point 40,-80?" in said
    assert domain_wizard.swapped_point_refusal(40.0, -80.0, source="hrrr") is None


def test_ordinary_points_on_a_global_source_are_left_alone():
    for lat, lon in ((-80.0, 40.0), (70.0, 20.0), (-33.87, 151.21), (37.5, -90.0)):
        assert domain_wizard.swapped_point_refusal(lat, lon, source="gfs") is None


def test_a_latitude_out_of_range_that_reads_as_a_longitude_names_the_swap():
    with pytest.raises(ValueError, match=r"did you mean --point 35\.2,-97\.5\?"):
        domain_wizard._parse_point("-97.5,35.2")
    with pytest.raises(ValueError) as refused:
        domain_wizard._parse_point("95,120")
    assert "did you mean" not in str(refused.value)


def test_the_prompt_asks_again_for_a_pole_point_with_a_longitude():
    from woof import domain_interactive

    with pytest.raises(ValueError, match="did you mean --point 37.5,-90"):
        domain_interactive._validate_point("-90,37.5")
    domain_interactive._validate_point("90,-97.5")


# --------------------------------------------------------------------------
# A cycle in the future
# --------------------------------------------------------------------------

def test_a_start_days_ahead_is_refused_with_its_date_and_the_wait():
    due = source_posting.expected_at("gfs", FUTURE, 0)
    said = source_readiness.future_start_refusal("gfs", FUTURE, due, now=NOW)
    assert said is not None
    assert "GFS cycle 2026-11-20T00Z is 4 d 9 h in the future" in said
    assert f"due about {due:%Y-%m-%d %H:%M}Z" in said
    assert source_readiness.span_words(due - NOW) in said
    assert "--wait-for-cycle" in said and "wait_for_cycle = true" in said


def test_the_next_cycle_is_an_ordinary_wait():
    cycle = datetime(2026, 11, 15, 18)
    due = source_posting.expected_at("gfs", cycle, 0)
    assert due - NOW < source_readiness.MAX_START_WAIT
    assert source_readiness.future_start_refusal("gfs", cycle, due, now=NOW) is None


def test_a_wait_of_days_is_said_to_the_nearest_hour():
    span = source_readiness.span_words
    assert span(timedelta(days=4, hours=7, minutes=59)) == "4 d 8 h"
    assert span(timedelta(hours=23, minutes=59, seconds=40)) == "1 d 0 h"
    assert span(timedelta(hours=5, minutes=20)) == "5 h 20 min"
    assert span(timedelta(hours=2)) == "2 h"


def test_when_words_carry_the_date_when_not_today():
    assert source_readiness.when_words(datetime(2026, 11, 15, 16, 0), now=NOW) == "16:00Z, in 48 min"
    assert source_readiness.when_words(datetime(2026, 11, 20, 3, 30), now=NOW) == (
        "2026-11-20 03:30Z, in 4 d 12 h")
    assert source_readiness.when_words(datetime(2026, 11, 14, 3, 30), now=NOW) == "2026-11-14 03:30Z"


def _fetch_args(tmp_path, **extra):
    return argparse.Namespace(cadence=None, member=None, hours=3, out=tmp_path,
                              late_after_minutes=None, wait_timeout_minutes=None,
                              **extra)


def test_the_fetch_refuses_a_cycle_days_ahead_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_as_posted, "now_utc", lambda: NOW)
    with pytest.raises(ValueError, match="2026-11-20T00Z is 4 d 9 h in the future"):
        fetch._posting_loop("gfs", FUTURE, _fetch_args(tmp_path), leads=(0, 3))
    loop = fetch._posting_loop("gfs", FUTURE, _fetch_args(tmp_path, wait_for_cycle=True),
                               leads=(0, 3))
    assert loop is not None


def test_go_refuses_before_a_stage_and_carries_the_ask_to_its_fetch():
    plan = {"source": "gfs", "cycle": "2026-11-20T00", "forecast_start_hour": 0}
    said = go_cli.future_cycle_refusal(plan, now=NOW)
    assert said is not None and "2026-11-20T00Z" in said
    assert go_cli.future_cycle_refusal({**plan, "wait_for_cycle": True}, now=NOW) is None
    assert go_cli.future_cycle_refusal({**plan, "as_posted": False}, now=NOW) is None
    assert go_cli.future_cycle_refusal({**plan, "cycle": "2026-11-15T12"}, now=NOW) is None
    command = go_cli.fetch_command({**plan, "hours": 3, "area": "30,-100,45,-80",
                                    "data": "d", "wait_for_cycle": True})
    assert "--wait-for-cycle" in command
    from woof.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["go", "x.toml", "--wait-for-cycle"]).wait_for_cycle
    assert parser.parse_args(["fetch", "--source", "gfs", "--wait-for-cycle"]).wait_for_cycle


def test_the_config_key_reaches_the_fetch_as_its_flag():
    from woof import runplan

    argv = runplan._fetch_arguments_from_hints(
        {"source": "gfs", "cycle": "2026-11-20T00", "hours": 3, "wait_for_cycle": True},
        out=__import__("pathlib").Path("o"))
    assert "--wait-for-cycle" in argv
    assert "wait_for_cycle" in fetch.FETCH_HINT_KEYS


def test_the_beat_line_says_the_date_and_how_long(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(go_cli, "_utc_now", lambda: NOW)
    schedule = tmp_path / "schedule.json"
    schedule.write_text(json.dumps({
        "schema": "gpuwm.posting-schedule.v1", "source": "gfs",
        "leads": [{"lead": 0, "state": "waiting", "expected_at": "2026-11-20T03:30:00Z"},
                  {"lead": 3, "state": "scheduled"}]}))
    assert go_cli._progress_note(schedule) == (
        ", 0 of 2 leads in, waiting for gfs f000, not posted yet "
        "(scheduled from about 2026-11-20 03:30Z, in 4 d 12 h)")


def test_the_wizard_says_go_will_refuse_the_cycle(capsys):
    domain_wizard._warn_future_cycle(FUTURE, source="gfs", now=NOW)
    err = capsys.readouterr().err
    assert "woof go will refuse this config as written" in err
    assert "2026-11-20T00Z" in err
    domain_wizard._warn_future_cycle(NOW.replace(hour=12), source="gfs",
                                     now=NOW + timedelta(hours=1))
    assert capsys.readouterr().err == ""
